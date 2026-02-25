use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::engine::db::Database;
use crate::error::{Result, SpectraError};
use crate::facet::relational::{
    encode_schema_metadata, index_meta_key, parse_schema_metadata, row_key, table_meta_key,
    validate_identifier, validate_index_name, validate_json_bytes, validate_pk,
    validate_table_name, validate_view_name, view_meta_key, IndexMetadata, TableColumnMetadata,
    TableSchemaMetadata, ViewMetadata,
};
use crate::sql::parser::{
    parse_sql, split_sql_statements, OrderDirection, SelectProjection, Statement,
};
use crate::sql::planner::plan;

#[derive(Debug, Clone)]
pub enum SqlResult {
    Affected {
        rows: u64,
        commit_ts: Option<u64>,
        message: String,
    },
    Rows(Vec<Vec<u8>>),
    Explain(String),
}

#[derive(Debug, Clone)]
struct PendingPut {
    key: Vec<u8>,
    value: Vec<u8>,
    valid_from: u64,
    valid_to: u64,
    schema_version: Option<u64>,
}

#[derive(Default)]
struct SqlSession {
    in_txn: bool,
    pending: VecDeque<PendingPut>,
}

#[derive(Debug, Clone)]
struct VisibleRow {
    pk: String,
    doc: Vec<u8>,
}

#[derive(Debug, Clone)]
struct JoinedRow {
    pk: String,
    left_doc: Vec<u8>,
    right_doc: Vec<u8>,
}

impl SqlSession {
    fn stage_put(
        &mut self,
        key: Vec<u8>,
        value: Vec<u8>,
        valid_from: u64,
        valid_to: u64,
        schema_version: Option<u64>,
    ) {
        self.pending.push_back(PendingPut {
            key,
            value,
            valid_from,
            valid_to,
            schema_version,
        });
    }

    fn read_staged(&self, key: &[u8], valid_at: Option<u64>) -> Option<Vec<u8>> {
        for p in self.pending.iter().rev() {
            if p.key.as_slice() != key {
                continue;
            }
            if let Some(ts) = valid_at {
                if !(p.valid_from <= ts && ts < p.valid_to) {
                    continue;
                }
            }
            return Some(p.value.clone());
        }
        None
    }

    fn commit(&mut self, db: &Database) -> Result<(u64, Option<u64>)> {
        if !self.in_txn {
            return Err(SpectraError::SqlExec(
                "COMMIT requires an active transaction".to_string(),
            ));
        }

        let rows = self.pending.len() as u64;
        let mut last_commit_ts = None;
        while let Some(p) = self.pending.pop_front() {
            let ts = db.put(&p.key, p.value, p.valid_from, p.valid_to, p.schema_version)?;
            last_commit_ts = Some(ts);
        }
        self.in_txn = false;
        Ok((rows, last_commit_ts))
    }

    fn rollback(&mut self) -> Result<u64> {
        if !self.in_txn {
            return Err(SpectraError::SqlExec(
                "ROLLBACK requires an active transaction".to_string(),
            ));
        }
        let rows = self.pending.len() as u64;
        self.pending.clear();
        self.in_txn = false;
        Ok(rows)
    }
}

pub fn execute_sql(db: &Database, query: &str) -> Result<SqlResult> {
    let chunks = split_sql_statements(query)?;
    let mut session = SqlSession::default();
    let mut last = None;

    for q in chunks {
        let stmt = parse_sql(&q)?;
        let stmt = plan(stmt)?;
        let res = execute_stmt(db, &mut session, stmt)?;
        last = Some(res);
    }

    if session.in_txn {
        return Err(SpectraError::SqlExec(
            "transaction left open; issue COMMIT or ROLLBACK".to_string(),
        ));
    }

    last.ok_or_else(|| SpectraError::SqlExec("no statements executed".to_string()))
}

fn execute_stmt(db: &Database, session: &mut SqlSession, stmt: Statement) -> Result<SqlResult> {
    match stmt {
        Statement::Begin => {
            if session.in_txn {
                return Err(SpectraError::SqlExec(
                    "nested BEGIN is not supported".to_string(),
                ));
            }
            session.in_txn = true;
            Ok(SqlResult::Affected {
                rows: 0,
                commit_ts: None,
                message: "BEGIN".to_string(),
            })
        }
        Statement::Commit => {
            let (rows, commit_ts) = session.commit(db)?;
            Ok(SqlResult::Affected {
                rows,
                commit_ts,
                message: format!("COMMIT ({rows} writes)"),
            })
        }
        Statement::Rollback => {
            let rows = session.rollback()?;
            Ok(SqlResult::Affected {
                rows,
                commit_ts: None,
                message: format!("ROLLBACK ({rows} staged writes discarded)"),
            })
        }
        Statement::CreateTable { table } => {
            validate_table_name(&table)?;
            let key = table_meta_key(&table);
            if read_live_key(db, session, &key, None, None)?.is_some() {
                return Err(SpectraError::SqlExec(format!(
                    "table {table} already exists"
                )));
            }

            let schema = TableSchemaMetadata {
                pk: "pk".to_string(),
                doc: "json".to_string(),
                columns: vec![
                    TableColumnMetadata {
                        name: "pk".to_string(),
                        type_name: "TEXT".to_string(),
                    },
                    TableColumnMetadata {
                        name: "doc".to_string(),
                        type_name: "JSON".to_string(),
                    },
                ],
            };
            let payload = encode_schema_metadata(&schema)?;
            let commit_ts = write_put(db, session, key, payload, 0, u64::MAX, Some(1))?;
            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: format!("created table {table}"),
            })
        }
        Statement::CreateView {
            view,
            source,
            pk,
            as_of,
            valid_at,
        } => {
            validate_view_name(&view)?;
            validate_table_name(&source)?;
            let source_meta = table_meta_key(&source);
            if read_live_key(db, session, &source_meta, None, None)?.is_none() {
                return Err(SpectraError::SqlExec(format!(
                    "source table {source} does not exist"
                )));
            }

            let key = view_meta_key(&view);
            if read_live_key(db, session, &key, None, None)?.is_some() {
                return Err(SpectraError::SqlExec(format!("view {view} already exists")));
            }

            let query = format!(
                "SELECT doc FROM {} WHERE pk='{}'{}{}",
                source,
                escape_sql_literal(&pk),
                as_of.map(|v| format!(" AS OF {v}")).unwrap_or_default(),
                valid_at
                    .map(|v| format!(" VALID AT {v}"))
                    .unwrap_or_default()
            );
            let meta = ViewMetadata {
                name: view.clone(),
                query,
                depends_on: vec![source],
            };
            let payload = serde_json::to_vec(&meta)?;
            let commit_ts = write_put(db, session, key, payload, 0, u64::MAX, Some(1))?;
            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: format!("created view {view}"),
            })
        }
        Statement::CreateIndex {
            index,
            table,
            column,
        } => {
            validate_index_name(&index)?;
            validate_table_name(&table)?;
            validate_identifier(&column)?;

            let table_meta = load_table_schema(db, session, &table)?;
            let valid_col = column.eq_ignore_ascii_case(&table_meta.pk)
                || column.eq_ignore_ascii_case("doc")
                || table_meta
                    .columns
                    .iter()
                    .any(|c| c.name.eq_ignore_ascii_case(&column));
            if !valid_col {
                return Err(SpectraError::SqlExec(format!(
                    "column {column} does not exist on table {table}"
                )));
            }

            let key = index_meta_key(&table, &index);
            if read_live_key(db, session, &key, None, None)?.is_some() {
                return Err(SpectraError::SqlExec(format!(
                    "index {index} already exists"
                )));
            }

            let meta = IndexMetadata {
                name: index.clone(),
                table,
                columns: vec![column],
                unique: false,
            };
            let payload = serde_json::to_vec(&meta)?;
            let commit_ts = write_put(db, session, key, payload, 0, u64::MAX, Some(1))?;
            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: format!("created index {index} (metadata-only)"),
            })
        }
        Statement::AlterTableAddColumn {
            table,
            column,
            column_type,
        } => {
            validate_table_name(&table)?;
            validate_identifier(&column)?;

            let mut schema = load_table_schema(db, session, &table)?;
            if schema
                .columns
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(&column))
            {
                return Ok(SqlResult::Affected {
                    rows: 0,
                    commit_ts: None,
                    message: format!("column {column} already exists on table {table}"),
                });
            }

            schema.columns.push(TableColumnMetadata {
                name: column.clone(),
                type_name: column_type.clone(),
            });
            let payload = encode_schema_metadata(&schema)?;
            let commit_ts = write_put(
                db,
                session,
                table_meta_key(&table),
                payload,
                0,
                u64::MAX,
                Some(1),
            )?;
            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: format!(
                    "altered table {table}: added column {column} {column_type} (metadata-only)"
                ),
            })
        }
        Statement::Insert { table, pk, doc } => {
            validate_table_name(&table)?;
            let _schema = load_table_schema(db, session, &table)?;
            validate_pk(&pk)?;

            validate_json_bytes(doc.as_bytes())?;
            let key = row_key(&table, &pk);
            let commit_ts = write_put(db, session, key, doc.into_bytes(), 0, u64::MAX, Some(1))?;
            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: "inserted 1 row".to_string(),
            })
        }
        Statement::ShowTables => {
            let prefix = b"__meta/table/";
            let metas = collect_visible_by_prefix(db, session, prefix, None, None)?;
            let rows = metas
                .into_keys()
                .map(|k| k[prefix.len()..].to_vec())
                .collect();
            Ok(SqlResult::Rows(rows))
        }
        Statement::Describe { table } => {
            validate_table_name(&table)?;
            let schema = load_table_schema(db, session, &table)?;
            let mut cols = schema.columns.clone();
            if cols.is_empty() {
                cols.push(TableColumnMetadata {
                    name: schema.pk.clone(),
                    type_name: "TEXT".to_string(),
                });
                cols.push(TableColumnMetadata {
                    name: "doc".to_string(),
                    type_name: "JSON".to_string(),
                });
            }

            let mut rows = Vec::with_capacity(cols.len());
            for col in cols {
                rows.push(serde_json::to_vec(&serde_json::json!({
                    "column": col.name,
                    "type": col.type_name,
                    "primary_key": col.name.eq_ignore_ascii_case(&schema.pk),
                }))?);
            }
            Ok(SqlResult::Rows(rows))
        }
        Statement::DropTable { table } => {
            validate_table_name(&table)?;
            let key = table_meta_key(&table);
            if read_live_key(db, session, &key, None, None)?.is_none() {
                return Err(SpectraError::SqlExec(format!(
                    "table {table} does not exist"
                )));
            }

            // Keep drop append-only while preventing dropped rows from resurfacing
            // if a table with the same name is recreated later.
            let row_prefix = format!("table/{table}/").into_bytes();
            let live_rows = collect_visible_by_prefix(db, session, &row_prefix, None, None)?;
            let mut writes = 0u64;
            let mut last_commit_ts = None;
            for (row_key, _) in live_rows {
                let ts = write_put(db, session, row_key, Vec::new(), 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                writes += 1;
            }

            let ts = write_put(db, session, key, Vec::new(), 0, u64::MAX, Some(1))?;
            last_commit_ts = ts.or(last_commit_ts);
            writes += 1;

            // Tombstone any index metadata for this table.
            let idx_prefix = format!("__meta/index/{table}/").into_bytes();
            let live_indexes = collect_visible_by_prefix(db, session, &idx_prefix, None, None)?;
            for (idx_key, _) in live_indexes {
                let ts = write_put(db, session, idx_key, Vec::new(), 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                writes += 1;
            }

            Ok(SqlResult::Affected {
                rows: writes,
                commit_ts: last_commit_ts,
                message: format!("dropped table {table} (metadata + row tombstones)"),
            })
        }
        Statement::DropView { view } => {
            validate_view_name(&view)?;
            let key = view_meta_key(&view);
            if read_live_key(db, session, &key, None, None)?.is_none() {
                return Err(SpectraError::SqlExec(format!("view {view} does not exist")));
            }
            let commit_ts = write_put(db, session, key, Vec::new(), 0, u64::MAX, Some(1))?;
            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: format!("dropped view {view}"),
            })
        }
        Statement::DropIndex { index, table } => {
            validate_index_name(&index)?;
            validate_table_name(&table)?;
            let key = index_meta_key(&table, &index);
            if read_live_key(db, session, &key, None, None)?.is_none() {
                return Err(SpectraError::SqlExec(format!(
                    "index {index} does not exist"
                )));
            }
            let commit_ts = write_put(db, session, key, Vec::new(), 0, u64::MAX, Some(1))?;
            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: format!("dropped index {index} on {table}"),
            })
        }
        Statement::Select {
            table,
            projection,
            join,
            pk_filter,
            as_of,
            valid_at,
            group_by_pk,
            order_by_pk,
            limit,
        } => {
            if group_by_pk && projection != SelectProjection::PkCount {
                return Err(SpectraError::SqlExec(
                    "GROUP BY pk currently requires projection pk, count(*)".to_string(),
                ));
            }
            if projection == SelectProjection::PkCount && !group_by_pk {
                return Err(SpectraError::SqlExec(
                    "pk, count(*) requires GROUP BY pk".to_string(),
                ));
            }

            if let Some(join_spec) = join {
                let _ = load_table_schema(db, session, &table)?;
                let _ = load_table_schema(db, session, &join_spec.right_table)?;

                let left_rows = fetch_rows_for_table(
                    db,
                    session,
                    &table,
                    pk_filter.as_deref(),
                    as_of,
                    valid_at,
                )?;
                let right_rows = fetch_rows_for_table(
                    db,
                    session,
                    &join_spec.right_table,
                    pk_filter.as_deref(),
                    as_of,
                    valid_at,
                )?;
                let mut joined = hash_join_on_pk(left_rows, right_rows);

                joined.sort_by(|a, b| a.pk.cmp(&b.pk));
                if matches!(order_by_pk, Some(OrderDirection::Desc)) {
                    joined.reverse();
                }
                if let Some(n) = limit {
                    let n = usize::try_from(n).unwrap_or(usize::MAX);
                    joined.truncate(n);
                }

                if projection == SelectProjection::PkCount {
                    return grouped_count_rows_from_joined(joined);
                }

                match projection {
                    SelectProjection::Doc => Err(SpectraError::SqlExec(
                        "JOIN does not support SELECT doc projection".to_string(),
                    )),
                    SelectProjection::PkDoc => {
                        let mut out = Vec::with_capacity(joined.len());
                        for row in joined {
                            out.push(serde_json::to_vec(&serde_json::json!({
                                "pk": row.pk,
                                "left_doc": decode_row_json(&row.left_doc),
                                "right_doc": decode_row_json(&row.right_doc),
                            }))?);
                        }
                        Ok(SqlResult::Rows(out))
                    }
                    SelectProjection::CountStar => {
                        Ok(SqlResult::Rows(vec![joined.len().to_string().into_bytes()]))
                    }
                    SelectProjection::PkCount => Err(SpectraError::SqlExec(
                        "invalid projection state".to_string(),
                    )),
                }
            } else {
                let (target_table, view_pk_filter, effective_as_of, effective_valid_at) =
                    resolve_select_target(db, session, &table, as_of, valid_at)?;
                let effective_pk = pk_filter.as_deref().or(view_pk_filter.as_deref());
                let mut rows = fetch_rows_for_table(
                    db,
                    session,
                    &target_table,
                    effective_pk,
                    effective_as_of,
                    effective_valid_at,
                )?;

                rows.sort_by(|a, b| a.pk.cmp(&b.pk));
                if matches!(order_by_pk, Some(OrderDirection::Desc)) {
                    rows.reverse();
                }
                if let Some(n) = limit {
                    let n = usize::try_from(n).unwrap_or(usize::MAX);
                    rows.truncate(n);
                }

                if projection == SelectProjection::PkCount {
                    return grouped_count_rows_from_visible(rows);
                }

                match projection {
                    SelectProjection::Doc => Ok(SqlResult::Rows(
                        rows.into_iter().map(|row| row.doc).collect(),
                    )),
                    SelectProjection::PkDoc => {
                        let mut out = Vec::with_capacity(rows.len());
                        for row in rows {
                            out.push(serde_json::to_vec(&serde_json::json!({
                                "pk": row.pk,
                                "doc": decode_row_json(&row.doc),
                            }))?);
                        }
                        Ok(SqlResult::Rows(out))
                    }
                    SelectProjection::CountStar => {
                        Ok(SqlResult::Rows(vec![rows.len().to_string().into_bytes()]))
                    }
                    SelectProjection::PkCount => Err(SpectraError::SqlExec(
                        "invalid projection state".to_string(),
                    )),
                }
            }
        }
        Statement::Explain(inner) => match *inner {
            Statement::Select {
                table,
                projection,
                join,
                pk_filter,
                as_of,
                valid_at,
                group_by_pk,
                order_by_pk,
                limit,
            } => {
                if projection != SelectProjection::Doc
                    || join.is_some()
                    || pk_filter.is_none()
                    || group_by_pk
                    || order_by_pk.is_some()
                    || limit.is_some()
                {
                    return Ok(SqlResult::Explain(
                        "explain not implemented for non-point SELECT".to_string(),
                    ));
                }

                let (target_table, _view_pk_filter, effective_as_of, effective_valid_at) =
                    resolve_select_target(db, session, &table, as_of, valid_at)?;
                let pk = pk_filter.ok_or_else(|| {
                    SpectraError::SqlExec("explain requires SELECT ... WHERE pk='...'".to_string())
                })?;
                let key = row_key(&target_table, &pk);
                let e = db.explain_get(&key, effective_as_of, effective_valid_at)?;
                Ok(SqlResult::Explain(format!(
                    "shard_id={} bloom_hit={:?} sstable_block={:?} commit_ts_used={} target={} source={}",
                    e.shard_id,
                    e.bloom_hit,
                    e.sstable_block,
                    e.commit_ts_used,
                    target_table,
                    table
                )))
            }
            other => Ok(SqlResult::Explain(format!(
                "explain not implemented for statement: {:?}",
                other
            ))),
        },
    }
}

fn load_table_schema(
    db: &Database,
    session: &SqlSession,
    table: &str,
) -> Result<TableSchemaMetadata> {
    let key = table_meta_key(table);
    let bytes = read_live_key(db, session, &key, None, None)?
        .ok_or_else(|| SpectraError::SqlExec(format!("table {table} does not exist")))?;
    parse_schema_metadata(&bytes)
}

fn resolve_select_target(
    db: &Database,
    session: &SqlSession,
    source: &str,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Result<(String, Option<String>, Option<u64>, Option<u64>)> {
    if read_live_key(db, session, &table_meta_key(source), None, None)?.is_some() {
        return Ok((source.to_string(), None, as_of, valid_at));
    }

    if let Some(bytes) = read_live_key(db, session, &view_meta_key(source), None, None)? {
        let meta: ViewMetadata = serde_json::from_slice(&bytes)?;
        let view_stmt = parse_sql(&meta.query)?;
        if let Statement::Select {
            table,
            pk_filter: view_pk_filter,
            as_of: view_as_of,
            valid_at: view_valid_at,
            ..
        } = view_stmt
        {
            return Ok((
                table,
                view_pk_filter,
                as_of.or(view_as_of),
                valid_at.or(view_valid_at),
            ));
        }
        return Err(SpectraError::SqlExec(format!(
            "view {source} has unsupported query shape"
        )));
    }

    Err(SpectraError::SqlExec(format!(
        "table or view {source} does not exist"
    )))
}

fn fetch_rows_for_table(
    db: &Database,
    session: &SqlSession,
    table: &str,
    pk_filter: Option<&str>,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Result<Vec<VisibleRow>> {
    if let Some(pk) = pk_filter {
        let key = row_key(table, pk);
        return match read_live_key(db, session, &key, as_of, valid_at)? {
            Some(doc) => Ok(vec![VisibleRow {
                pk: pk.to_string(),
                doc,
            }]),
            None => Ok(Vec::new()),
        };
    }
    scan_visible_rows(db, session, table, as_of, valid_at)
}

fn hash_join_on_pk(left_rows: Vec<VisibleRow>, right_rows: Vec<VisibleRow>) -> Vec<JoinedRow> {
    if left_rows.len() <= right_rows.len() {
        let mut left_map = HashMap::with_capacity(left_rows.len());
        for row in left_rows {
            left_map.insert(row.pk, row.doc);
        }
        let mut out = Vec::new();
        for right in right_rows {
            if let Some(left_doc) = left_map.get(&right.pk) {
                out.push(JoinedRow {
                    pk: right.pk,
                    left_doc: left_doc.clone(),
                    right_doc: right.doc,
                });
            }
        }
        return out;
    }

    let mut right_map = HashMap::with_capacity(right_rows.len());
    for row in right_rows {
        right_map.insert(row.pk, row.doc);
    }
    let mut out = Vec::new();
    for left in left_rows {
        if let Some(right_doc) = right_map.get(&left.pk) {
            out.push(JoinedRow {
                pk: left.pk,
                left_doc: left.doc,
                right_doc: right_doc.clone(),
            });
        }
    }
    out
}

fn grouped_count_rows_from_visible(rows: Vec<VisibleRow>) -> Result<SqlResult> {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for row in rows {
        *counts.entry(row.pk).or_insert(0) += 1;
    }

    let mut out = Vec::with_capacity(counts.len());
    for (pk, count) in counts {
        out.push(serde_json::to_vec(&serde_json::json!({
            "pk": pk,
            "count": count,
        }))?);
    }
    Ok(SqlResult::Rows(out))
}

fn grouped_count_rows_from_joined(rows: Vec<JoinedRow>) -> Result<SqlResult> {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for row in rows {
        *counts.entry(row.pk).or_insert(0) += 1;
    }

    let mut out = Vec::with_capacity(counts.len());
    for (pk, count) in counts {
        out.push(serde_json::to_vec(&serde_json::json!({
            "pk": pk,
            "count": count,
        }))?);
    }
    Ok(SqlResult::Rows(out))
}

fn decode_row_json(doc: &[u8]) -> serde_json::Value {
    serde_json::from_slice::<serde_json::Value>(doc)
        .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(doc).into_owned()))
}

fn scan_visible_rows(
    db: &Database,
    session: &SqlSession,
    table: &str,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Result<Vec<VisibleRow>> {
    let prefix = format!("table/{table}/").into_bytes();
    let docs = collect_visible_by_prefix(db, session, &prefix, as_of, valid_at)?;

    let mut out = Vec::with_capacity(docs.len());
    for (user_key, doc) in docs {
        let pk_bytes = &user_key[prefix.len()..];
        let pk = std::str::from_utf8(pk_bytes)
            .map_err(|_| SpectraError::SqlExec("row key contains non-utf8 pk".to_string()))?
            .to_string();
        out.push(VisibleRow { pk, doc });
    }
    Ok(out)
}

fn collect_visible_by_prefix(
    db: &Database,
    session: &SqlSession,
    prefix: &[u8],
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Result<BTreeMap<Vec<u8>, Vec<u8>>> {
    let mut best = BTreeMap::new();
    for row in db.scan_prefix(prefix, as_of, valid_at, None)? {
        best.insert(row.user_key, row.doc);
    }

    // Overlay staged writes as read-your-writes when querying the latest snapshot.
    if session.in_txn && as_of.is_none() {
        for staged in &session.pending {
            if !staged.key.starts_with(prefix) {
                continue;
            }
            if let Some(ts) = valid_at {
                if !(staged.valid_from <= ts && ts < staged.valid_to) {
                    continue;
                }
            }
            best.insert(staged.key.clone(), staged.value.clone());
        }
    }

    let mut out = BTreeMap::new();
    for (key, value) in best {
        if value.is_empty() {
            continue;
        }
        out.insert(key, value);
    }
    Ok(out)
}

fn read_live_key(
    db: &Database,
    session: &SqlSession,
    key: &[u8],
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Result<Option<Vec<u8>>> {
    match read_key(db, session, key, as_of, valid_at)? {
        Some(v) if v.is_empty() => Ok(None),
        other => Ok(other),
    }
}

fn read_key(
    db: &Database,
    session: &SqlSession,
    key: &[u8],
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Result<Option<Vec<u8>>> {
    if session.in_txn && as_of.is_none() {
        if let Some(v) = session.read_staged(key, valid_at) {
            return Ok(Some(v));
        }
    }
    db.get(key, as_of, valid_at)
}

fn write_put(
    db: &Database,
    session: &mut SqlSession,
    key: Vec<u8>,
    value: Vec<u8>,
    valid_from: u64,
    valid_to: u64,
    schema_version: Option<u64>,
) -> Result<Option<u64>> {
    if session.in_txn {
        session.stage_put(key, value, valid_from, valid_to, schema_version);
        Ok(None)
    } else {
        Ok(Some(db.put(
            &key,
            value,
            valid_from,
            valid_to,
            schema_version,
        )?))
    }
}

fn escape_sql_literal(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}
