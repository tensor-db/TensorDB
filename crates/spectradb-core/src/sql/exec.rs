use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::engine::db::Database;
use crate::error::{Result, SpectraError};
use crate::facet::relational::{
    encode_schema_metadata, index_meta_key, parse_schema_metadata, row_key, table_meta_key,
    validate_identifier, validate_index_name, validate_json_bytes, validate_pk,
    validate_table_name, validate_view_name, view_meta_key, IndexMetadata, TableColumnMetadata,
    TableSchemaMetadata, ViewMetadata,
};
use crate::sql::eval::{
    filter_rows, sql_value_to_json, AggAccumulator, EvalContext, SqlValue,
};
use crate::sql::parser::{
    extract_pk_eq_literal, is_aggregate_function, parse_sql, select_items_contain_aggregate,
    select_items_contain_window, split_sql_statements, BinOperator, CopyFormat, Expr, JoinSpec,
    JoinType, OrderDirection, SelectItem, Statement, TableRef,
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
pub struct VisibleRow {
    pub pk: String,
    pub doc: Vec<u8>,
}

#[derive(Debug, Clone)]
struct JoinedRow {
    pk: String,
    left_doc: Vec<u8>,
    right_doc: Vec<u8>,
    left_pk: String,
    right_pk: String,
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
        Statement::CreateTable { table, columns } => {
            validate_table_name(&table)?;
            let key = table_meta_key(&table);
            if read_live_key(db, session, &key, None, None)?.is_some() {
                return Err(SpectraError::SqlExec(format!(
                    "table {table} already exists"
                )));
            }

            // Determine if this is a legacy (pk TEXT PRIMARY KEY) or typed table
            let is_legacy = columns.len() == 1
                && columns[0].name.eq_ignore_ascii_case("pk")
                && columns[0].type_name == crate::sql::parser::SqlType::Text
                && columns[0].primary_key;

            let schema = if is_legacy {
                TableSchemaMetadata {
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
                    schema_mode: "legacy".to_string(),
                }
            } else {
                // Find the primary key column
                let pk_col = columns
                    .iter()
                    .find(|c| c.primary_key)
                    .ok_or_else(|| {
                        SpectraError::SqlExec(
                            "CREATE TABLE requires at least one PRIMARY KEY column".to_string(),
                        )
                    })?;

                let cols = columns
                    .iter()
                    .map(|c| TableColumnMetadata {
                        name: c.name.clone(),
                        type_name: c.type_name.name().to_string(),
                    })
                    .collect();

                TableSchemaMetadata {
                    pk: pk_col.name.clone(),
                    doc: "typed".to_string(),
                    columns: cols,
                    schema_mode: "typed".to_string(),
                }
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
        Statement::InsertTyped {
            table,
            columns,
            values,
        } => {
            validate_table_name(&table)?;
            let schema = load_table_schema(db, session, &table)?;

            // Build JSON from column names and value expressions
            let mut json_obj = serde_json::Map::new();
            let mut pk_val = None;

            for (col_name, val_expr) in columns.iter().zip(values.iter()) {
                let val = eval_const_expr(val_expr)?;
                let json_val = sql_value_to_json(&val);

                if col_name.eq_ignore_ascii_case(&schema.pk) {
                    pk_val = Some(match &val {
                        SqlValue::Text(s) => s.clone(),
                        SqlValue::Number(n) => n.to_string(),
                        _ => {
                            return Err(SpectraError::SqlExec(
                                "primary key must be text or number".to_string(),
                            ))
                        }
                    });
                }

                json_obj.insert(col_name.clone(), json_val);
            }

            let pk = pk_val.ok_or_else(|| {
                SpectraError::SqlExec(format!(
                    "INSERT must include primary key column '{}'",
                    schema.pk
                ))
            })?;
            validate_pk(&pk)?;

            let doc = serde_json::to_vec(&serde_json::Value::Object(json_obj))?;
            let key = row_key(&table, &pk);
            let commit_ts = write_put(db, session, key, doc, 0, u64::MAX, Some(1))?;
            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: "inserted 1 row".to_string(),
            })
        }
        Statement::Update {
            table,
            set_doc,
            set_assignments,
            filter,
            as_of,
            valid_at,
        } => {
            validate_table_name(&table)?;
            let _schema = load_table_schema(db, session, &table)?;

            let rows = fetch_rows_for_table(db, session, &table, None, as_of, valid_at)?;
            let rows = if let Some(ref f) = filter {
                filter_rows(rows, f)?
            } else {
                rows
            };

            let mut count = 0u64;
            let mut last_commit_ts = None;
            for row in &rows {
                let new_doc = if !set_assignments.is_empty() {
                    // Merge assignments into existing doc
                    let mut doc_val: serde_json::Value =
                        serde_json::from_slice(&row.doc).unwrap_or(serde_json::Value::Object(
                            serde_json::Map::new(),
                        ));
                    if let serde_json::Value::Object(ref mut map) = doc_val {
                        for (col, val_expr) in &set_assignments {
                            let mut ctx = EvalContext::new(&row.pk, &row.doc);
                            let val = ctx.eval(val_expr)?;
                            map.insert(col.clone(), sql_value_to_json(&val));
                        }
                    }
                    serde_json::to_vec(&doc_val)?
                } else {
                    // SET doc = 'json'
                    let mut ctx = EvalContext::new(&row.pk, &row.doc);
                    let val = ctx.eval(&set_doc)?;
                    match val {
                        SqlValue::Text(s) => {
                            validate_json_bytes(s.as_bytes())?;
                            s.into_bytes()
                        }
                        _ => {
                            return Err(SpectraError::SqlExec(
                                "UPDATE SET doc must evaluate to a string".to_string(),
                            ))
                        }
                    }
                };

                let key = row_key(&table, &row.pk);
                let ts = write_put(db, session, key, new_doc, 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                count += 1;
            }

            Ok(SqlResult::Affected {
                rows: count,
                commit_ts: last_commit_ts,
                message: format!("updated {count} row(s)"),
            })
        }
        Statement::Delete {
            table,
            filter,
            as_of,
            valid_at,
        } => {
            validate_table_name(&table)?;
            let _schema = load_table_schema(db, session, &table)?;

            let rows = fetch_rows_for_table(db, session, &table, None, as_of, valid_at)?;
            let rows = if let Some(ref f) = filter {
                filter_rows(rows, f)?
            } else {
                rows
            };

            let mut count = 0u64;
            let mut last_commit_ts = None;
            for row in &rows {
                let key = row_key(&table, &row.pk);
                // Write empty value as tombstone (append-only delete)
                let ts = write_put(db, session, key, Vec::new(), 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                count += 1;
            }

            Ok(SqlResult::Affected {
                rows: count,
                commit_ts: last_commit_ts,
                message: format!("deleted {count} row(s)"),
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
        Statement::CopyTo {
            table,
            path,
            format,
        } => execute_copy_to(db, session, &table, &path, format),
        Statement::CopyFrom {
            table,
            path,
            format,
        } => execute_copy_from(db, session, &table, &path, format),
        Statement::Select {
            ctes,
            from,
            items,
            join,
            filter,
            as_of,
            valid_at,
            group_by,
            having,
            order_by,
            limit,
        } => execute_select(
            db, session, ctes, from, items, join, filter, as_of, valid_at, group_by, having,
            order_by, limit,
        ),
        Statement::Explain(inner) => execute_explain(db, session, *inner),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_select(
    db: &Database,
    session: &mut SqlSession,
    ctes: Vec<crate::sql::parser::CteClause>,
    from: TableRef,
    items: Vec<SelectItem>,
    join: Option<JoinSpec>,
    filter: Option<Expr>,
    as_of: Option<u64>,
    valid_at: Option<u64>,
    group_by: Option<Vec<Expr>>,
    having: Option<Expr>,
    order_by: Option<Vec<(Expr, OrderDirection)>>,
    limit: Option<u64>,
) -> Result<SqlResult> {
    // Evaluate CTEs first
    let mut cte_data: HashMap<String, Vec<VisibleRow>> = HashMap::new();
    for cte in &ctes {
        let result = execute_stmt(db, session, *cte.query.clone())?;
        match result {
            SqlResult::Rows(rows) => {
                let visible_rows: Vec<VisibleRow> = rows
                    .into_iter()
                    .enumerate()
                    .map(|(i, doc)| VisibleRow {
                        pk: i.to_string(),
                        doc,
                    })
                    .collect();
                cte_data.insert(cte.name.clone(), visible_rows);
            }
            _ => {
                return Err(SpectraError::SqlExec(
                    "CTE must be a SELECT statement".to_string(),
                ))
            }
        }
    }

    let table_name = match &from {
        TableRef::Named(t) => t.clone(),
        TableRef::Subquery { alias, query } => {
            let result = execute_stmt(db, session, *query.clone())?;
            match result {
                SqlResult::Rows(rows) => {
                    let visible_rows: Vec<VisibleRow> = rows
                        .into_iter()
                        .enumerate()
                        .map(|(i, doc)| VisibleRow {
                            pk: i.to_string(),
                            doc,
                        })
                        .collect();
                    cte_data.insert(alias.clone(), visible_rows);
                    alias.clone()
                }
                _ => {
                    return Err(SpectraError::SqlExec(
                        "subquery in FROM must be a SELECT".to_string(),
                    ))
                }
            }
        }
    };

    // Check if this is a CTE reference
    let rows = if let Some(cte_rows) = cte_data.remove(&table_name) {
        cte_rows
    } else if let Some(join_spec) = join {
        return execute_join_select(
            db, session, &table_name, join_spec, &items, filter, as_of, valid_at, group_by,
            having, order_by, limit,
        );
    } else {
        // Resolve table or view
        let (target_table, view_pk_filter, effective_as_of, effective_valid_at) =
            resolve_select_target(db, session, &table_name, as_of, valid_at)?;

        // Optimization: extract pk = 'literal' from filter for point lookup
        let pk_from_filter = extract_pk_eq_literal(filter.as_ref());
        let effective_pk = pk_from_filter.clone().or(view_pk_filter);

        let mut rows = fetch_rows_for_table(
            db,
            session,
            &target_table,
            effective_pk.as_deref(),
            effective_as_of,
            effective_valid_at,
        )?;

        // Apply WHERE filter (if not fully resolved by pk optimization)
        if let Some(ref f) = filter {
            if pk_from_filter.is_none() || !is_simple_pk_eq(f) {
                rows = filter_rows(rows, f)?;
            }
        }
        rows
    };

    // Determine if we have aggregates or GROUP BY
    let has_aggregates = select_items_contain_aggregate(&items);
    let has_group_by = group_by.is_some();

    if has_group_by || has_aggregates {
        // Special case: plain count(*) without GROUP BY should return just the count number
        if !has_group_by && is_plain_count_star(&items) {
            return Ok(SqlResult::Rows(vec![rows.len().to_string().into_bytes()]));
        }
        return execute_grouped_select(rows, &items, group_by, having, order_by, limit);
    }

    // Apply ORDER BY
    let mut rows = rows;
    if let Some(ref orders) = order_by {
        rows.sort_by(|a, b| {
            for (expr, dir) in orders {
                let mut ctx_a = EvalContext::new(&a.pk, &a.doc);
                let mut ctx_b = EvalContext::new(&b.pk, &b.doc);
                let va = ctx_a.eval(expr).unwrap_or(SqlValue::Null);
                let vb = ctx_b.eval(expr).unwrap_or(SqlValue::Null);
                let cmp = va.to_sort_string().cmp(&vb.to_sort_string());
                let cmp = match dir {
                    OrderDirection::Asc => cmp,
                    OrderDirection::Desc => cmp.reverse(),
                };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    // Apply LIMIT
    if let Some(n) = limit {
        let n = usize::try_from(n).unwrap_or(usize::MAX);
        rows.truncate(n);
    }

    // Check for window functions
    if select_items_contain_window(&items) {
        return execute_window_select(&rows, &items);
    }

    // Project results
    project_rows(&rows, &items)
}

fn execute_grouped_select(
    rows: Vec<VisibleRow>,
    items: &[SelectItem],
    group_by: Option<Vec<Expr>>,
    having: Option<Expr>,
    order_by: Option<Vec<(Expr, OrderDirection)>>,
    limit: Option<u64>,
) -> Result<SqlResult> {
    let group_exprs = group_by.unwrap_or_default();

    // Group rows by key
    let mut groups: BTreeMap<String, Vec<&VisibleRow>> = BTreeMap::new();
    for row in &rows {
        let mut key_parts = Vec::new();
        let mut ctx = EvalContext::new(&row.pk, &row.doc);
        for expr in &group_exprs {
            let val = ctx.eval(expr)?;
            key_parts.push(val.to_sort_string());
        }
        let key = if key_parts.is_empty() {
            "__all__".to_string()
        } else {
            key_parts.join("\0")
        };
        groups.entry(key).or_default().push(row);
    }

    // If no rows and we have aggregates but no GROUP BY, still produce one row
    if groups.is_empty() && group_exprs.is_empty() {
        groups.insert("__all__".to_string(), Vec::new());
    }

    // Detect the legacy pk, count(*) GROUP BY pk pattern
    let is_legacy_pk_count = items.len() == 2
        && matches!(&items[0], SelectItem::Expr { expr: Expr::Column(c), alias: None } if c.eq_ignore_ascii_case("pk"))
        && matches!(&items[1], SelectItem::Expr { expr: Expr::Function { name, args }, alias: None }
            if name.eq_ignore_ascii_case("count") && args.len() == 1 && matches!(&args[0], Expr::Star));

    // Compute aggregates for each group
    let mut result_rows = Vec::new();
    for (_group_key, group_rows) in &groups {
        let mut row_json = serde_json::Map::new();

        // Use legacy format for pk, count(*) GROUP BY pk
        if is_legacy_pk_count {
            let pk_val = if let Some(first) = group_rows.first() {
                first.pk.clone()
            } else {
                String::new()
            };
            row_json.insert("pk".to_string(), serde_json::Value::String(pk_val));
            row_json.insert(
                "count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(group_rows.len() as u64)),
            );
        } else {
            for item in items {
                match item {
                    SelectItem::AllColumns => {
                        if let Some(first) = group_rows.first() {
                            if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&first.doc) {
                                if let serde_json::Value::Object(map) = val {
                                    for (k, v) in map {
                                        row_json.insert(k, v);
                                    }
                                }
                            }
                        }
                    }
                    SelectItem::Expr { expr, alias } => {
                        let val = eval_aggregate_expr(expr, group_rows)?;
                        let col_name = alias
                            .clone()
                            .unwrap_or_else(|| aggregate_display_name(expr));
                        row_json.insert(col_name, sql_value_to_json(&val));
                    }
                }
            }
        }

        result_rows.push((
            group_rows.first().map(|r| r.pk.clone()).unwrap_or_default(),
            serde_json::to_vec(&serde_json::Value::Object(row_json))?,
        ));
    }

    // Apply HAVING
    if let Some(ref having_expr) = having {
        // Re-evaluate having for each group - we need the group rows for this
        let mut filtered = Vec::new();
        for (idx, (_group_key, group_rows)) in groups.iter().enumerate() {
            let having_val = eval_aggregate_expr(having_expr, group_rows)?;
            if having_val.is_truthy() {
                if idx < result_rows.len() {
                    filtered.push(result_rows[idx].clone());
                }
            }
        }
        result_rows = filtered;
    }

    // Apply ORDER BY
    if let Some(ref orders) = order_by {
        result_rows.sort_by(|a, b| {
            // Parse the JSON to evaluate order expressions
            let doc_a = &a.1;
            let doc_b = &b.1;
            for (expr, dir) in orders {
                let mut ctx_a = EvalContext::new(&a.0, doc_a);
                let mut ctx_b = EvalContext::new(&b.0, doc_b);
                let va = ctx_a.eval(expr).unwrap_or(SqlValue::Null);
                let vb = ctx_b.eval(expr).unwrap_or(SqlValue::Null);
                let cmp = va.to_sort_string().cmp(&vb.to_sort_string());
                let cmp = match dir {
                    OrderDirection::Asc => cmp,
                    OrderDirection::Desc => cmp.reverse(),
                };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    // Apply LIMIT
    if let Some(n) = limit {
        let n = usize::try_from(n).unwrap_or(usize::MAX);
        result_rows.truncate(n);
    }

    Ok(SqlResult::Rows(
        result_rows.into_iter().map(|(_, doc)| doc).collect(),
    ))
}

fn eval_aggregate_expr(expr: &Expr, group_rows: &[&VisibleRow]) -> Result<SqlValue> {
    match expr {
        Expr::Function { name, args } if is_aggregate_function(name) => {
            let upper = name.to_uppercase();
            match upper.as_str() {
                "COUNT" => {
                    if args.len() == 1 && matches!(&args[0], Expr::Star) {
                        return Ok(SqlValue::Number(group_rows.len() as f64));
                    }
                    let mut count = 0u64;
                    for row in group_rows {
                        let mut ctx = EvalContext::new(&row.pk, &row.doc);
                        let val = ctx.eval(&args[0])?;
                        if !matches!(val, SqlValue::Null) {
                            count += 1;
                        }
                    }
                    Ok(SqlValue::Number(count as f64))
                }
                "SUM" | "AVG" | "MIN" | "MAX" => {
                    let mut acc = AggAccumulator::default();
                    for row in group_rows {
                        let mut ctx = EvalContext::new(&row.pk, &row.doc);
                        let val = ctx.eval(&args[0])?;
                        acc.accumulate(&val);
                    }
                    match upper.as_str() {
                        "SUM" => Ok(acc.sum_value()),
                        "AVG" => Ok(acc.avg()),
                        "MIN" => Ok(acc.min_value()),
                        "MAX" => Ok(acc.max_value()),
                        _ => unreachable!(),
                    }
                }
                _ => Err(SpectraError::SqlExec(format!(
                    "unknown aggregate function: {name}"
                ))),
            }
        }
        Expr::Column(_name) => {
            // For non-aggregate columns in GROUP BY, return the value from the first row
            if let Some(first) = group_rows.first() {
                let mut ctx = EvalContext::new(&first.pk, &first.doc);
                ctx.eval(expr)
            } else {
                Ok(SqlValue::Null)
            }
        }
        Expr::FieldAccess { .. } => {
            if let Some(first) = group_rows.first() {
                let mut ctx = EvalContext::new(&first.pk, &first.doc);
                ctx.eval(expr)
            } else {
                Ok(SqlValue::Null)
            }
        }
        Expr::BinOp { left, op, right } => {
            let lv = eval_aggregate_expr(left, group_rows)?;
            let rv = eval_aggregate_expr(right, group_rows)?;
            // Re-use eval_binop logic from eval module
            let mut ctx = EvalContext::new("", b"{}");
            let combined = Expr::BinOp {
                left: Box::new(to_literal(&lv)),
                op: *op,
                right: Box::new(to_literal(&rv)),
            };
            ctx.eval(&combined)
        }
        _ => {
            if let Some(first) = group_rows.first() {
                let mut ctx = EvalContext::new(&first.pk, &first.doc);
                ctx.eval(expr)
            } else {
                Ok(SqlValue::Null)
            }
        }
    }
}

fn to_literal(val: &SqlValue) -> Expr {
    match val {
        SqlValue::Null => Expr::Null,
        SqlValue::Bool(b) => Expr::BoolLit(*b),
        SqlValue::Number(n) => Expr::NumberLit(*n),
        SqlValue::Text(s) => Expr::StringLit(s.clone()),
    }
}

fn aggregate_display_name(expr: &Expr) -> String {
    match expr {
        Expr::Function { name, args } if is_aggregate_function(name) => {
            let lower = name.to_lowercase();
            if args.len() == 1 && matches!(&args[0], Expr::Star) {
                return lower;
            }
            if args.len() == 1 {
                let arg_name = expr_display_name(&args[0]);
                return format!("{lower}_{arg_name}");
            }
            lower
        }
        _ => expr_display_name(expr),
    }
}

fn expr_display_name(expr: &Expr) -> String {
    match expr {
        Expr::Column(name) => name.clone(),
        Expr::FieldAccess { column, path } => {
            let mut s = column.clone();
            for p in path {
                s.push('.');
                s.push_str(p);
            }
            s
        }
        Expr::Function { name, args } => {
            let arg_strs: Vec<String> = args.iter().map(expr_display_name).collect();
            format!("{}({})", name, arg_strs.join(", "))
        }
        Expr::WindowFunction { name, args, .. } => {
            let arg_strs: Vec<String> = args.iter().map(expr_display_name).collect();
            format!("{}({})", name, arg_strs.join(", "))
        }
        Expr::Star => "*".to_string(),
        _ => "?".to_string(),
    }
}

fn project_rows(rows: &[VisibleRow], items: &[SelectItem]) -> Result<SqlResult> {
    // Detect legacy projections for backward compatibility
    if let Some(legacy) = detect_legacy_projection(items) {
        return project_legacy(rows, legacy);
    }

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let mut row_json = serde_json::Map::new();
        let mut ctx = EvalContext::new(&row.pk, &row.doc);

        for item in items {
            match item {
                SelectItem::AllColumns => {
                    row_json.insert("pk".to_string(), serde_json::Value::String(row.pk.clone()));
                    let doc_val = decode_row_json(&row.doc);
                    row_json.insert("doc".to_string(), doc_val);
                }
                SelectItem::Expr { expr, alias } => {
                    let val = ctx.eval(expr)?;
                    let col_name = alias
                        .clone()
                        .unwrap_or_else(|| expr_display_name(expr));
                    row_json.insert(col_name, sql_value_to_json(&val));
                }
            }
        }

        // If single column, output just the value
        if row_json.len() == 1 {
            let val = row_json.into_values().next().unwrap();
            match val {
                serde_json::Value::String(s) => out.push(s.into_bytes()),
                other => out.push(serde_json::to_vec(&other)?),
            }
        } else {
            out.push(serde_json::to_vec(&serde_json::Value::Object(row_json))?);
        }
    }

    Ok(SqlResult::Rows(out))
}

fn detect_legacy_projection(items: &[SelectItem]) -> Option<LegacyProjection> {
    if items.len() == 1 {
        if let SelectItem::Expr { expr: Expr::Column(c), alias: None } = &items[0] {
            if c.eq_ignore_ascii_case("doc") {
                return Some(LegacyProjection::Doc);
            }
        }
        if let SelectItem::Expr {
            expr: Expr::Function { name, args },
            alias: None,
        } = &items[0]
        {
            if name.eq_ignore_ascii_case("count") && args.len() == 1 && matches!(&args[0], Expr::Star)
            {
                return Some(LegacyProjection::CountStar);
            }
        }
    }
    if items.len() == 2 {
        let first_is_pk = matches!(&items[0], SelectItem::Expr { expr: Expr::Column(c), alias: None } if c.eq_ignore_ascii_case("pk"));
        if first_is_pk {
            if let SelectItem::Expr { expr: Expr::Column(c), alias: None } = &items[1] {
                if c.eq_ignore_ascii_case("doc") {
                    return Some(LegacyProjection::PkDoc);
                }
            }
            if let SelectItem::Expr {
                expr: Expr::Function { name, args },
                alias: None,
            } = &items[1]
            {
                if name.eq_ignore_ascii_case("count")
                    && args.len() == 1
                    && matches!(&args[0], Expr::Star)
                {
                    return Some(LegacyProjection::PkCount);
                }
            }
        }
    }
    None
}

enum LegacyProjection {
    Doc,
    PkDoc,
    CountStar,
    PkCount,
}

fn project_legacy(rows: &[VisibleRow], projection: LegacyProjection) -> Result<SqlResult> {
    match projection {
        LegacyProjection::Doc => Ok(SqlResult::Rows(
            rows.iter().map(|row| row.doc.clone()).collect(),
        )),
        LegacyProjection::PkDoc => {
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                out.push(serde_json::to_vec(&serde_json::json!({
                    "pk": row.pk,
                    "doc": decode_row_json(&row.doc),
                }))?);
            }
            Ok(SqlResult::Rows(out))
        }
        LegacyProjection::CountStar => Ok(SqlResult::Rows(vec![
            rows.len().to_string().into_bytes(),
        ])),
        LegacyProjection::PkCount => {
            // This shouldn't normally be called for non-grouped queries
            // but handle it gracefully
            let mut counts: BTreeMap<String, u64> = BTreeMap::new();
            for row in rows {
                *counts.entry(row.pk.clone()).or_insert(0) += 1;
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
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_join_select(
    db: &Database,
    session: &mut SqlSession,
    left_table: &str,
    join_spec: JoinSpec,
    items: &[SelectItem],
    filter: Option<Expr>,
    as_of: Option<u64>,
    valid_at: Option<u64>,
    group_by: Option<Vec<Expr>>,
    having: Option<Expr>,
    order_by: Option<Vec<(Expr, OrderDirection)>>,
    limit: Option<u64>,
) -> Result<SqlResult> {
    let _ = load_table_schema(db, session, left_table)?;
    let _ = load_table_schema(db, session, &join_spec.right_table)?;

    // Extract pk filter from the WHERE clause for left table
    let pk_from_filter = extract_pk_eq_literal(filter.as_ref());

    let left_rows = fetch_rows_for_table(
        db,
        session,
        left_table,
        pk_from_filter.as_deref(),
        as_of,
        valid_at,
    )?;
    let right_rows = fetch_rows_for_table(
        db,
        session,
        &join_spec.right_table,
        pk_from_filter.as_deref(),
        as_of,
        valid_at,
    )?;

    let joined = match join_spec.join_type {
        JoinType::Inner => {
            if is_pk_eq_join(&join_spec.on_clause, left_table, &join_spec.right_table) {
                hash_join_on_pk(left_rows, right_rows)
            } else {
                nested_loop_join(left_rows, right_rows, &join_spec.on_clause, false, false)
            }
        }
        JoinType::Left => {
            if is_pk_eq_join(&join_spec.on_clause, left_table, &join_spec.right_table) {
                left_hash_join_on_pk(left_rows, right_rows)
            } else {
                nested_loop_join(left_rows, right_rows, &join_spec.on_clause, true, false)
            }
        }
        JoinType::Right => {
            if is_pk_eq_join(&join_spec.on_clause, left_table, &join_spec.right_table) {
                right_hash_join_on_pk(left_rows, right_rows)
            } else {
                nested_loop_join(left_rows, right_rows, &join_spec.on_clause, false, true)
            }
        }
        JoinType::Cross => cross_join(left_rows, right_rows),
    };

    // Apply WHERE filter on joined rows
    let joined = if let Some(ref f) = filter {
        if pk_from_filter.is_none() || !is_simple_pk_eq(f) {
            let visible: Vec<VisibleRow> = joined
                .iter()
                .map(|j| VisibleRow {
                    pk: j.pk.clone(),
                    doc: build_joined_doc(&j.left_doc, &j.right_doc),
                })
                .collect();
            let filtered = filter_rows(visible, f)?;
            let filtered_pks: std::collections::HashSet<usize> = filtered
                .iter()
                .enumerate()
                .map(|(i, _)| i)
                .collect();
            joined
                .into_iter()
                .enumerate()
                .filter(|(i, _)| filtered_pks.contains(i))
                .map(|(_, r)| r)
                .collect()
        } else {
            joined
        }
    } else {
        joined
    };

    // Handle GROUP BY and aggregates for joins
    if group_by.is_some() || select_items_contain_aggregate(items) {
        // Special case: plain count(*) without GROUP BY
        if group_by.is_none() && is_plain_count_star(items) {
            return Ok(SqlResult::Rows(vec![joined.len().to_string().into_bytes()]));
        }
        let rows: Vec<VisibleRow> = joined
            .iter()
            .map(|j| VisibleRow {
                pk: j.pk.clone(),
                doc: build_joined_doc(&j.left_doc, &j.right_doc),
            })
            .collect();
        return execute_grouped_select(rows, items, group_by, having, order_by, limit);
    }

    // Apply ORDER BY
    let mut joined = joined;
    if let Some(ref orders) = order_by {
        joined.sort_by(|a, b| {
            for (expr, dir) in orders {
                let doc_a = build_joined_doc(&a.left_doc, &a.right_doc);
                let doc_b = build_joined_doc(&b.left_doc, &b.right_doc);
                let mut ctx_a = EvalContext::new(&a.pk, &doc_a);
                let mut ctx_b = EvalContext::new(&b.pk, &doc_b);
                let va = ctx_a.eval(expr).unwrap_or(SqlValue::Null);
                let vb = ctx_b.eval(expr).unwrap_or(SqlValue::Null);
                let cmp = va.to_sort_string().cmp(&vb.to_sort_string());
                let cmp = match dir {
                    OrderDirection::Asc => cmp,
                    OrderDirection::Desc => cmp.reverse(),
                };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            a.pk.cmp(&b.pk)
        });
    } else {
        joined.sort_by(|a, b| a.pk.cmp(&b.pk));
    }

    // Apply LIMIT
    if let Some(n) = limit {
        let n = usize::try_from(n).unwrap_or(usize::MAX);
        joined.truncate(n);
    }

    // Project joined rows
    project_joined_rows(&joined, items)
}

fn project_joined_rows(rows: &[JoinedRow], items: &[SelectItem]) -> Result<SqlResult> {
    // Detect legacy projection for backward compat
    if let Some(legacy) = detect_legacy_projection(items) {
        match legacy {
            LegacyProjection::PkDoc => {
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    out.push(serde_json::to_vec(&serde_json::json!({
                        "pk": row.pk,
                        "left_doc": decode_row_json(&row.left_doc),
                        "right_doc": decode_row_json(&row.right_doc),
                    }))?);
                }
                return Ok(SqlResult::Rows(out));
            }
            LegacyProjection::CountStar => {
                return Ok(SqlResult::Rows(vec![rows.len().to_string().into_bytes()]));
            }
            LegacyProjection::PkCount => {
                let mut counts: BTreeMap<String, u64> = BTreeMap::new();
                for row in rows {
                    *counts.entry(row.pk.clone()).or_insert(0) += 1;
                }
                let mut out = Vec::with_capacity(counts.len());
                for (pk, count) in counts {
                    out.push(serde_json::to_vec(&serde_json::json!({
                        "pk": pk,
                        "count": count,
                    }))?);
                }
                return Ok(SqlResult::Rows(out));
            }
            LegacyProjection::Doc => {
                return Err(SpectraError::SqlExec(
                    "JOIN does not support SELECT doc projection".to_string(),
                ));
            }
        }
    }

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let combined_doc = build_joined_doc(&row.left_doc, &row.right_doc);
        let mut row_json = serde_json::Map::new();
        let mut ctx = EvalContext::new(&row.pk, &combined_doc);

        for item in items {
            match item {
                SelectItem::AllColumns => {
                    row_json.insert("pk".to_string(), serde_json::Value::String(row.pk.clone()));
                    row_json.insert("left_doc".to_string(), decode_row_json(&row.left_doc));
                    row_json.insert("right_doc".to_string(), decode_row_json(&row.right_doc));
                }
                SelectItem::Expr { expr, alias } => {
                    let val = ctx.eval(expr)?;
                    let col_name = alias.clone().unwrap_or_else(|| expr_display_name(expr));
                    row_json.insert(col_name, sql_value_to_json(&val));
                }
            }
        }

        if row_json.len() == 1 {
            let val = row_json.into_values().next().unwrap();
            match val {
                serde_json::Value::String(s) => out.push(s.into_bytes()),
                other => out.push(serde_json::to_vec(&other)?),
            }
        } else {
            out.push(serde_json::to_vec(&serde_json::Value::Object(row_json))?);
        }
    }

    Ok(SqlResult::Rows(out))
}

fn build_joined_doc(left_doc: &[u8], right_doc: &[u8]) -> Vec<u8> {
    let left = decode_row_json(left_doc);
    let right = decode_row_json(right_doc);
    serde_json::to_vec(&serde_json::json!({
        "left_doc": left,
        "right_doc": right,
    }))
    .unwrap_or_default()
}

fn is_pk_eq_join(on_clause: &Option<Expr>, left_table: &str, right_table: &str) -> bool {
    match on_clause {
        Some(Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        }) => {
            let left_is_pk = match left.as_ref() {
                Expr::Column(c) => c.eq_ignore_ascii_case("pk"),
                Expr::FieldAccess { column, path } => {
                    (column.eq_ignore_ascii_case(left_table) || column.eq_ignore_ascii_case("pk"))
                        && path.len() == 1
                        && path[0].eq_ignore_ascii_case("pk")
                }
                _ => false,
            };
            let right_is_pk = match right.as_ref() {
                Expr::Column(c) => c.eq_ignore_ascii_case("pk"),
                Expr::FieldAccess { column, path } => {
                    (column.eq_ignore_ascii_case(right_table)
                        || column.eq_ignore_ascii_case("pk"))
                        && path.len() == 1
                        && path[0].eq_ignore_ascii_case("pk")
                }
                _ => false,
            };
            left_is_pk && right_is_pk
        }
        _ => false,
    }
}

fn is_simple_pk_eq(expr: &Expr) -> bool {
    extract_pk_eq_literal(Some(expr)).is_some()
}

fn is_plain_count_star(items: &[SelectItem]) -> bool {
    items.len() == 1
        && matches!(
            &items[0],
            SelectItem::Expr {
                expr: Expr::Function { name, args },
                alias: None,
            } if name.eq_ignore_ascii_case("count") && args.len() == 1 && matches!(&args[0], Expr::Star)
        )
}

fn hash_join_on_pk(left_rows: Vec<VisibleRow>, right_rows: Vec<VisibleRow>) -> Vec<JoinedRow> {
    if left_rows.len() <= right_rows.len() {
        let mut left_map = HashMap::with_capacity(left_rows.len());
        for row in &left_rows {
            left_map.insert(row.pk.clone(), row.doc.clone());
        }
        let mut out = Vec::new();
        for right in &right_rows {
            if let Some(left_doc) = left_map.get(&right.pk) {
                out.push(JoinedRow {
                    pk: right.pk.clone(),
                    left_doc: left_doc.clone(),
                    right_doc: right.doc.clone(),
                    left_pk: right.pk.clone(),
                    right_pk: right.pk.clone(),
                });
            }
        }
        return out;
    }

    let mut right_map = HashMap::with_capacity(right_rows.len());
    for row in &right_rows {
        right_map.insert(row.pk.clone(), row.doc.clone());
    }
    let mut out = Vec::new();
    for left in &left_rows {
        if let Some(right_doc) = right_map.get(&left.pk) {
            out.push(JoinedRow {
                pk: left.pk.clone(),
                left_doc: left.doc.clone(),
                right_doc: right_doc.clone(),
                left_pk: left.pk.clone(),
                right_pk: left.pk.clone(),
            });
        }
    }
    out
}

fn left_hash_join_on_pk(
    left_rows: Vec<VisibleRow>,
    right_rows: Vec<VisibleRow>,
) -> Vec<JoinedRow> {
    let mut right_map = HashMap::with_capacity(right_rows.len());
    for row in &right_rows {
        right_map.insert(row.pk.clone(), row.doc.clone());
    }
    let mut out = Vec::new();
    for left in &left_rows {
        match right_map.get(&left.pk) {
            Some(right_doc) => {
                out.push(JoinedRow {
                    pk: left.pk.clone(),
                    left_doc: left.doc.clone(),
                    right_doc: right_doc.clone(),
                    left_pk: left.pk.clone(),
                    right_pk: left.pk.clone(),
                });
            }
            None => {
                out.push(JoinedRow {
                    pk: left.pk.clone(),
                    left_doc: left.doc.clone(),
                    right_doc: b"null".to_vec(),
                    left_pk: left.pk.clone(),
                    right_pk: String::new(),
                });
            }
        }
    }
    out
}

fn right_hash_join_on_pk(
    left_rows: Vec<VisibleRow>,
    right_rows: Vec<VisibleRow>,
) -> Vec<JoinedRow> {
    let mut left_map = HashMap::with_capacity(left_rows.len());
    for row in &left_rows {
        left_map.insert(row.pk.clone(), row.doc.clone());
    }
    let mut out = Vec::new();
    for right in &right_rows {
        match left_map.get(&right.pk) {
            Some(left_doc) => {
                out.push(JoinedRow {
                    pk: right.pk.clone(),
                    left_doc: left_doc.clone(),
                    right_doc: right.doc.clone(),
                    left_pk: right.pk.clone(),
                    right_pk: right.pk.clone(),
                });
            }
            None => {
                out.push(JoinedRow {
                    pk: right.pk.clone(),
                    left_doc: b"null".to_vec(),
                    right_doc: right.doc.clone(),
                    left_pk: String::new(),
                    right_pk: right.pk.clone(),
                });
            }
        }
    }
    out
}

fn nested_loop_join(
    left_rows: Vec<VisibleRow>,
    right_rows: Vec<VisibleRow>,
    on_clause: &Option<Expr>,
    left_outer: bool,
    right_outer: bool,
) -> Vec<JoinedRow> {
    let mut out = Vec::new();
    let mut left_matched = vec![false; left_rows.len()];
    let mut right_matched = vec![false; right_rows.len()];

    for (li, left) in left_rows.iter().enumerate() {
        for (ri, right) in right_rows.iter().enumerate() {
            let matches = match on_clause {
                Some(expr) => {
                    let combined_doc = build_joined_doc(&left.doc, &right.doc);
                    let mut ctx = EvalContext::new(&left.pk, &combined_doc);
                    ctx.eval(expr).map(|v| v.is_truthy()).unwrap_or(false)
                }
                None => true, // No ON clause = cross join behavior
            };
            if matches {
                left_matched[li] = true;
                right_matched[ri] = true;
                out.push(JoinedRow {
                    pk: left.pk.clone(),
                    left_doc: left.doc.clone(),
                    right_doc: right.doc.clone(),
                    left_pk: left.pk.clone(),
                    right_pk: right.pk.clone(),
                });
            }
        }
    }

    if left_outer {
        for (li, left) in left_rows.iter().enumerate() {
            if !left_matched[li] {
                out.push(JoinedRow {
                    pk: left.pk.clone(),
                    left_doc: left.doc.clone(),
                    right_doc: b"null".to_vec(),
                    left_pk: left.pk.clone(),
                    right_pk: String::new(),
                });
            }
        }
    }

    if right_outer {
        for (ri, right) in right_rows.iter().enumerate() {
            if !right_matched[ri] {
                out.push(JoinedRow {
                    pk: right.pk.clone(),
                    left_doc: b"null".to_vec(),
                    right_doc: right.doc.clone(),
                    left_pk: String::new(),
                    right_pk: right.pk.clone(),
                });
            }
        }
    }

    out
}

fn cross_join(left_rows: Vec<VisibleRow>, right_rows: Vec<VisibleRow>) -> Vec<JoinedRow> {
    let mut out = Vec::new();
    for left in &left_rows {
        for right in &right_rows {
            out.push(JoinedRow {
                pk: format!("{}_{}", left.pk, right.pk),
                left_doc: left.doc.clone(),
                right_doc: right.doc.clone(),
                left_pk: left.pk.clone(),
                right_pk: right.pk.clone(),
            });
        }
    }
    out
}

fn execute_window_select(
    rows: &[VisibleRow],
    items: &[SelectItem],
) -> Result<SqlResult> {
    // For each window function in the select items, compute the values for all rows
    // Then project with the window values injected

    // Collect window function specs from items
    struct WindowFuncSpec {
        item_index: usize,
        name: String,
        args: Vec<Expr>,
        partition_by: Vec<Expr>,
        order_by: Vec<(Expr, OrderDirection)>,
        alias: Option<String>,
    }

    let mut window_specs = Vec::new();
    for (idx, item) in items.iter().enumerate() {
        if let SelectItem::Expr {
            expr:
                Expr::WindowFunction {
                    name,
                    args,
                    partition_by,
                    order_by,
                },
            alias,
        } = item
        {
            window_specs.push(WindowFuncSpec {
                item_index: idx,
                name: name.clone(),
                args: args.clone(),
                partition_by: partition_by.clone(),
                order_by: order_by.clone(),
                alias: alias.clone(),
            });
        }
    }

    // Compute window values: Vec<row_idx -> value> for each window function
    let mut window_values: HashMap<usize, Vec<SqlValue>> = HashMap::new();

    for spec in &window_specs {
        let values = compute_window_function(rows, &spec.name, &spec.args, &spec.partition_by, &spec.order_by)?;
        window_values.insert(spec.item_index, values);
    }

    // Project with window values
    let mut out = Vec::with_capacity(rows.len());
    for (row_idx, row) in rows.iter().enumerate() {
        let mut row_json = serde_json::Map::new();
        let mut ctx = EvalContext::new(&row.pk, &row.doc);

        for (item_idx, item) in items.iter().enumerate() {
            match item {
                SelectItem::AllColumns => {
                    row_json.insert("pk".to_string(), serde_json::Value::String(row.pk.clone()));
                    let doc_val = decode_row_json(&row.doc);
                    row_json.insert("doc".to_string(), doc_val);
                }
                SelectItem::Expr { expr, alias } => {
                    let col_name = alias
                        .clone()
                        .unwrap_or_else(|| expr_display_name(expr));

                    // Check if this is a window function item
                    if let Some(vals) = window_values.get(&item_idx) {
                        row_json.insert(col_name, sql_value_to_json(&vals[row_idx]));
                    } else {
                        let val = ctx.eval(expr)?;
                        row_json.insert(col_name, sql_value_to_json(&val));
                    }
                }
            }
        }

        if row_json.len() == 1 {
            let val = row_json.into_values().next().unwrap();
            match val {
                serde_json::Value::String(s) => out.push(s.into_bytes()),
                other => out.push(serde_json::to_vec(&other)?),
            }
        } else {
            out.push(serde_json::to_vec(&serde_json::Value::Object(row_json))?);
        }
    }

    Ok(SqlResult::Rows(out))
}

fn compute_window_function(
    rows: &[VisibleRow],
    func_name: &str,
    _args: &[Expr],
    partition_by: &[Expr],
    order_by: &[(Expr, OrderDirection)],
) -> Result<Vec<SqlValue>> {
    let n = rows.len();
    let mut result = vec![SqlValue::Null; n];

    // Compute partition keys for all rows
    let mut partition_keys: Vec<String> = Vec::with_capacity(n);
    for row in rows {
        let key = if partition_by.is_empty() {
            String::new() // All rows in one partition
        } else {
            let mut ctx = EvalContext::new(&row.pk, &row.doc);
            let mut parts = Vec::new();
            for expr in partition_by {
                let v = ctx.eval(expr)?;
                parts.push(v.to_sort_string());
            }
            parts.join("|")
        };
        partition_keys.push(key);
    }

    // Compute sort keys within partitions
    let mut sort_keys: Vec<String> = Vec::with_capacity(n);
    for row in rows {
        let key = if order_by.is_empty() {
            String::new()
        } else {
            let mut ctx = EvalContext::new(&row.pk, &row.doc);
            let mut parts = Vec::new();
            for (expr, _dir) in order_by {
                let v = ctx.eval(expr)?;
                parts.push(v.to_sort_string());
            }
            parts.join("|")
        };
        sort_keys.push(key);
    }

    // Group row indices by partition
    let mut partitions: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, key) in partition_keys.iter().enumerate() {
        partitions.entry(key.clone()).or_default().push(idx);
    }

    // Sort each partition by order_by key
    for (_pkey, indices) in &mut partitions {
        if !order_by.is_empty() {
            indices.sort_by(|&a, &b| {
                for (oi, (_expr, dir)) in order_by.iter().enumerate() {
                    let mut ctx_a = EvalContext::new(&rows[a].pk, &rows[a].doc);
                    let mut ctx_b = EvalContext::new(&rows[b].pk, &rows[b].doc);
                    let va = ctx_a.eval(&order_by[oi].0).unwrap_or(SqlValue::Null);
                    let vb = ctx_b.eval(&order_by[oi].0).unwrap_or(SqlValue::Null);
                    let cmp = va.to_sort_string().cmp(&vb.to_sort_string());
                    let cmp = match dir {
                        OrderDirection::Asc => cmp,
                        OrderDirection::Desc => cmp.reverse(),
                    };
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }
    }

    match func_name.to_uppercase().as_str() {
        "ROW_NUMBER" => {
            for indices in partitions.values() {
                for (rank, &row_idx) in indices.iter().enumerate() {
                    result[row_idx] = SqlValue::Number((rank + 1) as f64);
                }
            }
        }
        "RANK" => {
            for indices in partitions.values() {
                let mut rank = 1usize;
                for (i, &row_idx) in indices.iter().enumerate() {
                    if i > 0 && sort_keys[row_idx] != sort_keys[indices[i - 1]] {
                        rank = i + 1;
                    }
                    result[row_idx] = SqlValue::Number(rank as f64);
                }
            }
        }
        "DENSE_RANK" => {
            for indices in partitions.values() {
                let mut rank = 1usize;
                for (i, &row_idx) in indices.iter().enumerate() {
                    if i > 0 && sort_keys[row_idx] != sort_keys[indices[i - 1]] {
                        rank += 1;
                    }
                    result[row_idx] = SqlValue::Number(rank as f64);
                }
            }
        }
        "LEAD" => {
            for indices in partitions.values() {
                for (i, &row_idx) in indices.iter().enumerate() {
                    if i + 1 < indices.len() {
                        let next_idx = indices[i + 1];
                        // Default: evaluate the first arg from the next row
                        let mut ctx = EvalContext::new(&rows[next_idx].pk, &rows[next_idx].doc);
                        if let Some(arg) = _args.first() {
                            result[row_idx] = ctx.eval(arg)?;
                        }
                    }
                    // Last row in partition: stays Null
                }
            }
        }
        "LAG" => {
            for indices in partitions.values() {
                for (i, &row_idx) in indices.iter().enumerate() {
                    if i > 0 {
                        let prev_idx = indices[i - 1];
                        let mut ctx = EvalContext::new(&rows[prev_idx].pk, &rows[prev_idx].doc);
                        if let Some(arg) = _args.first() {
                            result[row_idx] = ctx.eval(arg)?;
                        }
                    }
                    // First row in partition: stays Null
                }
            }
        }
        other => {
            return Err(SpectraError::SqlExec(format!(
                "unsupported window function: {other}"
            )));
        }
    }

    Ok(result)
}

fn execute_explain(
    db: &Database,
    session: &mut SqlSession,
    inner: Statement,
) -> Result<SqlResult> {
    match inner {
        Statement::Select {
            from: TableRef::Named(table),
            items,
            join,
            filter,
            as_of,
            valid_at,
            ..
        } => {
            let is_doc_only = detect_legacy_projection(&items)
                .map(|l| matches!(l, LegacyProjection::Doc))
                .unwrap_or(false);

            if !is_doc_only || join.is_some() || filter.is_none() {
                return Ok(SqlResult::Explain(
                    "explain not implemented for non-point SELECT".to_string(),
                ));
            }

            let pk = extract_pk_eq_literal(filter.as_ref());
            let pk = match pk {
                Some(pk) => pk,
                None => {
                    return Ok(SqlResult::Explain(
                        "explain not implemented for non-point SELECT".to_string(),
                    ))
                }
            };

            let (target_table, _view_pk_filter, effective_as_of, effective_valid_at) =
                resolve_select_target(db, session, &table, as_of, valid_at)?;
            let key = row_key(&target_table, &pk);
            let e = db.explain_get(&key, effective_as_of, effective_valid_at)?;
            Ok(SqlResult::Explain(format!(
                "shard_id={} bloom_hit={:?} sstable_block={:?} commit_ts_used={} target={} source={}",
                e.shard_id, e.bloom_hit, e.sstable_block, e.commit_ts_used, target_table, table
            )))
        }
        other => Ok(SqlResult::Explain(format!(
            "explain not implemented for statement: {:?}",
            other
        ))),
    }
}

fn execute_copy_to(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    path: &str,
    format: CopyFormat,
) -> Result<SqlResult> {
    validate_table_name(table)?;
    let schema = load_table_schema(db, session, table)?;
    let rows = fetch_rows_for_table(db, session, table, None, None, None)?;

    let mut output = String::new();
    match format {
        CopyFormat::Csv => {
            // Header
            let col_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
            output.push_str(&col_names.join(","));
            output.push('\n');
            for row in &rows {
                output.push_str(&row.pk);
                output.push(',');
                output.push_str(&String::from_utf8_lossy(&row.doc));
                output.push('\n');
            }
        }
        CopyFormat::Json => {
            let mut arr = Vec::new();
            for row in &rows {
                arr.push(serde_json::json!({
                    "pk": row.pk,
                    "doc": decode_row_json(&row.doc),
                }));
            }
            output = serde_json::to_string_pretty(&arr).unwrap_or_default();
        }
        CopyFormat::Ndjson => {
            for row in &rows {
                let obj = serde_json::json!({
                    "pk": row.pk,
                    "doc": decode_row_json(&row.doc),
                });
                output.push_str(&serde_json::to_string(&obj).unwrap_or_default());
                output.push('\n');
            }
        }
    }

    std::fs::write(path, &output)?;
    Ok(SqlResult::Affected {
        rows: rows.len() as u64,
        commit_ts: None,
        message: format!("exported {} rows to {}", rows.len(), path),
    })
}

fn execute_copy_from(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    path: &str,
    format: CopyFormat,
) -> Result<SqlResult> {
    validate_table_name(table)?;
    let _schema = load_table_schema(db, session, table)?;

    let content = std::fs::read_to_string(path)?;
    let mut count = 0u64;
    let mut last_commit_ts = None;

    match format {
        CopyFormat::Csv => {
            let mut lines = content.lines();
            let _header = lines.next(); // Skip header
            for line in lines {
                if line.trim().is_empty() {
                    continue;
                }
                let mut parts = line.splitn(2, ',');
                let pk = parts.next().unwrap_or("").trim().to_string();
                let doc = parts.next().unwrap_or("{}").trim().to_string();
                if pk.is_empty() {
                    continue;
                }
                validate_pk(&pk)?;
                validate_json_bytes(doc.as_bytes())?;
                let key = row_key(table, &pk);
                let ts = write_put(db, session, key, doc.into_bytes(), 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                count += 1;
            }
        }
        CopyFormat::Json => {
            let arr: Vec<serde_json::Value> = serde_json::from_str(&content)?;
            for val in arr {
                let pk = val
                    .get("pk")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| SpectraError::SqlExec("JSON row missing 'pk'".to_string()))?
                    .to_string();
                let doc = val
                    .get("doc")
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .unwrap_or_else(|| "{}".to_string());
                validate_pk(&pk)?;
                let key = row_key(table, &pk);
                let ts =
                    write_put(db, session, key, doc.into_bytes(), 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                count += 1;
            }
        }
        CopyFormat::Ndjson => {
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let val: serde_json::Value = serde_json::from_str(line)?;
                let pk = val
                    .get("pk")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        SpectraError::SqlExec("NDJSON row missing 'pk'".to_string())
                    })?
                    .to_string();
                let doc = val
                    .get("doc")
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .unwrap_or_else(|| "{}".to_string());
                validate_pk(&pk)?;
                let key = row_key(table, &pk);
                let ts =
                    write_put(db, session, key, doc.into_bytes(), 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                count += 1;
            }
        }
    }

    Ok(SqlResult::Affected {
        rows: count,
        commit_ts: last_commit_ts,
        message: format!("imported {count} rows from {path}"),
    })
}

fn eval_const_expr(expr: &Expr) -> Result<SqlValue> {
    match expr {
        Expr::StringLit(s) => Ok(SqlValue::Text(s.clone())),
        Expr::NumberLit(n) => Ok(SqlValue::Number(*n)),
        Expr::BoolLit(b) => Ok(SqlValue::Bool(*b)),
        Expr::Null => Ok(SqlValue::Null),
        _ => {
            // Evaluate with empty context for const expressions
            let mut ctx = EvalContext::new("", b"{}");
            ctx.eval(expr)
        }
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
            from: TableRef::Named(table),
            filter,
            as_of: view_as_of,
            valid_at: view_valid_at,
            ..
        } = view_stmt
        {
            let view_pk_filter = extract_pk_eq_literal(filter.as_ref());
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
