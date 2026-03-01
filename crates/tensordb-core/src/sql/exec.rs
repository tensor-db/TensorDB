use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::engine::db::Database;
use crate::error::{Result, TensorError};
use crate::facet::relational::{
    encode_schema_metadata, fts_index_meta_key, fts_index_prefix_for_table, index_entry_key,
    index_entry_prefix, index_meta_key, index_meta_prefix_for_table, index_scan_prefix,
    parse_schema_metadata, row_key, table_meta_key, ts_table_meta_key, validate_identifier,
    validate_index_name, validate_json_bytes, validate_pk, validate_table_name, validate_view_name,
    view_meta_key, FtsIndexMetadata, IndexMetadata, TableColumnMetadata, TableSchemaMetadata,
    TimeseriesMetadata, ViewMetadata,
};
use crate::facet::vector_persistence::{
    parse_vector_literal, vector_data_key, vector_data_prefix, vector_index_meta_key,
    VectorIndexMetadata, VectorRecord,
};
use crate::sql::eval::{
    filter_rows, parse_interval_seconds, sql_value_to_json, AggAccumulator, EvalContext,
    FirstAccumulator, LastAccumulator, SqlValue,
};

/// (target_table, view_pk_filter, effective_as_of, effective_valid_at)
type ResolvedTarget = (String, Option<String>, Option<u64>, Option<u64>);

/// Typed comparison for ORDER BY: uses numeric ordering when both values are numbers,
/// falls back to string comparison otherwise.
fn compare_sort_values(a: &SqlValue, b: &SqlValue) -> std::cmp::Ordering {
    match (a, b) {
        (SqlValue::Null, SqlValue::Null) => std::cmp::Ordering::Equal,
        (SqlValue::Null, _) => std::cmp::Ordering::Less,
        (_, SqlValue::Null) => std::cmp::Ordering::Greater,
        (SqlValue::Number(na), SqlValue::Number(nb)) => {
            na.partial_cmp(nb).unwrap_or(std::cmp::Ordering::Equal)
        }
        (SqlValue::Decimal(da), SqlValue::Decimal(db)) => da.cmp(db),
        (SqlValue::Decimal(da), SqlValue::Number(nb)) => {
            use std::str::FromStr;
            match rust_decimal::Decimal::from_str(&nb.to_string()) {
                Ok(db) => da.cmp(&db),
                Err(_) => std::cmp::Ordering::Equal,
            }
        }
        (SqlValue::Number(na), SqlValue::Decimal(db)) => {
            use std::str::FromStr;
            match rust_decimal::Decimal::from_str(&na.to_string()) {
                Ok(da) => da.cmp(db),
                Err(_) => std::cmp::Ordering::Equal,
            }
        }
        _ => a.to_sort_string().cmp(&b.to_sort_string()),
    }
}
use crate::sql::parser::{
    extract_pk_eq_literal, is_aggregate_function, parse_sql, select_items_contain_aggregate,
    select_items_contain_window, split_sql_statements, BinOperator, CopyFormat, Expr, JoinSpec,
    JoinType, OrderDirection, SelectItem, SetOpType, Statement, TableRef,
};
use crate::sql::planner::{explain_plan_with_stats, generate_plan_with_stats, plan, PlannerStats};

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

/// Per-column statistics collected by `ANALYZE`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStatistics {
    pub name: String,
    pub distinct_count: u64,
    pub null_count: u64,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    /// Top-N most frequent values: (value, count).
    pub top_values: Vec<(String, u64)>,
}

/// Statistics collected by `ANALYZE <table>`, stored under `__meta/stats/<table>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStats {
    pub row_count: u64,
    pub approx_byte_size: u64,
    pub avg_row_bytes: u64,
    pub last_updated_ms: u64,
    /// Per-column statistics (populated for typed tables).
    #[serde(default)]
    pub columns: Vec<ColumnStatistics>,
}

fn stats_meta_key(table: &str) -> Vec<u8> {
    format!("__meta/stats/{table}").into_bytes()
}

/// Collect all table names referenced in a statement (for stats lookup).
fn collect_statement_tables(stmt: &Statement) -> Vec<String> {
    match stmt {
        Statement::Select { from, joins, .. } => {
            let mut tables = Vec::new();
            if let TableRef::Named(t) = from {
                tables.push(t.clone());
            }
            for js in joins {
                tables.push(js.right_table.clone());
            }
            tables
        }
        _ => Vec::new(),
    }
}

/// Build PlannerStats from persisted ANALYZE data for the given tables.
fn build_planner_stats(db: &Database, session: &SqlSession, tables: &[&str]) -> PlannerStats {
    let mut stats = PlannerStats::new();
    for table in tables {
        // Load row count from ANALYZE stats
        let key = stats_meta_key(table);
        if let Ok(Some(data)) = read_live_key(db, session, &key, None, None) {
            if let Ok(ts) = serde_json::from_slice::<TableStats>(&data) {
                stats
                    .table_row_counts
                    .insert(table.to_lowercase(), ts.row_count);
            }
        }
        // Load index metadata
        let prefix = index_meta_prefix_for_table(table);
        if let Ok(index_metas) = db.scan_prefix(&prefix, None, None, None) {
            let mut cols = Vec::new();
            for meta_row in &index_metas {
                if let Ok(meta) = serde_json::from_slice::<IndexMetadata>(&meta_row.doc) {
                    if meta.columns.len() == 1 {
                        cols.push(meta.columns[0].clone());
                    }
                }
            }
            if !cols.is_empty() {
                stats.indexed_columns.insert(table.to_lowercase(), cols);
            }
        }
    }
    stats
}

/// A prepared statement that caches the parsed AST, avoiding re-parsing on
/// repeated executions. Supports parameter placeholders ($1, $2, ...).
#[derive(Debug, Clone)]
pub struct PreparedStatement {
    stmt: Statement,
    param_count: usize,
}

impl PreparedStatement {
    /// Parse and plan a SQL statement, caching the result for repeated execution.
    pub fn new(sql: &str) -> Result<Self> {
        let stmt = parse_sql(sql)?;
        let param_count = count_params(&stmt);
        let stmt = plan(stmt)?;
        Ok(Self { stmt, param_count })
    }

    /// Number of parameter placeholders ($1, $2, ...) in this statement.
    pub fn param_count(&self) -> usize {
        self.param_count
    }

    /// Execute this prepared statement against the given database.
    pub fn execute(&self, db: &Database) -> Result<SqlResult> {
        self.execute_with_params(db, &[])
    }

    /// Execute with bound parameter values. Parameters are positional ($1 = params[0]).
    pub fn execute_with_params(&self, db: &Database, params: &[&str]) -> Result<SqlResult> {
        let stmt = if params.is_empty() {
            self.stmt.clone()
        } else {
            if params.len() < self.param_count {
                return Err(TensorError::SqlExec(format!(
                    "expected {} parameters, got {}",
                    self.param_count,
                    params.len()
                )));
            }
            substitute_params(self.stmt.clone(), params)
        };

        let mut session = SqlSession::default();
        let result = execute_stmt(db, &mut session, stmt)?;
        if session.in_txn {
            return Err(TensorError::SqlExec(
                "transaction left open; issue COMMIT or ROLLBACK".to_string(),
            ));
        }
        Ok(result)
    }
}

/// Count parameter placeholders ($1, $2, ...) in a statement's expressions.
fn count_params(stmt: &Statement) -> usize {
    let mut max_param = 0usize;
    visit_exprs_in_stmt(stmt, &mut |e| {
        if let Expr::Column(name) = e {
            if let Some(n) = parse_param_index(name) {
                max_param = max_param.max(n);
            }
        }
    });
    max_param
}

/// Parse "$N" as parameter index N (1-based).
fn parse_param_index(name: &str) -> Option<usize> {
    name.strip_prefix('$')
        .and_then(|rest| rest.parse::<usize>().ok())
}

/// Substitute parameter placeholders with literal values.
fn substitute_params(stmt: Statement, params: &[&str]) -> Statement {
    match stmt {
        Statement::Select {
            ctes,
            from,
            items,
            joins,
            filter,
            as_of,
            valid_at,
            as_of_epoch,
            temporal,
            group_by,
            having,
            order_by,
            limit,
        } => Statement::Select {
            ctes,
            from,
            items: items
                .into_iter()
                .map(|i| subst_select_item(i, params))
                .collect(),
            joins,
            filter: filter.map(|f| subst_expr(f, params)),
            as_of,
            valid_at,
            as_of_epoch,
            temporal,
            group_by: group_by.map(|g| g.into_iter().map(|e| subst_expr(e, params)).collect()),
            having: having.map(|h| subst_expr(h, params)),
            order_by: order_by.map(|o| {
                o.into_iter()
                    .map(|(e, d)| (subst_expr(e, params), d))
                    .collect()
            }),
            limit,
        },
        other => other, // Other statement types: pass through
    }
}

fn subst_select_item(item: SelectItem, params: &[&str]) -> SelectItem {
    match item {
        SelectItem::Expr { expr, alias } => SelectItem::Expr {
            expr: subst_expr(expr, params),
            alias,
        },
        other => other,
    }
}

fn subst_expr(expr: Expr, params: &[&str]) -> Expr {
    match expr {
        Expr::Column(ref name) => {
            if let Some(idx) = parse_param_index(name) {
                if idx >= 1 && idx <= params.len() {
                    // Try to parse as number, otherwise use string
                    if let Ok(n) = params[idx - 1].parse::<f64>() {
                        Expr::NumberLit(n)
                    } else {
                        Expr::StringLit(params[idx - 1].to_string())
                    }
                } else {
                    expr
                }
            } else {
                expr
            }
        }
        Expr::BinOp { left, op, right } => Expr::BinOp {
            left: Box::new(subst_expr(*left, params)),
            op,
            right: Box::new(subst_expr(*right, params)),
        },
        Expr::Not(inner) => Expr::Not(Box::new(subst_expr(*inner, params))),
        Expr::Function { name, args } => Expr::Function {
            name,
            args: args.into_iter().map(|a| subst_expr(a, params)).collect(),
        },
        Expr::IsNull {
            expr: inner,
            negated,
        } => Expr::IsNull {
            expr: Box::new(subst_expr(*inner, params)),
            negated,
        },
        Expr::Between {
            expr: inner,
            low,
            high,
            negated,
        } => Expr::Between {
            expr: Box::new(subst_expr(*inner, params)),
            low: Box::new(subst_expr(*low, params)),
            high: Box::new(subst_expr(*high, params)),
            negated,
        },
        Expr::InList {
            expr: inner,
            list,
            negated,
        } => Expr::InList {
            expr: Box::new(subst_expr(*inner, params)),
            list: list.into_iter().map(|e| subst_expr(e, params)).collect(),
            negated,
        },
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => Expr::Case {
            operand: operand.map(|o| Box::new(subst_expr(*o, params))),
            when_clauses: when_clauses
                .into_iter()
                .map(|(w, t)| (subst_expr(w, params), subst_expr(t, params)))
                .collect(),
            else_clause: else_clause.map(|e| Box::new(subst_expr(*e, params))),
        },
        Expr::Cast {
            expr: inner,
            target_type,
        } => Expr::Cast {
            expr: Box::new(subst_expr(*inner, params)),
            target_type,
        },
        Expr::FieldAccess { column, path } => {
            if let Some(idx) = parse_param_index(&column) {
                if idx >= 1 && idx <= params.len() {
                    Expr::StringLit(params[idx - 1].to_string())
                } else {
                    Expr::FieldAccess { column, path }
                }
            } else {
                Expr::FieldAccess { column, path }
            }
        }
        Expr::WindowFunction {
            name,
            args,
            partition_by,
            order_by,
        } => Expr::WindowFunction {
            name,
            args: args.into_iter().map(|a| subst_expr(a, params)).collect(),
            partition_by: partition_by
                .into_iter()
                .map(|e| subst_expr(e, params))
                .collect(),
            order_by: order_by
                .into_iter()
                .map(|(e, d)| (subst_expr(e, params), d))
                .collect(),
        },
        // Literals and Star pass through unchanged
        other => other,
    }
}

/// Visit all expressions in a statement (for counting params, etc.).
fn visit_exprs_in_stmt(stmt: &Statement, visitor: &mut dyn FnMut(&Expr)) {
    if let Statement::Select {
        items,
        filter,
        group_by,
        having,
        order_by,
        ..
    } = stmt
    {
        for item in items {
            if let SelectItem::Expr { expr, .. } = item {
                visit_expr(expr, visitor);
            }
        }
        if let Some(f) = filter {
            visit_expr(f, visitor);
        }
        if let Some(g) = group_by {
            for e in g {
                visit_expr(e, visitor);
            }
        }
        if let Some(h) = having {
            visit_expr(h, visitor);
        }
        if let Some(o) = order_by {
            for (e, _) in o {
                visit_expr(e, visitor);
            }
        }
    }
}

fn visit_expr(expr: &Expr, visitor: &mut dyn FnMut(&Expr)) {
    visitor(expr);
    match expr {
        Expr::BinOp { left, right, .. } => {
            visit_expr(left, visitor);
            visit_expr(right, visitor);
        }
        Expr::Not(inner) => visit_expr(inner, visitor),
        Expr::Function { args, .. } => {
            for a in args {
                visit_expr(a, visitor);
            }
        }
        Expr::IsNull { expr: inner, .. } => visit_expr(inner, visitor),
        Expr::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            visit_expr(inner, visitor);
            visit_expr(low, visitor);
            visit_expr(high, visitor);
        }
        Expr::InList {
            expr: inner, list, ..
        } => {
            visit_expr(inner, visitor);
            for e in list {
                visit_expr(e, visitor);
            }
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(o) = operand {
                visit_expr(o, visitor);
            }
            for (w, t) in when_clauses {
                visit_expr(w, visitor);
                visit_expr(t, visitor);
            }
            if let Some(e) = else_clause {
                visit_expr(e, visitor);
            }
        }
        Expr::Cast { expr: inner, .. } => visit_expr(inner, visitor),
        Expr::WindowFunction {
            args,
            partition_by,
            order_by,
            ..
        } => {
            for a in args {
                visit_expr(a, visitor);
            }
            for e in partition_by {
                visit_expr(e, visitor);
            }
            for (e, _) in order_by {
                visit_expr(e, visitor);
            }
        }
        _ => {}
    }
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
    /// Stack of savepoints: (name, pending.len() at savepoint)
    savepoints: Vec<(String, usize)>,
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
    #[allow(dead_code)]
    left_pk: String,
    #[allow(dead_code)]
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
            return Err(TensorError::SqlExec(
                "COMMIT requires an active transaction".to_string(),
            ));
        }

        let rows = self.pending.len() as u64;
        // Advance global epoch atomically — all writes in this transaction
        // share this commit_epoch, providing cross-shard atomicity.
        let commit_epoch = if rows > 0 {
            Some(db.advance_epoch())
        } else {
            None
        };
        let mut max_commit_ts: u64 = 0;
        while let Some(p) = self.pending.pop_front() {
            let ts = db.put(&p.key, p.value, p.valid_from, p.valid_to, p.schema_version)?;
            if ts > max_commit_ts {
                max_commit_ts = ts;
            }
        }
        // Write TXN_COMMIT marker to WAL — crash recovery uses this to identify
        // complete transactions. Writes without a trailing marker are rolled back.
        if let Some(epoch) = commit_epoch {
            let marker_key = format!("__txn_commit/{epoch}").into_bytes();
            // Store the max commit_ts so PITR queries can resolve epoch → commit_ts.
            // Using max (not last) ensures all writes in the transaction are captured.
            let marker_val = max_commit_ts.to_le_bytes().to_vec();
            let _ = db.put(&marker_key, marker_val, 0, u64::MAX, None)?;
        }
        self.in_txn = false;
        self.savepoints.clear();
        let last_ts = if max_commit_ts > 0 {
            Some(max_commit_ts)
        } else {
            None
        };
        Ok((rows, last_ts))
    }

    fn rollback(&mut self) -> Result<u64> {
        if !self.in_txn {
            return Err(TensorError::SqlExec(
                "ROLLBACK requires an active transaction".to_string(),
            ));
        }
        let rows = self.pending.len() as u64;
        self.pending.clear();
        self.savepoints.clear();
        self.in_txn = false;
        Ok(rows)
    }

    fn savepoint(&mut self, name: String) -> Result<()> {
        if !self.in_txn {
            return Err(TensorError::SqlExec(
                "SAVEPOINT requires an active transaction".to_string(),
            ));
        }
        self.savepoints.push((name, self.pending.len()));
        Ok(())
    }

    fn rollback_to(&mut self, name: &str) -> Result<u64> {
        if !self.in_txn {
            return Err(TensorError::SqlExec(
                "ROLLBACK TO requires an active transaction".to_string(),
            ));
        }
        // Find the savepoint
        let pos = self
            .savepoints
            .iter()
            .rposition(|(n, _)| n == name)
            .ok_or_else(|| TensorError::SqlExec(format!("savepoint '{name}' does not exist")))?;
        let (_, pending_len) = self.savepoints[pos].clone();
        // Remove this savepoint and all later ones
        self.savepoints.truncate(pos);
        // Discard writes after the savepoint
        let discarded = self.pending.len() - pending_len;
        self.pending.truncate(pending_len);
        Ok(discarded as u64)
    }

    fn release_savepoint(&mut self, name: &str) -> Result<()> {
        if !self.in_txn {
            return Err(TensorError::SqlExec(
                "RELEASE requires an active transaction".to_string(),
            ));
        }
        let pos = self
            .savepoints
            .iter()
            .rposition(|(n, _)| n == name)
            .ok_or_else(|| TensorError::SqlExec(format!("savepoint '{name}' does not exist")))?;
        self.savepoints.remove(pos);
        Ok(())
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
        return Err(TensorError::SqlExec(
            "transaction left open; issue COMMIT or ROLLBACK".to_string(),
        ));
    }

    last.ok_or_else(|| TensorError::SqlExec("no statements executed".to_string()))
}

fn execute_stmt(db: &Database, session: &mut SqlSession, stmt: Statement) -> Result<SqlResult> {
    match stmt {
        Statement::Begin => {
            if session.in_txn {
                return Err(TensorError::SqlExec(
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
        Statement::Savepoint { name } => {
            session.savepoint(name.clone())?;
            Ok(SqlResult::Affected {
                rows: 0,
                commit_ts: None,
                message: format!("SAVEPOINT {name}"),
            })
        }
        Statement::RollbackTo { name } => {
            let discarded = session.rollback_to(&name)?;
            Ok(SqlResult::Affected {
                rows: discarded,
                commit_ts: None,
                message: format!("ROLLBACK TO {name} ({discarded} writes discarded)"),
            })
        }
        Statement::ReleaseSavepoint { name } => {
            session.release_savepoint(&name)?;
            Ok(SqlResult::Affected {
                rows: 0,
                commit_ts: None,
                message: format!("RELEASE {name}"),
            })
        }
        Statement::CreateTable { table, columns } => {
            validate_table_name(&table)?;
            let key = table_meta_key(&table);
            if read_live_key(db, session, &key, None, None)?.is_some() {
                return Err(TensorError::SqlExec(format!(
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
                let pk_col = columns.iter().find(|c| c.primary_key).ok_or_else(|| {
                    TensorError::SqlExec(
                        "CREATE TABLE requires at least one PRIMARY KEY column".to_string(),
                    )
                })?;

                let cols = columns
                    .iter()
                    .map(|c| {
                        let type_str = match c.type_name {
                            crate::sql::parser::SqlType::Vector { dims } => {
                                format!("VECTOR({dims})")
                            }
                            crate::sql::parser::SqlType::Decimal { precision, scale } => {
                                format!("DECIMAL({precision},{scale})")
                            }
                            _ => c.type_name.name().to_string(),
                        };
                        TableColumnMetadata {
                            name: c.name.clone(),
                            type_name: type_str,
                        }
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
                return Err(TensorError::SqlExec(format!(
                    "source table {source} does not exist"
                )));
            }

            let key = view_meta_key(&view);
            if read_live_key(db, session, &key, None, None)?.is_some() {
                return Err(TensorError::SqlExec(format!("view {view} already exists")));
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
            columns,
            unique,
        } => {
            validate_index_name(&index)?;
            validate_table_name(&table)?;
            for col in &columns {
                validate_identifier(col)?;
            }

            let table_meta = load_table_schema(db, session, &table)?;
            for col in &columns {
                let valid_col = col.eq_ignore_ascii_case(&table_meta.pk)
                    || col.eq_ignore_ascii_case("doc")
                    || table_meta
                        .columns
                        .iter()
                        .any(|c| c.name.eq_ignore_ascii_case(col));
                if !valid_col {
                    return Err(TensorError::SqlExec(format!(
                        "column {col} does not exist on table {table}"
                    )));
                }
            }

            let key = index_meta_key(&table, &index);
            if read_live_key(db, session, &key, None, None)?.is_some() {
                return Err(TensorError::SqlExec(format!(
                    "index {index} already exists"
                )));
            }

            let meta = IndexMetadata {
                name: index.clone(),
                table: table.clone(),
                columns: columns.clone(),
                unique,
            };
            let payload = serde_json::to_vec(&meta)?;
            let commit_ts = write_put(db, session, key, payload, 0, u64::MAX, Some(1))?;

            // Build the index: scan all existing rows and create index entries
            let rows = fetch_rows_for_table(db, session, &table, None, None, None)?;
            for row in &rows {
                let index_val = extract_index_value(&row.doc, &columns)?;
                if unique {
                    // Check for duplicates
                    let idx_prefix = index_entry_prefix(&table, &index, &index_val);
                    let existing = db.scan_prefix(&idx_prefix, None, None, Some(1))?;
                    if !existing.is_empty() {
                        return Err(TensorError::SqlExec(format!(
                            "UNIQUE constraint violated: duplicate value '{index_val}' in index {index}"
                        )));
                    }
                }
                let idx_key = index_entry_key(&table, &index, &index_val, &row.pk);
                write_put(db, session, idx_key, Vec::new(), 0, u64::MAX, Some(1))?;
            }

            Ok(SqlResult::Affected {
                rows: 1,
                commit_ts,
                message: format!(
                    "created index {index} on {table}({}) ({} rows indexed)",
                    columns.join(", "),
                    rows.len()
                ),
            })
        }
        Statement::CreateFulltextIndex {
            index,
            table,
            columns,
        } => execute_create_fulltext_index(db, session, &index, &table, &columns),
        Statement::DropFulltextIndex { index, table } => {
            execute_drop_fulltext_index(db, session, &index, &table)
        }
        Statement::CreateVectorIndex {
            index,
            table,
            column,
            index_type,
            params,
        } => execute_create_vector_index(db, session, &index, &table, &column, index_type, &params),
        Statement::DropVectorIndex { index, table } => {
            execute_drop_vector_index(db, session, &index, &table)
        }
        Statement::CreateTimeseriesTable {
            table,
            columns,
            bucket_interval,
        } => execute_create_timeseries_table(db, session, &table, columns, &bucket_interval),
        Statement::SetOp { op, left, right } => execute_set_op(db, session, op, *left, *right),
        Statement::InsertReturning {
            table,
            columns,
            values,
            returning,
        } => execute_insert_returning(db, session, &table, columns, values, &returning),
        Statement::CreateTableAs { table, query } => {
            execute_create_table_as(db, session, &table, *query)
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
            let commit_ts = write_put(
                db,
                session,
                key,
                doc.as_bytes().to_vec(),
                0,
                u64::MAX,
                Some(1),
            )?;
            // Update FTS indexes
            update_fts_indexes(db, session, &table, &pk, doc.as_bytes())?;
            // Update time-series buckets
            update_ts_buckets(db, session, &table, doc.as_bytes())?;
            // Update secondary indexes
            update_secondary_indexes(db, session, &table, &pk, doc.as_bytes())?;
            // Update vector indexes
            update_vector_indexes(db, session, &table, &pk, doc.as_bytes())?;
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
                            return Err(TensorError::SqlExec(
                                "primary key must be text or number".to_string(),
                            ))
                        }
                    });
                }

                json_obj.insert(col_name.clone(), json_val);
            }

            let pk = pk_val.ok_or_else(|| {
                TensorError::SqlExec(format!(
                    "INSERT must include primary key column '{}'",
                    schema.pk
                ))
            })?;
            validate_pk(&pk)?;

            let doc = serde_json::to_vec(&serde_json::Value::Object(json_obj))?;
            let key = row_key(&table, &pk);
            let commit_ts = write_put(db, session, key, doc.clone(), 0, u64::MAX, Some(1))?;
            // Update FTS indexes
            update_fts_indexes(db, session, &table, &pk, &doc)?;
            // Update time-series buckets
            update_ts_buckets(db, session, &table, &doc)?;
            // Update secondary indexes
            update_secondary_indexes(db, session, &table, &pk, &doc)?;
            // Update vector indexes
            update_vector_indexes(db, session, &table, &pk, &doc)?;
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
                    let mut doc_val: serde_json::Value = serde_json::from_slice(&row.doc)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
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
                            return Err(TensorError::SqlExec(
                                "UPDATE SET doc must evaluate to a string".to_string(),
                            ))
                        }
                    }
                };

                // Remove old index entries, add new ones
                remove_secondary_indexes(db, session, &table, &row.pk, &row.doc)?;
                remove_vector_indexes(db, session, &table, &row.pk)?;
                let key = row_key(&table, &row.pk);
                let ts = write_put(db, session, key, new_doc.clone(), 0, u64::MAX, Some(1))?;
                update_secondary_indexes(db, session, &table, &row.pk, &new_doc)?;
                update_vector_indexes(db, session, &table, &row.pk, &new_doc)?;
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
                // Remove index entries before deleting the row
                remove_secondary_indexes(db, session, &table, &row.pk, &row.doc)?;
                remove_vector_indexes(db, session, &table, &row.pk)?;
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
                return Err(TensorError::SqlExec(format!(
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

            // Clean up index metadata and index entries
            let idx_prefix = format!("__meta/index/{table}/").into_bytes();
            let live_indexes = collect_visible_by_prefix(db, session, &idx_prefix, None, None)?;
            for (idx_key, idx_val) in &live_indexes {
                // Clean up __idx/ entries for this index
                if let Ok(meta) = serde_json::from_slice::<IndexMetadata>(idx_val) {
                    let entry_prefix = index_scan_prefix(&table, &meta.name);
                    let entries =
                        collect_visible_by_prefix(db, session, &entry_prefix, None, None)?;
                    for (entry_key, _) in entries {
                        let ts =
                            write_put(db, session, entry_key, Vec::new(), 0, u64::MAX, Some(1))?;
                        last_commit_ts = ts.or(last_commit_ts);
                        writes += 1;
                    }
                }
                let ts = write_put(
                    db,
                    session,
                    idx_key.clone(),
                    Vec::new(),
                    0,
                    u64::MAX,
                    Some(1),
                )?;
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
                return Err(TensorError::SqlExec(format!("view {view} does not exist")));
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
                return Err(TensorError::SqlExec(format!(
                    "index {index} does not exist"
                )));
            }
            // Remove all __idx/<table>/<index>/ entries
            let idx_prefix = index_scan_prefix(&table, &index);
            let idx_entries = collect_visible_by_prefix(db, session, &idx_prefix, None, None)?;
            let mut writes = 0u64;
            let mut last_commit_ts = None;
            for (entry_key, _) in idx_entries {
                let ts = write_put(db, session, entry_key, Vec::new(), 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                writes += 1;
            }
            // Remove index metadata
            let ts = write_put(db, session, key, Vec::new(), 0, u64::MAX, Some(1))?;
            last_commit_ts = ts.or(last_commit_ts);
            writes += 1;
            Ok(SqlResult::Affected {
                rows: writes,
                commit_ts: last_commit_ts,
                message: format!("dropped index {index} on {table} ({writes} entries removed)"),
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
            joins,
            filter,
            as_of,
            valid_at,
            as_of_epoch,
            temporal,
            group_by,
            having,
            order_by,
            limit,
        } => execute_select(
            db,
            session,
            ctes,
            from,
            items,
            joins,
            filter,
            as_of,
            valid_at,
            as_of_epoch,
            temporal,
            group_by,
            having,
            order_by,
            limit,
        ),
        Statement::Explain(inner) => execute_explain(db, session, *inner),
        Statement::ExplainAnalyze(inner) => execute_explain_analyze(db, session, *inner),
        Statement::ExplainAi { key } => execute_explain_ai(db, session, &key),
        Statement::Analyze { table } => execute_analyze(db, session, &table),
        Statement::Ask { question } => execute_ask(db, &question),
        Statement::Backup { dest, since_epoch } => execute_backup(db, &dest, since_epoch),
        Statement::Restore { src } => execute_restore(db, &src),
    }
}

/// Resolve SQL:2011 temporal clauses into effective as_of / valid_at values.
///
/// Temporal clauses override legacy AS OF / VALID AT when present.
/// For range queries (FROM..TO, BETWEEN..AND), we currently map them to
/// the start of the range (conservative: returns the earliest version in range).
/// SYSTEM_TIME ALL maps to as_of=None (latest system time — all versions
/// are preserved through the append-only ledger and can be retrieved via scan).
fn resolve_temporal_clauses(
    mut as_of: Option<u64>,
    mut valid_at: Option<u64>,
    temporal: &[crate::sql::parser::TemporalClause],
) -> Result<(Option<u64>, Option<u64>)> {
    use crate::sql::parser::TemporalClause;
    for clause in temporal {
        match clause {
            TemporalClause::SystemTimeAsOf(ts) => {
                as_of = Some(*ts);
            }
            TemporalClause::SystemTimeFromTo(t1, _t2) => {
                // Return versions visible at start of range
                as_of = Some(*t1);
            }
            TemporalClause::SystemTimeBetween(t1, _t2) => {
                // Return versions visible at start of range
                as_of = Some(*t1);
            }
            TemporalClause::SystemTimeAll => {
                // All historical versions — don't filter by system time
                as_of = None;
            }
            TemporalClause::ApplicationTimeAsOf(ts) => {
                valid_at = Some(*ts);
            }
            TemporalClause::ApplicationTimeFromTo(t1, _t2) => {
                valid_at = Some(*t1);
            }
            TemporalClause::ApplicationTimeBetween(t1, _t2) => {
                valid_at = Some(*t1);
            }
        }
    }
    Ok((as_of, valid_at))
}

#[allow(clippy::too_many_arguments)]
fn execute_select(
    db: &Database,
    session: &mut SqlSession,
    ctes: Vec<crate::sql::parser::CteClause>,
    from: TableRef,
    items: Vec<SelectItem>,
    joins: Vec<JoinSpec>,
    filter: Option<Expr>,
    as_of: Option<u64>,
    valid_at: Option<u64>,
    as_of_epoch: Option<u64>,
    temporal: Vec<crate::sql::parser::TemporalClause>,
    group_by: Option<Vec<Expr>>,
    having: Option<Expr>,
    order_by: Option<Vec<(Expr, OrderDirection)>>,
    limit: Option<u64>,
) -> Result<SqlResult> {
    // Resolve SQL:2011 temporal clauses into as_of/valid_at overrides
    let (mut as_of, valid_at) = resolve_temporal_clauses(as_of, valid_at, &temporal)?;

    // PITR: resolve epoch to commit timestamp.
    // Each transaction's TXN_COMMIT marker stores the max commit_ts of all writes
    // in that transaction. Using that as as_of shows exactly the state at that epoch.
    // advance_epoch() bumps all shard counters, ensuring writes after epoch N have
    // commit_ts > max_commit_ts_of_epoch_N.
    if let Some(epoch) = as_of_epoch {
        let marker_key = format!("__txn_commit/{epoch}");
        if let Some(v) = db.get(marker_key.as_bytes(), None, None)? {
            if v.len() >= 8 {
                as_of = Some(u64::from_le_bytes(v[..8].try_into().unwrap()));
            }
        } else {
            // Epoch not found — use epoch value directly (approximate PITR)
            as_of = Some(epoch);
        }
    }
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
                return Err(TensorError::SqlExec(
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
                    return Err(TensorError::SqlExec(
                        "subquery in FROM must be a SELECT".to_string(),
                    ))
                }
            }
        }
        TableRef::TableFunction { name, args, alias } => {
            let rows = if name.eq_ignore_ascii_case("vector_search") {
                execute_vector_search_fn(db, session, args)?
            } else {
                execute_table_function(name, args)?
            };
            let virt_name = alias.clone().unwrap_or_else(|| name.clone());
            cte_data.insert(virt_name.clone(), rows);
            virt_name
        }
    };

    // AI virtual tables: intercept before normal table resolution
    if table_name == "ai_top_risks" {
        return execute_ai_top_risks(db, filter.as_ref(), limit);
    }
    if table_name == "ai_cluster_summary" {
        return execute_ai_cluster_summary(db, filter.as_ref());
    }

    // Check if this is a CTE reference (or table function result)
    let rows = if let Some(cte_rows) = cte_data.remove(&table_name) {
        // Apply WHERE filter on CTE/table-function rows
        if let Some(ref f) = filter {
            filter_rows(cte_rows, f)?
        } else {
            cte_rows
        }
    } else if !joins.is_empty() {
        return execute_join_select(
            db,
            session,
            &table_name,
            joins,
            &items,
            filter,
            as_of,
            valid_at,
            group_by,
            having,
            order_by,
            limit,
        );
    } else {
        // Resolve table or view
        let (target_table, view_pk_filter, effective_as_of, effective_valid_at) =
            resolve_select_target(db, session, &table_name, as_of, valid_at)?;

        // Check for MATCH() in filter — use FTS posting lists for pre-filtering
        let match_pks = if let Some(ref f) = filter {
            extract_match_filter(f, &target_table, db)?
        } else {
            None
        };

        // Optimization: extract pk = 'literal' from filter for point lookup
        let pk_from_filter = extract_pk_eq_literal(filter.as_ref());
        let effective_pk = pk_from_filter.clone().or(view_pk_filter);

        let mut rows = if let Some(ref fts_pks) = match_pks {
            // FTS-driven fetch: only fetch rows that match FTS query
            let mut fts_rows = Vec::new();
            for pk in fts_pks {
                let fetched = fetch_rows_for_table(
                    db,
                    session,
                    &target_table,
                    Some(pk),
                    effective_as_of,
                    effective_valid_at,
                )?;
                fts_rows.extend(fetched);
            }
            fts_rows
        } else if effective_pk.is_some() {
            // PK point lookup — already extracted from filter
            fetch_rows_for_table(
                db,
                session,
                &target_table,
                effective_pk.as_deref(),
                effective_as_of,
                effective_valid_at,
            )?
        } else if let Some(idx_rows) = filter.as_ref().and_then(|f| {
            try_index_scan(
                db,
                session,
                &target_table,
                f,
                effective_as_of,
                effective_valid_at,
            )
            .or_else(|| {
                try_composite_index_scan(
                    db,
                    session,
                    &target_table,
                    f,
                    effective_as_of,
                    effective_valid_at,
                )
            })
            .or_else(|| {
                try_index_in_scan(
                    db,
                    session,
                    &target_table,
                    f,
                    effective_as_of,
                    effective_valid_at,
                )
            })
            .or_else(|| {
                try_index_range_scan(
                    db,
                    session,
                    &target_table,
                    f,
                    effective_as_of,
                    effective_valid_at,
                )
            })
        }) {
            idx_rows
        } else {
            fetch_rows_for_table(
                db,
                session,
                &target_table,
                None,
                effective_as_of,
                effective_valid_at,
            )?
        };

        // Apply WHERE filter (if not fully resolved by pk optimization, FTS, or index scan)
        if let Some(ref f) = filter {
            if match_pks.is_none() && (pk_from_filter.is_none() || !is_simple_pk_eq(f)) {
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

    // Check for window functions BEFORE ORDER BY/LIMIT so they see all rows
    let mut rows = rows;
    if select_items_contain_window(&items) {
        return execute_window_select(&mut rows, &items, order_by.as_ref(), limit);
    }

    // Apply ORDER BY
    if let Some(ref orders) = order_by {
        rows.sort_by(|a, b| {
            for (expr, dir) in orders {
                let mut ctx_a = EvalContext::new(&a.pk, &a.doc);
                let mut ctx_b = EvalContext::new(&b.pk, &b.doc);
                let va = ctx_a.eval(expr).unwrap_or(SqlValue::Null);
                let vb = ctx_b.eval(expr).unwrap_or(SqlValue::Null);
                let cmp = compare_sort_values(&va, &vb);
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
    let group_by_ref = group_by.clone();
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
    for group_rows in groups.values() {
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
                            if let Ok(serde_json::Value::Object(map)) =
                                serde_json::from_slice::<serde_json::Value>(&first.doc)
                            {
                                for (k, v) in map {
                                    row_json.insert(k, v);
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
        for (idx, group_rows) in groups.values().enumerate() {
            let having_val = eval_aggregate_expr(having_expr, group_rows)?;
            if having_val.is_truthy() && idx < result_rows.len() {
                filtered.push(result_rows[idx].clone());
            }
        }
        result_rows = filtered;
    }

    // Detect TIME_BUCKET_GAPFILL in GROUP BY and fill gaps
    if let Some(ref gb) = group_by_ref {
        if let Some(gapfill_info) = detect_gapfill(gb) {
            result_rows = apply_gapfill(result_rows, &gapfill_info, items)?;
        }
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
                let cmp = compare_sort_values(&va, &vb);
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
                "FIRST" => {
                    // FIRST(value, timestamp) — value at smallest timestamp
                    if args.len() < 2 {
                        return Err(TensorError::SqlExec(
                            "FIRST() requires 2 arguments: FIRST(value, timestamp)".to_string(),
                        ));
                    }
                    let mut acc = FirstAccumulator::default();
                    for row in group_rows {
                        let mut ctx = EvalContext::new(&row.pk, &row.doc);
                        let val = ctx.eval(&args[0])?;
                        let ts = ctx.eval(&args[1])?;
                        acc.accumulate(&val, &ts);
                    }
                    Ok(acc.result())
                }
                "LAST" => {
                    // LAST(value, timestamp) — value at largest timestamp
                    if args.len() < 2 {
                        return Err(TensorError::SqlExec(
                            "LAST() requires 2 arguments: LAST(value, timestamp)".to_string(),
                        ));
                    }
                    let mut acc = LastAccumulator::default();
                    for row in group_rows {
                        let mut ctx = EvalContext::new(&row.pk, &row.doc);
                        let val = ctx.eval(&args[0])?;
                        let ts = ctx.eval(&args[1])?;
                        acc.accumulate(&val, &ts);
                    }
                    Ok(acc.result())
                }
                "STRING_AGG" | "GROUP_CONCAT" => {
                    // STRING_AGG(expr, separator)
                    let separator = if args.len() >= 2 {
                        match eval_const_expr(&args[1])? {
                            SqlValue::Text(s) => s,
                            _ => ",".to_string(),
                        }
                    } else {
                        ",".to_string()
                    };
                    let mut parts = Vec::new();
                    for row in group_rows {
                        let mut ctx = EvalContext::new(&row.pk, &row.doc);
                        let val = ctx.eval(&args[0])?;
                        if !matches!(val, SqlValue::Null) {
                            parts.push(val.to_sort_string());
                        }
                    }
                    if parts.is_empty() {
                        Ok(SqlValue::Null)
                    } else {
                        Ok(SqlValue::Text(parts.join(&separator)))
                    }
                }
                "STDDEV" | "STDDEV_POP" | "STDDEV_SAMP" | "VARIANCE" | "VAR_POP" | "VAR_SAMP" => {
                    let mut values = Vec::new();
                    for row in group_rows {
                        let mut ctx = EvalContext::new(&row.pk, &row.doc);
                        let val = ctx.eval(&args[0])?;
                        if let Some(n) = val.to_f64() {
                            values.push(n);
                        }
                    }
                    if values.is_empty() {
                        return Ok(SqlValue::Null);
                    }
                    let n = values.len() as f64;
                    let mean = values.iter().sum::<f64>() / n;
                    let sum_sq: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();

                    let is_sample = matches!(upper.as_str(), "STDDEV_SAMP" | "VAR_SAMP" | "STDDEV");
                    let divisor = if is_sample && values.len() > 1 {
                        n - 1.0
                    } else {
                        n
                    };
                    let variance = sum_sq / divisor;

                    if matches!(upper.as_str(), "STDDEV" | "STDDEV_POP" | "STDDEV_SAMP") {
                        Ok(SqlValue::Number(variance.sqrt()))
                    } else {
                        Ok(SqlValue::Number(variance))
                    }
                }
                "APPROX_COUNT_DISTINCT" => {
                    let mut hll = crate::storage::zone_map::HyperLogLog::new();
                    for row in group_rows {
                        let mut ctx = EvalContext::new(&row.pk, &row.doc);
                        let val = ctx.eval(&args[0])?;
                        if !matches!(val, SqlValue::Null) {
                            hll.add(val.to_sort_string().as_bytes());
                        }
                    }
                    Ok(SqlValue::Number(hll.count() as f64))
                }
                _ => Err(TensorError::SqlExec(format!(
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
        Expr::Function { name, args } => {
            // Non-aggregate function wrapping aggregate args (e.g. ROUND(VAR_POP(age), 2))
            // Recursively resolve args through aggregate eval, then compute scalar function
            let resolved_args: Vec<Expr> = args
                .iter()
                .map(|a| eval_aggregate_expr(a, group_rows).map(|v| to_literal(&v)))
                .collect::<Result<Vec<_>>>()?;
            let resolved = Expr::Function {
                name: name.clone(),
                args: resolved_args,
            };
            let mut ctx = EvalContext::new("", b"{}");
            ctx.eval(&resolved)
        }
        Expr::Cast {
            expr: inner,
            target_type,
        } => {
            let val = eval_aggregate_expr(inner, group_rows)?;
            let resolved = Expr::Cast {
                expr: Box::new(to_literal(&val)),
                target_type: target_type.clone(),
            };
            let mut ctx = EvalContext::new("", b"{}");
            ctx.eval(&resolved)
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            // For CASE in aggregate context, evaluate each branch through aggregate eval
            let resolved_operand = if let Some(op) = operand {
                Some(Box::new(to_literal(&eval_aggregate_expr(op, group_rows)?)))
            } else {
                None
            };
            let resolved_whens: Vec<(Expr, Expr)> = when_clauses
                .iter()
                .map(|(cond, then)| {
                    Ok((
                        to_literal(&eval_aggregate_expr(cond, group_rows)?),
                        to_literal(&eval_aggregate_expr(then, group_rows)?),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let resolved_else = if let Some(el) = else_clause {
                Some(Box::new(to_literal(&eval_aggregate_expr(el, group_rows)?)))
            } else {
                None
            };
            let resolved = Expr::Case {
                operand: resolved_operand,
                when_clauses: resolved_whens,
                else_clause: resolved_else,
            };
            let mut ctx = EvalContext::new("", b"{}");
            ctx.eval(&resolved)
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

/// Information about a TIME_BUCKET_GAPFILL call detected in GROUP BY.
struct GapfillInfo {
    bucket_seconds: u64,
}

/// Check if GROUP BY contains a `TIME_BUCKET_GAPFILL(...)` call.
fn detect_gapfill(group_exprs: &[Expr]) -> Option<GapfillInfo> {
    for expr in group_exprs {
        if let Expr::Function { name, args } = expr {
            if name.eq_ignore_ascii_case("TIME_BUCKET_GAPFILL") && args.len() >= 2 {
                let bucket_seconds = match &args[0] {
                    Expr::StringLit(s) => parse_interval_seconds(s),
                    Expr::NumberLit(n) => Some(*n as u64),
                    _ => None,
                };
                if let Some(secs) = bucket_seconds {
                    return Some(GapfillInfo {
                        bucket_seconds: secs,
                    });
                }
            }
        }
    }
    None
}

/// Fill gaps in time-series grouped results by adding empty bucket rows.
fn apply_gapfill(
    mut result_rows: Vec<(String, Vec<u8>)>,
    info: &GapfillInfo,
    items: &[SelectItem],
) -> Result<Vec<(String, Vec<u8>)>> {
    if result_rows.is_empty() {
        return Ok(result_rows);
    }

    // Find the gapfill column name from the SELECT items
    let gf_col_name = find_gapfill_column_name(items);
    let gf_col_name = match gf_col_name {
        Some(n) => n,
        None => return Ok(result_rows), // Can't identify the column
    };

    // Extract all bucket values from existing rows
    let mut existing_buckets: Vec<u64> = Vec::new();
    for (_, doc) in &result_rows {
        if let Ok(serde_json::Value::Object(map)) = serde_json::from_slice::<serde_json::Value>(doc)
        {
            if let Some(v) = map.get(&gf_col_name).and_then(|v| v.as_f64()) {
                existing_buckets.push(v as u64);
            }
        }
    }

    if existing_buckets.is_empty() {
        return Ok(result_rows);
    }

    let min_bucket = *existing_buckets.iter().min().unwrap();
    let max_bucket = *existing_buckets.iter().max().unwrap();

    // Generate all buckets in range
    let mut current = min_bucket;
    let existing_set: std::collections::HashSet<u64> = existing_buckets.into_iter().collect();

    while current <= max_bucket {
        if !existing_set.contains(&current) {
            // Create a gap-fill row with NULL aggregates
            let mut row_json = serde_json::Map::new();
            row_json.insert(
                gf_col_name.clone(),
                serde_json::Value::Number(serde_json::Number::from(current)),
            );
            // For each aggregate item, insert null
            for item in items {
                if let SelectItem::Expr { expr, alias } = item {
                    if let Expr::Function { name, .. } = expr {
                        if is_aggregate_function(name) {
                            let col_name = alias
                                .clone()
                                .unwrap_or_else(|| aggregate_display_name(expr));
                            row_json.insert(col_name, serde_json::Value::Null);
                        } else if name.eq_ignore_ascii_case("LOCF")
                            || name.eq_ignore_ascii_case("INTERPOLATE")
                        {
                            let col_name = alias
                                .clone()
                                .unwrap_or_else(|| aggregate_display_name(expr));
                            row_json.insert(col_name, serde_json::Value::Null);
                        }
                    }
                }
            }
            result_rows.push((
                String::new(),
                serde_json::to_vec(&serde_json::Value::Object(row_json))?,
            ));
        }
        current += info.bucket_seconds;
    }

    // Sort by bucket value
    let gf_col_sort = gf_col_name.clone();
    result_rows.sort_by(|a, b| {
        let va = serde_json::from_slice::<serde_json::Value>(&a.1)
            .ok()
            .and_then(|v| v.get(&gf_col_sort).and_then(|v| v.as_f64()));
        let vb = serde_json::from_slice::<serde_json::Value>(&b.1)
            .ok()
            .and_then(|v| v.get(&gf_col_sort).and_then(|v| v.as_f64()));
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply LOCF: for rows with null values, carry forward the last non-null value
    apply_locf_interpolation(&mut result_rows, items, &gf_col_name)?;

    Ok(result_rows)
}

/// Find the column name used for the gapfill bucket in SELECT items.
fn find_gapfill_column_name(items: &[SelectItem]) -> Option<String> {
    for item in items {
        if let SelectItem::Expr { expr, alias } = item {
            if let Expr::Function { name, .. } = expr {
                if name.eq_ignore_ascii_case("TIME_BUCKET_GAPFILL") {
                    return Some(
                        alias
                            .clone()
                            .unwrap_or_else(|| aggregate_display_name(expr)),
                    );
                }
            }
        }
    }
    None
}

/// Apply LOCF (Last Observation Carried Forward) and INTERPOLATE to gap-filled rows.
fn apply_locf_interpolation(
    rows: &mut [(String, Vec<u8>)],
    items: &[SelectItem],
    _bucket_col: &str,
) -> Result<()> {
    // Collect column names that need LOCF or INTERPOLATE
    let mut locf_cols: Vec<String> = Vec::new();
    let mut interp_cols: Vec<String> = Vec::new();

    for item in items {
        if let SelectItem::Expr { expr, alias } = item {
            if let Expr::Function { name, args } = expr {
                let col_name = alias
                    .clone()
                    .unwrap_or_else(|| aggregate_display_name(expr));
                if name.eq_ignore_ascii_case("LOCF") && !args.is_empty() {
                    locf_cols.push(col_name);
                } else if name.eq_ignore_ascii_case("INTERPOLATE") && !args.is_empty() {
                    interp_cols.push(col_name);
                }
            }
        }
    }

    if locf_cols.is_empty() && interp_cols.is_empty() {
        return Ok(());
    }

    // LOCF: carry forward last non-null
    let mut last_values: HashMap<String, serde_json::Value> = HashMap::new();
    for (_, doc) in rows.iter_mut() {
        if let Ok(serde_json::Value::Object(ref mut map)) =
            serde_json::from_slice::<serde_json::Value>(doc)
        {
            for col in &locf_cols {
                if let Some(v) = map.get(col) {
                    if !v.is_null() {
                        last_values.insert(col.clone(), v.clone());
                    } else if let Some(last) = last_values.get(col) {
                        map.insert(col.clone(), last.clone());
                    }
                } else if let Some(last) = last_values.get(col) {
                    map.insert(col.clone(), last.clone());
                }
            }
            *doc = serde_json::to_vec(&serde_json::Value::Object(map.clone()))?;
        }
    }

    Ok(())
}

fn to_literal(val: &SqlValue) -> Expr {
    match val {
        SqlValue::Null => Expr::Null,
        SqlValue::Bool(b) => Expr::BoolLit(*b),
        SqlValue::Number(n) => Expr::NumberLit(*n),
        SqlValue::Text(s) => Expr::StringLit(s.clone()),
        SqlValue::Decimal(d) => {
            use rust_decimal::prelude::ToPrimitive;
            Expr::NumberLit(d.to_f64().unwrap_or(0.0))
        }
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
                    let col_name = alias.clone().unwrap_or_else(|| expr_display_name(expr));
                    row_json.insert(col_name, sql_value_to_json(&val));
                }
            }
        }

        out.push(serde_json::to_vec(&serde_json::Value::Object(row_json))?);
    }

    Ok(SqlResult::Rows(out))
}

fn detect_legacy_projection(items: &[SelectItem]) -> Option<LegacyProjection> {
    if items.len() == 1 {
        if let SelectItem::Expr {
            expr: Expr::Column(c),
            alias: None,
        } = &items[0]
        {
            if c.eq_ignore_ascii_case("doc") {
                return Some(LegacyProjection::Doc);
            }
        }
        if let SelectItem::Expr {
            expr: Expr::Function { name, args },
            alias: None,
        } = &items[0]
        {
            if name.eq_ignore_ascii_case("count")
                && args.len() == 1
                && matches!(&args[0], Expr::Star)
            {
                return Some(LegacyProjection::CountStar);
            }
        }
    }
    if items.len() == 2 {
        let first_is_pk = matches!(&items[0], SelectItem::Expr { expr: Expr::Column(c), alias: None } if c.eq_ignore_ascii_case("pk"));
        if first_is_pk {
            if let SelectItem::Expr {
                expr: Expr::Column(c),
                alias: None,
            } = &items[1]
            {
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
        LegacyProjection::CountStar => {
            Ok(SqlResult::Rows(vec![rows.len().to_string().into_bytes()]))
        }
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
/// Check whether an index exists on `table` that covers `column` as a single-column index.
fn has_single_column_index(
    db: &Database,
    _session: &SqlSession,
    table: &str,
    column: &str,
) -> bool {
    let prefix = index_meta_prefix_for_table(table);
    let metas = match db.scan_prefix(&prefix, None, None, None) {
        Ok(m) => m,
        Err(_) => return false,
    };
    metas.iter().any(|meta_row| {
        if let Ok(meta) = serde_json::from_slice::<IndexMetadata>(&meta_row.doc) {
            meta.columns.len() == 1 && meta.columns[0].eq_ignore_ascii_case(column)
        } else {
            false
        }
    })
}

/// Greedy join reordering: reorder joins so that index-accelerated (INL) joins
/// are executed first, reducing intermediate result sizes. Ensures that each
/// join step only references tables already available in the pipeline.
fn reorder_joins(
    db: &Database,
    session: &SqlSession,
    left_table: &str,
    joins: &[JoinSpec],
) -> Vec<JoinSpec> {
    if joins.len() <= 1 {
        return joins.to_vec();
    }

    let mut available_tables: Vec<String> = vec![left_table.to_string()];
    let mut remaining: Vec<(usize, &JoinSpec)> = joins.iter().enumerate().collect();
    let mut result = Vec::with_capacity(joins.len());

    while !remaining.is_empty() {
        // Score each remaining join
        let mut best_idx = None;
        let mut best_score = i64::MIN;

        for (pos, &(_, js)) in remaining.iter().enumerate() {
            // Check if this join is valid (ON clause references available tables)
            let valid = is_join_reachable(&js.on_clause, &available_tables, &js.right_table);
            if !valid {
                continue;
            }

            let mut score: i64 = 0;

            // Bonus for INL-accelerable joins (right table has index on join column)
            if let Some((_left_col, right_col)) = extract_join_eq_columns_from_available(
                &js.on_clause,
                &available_tables,
                &js.right_table,
            ) {
                if has_single_column_index(db, session, &js.right_table, &right_col) {
                    score += 100; // Strongly prefer indexed joins
                }
            }

            // Slight preference for original ordering (stability)
            score -= pos as i64;

            if score > best_score {
                best_score = score;
                best_idx = Some(pos);
            }
        }

        match best_idx {
            Some(pos) => {
                let (_, js) = remaining.remove(pos);
                available_tables.push(js.right_table.clone());
                result.push(js.clone());
            }
            None => {
                // No valid join found — append remaining in original order
                // (execution will handle errors)
                for (_, js) in remaining.drain(..) {
                    result.push(js.clone());
                }
            }
        }
    }

    result
}

/// Check if a join's ON clause references only tables in `available` and `right_table`.
fn is_join_reachable(on_clause: &Option<Expr>, available: &[String], right_table: &str) -> bool {
    match on_clause {
        None => true, // CROSS JOIN: always reachable
        Some(Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        }) => {
            let check_side = |e: &Expr| -> bool {
                match e {
                    Expr::FieldAccess { column, .. } => {
                        available.iter().any(|t| t.eq_ignore_ascii_case(column))
                            || column.eq_ignore_ascii_case(right_table)
                    }
                    Expr::Column(_) => true, // Unqualified: assumed reachable
                    _ => true,
                }
            };
            check_side(left) && check_side(right)
        }
        _ => true, // Complex ON clause: assume reachable
    }
}

/// Like extract_join_eq_columns but accepts a set of available tables instead
/// of a single left_table. Returns (left_col, right_col) where left_col belongs
/// to one of the available tables.
fn extract_join_eq_columns_from_available(
    on_clause: &Option<Expr>,
    available_tables: &[String],
    right_table: &str,
) -> Option<(String, String)> {
    match on_clause {
        Some(Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        }) => {
            let resolve = |e: &Expr| -> Option<(String, String)> {
                match e {
                    Expr::Column(c) => Some((String::new(), c.clone())),
                    Expr::FieldAccess { column, path } if path.len() == 1 => {
                        Some((column.clone(), path[0].clone()))
                    }
                    _ => None,
                }
            };
            let (ltbl, lcol) = resolve(left)?;
            let (rtbl, rcol) = resolve(right)?;

            let left_matches_available = ltbl.is_empty()
                || available_tables
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(&ltbl));
            let right_matches_right = rtbl.is_empty() || rtbl.eq_ignore_ascii_case(right_table);

            if left_matches_available && right_matches_right {
                Some((lcol, rcol))
            } else {
                // Try flipped
                let right_matches_available = rtbl.is_empty()
                    || available_tables
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(&rtbl));
                let left_matches_right = ltbl.is_empty() || ltbl.eq_ignore_ascii_case(right_table);
                if right_matches_available && left_matches_right {
                    Some((rcol, lcol))
                } else {
                    None
                }
            }
        }
        _ => None,
    }
}

/// Split a WHERE clause into AND-conjuncts.
fn split_conjuncts(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::BinOp {
            left,
            op: BinOperator::And,
            right,
        } => {
            let mut parts = split_conjuncts(left);
            parts.extend(split_conjuncts(right));
            parts
        }
        other => vec![other.clone()],
    }
}

/// Recombine conjuncts into a single AND expression.
fn combine_conjuncts(parts: Vec<Expr>) -> Option<Expr> {
    parts.into_iter().reduce(|a, b| Expr::BinOp {
        left: Box::new(a),
        op: BinOperator::And,
        right: Box::new(b),
    })
}

/// Collect all table qualifiers referenced by an expression.
fn collect_table_refs(expr: &Expr) -> Vec<String> {
    let mut tables = Vec::new();
    collect_table_refs_inner(expr, &mut tables);
    tables
}

fn collect_table_refs_inner(expr: &Expr, tables: &mut Vec<String>) {
    match expr {
        Expr::FieldAccess { column, .. } => {
            if !column.eq_ignore_ascii_case("doc") {
                let lower = column.to_lowercase();
                if !tables.contains(&lower) {
                    tables.push(lower);
                }
            }
        }
        Expr::BinOp { left, right, .. } => {
            collect_table_refs_inner(left, tables);
            collect_table_refs_inner(right, tables);
        }
        Expr::Not(inner) => collect_table_refs_inner(inner, tables),
        Expr::IsNull { expr: inner, .. } => collect_table_refs_inner(inner, tables),
        Expr::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            collect_table_refs_inner(inner, tables);
            collect_table_refs_inner(low, tables);
            collect_table_refs_inner(high, tables);
        }
        Expr::InList {
            expr: inner, list, ..
        } => {
            collect_table_refs_inner(inner, tables);
            for item in list {
                collect_table_refs_inner(item, tables);
            }
        }
        Expr::Function { args, .. } | Expr::WindowFunction { args, .. } => {
            for arg in args {
                collect_table_refs_inner(arg, tables);
            }
        }
        _ => {}
    }
}

/// Predicate pushdown: partition a WHERE clause into per-table predicates
/// that can be applied before joining, and residual cross-table predicates.
fn pushdown_predicates(
    filter: &Option<Expr>,
    left_table: &str,
    joins: &[JoinSpec],
) -> (HashMap<String, Vec<Expr>>, Option<Expr>) {
    let filter = match filter {
        Some(f) => f,
        None => return (HashMap::new(), None),
    };

    let conjuncts = split_conjuncts(filter);
    let mut per_table: HashMap<String, Vec<Expr>> = HashMap::new();
    let mut residual = Vec::new();

    // All tables involved in the query
    let mut all_tables: Vec<String> = vec![left_table.to_lowercase()];
    for js in joins {
        all_tables.push(js.right_table.to_lowercase());
    }

    for conjunct in conjuncts {
        let refs = collect_table_refs(&conjunct);

        if refs.len() == 1 && all_tables.contains(&refs[0]) {
            // Single-table predicate: push down
            per_table.entry(refs[0].clone()).or_default().push(conjunct);
        } else if refs.is_empty() {
            // No table refs (e.g., `1 = 1`): keep as residual
            residual.push(conjunct);
        } else {
            // Cross-table or unqualified: keep as residual
            residual.push(conjunct);
        }
    }

    (per_table, combine_conjuncts(residual))
}

#[allow(clippy::too_many_arguments)]
fn execute_join_select(
    db: &Database,
    session: &mut SqlSession,
    left_table: &str,
    joins: Vec<JoinSpec>,
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

    // Reorder joins for optimal execution (index-accelerated joins first)
    let joins = if joins.len() > 1 {
        reorder_joins(db, session, left_table, &joins)
    } else {
        joins
    };

    // Predicate pushdown: split WHERE into per-table predicates and residual
    let (pushed_preds, residual_filter) = pushdown_predicates(&filter, left_table, &joins);

    // Extract pk filter from the WHERE clause for left table
    let pk_from_filter = extract_pk_eq_literal(filter.as_ref());

    let mut left_rows = fetch_rows_for_table(
        db,
        session,
        left_table,
        pk_from_filter.as_deref(),
        as_of,
        valid_at,
    )?;

    // Apply pushed-down predicate for left table
    if let Some(preds) = pushed_preds.get(&left_table.to_lowercase()) {
        if let Some(pred) = combine_conjuncts(preds.clone()) {
            left_rows = filter_rows(left_rows, &pred)?;
        }
    }

    let is_nway = joins.len() > 1;

    if is_nway {
        // N-way join: table-keyed pipeline
        // Wrap left table rows under their table name for proper column resolution
        let mut current_table = left_table.to_string();
        let mut current_rows: Vec<VisibleRow> = left_rows
            .into_iter()
            .map(|r| {
                let doc_val: serde_json::Value =
                    serde_json::from_slice(&r.doc).unwrap_or(serde_json::Value::Null);
                let mut obj = serde_json::Map::new();
                obj.insert(left_table.to_string(), doc_val);
                VisibleRow {
                    pk: r.pk,
                    doc: serde_json::to_vec(&serde_json::Value::Object(obj)).unwrap_or_default(),
                }
            })
            .collect();

        for join_spec in &joins {
            let _ = load_table_schema(db, session, &join_spec.right_table)?;
            let right_pred = pushed_preds
                .get(&join_spec.right_table.to_lowercase())
                .and_then(|preds| combine_conjuncts(preds.clone()));
            let joined = execute_single_join(
                db,
                session,
                &current_rows,
                &current_table,
                join_spec,
                pk_from_filter.as_deref(),
                as_of,
                valid_at,
                right_pred.as_ref(),
            )?;
            // Merge right table under its name key
            let right_name = join_spec.right_table.clone();
            current_rows = joined
                .into_iter()
                .map(|j| {
                    let mut obj: serde_json::Map<String, serde_json::Value> =
                        serde_json::from_slice(&j.left_doc).unwrap_or_default();
                    let right_val: serde_json::Value =
                        serde_json::from_slice(&j.right_doc).unwrap_or(serde_json::Value::Null);
                    obj.insert(right_name.clone(), right_val);
                    VisibleRow {
                        pk: j.pk,
                        doc: serde_json::to_vec(&serde_json::Value::Object(obj))
                            .unwrap_or_default(),
                    }
                })
                .collect();
            current_table = join_spec.right_table.clone();
        }

        // N-way result: use standard VisibleRow pipeline
        // Apply only residual (cross-table) predicates — table-specific ones were pushed down
        let mut rows = current_rows;
        if let Some(ref f) = residual_filter {
            rows = filter_rows(rows, f)?;
        }
        if group_by.is_some() || select_items_contain_aggregate(items) {
            if group_by.is_none() && is_plain_count_star(items) {
                return Ok(SqlResult::Rows(vec![rows.len().to_string().into_bytes()]));
            }
            return execute_grouped_select(rows, items, group_by, having, order_by, limit);
        }
        if let Some(ref orders) = order_by {
            rows.sort_by(|a, b| {
                for (expr, dir) in orders {
                    let mut ctx_a = EvalContext::new(&a.pk, &a.doc);
                    let mut ctx_b = EvalContext::new(&b.pk, &b.doc);
                    let va = ctx_a.eval(expr).unwrap_or(SqlValue::Null);
                    let vb = ctx_b.eval(expr).unwrap_or(SqlValue::Null);
                    let cmp = compare_sort_values(&va, &vb);
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
            rows.sort_by(|a, b| a.pk.cmp(&b.pk));
        }
        if let Some(n) = limit {
            let n = usize::try_from(n).unwrap_or(usize::MAX);
            rows.truncate(n);
        }
        return project_rows(&rows, items);
    }

    // Single join: preserve JoinedRow (left_doc/right_doc) for backward compat
    let join_spec = &joins[0];
    let _ = load_table_schema(db, session, &join_spec.right_table)?;
    let right_pred = pushed_preds
        .get(&join_spec.right_table.to_lowercase())
        .and_then(|preds| combine_conjuncts(preds.clone()));
    let joined = execute_single_join(
        db,
        session,
        &left_rows,
        left_table,
        join_spec,
        pk_from_filter.as_deref(),
        as_of,
        valid_at,
        right_pred.as_ref(),
    )?;

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
            let filtered_pks: std::collections::HashSet<usize> =
                filtered.iter().enumerate().map(|(i, _)| i).collect();
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
                let cmp = compare_sort_values(&va, &vb);
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
                return Err(TensorError::SqlExec(
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

        out.push(serde_json::to_vec(&serde_json::Value::Object(row_json))?);
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

/// Execute a single join step between current rows and a right table.
#[allow(clippy::too_many_arguments)]
fn execute_single_join(
    db: &Database,
    session: &SqlSession,
    left_rows: &[VisibleRow],
    left_table: &str,
    join_spec: &JoinSpec,
    pk_from_filter: Option<&str>,
    as_of: Option<u64>,
    valid_at: Option<u64>,
    right_predicate: Option<&Expr>,
) -> Result<Vec<JoinedRow>> {
    // Helper: fetch and optionally filter right rows
    let fetch_right = |pk_filter: Option<&str>| -> Result<Vec<VisibleRow>> {
        let mut rows = fetch_rows_for_table(
            db,
            session,
            &join_spec.right_table,
            pk_filter,
            as_of,
            valid_at,
        )?;
        if let Some(pred) = right_predicate {
            rows = filter_rows(rows, pred)?;
        }
        Ok(rows)
    };

    match join_spec.join_type {
        JoinType::Inner | JoinType::Left => {
            let is_pk_join =
                is_pk_eq_join(&join_spec.on_clause, left_table, &join_spec.right_table);
            let left_outer = matches!(join_spec.join_type, JoinType::Left);

            if !is_pk_join {
                if let Some(inl_result) = try_index_nested_loop_join(
                    db,
                    session,
                    left_rows,
                    left_table,
                    &join_spec.right_table,
                    &join_spec.on_clause,
                    left_outer,
                    as_of,
                    valid_at,
                ) {
                    return Ok(inl_result);
                }
                let right_rows = fetch_right(pk_from_filter)?;
                Ok(nested_loop_join(
                    left_rows.to_vec(),
                    right_rows,
                    &join_spec.on_clause,
                    left_outer,
                    false,
                ))
            } else {
                let right_rows = fetch_right(pk_from_filter)?;
                if left_outer {
                    Ok(left_hash_join_on_pk(left_rows.to_vec(), right_rows))
                } else {
                    Ok(hash_join_on_pk(left_rows.to_vec(), right_rows))
                }
            }
        }
        JoinType::Right => {
            let right_rows = fetch_right(pk_from_filter)?;
            if is_pk_eq_join(&join_spec.on_clause, left_table, &join_spec.right_table) {
                Ok(right_hash_join_on_pk(left_rows.to_vec(), right_rows))
            } else {
                Ok(nested_loop_join(
                    left_rows.to_vec(),
                    right_rows,
                    &join_spec.on_clause,
                    false,
                    true,
                ))
            }
        }
        JoinType::Cross => {
            let right_rows = fetch_right(pk_from_filter)?;
            Ok(cross_join(left_rows.to_vec(), right_rows))
        }
    }
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
                    (column.eq_ignore_ascii_case(right_table) || column.eq_ignore_ascii_case("pk"))
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

fn left_hash_join_on_pk(left_rows: Vec<VisibleRow>, right_rows: Vec<VisibleRow>) -> Vec<JoinedRow> {
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

/// Extract (left_col, right_col) from ON clause like `left_table.col = right_table.col`
/// or `col_a = col_b`. Returns (left_column, right_column) where "left" means the column
/// referencing the left table and "right" means the column referencing the right table.
fn extract_join_eq_columns(
    on_clause: &Option<Expr>,
    left_table: &str,
    right_table: &str,
) -> Option<(String, String)> {
    match on_clause {
        Some(Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        }) => {
            let resolve = |e: &Expr| -> Option<(String, String)> {
                match e {
                    Expr::Column(c) => Some((String::new(), c.clone())),
                    Expr::FieldAccess { column, path } if path.len() == 1 => {
                        Some((column.clone(), path[0].clone()))
                    }
                    _ => None,
                }
            };
            let (ltbl, lcol) = resolve(left)?;
            let (rtbl, rcol) = resolve(right)?;

            // Determine which side references which table
            if (ltbl.is_empty() || ltbl.eq_ignore_ascii_case(left_table))
                && (rtbl.is_empty() || rtbl.eq_ignore_ascii_case(right_table))
            {
                Some((lcol, rcol))
            } else if (ltbl.is_empty() || ltbl.eq_ignore_ascii_case(right_table))
                && (rtbl.is_empty() || rtbl.eq_ignore_ascii_case(left_table))
            {
                Some((rcol, lcol))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Resolve a column value from a doc that may be flat or table-keyed (N-way join).
/// Tries: doc[column] (flat), doc[table_hint][column] (table-keyed with hint),
/// then searches all sub-objects as fallback.
fn resolve_doc_column<'a>(
    doc: &'a serde_json::Value,
    column: &str,
    table_hint: &str,
) -> Option<&'a serde_json::Value> {
    // 1. Flat: doc[column]
    if let Some(v) = doc.get(column) {
        return Some(v);
    }
    // 2. Table-keyed: doc[table_hint][column]
    if !table_hint.is_empty() {
        if let Some(table_obj) = doc.get(table_hint) {
            if let Some(v) = table_obj.get(column) {
                return Some(v);
            }
        }
    }
    // 3. Search all sub-objects (fallback for multi-table keyed docs)
    if let Some(obj) = doc.as_object() {
        for (_key, sub) in obj {
            if sub.is_object() {
                if let Some(v) = sub.get(column) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Index nested loop join: for each left row, use an index on the right table
/// to find matching rows instead of scanning the entire right table.
#[allow(clippy::too_many_arguments)]
fn try_index_nested_loop_join(
    db: &Database,
    session: &SqlSession,
    left_rows: &[VisibleRow],
    left_table: &str,
    right_table: &str,
    on_clause: &Option<Expr>,
    left_outer: bool,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Option<Vec<JoinedRow>> {
    let (left_col, right_col) = extract_join_eq_columns(on_clause, left_table, right_table)?;

    // Skip if join is on pk (already handled by hash join)
    if right_col.eq_ignore_ascii_case("pk") || right_col.eq_ignore_ascii_case("id") {
        return None;
    }

    // Find a single-column index on right_col in the right table
    let prefix = index_meta_prefix_for_table(right_table);
    let index_metas = db.scan_prefix(&prefix, None, None, None).ok()?;
    let matching_index = index_metas.iter().find_map(|meta_row| {
        let meta: IndexMetadata = serde_json::from_slice(&meta_row.doc).ok()?;
        if meta.columns.len() == 1 && meta.columns[0] == right_col {
            Some(meta)
        } else {
            None
        }
    })?;

    let mut out = Vec::new();
    for left in left_rows {
        // Extract the join key value from the left row (handles flat or table-keyed docs)
        let left_doc: serde_json::Value = serde_json::from_slice(&left.doc).ok()?;
        let left_val = match resolve_doc_column(&left_doc, &left_col, left_table) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Number(n)) => n.to_string(),
            Some(serde_json::Value::Bool(b)) => b.to_string(),
            _ => continue,
        };

        // Index lookup on right table
        let idx_prefix = index_entry_prefix(right_table, &matching_index.name, &left_val);
        let idx_entries = match db.scan_prefix(&idx_prefix, as_of, valid_at, None) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let mut matched = false;
        for entry in &idx_entries {
            if entry.doc == INDEX_TOMBSTONE {
                continue;
            }
            let key_str = String::from_utf8_lossy(&entry.user_key);
            let right_pk = match key_str.rsplit('/').next() {
                Some(pk) => pk,
                None => continue,
            };
            let row_k = row_key(right_table, right_pk);
            if let Ok(Some(right_doc)) = read_live_key(db, session, &row_k, as_of, valid_at) {
                if !right_doc.is_empty() {
                    matched = true;
                    out.push(JoinedRow {
                        pk: left.pk.clone(),
                        left_doc: left.doc.clone(),
                        right_doc,
                        left_pk: left.pk.clone(),
                        right_pk: right_pk.to_string(),
                    });
                }
            }
        }

        if left_outer && !matched {
            out.push(JoinedRow {
                pk: left.pk.clone(),
                left_doc: left.doc.clone(),
                right_doc: b"null".to_vec(),
                left_pk: left.pk.clone(),
                right_pk: String::new(),
            });
        }
    }
    Some(out)
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
    rows: &mut [VisibleRow],
    items: &[SelectItem],
    order_by: Option<&Vec<(Expr, OrderDirection)>>,
    limit: Option<u64>,
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
            ..
        } = item
        {
            window_specs.push(WindowFuncSpec {
                item_index: idx,
                name: name.clone(),
                args: args.clone(),
                partition_by: partition_by.clone(),
                order_by: order_by.clone(),
            });
        }
    }

    // Compute window values over ALL rows (before ORDER BY/LIMIT)
    let mut window_values: HashMap<usize, Vec<SqlValue>> = HashMap::new();

    for spec in &window_specs {
        let values = compute_window_function(
            rows,
            &spec.name,
            &spec.args,
            &spec.partition_by,
            &spec.order_by,
        )?;
        window_values.insert(spec.item_index, values);
    }

    // Apply ORDER BY (with original row indices preserved)
    let mut indices: Vec<usize> = (0..rows.len()).collect();
    if let Some(orders) = order_by {
        indices.sort_by(|&ai, &bi| {
            let a = &rows[ai];
            let b = &rows[bi];
            for (expr, dir) in orders {
                let mut ctx_a = EvalContext::new(&a.pk, &a.doc);
                let mut ctx_b = EvalContext::new(&b.pk, &b.doc);
                let va = ctx_a.eval(expr).unwrap_or(SqlValue::Null);
                let vb = ctx_b.eval(expr).unwrap_or(SqlValue::Null);
                let cmp = compare_sort_values(&va, &vb);
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
        indices.truncate(n);
    }

    // Project with window values using the ordered/limited indices
    let mut out = Vec::with_capacity(indices.len());
    for &row_idx in &indices {
        let row = &rows[row_idx];
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
                    let col_name = alias.clone().unwrap_or_else(|| expr_display_name(expr));

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

        out.push(serde_json::to_vec(&serde_json::Value::Object(row_json))?);
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
    for indices in partitions.values_mut() {
        if !order_by.is_empty() {
            indices.sort_by(|&a, &b| {
                for (oi, (_expr, dir)) in order_by.iter().enumerate() {
                    let mut ctx_a = EvalContext::new(&rows[a].pk, &rows[a].doc);
                    let mut ctx_b = EvalContext::new(&rows[b].pk, &rows[b].doc);
                    let va = ctx_a.eval(&order_by[oi].0).unwrap_or(SqlValue::Null);
                    let vb = ctx_b.eval(&order_by[oi].0).unwrap_or(SqlValue::Null);
                    let cmp = compare_sort_values(&va, &vb);
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
            return Err(TensorError::SqlExec(format!(
                "unsupported window function: {other}"
            )));
        }
    }

    Ok(result)
}

fn execute_analyze(db: &Database, session: &mut SqlSession, table: &str) -> Result<SqlResult> {
    validate_table_name(table)?;

    // Ensure the table exists and load schema
    let schema = load_table_schema(db, session, table)?;

    // Scan all visible rows to compute stats
    let rows = scan_visible_rows(db, session, table, None, None)?;
    let row_count = rows.len() as u64;
    let approx_byte_size: u64 = rows.iter().map(|r| r.doc.len() as u64).sum();
    let avg_row_bytes = if row_count > 0 {
        approx_byte_size / row_count
    } else {
        0
    };
    let last_updated_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // Compute column-level statistics for typed tables
    let columns = if schema.schema_mode == "typed" {
        compute_column_stats(&rows, &schema)
    } else {
        Vec::new()
    };

    let col_count = columns.len();

    let stats = TableStats {
        row_count,
        approx_byte_size,
        avg_row_bytes,
        last_updated_ms,
        columns,
    };

    let stats_bytes = serde_json::to_vec(&stats)?;
    let key = stats_meta_key(table);
    let commit_ts = db.put(&key, stats_bytes, 0, u64::MAX, None)?;

    let msg = if col_count > 0 {
        format!("ANALYZE {table}: {row_count} rows, ~{approx_byte_size} bytes, {col_count} column stats")
    } else {
        format!("ANALYZE {table}: {row_count} rows, ~{approx_byte_size} bytes")
    };

    Ok(SqlResult::Affected {
        rows: row_count,
        commit_ts: Some(commit_ts),
        message: msg,
    })
}

/// Compute per-column statistics from a set of visible rows.
fn compute_column_stats(
    rows: &[VisibleRow],
    schema: &TableSchemaMetadata,
) -> Vec<ColumnStatistics> {
    const TOP_N: usize = 10;

    schema
        .columns
        .iter()
        .map(|col| {
            let mut distinct: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut null_count = 0u64;
            let mut min_val: Option<String> = None;
            let mut max_val: Option<String> = None;
            let mut freq: HashMap<String, u64> = HashMap::new();

            for row in rows {
                let val_str = if col.name.eq_ignore_ascii_case(&schema.pk) {
                    Some(row.pk.clone())
                } else {
                    match serde_json::from_slice::<serde_json::Value>(&row.doc) {
                        Ok(serde_json::Value::Object(ref map)) => {
                            map.get(&col.name).and_then(|v| {
                                if v.is_null() {
                                    None
                                } else {
                                    Some(match v {
                                        serde_json::Value::String(s) => s.clone(),
                                        other => other.to_string(),
                                    })
                                }
                            })
                        }
                        _ => None,
                    }
                };

                match val_str {
                    Some(v) => {
                        distinct.insert(v.clone());
                        *freq.entry(v.clone()).or_insert(0) += 1;
                        match &min_val {
                            None => min_val = Some(v.clone()),
                            Some(m) if v < *m => min_val = Some(v.clone()),
                            _ => {}
                        }
                        match &max_val {
                            None => max_val = Some(v.clone()),
                            Some(m) if v > *m => max_val = Some(v),
                            _ => {}
                        }
                    }
                    None => null_count += 1,
                }
            }

            // Top-N by frequency
            let mut freq_vec: Vec<(String, u64)> = freq.into_iter().collect();
            freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
            freq_vec.truncate(TOP_N);

            ColumnStatistics {
                name: col.name.clone(),
                distinct_count: distinct.len() as u64,
                null_count,
                min_value: min_val,
                max_value: max_val,
                top_values: freq_vec,
            }
        })
        .collect()
}

// ==================== ASK (NL → SQL) ====================

fn execute_ask(db: &Database, question: &str) -> Result<SqlResult> {
    #[cfg(feature = "llm")]
    {
        let (sql, result) = db.ask(question)?;
        // Wrap the result: first row is the generated SQL, then the actual result rows
        match result {
            SqlResult::Rows(rows) => {
                let mut out = Vec::with_capacity(rows.len() + 1);
                let header = serde_json::json!({ "__generated_sql": sql });
                out.push(serde_json::to_vec(&header).unwrap_or_default());
                out.extend(rows);
                Ok(SqlResult::Rows(out))
            }
            SqlResult::Affected {
                rows,
                commit_ts,
                message,
            } => Ok(SqlResult::Affected {
                rows,
                commit_ts,
                message: format!("-- Generated SQL: {sql}\n{message}"),
            }),
            SqlResult::Explain(text) => Ok(SqlResult::Explain(format!(
                "-- Generated SQL: {sql}\n{text}"
            ))),
        }
    }
    #[cfg(not(feature = "llm"))]
    {
        let _ = question;
        let _ = db;
        Err(crate::error::TensorError::FeatureNotEnabled(
            "llm".to_string(),
        ))
    }
}

// ==================== Full-Text Search Functions ====================

fn execute_create_fulltext_index(
    db: &Database,
    session: &mut SqlSession,
    index: &str,
    table: &str,
    columns: &[String],
) -> Result<SqlResult> {
    validate_index_name(index)?;
    validate_table_name(table)?;
    let schema = load_table_schema(db, session, table)?;

    // Validate all columns exist
    for col in columns {
        let valid = col.eq_ignore_ascii_case("doc")
            || schema
                .columns
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(col));
        if !valid {
            return Err(TensorError::SqlExec(format!(
                "column {col} does not exist on table {table}"
            )));
        }
    }

    // Check if FTS index already exists
    let key = fts_index_meta_key(table, index);
    if read_live_key(db, session, &key, None, None)?.is_some() {
        return Err(TensorError::SqlExec(format!(
            "fulltext index {index} already exists on {table}"
        )));
    }

    let meta = FtsIndexMetadata {
        name: index.to_string(),
        table: table.to_string(),
        columns: columns.to_vec(),
    };
    let payload = serde_json::to_vec(&meta)?;
    let commit_ts = write_put(db, session, key, payload, 0, u64::MAX, Some(1))?;

    // Backfill: index all existing rows
    let rows = scan_visible_rows(db, session, table, None, None)?;
    let mut indexed_count = 0u64;
    for row in &rows {
        index_row_for_fts(db, session, table, &row.pk, &row.doc, columns)?;
        indexed_count += 1;
    }

    Ok(SqlResult::Affected {
        rows: 1,
        commit_ts,
        message: format!(
            "created fulltext index {index} on {table}({}) — indexed {indexed_count} rows",
            columns.join(", ")
        ),
    })
}

fn execute_drop_fulltext_index(
    db: &Database,
    session: &mut SqlSession,
    index: &str,
    table: &str,
) -> Result<SqlResult> {
    validate_index_name(index)?;
    validate_table_name(table)?;

    let key = fts_index_meta_key(table, index);
    if read_live_key(db, session, &key, None, None)?.is_none() {
        return Err(TensorError::SqlExec(format!(
            "fulltext index {index} does not exist on {table}"
        )));
    }

    // Delete the metadata (append tombstone-style: overwrite with empty)
    write_put(db, session, key, b"{}".to_vec(), 0, u64::MAX, Some(1))?;

    Ok(SqlResult::Affected {
        rows: 1,
        commit_ts: None,
        message: format!("dropped fulltext index {index} on {table}"),
    })
}

fn execute_create_vector_index(
    db: &Database,
    session: &mut SqlSession,
    index: &str,
    table: &str,
    column: &str,
    index_type: crate::sql::parser::VectorIndexType,
    params: &[(String, String)],
) -> Result<SqlResult> {
    validate_index_name(index)?;
    validate_table_name(table)?;
    let schema = load_table_schema(db, session, table)?;

    // Validate the column exists and is a VECTOR type
    let col_meta = schema
        .columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(column));
    let dims: u16 = match col_meta {
        Some(c) if c.type_name.starts_with("VECTOR(") => {
            // Parse dims from "VECTOR(384)"
            let inner = &c.type_name[7..c.type_name.len() - 1];
            inner.parse::<u16>().map_err(|_| {
                TensorError::SqlExec(format!(
                    "invalid VECTOR dimensions in schema: {}",
                    c.type_name
                ))
            })?
        }
        Some(c) => {
            return Err(TensorError::SqlExec(format!(
                "column {} has type {}, expected VECTOR",
                column, c.type_name
            )));
        }
        None => {
            return Err(TensorError::SqlExec(format!(
                "column {} does not exist on table {}",
                column, table
            )));
        }
    };

    // Check if vector index already exists
    let key = vector_index_meta_key(table, column);
    if let Some(existing) = read_live_key(db, session, key.as_bytes(), None, None)? {
        if existing != b"{}" {
            return Err(TensorError::SqlExec(format!(
                "vector index already exists on {table}({column})"
            )));
        }
    }

    // Extract metric from params (default: cosine)
    let metric = params
        .iter()
        .find(|(k, _)| k == "metric")
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| "cosine".to_string());

    let meta = VectorIndexMetadata::new(
        index.to_string(),
        table.to_string(),
        column.to_string(),
        dims,
        index_type,
        metric,
        params.to_vec(),
    );

    let payload = serde_json::to_vec(&meta)?;
    let commit_ts = write_put(db, session, key.into_bytes(), payload, 0, u64::MAX, Some(1))?;

    Ok(SqlResult::Affected {
        rows: 1,
        commit_ts,
        message: format!(
            "created vector index {index} on {table}({column}) using {}",
            meta.index_type
        ),
    })
}

fn execute_drop_vector_index(
    db: &Database,
    session: &mut SqlSession,
    index: &str,
    table: &str,
) -> Result<SqlResult> {
    validate_index_name(index)?;
    validate_table_name(table)?;

    // Find the vector index metadata by scanning all vector indexes on this table
    let prefix = format!("__meta/vector_index/{table}/");
    let metas = db.scan_prefix(prefix.as_bytes(), None, None, None)?;

    let mut found = false;
    for meta_row in &metas {
        if let Ok(meta) = serde_json::from_slice::<VectorIndexMetadata>(&meta_row.doc) {
            if meta.index_name == index {
                let key = vector_index_meta_key(table, &meta.column);
                write_put(
                    db,
                    session,
                    key.into_bytes(),
                    b"{}".to_vec(),
                    0,
                    u64::MAX,
                    Some(1),
                )?;
                found = true;
                break;
            }
        }
    }

    if !found {
        return Err(TensorError::SqlExec(format!(
            "vector index {index} does not exist on {table}"
        )));
    }

    Ok(SqlResult::Affected {
        rows: 1,
        commit_ts: None,
        message: format!("dropped vector index {index} on {table}"),
    })
}

/// Store vector data for all vector indexes on a table.
/// For each vector index on this table, extracts the vector column value from the doc,
/// parses it as a vector literal, and stores a VectorRecord under `__vec/{table}/{column}/{pk}`.
fn update_vector_indexes(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    pk: &str,
    doc: &[u8],
) -> Result<()> {
    let prefix = format!("__meta/vector_index/{table}/");
    let metas = db.scan_prefix(prefix.as_bytes(), None, None, None)?;

    for meta_row in &metas {
        // Skip tombstoned metadata
        if meta_row.doc == b"{}" {
            continue;
        }
        if let Ok(meta) = serde_json::from_slice::<VectorIndexMetadata>(&meta_row.doc) {
            if meta.table != table {
                continue;
            }
            // Extract the vector column value from the JSON doc
            if let Ok(serde_json::Value::Object(map)) =
                serde_json::from_slice::<serde_json::Value>(doc)
            {
                if let Some(val) = map.get(&meta.column) {
                    let vec_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Array(_) => val.to_string(),
                        _ => continue,
                    };
                    if let Ok(floats) = parse_vector_literal(&vec_str) {
                        if floats.len() == meta.dims as usize {
                            let record = VectorRecord::from_f32(&floats);
                            let key = vector_data_key(table, &meta.column, pk);
                            write_put(
                                db,
                                session,
                                key.into_bytes(),
                                record.to_bytes(),
                                0,
                                u64::MAX,
                                Some(1),
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Remove vector data for all vector indexes on a table when a row is deleted or updated.
fn remove_vector_indexes(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    pk: &str,
) -> Result<()> {
    let prefix = format!("__meta/vector_index/{table}/");
    let metas = db.scan_prefix(prefix.as_bytes(), None, None, None)?;

    for meta_row in &metas {
        if meta_row.doc == b"{}" {
            continue;
        }
        if let Ok(meta) = serde_json::from_slice::<VectorIndexMetadata>(&meta_row.doc) {
            if meta.table != table {
                continue;
            }
            // Write an empty tombstone for the vector data
            let key = vector_data_key(table, &meta.column, pk);
            write_put(
                db,
                session,
                key.into_bytes(),
                Vec::new(),
                0,
                u64::MAX,
                Some(1),
            )?;
        }
    }
    Ok(())
}

/// Execute `vector_search('table', 'column', '[0.1, 0.2, ...]', k)` table function.
/// Scans all stored vector records for the given table/column, computes distances,
/// and returns the k nearest neighbors.
fn execute_vector_search_fn(
    db: &Database,
    session: &mut SqlSession,
    args: &[Expr],
) -> Result<Vec<VisibleRow>> {
    use crate::facet::vector_search::DistanceMetric;

    if args.len() < 3 {
        return Err(TensorError::SqlExec(
            "vector_search() requires at least 3 arguments: table, column, query_vector [, k]"
                .to_string(),
        ));
    }

    let table = match &args[0] {
        Expr::StringLit(s) => s.clone(),
        _ => {
            return Err(TensorError::SqlExec(
                "vector_search() first argument must be a table name string".to_string(),
            ))
        }
    };
    let column = match &args[1] {
        Expr::StringLit(s) => s.clone(),
        _ => {
            return Err(TensorError::SqlExec(
                "vector_search() second argument must be a column name string".to_string(),
            ))
        }
    };
    let query_str = match &args[2] {
        Expr::StringLit(s) => s.clone(),
        _ => {
            return Err(TensorError::SqlExec(
                "vector_search() third argument must be a vector literal string".to_string(),
            ))
        }
    };
    let k = if args.len() > 3 {
        match eval_const_expr(&args[3])? {
            SqlValue::Number(n) => n as usize,
            _ => 10,
        }
    } else {
        10
    };

    let query = parse_vector_literal(&query_str)?;

    // Load metric from index metadata (default to cosine)
    let meta_key = vector_index_meta_key(&table, &column);
    let metric =
        if let Some(meta_bytes) = read_live_key(db, session, meta_key.as_bytes(), None, None)? {
            if meta_bytes != b"{}" {
                if let Ok(meta) = serde_json::from_slice::<VectorIndexMetadata>(&meta_bytes) {
                    match meta.metric.as_str() {
                        "euclidean" | "l2" => DistanceMetric::Euclidean,
                        "dot_product" | "dot" => DistanceMetric::DotProduct,
                        _ => DistanceMetric::Cosine,
                    }
                } else {
                    DistanceMetric::Cosine
                }
            } else {
                DistanceMetric::Cosine
            }
        } else {
            DistanceMetric::Cosine
        };

    // Scan all vector records for this table/column
    let prefix = vector_data_prefix(&table, &column);
    let records = db.scan_prefix(prefix.as_bytes(), None, None, None)?;

    // Compute distances and find top-k
    let mut scored: Vec<(String, f32, Vec<f32>)> = Vec::new();
    for rec in &records {
        if rec.doc.is_empty() {
            continue; // tombstone
        }
        if let Ok(vr) = VectorRecord::from_bytes(&rec.doc) {
            let vec = vr.to_f32_vec();
            if vec.len() == query.len() {
                let dist = metric.compute(&query, &vec);
                // Extract pk from the key: __vec/{table}/{column}/{pk}
                let key_str = String::from_utf8_lossy(&rec.user_key);
                let pk = key_str.rsplit('/').next().unwrap_or(&key_str).to_string();
                scored.push((pk, dist, vec));
            }
        }
    }

    // Sort by distance and take top k
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);

    // Build result rows
    let mut rows = Vec::new();
    for (rank, (pk, distance, _vec)) in scored.iter().enumerate() {
        let mut obj = serde_json::Map::new();
        obj.insert("pk".to_string(), serde_json::Value::String(pk.clone()));
        obj.insert("distance".to_string(), serde_json::json!(*distance));
        obj.insert(
            "score".to_string(),
            serde_json::json!(1.0 / (1.0 + *distance as f64)),
        );
        obj.insert("rank".to_string(), serde_json::json!(rank + 1));
        let doc = serde_json::to_vec(&serde_json::Value::Object(obj))?;
        rows.push(VisibleRow {
            pk: pk.clone(),
            doc,
        });
    }

    Ok(rows)
}

/// Index a single row for all FTS indexes on a table.
fn update_fts_indexes(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    pk: &str,
    doc: &[u8],
) -> Result<()> {
    // Find all FTS indexes for this table
    let prefix = fts_index_prefix_for_table(table);
    let fts_metas = db.scan_prefix(&prefix, None, None, None)?;

    for meta_row in &fts_metas {
        if let Ok(meta) = serde_json::from_slice::<FtsIndexMetadata>(&meta_row.doc) {
            if meta.table == table && !meta.columns.is_empty() {
                index_row_for_fts(db, session, table, pk, doc, &meta.columns)?;
            }
        }
    }
    Ok(())
}

/// Index a single row's text columns into the FTS inverted index.
fn index_row_for_fts(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    pk: &str,
    doc: &[u8],
    columns: &[String],
) -> Result<()> {
    use crate::facet::fts::{fts_token_key, merge_posting, stem, tokenize};

    // Extract text from the specified columns
    let text = extract_fts_text(doc, columns);
    let tokens = tokenize(&text);

    for token in &tokens {
        let stemmed = stem(token);
        let key = fts_token_key(table, &stemmed);

        // Read existing posting list, merge in the new pk
        let existing = db.get(&key, None, None)?.unwrap_or_default();
        let updated = merge_posting(&existing, pk);
        write_put(db, session, key, updated, 0, u64::MAX, Some(1))?;
    }

    Ok(())
}

/// Extract text from specified columns of a JSON document.
fn extract_fts_text(doc: &[u8], columns: &[String]) -> String {
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(doc) {
        let mut parts = Vec::new();
        for col in columns {
            if col.eq_ignore_ascii_case("doc") {
                // Index all string values from the entire document
                collect_json_strings(&v, &mut parts);
            } else if let Some(val) = v.get(col) {
                match val {
                    serde_json::Value::String(s) => parts.push(s.clone()),
                    other => collect_json_strings(other, &mut parts),
                }
            }
        }
        parts.join(" ")
    } else {
        String::from_utf8_lossy(doc).to_string()
    }
}

/// Recursively collect all string values from a JSON value.
fn collect_json_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(arr) => {
            for item in arr {
                collect_json_strings(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for val in map.values() {
                collect_json_strings(val, out);
            }
        }
        serde_json::Value::Number(n) => out.push(n.to_string()),
        serde_json::Value::Bool(b) => out.push(b.to_string()),
        serde_json::Value::Null => {}
    }
}

/// Check if a filter expression contains a MATCH() call, and if so,
/// extract (column, query) and return matching PKs.
pub fn extract_match_filter(
    expr: &Expr,
    table: &str,
    db: &Database,
) -> Result<Option<Vec<String>>> {
    match expr {
        Expr::Function { name, args } if name.eq_ignore_ascii_case("MATCH") => {
            if args.len() != 2 {
                return Err(TensorError::SqlExec(
                    "MATCH() requires exactly 2 arguments: MATCH(column, 'query')".to_string(),
                ));
            }
            let _column = match &args[0] {
                Expr::Column(c) => c.clone(),
                _ => {
                    return Err(TensorError::SqlExec(
                        "MATCH() first argument must be a column name".to_string(),
                    ))
                }
            };
            let query = match &args[1] {
                Expr::StringLit(s) => s.clone(),
                _ => {
                    return Err(TensorError::SqlExec(
                        "MATCH() second argument must be a string literal".to_string(),
                    ))
                }
            };

            // Use FTS posting list intersection
            let lookup = |key: &[u8]| -> Result<Option<Vec<u8>>> { db.get(key, None, None) };
            let pks = crate::facet::fts::match_query_pks(&query, table, lookup)?;
            Ok(Some(pks))
        }
        Expr::BinOp {
            left,
            op: BinOperator::And,
            right,
        } => {
            // Check left side for MATCH
            if let Some(pks) = extract_match_filter(left, table, db)? {
                return Ok(Some(pks));
            }
            // Check right side for MATCH
            extract_match_filter(right, table, db)
        }
        _ => Ok(None),
    }
}

/// Compute BM25 relevance score for a document given a query.
pub fn bm25_score(doc_text: &str, query: &str, avg_doc_len: f64, total_docs: u64) -> f64 {
    use crate::facet::fts::{stem, tokenize};

    const K1: f64 = 1.2;
    const B: f64 = 0.75;

    let doc_tokens = tokenize(doc_text);
    let query_tokens = tokenize(query);
    let doc_len = doc_tokens.len() as f64;

    let mut score = 0.0;
    for qt in &query_tokens {
        let stemmed = stem(qt);
        let tf = doc_tokens.iter().filter(|t| stem(t) == stemmed).count() as f64;
        if tf == 0.0 {
            continue;
        }

        // IDF approximation: log((N - df + 0.5) / (df + 0.5) + 1)
        // Since we don't have df readily available, use simplified IDF
        let idf = ((total_docs as f64 + 1.0) / 2.0).ln().max(0.1);

        let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * doc_len / avg_doc_len));
        score += idf * tf_norm;
    }
    score
}

// ==================== AI SQL Functions ====================

/// Execute `EXPLAIN AI '<key>'` — return AI insights, provenance, and risk score for a key.
fn execute_explain_ai(db: &Database, _session: &mut SqlSession, key: &str) -> Result<SqlResult> {
    use crate::ai::{hex_encode, insight_prefix_for_source, quick_risk_score, AiInsight};

    let user_key = format!("table/{key}").into_bytes();
    let mut output = String::new();
    output.push_str(&format!("EXPLAIN AI for key: {key}\n\n"));

    // 1. Check if the raw key data exists and compute inline risk score
    let current_value = db.get(&user_key, None, None)?;
    if let Some(ref val) = current_value {
        let risk = quick_risk_score(val);
        output.push_str(&format!("Current value size: {} bytes\n", val.len()));
        output.push_str(&format!("Inline risk score: {risk:.3}\n"));
        let preview = String::from_utf8_lossy(&val[..val.len().min(200)]);
        output.push_str(&format!("Value preview: {preview}\n\n"));
    } else {
        // Try raw key without table/ prefix
        let raw_value = db.get(key.as_bytes(), None, None)?;
        if let Some(ref val) = raw_value {
            let risk = quick_risk_score(val);
            output.push_str(&format!("Current value size: {} bytes\n", val.len()));
            output.push_str(&format!("Inline risk score: {risk:.3}\n\n"));
        } else {
            output.push_str("Key not found in database\n\n");
        }
    }

    // 2. Scan for existing AI insights about this key
    let insight_prefix = insight_prefix_for_source(&user_key);
    let insights = db.scan_prefix(&insight_prefix, None, None, None)?;

    if insights.is_empty() {
        // Also try raw key
        let raw_prefix = insight_prefix_for_source(key.as_bytes());
        let raw_insights = db.scan_prefix(&raw_prefix, None, None, None)?;
        if raw_insights.is_empty() {
            output.push_str("No AI insights found for this key.\n");
            output.push_str(&format!("Source key (hex): {}\n", hex_encode(&user_key)));
        } else {
            output.push_str(&format!("AI Insights ({}):\n", raw_insights.len()));
            for row in &raw_insights {
                if let Ok(insight) = serde_json::from_slice::<AiInsight>(&row.doc) {
                    format_insight(&mut output, &insight);
                }
            }
        }
    } else {
        output.push_str(&format!("AI Insights ({}):\n", insights.len()));
        for row in &insights {
            if let Ok(insight) = serde_json::from_slice::<AiInsight>(&row.doc) {
                format_insight(&mut output, &insight);
            }
        }
    }

    Ok(SqlResult::Explain(output))
}

fn format_insight(output: &mut String, insight: &crate::ai::AiInsight) {
    output.push_str(&format!("  --- Insight {} ---\n", insight.insight_id));
    output.push_str(&format!("  Risk score: {:.3}\n", insight.risk_score));
    output.push_str(&format!("  Tags: {}\n", insight.tags.join(", ")));
    output.push_str(&format!("  Summary: {}\n", insight.summary));
    output.push_str(&format!("  Cluster: {}\n", insight.cluster_id));
    output.push_str(&format!(
        "  Troubleshooting: {}\n",
        insight.troubleshooting_hint
    ));
    if !insight.provenance.matched_signals.is_empty() {
        output.push_str(&format!(
            "  Matched signals: {}\n",
            insight.provenance.matched_signals.join(", ")
        ));
    }
    if !insight.provenance.risk_factors.is_empty() {
        output.push_str(&format!(
            "  Risk factors: {}\n",
            insight.provenance.risk_factors.join(", ")
        ));
    }
    output.push_str(&format!("  Model: {}\n\n", insight.model_id));
}

/// Execute `SELECT * FROM ai_top_risks` — returns top N risks from AI insights.
/// Use `LIMIT N` to control how many results. Optionally filter with WHERE.
fn execute_ai_top_risks(
    db: &Database,
    _filter: Option<&Expr>,
    limit: Option<u64>,
) -> Result<SqlResult> {
    use crate::ai::AiInsight;

    let max_results = limit.unwrap_or(10) as usize;

    // Scan all AI insights
    let all_insights = db.scan_prefix(b"__ai/insight/", None, None, None)?;

    // Parse and sort by risk score descending
    let mut parsed: Vec<AiInsight> = all_insights
        .iter()
        .filter_map(|row| serde_json::from_slice::<AiInsight>(&row.doc).ok())
        .collect();

    parsed.sort_by(|a, b| {
        b.risk_score
            .partial_cmp(&a.risk_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    parsed.truncate(max_results);

    let rows: Vec<Vec<u8>> = parsed
        .iter()
        .map(|insight| {
            serde_json::json!({
                "insight_id": insight.insight_id,
                "risk_score": insight.risk_score,
                "tags": insight.tags,
                "summary": insight.summary,
                "cluster_id": insight.cluster_id,
                "source_key": insight.source_key_hex,
                "model": insight.model_id,
            })
            .to_string()
            .into_bytes()
        })
        .collect();

    Ok(SqlResult::Rows(rows))
}

/// Execute `SELECT * FROM ai_cluster_summary WHERE cluster_id = '...'`
/// Returns all insights belonging to a specific cluster.
fn execute_ai_cluster_summary(db: &Database, filter: Option<&Expr>) -> Result<SqlResult> {
    use crate::ai::{correlation_prefix_for_cluster, AiCorrelationRef};

    // Extract cluster_id from filter: WHERE cluster_id = 'xxx'
    let cluster_id = if let Some(expr) = filter {
        extract_string_eq(expr, "cluster_id")
    } else {
        None
    };

    let cluster_id = cluster_id.ok_or_else(|| {
        TensorError::SqlExec("ai_cluster_summary requires WHERE cluster_id = '<id>'".to_string())
    })?;

    let prefix = correlation_prefix_for_cluster(&cluster_id);
    let correlations = db.scan_prefix(&prefix, None, None, None)?;

    let mut parsed: Vec<AiCorrelationRef> = correlations
        .iter()
        .filter_map(|row| serde_json::from_slice::<AiCorrelationRef>(&row.doc).ok())
        .collect();

    // Sort by risk score descending
    parsed.sort_by(|a, b| {
        b.risk_score
            .partial_cmp(&a.risk_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_risk: f64 = parsed.iter().map(|c| c.risk_score).sum();
    let avg_risk = if parsed.is_empty() {
        0.0
    } else {
        total_risk / parsed.len() as f64
    };

    // Build summary row + individual correlation rows
    let mut rows: Vec<Vec<u8>> = Vec::new();

    // Summary row
    let all_tags: Vec<String> = parsed
        .iter()
        .flat_map(|c| c.tags.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    rows.push(
        serde_json::json!({
            "cluster_id": cluster_id,
            "event_count": parsed.len(),
            "avg_risk_score": (avg_risk * 1000.0).round() / 1000.0,
            "max_risk_score": parsed.first().map(|c| c.risk_score).unwrap_or(0.0),
            "all_tags": all_tags,
            "type": "summary",
        })
        .to_string()
        .into_bytes(),
    );

    // Individual events
    for corr in &parsed {
        rows.push(
            serde_json::json!({
                "insight_id": corr.insight_id,
                "source_key": corr.source_key_hex,
                "risk_score": corr.risk_score,
                "tags": corr.tags,
                "summary": corr.summary,
                "type": "event",
            })
            .to_string()
            .into_bytes(),
        );
    }

    Ok(SqlResult::Rows(rows))
}

/// Extract a string value from an equality filter: `column = 'value'`
fn extract_string_eq(expr: &Expr, column_name: &str) -> Option<String> {
    match expr {
        Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        } => {
            if let Expr::Column(ref col) = **left {
                if col.eq_ignore_ascii_case(column_name) {
                    if let Expr::StringLit(ref val) = **right {
                        return Some(val.clone());
                    }
                }
            }
            if let Expr::Column(ref col) = **right {
                if col.eq_ignore_ascii_case(column_name) {
                    if let Expr::StringLit(ref val) = **left {
                        return Some(val.clone());
                    }
                }
            }
            None
        }
        Expr::BinOp {
            left,
            op: BinOperator::And,
            right,
        } => extract_string_eq(left, column_name).or_else(|| extract_string_eq(right, column_name)),
        _ => None,
    }
}

// ==================== Time-Series SQL Functions ====================

/// Execute `CREATE TIMESERIES TABLE <name> (<columns>) WITH (bucket_size = '<interval>')`
fn execute_create_timeseries_table(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    columns: Vec<crate::sql::parser::ColumnDef>,
    bucket_interval: &str,
) -> Result<SqlResult> {
    use crate::facet::relational::validate_table_name;

    validate_table_name(table)?;
    let key = table_meta_key(table);
    if read_live_key(db, session, &key, None, None)?.is_some() {
        return Err(TensorError::SqlExec(format!(
            "table {table} already exists"
        )));
    }

    // Parse bucket interval
    let bucket_seconds = parse_interval_seconds(bucket_interval).ok_or_else(|| {
        TensorError::SqlExec(format!("invalid bucket_size interval: '{bucket_interval}'"))
    })?;

    // Find the timestamp column (first column of type INTEGER or REAL named 'ts' or 'timestamp')
    let ts_col = columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case("ts") || c.name.eq_ignore_ascii_case("timestamp"))
        .or_else(|| {
            columns.iter().find(|c| {
                matches!(
                    c.type_name,
                    crate::sql::parser::SqlType::Integer | crate::sql::parser::SqlType::Real
                ) && !c.primary_key
            })
        })
        .map(|c| c.name.clone())
        .unwrap_or_else(|| "ts".to_string());

    // Find value columns (numeric columns that aren't PK or timestamp)
    let value_columns: Vec<String> = columns
        .iter()
        .filter(|c| {
            !c.primary_key
                && !c.name.eq_ignore_ascii_case(&ts_col)
                && matches!(
                    c.type_name,
                    crate::sql::parser::SqlType::Integer | crate::sql::parser::SqlType::Real
                )
        })
        .map(|c| c.name.clone())
        .collect();

    // Find the primary key column
    let pk_col = columns.iter().find(|c| c.primary_key).ok_or_else(|| {
        TensorError::SqlExec(
            "CREATE TIMESERIES TABLE requires at least one PRIMARY KEY column".to_string(),
        )
    })?;

    // Create the typed table schema
    let cols = columns
        .iter()
        .map(|c| TableColumnMetadata {
            name: c.name.clone(),
            type_name: c.type_name.name().to_string(),
        })
        .collect();

    let schema = TableSchemaMetadata {
        pk: pk_col.name.clone(),
        doc: "typed".to_string(),
        columns: cols,
        schema_mode: "timeseries".to_string(),
    };

    let payload = encode_schema_metadata(&schema)?;
    let commit_ts = write_put(db, session, key, payload, 0, u64::MAX, Some(1))?;

    // Store timeseries metadata
    let ts_meta = TimeseriesMetadata {
        table: table.to_string(),
        bucket_interval: bucket_interval.to_string(),
        bucket_seconds,
        ts_column: ts_col.clone(),
        value_columns: value_columns.clone(),
    };
    let ts_meta_bytes = serde_json::to_vec(&ts_meta)?;
    let ts_key = ts_table_meta_key(table);
    write_put(db, session, ts_key, ts_meta_bytes, 0, u64::MAX, Some(1))?;

    let val_cols = if value_columns.is_empty() {
        String::new()
    } else {
        format!(", value columns: {}", value_columns.join(", "))
    };

    Ok(SqlResult::Affected {
        rows: 1,
        commit_ts,
        message: format!(
            "created timeseries table {table} (bucket_size={bucket_interval}, ts_column={ts_col}{val_cols})"
        ),
    })
}

/// Update time-series bucket storage when a row is inserted into a TS table.
fn update_ts_buckets(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    doc: &[u8],
) -> Result<()> {
    use crate::facet::timeseries::{bucket_id, ts_bucket_key, Bucket};

    // Check if this is a timeseries table
    let ts_key = ts_table_meta_key(table);
    let ts_meta_bytes = match read_live_key(db, session, &ts_key, None, None)? {
        Some(b) => b,
        None => return Ok(()), // Not a timeseries table
    };
    let ts_meta: TimeseriesMetadata = serde_json::from_slice(&ts_meta_bytes)
        .map_err(|e| TensorError::SqlExec(format!("invalid timeseries metadata: {e}")))?;

    // Parse the doc as JSON to extract ts and value columns
    let json: serde_json::Value = serde_json::from_slice(doc)
        .map_err(|e| TensorError::SqlExec(format!("invalid JSON for timeseries row: {e}")))?;

    let timestamp = json
        .get(&ts_meta.ts_column)
        .and_then(|v| v.as_f64())
        .or_else(|| {
            json.get(&ts_meta.ts_column)
                .and_then(|v| v.as_u64().map(|u| u as f64))
        });

    let timestamp = match timestamp {
        Some(ts) => ts as u64,
        None => return Ok(()), // No timestamp column in this row
    };

    // For each value column, update the corresponding bucket
    for val_col in &ts_meta.value_columns {
        let value = match json.get(val_col).and_then(|v| v.as_f64()) {
            Some(v) => v,
            None => continue,
        };

        let bid = bucket_id(timestamp, ts_meta.bucket_seconds);
        let bkey = ts_bucket_key(table, val_col, bid);

        // Load existing bucket or create new
        let mut bucket = match read_live_key(db, session, &bkey, None, None)? {
            Some(existing) => Bucket::decode(&existing)?,
            None => Bucket::new(),
        };

        bucket.add(timestamp, value);
        let encoded = bucket.encode();
        write_put(db, session, bkey, encoded, 0, u64::MAX, Some(1))?;
    }

    Ok(())
}

/// Load timeseries metadata for a table, if it exists.
#[allow(dead_code)]
fn load_ts_metadata(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
) -> Result<Option<TimeseriesMetadata>> {
    let ts_key = ts_table_meta_key(table);
    match read_live_key(db, session, &ts_key, None, None)? {
        Some(bytes) => {
            let meta: TimeseriesMetadata = serde_json::from_slice(&bytes)
                .map_err(|e| TensorError::SqlExec(format!("invalid timeseries metadata: {e}")))?;
            Ok(Some(meta))
        }
        None => Ok(None),
    }
}

fn execute_explain_analyze(
    db: &Database,
    session: &mut SqlSession,
    inner: Statement,
) -> Result<SqlResult> {
    // Build planner stats from persisted ANALYZE data
    let tables = collect_statement_tables(&inner);
    let table_refs: Vec<&str> = tables.iter().map(|s| s.as_str()).collect();
    let planner_stats = build_planner_stats(db, session, &table_refs);

    // Generate the plan tree before execution
    let plan_tree = generate_plan_with_stats(&inner, Some(&planner_stats));

    // Capture stats before execution
    let stats_before = db.stats().ok();

    let start = Instant::now();
    let result = execute_stmt(db, session, inner)?;
    let elapsed = start.elapsed();

    // Capture stats after execution to compute deltas
    let stats_after = db.stats().ok();

    let (rows_returned, detail) = match &result {
        SqlResult::Rows(rows) => (rows.len() as u64, String::new()),
        SqlResult::Affected { rows, message, .. } => (*rows, format!(" message={message}")),
        SqlResult::Explain(text) => (0, format!(" explain={text}")),
    };

    let mut output = format!(
        "execution_time_us={} rows_returned={rows_returned}{detail}",
        elapsed.as_micros(),
    );

    if let (Some(before), Some(after)) = (stats_before, stats_after) {
        let reads_delta = after.gets.saturating_sub(before.gets);
        let writes_delta = after.puts.saturating_sub(before.puts);
        let bloom_neg_delta = after.bloom_negatives.saturating_sub(before.bloom_negatives);
        output.push_str(&format!(
            "\noperations: reads={reads_delta} writes={writes_delta} bloom_negatives={bloom_neg_delta}"
        ));
    }

    if let Some(plan) = plan_tree {
        output.push_str(&format!("\nplan_cost={:.1}", plan.cost()));
    }

    Ok(SqlResult::Explain(output))
}

fn execute_explain(db: &Database, session: &mut SqlSession, inner: Statement) -> Result<SqlResult> {
    // Build planner stats and try cost-based plan tree
    let tables = collect_statement_tables(&inner);
    let table_refs: Vec<&str> = tables.iter().map(|s| s.as_str()).collect();
    let planner_stats = build_planner_stats(db, session, &table_refs);
    let plan_output = explain_plan_with_stats(&inner, Some(&planner_stats));

    match inner {
        Statement::Select {
            ref from,
            ref items,
            ref joins,
            ref filter,
            ref as_of,
            ref valid_at,
            ..
        } => {
            let mut output = plan_output;

            // For point lookups, also include storage-level detail
            if let TableRef::Named(ref table) = from {
                let is_doc_only = detect_legacy_projection(items)
                    .map(|l| matches!(l, LegacyProjection::Doc))
                    .unwrap_or(false);

                if is_doc_only && joins.is_empty() && filter.is_some() {
                    if let Some(pk) = extract_pk_eq_literal(filter.as_ref()) {
                        let (target_table, _, effective_as_of, effective_valid_at) =
                            resolve_select_target(db, session, table, *as_of, *valid_at)?;
                        let key = row_key(&target_table, &pk);
                        let e = db.explain_get(&key, effective_as_of, effective_valid_at)?;
                        output.push_str(&format!(
                            "\nStorage: shard_id={} bloom_hit={:?} sstable_block={:?} commit_ts_used={}",
                            e.shard_id, e.bloom_hit, e.sstable_block, e.commit_ts_used
                        ));
                    }
                }
            }

            Ok(SqlResult::Explain(output))
        }
        _ => Ok(SqlResult::Explain(plan_output)),
    }
}

/// Quote a CSV field per RFC 4180: enclose in double quotes if it contains comma,
/// double quote, or newline. Double quotes inside are escaped by doubling.
fn csv_quote(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

/// Parse a CSV line into fields, handling RFC 4180 quoted fields.
fn csv_parse_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next(); // consume escaped quote
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == ',' {
            fields.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
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
    let is_typed = schema.columns.len() > 2
        || (schema.columns.len() == 2 && !schema.columns.iter().any(|c| c.name == "doc"));

    let mut output = String::new();
    match format {
        CopyFormat::Csv => {
            let col_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
            output.push_str(
                &col_names
                    .iter()
                    .map(|n| csv_quote(n))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            output.push('\n');
            for row in &rows {
                if is_typed {
                    // Typed table: extract each column from the JSON doc
                    let doc_val: serde_json::Value = serde_json::from_slice(&row.doc)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    let mut fields = Vec::new();
                    for col in &schema.columns {
                        let val = doc_val.get(&col.name).unwrap_or(&serde_json::Value::Null);
                        let s = match val {
                            serde_json::Value::Null => String::new(),
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            other => other.to_string(),
                        };
                        fields.push(csv_quote(&s));
                    }
                    output.push_str(&fields.join(","));
                } else {
                    // Legacy table: pk,doc
                    output.push_str(&csv_quote(&row.pk));
                    output.push(',');
                    output.push_str(&csv_quote(&String::from_utf8_lossy(&row.doc)));
                }
                output.push('\n');
            }
        }
        CopyFormat::Json => {
            let mut arr = Vec::new();
            for row in &rows {
                if is_typed {
                    let doc_val: serde_json::Value = serde_json::from_slice(&row.doc)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    arr.push(doc_val);
                } else {
                    arr.push(serde_json::json!({
                        "pk": row.pk,
                        "doc": decode_row_json(&row.doc),
                    }));
                }
            }
            output = serde_json::to_string_pretty(&arr).unwrap_or_default();
        }
        CopyFormat::Ndjson => {
            for row in &rows {
                if is_typed {
                    let doc_val: serde_json::Value = serde_json::from_slice(&row.doc)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    output.push_str(&serde_json::to_string(&doc_val).unwrap_or_default());
                } else {
                    let obj = serde_json::json!({
                        "pk": row.pk,
                        "doc": decode_row_json(&row.doc),
                    });
                    output.push_str(&serde_json::to_string(&obj).unwrap_or_default());
                }
                output.push('\n');
            }
        }
        CopyFormat::Parquet => {
            #[cfg(feature = "parquet")]
            {
                return write_parquet(&schema, &rows, is_typed, path);
            }
            #[cfg(not(feature = "parquet"))]
            {
                return Err(TensorError::SqlExec(
                    "Parquet support requires --features parquet".to_string(),
                ));
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
    let schema = load_table_schema(db, session, table)?;
    let is_typed = schema.columns.len() > 2
        || (schema.columns.len() == 2 && !schema.columns.iter().any(|c| c.name == "doc"));

    let mut count = 0u64;
    let mut last_commit_ts = None;

    match format {
        CopyFormat::Csv => {
            let content = std::fs::read_to_string(path)?;
            let mut lines = content.lines();
            let header_line = lines.next().unwrap_or("");
            let headers: Vec<String> = csv_parse_line(header_line)
                .into_iter()
                .map(|s| s.trim().to_string())
                .collect();

            for line in lines {
                if line.trim().is_empty() {
                    continue;
                }
                let fields = csv_parse_line(line);

                if is_typed && !headers.is_empty() {
                    // Typed table: build JSON from CSV columns
                    let mut json_obj = serde_json::Map::new();
                    for (i, header) in headers.iter().enumerate() {
                        let val = fields.get(i).map(|s| s.as_str()).unwrap_or("");
                        let col_meta = schema.columns.iter().find(|c| c.name == *header);
                        let json_val = csv_field_to_json(val, col_meta);
                        json_obj.insert(header.clone(), json_val);
                    }
                    let pk = json_obj
                        .get(&schema.pk)
                        .and_then(|v| match v {
                            serde_json::Value::String(s) => Some(s.clone()),
                            serde_json::Value::Number(n) => Some(n.to_string()),
                            _ => None,
                        })
                        .unwrap_or_else(|| format!("row_{count}"));
                    validate_pk(&pk)?;
                    let doc = serde_json::to_vec(&serde_json::Value::Object(json_obj))?;
                    let key = row_key(table, &pk);
                    let ts = write_put(db, session, key, doc, 0, u64::MAX, Some(1))?;
                    last_commit_ts = ts.or(last_commit_ts);
                    count += 1;
                } else {
                    // Legacy table: first field is pk, second is doc JSON
                    let pk = fields
                        .first()
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default();
                    let doc = fields
                        .get(1)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_else(|| "{}".to_string());
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
        }
        CopyFormat::Json => {
            let content = std::fs::read_to_string(path)?;
            let arr: Vec<serde_json::Value> = serde_json::from_str(&content)?;
            for val in arr {
                let (pk, doc) = extract_pk_and_doc(&val, &schema, is_typed)?;
                validate_pk(&pk)?;
                let key = row_key(table, &pk);
                let ts = write_put(db, session, key, doc, 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                count += 1;
            }
        }
        CopyFormat::Ndjson => {
            let content = std::fs::read_to_string(path)?;
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let val: serde_json::Value = serde_json::from_str(line)?;
                let (pk, doc) = extract_pk_and_doc(&val, &schema, is_typed)?;
                validate_pk(&pk)?;
                let key = row_key(table, &pk);
                let ts = write_put(db, session, key, doc, 0, u64::MAX, Some(1))?;
                last_commit_ts = ts.or(last_commit_ts);
                count += 1;
            }
        }
        CopyFormat::Parquet => {
            #[cfg(feature = "parquet")]
            {
                return read_parquet_into_table(db, session, table, &schema, is_typed, path);
            }
            #[cfg(not(feature = "parquet"))]
            {
                return Err(TensorError::SqlExec(
                    "Parquet support requires --features parquet".to_string(),
                ));
            }
        }
    }

    Ok(SqlResult::Affected {
        rows: count,
        commit_ts: last_commit_ts,
        message: format!("imported {count} rows from {path}"),
    })
}

/// Convert a CSV text field to a JSON value using column type info.
fn csv_field_to_json(val: &str, col_meta: Option<&TableColumnMetadata>) -> serde_json::Value {
    if val.is_empty() {
        return serde_json::Value::Null;
    }
    let type_hint = col_meta.map(|c| c.type_name.to_uppercase());
    match type_hint.as_deref() {
        Some("INTEGER") | Some("INT") | Some("BIGINT") => {
            if let Ok(n) = val.parse::<i64>() {
                serde_json::Value::Number(serde_json::Number::from(n))
            } else {
                serde_json::Value::String(val.to_string())
            }
        }
        Some("REAL") | Some("FLOAT") | Some("DOUBLE") => {
            if let Ok(n) = val.parse::<f64>() {
                serde_json::Number::from_f64(n)
                    .map(serde_json::Value::Number)
                    .unwrap_or_else(|| serde_json::Value::String(val.to_string()))
            } else {
                serde_json::Value::String(val.to_string())
            }
        }
        Some("BOOLEAN") | Some("BOOL") => {
            let lower = val.to_lowercase();
            serde_json::Value::Bool(lower == "true" || lower == "1" || lower == "yes")
        }
        _ => serde_json::Value::String(val.to_string()),
    }
}

/// Extract pk and doc bytes from a JSON value for import.
fn extract_pk_and_doc(
    val: &serde_json::Value,
    schema: &TableSchemaMetadata,
    is_typed: bool,
) -> Result<(String, Vec<u8>)> {
    if is_typed {
        // Typed table: the JSON object IS the row, pk is extracted from schema.pk field
        let pk = val
            .get(&schema.pk)
            .and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .ok_or_else(|| {
                TensorError::SqlExec(format!("JSON row missing pk column '{}'", schema.pk))
            })?;
        let doc = serde_json::to_vec(val)?;
        Ok((pk, doc))
    } else {
        // Legacy table: expect {"pk": ..., "doc": ...}
        let pk = val
            .get("pk")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensorError::SqlExec("JSON row missing 'pk'".to_string()))?
            .to_string();
        let doc = val
            .get("doc")
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_else(|| "{}".to_string());
        Ok((pk, doc.into_bytes()))
    }
}

// ---------- Table functions ----------

fn execute_table_function(name: &str, args: &[Expr]) -> Result<Vec<VisibleRow>> {
    let upper = name.to_uppercase();
    match upper.as_str() {
        "READ_CSV" => {
            let path = match args.first() {
                Some(Expr::StringLit(s)) => s.clone(),
                _ => {
                    return Err(TensorError::SqlExec(
                        "read_csv() requires a file path as first argument".to_string(),
                    ))
                }
            };
            let content = std::fs::read_to_string(&path)
                .map_err(|e| TensorError::SqlExec(format!("failed to read file '{path}': {e}")))?;
            let mut lines = content.lines();
            let header_line = lines
                .next()
                .ok_or_else(|| TensorError::SqlExec("CSV file is empty".to_string()))?;
            let headers = csv_parse_line(header_line);

            let mut rows = Vec::new();
            for (i, line) in lines.enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let fields = csv_parse_line(line);
                let mut obj = serde_json::Map::new();
                for (col_idx, header) in headers.iter().enumerate() {
                    let val = fields.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                    let header_clean = header.trim().to_string();
                    // Auto-detect type
                    let json_val = if val.is_empty() {
                        serde_json::Value::Null
                    } else if let Ok(n) = val.parse::<i64>() {
                        serde_json::Value::Number(serde_json::Number::from(n))
                    } else if let Ok(n) = val.parse::<f64>() {
                        serde_json::Number::from_f64(n)
                            .map(serde_json::Value::Number)
                            .unwrap_or_else(|| serde_json::Value::String(val.to_string()))
                    } else if val.eq_ignore_ascii_case("true") || val.eq_ignore_ascii_case("false")
                    {
                        serde_json::Value::Bool(val.eq_ignore_ascii_case("true"))
                    } else {
                        serde_json::Value::String(val.to_string())
                    };
                    obj.insert(header_clean, json_val);
                }
                let doc = serde_json::to_vec(&serde_json::Value::Object(obj))?;
                rows.push(VisibleRow {
                    pk: i.to_string(),
                    doc,
                });
            }
            Ok(rows)
        }
        "READ_JSON" => {
            let path = match args.first() {
                Some(Expr::StringLit(s)) => s.clone(),
                _ => {
                    return Err(TensorError::SqlExec(
                        "read_json() requires a file path as first argument".to_string(),
                    ))
                }
            };
            let content = std::fs::read_to_string(&path)
                .map_err(|e| TensorError::SqlExec(format!("failed to read file '{path}': {e}")))?;
            let arr: Vec<serde_json::Value> = serde_json::from_str(&content)?;
            let mut rows = Vec::new();
            for (i, val) in arr.iter().enumerate() {
                let doc = serde_json::to_vec(val)?;
                rows.push(VisibleRow {
                    pk: i.to_string(),
                    doc,
                });
            }
            Ok(rows)
        }
        "READ_NDJSON" => {
            let path = match args.first() {
                Some(Expr::StringLit(s)) => s.clone(),
                _ => {
                    return Err(TensorError::SqlExec(
                        "read_ndjson() requires a file path as first argument".to_string(),
                    ))
                }
            };
            let content = std::fs::read_to_string(&path)
                .map_err(|e| TensorError::SqlExec(format!("failed to read file '{path}': {e}")))?;
            let mut rows = Vec::new();
            for (i, line) in content.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let val: serde_json::Value = serde_json::from_str(line)?;
                let doc = serde_json::to_vec(&val)?;
                rows.push(VisibleRow {
                    pk: i.to_string(),
                    doc,
                });
            }
            Ok(rows)
        }
        "READ_PARQUET" => {
            #[cfg(feature = "parquet")]
            {
                let path = match args.first() {
                    Some(Expr::StringLit(s)) => s.clone(),
                    _ => {
                        return Err(TensorError::SqlExec(
                            "read_parquet() requires a file path as first argument".to_string(),
                        ))
                    }
                };
                read_parquet_rows(&path)
            }
            #[cfg(not(feature = "parquet"))]
            {
                let _ = args;
                Err(TensorError::SqlExec(
                    "Parquet support requires --features parquet".to_string(),
                ))
            }
        }
        _ => Err(TensorError::SqlExec(format!(
            "unknown table function: {name}()"
        ))),
    }
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
        .ok_or_else(|| TensorError::SqlExec(format!("table {table} does not exist")))?;
    parse_schema_metadata(&bytes)
}

fn resolve_select_target(
    db: &Database,
    session: &SqlSession,
    source: &str,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Result<ResolvedTarget> {
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
        return Err(TensorError::SqlExec(format!(
            "view {source} has unsupported query shape"
        )));
    }

    Err(TensorError::SqlExec(format!(
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
            .map_err(|_| TensorError::SqlExec("row key contains non-utf8 pk".to_string()))?
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

// ---------- Set Operations (UNION, INTERSECT, EXCEPT) ----------

fn execute_set_op(
    db: &Database,
    session: &mut SqlSession,
    op: SetOpType,
    left: Statement,
    right: Statement,
) -> Result<SqlResult> {
    let left_result = execute_stmt(db, session, left)?;
    let right_result = execute_stmt(db, session, right)?;

    let left_rows = match left_result {
        SqlResult::Rows(r) => r,
        _ => {
            return Err(TensorError::SqlExec(
                "set operation requires SELECT statements".to_string(),
            ))
        }
    };
    let right_rows = match right_result {
        SqlResult::Rows(r) => r,
        _ => {
            return Err(TensorError::SqlExec(
                "set operation requires SELECT statements".to_string(),
            ))
        }
    };

    let result = match op {
        SetOpType::UnionAll => {
            let mut combined = left_rows;
            combined.extend(right_rows);
            combined
        }
        SetOpType::Union => {
            let mut combined = left_rows;
            combined.extend(right_rows);
            // Deduplicate by content
            let mut seen = std::collections::HashSet::new();
            combined.retain(|row| seen.insert(row.clone()));
            combined
        }
        SetOpType::Intersect => {
            let right_set: std::collections::HashSet<Vec<u8>> = right_rows.into_iter().collect();
            let mut result = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for row in left_rows {
                if right_set.contains(&row) && seen.insert(row.clone()) {
                    result.push(row);
                }
            }
            result
        }
        SetOpType::Except => {
            let right_set: std::collections::HashSet<Vec<u8>> = right_rows.into_iter().collect();
            let mut result = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for row in left_rows {
                if !right_set.contains(&row) && seen.insert(row.clone()) {
                    result.push(row);
                }
            }
            result
        }
    };

    Ok(SqlResult::Rows(result))
}

// ---------- INSERT ... RETURNING ----------

fn execute_insert_returning(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    columns: Vec<String>,
    values: Vec<Expr>,
    returning: &[SelectItem],
) -> Result<SqlResult> {
    validate_table_name(table)?;
    let schema = load_table_schema(db, session, table)?;

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
                    return Err(TensorError::SqlExec(
                        "primary key must be text or number".to_string(),
                    ))
                }
            });
        }

        json_obj.insert(col_name.clone(), json_val);
    }

    let pk = pk_val.ok_or_else(|| {
        TensorError::SqlExec(format!(
            "INSERT must include primary key column '{}'",
            schema.pk
        ))
    })?;
    validate_pk(&pk)?;

    let doc = serde_json::to_vec(&serde_json::Value::Object(json_obj))?;
    let key = row_key(table, &pk);
    write_put(db, session, key, doc.clone(), 0, u64::MAX, Some(1))?;
    // Update FTS indexes
    update_fts_indexes(db, session, table, &pk, &doc)?;
    // Update time-series buckets
    update_ts_buckets(db, session, table, &doc)?;
    // Update vector indexes
    update_vector_indexes(db, session, table, &pk, &doc)?;

    // Build RETURNING result
    let mut ctx = EvalContext::new(&pk, &doc);
    let mut row_json = serde_json::Map::new();
    for item in returning {
        match item {
            SelectItem::AllColumns => {
                // Return all columns
                if let Ok(serde_json::Value::Object(map)) =
                    serde_json::from_slice::<serde_json::Value>(&doc)
                {
                    for (k, v) in map {
                        row_json.insert(k, v);
                    }
                }
            }
            SelectItem::Expr { expr, alias } => {
                let val = ctx.eval(expr)?;
                let col_name = alias.clone().unwrap_or_else(|| expr_display_name(expr));
                row_json.insert(col_name, sql_value_to_json(&val));
            }
        }
    }

    let row_bytes = serde_json::to_vec(&serde_json::Value::Object(row_json))?;
    Ok(SqlResult::Rows(vec![row_bytes]))
}

// ---------- CREATE TABLE AS SELECT ----------

fn execute_create_table_as(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    query: Statement,
) -> Result<SqlResult> {
    validate_table_name(table)?;

    // Check table doesn't exist
    let key = table_meta_key(table);
    if read_live_key(db, session, &key, None, None)?.is_some() {
        return Err(TensorError::SqlExec(format!(
            "table {table} already exists"
        )));
    }

    // Execute the query
    let result = execute_stmt(db, session, query)?;
    let rows = match result {
        SqlResult::Rows(r) => r,
        _ => {
            return Err(TensorError::SqlExec(
                "CREATE TABLE AS requires a SELECT statement".to_string(),
            ))
        }
    };

    // Infer schema from first row
    let columns = if let Some(first) = rows.first() {
        if let Ok(serde_json::Value::Object(map)) =
            serde_json::from_slice::<serde_json::Value>(first)
        {
            let mut cols = Vec::new();
            let mut has_pk = false;
            for (k, v) in &map {
                let type_name = match v {
                    serde_json::Value::Number(_) => "REAL",
                    serde_json::Value::Bool(_) => "BOOLEAN",
                    serde_json::Value::String(_) => "TEXT",
                    _ => "TEXT",
                };
                let is_pk = k.eq_ignore_ascii_case("pk") || k.eq_ignore_ascii_case("id");
                if is_pk {
                    has_pk = true;
                }
                cols.push(TableColumnMetadata {
                    name: k.clone(),
                    type_name: type_name.to_string(),
                });
            }
            if !has_pk && !cols.is_empty() {
                // Mark first column as PK-like
                cols[0].type_name = format!("{} (pk)", cols[0].type_name);
            }
            cols
        } else {
            vec![
                TableColumnMetadata {
                    name: "pk".to_string(),
                    type_name: "TEXT".to_string(),
                },
                TableColumnMetadata {
                    name: "doc".to_string(),
                    type_name: "JSON".to_string(),
                },
            ]
        }
    } else {
        vec![
            TableColumnMetadata {
                name: "pk".to_string(),
                type_name: "TEXT".to_string(),
            },
            TableColumnMetadata {
                name: "doc".to_string(),
                type_name: "JSON".to_string(),
            },
        ]
    };

    // Find PK column name
    let pk_name = columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case("pk") || c.name.eq_ignore_ascii_case("id"))
        .map(|c| c.name.clone())
        .unwrap_or_else(|| {
            columns
                .first()
                .map(|c| c.name.clone())
                .unwrap_or("pk".into())
        });

    // Create schema metadata
    let schema = TableSchemaMetadata {
        pk: pk_name.clone(),
        doc: "json".to_string(),
        columns,
        schema_mode: "legacy".to_string(),
    };
    let payload = encode_schema_metadata(&schema)?;
    write_put(db, session, key, payload, 0, u64::MAX, Some(1))?;

    // Insert rows
    let mut count = 0u64;
    for (i, row_data) in rows.iter().enumerate() {
        let pk = if let Ok(serde_json::Value::Object(ref map)) =
            serde_json::from_slice::<serde_json::Value>(row_data)
        {
            map.get(&pk_name)
                .and_then(|v| match v {
                    serde_json::Value::String(s) => Some(s.clone()),
                    serde_json::Value::Number(n) => Some(n.to_string()),
                    _ => None,
                })
                .unwrap_or_else(|| format!("row_{i}"))
        } else {
            format!("row_{i}")
        };
        let key = row_key(table, &pk);
        write_put(db, session, key, row_data.clone(), 0, u64::MAX, Some(1))?;
        count += 1;
    }

    Ok(SqlResult::Affected {
        rows: count,
        commit_ts: None,
        message: format!("created table {table} with {count} row(s)"),
    })
}

// ---------- Parquet support (behind feature flag) ----------

#[cfg(feature = "parquet")]
fn write_parquet(
    schema: &TableSchemaMetadata,
    rows: &[VisibleRow],
    is_typed: bool,
    path: &str,
) -> Result<SqlResult> {
    use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    // Build Arrow schema from table schema
    let fields: Vec<Field> = if is_typed {
        schema
            .columns
            .iter()
            .map(|col| {
                let dt = match col.type_name.to_uppercase().as_str() {
                    "INTEGER" | "INT" | "BIGINT" => DataType::Int64,
                    "REAL" | "FLOAT" | "DOUBLE" => DataType::Float64,
                    "BOOLEAN" | "BOOL" => DataType::Boolean,
                    _ => DataType::Utf8,
                };
                Field::new(&col.name, dt, true)
            })
            .collect()
    } else {
        vec![
            Field::new("pk", DataType::Utf8, false),
            Field::new("doc", DataType::Utf8, true),
        ]
    };

    let arrow_schema = Arc::new(Schema::new(fields));
    let file = std::fs::File::create(path)
        .map_err(|e| TensorError::SqlExec(format!("failed to create file '{path}': {e}")))?;
    let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), None)
        .map_err(|e| TensorError::SqlExec(format!("parquet writer error: {e}")))?;

    if is_typed {
        // Build column arrays from row data
        let mut builders: Vec<Box<dyn std::any::Any>> = schema
            .columns
            .iter()
            .map(|col| -> Box<dyn std::any::Any> {
                match col.type_name.to_uppercase().as_str() {
                    "INTEGER" | "INT" | "BIGINT" => Box::new(Int64Builder::new()),
                    "REAL" | "FLOAT" | "DOUBLE" => Box::new(Float64Builder::new()),
                    "BOOLEAN" | "BOOL" => Box::new(BooleanBuilder::new()),
                    _ => Box::new(StringBuilder::new()),
                }
            })
            .collect();

        for row in rows {
            let doc_val: serde_json::Value = serde_json::from_slice(&row.doc)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            for (i, col) in schema.columns.iter().enumerate() {
                let val = doc_val.get(&col.name).unwrap_or(&serde_json::Value::Null);
                match col.type_name.to_uppercase().as_str() {
                    "INTEGER" | "INT" | "BIGINT" => {
                        let b = builders[i].downcast_mut::<Int64Builder>().unwrap();
                        match val {
                            serde_json::Value::Number(n) => b.append_value(
                                n.as_i64().unwrap_or(n.as_f64().unwrap_or(0.0) as i64),
                            ),
                            serde_json::Value::Null => b.append_null(),
                            _ => b.append_null(),
                        }
                    }
                    "REAL" | "FLOAT" | "DOUBLE" => {
                        let b = builders[i].downcast_mut::<Float64Builder>().unwrap();
                        match val {
                            serde_json::Value::Number(n) => {
                                b.append_value(n.as_f64().unwrap_or(0.0))
                            }
                            serde_json::Value::Null => b.append_null(),
                            _ => b.append_null(),
                        }
                    }
                    "BOOLEAN" | "BOOL" => {
                        let b = builders[i].downcast_mut::<BooleanBuilder>().unwrap();
                        match val {
                            serde_json::Value::Bool(v) => b.append_value(*v),
                            serde_json::Value::Null => b.append_null(),
                            _ => b.append_null(),
                        }
                    }
                    _ => {
                        let b = builders[i].downcast_mut::<StringBuilder>().unwrap();
                        match val {
                            serde_json::Value::String(s) => b.append_value(s),
                            serde_json::Value::Null => b.append_null(),
                            other => b.append_value(other.to_string()),
                        }
                    }
                }
            }
        }

        let arrays: Vec<ArrayRef> = builders
            .iter_mut()
            .enumerate()
            .map(|(i, builder)| -> ArrayRef {
                match schema.columns[i].type_name.to_uppercase().as_str() {
                    "INTEGER" | "INT" | "BIGINT" => {
                        Arc::new(builder.downcast_mut::<Int64Builder>().unwrap().finish())
                    }
                    "REAL" | "FLOAT" | "DOUBLE" => {
                        Arc::new(builder.downcast_mut::<Float64Builder>().unwrap().finish())
                    }
                    "BOOLEAN" | "BOOL" => {
                        Arc::new(builder.downcast_mut::<BooleanBuilder>().unwrap().finish())
                    }
                    _ => Arc::new(builder.downcast_mut::<StringBuilder>().unwrap().finish()),
                }
            })
            .collect();

        let batch = RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| TensorError::SqlExec(format!("arrow batch error: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| TensorError::SqlExec(format!("parquet write error: {e}")))?;
    } else {
        // Legacy: pk + doc as strings
        let mut pk_builder = StringBuilder::new();
        let mut doc_builder = StringBuilder::new();
        for row in rows {
            pk_builder.append_value(&row.pk);
            doc_builder.append_value(String::from_utf8_lossy(&row.doc));
        }
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(pk_builder.finish()),
            Arc::new(doc_builder.finish()),
        ];
        let batch = RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| TensorError::SqlExec(format!("arrow batch error: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| TensorError::SqlExec(format!("parquet write error: {e}")))?;
    }

    writer
        .close()
        .map_err(|e| TensorError::SqlExec(format!("parquet close error: {e}")))?;
    Ok(SqlResult::Affected {
        rows: rows.len() as u64,
        commit_ts: None,
        message: format!("exported {} rows to {}", rows.len(), path),
    })
}

#[cfg(feature = "parquet")]
fn read_parquet_into_table(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    schema: &TableSchemaMetadata,
    is_typed: bool,
    path: &str,
) -> Result<SqlResult> {
    let rows = read_parquet_rows(path)?;
    let mut count = 0u64;
    let mut last_commit_ts = None;

    for row in &rows {
        let doc_val: serde_json::Value = serde_json::from_slice(&row.doc)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let (pk, doc) = if is_typed {
            let pk = doc_val
                .get(&schema.pk)
                .and_then(|v| match v {
                    serde_json::Value::String(s) => Some(s.clone()),
                    serde_json::Value::Number(n) => Some(n.to_string()),
                    _ => None,
                })
                .unwrap_or_else(|| format!("row_{count}"));
            (pk, row.doc.clone())
        } else {
            let pk = doc_val
                .get("pk")
                .and_then(|v| v.as_str())
                .unwrap_or(&row.pk)
                .to_string();
            (pk, row.doc.clone())
        };
        validate_pk(&pk)?;
        let key = row_key(table, &pk);
        let ts = write_put(db, session, key, doc, 0, u64::MAX, Some(1))?;
        last_commit_ts = ts.or(last_commit_ts);
        count += 1;
    }

    Ok(SqlResult::Affected {
        rows: count,
        commit_ts: last_commit_ts,
        message: format!("imported {count} rows from {path}"),
    })
}

#[cfg(feature = "parquet")]
fn read_parquet_rows(path: &str) -> Result<Vec<VisibleRow>> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::DataType;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)
        .map_err(|e| TensorError::SqlExec(format!("failed to open file '{path}': {e}")))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| TensorError::SqlExec(format!("parquet reader error: {e}")))?;
    let reader = builder
        .build()
        .map_err(|e| TensorError::SqlExec(format!("parquet build error: {e}")))?;

    let mut all_rows = Vec::new();
    let mut row_idx = 0usize;

    for batch_result in reader {
        let batch =
            batch_result.map_err(|e| TensorError::SqlExec(format!("parquet read error: {e}")))?;
        let schema = batch.schema();

        for row_i in 0..batch.num_rows() {
            let mut obj = serde_json::Map::new();
            for (col_i, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_i);
                let val = if col.is_null(row_i) {
                    serde_json::Value::Null
                } else {
                    match field.data_type() {
                        DataType::Int8 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::Int8Type>()
                            .value(row_i)),
                        DataType::Int16 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::Int16Type>()
                            .value(row_i)),
                        DataType::Int32 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::Int32Type>()
                            .value(row_i)),
                        DataType::Int64 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::Int64Type>()
                            .value(row_i)),
                        DataType::UInt8 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::UInt8Type>()
                            .value(row_i)),
                        DataType::UInt16 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::UInt16Type>()
                            .value(row_i)),
                        DataType::UInt32 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::UInt32Type>()
                            .value(row_i)),
                        DataType::UInt64 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::UInt64Type>()
                            .value(row_i)),
                        DataType::Float32 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::Float32Type>()
                            .value(row_i)),
                        DataType::Float64 => serde_json::json!(col
                            .as_primitive::<arrow::datatypes::Float64Type>()
                            .value(row_i)),
                        DataType::Boolean => serde_json::json!(col.as_boolean().value(row_i)),
                        DataType::Utf8 => {
                            let arr = col
                                .as_any()
                                .downcast_ref::<arrow::array::StringArray>()
                                .unwrap();
                            serde_json::json!(arr.value(row_i))
                        }
                        DataType::LargeUtf8 => {
                            let arr = col
                                .as_any()
                                .downcast_ref::<arrow::array::LargeStringArray>()
                                .unwrap();
                            serde_json::json!(arr.value(row_i))
                        }
                        _ => {
                            // Fallback: use arrow's display format
                            serde_json::Value::String(format!(
                                "{}",
                                arrow::util::display::ArrayFormatter::try_new(
                                    col.as_ref(),
                                    &Default::default()
                                )
                                .map(|f| f.value(row_i).to_string())
                                .unwrap_or_else(|_| "?".to_string())
                            ))
                        }
                    }
                };
                obj.insert(field.name().clone(), val);
            }
            let doc = serde_json::to_vec(&serde_json::Value::Object(obj))?;
            all_rows.push(VisibleRow {
                pk: row_idx.to_string(),
                doc,
            });
            row_idx += 1;
        }
    }

    Ok(all_rows)
}

// ==================== Index-Accelerated Scan ====================

/// Try to use a secondary index to answer a WHERE clause.
/// Returns Some(rows) if an index can satisfy the predicate, None otherwise.
fn try_index_scan(
    db: &Database,
    session: &SqlSession,
    table: &str,
    filter: &Expr,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Option<Vec<VisibleRow>> {
    // Extract column = 'literal' patterns from the filter
    let (col, val) = extract_eq_column_string_literal(filter)?;

    // Skip pk column (already handled by pk optimization)
    if col == "pk" || col == "id" {
        return None;
    }

    // Find an index on this column
    let prefix = index_meta_prefix_for_table(table);
    let index_metas = db.scan_prefix(&prefix, None, None, None).ok()?;

    let matching_index = index_metas.iter().find_map(|meta_row| {
        let meta: IndexMetadata = serde_json::from_slice(&meta_row.doc).ok()?;
        // Single-column index that matches our predicate column
        if meta.columns.len() == 1 && meta.columns[0] == col {
            Some(meta)
        } else {
            None
        }
    })?;

    // Do the index lookup: scan __idx/<table>/<index>/<value>/
    let idx_prefix = index_entry_prefix(table, &matching_index.name, &val);
    let idx_entries = db.scan_prefix(&idx_prefix, as_of, valid_at, None).ok()?;

    let mut result = Vec::new();
    for entry in &idx_entries {
        if entry.doc == INDEX_TOMBSTONE {
            continue;
        }
        // Extract PK from the index key: __idx/<table>/<index>/<value>/<pk>
        let key_str = String::from_utf8_lossy(&entry.user_key);
        let pk = key_str.rsplit('/').next()?;

        let row_k = row_key(table, pk);
        if let Some(doc) = read_live_key(db, session, &row_k, as_of, valid_at).ok()? {
            if !doc.is_empty() {
                result.push(VisibleRow {
                    pk: pk.to_string(),
                    doc,
                });
            }
        }
    }
    Some(result)
}

/// Try to use a composite index to answer an AND predicate.
fn try_composite_index_scan(
    db: &Database,
    session: &SqlSession,
    table: &str,
    filter: &Expr,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Option<Vec<VisibleRow>> {
    // Extract AND-connected equality predicates
    let mut eq_preds: Vec<(String, String)> = Vec::new();
    collect_eq_predicates(filter, &mut eq_preds);
    if eq_preds.len() < 2 {
        return None;
    }

    // Find a composite index that covers all predicate columns
    let prefix = index_meta_prefix_for_table(table);
    let index_metas = db.scan_prefix(&prefix, None, None, None).ok()?;

    let matching_index = index_metas.iter().find_map(|meta_row| {
        let meta: IndexMetadata = serde_json::from_slice(&meta_row.doc).ok()?;
        if meta.columns.len() >= 2 {
            // Check if all index columns have a predicate
            let all_covered = meta
                .columns
                .iter()
                .all(|c| eq_preds.iter().any(|(pc, _)| pc == c));
            if all_covered {
                return Some(meta);
            }
        }
        None
    })?;

    // Build the composite index value in column order
    let mut parts = Vec::new();
    for col in &matching_index.columns {
        let val = eq_preds.iter().find(|(c, _)| c == col)?.1.clone();
        parts.push(val);
    }
    let composite_val = parts.join("\x00");

    let idx_prefix = index_entry_prefix(table, &matching_index.name, &composite_val);
    let idx_entries = db.scan_prefix(&idx_prefix, as_of, valid_at, None).ok()?;

    let mut result = Vec::new();
    for entry in &idx_entries {
        if entry.doc == INDEX_TOMBSTONE {
            continue;
        }
        let key_str = String::from_utf8_lossy(&entry.user_key);
        let pk = key_str.rsplit('/').next()?;
        let row_k = row_key(table, pk);
        if let Some(doc) = read_live_key(db, session, &row_k, as_of, valid_at).ok()? {
            if !doc.is_empty() {
                result.push(VisibleRow {
                    pk: pk.to_string(),
                    doc,
                });
            }
        }
    }
    Some(result)
}

/// Try index scan for IN-list predicates: col IN ('a', 'b', 'c')
fn try_index_in_scan(
    db: &Database,
    session: &SqlSession,
    table: &str,
    filter: &Expr,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Option<Vec<VisibleRow>> {
    let (col, values) = extract_in_list_column_literals(filter)?;

    // Find index on this column
    let prefix = index_meta_prefix_for_table(table);
    let index_metas = db.scan_prefix(&prefix, None, None, None).ok()?;

    let matching_index = index_metas.iter().find_map(|meta_row| {
        let meta: IndexMetadata = serde_json::from_slice(&meta_row.doc).ok()?;
        if meta.columns.len() == 1 && meta.columns[0] == col {
            Some(meta)
        } else {
            None
        }
    })?;

    let mut result = Vec::new();
    let mut seen_pks = std::collections::HashSet::new();
    for val in &values {
        let idx_prefix = index_entry_prefix(table, &matching_index.name, val);
        if let Ok(idx_entries) = db.scan_prefix(&idx_prefix, as_of, valid_at, None) {
            for entry in &idx_entries {
                if entry.doc == INDEX_TOMBSTONE {
                    continue;
                }
                let key_str = String::from_utf8_lossy(&entry.user_key);
                if let Some(pk) = key_str.rsplit('/').next() {
                    if !seen_pks.insert(pk.to_string()) {
                        continue;
                    }
                    let row_k = row_key(table, pk);
                    if let Ok(Some(doc)) = read_live_key(db, session, &row_k, as_of, valid_at) {
                        if !doc.is_empty() {
                            result.push(VisibleRow {
                                pk: pk.to_string(),
                                doc,
                            });
                        }
                    }
                }
            }
        }
    }
    Some(result)
}

/// Extract col = 'string_literal' from a simple binary equality expression.
fn extract_eq_column_string_literal(expr: &Expr) -> Option<(String, String)> {
    match expr {
        Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (Expr::Column(c), Expr::StringLit(v)) | (Expr::StringLit(v), Expr::Column(c)) => {
                Some((c.clone(), v.clone()))
            }
            (Expr::Column(c), Expr::NumberLit(n)) | (Expr::NumberLit(n), Expr::Column(c)) => {
                // Use serde_json Number formatting to match extract_index_value() output
                let s = serde_json::Number::from_f64(*n)
                    .map(|num| num.to_string())
                    .unwrap_or_else(|| n.to_string());
                Some((c.clone(), s))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Collect all col = 'literal' predicates from an AND tree.
fn collect_eq_predicates(expr: &Expr, out: &mut Vec<(String, String)>) {
    match expr {
        Expr::BinOp {
            left,
            op: BinOperator::And,
            right,
        } => {
            collect_eq_predicates(left, out);
            collect_eq_predicates(right, out);
        }
        _ => {
            if let Some(pair) = extract_eq_column_string_literal(expr) {
                out.push(pair);
            }
        }
    }
}

/// Extract col IN ('a', 'b', ...) from an InList expression.
fn extract_in_list_column_literals(expr: &Expr) -> Option<(String, Vec<String>)> {
    match expr {
        Expr::InList {
            expr: inner,
            list,
            negated: false,
        } => {
            let col = match inner.as_ref() {
                Expr::Column(c) => c.clone(),
                _ => return None,
            };
            let values: Vec<String> = list
                .iter()
                .filter_map(|e| match e {
                    Expr::StringLit(s) => Some(s.clone()),
                    Expr::NumberLit(n) => Some(
                        serde_json::Number::from_f64(*n)
                            .map(|num| num.to_string())
                            .unwrap_or_else(|| n.to_string()),
                    ),
                    _ => None,
                })
                .collect();
            if values.len() == list.len() {
                Some((col, values))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to use a single-column index for a range predicate (>, <, >=, <=).
fn try_index_range_scan(
    db: &Database,
    session: &SqlSession,
    table: &str,
    filter: &Expr,
    as_of: Option<u64>,
    valid_at: Option<u64>,
) -> Option<Vec<VisibleRow>> {
    let (col, op, val) = extract_range_predicate(filter)?;

    // Skip pk (already handled by pk optimization)
    if col == "pk" || col == "id" {
        return None;
    }

    // Find a single-column index on this column
    let prefix = index_meta_prefix_for_table(table);
    let index_metas = db.scan_prefix(&prefix, None, None, None).ok()?;

    let matching_index = index_metas.iter().find_map(|meta_row| {
        let meta: IndexMetadata = serde_json::from_slice(&meta_row.doc).ok()?;
        if meta.columns.len() == 1 && meta.columns[0] == col {
            Some(meta)
        } else {
            None
        }
    })?;

    // Scan all entries in this index and filter by range comparison
    let scan_prefix = index_scan_prefix(table, &matching_index.name);
    let idx_entries = db.scan_prefix(&scan_prefix, as_of, valid_at, None).ok()?;

    // Parse the bound value as f64 for numeric comparison
    let bound_num = val.parse::<f64>().ok();

    let mut result = Vec::new();
    let mut seen_pks = std::collections::HashSet::new();
    for entry in &idx_entries {
        if entry.doc == INDEX_TOMBSTONE {
            continue;
        }
        let key_str = String::from_utf8_lossy(&entry.user_key);
        // Key format: __idx/<table>/<index>/<value>/<pk>
        // Strip the scan prefix to get <value>/<pk>
        let prefix_str = String::from_utf8_lossy(&scan_prefix);
        let suffix = key_str
            .strip_prefix(prefix_str.as_ref())
            .unwrap_or(&key_str);
        let mut parts_iter = suffix.rsplitn(2, '/');
        let pk = parts_iter.next()?;
        let idx_val = parts_iter.next().unwrap_or("");

        // Compare: try numeric first, fall back to string
        let matches = if let (Some(bound), Ok(entry_num)) = (bound_num, idx_val.parse::<f64>()) {
            match op {
                BinOperator::Gt => entry_num > bound,
                BinOperator::Lt => entry_num < bound,
                BinOperator::GtEq => entry_num >= bound,
                BinOperator::LtEq => entry_num <= bound,
                _ => false,
            }
        } else {
            match op {
                BinOperator::Gt => idx_val > val.as_str(),
                BinOperator::Lt => idx_val < val.as_str(),
                BinOperator::GtEq => idx_val >= val.as_str(),
                BinOperator::LtEq => idx_val <= val.as_str(),
                _ => false,
            }
        };

        if matches && seen_pks.insert(pk.to_string()) {
            let row_k = row_key(table, pk);
            if let Ok(Some(doc)) = read_live_key(db, session, &row_k, as_of, valid_at) {
                if !doc.is_empty() {
                    result.push(VisibleRow {
                        pk: pk.to_string(),
                        doc,
                    });
                }
            }
        }
    }
    Some(result)
}

/// Extract col OP literal from a range predicate (>, <, >=, <=).
fn extract_range_predicate(expr: &Expr) -> Option<(String, BinOperator, String)> {
    match expr {
        Expr::BinOp { left, op, right }
            if matches!(
                op,
                BinOperator::Gt | BinOperator::Lt | BinOperator::GtEq | BinOperator::LtEq
            ) =>
        {
            match (left.as_ref(), right.as_ref()) {
                (Expr::Column(c), Expr::StringLit(v)) => Some((c.clone(), *op, v.clone())),
                (Expr::Column(c), Expr::NumberLit(n)) => {
                    let s = serde_json::Number::from_f64(*n)
                        .map(|num| num.to_string())
                        .unwrap_or_else(|| n.to_string());
                    Some((c.clone(), *op, s))
                }
                // Reversed: literal OP column → flip the operator
                (Expr::StringLit(v), Expr::Column(c)) => {
                    Some((c.clone(), flip_range_op(op), v.clone()))
                }
                (Expr::NumberLit(n), Expr::Column(c)) => {
                    let s = serde_json::Number::from_f64(*n)
                        .map(|num| num.to_string())
                        .unwrap_or_else(|| n.to_string());
                    Some((c.clone(), flip_range_op(op), s))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn flip_range_op(op: &BinOperator) -> BinOperator {
    match op {
        BinOperator::Gt => BinOperator::Lt,
        BinOperator::Lt => BinOperator::Gt,
        BinOperator::GtEq => BinOperator::LtEq,
        BinOperator::LtEq => BinOperator::GtEq,
        other => *other,
    }
}

// ==================== Secondary Index Maintenance ====================

/// Extract a composite index value from a JSON document.
/// For single-column indexes, returns the column value as a string.
/// For composite indexes, returns values joined with `\x00`.
fn extract_index_value(doc: &[u8], columns: &[String]) -> Result<String> {
    let parsed: serde_json::Value = serde_json::from_slice(doc).unwrap_or(serde_json::Value::Null);
    let mut parts = Vec::with_capacity(columns.len());
    for col in columns {
        let val = parsed.get(col);
        let s = match val {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Number(n)) => n.to_string(),
            Some(serde_json::Value::Bool(b)) => b.to_string(),
            Some(serde_json::Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        };
        parts.push(s);
    }
    Ok(parts.join("\x00"))
}

/// Update all secondary indexes for a table after an INSERT or UPDATE.
fn update_secondary_indexes(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    pk: &str,
    doc: &[u8],
) -> Result<()> {
    let prefix = index_meta_prefix_for_table(table);
    let index_metas = db.scan_prefix(&prefix, None, None, None)?;

    for meta_row in &index_metas {
        if let Ok(meta) = serde_json::from_slice::<IndexMetadata>(&meta_row.doc) {
            let index_val = extract_index_value(doc, &meta.columns)?;

            if meta.unique {
                let idx_prefix = index_entry_prefix(table, &meta.name, &index_val);
                let existing = db.scan_prefix(&idx_prefix, None, None, Some(10))?;
                // Filter out tombstoned entries and entries for our own pk
                let other = existing
                    .iter()
                    .any(|r| r.doc != INDEX_TOMBSTONE && !r.user_key.ends_with(pk.as_bytes()));
                if other {
                    return Err(TensorError::SqlExec(format!(
                        "UNIQUE constraint violated: duplicate value in index {}",
                        meta.name
                    )));
                }
            }

            let idx_key = index_entry_key(table, &meta.name, &index_val, pk);
            write_put(db, session, idx_key, Vec::new(), 0, u64::MAX, Some(1))?;
        }
    }
    Ok(())
}

/// Tombstone marker for deleted index entries.
const INDEX_TOMBSTONE: &[u8] = b"\x00DELETED";

/// Remove all secondary index entries for a row being deleted.
fn remove_secondary_indexes(
    db: &Database,
    session: &mut SqlSession,
    table: &str,
    pk: &str,
    old_doc: &[u8],
) -> Result<()> {
    let prefix = index_meta_prefix_for_table(table);
    let index_metas = db.scan_prefix(&prefix, None, None, None)?;

    for meta_row in &index_metas {
        if let Ok(meta) = serde_json::from_slice::<IndexMetadata>(&meta_row.doc) {
            let index_val = extract_index_value(old_doc, &meta.columns)?;
            let idx_key = index_entry_key(table, &meta.name, &index_val, pk);
            // Write tombstone marker (distinct from live empty-doc index entries)
            write_put(
                db,
                session,
                idx_key,
                INDEX_TOMBSTONE.to_vec(),
                0,
                u64::MAX,
                Some(1),
            )?;
        }
    }
    Ok(())
}

// ==================== Backup & Restore ====================

fn execute_backup(db: &Database, dest: &str, since_epoch: Option<u64>) -> Result<SqlResult> {
    use std::fs;
    use std::path::Path;

    let dest_path = Path::new(dest);
    fs::create_dir_all(dest_path).map_err(|e| {
        TensorError::SqlExec(format!("failed to create backup directory {dest}: {e}"))
    })?;

    // 1. Flush all shard memtables
    db.sync();

    let current_epoch = db.current_epoch();

    if let Some(from_epoch) = since_epoch {
        // Incremental backup: export only data written after the given epoch.
        // Uses the TXN_COMMIT marker to find the commit_ts boundary.
        let marker_key = format!("__txn_commit/{from_epoch}");
        let since_ts = if let Some(v) = db.get(marker_key.as_bytes(), None, None)? {
            if v.len() >= 8 {
                u64::from_le_bytes(v[..8].try_into().unwrap())
            } else {
                from_epoch
            }
        } else {
            from_epoch
        };

        // Scan all data and export rows with commit_ts > since_ts
        let mut record_count = 0u64;
        let mut incremental_data: Vec<serde_json::Value> = Vec::new();

        // Scan all prefixes (tables, indexes, metadata, txn markers)
        for prefix in &[
            b"__table/".as_slice(),
            b"__meta/".as_slice(),
            b"__idx/".as_slice(),
            b"__txn_commit/".as_slice(),
        ] {
            for row in db.scan_prefix(prefix, None, None, None)? {
                if row.commit_ts > since_ts {
                    incremental_data.push(serde_json::json!({
                        "key": crate::ai::hex_encode(&row.user_key),
                        "value": crate::ai::hex_encode(&row.doc),
                        "commit_ts": row.commit_ts,
                    }));
                    record_count += 1;
                }
            }
        }

        // Write incremental backup as JSON
        let metadata = serde_json::json!({
            "version": "1.0",
            "type": "incremental",
            "since_epoch": from_epoch,
            "since_commit_ts": since_ts,
            "current_epoch": current_epoch,
            "record_count": record_count,
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "shard_count": db.config().shard_count,
        });

        let meta_path = dest_path.join("backup_metadata.json");
        fs::write(&meta_path, serde_json::to_vec_pretty(&metadata)?)
            .map_err(|e| TensorError::SqlExec(format!("failed to write backup metadata: {e}")))?;

        let data_path = dest_path.join("incremental_data.json");
        fs::write(
            &data_path,
            serde_json::to_vec_pretty(&serde_json::json!(incremental_data))?,
        )
        .map_err(|e| TensorError::SqlExec(format!("failed to write incremental data: {e}")))?;

        return Ok(SqlResult::Affected {
            rows: record_count,
            commit_ts: None,
            message: format!("incremental backup complete: {record_count} records since epoch {from_epoch} to {dest}"),
        });
    }

    // Full backup
    // 2. Copy manifest
    let root = db.root();
    let manifest_src = root.join("MANIFEST.json");
    let manifest_dst = dest_path.join("MANIFEST.json");
    if manifest_src.exists() {
        fs::copy(&manifest_src, &manifest_dst)
            .map_err(|e| TensorError::SqlExec(format!("failed to copy manifest: {e}")))?;
    }

    // 3. Copy shard directories (SSTables + WAL)
    let mut file_count = 0u64;
    for shard_id in 0..db.config().shard_count {
        let shard_dir_name = format!("shard-{shard_id}");
        let src_shard = root.join(&shard_dir_name);
        let dst_shard = dest_path.join(&shard_dir_name);

        if !src_shard.exists() {
            continue;
        }

        fs::create_dir_all(&dst_shard)
            .map_err(|e| TensorError::SqlExec(format!("failed to create shard backup dir: {e}")))?;

        for entry in fs::read_dir(&src_shard)
            .map_err(|e| TensorError::SqlExec(format!("failed to read shard dir: {e}")))?
        {
            let entry = entry.map_err(|e| TensorError::SqlExec(format!("dir entry error: {e}")))?;
            let src_file = entry.path();
            if src_file.is_file() {
                let file_name = entry.file_name();
                let dst_file = dst_shard.join(&file_name);
                fs::copy(&src_file, &dst_file).map_err(|e| {
                    TensorError::SqlExec(format!("failed to copy {}: {e}", src_file.display()))
                })?;
                file_count += 1;
            }
        }
    }

    // 4. Write backup metadata
    let metadata = serde_json::json!({
        "version": "1.0",
        "type": "full",
        "current_epoch": current_epoch,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "shard_count": db.config().shard_count,
        "file_count": file_count,
    });
    let meta_path = dest_path.join("backup_metadata.json");
    fs::write(&meta_path, serde_json::to_vec_pretty(&metadata)?)
        .map_err(|e| TensorError::SqlExec(format!("failed to write backup metadata: {e}")))?;

    Ok(SqlResult::Affected {
        rows: file_count,
        commit_ts: None,
        message: format!("backup complete: {file_count} files copied to {dest}"),
    })
}

fn execute_restore(db: &Database, src: &str) -> Result<SqlResult> {
    use std::path::Path;

    let src_path = Path::new(src);
    if !src_path.exists() {
        return Err(TensorError::SqlExec(format!(
            "backup directory does not exist: {src}"
        )));
    }

    let meta_path = src_path.join("backup_metadata.json");
    if !meta_path.exists() {
        return Err(TensorError::SqlExec(format!(
            "not a valid backup: missing backup_metadata.json in {src}"
        )));
    }

    // Verify backup metadata
    let meta_bytes = std::fs::read(&meta_path)
        .map_err(|e| TensorError::SqlExec(format!("failed to read backup metadata: {e}")))?;
    let _metadata: serde_json::Value = serde_json::from_slice(&meta_bytes)?;

    // Copy files from backup to database root
    let root = db.root();
    let mut file_count = 0u64;

    // Copy manifest
    let manifest_src = src_path.join("MANIFEST.json");
    if manifest_src.exists() {
        std::fs::copy(&manifest_src, root.join("MANIFEST.json"))
            .map_err(|e| TensorError::SqlExec(format!("failed to restore manifest: {e}")))?;
        file_count += 1;
    }

    // Copy shard directories
    for shard_id in 0..db.config().shard_count {
        let shard_dir_name = format!("shard-{shard_id}");
        let src_shard = src_path.join(&shard_dir_name);
        let dst_shard = root.join(&shard_dir_name);

        if !src_shard.exists() {
            continue;
        }

        std::fs::create_dir_all(&dst_shard)
            .map_err(|e| TensorError::SqlExec(format!("failed to create shard dir: {e}")))?;

        for entry in std::fs::read_dir(&src_shard)
            .map_err(|e| TensorError::SqlExec(format!("failed to read backup shard dir: {e}")))?
        {
            let entry = entry.map_err(|e| TensorError::SqlExec(format!("dir entry error: {e}")))?;
            let src_file = entry.path();
            if src_file.is_file() {
                let dst_file = dst_shard.join(entry.file_name());
                std::fs::copy(&src_file, &dst_file).map_err(|e| {
                    TensorError::SqlExec(format!("failed to restore {}: {e}", src_file.display()))
                })?;
                file_count += 1;
            }
        }
    }

    Ok(SqlResult::Affected {
        rows: file_count,
        commit_ts: None,
        message: format!(
            "restore complete: {file_count} files restored from {src} (restart required)"
        ),
    })
}
