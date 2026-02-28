use crate::error::{Result, TensorError};

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),
    FieldAccess {
        column: String,
        path: Vec<String>,
    },
    StringLit(String),
    NumberLit(f64),
    BoolLit(bool),
    Null,
    BinOp {
        left: Box<Expr>,
        op: BinOperator,
        right: Box<Expr>,
    },
    Not(Box<Expr>),
    Function {
        name: String,
        args: Vec<Expr>,
    },
    Star,
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    WindowFunction {
        name: String,
        args: Vec<Expr>,
        partition_by: Vec<Expr>,
        order_by: Vec<(Expr, OrderDirection)>,
    },
    Case {
        operand: Option<Box<Expr>>,
        when_clauses: Vec<(Expr, Expr)>,
        else_clause: Option<Box<Expr>>,
    },
    Cast {
        expr: Box<Expr>,
        target_type: String,
    },
}

pub fn is_window_function(name: &str) -> bool {
    matches!(
        name.to_uppercase().as_str(),
        "ROW_NUMBER" | "RANK" | "DENSE_RANK" | "LEAD" | "LAG"
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOperator {
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    Like,
    ILike,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    Expr { expr: Expr, alias: Option<String> },
    AllColumns,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Cross,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JoinSpec {
    pub join_type: JoinType,
    pub right_table: String,
    pub right_alias: Option<String>,
    pub on_clause: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CteClause {
    pub name: String,
    pub query: Box<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableRef {
    Named(String),
    Subquery {
        query: Box<Statement>,
        alias: String,
    },
    /// Table function: `read_csv('path')`, `read_json('path')`, etc.
    TableFunction {
        name: String,
        args: Vec<Expr>,
        alias: Option<String>,
    },
}

/// SQL:2011 temporal query clause.
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalClause {
    /// `FOR SYSTEM_TIME AS OF <ts>` — snapshot at a specific system time
    SystemTimeAsOf(u64),
    /// `FOR SYSTEM_TIME FROM <t1> TO <t2>` — half-open range [t1, t2)
    SystemTimeFromTo(u64, u64),
    /// `FOR SYSTEM_TIME BETWEEN <t1> AND <t2>` — inclusive range [t1, t2]
    SystemTimeBetween(u64, u64),
    /// `FOR SYSTEM_TIME ALL` — return all historical versions
    SystemTimeAll,
    /// `FOR APPLICATION_TIME AS OF <ts>` — business time point query
    ApplicationTimeAsOf(u64),
    /// `FOR APPLICATION_TIME FROM <t1> TO <t2>` — half-open range [t1, t2)
    ApplicationTimeFromTo(u64, u64),
    /// `FOR APPLICATION_TIME BETWEEN <t1> AND <t2>` — inclusive range [t1, t2]
    ApplicationTimeBetween(u64, u64),
}

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum Statement {
    Begin,
    Commit,
    Rollback,
    Savepoint {
        name: String,
    },
    RollbackTo {
        name: String,
    },
    ReleaseSavepoint {
        name: String,
    },
    CreateTable {
        table: String,
        columns: Vec<ColumnDef>,
    },
    CreateView {
        view: String,
        source: String,
        pk: String,
        as_of: Option<u64>,
        valid_at: Option<u64>,
    },
    CreateIndex {
        index: String,
        table: String,
        columns: Vec<String>,
        unique: bool,
    },
    /// `CREATE FULLTEXT INDEX <name> ON <table> (<columns>)`
    CreateFulltextIndex {
        index: String,
        table: String,
        columns: Vec<String>,
    },
    /// `DROP FULLTEXT INDEX <name> ON <table>`
    DropFulltextIndex {
        index: String,
        table: String,
    },
    /// `CREATE TIMESERIES TABLE <name> (ts TIMESTAMP, value REAL, ...)
    ///  WITH (bucket_size = '<interval>')`
    CreateTimeseriesTable {
        table: String,
        columns: Vec<ColumnDef>,
        bucket_interval: String,
    },
    AlterTableAddColumn {
        table: String,
        column: String,
        column_type: String,
    },
    Insert {
        table: String,
        pk: String,
        doc: String,
    },
    InsertTyped {
        table: String,
        columns: Vec<String>,
        values: Vec<Expr>,
    },
    Update {
        table: String,
        set_doc: Expr,
        set_assignments: Vec<(String, Expr)>,
        filter: Option<Expr>,
        as_of: Option<u64>,
        valid_at: Option<u64>,
    },
    Delete {
        table: String,
        filter: Option<Expr>,
        as_of: Option<u64>,
        valid_at: Option<u64>,
    },
    Select {
        ctes: Vec<CteClause>,
        from: TableRef,
        items: Vec<SelectItem>,
        joins: Vec<JoinSpec>,
        filter: Option<Expr>,
        as_of: Option<u64>,
        valid_at: Option<u64>,
        /// Point-in-time recovery: query state as of a specific epoch
        as_of_epoch: Option<u64>,
        /// SQL:2011 temporal clauses (FOR SYSTEM_TIME ..., FOR APPLICATION_TIME ...)
        temporal: Vec<TemporalClause>,
        group_by: Option<Vec<Expr>>,
        having: Option<Expr>,
        order_by: Option<Vec<(Expr, OrderDirection)>>,
        limit: Option<u64>,
    },
    CopyTo {
        table: String,
        path: String,
        format: CopyFormat,
    },
    CopyFrom {
        table: String,
        path: String,
        format: CopyFormat,
    },
    ShowTables,
    Describe {
        table: String,
    },
    DropTable {
        table: String,
    },
    DropView {
        view: String,
    },
    DropIndex {
        index: String,
        table: String,
    },
    /// Set operations: UNION, UNION ALL, INTERSECT, EXCEPT
    SetOp {
        op: SetOpType,
        left: Box<Statement>,
        right: Box<Statement>,
    },
    /// INSERT ... RETURNING
    InsertReturning {
        table: String,
        columns: Vec<String>,
        values: Vec<Expr>,
        returning: Vec<SelectItem>,
    },
    /// CREATE TABLE AS SELECT
    CreateTableAs {
        table: String,
        query: Box<Statement>,
    },
    Explain(Box<Statement>),
    ExplainAnalyze(Box<Statement>),
    /// `EXPLAIN AI '<key>'` — returns AI insights, provenance, and risk score for a key
    ExplainAi {
        key: String,
    },
    Analyze {
        table: String,
    },
    /// `ASK '<natural language question>'` — translates NL to SQL via embedded LLM
    Ask {
        question: String,
    },
    /// `BACKUP DATABASE TO '<path>'` or `BACKUP DATABASE TO '<path>' SINCE EPOCH <n>`
    Backup {
        dest: String,
        since_epoch: Option<u64>,
    },
    /// `RESTORE DATABASE FROM '<path>'`
    Restore {
        src: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub type_name: SqlType,
    pub primary_key: bool,
    pub not_null: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlType {
    Integer,
    Real,
    Text,
    Boolean,
    Blob,
    Json,
    Decimal { precision: u8, scale: u8 },
}

impl SqlType {
    pub fn from_str_name(s: &str) -> Option<SqlType> {
        match s.to_uppercase().as_str() {
            "INTEGER" | "INT" | "BIGINT" => Some(SqlType::Integer),
            "REAL" | "FLOAT" | "DOUBLE" => Some(SqlType::Real),
            "TEXT" | "VARCHAR" | "STRING" => Some(SqlType::Text),
            "BOOLEAN" | "BOOL" => Some(SqlType::Boolean),
            "BLOB" | "BYTES" => Some(SqlType::Blob),
            "JSON" => Some(SqlType::Json),
            "DECIMAL" | "NUMERIC" => Some(SqlType::Decimal {
                precision: 38,
                scale: 10,
            }),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SqlType::Integer => "INTEGER",
            SqlType::Real => "REAL",
            SqlType::Text => "TEXT",
            SqlType::Boolean => "BOOLEAN",
            SqlType::Blob => "BLOB",
            SqlType::Json => "JSON",
            SqlType::Decimal { .. } => "DECIMAL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpType {
    Union,
    UnionAll,
    Intersect,
    Except,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyFormat {
    Csv,
    Json,
    Ndjson,
    Parquet,
}

// Legacy re-exports for backward compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectProjection {
    Doc,
    PkDoc,
    PkCount,
    CountStar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    Number(String), // Store as string to parse as both u64 and f64
    StringLit(String),
    Symbol(char),
    // Multi-char operators
    NotEq, // !=
    LtEq,  // <=
    GtEq,  // >=
}

pub fn parse_sql(input: &str) -> Result<Statement> {
    let mut toks = tokenize(input)?;
    if matches!(toks.last(), Some(Token::Symbol(';'))) {
        toks.pop();
    }
    let mut p = Parser { toks, i: 0 };
    let stmt = p.parse_statement()?;
    // Check for set operations after the first statement
    let stmt = p.try_parse_set_op(stmt)?;
    if p.i != p.toks.len() {
        return Err(TensorError::SqlParse(
            "unexpected trailing tokens".to_string(),
        ));
    }
    Ok(stmt)
}

pub fn split_sql_statements(input: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut start = 0usize;

    for (idx, ch) in input.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '\'' {
                in_string = false;
            }
            continue;
        }

        if ch == '\'' {
            in_string = true;
            continue;
        }

        if ch == ';' {
            let seg = input[start..idx].trim();
            if !seg.is_empty() {
                out.push(seg.to_string());
            }
            start = idx + ch.len_utf8();
        }
    }

    if in_string {
        return Err(TensorError::SqlParse(
            "unterminated string literal".to_string(),
        ));
    }

    let tail = input[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }

    if out.is_empty() {
        return Err(TensorError::SqlParse("no SQL statements found".to_string()));
    }

    Ok(out)
}

struct Parser {
    toks: Vec<Token>,
    i: usize,
}

impl Parser {
    fn parse_statement(&mut self) -> Result<Statement> {
        if self.peek_kw("EXPLAIN") {
            self.expect_kw("EXPLAIN")?;
            if self.peek_kw("ANALYZE") {
                self.expect_kw("ANALYZE")?;
                let inner = self.parse_statement()?;
                return Ok(Statement::ExplainAnalyze(Box::new(inner)));
            }
            if self.peek_kw("AI") {
                self.expect_kw("AI")?;
                let key = self.expect_string()?;
                return Ok(Statement::ExplainAi { key });
            }
            let inner = self.parse_statement()?;
            return Ok(Statement::Explain(Box::new(inner)));
        }
        if self.peek_kw("ANALYZE") {
            self.expect_kw("ANALYZE")?;
            let table = self.expect_ident()?;
            return Ok(Statement::Analyze { table });
        }
        if self.peek_kw("BEGIN") {
            self.expect_kw("BEGIN")?;
            return Ok(Statement::Begin);
        }
        if self.peek_kw("COMMIT") {
            self.expect_kw("COMMIT")?;
            return Ok(Statement::Commit);
        }
        if self.peek_kw("ROLLBACK") {
            self.expect_kw("ROLLBACK")?;
            if self.peek_kw("TO") {
                self.expect_kw("TO")?;
                // optional SAVEPOINT keyword
                if self.peek_kw("SAVEPOINT") {
                    self.expect_kw("SAVEPOINT")?;
                }
                let name = self.expect_ident()?;
                return Ok(Statement::RollbackTo { name });
            }
            return Ok(Statement::Rollback);
        }
        if self.peek_kw("SAVEPOINT") {
            self.expect_kw("SAVEPOINT")?;
            let name = self.expect_ident()?;
            return Ok(Statement::Savepoint { name });
        }
        if self.peek_kw("RELEASE") {
            self.expect_kw("RELEASE")?;
            // optional SAVEPOINT keyword
            if self.peek_kw("SAVEPOINT") {
                self.expect_kw("SAVEPOINT")?;
            }
            let name = self.expect_ident()?;
            return Ok(Statement::ReleaseSavepoint { name });
        }
        if self.peek_kw("CREATE") {
            return self.parse_create();
        }
        if self.peek_kw("ALTER") {
            return self.parse_alter();
        }
        if self.peek_kw("INSERT") {
            return self.parse_insert();
        }
        if self.peek_kw("UPDATE") {
            return self.parse_update();
        }
        if self.peek_kw("DELETE") {
            return self.parse_delete();
        }
        if self.peek_kw("SELECT") || self.peek_kw("WITH") {
            return self.parse_select_or_cte();
        }
        if self.peek_kw("SHOW") {
            return self.parse_show();
        }
        if self.peek_kw("DESCRIBE") {
            return self.parse_describe();
        }
        if self.peek_kw("DROP") {
            return self.parse_drop();
        }
        if self.peek_kw("COPY") {
            return self.parse_copy();
        }
        if self.peek_kw("ASK") {
            self.expect_kw("ASK")?;
            let question = self.expect_string()?;
            return Ok(Statement::Ask { question });
        }
        if self.peek_kw("BACKUP") {
            self.expect_kw("BACKUP")?;
            if self.peek_kw("DATABASE") {
                self.expect_kw("DATABASE")?;
            }
            self.expect_kw("TO")?;
            let dest = self.expect_string()?;
            let since_epoch = if self.peek_kw("SINCE") {
                self.expect_kw("SINCE")?;
                self.expect_kw("EPOCH")?;
                Some(self.expect_number_u64()?)
            } else {
                None
            };
            return Ok(Statement::Backup { dest, since_epoch });
        }
        if self.peek_kw("RESTORE") {
            self.expect_kw("RESTORE")?;
            if self.peek_kw("DATABASE") {
                self.expect_kw("DATABASE")?;
            }
            self.expect_kw("FROM")?;
            let src = self.expect_string()?;
            return Ok(Statement::Restore { src });
        }
        Err(TensorError::SqlParse(
            "unsupported SQL statement".to_string(),
        ))
    }

    fn parse_create(&mut self) -> Result<Statement> {
        self.expect_kw("CREATE")?;
        if self.peek_kw("TIMESERIES") {
            return self.parse_create_timeseries_table();
        }
        if self.peek_kw("TABLE") {
            return self.parse_create_table_after_create();
        }
        if self.peek_kw("VIEW") {
            return self.parse_create_view_after_create();
        }
        if self.peek_kw("FULLTEXT") {
            return self.parse_create_fulltext_index();
        }
        if self.peek_kw("UNIQUE") {
            self.expect_kw("UNIQUE")?;
            return self.parse_create_index_after_create(true);
        }
        if self.peek_kw("INDEX") {
            return self.parse_create_index_after_create(false);
        }
        Err(TensorError::SqlParse(
            "expected TABLE, TIMESERIES TABLE, VIEW, FULLTEXT INDEX, UNIQUE INDEX, or INDEX after CREATE"
                .to_string(),
        ))
    }

    fn parse_create_table_after_create(&mut self) -> Result<Statement> {
        self.expect_kw("TABLE")?;
        let table = self.expect_ident()?;

        // CREATE TABLE ... AS SELECT ...
        if self.peek_kw("AS") {
            self.expect_kw("AS")?;
            let query = self.parse_select_or_cte()?;
            return Ok(Statement::CreateTableAs {
                table,
                query: Box::new(query),
            });
        }

        self.expect_symbol('(')?;

        let mut columns = Vec::new();

        // Parse column definitions
        loop {
            let col_name = self.expect_ident()?;
            let type_name_str = self.expect_ident()?;
            let mut type_name = SqlType::from_str_name(&type_name_str).ok_or_else(|| {
                TensorError::SqlParse(format!("unknown column type: {type_name_str}"))
            })?;

            // Handle DECIMAL(precision, scale) / NUMERIC(p,s) / VARCHAR(n) syntax
            if matches!(type_name, SqlType::Decimal { .. }) && self.peek_symbol('(') {
                self.expect_symbol('(')?;
                let precision = self.expect_number_u64()? as u8;
                let scale = if self.peek_symbol(',') {
                    self.expect_symbol(',')?;
                    self.expect_number_u64()? as u8
                } else {
                    0
                };
                self.expect_symbol(')')?;
                type_name = SqlType::Decimal { precision, scale };
            } else if self.peek_symbol('(') {
                // Skip size specifiers for VARCHAR(n) etc.
                self.expect_symbol('(')?;
                let _ = self.expect_number_u64()?;
                self.expect_symbol(')')?;
            }

            let mut primary_key = false;
            let mut not_null = false;

            // Parse column constraints
            loop {
                if self.peek_kw("PRIMARY") {
                    self.expect_kw("PRIMARY")?;
                    self.expect_kw("KEY")?;
                    primary_key = true;
                } else if self.peek_kw("NOT") {
                    self.expect_kw("NOT")?;
                    self.expect_kw("NULL")?;
                    not_null = true;
                } else {
                    break;
                }
            }

            columns.push(ColumnDef {
                name: col_name,
                type_name,
                primary_key,
                not_null,
            });

            if self.peek_symbol(',') {
                self.expect_symbol(',')?;
            } else {
                break;
            }
        }
        self.expect_symbol(')')?;

        Ok(Statement::CreateTable { table, columns })
    }

    fn parse_create_view_after_create(&mut self) -> Result<Statement> {
        self.expect_kw("VIEW")?;
        let view = self.expect_ident()?;
        self.expect_kw("AS")?;
        let select = self.parse_select_or_cte()?;
        let Statement::Select {
            from,
            items,
            joins,
            filter,
            as_of,
            valid_at,
            group_by,
            having: _,
            order_by,
            limit,
            ..
        } = select
        else {
            return Err(TensorError::SqlParse(
                "expected SELECT after CREATE VIEW ... AS".to_string(),
            ));
        };

        let table = match &from {
            TableRef::Named(t) => t.clone(),
            _ => {
                return Err(TensorError::SqlParse(
                    "CREATE VIEW requires simple table reference".to_string(),
                ))
            }
        };

        // Validate: must be SELECT doc FROM table WHERE pk='...'
        let is_doc_projection = items.len() == 1
            && matches!(&items[0], SelectItem::Expr { expr: Expr::Column(c), alias: None } if c.eq_ignore_ascii_case("doc"));
        if !is_doc_projection {
            return Err(TensorError::SqlParse(
                "CREATE VIEW requires SELECT doc projection".to_string(),
            ));
        }
        if !joins.is_empty() {
            return Err(TensorError::SqlParse(
                "CREATE VIEW does not support JOIN".to_string(),
            ));
        }

        let pk = extract_pk_eq_literal(filter.as_ref()).ok_or_else(|| {
            TensorError::SqlParse("CREATE VIEW requires WHERE pk='...'".to_string())
        })?;

        if group_by.is_some() || order_by.is_some() || limit.is_some() {
            return Err(TensorError::SqlParse(
                "CREATE VIEW does not support GROUP BY, ORDER BY, or LIMIT".to_string(),
            ));
        }

        Ok(Statement::CreateView {
            view,
            source: table,
            pk,
            as_of,
            valid_at,
        })
    }

    fn parse_create_fulltext_index(&mut self) -> Result<Statement> {
        self.expect_kw("FULLTEXT")?;
        self.expect_kw("INDEX")?;
        let index = self.expect_ident()?;
        self.expect_kw("ON")?;
        let table = self.expect_ident()?;
        self.expect_symbol('(')?;
        let mut columns = Vec::new();
        columns.push(self.expect_ident()?);
        while self.peek_symbol(',') {
            self.expect_symbol(',')?;
            columns.push(self.expect_ident()?);
        }
        self.expect_symbol(')')?;
        Ok(Statement::CreateFulltextIndex {
            index,
            table,
            columns,
        })
    }

    /// Parse `CREATE TIMESERIES TABLE <name> (<columns>) WITH (bucket_size = '<interval>')`
    fn parse_create_timeseries_table(&mut self) -> Result<Statement> {
        self.expect_kw("TIMESERIES")?;
        self.expect_kw("TABLE")?;
        let table = self.expect_ident()?;
        self.expect_symbol('(')?;

        let mut columns = Vec::new();
        loop {
            let col_name = self.expect_ident()?;
            let type_name_str = self.expect_ident()?;
            let type_name = SqlType::from_str_name(&type_name_str).ok_or_else(|| {
                TensorError::SqlParse(format!("unknown column type: {type_name_str}"))
            })?;

            let mut primary_key = false;
            let mut not_null = false;
            loop {
                if self.peek_kw("PRIMARY") {
                    self.expect_kw("PRIMARY")?;
                    self.expect_kw("KEY")?;
                    primary_key = true;
                } else if self.peek_kw("NOT") {
                    self.expect_kw("NOT")?;
                    self.expect_kw("NULL")?;
                    not_null = true;
                } else {
                    break;
                }
            }

            columns.push(ColumnDef {
                name: col_name,
                type_name,
                primary_key,
                not_null,
            });

            if self.peek_symbol(',') {
                self.expect_symbol(',')?;
            } else {
                break;
            }
        }
        self.expect_symbol(')')?;

        // Parse optional WITH (bucket_size = '<interval>')
        let mut bucket_interval = "1h".to_string();
        if self.peek_kw("WITH") {
            self.expect_kw("WITH")?;
            self.expect_symbol('(')?;
            loop {
                let key = self.expect_ident()?;
                self.expect_symbol('=')?;
                let val = self.expect_string()?;
                if key.eq_ignore_ascii_case("bucket_size") {
                    bucket_interval = val;
                }
                if !self.peek_symbol(',') {
                    break;
                }
                self.expect_symbol(',')?;
            }
            self.expect_symbol(')')?;
        }

        Ok(Statement::CreateTimeseriesTable {
            table,
            columns,
            bucket_interval,
        })
    }

    fn parse_create_index_after_create(&mut self, unique: bool) -> Result<Statement> {
        self.expect_kw("INDEX")?;
        let index = self.expect_ident()?;
        self.expect_kw("ON")?;
        let table = self.expect_ident()?;
        self.expect_symbol('(')?;
        let mut columns = vec![self.expect_ident()?];
        while self.peek_symbol(',') {
            self.expect_symbol(',')?;
            columns.push(self.expect_ident()?);
        }
        self.expect_symbol(')')?;
        Ok(Statement::CreateIndex {
            index,
            table,
            columns,
            unique,
        })
    }

    fn parse_alter(&mut self) -> Result<Statement> {
        self.expect_kw("ALTER")?;
        self.expect_kw("TABLE")?;
        let table = self.expect_ident()?;
        self.expect_kw("ADD")?;
        self.expect_kw("COLUMN")?;
        let column = self.expect_ident()?;
        let column_type = self.expect_ident()?;
        Ok(Statement::AlterTableAddColumn {
            table,
            column,
            column_type,
        })
    }

    fn parse_insert(&mut self) -> Result<Statement> {
        self.expect_kw("INSERT")?;
        self.expect_kw("INTO")?;
        let table = self.expect_ident()?;
        self.expect_symbol('(')?;

        // Collect column names
        let mut col_names = Vec::new();
        col_names.push(self.expect_ident()?);
        while self.peek_symbol(',') {
            self.expect_symbol(',')?;
            col_names.push(self.expect_ident()?);
        }
        self.expect_symbol(')')?;

        self.expect_kw("VALUES")?;
        self.expect_symbol('(')?;

        // Check if this is the legacy (pk, doc) form
        if col_names.len() == 2
            && col_names[0].eq_ignore_ascii_case("pk")
            && col_names[1].eq_ignore_ascii_case("doc")
        {
            let pk = self.expect_string()?;
            self.expect_symbol(',')?;
            let doc = self.expect_string()?;
            self.expect_symbol(')')?;
            return Ok(Statement::Insert { table, pk, doc });
        }

        // Typed insert: collect value expressions
        let mut values = Vec::new();
        values.push(self.parse_expr()?);
        while self.peek_symbol(',') {
            self.expect_symbol(',')?;
            values.push(self.parse_expr()?);
        }
        self.expect_symbol(')')?;

        if col_names.len() != values.len() {
            return Err(TensorError::SqlParse(
                "column count does not match value count".to_string(),
            ));
        }

        // Check for RETURNING clause
        if self.peek_kw("RETURNING") {
            self.expect_kw("RETURNING")?;
            let mut returning = Vec::new();
            returning.push(self.parse_select_item()?);
            while self.peek_symbol(',') {
                self.expect_symbol(',')?;
                returning.push(self.parse_select_item()?);
            }
            return Ok(Statement::InsertReturning {
                table,
                columns: col_names,
                values,
                returning,
            });
        }

        Ok(Statement::InsertTyped {
            table,
            columns: col_names,
            values,
        })
    }

    fn parse_update(&mut self) -> Result<Statement> {
        self.expect_kw("UPDATE")?;
        let table = self.expect_ident()?;
        self.expect_kw("SET")?;

        let mut set_doc = None;
        let mut set_assignments = Vec::new();

        // Parse SET clause: either `doc = '...'` or `col1 = expr, col2 = expr, ...`
        let first_col = self.expect_ident()?;
        self.expect_symbol('=')?;
        let first_val = self.parse_expr()?;

        if first_col.eq_ignore_ascii_case("doc") {
            set_doc = Some(first_val);
        } else {
            set_assignments.push((first_col, first_val));
        }

        while self.peek_symbol(',') {
            self.expect_symbol(',')?;
            let col = self.expect_ident()?;
            self.expect_symbol('=')?;
            let val = self.parse_expr()?;
            set_assignments.push((col, val));
        }

        let filter = if self.peek_kw("WHERE") {
            self.expect_kw("WHERE")?;
            Some(self.parse_expr()?)
        } else {
            None
        };

        let mut as_of = None;
        let mut valid_at = None;
        self.parse_temporal_clauses(&mut as_of, &mut valid_at)?;

        Ok(Statement::Update {
            table,
            set_doc: set_doc.unwrap_or(Expr::Null),
            set_assignments,
            filter,
            as_of,
            valid_at,
        })
    }

    fn parse_delete(&mut self) -> Result<Statement> {
        self.expect_kw("DELETE")?;
        self.expect_kw("FROM")?;
        let table = self.expect_ident()?;

        let filter = if self.peek_kw("WHERE") {
            self.expect_kw("WHERE")?;
            Some(self.parse_expr()?)
        } else {
            None
        };

        let mut as_of = None;
        let mut valid_at = None;
        self.parse_temporal_clauses(&mut as_of, &mut valid_at)?;

        Ok(Statement::Delete {
            table,
            filter,
            as_of,
            valid_at,
        })
    }

    fn parse_select_or_cte(&mut self) -> Result<Statement> {
        let mut ctes = Vec::new();
        if self.peek_kw("WITH") {
            self.expect_kw("WITH")?;
            loop {
                let name = self.expect_ident()?;
                self.expect_kw("AS")?;
                self.expect_symbol('(')?;
                let query = self.parse_select_or_cte()?;
                self.expect_symbol(')')?;
                ctes.push(CteClause {
                    name,
                    query: Box::new(query),
                });
                if self.peek_symbol(',') {
                    self.expect_symbol(',')?;
                } else {
                    break;
                }
            }
        }
        self.parse_select_inner(ctes)
    }

    fn parse_select_inner(&mut self, ctes: Vec<CteClause>) -> Result<Statement> {
        self.expect_kw("SELECT")?;

        // Parse select items
        let mut items = Vec::new();
        items.push(self.parse_select_item()?);
        while self.peek_symbol(',') {
            self.expect_symbol(',')?;
            items.push(self.parse_select_item()?);
        }

        self.expect_kw("FROM")?;
        let from = self.parse_table_ref()?;

        // Parse JOINs (supports N-way: FROM a JOIN b ON ... JOIN c ON ...)
        let mut joins = Vec::new();
        while let Some(j) = self.try_parse_join(&from)? {
            joins.push(j);
        }

        // Parse WHERE
        let filter = if self.peek_kw("WHERE") {
            self.expect_kw("WHERE")?;
            Some(self.parse_expr()?)
        } else {
            None
        };

        let mut as_of = None;
        let mut as_of_epoch = None;
        let mut valid_at = None;
        let mut temporal = Vec::new();
        let mut group_by = None;
        let mut having = None;
        let mut order_by = None;
        let mut limit = None;

        loop {
            if self.peek_kw("AS") && self.peek_kw_at(1, "OF") {
                self.expect_kw("AS")?;
                self.expect_kw("OF")?;
                // AS OF EPOCH <n> — point-in-time recovery by epoch
                if self.peek_kw("EPOCH") {
                    if as_of_epoch.is_some() {
                        return Err(TensorError::SqlParse(
                            "AS OF EPOCH specified more than once".to_string(),
                        ));
                    }
                    self.expect_kw("EPOCH")?;
                    as_of_epoch = Some(self.expect_number_u64()?);
                } else {
                    if as_of.is_some() {
                        return Err(TensorError::SqlParse(
                            "AS OF specified more than once".to_string(),
                        ));
                    }
                    as_of = Some(self.expect_number_u64()?);
                }
                continue;
            }

            if self.peek_kw("VALID") {
                if valid_at.is_some() {
                    return Err(TensorError::SqlParse(
                        "VALID AT specified more than once".to_string(),
                    ));
                }
                self.expect_kw("VALID")?;
                self.expect_kw("AT")?;
                valid_at = Some(self.expect_number_u64()?);
                continue;
            }

            // SQL:2011 temporal clauses: FOR SYSTEM_TIME / FOR APPLICATION_TIME
            if self.peek_kw("FOR") {
                self.expect_kw("FOR")?;
                let clause = self.parse_temporal_for_clause()?;
                temporal.push(clause);
                continue;
            }

            if self.peek_kw("GROUP") {
                if group_by.is_some() {
                    return Err(TensorError::SqlParse(
                        "GROUP BY specified more than once".to_string(),
                    ));
                }
                self.expect_kw("GROUP")?;
                self.expect_kw("BY")?;
                let mut exprs = Vec::new();
                exprs.push(self.parse_expr()?);
                while self.peek_symbol(',') {
                    self.expect_symbol(',')?;
                    exprs.push(self.parse_expr()?);
                }
                group_by = Some(exprs);
                continue;
            }

            if self.peek_kw("HAVING") {
                if having.is_some() {
                    return Err(TensorError::SqlParse(
                        "HAVING specified more than once".to_string(),
                    ));
                }
                if group_by.is_none() {
                    return Err(TensorError::SqlParse(
                        "HAVING requires GROUP BY".to_string(),
                    ));
                }
                self.expect_kw("HAVING")?;
                having = Some(self.parse_expr()?);
                continue;
            }

            if self.peek_kw("ORDER") {
                if order_by.is_some() {
                    return Err(TensorError::SqlParse(
                        "ORDER BY specified more than once".to_string(),
                    ));
                }
                self.expect_kw("ORDER")?;
                self.expect_kw("BY")?;
                let mut orders = Vec::new();
                loop {
                    let expr = self.parse_expr()?;
                    let direction = if self.peek_kw("DESC") {
                        self.expect_kw("DESC")?;
                        OrderDirection::Desc
                    } else {
                        if self.peek_kw("ASC") {
                            self.expect_kw("ASC")?;
                        }
                        OrderDirection::Asc
                    };
                    orders.push((expr, direction));
                    if self.peek_symbol(',') {
                        self.expect_symbol(',')?;
                    } else {
                        break;
                    }
                }
                order_by = Some(orders);
                continue;
            }

            if self.peek_kw("LIMIT") {
                if limit.is_some() {
                    return Err(TensorError::SqlParse(
                        "LIMIT specified more than once".to_string(),
                    ));
                }
                self.expect_kw("LIMIT")?;
                limit = Some(self.expect_number_u64()?);
                continue;
            }

            break;
        }

        Ok(Statement::Select {
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
        })
    }

    fn parse_select_item(&mut self) -> Result<SelectItem> {
        if self.peek_symbol('*') {
            self.expect_symbol('*')?;
            return Ok(SelectItem::AllColumns);
        }

        let expr = self.parse_expr()?;
        let alias = if self.peek_kw("AS") {
            self.expect_kw("AS")?;
            Some(self.expect_ident()?)
        } else if self.peek_ident() && !self.peek_kw("FROM") {
            // Implicit alias (no AS keyword) only if next is identifier not FROM
            // Actually, don't do implicit aliases - they create ambiguity
            None
        } else {
            None
        };

        Ok(SelectItem::Expr { expr, alias })
    }

    fn parse_table_ref(&mut self) -> Result<TableRef> {
        if self.peek_symbol('(') {
            self.expect_symbol('(')?;
            let query = self.parse_select_or_cte()?;
            self.expect_symbol(')')?;
            let alias = if self.peek_kw("AS") {
                self.expect_kw("AS")?;
                self.expect_ident()?
            } else {
                self.expect_ident()?
            };
            Ok(TableRef::Subquery {
                query: Box::new(query),
                alias,
            })
        } else {
            let mut name = self.expect_ident()?;
            // Handle dot-qualified names: schema.table
            while self.peek_symbol('.') {
                self.expect_symbol('.')?;
                let part = self.expect_ident()?;
                name = format!("{name}.{part}");
            }
            // Check if this is a table function call: name(args)
            if self.peek_symbol('(') {
                self.expect_symbol('(')?;
                let mut args = Vec::new();
                if !self.peek_symbol(')') {
                    args.push(self.parse_expr()?);
                    while self.peek_symbol(',') {
                        self.expect_symbol(',')?;
                        args.push(self.parse_expr()?);
                    }
                }
                self.expect_symbol(')')?;
                let alias = if self.peek_kw("AS") {
                    self.expect_kw("AS")?;
                    Some(self.expect_ident()?)
                } else if self.peek_ident()
                    && !self.peek_kw("WHERE")
                    && !self.peek_kw("ORDER")
                    && !self.peek_kw("LIMIT")
                    && !self.peek_kw("GROUP")
                    && !self.peek_kw("HAVING")
                    && !self.peek_kw("JOIN")
                    && !self.peek_kw("INNER")
                    && !self.peek_kw("LEFT")
                    && !self.peek_kw("RIGHT")
                    && !self.peek_kw("CROSS")
                    && !self.peek_kw("UNION")
                    && !self.peek_kw("INTERSECT")
                    && !self.peek_kw("EXCEPT")
                {
                    Some(self.expect_ident()?)
                } else {
                    None
                };
                Ok(TableRef::TableFunction { name, args, alias })
            } else {
                Ok(TableRef::Named(name))
            }
        }
    }

    fn try_parse_join(&mut self, left_table: &TableRef) -> Result<Option<JoinSpec>> {
        let join_type = if self.peek_kw("JOIN") || self.peek_kw("INNER") {
            if self.peek_kw("INNER") {
                self.expect_kw("INNER")?;
            }
            self.expect_kw("JOIN")?;
            JoinType::Inner
        } else if self.peek_kw("LEFT") {
            self.expect_kw("LEFT")?;
            if self.peek_kw("OUTER") {
                self.expect_kw("OUTER")?;
            }
            self.expect_kw("JOIN")?;
            JoinType::Left
        } else if self.peek_kw("RIGHT") {
            self.expect_kw("RIGHT")?;
            if self.peek_kw("OUTER") {
                self.expect_kw("OUTER")?;
            }
            self.expect_kw("JOIN")?;
            JoinType::Right
        } else if self.peek_kw("CROSS") {
            self.expect_kw("CROSS")?;
            self.expect_kw("JOIN")?;
            JoinType::Cross
        } else {
            return Ok(None);
        };

        let right_table = self.expect_ident()?;
        let right_alias = None;

        let on_clause = if join_type == JoinType::Cross {
            None
        } else if self.peek_kw("ON") {
            self.expect_kw("ON")?;

            // Try parsing the old simple form: pk, left_t.pk=right_t.pk
            let saved = self.i;
            if self.peek_kw("PK") {
                self.expect_ident_eq("pk")?;
                // Old-style: just "pk" means pk=pk
                let left_name = match left_table {
                    TableRef::Named(n) => n.clone(),
                    _ => "left".to_string(),
                };
                Some(Expr::BinOp {
                    left: Box::new(Expr::FieldAccess {
                        column: left_name,
                        path: vec!["pk".to_string()],
                    }),
                    op: BinOperator::Eq,
                    right: Box::new(Expr::FieldAccess {
                        column: right_table.clone(),
                        path: vec!["pk".to_string()],
                    }),
                })
            } else {
                // Try to parse as expression
                self.i = saved;
                Some(self.parse_expr()?)
            }
        } else {
            None
        };

        Ok(Some(JoinSpec {
            join_type,
            right_table,
            right_alias,
            on_clause,
        }))
    }

    fn parse_temporal_clauses(
        &mut self,
        as_of: &mut Option<u64>,
        valid_at: &mut Option<u64>,
    ) -> Result<()> {
        loop {
            if self.peek_kw("AS") && self.peek_kw_at(1, "OF") {
                self.expect_kw("AS")?;
                self.expect_kw("OF")?;
                *as_of = Some(self.expect_number_u64()?);
                continue;
            }
            if self.peek_kw("VALID") {
                self.expect_kw("VALID")?;
                self.expect_kw("AT")?;
                *valid_at = Some(self.expect_number_u64()?);
                continue;
            }
            break;
        }
        Ok(())
    }

    /// Parse a SQL:2011 temporal FOR clause after the `FOR` keyword has been consumed.
    /// Parses: SYSTEM_TIME {AS OF <n> | FROM <n> TO <n> | BETWEEN <n> AND <n> | ALL}
    ///         APPLICATION_TIME {AS OF <n> | FROM <n> TO <n> | BETWEEN <n> AND <n>}
    fn parse_temporal_for_clause(&mut self) -> Result<TemporalClause> {
        if self.peek_kw("SYSTEM_TIME") {
            self.expect_kw("SYSTEM_TIME")?;

            if self.peek_kw("AS") {
                self.expect_kw("AS")?;
                self.expect_kw("OF")?;
                let ts = self.expect_number_u64()?;
                return Ok(TemporalClause::SystemTimeAsOf(ts));
            }
            if self.peek_kw("FROM") {
                self.expect_kw("FROM")?;
                let t1 = self.expect_number_u64()?;
                self.expect_kw("TO")?;
                let t2 = self.expect_number_u64()?;
                return Ok(TemporalClause::SystemTimeFromTo(t1, t2));
            }
            if self.peek_kw("BETWEEN") {
                self.expect_kw("BETWEEN")?;
                let t1 = self.expect_number_u64()?;
                self.expect_kw("AND")?;
                let t2 = self.expect_number_u64()?;
                return Ok(TemporalClause::SystemTimeBetween(t1, t2));
            }
            if self.peek_kw("ALL") {
                self.expect_kw("ALL")?;
                return Ok(TemporalClause::SystemTimeAll);
            }
            return Err(TensorError::SqlParse(
                "expected AS OF, FROM ... TO, BETWEEN ... AND, or ALL after SYSTEM_TIME"
                    .to_string(),
            ));
        }

        if self.peek_kw("APPLICATION_TIME") {
            self.expect_kw("APPLICATION_TIME")?;

            if self.peek_kw("AS") {
                self.expect_kw("AS")?;
                self.expect_kw("OF")?;
                let ts = self.expect_number_u64()?;
                return Ok(TemporalClause::ApplicationTimeAsOf(ts));
            }
            if self.peek_kw("FROM") {
                self.expect_kw("FROM")?;
                let t1 = self.expect_number_u64()?;
                self.expect_kw("TO")?;
                let t2 = self.expect_number_u64()?;
                return Ok(TemporalClause::ApplicationTimeFromTo(t1, t2));
            }
            if self.peek_kw("BETWEEN") {
                self.expect_kw("BETWEEN")?;
                let t1 = self.expect_number_u64()?;
                self.expect_kw("AND")?;
                let t2 = self.expect_number_u64()?;
                return Ok(TemporalClause::ApplicationTimeBetween(t1, t2));
            }
            return Err(TensorError::SqlParse(
                "expected AS OF, FROM ... TO, or BETWEEN ... AND after APPLICATION_TIME"
                    .to_string(),
            ));
        }

        Err(TensorError::SqlParse(
            "expected SYSTEM_TIME or APPLICATION_TIME after FOR".to_string(),
        ))
    }

    // Expression parser: precedence climbing
    // Low to high: OR → AND → NOT → comparison → addition → multiplication → unary → primary

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_and_expr()?;
        while self.peek_kw("OR") {
            self.expect_kw("OR")?;
            let right = self.parse_and_expr()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op: BinOperator::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_not_expr()?;
        while self.peek_kw("AND") {
            self.expect_kw("AND")?;
            let right = self.parse_not_expr()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op: BinOperator::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not_expr(&mut self) -> Result<Expr> {
        if self.peek_kw("NOT") {
            self.expect_kw("NOT")?;
            let inner = self.parse_not_expr()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let left = self.parse_addition()?;

        // IS NULL / IS NOT NULL
        if self.peek_kw("IS") {
            self.expect_kw("IS")?;
            let negated = if self.peek_kw("NOT") {
                self.expect_kw("NOT")?;
                true
            } else {
                false
            };
            self.expect_kw("NULL")?;
            return Ok(Expr::IsNull {
                expr: Box::new(left),
                negated,
            });
        }

        // NOT BETWEEN / NOT IN / NOT LIKE
        if self.peek_kw("NOT") {
            let saved = self.i;
            self.expect_kw("NOT")?;
            if self.peek_kw("BETWEEN") {
                self.expect_kw("BETWEEN")?;
                let low = self.parse_addition()?;
                self.expect_kw("AND")?;
                let high = self.parse_addition()?;
                return Ok(Expr::Between {
                    expr: Box::new(left),
                    low: Box::new(low),
                    high: Box::new(high),
                    negated: true,
                });
            }
            if self.peek_kw("IN") {
                self.expect_kw("IN")?;
                self.expect_symbol('(')?;
                let mut list = Vec::new();
                list.push(self.parse_expr()?);
                while self.peek_symbol(',') {
                    self.expect_symbol(',')?;
                    list.push(self.parse_expr()?);
                }
                self.expect_symbol(')')?;
                return Ok(Expr::InList {
                    expr: Box::new(left),
                    list,
                    negated: true,
                });
            }
            if self.peek_kw("LIKE") {
                self.expect_kw("LIKE")?;
                let right = self.parse_addition()?;
                return Ok(Expr::Not(Box::new(Expr::BinOp {
                    left: Box::new(left),
                    op: BinOperator::Like,
                    right: Box::new(right),
                })));
            }
            if self.peek_kw("ILIKE") {
                self.expect_kw("ILIKE")?;
                let right = self.parse_addition()?;
                return Ok(Expr::Not(Box::new(Expr::BinOp {
                    left: Box::new(left),
                    op: BinOperator::ILike,
                    right: Box::new(right),
                })));
            }
            self.i = saved;
        }

        // BETWEEN
        if self.peek_kw("BETWEEN") {
            self.expect_kw("BETWEEN")?;
            let low = self.parse_addition()?;
            self.expect_kw("AND")?;
            let high = self.parse_addition()?;
            return Ok(Expr::Between {
                expr: Box::new(left),
                low: Box::new(low),
                high: Box::new(high),
                negated: false,
            });
        }

        // IN
        if self.peek_kw("IN") {
            self.expect_kw("IN")?;
            self.expect_symbol('(')?;
            let mut list = Vec::new();
            list.push(self.parse_expr()?);
            while self.peek_symbol(',') {
                self.expect_symbol(',')?;
                list.push(self.parse_expr()?);
            }
            self.expect_symbol(')')?;
            return Ok(Expr::InList {
                expr: Box::new(left),
                list,
                negated: false,
            });
        }

        // LIKE
        if self.peek_kw("LIKE") {
            self.expect_kw("LIKE")?;
            let right = self.parse_addition()?;
            return Ok(Expr::BinOp {
                left: Box::new(left),
                op: BinOperator::Like,
                right: Box::new(right),
            });
        }

        // ILIKE (case-insensitive LIKE)
        if self.peek_kw("ILIKE") {
            self.expect_kw("ILIKE")?;
            let right = self.parse_addition()?;
            return Ok(Expr::BinOp {
                left: Box::new(left),
                op: BinOperator::ILike,
                right: Box::new(right),
            });
        }

        // Standard comparison operators
        if let Some(op) = self.peek_comparison_op() {
            self.consume_comparison_op()?;
            let right = self.parse_addition()?;
            return Ok(Expr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            });
        }

        Ok(left)
    }

    fn parse_addition(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplication()?;
        loop {
            if self.peek_symbol('+') {
                self.expect_symbol('+')?;
                let right = self.parse_multiplication()?;
                left = Expr::BinOp {
                    left: Box::new(left),
                    op: BinOperator::Add,
                    right: Box::new(right),
                };
            } else if self.peek_symbol('-') {
                self.expect_symbol('-')?;
                let right = self.parse_multiplication()?;
                left = Expr::BinOp {
                    left: Box::new(left),
                    op: BinOperator::Sub,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            if self.peek_symbol('/') {
                self.expect_symbol('/')?;
                let right = self.parse_unary()?;
                left = Expr::BinOp {
                    left: Box::new(left),
                    op: BinOperator::Div,
                    right: Box::new(right),
                };
            } else if self.peek_symbol('%') {
                self.expect_symbol('%')?;
                let right = self.parse_unary()?;
                left = Expr::BinOp {
                    left: Box::new(left),
                    op: BinOperator::Mod,
                    right: Box::new(right),
                };
            } else {
                // Note: * is handled carefully - only as multiply if preceded by expression
                // The tricky part is distinguishing SELECT * FROM ... from SELECT 2 * 3
                // In multiplication context, * after an expression is always multiply
                if self.peek_symbol('*') && !self.peek_kw_at(1, "FROM") {
                    // Check if next-next is something that looks like part of an expression
                    // or if we're in a multiplication context
                    // For safety, we only treat * as multiply in parse_multiplication
                    self.expect_symbol('*')?;
                    let right = self.parse_unary()?;
                    left = Expr::BinOp {
                        left: Box::new(left),
                        op: BinOperator::Mul,
                        right: Box::new(right),
                    };
                } else {
                    break;
                }
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        if self.peek_symbol('-') {
            self.expect_symbol('-')?;
            let inner = self.parse_primary()?;
            return Ok(Expr::BinOp {
                left: Box::new(Expr::NumberLit(0.0)),
                op: BinOperator::Sub,
                right: Box::new(inner),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        // Parenthesized expression
        if self.peek_symbol('(') {
            self.expect_symbol('(')?;
            let expr = self.parse_expr()?;
            self.expect_symbol(')')?;
            return Ok(expr);
        }

        // Star
        if self.peek_symbol('*') {
            self.expect_symbol('*')?;
            return Ok(Expr::Star);
        }

        // String literal
        if let Some(Token::StringLit(_)) = self.toks.get(self.i) {
            let s = self.expect_string()?;
            return Ok(Expr::StringLit(s));
        }

        // Number literal
        if let Some(Token::Number(_)) = self.toks.get(self.i) {
            let n = self.expect_number_f64()?;
            return Ok(Expr::NumberLit(n));
        }

        // Boolean and NULL literals
        if self.peek_kw("TRUE") {
            self.expect_kw("TRUE")?;
            return Ok(Expr::BoolLit(true));
        }
        if self.peek_kw("FALSE") {
            self.expect_kw("FALSE")?;
            return Ok(Expr::BoolLit(false));
        }
        if self.peek_kw("NULL") {
            self.expect_kw("NULL")?;
            return Ok(Expr::Null);
        }

        // CASE expression
        if self.peek_kw("CASE") {
            return self.parse_case_expr();
        }

        // CAST(expr AS type)
        if self.peek_kw("CAST") {
            self.expect_kw("CAST")?;
            self.expect_symbol('(')?;
            let expr = self.parse_expr()?;
            self.expect_kw("AS")?;
            let type_name = self.expect_ident()?;
            self.expect_symbol(')')?;
            return Ok(Expr::Cast {
                expr: Box::new(expr),
                target_type: type_name.to_uppercase(),
            });
        }

        // MATCH(field, 'query') for FTS
        if self.peek_kw("MATCH") {
            self.expect_kw("MATCH")?;
            self.expect_symbol('(')?;
            let mut args = Vec::new();
            args.push(self.parse_expr()?);
            while self.peek_symbol(',') {
                self.expect_symbol(',')?;
                args.push(self.parse_expr()?);
            }
            self.expect_symbol(')')?;
            return Ok(Expr::Function {
                name: "MATCH".to_string(),
                args,
            });
        }

        // Identifier: could be column, function call, or qualified reference
        if self.peek_ident() {
            let ident = self.expect_ident()?;

            // Function call: ident(...)
            if self.peek_symbol('(') {
                self.expect_symbol('(')?;
                let mut args = Vec::new();
                if !self.peek_symbol(')') {
                    args.push(self.parse_expr()?);
                    while self.peek_symbol(',') {
                        self.expect_symbol(',')?;
                        args.push(self.parse_expr()?);
                    }
                }
                self.expect_symbol(')')?;

                // Check for OVER clause (window function)
                if self.peek_kw("OVER") {
                    self.expect_kw("OVER")?;
                    self.expect_symbol('(')?;
                    let mut partition_by = Vec::new();
                    let mut win_order_by = Vec::new();

                    if self.peek_kw("PARTITION") {
                        self.expect_kw("PARTITION")?;
                        self.expect_kw("BY")?;
                        partition_by.push(self.parse_expr()?);
                        while self.peek_symbol(',') {
                            self.expect_symbol(',')?;
                            partition_by.push(self.parse_expr()?);
                        }
                    }

                    if self.peek_kw("ORDER") {
                        self.expect_kw("ORDER")?;
                        self.expect_kw("BY")?;
                        let expr = self.parse_expr()?;
                        let dir = if self.peek_kw("DESC") {
                            self.expect_kw("DESC")?;
                            OrderDirection::Desc
                        } else {
                            if self.peek_kw("ASC") {
                                self.expect_kw("ASC")?;
                            }
                            OrderDirection::Asc
                        };
                        win_order_by.push((expr, dir));
                        while self.peek_symbol(',') {
                            self.expect_symbol(',')?;
                            let expr = self.parse_expr()?;
                            let dir = if self.peek_kw("DESC") {
                                self.expect_kw("DESC")?;
                                OrderDirection::Desc
                            } else {
                                if self.peek_kw("ASC") {
                                    self.expect_kw("ASC")?;
                                }
                                OrderDirection::Asc
                            };
                            win_order_by.push((expr, dir));
                        }
                    }

                    self.expect_symbol(')')?;
                    return Ok(Expr::WindowFunction {
                        name: ident,
                        args,
                        partition_by,
                        order_by: win_order_by,
                    });
                }

                return Ok(Expr::Function { name: ident, args });
            }

            // Qualified reference: ident.field or ident.field.field
            if self.peek_symbol('.') {
                self.expect_symbol('.')?;
                let mut path = Vec::new();
                let field = self.expect_ident()?;
                path.push(field);
                while self.peek_symbol('.') {
                    self.expect_symbol('.')?;
                    path.push(self.expect_ident()?);
                }
                return Ok(Expr::FieldAccess {
                    column: ident,
                    path,
                });
            }

            return Ok(Expr::Column(ident));
        }

        Err(TensorError::SqlParse(format!(
            "unexpected token in expression at position {}",
            self.i
        )))
    }

    fn parse_case_expr(&mut self) -> Result<Expr> {
        self.expect_kw("CASE")?;

        // Check for simple CASE (CASE expr WHEN ...) vs searched CASE (CASE WHEN ...)
        let operand = if !self.peek_kw("WHEN") {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let mut when_clauses = Vec::new();
        while self.peek_kw("WHEN") {
            self.expect_kw("WHEN")?;
            let condition = self.parse_expr()?;
            self.expect_kw("THEN")?;
            let result = self.parse_expr()?;
            when_clauses.push((condition, result));
        }

        let else_clause = if self.peek_kw("ELSE") {
            self.expect_kw("ELSE")?;
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        self.expect_kw("END")?;

        Ok(Expr::Case {
            operand,
            when_clauses,
            else_clause,
        })
    }

    fn peek_comparison_op(&self) -> Option<BinOperator> {
        match self.toks.get(self.i) {
            Some(Token::Symbol('=')) => Some(BinOperator::Eq),
            Some(Token::NotEq) => Some(BinOperator::NotEq),
            Some(Token::LtEq) => Some(BinOperator::LtEq),
            Some(Token::GtEq) => Some(BinOperator::GtEq),
            Some(Token::Symbol('<')) => Some(BinOperator::Lt),
            Some(Token::Symbol('>')) => Some(BinOperator::Gt),
            _ => None,
        }
    }

    fn consume_comparison_op(&mut self) -> Result<()> {
        match self.toks.get(self.i) {
            Some(Token::Symbol('='))
            | Some(Token::NotEq)
            | Some(Token::LtEq)
            | Some(Token::GtEq)
            | Some(Token::Symbol('<'))
            | Some(Token::Symbol('>')) => {
                self.i += 1;
                Ok(())
            }
            _ => Err(TensorError::SqlParse(
                "expected comparison operator".to_string(),
            )),
        }
    }

    fn parse_show(&mut self) -> Result<Statement> {
        self.expect_kw("SHOW")?;
        self.expect_kw("TABLES")?;
        Ok(Statement::ShowTables)
    }

    fn parse_describe(&mut self) -> Result<Statement> {
        self.expect_kw("DESCRIBE")?;
        let table = self.expect_ident()?;
        Ok(Statement::Describe { table })
    }

    fn parse_drop(&mut self) -> Result<Statement> {
        self.expect_kw("DROP")?;
        if self.peek_kw("TABLE") {
            self.expect_kw("TABLE")?;
            let table = self.expect_ident()?;
            return Ok(Statement::DropTable { table });
        }
        if self.peek_kw("VIEW") {
            self.expect_kw("VIEW")?;
            let view = self.expect_ident()?;
            return Ok(Statement::DropView { view });
        }
        if self.peek_kw("FULLTEXT") {
            self.expect_kw("FULLTEXT")?;
            self.expect_kw("INDEX")?;
            let index = self.expect_ident()?;
            self.expect_kw("ON")?;
            let table = self.expect_ident()?;
            return Ok(Statement::DropFulltextIndex { index, table });
        }
        if self.peek_kw("INDEX") {
            self.expect_kw("INDEX")?;
            let index = self.expect_ident()?;
            self.expect_kw("ON")?;
            let table = self.expect_ident()?;
            return Ok(Statement::DropIndex { index, table });
        }
        Err(TensorError::SqlParse(
            "expected TABLE, VIEW, FULLTEXT INDEX, or INDEX after DROP".to_string(),
        ))
    }

    fn parse_copy(&mut self) -> Result<Statement> {
        self.expect_kw("COPY")?;
        let table = self.expect_ident()?;

        if self.peek_kw("TO") {
            self.expect_kw("TO")?;
            let path = self.expect_string()?;
            let format = self.parse_copy_format()?;
            Ok(Statement::CopyTo {
                table,
                path,
                format,
            })
        } else if self.peek_kw("FROM") {
            self.expect_kw("FROM")?;
            let path = self.expect_string()?;
            let format = self.parse_copy_format()?;
            Ok(Statement::CopyFrom {
                table,
                path,
                format,
            })
        } else {
            Err(TensorError::SqlParse(
                "expected TO or FROM after COPY table".to_string(),
            ))
        }
    }

    fn parse_copy_format(&mut self) -> Result<CopyFormat> {
        if self.peek_kw("WITH") {
            self.expect_kw("WITH")?;
        }
        if self.peek_kw("FORMAT") {
            self.expect_kw("FORMAT")?;
        }
        if self.peek_kw("CSV") {
            self.expect_kw("CSV")?;
            return Ok(CopyFormat::Csv);
        }
        if self.peek_kw("JSON") {
            self.expect_kw("JSON")?;
            return Ok(CopyFormat::Json);
        }
        if self.peek_kw("NDJSON") {
            self.expect_kw("NDJSON")?;
            return Ok(CopyFormat::Ndjson);
        }
        if self.peek_kw("PARQUET") {
            self.expect_kw("PARQUET")?;
            return Ok(CopyFormat::Parquet);
        }
        // Default to CSV
        Ok(CopyFormat::Csv)
    }

    /// Check for UNION / UNION ALL / INTERSECT / EXCEPT after a SELECT.
    fn try_parse_set_op(&mut self, left: Statement) -> Result<Statement> {
        if self.peek_kw("UNION") {
            self.expect_kw("UNION")?;
            let op = if self.peek_kw("ALL") {
                self.expect_kw("ALL")?;
                SetOpType::UnionAll
            } else {
                SetOpType::Union
            };
            let right = self.parse_select_or_cte()?;
            let right = self.try_parse_set_op(right)?;
            return Ok(Statement::SetOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        if self.peek_kw("INTERSECT") {
            self.expect_kw("INTERSECT")?;
            let right = self.parse_select_or_cte()?;
            let right = self.try_parse_set_op(right)?;
            return Ok(Statement::SetOp {
                op: SetOpType::Intersect,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        if self.peek_kw("EXCEPT") {
            self.expect_kw("EXCEPT")?;
            let right = self.parse_select_or_cte()?;
            let right = self.try_parse_set_op(right)?;
            return Ok(Statement::SetOp {
                op: SetOpType::Except,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    // ---- Token helpers ----

    fn peek_kw(&self, kw: &str) -> bool {
        matches!(self.toks.get(self.i), Some(Token::Ident(s)) if s.eq_ignore_ascii_case(kw))
    }

    fn peek_kw_at(&self, offset: usize, kw: &str) -> bool {
        matches!(self.toks.get(self.i + offset), Some(Token::Ident(s)) if s.eq_ignore_ascii_case(kw))
    }

    fn peek_ident(&self) -> bool {
        matches!(self.toks.get(self.i), Some(Token::Ident(_)))
    }

    fn peek_symbol(&self, sym: char) -> bool {
        matches!(self.toks.get(self.i), Some(Token::Symbol(c)) if *c == sym)
    }

    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        match self.toks.get(self.i) {
            Some(Token::Ident(s)) if s.eq_ignore_ascii_case(kw) => {
                self.i += 1;
                Ok(())
            }
            _ => Err(TensorError::SqlParse(format!("expected keyword {kw}"))),
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.toks.get(self.i) {
            Some(Token::Ident(s)) => {
                self.i += 1;
                Ok(s.clone())
            }
            _ => Err(TensorError::SqlParse("expected identifier".to_string())),
        }
    }

    fn expect_ident_eq(&mut self, expected: &str) -> Result<()> {
        let got = self.expect_ident()?;
        if got.eq_ignore_ascii_case(expected) {
            Ok(())
        } else {
            Err(TensorError::SqlParse(format!(
                "expected identifier {expected}, got {got}"
            )))
        }
    }

    fn expect_string(&mut self) -> Result<String> {
        match self.toks.get(self.i) {
            Some(Token::StringLit(s)) => {
                self.i += 1;
                Ok(s.clone())
            }
            _ => Err(TensorError::SqlParse("expected string literal".to_string())),
        }
    }

    fn expect_symbol(&mut self, symbol: char) -> Result<()> {
        match self.toks.get(self.i) {
            Some(Token::Symbol(c)) if *c == symbol => {
                self.i += 1;
                Ok(())
            }
            _ => Err(TensorError::SqlParse(format!("expected symbol {symbol}"))),
        }
    }

    fn expect_number_u64(&mut self) -> Result<u64> {
        match self.toks.get(self.i) {
            Some(Token::Number(s)) => {
                let n = s
                    .parse::<u64>()
                    .map_err(|_| TensorError::SqlParse("invalid number".to_string()))?;
                self.i += 1;
                Ok(n)
            }
            _ => Err(TensorError::SqlParse(
                "expected numeric literal".to_string(),
            )),
        }
    }

    fn expect_number_f64(&mut self) -> Result<f64> {
        match self.toks.get(self.i) {
            Some(Token::Number(s)) => {
                let n = s
                    .parse::<f64>()
                    .map_err(|_| TensorError::SqlParse("invalid number".to_string()))?;
                self.i += 1;
                Ok(n)
            }
            _ => Err(TensorError::SqlParse(
                "expected numeric literal".to_string(),
            )),
        }
    }
}

pub fn extract_pk_eq_literal(expr: Option<&Expr>) -> Option<String> {
    match expr? {
        Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        } => {
            // pk = 'literal'
            if let (Expr::Column(col), Expr::StringLit(val)) = (left.as_ref(), right.as_ref()) {
                if col.eq_ignore_ascii_case("pk") {
                    return Some(val.clone());
                }
            }
            // 'literal' = pk
            if let (Expr::StringLit(val), Expr::Column(col)) = (left.as_ref(), right.as_ref()) {
                if col.eq_ignore_ascii_case("pk") {
                    return Some(val.clone());
                }
            }
            None
        }
        _ => None,
    }
}

pub fn is_aggregate_function(name: &str) -> bool {
    matches!(
        name.to_uppercase().as_str(),
        "COUNT"
            | "SUM"
            | "AVG"
            | "MIN"
            | "MAX"
            | "FIRST"
            | "LAST"
            | "STRING_AGG"
            | "GROUP_CONCAT"
            | "STDDEV"
            | "STDDEV_POP"
            | "STDDEV_SAMP"
            | "VARIANCE"
            | "VAR_POP"
            | "VAR_SAMP"
            | "APPROX_COUNT_DISTINCT"
    )
}

pub fn select_items_contain_aggregate(items: &[SelectItem]) -> bool {
    items.iter().any(|item| match item {
        SelectItem::Expr { expr, .. } => expr_contains_aggregate(expr),
        SelectItem::AllColumns => false,
    })
}

pub fn select_items_contain_window(items: &[SelectItem]) -> bool {
    items.iter().any(|item| match item {
        SelectItem::Expr { expr, .. } => expr_contains_window(expr),
        SelectItem::AllColumns => false,
    })
}

fn expr_contains_window(expr: &Expr) -> bool {
    match expr {
        Expr::WindowFunction { .. } => true,
        Expr::BinOp { left, right, .. } => {
            expr_contains_window(left) || expr_contains_window(right)
        }
        Expr::Not(inner) => expr_contains_window(inner),
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                if expr_contains_window(op) {
                    return true;
                }
            }
            for (cond, result) in when_clauses {
                if expr_contains_window(cond) || expr_contains_window(result) {
                    return true;
                }
            }
            if let Some(el) = else_clause {
                if expr_contains_window(el) {
                    return true;
                }
            }
            false
        }
        Expr::Cast { expr, .. } => expr_contains_window(expr),
        _ => false,
    }
}

fn expr_contains_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Function { name, args } => {
            if is_aggregate_function(name) {
                return true;
            }
            args.iter().any(expr_contains_aggregate)
        }
        Expr::BinOp { left, right, .. } => {
            expr_contains_aggregate(left) || expr_contains_aggregate(right)
        }
        Expr::Not(inner) => expr_contains_aggregate(inner),
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                if expr_contains_aggregate(op) {
                    return true;
                }
            }
            for (cond, result) in when_clauses {
                if expr_contains_aggregate(cond) || expr_contains_aggregate(result) {
                    return true;
                }
            }
            if let Some(el) = else_clause {
                if expr_contains_aggregate(el) {
                    return true;
                }
            }
            false
        }
        Expr::Cast { expr, .. } => expr_contains_aggregate(expr),
        _ => false,
    }
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut out = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Multi-character operators
        if c == '!' && i + 1 < chars.len() && chars[i + 1] == '=' {
            out.push(Token::NotEq);
            i += 2;
            continue;
        }
        if c == '<' && i + 1 < chars.len() && chars[i + 1] == '=' {
            out.push(Token::LtEq);
            i += 2;
            continue;
        }
        if c == '>' && i + 1 < chars.len() && chars[i + 1] == '=' {
            out.push(Token::GtEq);
            i += 2;
            continue;
        }
        if c == '<' && i + 1 < chars.len() && chars[i + 1] == '>' {
            out.push(Token::NotEq);
            i += 2;
            continue;
        }

        if "(),;=*.<>+-/%".contains(c) {
            out.push(Token::Symbol(c));
            i += 1;
            continue;
        }
        if c == '\'' {
            i += 1;
            let mut s = String::new();
            while i < chars.len() {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    s.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                if chars[i] == '\'' {
                    i += 1;
                    break;
                }
                s.push(chars[i]);
                i += 1;
            }
            out.push(Token::StringLit(s));
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            // Check for decimal point
            if i < chars.len()
                && chars[i] == '.'
                && i + 1 < chars.len()
                && chars[i + 1].is_ascii_digit()
            {
                i += 1; // skip the dot
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num_str: String = chars[start..i].iter().collect();
            out.push(Token::Number(num_str));
            continue;
        }
        // Parameter placeholders: $1, $2, ...
        if c == '$' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            let start = i;
            i += 1; // skip $
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            out.push(Token::Ident(input[start..i].to_string()));
            continue;
        }

        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            let start = i;
            i += 1;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '-')
            {
                i += 1;
            }
            out.push(Token::Ident(input[start..i].to_string()));
            continue;
        }

        return Err(TensorError::SqlParse(format!("unexpected character {c}")));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_begin_commit_rollback() {
        assert_eq!(parse_sql("BEGIN;").unwrap(), Statement::Begin);
        assert_eq!(parse_sql("COMMIT;").unwrap(), Statement::Commit);
        assert_eq!(parse_sql("ROLLBACK;").unwrap(), Statement::Rollback);
    }

    #[test]
    fn parses_create_table_legacy() {
        let stmt = parse_sql("CREATE TABLE users (pk TEXT PRIMARY KEY);").unwrap();
        assert!(
            matches!(stmt, Statement::CreateTable { table, columns } if table == "users" && columns.len() == 1)
        );
    }

    #[test]
    fn parses_create_table_typed() {
        let stmt = parse_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, balance REAL);",
        )
        .unwrap();
        if let Statement::CreateTable { table, columns } = stmt {
            assert_eq!(table, "users");
            assert_eq!(columns.len(), 3);
            assert_eq!(columns[0].name, "id");
            assert_eq!(columns[0].type_name, SqlType::Integer);
            assert!(columns[0].primary_key);
            assert_eq!(columns[1].name, "name");
            assert_eq!(columns[1].type_name, SqlType::Text);
            assert!(columns[1].not_null);
            assert_eq!(columns[2].name, "balance");
            assert_eq!(columns[2].type_name, SqlType::Real);
        } else {
            panic!("expected CreateTable");
        }
    }

    #[test]
    fn parses_create_view_as_select() {
        let stmt = parse_sql(
            "CREATE VIEW v_orders AS SELECT doc FROM orders WHERE pk='k1' AS OF 12 VALID AT 7;",
        )
        .unwrap();

        assert_eq!(
            stmt,
            Statement::CreateView {
                view: "v_orders".to_string(),
                source: "orders".to_string(),
                pk: "k1".to_string(),
                as_of: Some(12),
                valid_at: Some(7),
            }
        );
    }

    #[test]
    fn parses_create_index() {
        let stmt = parse_sql("CREATE INDEX idx_pk ON orders (pk);").unwrap();
        assert_eq!(
            stmt,
            Statement::CreateIndex {
                index: "idx_pk".to_string(),
                table: "orders".to_string(),
                columns: vec!["pk".to_string()],
                unique: false,
            }
        );
    }

    #[test]
    fn parses_create_unique_index() {
        let stmt = parse_sql("CREATE UNIQUE INDEX idx_email ON users (email);").unwrap();
        assert_eq!(
            stmt,
            Statement::CreateIndex {
                index: "idx_email".to_string(),
                table: "users".to_string(),
                columns: vec!["email".to_string()],
                unique: true,
            }
        );
    }

    #[test]
    fn parses_create_composite_index() {
        let stmt = parse_sql("CREATE INDEX idx_city_state ON addresses (city, state);").unwrap();
        assert_eq!(
            stmt,
            Statement::CreateIndex {
                index: "idx_city_state".to_string(),
                table: "addresses".to_string(),
                columns: vec!["city".to_string(), "state".to_string()],
                unique: false,
            }
        );
    }

    #[test]
    fn parses_backup_restore() {
        let stmt = parse_sql("BACKUP DATABASE TO '/tmp/backup';").unwrap();
        assert_eq!(
            stmt,
            Statement::Backup {
                dest: "/tmp/backup".to_string(),
                since_epoch: None,
            }
        );

        let stmt = parse_sql("RESTORE DATABASE FROM '/tmp/backup';").unwrap();
        assert_eq!(
            stmt,
            Statement::Restore {
                src: "/tmp/backup".to_string()
            }
        );
    }

    #[test]
    fn parses_alter_table_add_column() {
        let stmt = parse_sql("ALTER TABLE orders ADD COLUMN status TEXT;").unwrap();
        assert_eq!(
            stmt,
            Statement::AlterTableAddColumn {
                table: "orders".to_string(),
                column: "status".to_string(),
                column_type: "TEXT".to_string(),
            }
        );
    }

    #[test]
    fn parses_select_with_where_expression() {
        let stmt = parse_sql("SELECT doc FROM t WHERE pk = 'k1';").unwrap();
        if let Statement::Select { filter, .. } = stmt {
            assert!(filter.is_some());
            let pk = extract_pk_eq_literal(filter.as_ref());
            assert_eq!(pk, Some("k1".to_string()));
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_select_with_complex_where() {
        let stmt = parse_sql("SELECT doc FROM t WHERE doc.balance > 100 AND pk = 'k1';").unwrap();
        if let Statement::Select { filter, .. } = stmt {
            assert!(filter.is_some());
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_select_scan_with_order_and_limit() {
        let stmt = parse_sql("SELECT pk, doc FROM orders ORDER BY pk DESC LIMIT 10;").unwrap();
        if let Statement::Select {
            order_by, limit, ..
        } = stmt
        {
            assert!(order_by.is_some());
            let orders = order_by.unwrap();
            assert_eq!(orders.len(), 1);
            assert_eq!(orders[0].1, OrderDirection::Desc);
            assert_eq!(limit, Some(10));
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_select_count_with_as_of_valid_at() {
        let stmt = parse_sql("SELECT count(*) FROM orders AS OF 22 VALID AT 7;").unwrap();
        if let Statement::Select {
            as_of, valid_at, ..
        } = stmt
        {
            assert_eq!(as_of, Some(22));
            assert_eq!(valid_at, Some(7));
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_join_with_qualified_on_predicate() {
        let stmt =
            parse_sql("SELECT pk, doc FROM left_t JOIN right_t ON left_t.pk=right_t.pk;").unwrap();
        if let Statement::Select { joins, .. } = stmt {
            assert_eq!(joins.len(), 1);
            let j = &joins[0];
            assert_eq!(j.join_type, JoinType::Inner);
            assert_eq!(j.right_table, "right_t");
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_pk_count_group_by_pk() {
        let stmt =
            parse_sql("SELECT pk, count(*) FROM orders GROUP BY pk ORDER BY pk DESC LIMIT 3;")
                .unwrap();
        if let Statement::Select {
            group_by,
            order_by,
            limit,
            ..
        } = stmt
        {
            assert!(group_by.is_some());
            assert!(order_by.is_some());
            assert_eq!(limit, Some(3));
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_show_describe_and_drop_statements() {
        assert_eq!(parse_sql("SHOW TABLES;").unwrap(), Statement::ShowTables);
        assert_eq!(
            parse_sql("DESCRIBE users;").unwrap(),
            Statement::Describe {
                table: "users".to_string(),
            }
        );
        assert_eq!(
            parse_sql("DROP TABLE users;").unwrap(),
            Statement::DropTable {
                table: "users".to_string(),
            }
        );
        assert_eq!(
            parse_sql("DROP VIEW v_users;").unwrap(),
            Statement::DropView {
                view: "v_users".to_string(),
            }
        );
        assert_eq!(
            parse_sql("DROP INDEX idx_users ON users;").unwrap(),
            Statement::DropIndex {
                index: "idx_users".to_string(),
                table: "users".to_string(),
            }
        );
    }

    #[test]
    fn keeps_create_table_syntax_working() {
        let stmt = parse_sql("CREATE TABLE users (pk TEXT PRIMARY KEY);").unwrap();
        if let Statement::CreateTable { table, columns } = stmt {
            assert_eq!(table, "users");
            assert_eq!(columns.len(), 1);
            assert_eq!(columns[0].name, "pk");
            assert!(columns[0].primary_key);
        } else {
            panic!("expected CreateTable");
        }
    }

    #[test]
    fn rejects_invalid_create_index_shape() {
        let err = parse_sql("CREATE INDEX idx ON t pk").unwrap_err();
        assert!(format!("{err}").contains("expected symbol ("));
    }

    #[test]
    fn splits_sql_batches_with_string_literals() {
        let stmts = split_sql_statements(
            "BEGIN; INSERT INTO t (pk, doc) VALUES ('k', '{\"a\":1;\"x\":2}'); COMMIT;",
        )
        .unwrap();
        assert_eq!(stmts.len(), 3);
        assert_eq!(stmts[0], "BEGIN");
        assert!(stmts[1].starts_with("INSERT INTO t"));
        assert_eq!(stmts[2], "COMMIT");
    }

    #[test]
    fn parses_update_with_where() {
        let stmt = parse_sql("UPDATE t SET doc = '{\"v\":3}' WHERE pk = 'k1';").unwrap();
        if let Statement::Update { table, filter, .. } = stmt {
            assert_eq!(table, "t");
            assert!(filter.is_some());
        } else {
            panic!("expected Update");
        }
    }

    #[test]
    fn parses_delete_with_where() {
        let stmt = parse_sql("DELETE FROM t WHERE pk = 'k1';").unwrap();
        if let Statement::Delete { table, filter, .. } = stmt {
            assert_eq!(table, "t");
            assert!(filter.is_some());
        } else {
            panic!("expected Delete");
        }
    }

    #[test]
    fn parses_left_join() {
        let stmt = parse_sql("SELECT pk, doc FROM a LEFT JOIN b ON a.pk = b.pk;").unwrap();
        if let Statement::Select { joins, .. } = stmt {
            assert_eq!(joins.len(), 1);
            let j = &joins[0];
            assert_eq!(j.join_type, JoinType::Left);
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_having() {
        let stmt =
            parse_sql("SELECT pk, count(*) FROM t GROUP BY pk HAVING count(*) > 1;").unwrap();
        if let Statement::Select { having, .. } = stmt {
            assert!(having.is_some());
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_between() {
        let stmt = parse_sql("SELECT doc FROM t WHERE doc.balance BETWEEN 10 AND 100;").unwrap();
        if let Statement::Select { filter, .. } = stmt {
            assert!(matches!(filter, Some(Expr::Between { .. })));
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_in_list() {
        let stmt = parse_sql("SELECT doc FROM t WHERE pk IN ('a', 'b', 'c');").unwrap();
        if let Statement::Select { filter, .. } = stmt {
            assert!(matches!(filter, Some(Expr::InList { .. })));
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_comparison_operators() {
        parse_sql("SELECT doc FROM t WHERE doc.x != 5;").unwrap();
        parse_sql("SELECT doc FROM t WHERE doc.x <= 5;").unwrap();
        parse_sql("SELECT doc FROM t WHERE doc.x >= 5;").unwrap();
        parse_sql("SELECT doc FROM t WHERE doc.x < 5;").unwrap();
        parse_sql("SELECT doc FROM t WHERE doc.x > 5;").unwrap();
    }

    #[test]
    fn parses_like() {
        let stmt = parse_sql("SELECT doc FROM t WHERE pk LIKE 'test%';").unwrap();
        if let Statement::Select { filter, .. } = stmt {
            assert!(matches!(
                filter,
                Some(Expr::BinOp {
                    op: BinOperator::Like,
                    ..
                })
            ));
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_cte() {
        let stmt = parse_sql("WITH cte1 AS (SELECT doc FROM t) SELECT doc FROM cte1;").unwrap();
        if let Statement::Select { ctes, .. } = stmt {
            assert_eq!(ctes.len(), 1);
            assert_eq!(ctes[0].name, "cte1");
        } else {
            panic!("expected Select");
        }
    }

    #[test]
    fn parses_subquery_in_from() {
        let stmt = parse_sql("SELECT doc FROM (SELECT doc FROM t) sub;").unwrap();
        if let Statement::Select {
            from: TableRef::Subquery { alias, .. },
            ..
        } = stmt
        {
            assert_eq!(alias, "sub");
        } else {
            panic!("expected Select with subquery");
        }
    }
}
