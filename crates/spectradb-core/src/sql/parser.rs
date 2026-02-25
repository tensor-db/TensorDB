use crate::error::{Result, SpectraError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Begin,
    Commit,
    Rollback,
    CreateTable {
        table: String,
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
        column: String,
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
    Select {
        table: String,
        projection: SelectProjection,
        join: Option<JoinSpec>,
        pk_filter: Option<String>,
        as_of: Option<u64>,
        valid_at: Option<u64>,
        group_by_pk: bool,
        order_by_pk: Option<OrderDirection>,
        limit: Option<u64>,
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
    Explain(Box<Statement>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectProjection {
    Doc,
    PkDoc,
    PkCount,
    CountStar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinSpec {
    pub right_table: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    Number(u64),
    StringLit(String),
    Symbol(char),
}

pub fn parse_sql(input: &str) -> Result<Statement> {
    let mut toks = tokenize(input)?;
    if matches!(toks.last(), Some(Token::Symbol(';'))) {
        toks.pop();
    }
    let mut p = Parser { toks, i: 0 };
    let stmt = p.parse_statement()?;
    if p.i != p.toks.len() {
        return Err(SpectraError::SqlParse(
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
        return Err(SpectraError::SqlParse(
            "unterminated string literal".to_string(),
        ));
    }

    let tail = input[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }

    if out.is_empty() {
        return Err(SpectraError::SqlParse(
            "no SQL statements found".to_string(),
        ));
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
            let inner = self.parse_statement()?;
            return Ok(Statement::Explain(Box::new(inner)));
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
            return Ok(Statement::Rollback);
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
        if self.peek_kw("SELECT") {
            return self.parse_select();
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
        Err(SpectraError::SqlParse(
            "unsupported SQL statement".to_string(),
        ))
    }

    fn parse_create(&mut self) -> Result<Statement> {
        self.expect_kw("CREATE")?;
        if self.peek_kw("TABLE") {
            return self.parse_create_table_after_create();
        }
        if self.peek_kw("VIEW") {
            return self.parse_create_view_after_create();
        }
        if self.peek_kw("INDEX") {
            return self.parse_create_index_after_create();
        }
        Err(SpectraError::SqlParse(
            "expected TABLE, VIEW, or INDEX after CREATE".to_string(),
        ))
    }

    fn parse_create_table_after_create(&mut self) -> Result<Statement> {
        self.expect_kw("TABLE")?;
        let table = self.expect_ident()?;
        self.expect_symbol('(')?;
        self.expect_ident_eq("pk")?;
        self.expect_kw("TEXT")?;
        self.expect_kw("PRIMARY")?;
        self.expect_kw("KEY")?;
        self.expect_symbol(')')?;
        Ok(Statement::CreateTable { table })
    }

    fn parse_create_view_after_create(&mut self) -> Result<Statement> {
        self.expect_kw("VIEW")?;
        let view = self.expect_ident()?;
        self.expect_kw("AS")?;
        let select = self.parse_select()?;
        let Statement::Select {
            table: source,
            projection,
            join,
            pk_filter,
            as_of,
            valid_at,
            group_by_pk,
            order_by_pk,
            limit,
        } = select
        else {
            return Err(SpectraError::SqlParse(
                "expected SELECT after CREATE VIEW ... AS".to_string(),
            ));
        };
        if projection != SelectProjection::Doc {
            return Err(SpectraError::SqlParse(
                "CREATE VIEW requires SELECT doc projection".to_string(),
            ));
        }
        if join.is_some() {
            return Err(SpectraError::SqlParse(
                "CREATE VIEW does not support JOIN".to_string(),
            ));
        }
        let pk = pk_filter.ok_or_else(|| {
            SpectraError::SqlParse("CREATE VIEW requires WHERE pk='...'".to_string())
        })?;
        if group_by_pk || order_by_pk.is_some() || limit.is_some() {
            return Err(SpectraError::SqlParse(
                "CREATE VIEW does not support GROUP BY, ORDER BY, or LIMIT".to_string(),
            ));
        }

        Ok(Statement::CreateView {
            view,
            source,
            pk,
            as_of,
            valid_at,
        })
    }

    fn parse_create_index_after_create(&mut self) -> Result<Statement> {
        self.expect_kw("INDEX")?;
        let index = self.expect_ident()?;
        self.expect_kw("ON")?;
        let table = self.expect_ident()?;
        self.expect_symbol('(')?;
        let column = self.expect_ident()?;
        self.expect_symbol(')')?;
        Ok(Statement::CreateIndex {
            index,
            table,
            column,
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
        self.expect_ident_eq("pk")?;
        self.expect_symbol(',')?;
        self.expect_ident_eq("doc")?;
        self.expect_symbol(')')?;
        self.expect_kw("VALUES")?;
        self.expect_symbol('(')?;
        let pk = self.expect_string()?;
        self.expect_symbol(',')?;
        let doc = self.expect_string()?;
        self.expect_symbol(')')?;
        Ok(Statement::Insert { table, pk, doc })
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
        if self.peek_kw("INDEX") {
            self.expect_kw("INDEX")?;
            let index = self.expect_ident()?;
            self.expect_kw("ON")?;
            let table = self.expect_ident()?;
            return Ok(Statement::DropIndex { index, table });
        }
        Err(SpectraError::SqlParse(
            "expected TABLE, VIEW, or INDEX after DROP".to_string(),
        ))
    }

    fn parse_select(&mut self) -> Result<Statement> {
        self.expect_kw("SELECT")?;
        let projection = if self.peek_kw("DOC") {
            self.expect_ident_eq("doc")?;
            SelectProjection::Doc
        } else if self.peek_kw("PK") {
            self.expect_ident_eq("pk")?;
            self.expect_symbol(',')?;
            if self.peek_kw("DOC") {
                self.expect_ident_eq("doc")?;
                SelectProjection::PkDoc
            } else if self.peek_kw("COUNT") {
                self.expect_kw("COUNT")?;
                self.expect_symbol('(')?;
                self.expect_symbol('*')?;
                self.expect_symbol(')')?;
                SelectProjection::PkCount
            } else {
                return Err(SpectraError::SqlParse(
                    "expected projection after pk,: doc or count(*)".to_string(),
                ));
            }
        } else if self.peek_kw("COUNT") {
            self.expect_kw("COUNT")?;
            self.expect_symbol('(')?;
            self.expect_symbol('*')?;
            self.expect_symbol(')')?;
            SelectProjection::CountStar
        } else {
            return Err(SpectraError::SqlParse(
                "expected SELECT projection: doc, pk doc, pk count(*), or count(*)".to_string(),
            ));
        };
        self.expect_kw("FROM")?;
        let table = self.expect_ident()?;

        let mut join = None;
        if self.peek_kw("JOIN") {
            self.expect_kw("JOIN")?;
            let right_table = self.expect_ident()?;
            self.expect_kw("ON")?;
            self.parse_join_predicate(&table, &right_table)?;
            join = Some(JoinSpec { right_table });
        }

        let mut pk_filter = None;
        if self.peek_kw("WHERE") {
            self.expect_kw("WHERE")?;
            self.expect_ident_eq("pk")?;
            self.expect_symbol('=')?;
            pk_filter = Some(self.expect_string()?);
        }

        let mut as_of = None;
        let mut valid_at = None;
        let mut group_by_pk = false;
        let mut order_by_pk = None;
        let mut limit = None;

        loop {
            if self.peek_kw("AS") {
                if as_of.is_some() {
                    return Err(SpectraError::SqlParse(
                        "AS OF specified more than once".to_string(),
                    ));
                }
                self.expect_kw("AS")?;
                self.expect_kw("OF")?;
                as_of = Some(self.expect_number()?);
                continue;
            }

            if self.peek_kw("VALID") {
                if valid_at.is_some() {
                    return Err(SpectraError::SqlParse(
                        "VALID AT specified more than once".to_string(),
                    ));
                }
                self.expect_kw("VALID")?;
                self.expect_kw("AT")?;
                valid_at = Some(self.expect_number()?);
                continue;
            }

            if self.peek_kw("GROUP") {
                if group_by_pk {
                    return Err(SpectraError::SqlParse(
                        "GROUP BY specified more than once".to_string(),
                    ));
                }
                self.expect_kw("GROUP")?;
                self.expect_kw("BY")?;
                self.expect_ident_eq("pk")?;
                group_by_pk = true;
                continue;
            }

            if self.peek_kw("ORDER") {
                if order_by_pk.is_some() {
                    return Err(SpectraError::SqlParse(
                        "ORDER BY specified more than once".to_string(),
                    ));
                }
                self.expect_kw("ORDER")?;
                self.expect_kw("BY")?;
                self.expect_ident_eq("pk")?;
                let direction = if self.peek_kw("DESC") {
                    self.expect_kw("DESC")?;
                    OrderDirection::Desc
                } else {
                    if self.peek_kw("ASC") {
                        self.expect_kw("ASC")?;
                    }
                    OrderDirection::Asc
                };
                order_by_pk = Some(direction);
                continue;
            }

            if self.peek_kw("LIMIT") {
                if limit.is_some() {
                    return Err(SpectraError::SqlParse(
                        "LIMIT specified more than once".to_string(),
                    ));
                }
                self.expect_kw("LIMIT")?;
                limit = Some(self.expect_number()?);
                continue;
            }

            break;
        }

        if projection == SelectProjection::PkCount && !group_by_pk {
            return Err(SpectraError::SqlParse(
                "pk, count(*) requires GROUP BY pk".to_string(),
            ));
        }
        if group_by_pk && projection != SelectProjection::PkCount {
            return Err(SpectraError::SqlParse(
                "GROUP BY pk currently requires projection pk, count(*)".to_string(),
            ));
        }
        if join.is_some() && projection == SelectProjection::Doc {
            return Err(SpectraError::SqlParse(
                "JOIN does not support SELECT doc projection".to_string(),
            ));
        }

        Ok(Statement::Select {
            table,
            projection,
            join,
            pk_filter,
            as_of,
            valid_at,
            group_by_pk,
            order_by_pk,
            limit,
        })
    }

    fn parse_join_predicate(&mut self, left: &str, right: &str) -> Result<()> {
        if self.peek_kw("PK") {
            self.expect_ident_eq("pk")?;
            return Ok(());
        }

        let left_on = self.expect_ident()?;
        if !left_on.eq_ignore_ascii_case(left) {
            return Err(SpectraError::SqlParse(format!(
                "JOIN predicate must start with {left}.pk"
            )));
        }
        self.expect_symbol('.')?;
        self.expect_ident_eq("pk")?;
        self.expect_symbol('=')?;
        let right_on = self.expect_ident()?;
        if !right_on.eq_ignore_ascii_case(right) {
            return Err(SpectraError::SqlParse(format!(
                "JOIN predicate must end with {right}.pk"
            )));
        }
        self.expect_symbol('.')?;
        self.expect_ident_eq("pk")?;
        Ok(())
    }

    fn peek_kw(&self, kw: &str) -> bool {
        matches!(self.toks.get(self.i), Some(Token::Ident(s)) if s.eq_ignore_ascii_case(kw))
    }

    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        match self.toks.get(self.i) {
            Some(Token::Ident(s)) if s.eq_ignore_ascii_case(kw) => {
                self.i += 1;
                Ok(())
            }
            _ => Err(SpectraError::SqlParse(format!("expected keyword {kw}"))),
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.toks.get(self.i) {
            Some(Token::Ident(s)) => {
                self.i += 1;
                Ok(s.clone())
            }
            _ => Err(SpectraError::SqlParse("expected identifier".to_string())),
        }
    }

    fn expect_ident_eq(&mut self, expected: &str) -> Result<()> {
        let got = self.expect_ident()?;
        if got.eq_ignore_ascii_case(expected) {
            Ok(())
        } else {
            Err(SpectraError::SqlParse(format!(
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
            _ => Err(SpectraError::SqlParse(
                "expected string literal".to_string(),
            )),
        }
    }

    fn expect_symbol(&mut self, symbol: char) -> Result<()> {
        match self.toks.get(self.i) {
            Some(Token::Symbol(c)) if *c == symbol => {
                self.i += 1;
                Ok(())
            }
            _ => Err(SpectraError::SqlParse(format!("expected symbol {symbol}"))),
        }
    }

    fn expect_number(&mut self) -> Result<u64> {
        match self.toks.get(self.i) {
            Some(Token::Number(n)) => {
                self.i += 1;
                Ok(*n)
            }
            _ => Err(SpectraError::SqlParse(
                "expected numeric literal".to_string(),
            )),
        }
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
        if "(),;=*.".contains(c) {
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
            let n: u64 = input[start..i]
                .parse()
                .map_err(|_| SpectraError::SqlParse("invalid number".to_string()))?;
            out.push(Token::Number(n));
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

        return Err(SpectraError::SqlParse(format!("unexpected character {c}")));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        parse_sql, split_sql_statements, JoinSpec, OrderDirection, SelectProjection, Statement,
    };

    #[test]
    fn parses_begin_commit_rollback() {
        assert_eq!(parse_sql("BEGIN;").unwrap(), Statement::Begin);
        assert_eq!(parse_sql("COMMIT;").unwrap(), Statement::Commit);
        assert_eq!(parse_sql("ROLLBACK;").unwrap(), Statement::Rollback);
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
                column: "pk".to_string(),
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
    fn parses_select_from_identifier() {
        let stmt = parse_sql("SELECT doc FROM v_orders WHERE pk='k1';").unwrap();
        assert_eq!(
            stmt,
            Statement::Select {
                table: "v_orders".to_string(),
                projection: SelectProjection::Doc,
                join: None,
                pk_filter: Some("k1".to_string()),
                as_of: None,
                valid_at: None,
                group_by_pk: false,
                order_by_pk: None,
                limit: None,
            }
        );
    }

    #[test]
    fn parses_select_scan_with_order_and_limit() {
        let stmt = parse_sql("SELECT pk, doc FROM orders ORDER BY pk DESC LIMIT 10;").unwrap();
        assert_eq!(
            stmt,
            Statement::Select {
                table: "orders".to_string(),
                projection: SelectProjection::PkDoc,
                join: None,
                pk_filter: None,
                as_of: None,
                valid_at: None,
                group_by_pk: false,
                order_by_pk: Some(OrderDirection::Desc),
                limit: Some(10),
            }
        );
    }

    #[test]
    fn parses_select_count_with_as_of_valid_at() {
        let stmt = parse_sql("SELECT count(*) FROM orders AS OF 22 VALID AT 7;").unwrap();
        assert_eq!(
            stmt,
            Statement::Select {
                table: "orders".to_string(),
                projection: SelectProjection::CountStar,
                join: None,
                pk_filter: None,
                as_of: Some(22),
                valid_at: Some(7),
                group_by_pk: false,
                order_by_pk: None,
                limit: None,
            }
        );
    }

    #[test]
    fn parses_join_with_qualified_on_predicate() {
        let stmt =
            parse_sql("SELECT pk, doc FROM left_t JOIN right_t ON left_t.pk=right_t.pk;").unwrap();
        assert_eq!(
            stmt,
            Statement::Select {
                table: "left_t".to_string(),
                projection: SelectProjection::PkDoc,
                join: Some(JoinSpec {
                    right_table: "right_t".to_string(),
                }),
                pk_filter: None,
                as_of: None,
                valid_at: None,
                group_by_pk: false,
                order_by_pk: None,
                limit: None,
            }
        );
    }

    #[test]
    fn parses_pk_count_group_by_pk() {
        let stmt =
            parse_sql("SELECT pk, count(*) FROM orders GROUP BY pk ORDER BY pk DESC LIMIT 3;")
                .unwrap();
        assert_eq!(
            stmt,
            Statement::Select {
                table: "orders".to_string(),
                projection: SelectProjection::PkCount,
                join: None,
                pk_filter: None,
                as_of: None,
                valid_at: None,
                group_by_pk: true,
                order_by_pk: Some(OrderDirection::Desc),
                limit: Some(3),
            }
        );
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
        assert_eq!(
            stmt,
            Statement::CreateTable {
                table: "users".to_string()
            }
        );
    }

    #[test]
    fn rejects_invalid_create_index_shape() {
        let err = parse_sql("CREATE INDEX idx ON t pk").unwrap_err();
        assert!(format!("{err}").contains("expected symbol ("));
    }

    #[test]
    fn rejects_pk_count_without_group_by() {
        let err = parse_sql("SELECT pk, count(*) FROM orders;").unwrap_err();
        assert!(format!("{err}").contains("requires GROUP BY pk"));
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
}
