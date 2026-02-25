use crate::error::{Result, SpectraError};
use crate::sql::parser::{BinOperator, Expr};

#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
}

impl SqlValue {
    pub fn is_truthy(&self) -> bool {
        match self {
            SqlValue::Null => false,
            SqlValue::Bool(b) => *b,
            SqlValue::Number(n) => *n != 0.0,
            SqlValue::Text(s) => !s.is_empty(),
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            SqlValue::Number(n) => Some(*n),
            SqlValue::Text(s) => s.parse::<f64>().ok(),
            SqlValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            SqlValue::Null => None,
        }
    }

    pub fn to_sort_string(&self) -> String {
        match self {
            SqlValue::Null => String::new(),
            SqlValue::Bool(b) => b.to_string(),
            SqlValue::Number(n) => n.to_string(),
            SqlValue::Text(s) => s.clone(),
        }
    }

    fn cmp_partial(&self, other: &SqlValue) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (SqlValue::Null, SqlValue::Null) => Some(std::cmp::Ordering::Equal),
            (SqlValue::Null, _) | (_, SqlValue::Null) => None,
            (SqlValue::Number(a), SqlValue::Number(b)) => a.partial_cmp(b),
            (SqlValue::Text(a), SqlValue::Text(b)) => Some(a.cmp(b)),
            (SqlValue::Bool(a), SqlValue::Bool(b)) => Some(a.cmp(b)),
            (SqlValue::Number(a), SqlValue::Text(b)) => {
                if let Ok(bv) = b.parse::<f64>() {
                    a.partial_cmp(&bv)
                } else {
                    None
                }
            }
            (SqlValue::Text(a), SqlValue::Number(b)) => {
                if let Ok(av) = a.parse::<f64>() {
                    av.partial_cmp(b)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

pub struct EvalContext<'a> {
    pub pk: &'a str,
    pub doc: &'a [u8],
    doc_parsed: Option<serde_json::Value>,
}

impl<'a> EvalContext<'a> {
    pub fn new(pk: &'a str, doc: &'a [u8]) -> Self {
        Self {
            pk,
            doc,
            doc_parsed: None,
        }
    }

    fn parsed_doc(&mut self) -> &serde_json::Value {
        if self.doc_parsed.is_none() {
            self.doc_parsed = Some(
                serde_json::from_slice(self.doc)
                    .unwrap_or(serde_json::Value::Null),
            );
        }
        self.doc_parsed.as_ref().unwrap()
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<SqlValue> {
        match expr {
            Expr::Column(name) => {
                if name.eq_ignore_ascii_case("pk") {
                    Ok(SqlValue::Text(self.pk.to_string()))
                } else if name.eq_ignore_ascii_case("doc") {
                    let s = String::from_utf8_lossy(self.doc).to_string();
                    Ok(SqlValue::Text(s))
                } else {
                    // Try to look up the column in the doc JSON
                    let doc = self.parsed_doc().clone();
                    Ok(json_to_sql_value(doc.get(name)))
                }
            }
            Expr::FieldAccess { column, path } => {
                if column.eq_ignore_ascii_case("doc") {
                    let mut val = self.parsed_doc().clone();
                    for field in path {
                        val = match val.get(field) {
                            Some(v) => v.clone(),
                            None => return Ok(SqlValue::Null),
                        };
                    }
                    Ok(json_value_to_sql(&val))
                } else {
                    Ok(SqlValue::Null)
                }
            }
            Expr::StringLit(s) => Ok(SqlValue::Text(s.clone())),
            Expr::NumberLit(n) => Ok(SqlValue::Number(*n)),
            Expr::BoolLit(b) => Ok(SqlValue::Bool(*b)),
            Expr::Null => Ok(SqlValue::Null),
            Expr::Star => Ok(SqlValue::Null),
            Expr::BinOp { left, op, right } => {
                let lv = self.eval(left)?;
                match op {
                    BinOperator::And => {
                        if !lv.is_truthy() {
                            return Ok(SqlValue::Bool(false));
                        }
                        let rv = self.eval(right)?;
                        Ok(SqlValue::Bool(rv.is_truthy()))
                    }
                    BinOperator::Or => {
                        if lv.is_truthy() {
                            return Ok(SqlValue::Bool(true));
                        }
                        let rv = self.eval(right)?;
                        Ok(SqlValue::Bool(rv.is_truthy()))
                    }
                    _ => {
                        let rv = self.eval(right)?;
                        eval_binop(&lv, op, &rv)
                    }
                }
            }
            Expr::Not(inner) => {
                let v = self.eval(inner)?;
                Ok(SqlValue::Bool(!v.is_truthy()))
            }
            Expr::Function { name, args } => {
                eval_scalar_function(name, args, self)
            }
            Expr::IsNull { expr, negated } => {
                let v = self.eval(expr)?;
                let is_null = matches!(v, SqlValue::Null);
                Ok(SqlValue::Bool(if *negated { !is_null } else { is_null }))
            }
            Expr::Between { expr, low, high, negated } => {
                let v = self.eval(expr)?;
                let lo = self.eval(low)?;
                let hi = self.eval(high)?;
                let in_range = match (v.cmp_partial(&lo), v.cmp_partial(&hi)) {
                    (Some(lo_cmp), Some(hi_cmp)) => {
                        lo_cmp != std::cmp::Ordering::Less && hi_cmp != std::cmp::Ordering::Greater
                    }
                    _ => false,
                };
                Ok(SqlValue::Bool(if *negated { !in_range } else { in_range }))
            }
            Expr::InList { expr, list, negated } => {
                let v = self.eval(expr)?;
                let mut found = false;
                for item in list {
                    let iv = self.eval(item)?;
                    if let Some(std::cmp::Ordering::Equal) = v.cmp_partial(&iv) {
                        found = true;
                        break;
                    }
                }
                Ok(SqlValue::Bool(if *negated { !found } else { found }))
            }
            Expr::WindowFunction { .. } => {
                // Window functions cannot be evaluated per-row in EvalContext.
                // They are computed by the executor's window function processor.
                // If we reach here, the window value should have been injected already.
                Ok(SqlValue::Null)
            }
        }
    }
}

fn eval_binop(lv: &SqlValue, op: &BinOperator, rv: &SqlValue) -> Result<SqlValue> {
    match op {
        BinOperator::Eq => {
            let result = match lv.cmp_partial(rv) {
                Some(std::cmp::Ordering::Equal) => true,
                _ => false,
            };
            Ok(SqlValue::Bool(result))
        }
        BinOperator::NotEq => {
            let result = match lv.cmp_partial(rv) {
                Some(std::cmp::Ordering::Equal) => false,
                Some(_) => true,
                None => false,
            };
            Ok(SqlValue::Bool(result))
        }
        BinOperator::Lt => {
            let result = matches!(lv.cmp_partial(rv), Some(std::cmp::Ordering::Less));
            Ok(SqlValue::Bool(result))
        }
        BinOperator::Gt => {
            let result = matches!(lv.cmp_partial(rv), Some(std::cmp::Ordering::Greater));
            Ok(SqlValue::Bool(result))
        }
        BinOperator::LtEq => {
            let result = matches!(
                lv.cmp_partial(rv),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
            );
            Ok(SqlValue::Bool(result))
        }
        BinOperator::GtEq => {
            let result = matches!(
                lv.cmp_partial(rv),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            );
            Ok(SqlValue::Bool(result))
        }
        BinOperator::Like => {
            let (ltext, rtext) = match (lv, rv) {
                (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
                _ => return Ok(SqlValue::Bool(false)),
            };
            Ok(SqlValue::Bool(like_match(&ltext, &rtext)))
        }
        BinOperator::Add => {
            let result = match (lv.to_f64(), rv.to_f64()) {
                (Some(a), Some(b)) => SqlValue::Number(a + b),
                _ => SqlValue::Null,
            };
            Ok(result)
        }
        BinOperator::Sub => {
            let result = match (lv.to_f64(), rv.to_f64()) {
                (Some(a), Some(b)) => SqlValue::Number(a - b),
                _ => SqlValue::Null,
            };
            Ok(result)
        }
        BinOperator::Mul => {
            let result = match (lv.to_f64(), rv.to_f64()) {
                (Some(a), Some(b)) => SqlValue::Number(a * b),
                _ => SqlValue::Null,
            };
            Ok(result)
        }
        BinOperator::Div => {
            let result = match (lv.to_f64(), rv.to_f64()) {
                (Some(a), Some(b)) if b != 0.0 => SqlValue::Number(a / b),
                _ => SqlValue::Null,
            };
            Ok(result)
        }
        BinOperator::Mod => {
            let result = match (lv.to_f64(), rv.to_f64()) {
                (Some(a), Some(b)) if b != 0.0 => SqlValue::Number(a % b),
                _ => SqlValue::Null,
            };
            Ok(result)
        }
        BinOperator::And | BinOperator::Or => {
            unreachable!("AND/OR handled with short-circuit in eval")
        }
    }
}

fn eval_scalar_function(name: &str, args: &[Expr], ctx: &mut EvalContext) -> Result<SqlValue> {
    let upper = name.to_uppercase();
    match upper.as_str() {
        "COALESCE" => {
            for arg in args {
                let v = ctx.eval(arg)?;
                if !matches!(v, SqlValue::Null) {
                    return Ok(v);
                }
            }
            Ok(SqlValue::Null)
        }
        "UPPER" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Text(s) => Ok(SqlValue::Text(s.to_uppercase())),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "LOWER" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Text(s) => Ok(SqlValue::Text(s.to_lowercase())),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "LENGTH" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Text(s) => Ok(SqlValue::Number(s.len() as f64)),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "ABS" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Number(n) => Ok(SqlValue::Number(n.abs())),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "TYPEOF" => {
            if let Some(arg) = args.first() {
                let v = ctx.eval(arg)?;
                let t = match v {
                    SqlValue::Null => "null",
                    SqlValue::Bool(_) => "boolean",
                    SqlValue::Number(_) => "number",
                    SqlValue::Text(_) => "text",
                };
                Ok(SqlValue::Text(t.to_string()))
            } else {
                Ok(SqlValue::Null)
            }
        }
        _ => Err(SpectraError::SqlExec(format!(
            "unknown function: {name}"
        ))),
    }
}

fn like_match(text: &str, pattern: &str) -> bool {
    let text: Vec<char> = text.chars().collect();
    let pattern: Vec<char> = pattern.chars().collect();
    like_match_inner(&text, &pattern, 0, 0)
}

fn like_match_inner(text: &[char], pattern: &[char], ti: usize, pi: usize) -> bool {
    if pi == pattern.len() {
        return ti == text.len();
    }
    if pattern[pi] == '%' {
        // Skip consecutive %'s
        let mut next_pi = pi;
        while next_pi < pattern.len() && pattern[next_pi] == '%' {
            next_pi += 1;
        }
        if next_pi == pattern.len() {
            return true;
        }
        for start in ti..=text.len() {
            if like_match_inner(text, pattern, start, next_pi) {
                return true;
            }
        }
        return false;
    }
    if ti >= text.len() {
        return false;
    }
    if pattern[pi] == '_' || pattern[pi] == text[ti] {
        return like_match_inner(text, pattern, ti + 1, pi + 1);
    }
    false
}

pub fn json_value_to_sql(v: &serde_json::Value) -> SqlValue {
    match v {
        serde_json::Value::Null => SqlValue::Null,
        serde_json::Value::Bool(b) => SqlValue::Bool(*b),
        serde_json::Value::Number(n) => {
            SqlValue::Number(n.as_f64().unwrap_or(0.0))
        }
        serde_json::Value::String(s) => SqlValue::Text(s.clone()),
        other => SqlValue::Text(other.to_string()),
    }
}

fn json_to_sql_value(v: Option<&serde_json::Value>) -> SqlValue {
    match v {
        None => SqlValue::Null,
        Some(val) => json_value_to_sql(val),
    }
}

pub fn sql_value_to_json(v: &SqlValue) -> serde_json::Value {
    match v {
        SqlValue::Null => serde_json::Value::Null,
        SqlValue::Bool(b) => serde_json::Value::Bool(*b),
        SqlValue::Number(n) => {
            serde_json::Number::from_f64(*n)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        SqlValue::Text(s) => serde_json::Value::String(s.clone()),
    }
}

pub fn filter_rows(rows: Vec<super::exec::VisibleRow>, filter: &Expr) -> Result<Vec<super::exec::VisibleRow>> {
    let mut out = Vec::new();
    for row in rows {
        let mut ctx = EvalContext::new(&row.pk, &row.doc);
        let val = ctx.eval(filter)?;
        if val.is_truthy() {
            out.push(row);
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct AggAccumulator {
    pub count: u64,
    pub sum: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub has_values: bool,
}

impl Default for AggAccumulator {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: None,
            max: None,
            has_values: false,
        }
    }
}

impl AggAccumulator {
    pub fn accumulate(&mut self, val: &SqlValue) {
        self.count += 1;
        if let Some(n) = val.to_f64() {
            self.sum += n;
            self.has_values = true;
            self.min = Some(match self.min {
                Some(m) if m <= n => m,
                _ => n,
            });
            self.max = Some(match self.max {
                Some(m) if m >= n => m,
                _ => n,
            });
        }
    }

    pub fn avg(&self) -> SqlValue {
        if self.has_values && self.count > 0 {
            SqlValue::Number(self.sum / self.count as f64)
        } else {
            SqlValue::Null
        }
    }

    pub fn sum_value(&self) -> SqlValue {
        if self.has_values {
            SqlValue::Number(self.sum)
        } else {
            SqlValue::Null
        }
    }

    pub fn min_value(&self) -> SqlValue {
        match self.min {
            Some(v) => SqlValue::Number(v),
            None => SqlValue::Null,
        }
    }

    pub fn max_value(&self) -> SqlValue {
        match self.max {
            Some(v) => SqlValue::Number(v),
            None => SqlValue::Null,
        }
    }

    pub fn count_value(&self) -> SqlValue {
        SqlValue::Number(self.count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn like_match_basic() {
        assert!(like_match("hello", "hello"));
        assert!(!like_match("hello", "world"));
        assert!(like_match("hello", "%lo"));
        assert!(like_match("hello", "hel%"));
        assert!(like_match("hello", "%ell%"));
        assert!(like_match("hello", "h_llo"));
        assert!(!like_match("hello", "h_lo"));
        assert!(like_match("", "%"));
        assert!(like_match("abc", "%%%"));
    }

    #[test]
    fn eval_field_access() {
        let doc = br#"{"name":"Alice","age":30}"#;
        let expr = Expr::FieldAccess {
            column: "doc".to_string(),
            path: vec!["name".to_string()],
        };
        let mut ctx = EvalContext::new("k1", doc);
        assert_eq!(ctx.eval(&expr).unwrap(), SqlValue::Text("Alice".to_string()));

        let expr_age = Expr::FieldAccess {
            column: "doc".to_string(),
            path: vec!["age".to_string()],
        };
        assert_eq!(ctx.eval(&expr_age).unwrap(), SqlValue::Number(30.0));
    }

    #[test]
    fn eval_comparison() {
        let doc = br#"{"balance":100}"#;
        let expr = Expr::BinOp {
            left: Box::new(Expr::FieldAccess {
                column: "doc".to_string(),
                path: vec!["balance".to_string()],
            }),
            op: BinOperator::Gt,
            right: Box::new(Expr::NumberLit(50.0)),
        };
        let mut ctx = EvalContext::new("k1", doc);
        assert_eq!(ctx.eval(&expr).unwrap(), SqlValue::Bool(true));
    }

    #[test]
    fn eval_and_or() {
        let doc = br#"{"a":1,"b":2}"#;
        let expr = Expr::BinOp {
            left: Box::new(Expr::BinOp {
                left: Box::new(Expr::FieldAccess {
                    column: "doc".to_string(),
                    path: vec!["a".to_string()],
                }),
                op: BinOperator::Eq,
                right: Box::new(Expr::NumberLit(1.0)),
            }),
            op: BinOperator::And,
            right: Box::new(Expr::BinOp {
                left: Box::new(Expr::FieldAccess {
                    column: "doc".to_string(),
                    path: vec!["b".to_string()],
                }),
                op: BinOperator::Gt,
                right: Box::new(Expr::NumberLit(1.0)),
            }),
        };
        let mut ctx = EvalContext::new("k1", doc);
        assert_eq!(ctx.eval(&expr).unwrap(), SqlValue::Bool(true));
    }
}
