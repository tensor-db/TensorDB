use crate::error::{Result, TensorError};
use crate::sql::parser::{BinOperator, Expr};

#[derive(Debug, Clone, PartialEq, Default)]
pub enum SqlValue {
    #[default]
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    Decimal(rust_decimal::Decimal),
}

impl SqlValue {
    pub fn is_truthy(&self) -> bool {
        match self {
            SqlValue::Null => false,
            SqlValue::Bool(b) => *b,
            SqlValue::Number(n) => *n != 0.0,
            SqlValue::Text(s) => !s.is_empty(),
            SqlValue::Decimal(d) => !d.is_zero(),
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            SqlValue::Number(n) => Some(*n),
            SqlValue::Text(s) => s.parse::<f64>().ok(),
            SqlValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            SqlValue::Null => None,
            SqlValue::Decimal(d) => {
                use rust_decimal::prelude::ToPrimitive;
                d.to_f64()
            }
        }
    }

    pub fn to_decimal(&self) -> Option<rust_decimal::Decimal> {
        use std::str::FromStr;
        match self {
            SqlValue::Decimal(d) => Some(*d),
            SqlValue::Number(n) => rust_decimal::Decimal::from_str(&n.to_string()).ok(),
            SqlValue::Text(s) => rust_decimal::Decimal::from_str(s).ok(),
            SqlValue::Bool(b) => Some(if *b {
                rust_decimal::Decimal::ONE
            } else {
                rust_decimal::Decimal::ZERO
            }),
            SqlValue::Null => None,
        }
    }

    pub fn to_sort_string(&self) -> String {
        match self {
            SqlValue::Null => String::new(),
            SqlValue::Bool(b) => b.to_string(),
            SqlValue::Number(n) => n.to_string(),
            SqlValue::Text(s) => s.clone(),
            SqlValue::Decimal(d) => d.to_string(),
        }
    }

    fn cmp_partial(&self, other: &SqlValue) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (SqlValue::Null, SqlValue::Null) => Some(std::cmp::Ordering::Equal),
            (SqlValue::Null, _) | (_, SqlValue::Null) => None,
            (SqlValue::Number(a), SqlValue::Number(b)) => a.partial_cmp(b),
            (SqlValue::Decimal(a), SqlValue::Decimal(b)) => Some(a.cmp(b)),
            (SqlValue::Decimal(a), SqlValue::Number(b)) => {
                use std::str::FromStr;
                rust_decimal::Decimal::from_str(&b.to_string())
                    .ok()
                    .map(|bd| a.cmp(&bd))
            }
            (SqlValue::Number(a), SqlValue::Decimal(b)) => {
                use std::str::FromStr;
                rust_decimal::Decimal::from_str(&a.to_string())
                    .ok()
                    .map(|ad| ad.cmp(b))
            }
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
            self.doc_parsed =
                Some(serde_json::from_slice(self.doc).unwrap_or(serde_json::Value::Null));
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
                    // Table-qualified column (e.g., users.name):
                    let doc = self.parsed_doc();
                    // 1. Table-keyed structure: doc[table][column] (N-way join)
                    if let Some(table_obj) = doc.get(column) {
                        if table_obj.is_object() {
                            let mut val = table_obj.clone();
                            for field in path {
                                val = match val.get(field) {
                                    Some(v) => v.clone(),
                                    None => {
                                        val = serde_json::Value::Null;
                                        break;
                                    }
                                };
                            }
                            if !val.is_null() {
                                return Ok(json_value_to_sql(&val));
                            }
                        }
                    }
                    // 2. Nested join doc: left_doc/right_doc structure
                    for key in &["left_doc", "right_doc"] {
                        if let Some(sub) = doc.get(*key) {
                            // 2a. Check if sub has a table-keyed sub-object
                            if let Some(table_obj) = sub.get(column) {
                                if table_obj.is_object() {
                                    let mut val = table_obj.clone();
                                    for field in path {
                                        val = match val.get(field) {
                                            Some(v) => v.clone(),
                                            None => {
                                                val = serde_json::Value::Null;
                                                break;
                                            }
                                        };
                                    }
                                    if !val.is_null() {
                                        return Ok(json_value_to_sql(&val));
                                    }
                                }
                            }
                            // 2b. Direct column access in left_doc/right_doc
                            let mut val = sub.clone();
                            for field in path {
                                val = match val.get(field) {
                                    Some(v) => v.clone(),
                                    None => {
                                        val = serde_json::Value::Null;
                                        break;
                                    }
                                };
                            }
                            if !val.is_null() {
                                return Ok(json_value_to_sql(&val));
                            }
                        }
                    }
                    // 3. Flat doc fallback: doc[path[0]][path[1]]...
                    if !path.is_empty() {
                        let mut val = doc.clone();
                        for field in path {
                            val = match val.get(field) {
                                Some(v) => v.clone(),
                                None => return Ok(SqlValue::Null),
                            };
                        }
                        return Ok(json_value_to_sql(&val));
                    }
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
            Expr::Function { name, args, .. } => eval_scalar_function(name, args, self),
            Expr::IsNull { expr, negated } => {
                let v = self.eval(expr)?;
                let is_null = matches!(v, SqlValue::Null);
                Ok(SqlValue::Bool(if *negated { !is_null } else { is_null }))
            }
            Expr::Between {
                expr,
                low,
                high,
                negated,
            } => {
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
            Expr::InList {
                expr,
                list,
                negated,
            } => {
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
            Expr::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op_expr) = operand {
                    // Simple CASE: CASE expr WHEN val THEN result ...
                    let op_val = self.eval(op_expr)?;
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = self.eval(when_expr)?;
                        if matches!(
                            op_val.cmp_partial(&when_val),
                            Some(std::cmp::Ordering::Equal)
                        ) {
                            return self.eval(then_expr);
                        }
                    }
                } else {
                    // Searched CASE: CASE WHEN condition THEN result ...
                    for (cond_expr, then_expr) in when_clauses {
                        let cond_val = self.eval(cond_expr)?;
                        if cond_val.is_truthy() {
                            return self.eval(then_expr);
                        }
                    }
                }
                if let Some(else_expr) = else_clause {
                    self.eval(else_expr)
                } else {
                    Ok(SqlValue::Null)
                }
            }
            Expr::Cast { expr, target_type } => {
                let val = self.eval(expr)?;
                match target_type.as_str() {
                    "INTEGER" | "INT" => match val.to_f64() {
                        Some(n) => Ok(SqlValue::Number((n as i64) as f64)),
                        None => Ok(SqlValue::Null),
                    },
                    "REAL" | "FLOAT" | "DOUBLE" => match &val {
                        SqlValue::Number(n) => Ok(SqlValue::Number(*n)),
                        SqlValue::Text(s) => match s.parse::<f64>() {
                            Ok(n) => Ok(SqlValue::Number(n)),
                            Err(_) => Ok(SqlValue::Null),
                        },
                        SqlValue::Bool(b) => Ok(SqlValue::Number(if *b { 1.0 } else { 0.0 })),
                        SqlValue::Decimal(d) => {
                            use rust_decimal::prelude::ToPrimitive;
                            Ok(SqlValue::Number(d.to_f64().unwrap_or(0.0)))
                        }
                        SqlValue::Null => Ok(SqlValue::Null),
                    },
                    "TEXT" | "VARCHAR" | "STRING" => match &val {
                        SqlValue::Null => Ok(SqlValue::Null),
                        SqlValue::Text(s) => Ok(SqlValue::Text(s.clone())),
                        SqlValue::Number(n) => Ok(SqlValue::Text(format_number(*n))),
                        SqlValue::Bool(b) => Ok(SqlValue::Text(b.to_string())),
                        SqlValue::Decimal(d) => Ok(SqlValue::Text(d.to_string())),
                    },
                    "BOOLEAN" | "BOOL" => match &val {
                        SqlValue::Bool(b) => Ok(SqlValue::Bool(*b)),
                        SqlValue::Number(n) => Ok(SqlValue::Bool(*n != 0.0)),
                        SqlValue::Text(s) => {
                            let lower = s.to_lowercase();
                            Ok(SqlValue::Bool(lower == "true" || lower == "1"))
                        }
                        SqlValue::Null => Ok(SqlValue::Null),
                        SqlValue::Decimal(d) => Ok(SqlValue::Bool(!d.is_zero())),
                    },
                    "DECIMAL" | "NUMERIC" => match val.to_decimal() {
                        Some(d) => Ok(SqlValue::Decimal(d)),
                        None => Ok(SqlValue::Null),
                    },
                    _ => Err(TensorError::SqlExec(format!(
                        "unsupported CAST target type: {target_type}"
                    ))),
                }
            }
        }
    }
}

fn eval_binop(lv: &SqlValue, op: &BinOperator, rv: &SqlValue) -> Result<SqlValue> {
    match op {
        BinOperator::Eq => {
            let result = matches!(lv.cmp_partial(rv), Some(std::cmp::Ordering::Equal));
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
            // Use Decimal arithmetic when either operand is Decimal
            if matches!(lv, SqlValue::Decimal(_)) || matches!(rv, SqlValue::Decimal(_)) {
                match (lv.to_decimal(), rv.to_decimal()) {
                    (Some(a), Some(b)) => Ok(SqlValue::Decimal(a + b)),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                match (lv.to_f64(), rv.to_f64()) {
                    (Some(a), Some(b)) => Ok(SqlValue::Number(a + b)),
                    _ => Ok(SqlValue::Null),
                }
            }
        }
        BinOperator::Sub => {
            if matches!(lv, SqlValue::Decimal(_)) || matches!(rv, SqlValue::Decimal(_)) {
                match (lv.to_decimal(), rv.to_decimal()) {
                    (Some(a), Some(b)) => Ok(SqlValue::Decimal(a - b)),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                match (lv.to_f64(), rv.to_f64()) {
                    (Some(a), Some(b)) => Ok(SqlValue::Number(a - b)),
                    _ => Ok(SqlValue::Null),
                }
            }
        }
        BinOperator::Mul => {
            if matches!(lv, SqlValue::Decimal(_)) || matches!(rv, SqlValue::Decimal(_)) {
                match (lv.to_decimal(), rv.to_decimal()) {
                    (Some(a), Some(b)) => Ok(SqlValue::Decimal(a * b)),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                match (lv.to_f64(), rv.to_f64()) {
                    (Some(a), Some(b)) => Ok(SqlValue::Number(a * b)),
                    _ => Ok(SqlValue::Null),
                }
            }
        }
        BinOperator::Div => {
            if matches!(lv, SqlValue::Decimal(_)) || matches!(rv, SqlValue::Decimal(_)) {
                match (lv.to_decimal(), rv.to_decimal()) {
                    (Some(a), Some(b)) if !b.is_zero() => Ok(SqlValue::Decimal(a / b)),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                match (lv.to_f64(), rv.to_f64()) {
                    (Some(a), Some(b)) if b != 0.0 => Ok(SqlValue::Number(a / b)),
                    _ => Ok(SqlValue::Null),
                }
            }
        }
        BinOperator::Mod => {
            if matches!(lv, SqlValue::Decimal(_)) || matches!(rv, SqlValue::Decimal(_)) {
                match (lv.to_decimal(), rv.to_decimal()) {
                    (Some(a), Some(b)) if !b.is_zero() => Ok(SqlValue::Decimal(a % b)),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                match (lv.to_f64(), rv.to_f64()) {
                    (Some(a), Some(b)) if b != 0.0 => Ok(SqlValue::Number(a % b)),
                    _ => Ok(SqlValue::Null),
                }
            }
        }
        BinOperator::ILike => {
            let text = lv.to_sort_string();
            let pattern = rv.to_sort_string();
            Ok(SqlValue::Bool(ilike_match(&text, &pattern)))
        }
        BinOperator::VectorDistance => {
            // <-> operator: compute vector distance between two vector literals
            let l_str = lv.to_sort_string();
            let r_str = rv.to_sort_string();
            match (
                crate::facet::vector_persistence::parse_vector_literal(&l_str),
                crate::facet::vector_persistence::parse_vector_literal(&r_str),
            ) {
                (Ok(v1), Ok(v2)) if v1.len() == v2.len() => {
                    let dist =
                        crate::facet::vector_search::DistanceMetric::Euclidean.compute(&v1, &v2);
                    Ok(SqlValue::Number(dist as f64))
                }
                _ => Ok(SqlValue::Null),
            }
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
                    SqlValue::Decimal(_) => "decimal",
                };
                Ok(SqlValue::Text(t.to_string()))
            } else {
                Ok(SqlValue::Null)
            }
        }
        // TIME_BUCKET(interval, timestamp) — truncate timestamp to bucket boundary
        "TIME_BUCKET" => {
            if args.len() >= 2 {
                let interval = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => parse_interval_seconds(&s),
                    SqlValue::Number(n) => Some(n as u64),
                    _ => None,
                };
                let ts = ctx.eval(&args[1])?.to_f64();
                match (interval, ts) {
                    (Some(bucket_secs), Some(ts_val)) if bucket_secs > 0 => {
                        let bucket_start = (ts_val as u64 / bucket_secs) * bucket_secs;
                        Ok(SqlValue::Number(bucket_start as f64))
                    }
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        // TIME_BUCKET_GAPFILL(interval, timestamp) — same as TIME_BUCKET but
        // signals the executor to fill gaps. In per-row eval context, it
        // behaves identically to TIME_BUCKET; gap filling happens in the
        // grouped query post-processing.
        "TIME_BUCKET_GAPFILL" => {
            if args.len() >= 2 {
                let interval = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => parse_interval_seconds(&s),
                    SqlValue::Number(n) => Some(n as u64),
                    _ => None,
                };
                let ts = ctx.eval(&args[1])?.to_f64();
                match (interval, ts) {
                    (Some(bucket_secs), Some(ts_val)) if bucket_secs > 0 => {
                        let bucket_start = (ts_val as u64 / bucket_secs) * bucket_secs;
                        Ok(SqlValue::Number(bucket_start as f64))
                    }
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        // LOCF(value) — Last Observation Carried Forward (placeholder in per-row eval;
        // actual gap filling happens post-aggregation)
        "LOCF" => {
            if let Some(arg) = args.first() {
                ctx.eval(arg)
            } else {
                Ok(SqlValue::Null)
            }
        }
        // INTERPOLATE(value) — Linear interpolation (placeholder in per-row eval;
        // actual interpolation happens post-aggregation)
        "INTERPOLATE" => {
            if let Some(arg) = args.first() {
                ctx.eval(arg)
            } else {
                Ok(SqlValue::Null)
            }
        }
        // DELTA(value) — difference from previous value (window-like, placeholder)
        "DELTA" => {
            if let Some(arg) = args.first() {
                ctx.eval(arg)
            } else {
                Ok(SqlValue::Null)
            }
        }
        // RATE(value) — per-second rate of change (window-like, placeholder)
        "RATE" => {
            if let Some(arg) = args.first() {
                ctx.eval(arg)
            } else {
                Ok(SqlValue::Null)
            }
        }
        // --- String functions ---
        "SUBSTR" | "SUBSTRING" => {
            if args.len() >= 2 {
                let s = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let start = ctx.eval(&args[1])?.to_f64().unwrap_or(1.0) as i64;
                let chars: Vec<char> = s.chars().collect();
                let start_idx = if start < 1 { 0 } else { (start - 1) as usize };
                if start_idx >= chars.len() {
                    return Ok(SqlValue::Text(String::new()));
                }
                let result = if args.len() >= 3 {
                    let len = ctx.eval(&args[2])?.to_f64().unwrap_or(0.0) as usize;
                    let end = (start_idx + len).min(chars.len());
                    chars[start_idx..end].iter().collect()
                } else {
                    chars[start_idx..].iter().collect()
                };
                Ok(SqlValue::Text(result))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "TRIM" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Text(s) => Ok(SqlValue::Text(s.trim().to_string())),
                    SqlValue::Null => Ok(SqlValue::Null),
                    other => Ok(SqlValue::Text(other.to_sort_string().trim().to_string())),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "LTRIM" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Text(s) => Ok(SqlValue::Text(s.trim_start().to_string())),
                    SqlValue::Null => Ok(SqlValue::Null),
                    other => Ok(SqlValue::Text(
                        other.to_sort_string().trim_start().to_string(),
                    )),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "RTRIM" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Text(s) => Ok(SqlValue::Text(s.trim_end().to_string())),
                    SqlValue::Null => Ok(SqlValue::Null),
                    other => Ok(SqlValue::Text(
                        other.to_sort_string().trim_end().to_string(),
                    )),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "REPLACE" => {
            if args.len() >= 3 {
                let s = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let from = match ctx.eval(&args[1])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let to = match ctx.eval(&args[2])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                Ok(SqlValue::Text(s.replace(&from, &to)))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "CONCAT" => {
            let mut result = String::new();
            for arg in args {
                match ctx.eval(arg)? {
                    SqlValue::Null => {} // NULL is skipped in CONCAT
                    other => result.push_str(&other.to_sort_string()),
                }
            }
            Ok(SqlValue::Text(result))
        }
        "CONCAT_WS" => {
            if args.is_empty() {
                return Ok(SqlValue::Null);
            }
            let sep = match ctx.eval(&args[0])? {
                SqlValue::Null => return Ok(SqlValue::Null),
                other => other.to_sort_string(),
            };
            let parts: Vec<String> = args[1..]
                .iter()
                .filter_map(|a| match ctx.eval(a).ok()? {
                    SqlValue::Null => None,
                    other => Some(other.to_sort_string()),
                })
                .collect();
            Ok(SqlValue::Text(parts.join(&sep)))
        }
        "LEFT" => {
            if args.len() >= 2 {
                let s = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let n = ctx.eval(&args[1])?.to_f64().unwrap_or(0.0) as usize;
                let result: String = s.chars().take(n).collect();
                Ok(SqlValue::Text(result))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "RIGHT" => {
            if args.len() >= 2 {
                let s = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let n = ctx.eval(&args[1])?.to_f64().unwrap_or(0.0) as usize;
                let chars: Vec<char> = s.chars().collect();
                let start = chars.len().saturating_sub(n);
                let result: String = chars[start..].iter().collect();
                Ok(SqlValue::Text(result))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "LPAD" => {
            if args.len() >= 2 {
                let s = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let target_len = ctx.eval(&args[1])?.to_f64().unwrap_or(0.0) as usize;
                let pad = if args.len() >= 3 {
                    match ctx.eval(&args[2])? {
                        SqlValue::Text(s) => s,
                        _ => " ".to_string(),
                    }
                } else {
                    " ".to_string()
                };
                let char_len = s.chars().count();
                if char_len >= target_len {
                    Ok(SqlValue::Text(s.chars().take(target_len).collect()))
                } else {
                    let needed = target_len - char_len;
                    let mut result = String::new();
                    let pad_chars: Vec<char> = pad.chars().collect();
                    if !pad_chars.is_empty() {
                        for i in 0..needed {
                            result.push(pad_chars[i % pad_chars.len()]);
                        }
                    }
                    result.push_str(&s);
                    Ok(SqlValue::Text(result))
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "RPAD" => {
            if args.len() >= 2 {
                let s = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let target_len = ctx.eval(&args[1])?.to_f64().unwrap_or(0.0) as usize;
                let pad = if args.len() >= 3 {
                    match ctx.eval(&args[2])? {
                        SqlValue::Text(s) => s,
                        _ => " ".to_string(),
                    }
                } else {
                    " ".to_string()
                };
                let char_len = s.chars().count();
                if char_len >= target_len {
                    Ok(SqlValue::Text(s.chars().take(target_len).collect()))
                } else {
                    let needed = target_len - char_len;
                    let mut result = s;
                    let pad_chars: Vec<char> = pad.chars().collect();
                    if !pad_chars.is_empty() {
                        for i in 0..needed {
                            result.push(pad_chars[i % pad_chars.len()]);
                        }
                    }
                    Ok(SqlValue::Text(result))
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "REVERSE" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Text(s) => Ok(SqlValue::Text(s.chars().rev().collect())),
                    SqlValue::Null => Ok(SqlValue::Null),
                    other => Ok(SqlValue::Text(
                        other.to_sort_string().chars().rev().collect(),
                    )),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "SPLIT_PART" => {
            if args.len() >= 3 {
                let s = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let delimiter = match ctx.eval(&args[1])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let part_num = ctx.eval(&args[2])?.to_f64().unwrap_or(0.0) as usize;
                if part_num == 0 {
                    return Ok(SqlValue::Text(String::new()));
                }
                let parts: Vec<&str> = s.split(&delimiter).collect();
                if part_num > parts.len() {
                    Ok(SqlValue::Text(String::new()))
                } else {
                    Ok(SqlValue::Text(parts[part_num - 1].to_string()))
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "REPEAT" => {
            if args.len() >= 2 {
                let s = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let n = ctx.eval(&args[1])?.to_f64().unwrap_or(0.0) as usize;
                Ok(SqlValue::Text(s.repeat(n.min(10000))))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "POSITION" | "STRPOS" => {
            if args.len() >= 2 {
                let haystack = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                let needle = match ctx.eval(&args[1])? {
                    SqlValue::Text(s) => s,
                    SqlValue::Null => return Ok(SqlValue::Null),
                    other => other.to_sort_string(),
                };
                match haystack.find(&needle) {
                    Some(pos) => {
                        let char_pos = haystack[..pos].chars().count() + 1;
                        Ok(SqlValue::Number(char_pos as f64))
                    }
                    None => Ok(SqlValue::Number(0.0)),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "INITCAP" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Text(s) => {
                        let result: String = s
                            .split_whitespace()
                            .map(|word| {
                                let mut chars = word.chars();
                                match chars.next() {
                                    Some(c) => {
                                        let upper: String = c.to_uppercase().collect();
                                        let rest: String = chars.as_str().to_lowercase();
                                        format!("{upper}{rest}")
                                    }
                                    None => String::new(),
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        Ok(SqlValue::Text(result))
                    }
                    SqlValue::Null => Ok(SqlValue::Null),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        // --- Math functions ---
        "ROUND" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)? {
                    SqlValue::Number(n) => {
                        let decimals = if args.len() >= 2 {
                            ctx.eval(&args[1])?.to_f64().unwrap_or(0.0) as i32
                        } else {
                            0
                        };
                        let factor = 10f64.powi(decimals);
                        Ok(SqlValue::Number((n * factor).round() / factor))
                    }
                    SqlValue::Null => Ok(SqlValue::Null),
                    other => match other.to_f64() {
                        Some(n) => Ok(SqlValue::Number(n.round())),
                        None => Ok(SqlValue::Null),
                    },
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "CEIL" | "CEILING" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)?.to_f64() {
                    Some(n) => Ok(SqlValue::Number(n.ceil())),
                    None => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "FLOOR" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)?.to_f64() {
                    Some(n) => Ok(SqlValue::Number(n.floor())),
                    None => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "MOD" => {
            if args.len() >= 2 {
                let a = ctx.eval(&args[0])?.to_f64();
                let b = ctx.eval(&args[1])?.to_f64();
                match (a, b) {
                    (Some(a), Some(b)) if b != 0.0 => Ok(SqlValue::Number(a % b)),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "POWER" | "POW" => {
            if args.len() >= 2 {
                let base = ctx.eval(&args[0])?.to_f64();
                let exp = ctx.eval(&args[1])?.to_f64();
                match (base, exp) {
                    (Some(b), Some(e)) => Ok(SqlValue::Number(b.powf(e))),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "SQRT" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)?.to_f64() {
                    Some(n) if n >= 0.0 => Ok(SqlValue::Number(n.sqrt())),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "LOG" | "LOG10" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)?.to_f64() {
                    Some(n) if n > 0.0 => Ok(SqlValue::Number(n.log10())),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "LN" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)?.to_f64() {
                    Some(n) if n > 0.0 => Ok(SqlValue::Number(n.ln())),
                    _ => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "EXP" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)?.to_f64() {
                    Some(n) => Ok(SqlValue::Number(n.exp())),
                    None => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "SIGN" => {
            if let Some(arg) = args.first() {
                match ctx.eval(arg)?.to_f64() {
                    Some(n) => {
                        let s = if n > 0.0 {
                            1.0
                        } else if n < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        Ok(SqlValue::Number(s))
                    }
                    None => Ok(SqlValue::Null),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "RANDOM" => Ok(SqlValue::Number(fastrand_f64())),
        "PI" => Ok(SqlValue::Number(std::f64::consts::PI)),
        // --- Utility functions ---
        "NULLIF" => {
            if args.len() >= 2 {
                let a = ctx.eval(&args[0])?;
                let b = ctx.eval(&args[1])?;
                if matches!(a.cmp_partial(&b), Some(std::cmp::Ordering::Equal)) {
                    Ok(SqlValue::Null)
                } else {
                    Ok(a)
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "GREATEST" => {
            let mut best: Option<SqlValue> = None;
            for arg in args {
                let v = ctx.eval(arg)?;
                if matches!(v, SqlValue::Null) {
                    continue;
                }
                best = Some(match best {
                    None => v,
                    Some(ref cur) => {
                        if matches!(v.cmp_partial(cur), Some(std::cmp::Ordering::Greater)) {
                            v
                        } else {
                            best.unwrap()
                        }
                    }
                });
            }
            Ok(best.unwrap_or(SqlValue::Null))
        }
        "LEAST" => {
            let mut best: Option<SqlValue> = None;
            for arg in args {
                let v = ctx.eval(arg)?;
                if matches!(v, SqlValue::Null) {
                    continue;
                }
                best = Some(match best {
                    None => v,
                    Some(ref cur) => {
                        if matches!(v.cmp_partial(cur), Some(std::cmp::Ordering::Less)) {
                            v
                        } else {
                            best.unwrap()
                        }
                    }
                });
            }
            Ok(best.unwrap_or(SqlValue::Null))
        }
        "IF" | "IIF" => {
            if args.len() >= 3 {
                let cond = ctx.eval(&args[0])?;
                if cond.is_truthy() {
                    ctx.eval(&args[1])
                } else {
                    ctx.eval(&args[2])
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        // --- Date/Time functions ---
        "NOW" | "CURRENT_TIMESTAMP" => {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Ok(SqlValue::Number(ts as f64))
        }
        "EXTRACT" => {
            // EXTRACT('field', timestamp) — simplified epoch-based extraction
            if args.len() >= 2 {
                let field = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s.to_uppercase(),
                    _ => return Ok(SqlValue::Null),
                };
                let ts = match ctx.eval(&args[1])?.to_f64() {
                    Some(n) => n as i64,
                    None => return Ok(SqlValue::Null),
                };
                let result = extract_from_epoch(ts, &field);
                Ok(SqlValue::Number(result))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "DATE_TRUNC" => {
            // DATE_TRUNC('field', timestamp) — truncate to boundary
            if args.len() >= 2 {
                let field = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s.to_uppercase(),
                    _ => return Ok(SqlValue::Null),
                };
                let ts = match ctx.eval(&args[1])?.to_f64() {
                    Some(n) => n as i64,
                    None => return Ok(SqlValue::Null),
                };
                let result = date_trunc_epoch(ts, &field);
                Ok(SqlValue::Number(result as f64))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "DATE_PART" => {
            // Alias for EXTRACT
            if args.len() >= 2 {
                let field = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s.to_uppercase(),
                    _ => return Ok(SqlValue::Null),
                };
                let ts = match ctx.eval(&args[1])?.to_f64() {
                    Some(n) => n as i64,
                    None => return Ok(SqlValue::Null),
                };
                let result = extract_from_epoch(ts, &field);
                Ok(SqlValue::Number(result))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "TO_CHAR" => {
            // Simplified TO_CHAR: just formats number or timestamp as string
            if let Some(arg) = args.first() {
                let val = ctx.eval(arg)?;
                match val {
                    SqlValue::Null => Ok(SqlValue::Null),
                    SqlValue::Number(n) => Ok(SqlValue::Text(format_number(n))),
                    other => Ok(SqlValue::Text(other.to_sort_string())),
                }
            } else {
                Ok(SqlValue::Null)
            }
        }
        "EPOCH" => {
            // Return current unix epoch
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Ok(SqlValue::Number(ts as f64))
        }
        // MATCH(column, 'query') — always returns true when used in WHERE
        // because FTS pre-filtering already selected matching rows.
        // When used in SELECT, returns the query for reference.
        "MATCH" => Ok(SqlValue::Bool(true)),
        // HIGHLIGHT(column, 'query') — returns text with <<match>> markers
        "HIGHLIGHT" => {
            if args.len() >= 2 {
                let text = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    _ => return Ok(SqlValue::Null),
                };
                let query = match ctx.eval(&args[1])? {
                    SqlValue::Text(s) => s,
                    _ => return Ok(SqlValue::Null),
                };
                let highlighted = highlight_text(&text, &query);
                Ok(SqlValue::Text(highlighted))
            } else {
                Ok(SqlValue::Null)
            }
        }
        // --- Vector functions ---
        "VECTOR_DISTANCE" => {
            // VECTOR_DISTANCE(v1, v2[, metric])
            if args.len() >= 2 {
                let v1_str = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    _ => return Ok(SqlValue::Null),
                };
                let v2_str = match ctx.eval(&args[1])? {
                    SqlValue::Text(s) => s,
                    _ => return Ok(SqlValue::Null),
                };
                let v1 = crate::facet::vector_persistence::parse_vector_literal(&v1_str)?;
                let v2 = crate::facet::vector_persistence::parse_vector_literal(&v2_str)?;
                if v1.len() != v2.len() {
                    return Err(TensorError::SqlExec(
                        "VECTOR_DISTANCE: dimension mismatch".to_string(),
                    ));
                }
                let metric_str = if args.len() >= 3 {
                    match ctx.eval(&args[2])? {
                        SqlValue::Text(s) => s.to_lowercase(),
                        _ => "euclidean".to_string(),
                    }
                } else {
                    "euclidean".to_string()
                };
                let metric =
                    crate::facet::vector_ops::VectorSqlBridge::parse_distance_metric(&metric_str)
                        .unwrap_or(crate::facet::vector_search::DistanceMetric::Euclidean);
                let dist = metric.compute(&v1, &v2) as f64;
                Ok(SqlValue::Number(dist))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "COSINE_SIMILARITY" => {
            // COSINE_SIMILARITY(v1, v2) -> 1 - cosine_distance
            if args.len() >= 2 {
                let v1_str = match ctx.eval(&args[0])? {
                    SqlValue::Text(s) => s,
                    _ => return Ok(SqlValue::Null),
                };
                let v2_str = match ctx.eval(&args[1])? {
                    SqlValue::Text(s) => s,
                    _ => return Ok(SqlValue::Null),
                };
                let v1 = crate::facet::vector_persistence::parse_vector_literal(&v1_str)?;
                let v2 = crate::facet::vector_persistence::parse_vector_literal(&v2_str)?;
                if v1.len() != v2.len() {
                    return Err(TensorError::SqlExec(
                        "COSINE_SIMILARITY: dimension mismatch".to_string(),
                    ));
                }
                let cos_dist =
                    crate::facet::vector_search::DistanceMetric::Cosine.compute(&v1, &v2);
                Ok(SqlValue::Number((1.0 - cos_dist) as f64))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "VECTOR_NORM" => {
            if let Some(arg) = args.first() {
                let s = match ctx.eval(arg)? {
                    SqlValue::Text(s) => s,
                    _ => return Ok(SqlValue::Null),
                };
                let v = crate::facet::vector_persistence::parse_vector_literal(&s)?;
                let norm: f64 = v
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                Ok(SqlValue::Number(norm))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "VECTOR_DIMS" => {
            if let Some(arg) = args.first() {
                let s = match ctx.eval(arg)? {
                    SqlValue::Text(s) => s,
                    _ => return Ok(SqlValue::Null),
                };
                let v = crate::facet::vector_persistence::parse_vector_literal(&s)?;
                Ok(SqlValue::Number(v.len() as f64))
            } else {
                Ok(SqlValue::Null)
            }
        }
        "HYBRID_SCORE" => {
            // HYBRID_SCORE(vector_distance, text_score, vector_weight, text_weight)
            if args.len() >= 4 {
                let vdist = ctx.eval(&args[0])?.to_f64().unwrap_or(f64::MAX);
                let tscore = ctx.eval(&args[1])?.to_f64().unwrap_or(0.0);
                let vweight = ctx.eval(&args[2])?.to_f64().unwrap_or(0.5);
                let tweight = ctx.eval(&args[3])?.to_f64().unwrap_or(0.5);
                let score = crate::facet::vector_hybrid::compute_hybrid_score(
                    vdist, tscore, vweight, tweight,
                );
                Ok(SqlValue::Number(score))
            } else {
                Ok(SqlValue::Null)
            }
        }
        _ => Err(TensorError::SqlExec(format!("unknown function: {name}"))),
    }
}

/// Highlight matching query terms in text with <<>> markers.
fn highlight_text(text: &str, query: &str) -> String {
    use crate::facet::fts::{stem, tokenize};

    let query_stems: Vec<String> = tokenize(query).iter().map(|t| stem(t)).collect();
    let mut result = String::new();

    for word in text.split_whitespace() {
        if !result.is_empty() {
            result.push(' ');
        }
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        let stemmed = stem(&cleaned);
        if query_stems.contains(&stemmed) {
            result.push_str("<<");
            result.push_str(word);
            result.push_str(">>");
        } else {
            result.push_str(word);
        }
    }
    result
}

/// Simple pseudo-random f64 in [0, 1) using thread-local state.
fn fastrand_f64() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0x12345678_9abcdef0) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        // Mix in current time on first call
        if x == 0x12345678_9abcdef0 {
            x ^= std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
        }
        // xorshift64
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 11) as f64 / (1u64 << 53) as f64
    })
}

/// Extract a date field from a Unix epoch timestamp (seconds).
fn extract_from_epoch(epoch_secs: i64, field: &str) -> f64 {
    // Simple date math without external crate
    const SECS_PER_MINUTE: i64 = 60;
    const SECS_PER_HOUR: i64 = 3600;
    const SECS_PER_DAY: i64 = 86400;

    match field {
        "EPOCH" => epoch_secs as f64,
        "SECOND" => (epoch_secs % 60) as f64,
        "MINUTE" => ((epoch_secs % SECS_PER_HOUR) / SECS_PER_MINUTE) as f64,
        "HOUR" => ((epoch_secs % SECS_PER_DAY) / SECS_PER_HOUR) as f64,
        "DOW" | "DAYOFWEEK" => {
            // Unix epoch (1970-01-01) was a Thursday (4)
            let days = epoch_secs.div_euclid(SECS_PER_DAY);
            ((days + 4) % 7) as f64
        }
        "DOY" | "DAYOFYEAR" => {
            let (y, m, d) = epoch_to_ymd(epoch_secs);
            day_of_year(y, m, d) as f64
        }
        "DAY" => {
            let (_, _, d) = epoch_to_ymd(epoch_secs);
            d as f64
        }
        "MONTH" => {
            let (_, m, _) = epoch_to_ymd(epoch_secs);
            m as f64
        }
        "YEAR" => {
            let (y, _, _) = epoch_to_ymd(epoch_secs);
            y as f64
        }
        "QUARTER" => {
            let (_, m, _) = epoch_to_ymd(epoch_secs);
            ((m - 1) / 3 + 1) as f64
        }
        "WEEK" | "ISOWEEK" => {
            let (y, m, d) = epoch_to_ymd(epoch_secs);
            let doy = day_of_year(y, m, d);
            ((doy - 1) / 7 + 1) as f64
        }
        _ => 0.0,
    }
}

/// Truncate a Unix epoch timestamp to a date boundary.
fn date_trunc_epoch(epoch_secs: i64, field: &str) -> i64 {
    const SECS_PER_MINUTE: i64 = 60;
    const SECS_PER_HOUR: i64 = 3600;
    const SECS_PER_DAY: i64 = 86400;

    match field {
        "SECOND" => epoch_secs,
        "MINUTE" => (epoch_secs / SECS_PER_MINUTE) * SECS_PER_MINUTE,
        "HOUR" => (epoch_secs / SECS_PER_HOUR) * SECS_PER_HOUR,
        "DAY" => (epoch_secs / SECS_PER_DAY) * SECS_PER_DAY,
        "WEEK" => {
            let days = epoch_secs / SECS_PER_DAY;
            // Monday-based: epoch day 0 (1970-01-01) is Thursday (3 days after Monday)
            let monday = days - ((days + 3) % 7);
            monday * SECS_PER_DAY
        }
        "MONTH" => {
            let (y, m, _) = epoch_to_ymd(epoch_secs);
            ymd_to_epoch(y, m, 1)
        }
        "QUARTER" => {
            let (y, m, _) = epoch_to_ymd(epoch_secs);
            let q_month = ((m - 1) / 3) * 3 + 1;
            ymd_to_epoch(y, q_month, 1)
        }
        "YEAR" => {
            let (y, _, _) = epoch_to_ymd(epoch_secs);
            ymd_to_epoch(y, 1, 1)
        }
        _ => epoch_secs,
    }
}

/// Convert Unix epoch seconds to (year, month, day).
fn epoch_to_ymd(epoch_secs: i64) -> (i32, u32, u32) {
    let days = (epoch_secs / 86400) as i32;
    // Civil days algorithm from Howard Hinnant
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Convert (year, month, day) to Unix epoch seconds.
fn ymd_to_epoch(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u32;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i32 - 719468;
    days as i64 * 86400
}

/// Day of year (1-366).
fn day_of_year(y: i32, m: u32, d: u32) -> u32 {
    let is_leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_days: [u32; 12] = [
        31,
        if is_leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut doy = d;
    for md in month_days.iter().take((m as usize - 1).min(11)) {
        doy += md;
    }
    doy
}

/// Case-insensitive LIKE match.
pub fn ilike_match(text: &str, pattern: &str) -> bool {
    let text_lower: Vec<char> = text.to_lowercase().chars().collect();
    let pattern_lower: Vec<char> = pattern.to_lowercase().chars().collect();
    like_match_inner(&text_lower, &pattern_lower, 0, 0)
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
        serde_json::Value::Number(n) => SqlValue::Number(n.as_f64().unwrap_or(0.0)),
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

fn format_number(n: f64) -> String {
    if n == (n as i64) as f64 {
        format!("{}", n as i64)
    } else {
        n.to_string()
    }
}

pub fn sql_value_to_json(v: &SqlValue) -> serde_json::Value {
    match v {
        SqlValue::Null => serde_json::Value::Null,
        SqlValue::Bool(b) => serde_json::Value::Bool(*b),
        SqlValue::Number(n) => serde_json::Number::from_f64(*n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        SqlValue::Text(s) => serde_json::Value::String(s.clone()),
        SqlValue::Decimal(d) => serde_json::Value::String(d.to_string()),
    }
}

pub fn filter_rows(
    rows: Vec<super::exec::VisibleRow>,
    filter: &Expr,
) -> Result<Vec<super::exec::VisibleRow>> {
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

/// Parse a human-readable interval string into seconds.
/// Supports: '1s', '30s', '1m', '5m', '1h', '6h', '1d', '7d', '1w'.
pub fn parse_interval_seconds(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase();
    if let Ok(n) = s.parse::<u64>() {
        return Some(n);
    }
    let (num_str, unit) = if s.ends_with("ms") {
        // Milliseconds: round to nearest second, minimum 1
        let n: &str = &s[..s.len() - 2];
        let ms = n.parse::<u64>().ok()?;
        return Some((ms / 1000).max(1));
    } else {
        let last = s.chars().last()?;
        (&s[..s.len() - last.len_utf8()], last)
    };
    let n: u64 = if num_str.is_empty() {
        1
    } else {
        num_str.parse().ok()?
    };
    let multiplier = match unit {
        's' => 1,
        'm' => 60,
        'h' => 3600,
        'd' => 86400,
        'w' => 604800,
        _ => return None,
    };
    Some(n * multiplier)
}

/// Accumulator for FIRST(value, timestamp) aggregate — returns the value
/// at the smallest timestamp.
#[derive(Debug, Clone, Default)]
pub struct FirstAccumulator {
    pub min_ts: Option<f64>,
    pub value: SqlValue,
}

impl FirstAccumulator {
    pub fn accumulate(&mut self, val: &SqlValue, ts: &SqlValue) {
        if let Some(ts_f) = ts.to_f64() {
            if self.min_ts.is_none() || ts_f < self.min_ts.unwrap() {
                self.min_ts = Some(ts_f);
                self.value = val.clone();
            }
        }
    }

    pub fn result(&self) -> SqlValue {
        self.value.clone()
    }
}

/// Accumulator for LAST(value, timestamp) aggregate — returns the value
/// at the largest timestamp.
#[derive(Debug, Clone, Default)]
pub struct LastAccumulator {
    pub max_ts: Option<f64>,
    pub value: SqlValue,
}

impl LastAccumulator {
    pub fn accumulate(&mut self, val: &SqlValue, ts: &SqlValue) {
        if let Some(ts_f) = ts.to_f64() {
            if self.max_ts.is_none() || ts_f > self.max_ts.unwrap() {
                self.max_ts = Some(ts_f);
                self.value = val.clone();
            }
        }
    }

    pub fn result(&self) -> SqlValue {
        self.value.clone()
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
        assert_eq!(
            ctx.eval(&expr).unwrap(),
            SqlValue::Text("Alice".to_string())
        );

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
