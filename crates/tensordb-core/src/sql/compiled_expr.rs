//! Expression compilation: convert common predicate patterns to closures
//! for faster evaluation during table scans.
//!
//! The interpreter-based eval path (EvalContext::eval) is general but has
//! per-row overhead from matching on Expr variants. For hot predicates, we
//! pattern-match common shapes to direct Rust closures that bypass the
//! interpreter.

use crate::sql::eval::SqlValue;
use crate::sql::parser::{BinOperator, Expr};
use std::collections::HashSet;

/// A compiled predicate that can be applied directly to JSON document bytes.
pub enum CompiledPredicate {
    /// `column = literal_string`
    EqString { column: String, value: String },
    /// `column = literal_number`
    EqNumber { column: String, value: f64 },
    /// `column > literal_number`
    GtNumber { column: String, value: f64 },
    /// `column < literal_number`
    LtNumber { column: String, value: f64 },
    /// `column >= literal_number AND column <= literal_number` (range)
    RangeNumber { column: String, low: f64, high: f64 },
    /// `column IN ('a', 'b', 'c')` — HashSet lookup
    InStringSet {
        column: String,
        values: HashSet<String>,
    },
    /// `column IN (1, 2, 3)` — HashSet lookup on integer values
    InNumberSet {
        column: String,
        values: HashSet<i64>,
    },
    /// Fallback — use interpreter
    Interpreted,
}

impl CompiledPredicate {
    /// Evaluate the compiled predicate against a JSON document.
    /// Returns true if the row matches.
    pub fn matches(&self, doc: &[u8]) -> bool {
        match self {
            CompiledPredicate::EqString { column, value } => {
                extract_string_field(doc, column).is_some_and(|s| s == *value)
            }
            CompiledPredicate::EqNumber { column, value } => {
                extract_number_field(doc, column).is_some_and(|n| (n - value).abs() < f64::EPSILON)
            }
            CompiledPredicate::GtNumber { column, value } => {
                extract_number_field(doc, column).is_some_and(|n| n > *value)
            }
            CompiledPredicate::LtNumber { column, value } => {
                extract_number_field(doc, column).is_some_and(|n| n < *value)
            }
            CompiledPredicate::RangeNumber { column, low, high } => {
                extract_number_field(doc, column).is_some_and(|n| n >= *low && n <= *high)
            }
            CompiledPredicate::InStringSet { column, values } => {
                extract_string_field(doc, column).is_some_and(|s| values.contains(s.as_str()))
            }
            CompiledPredicate::InNumberSet { column, values } => {
                extract_number_field(doc, column).is_some_and(|n| values.contains(&(n as i64)))
            }
            CompiledPredicate::Interpreted => true, // caller should use eval fallback
        }
    }

    /// Returns true if this is the fallback interpreted mode.
    pub fn is_interpreted(&self) -> bool {
        matches!(self, CompiledPredicate::Interpreted)
    }
}

/// Try to compile an Expr into a CompiledPredicate.
/// Returns `CompiledPredicate::Interpreted` if the expression can't be compiled.
pub fn try_compile_predicate(expr: &Expr) -> CompiledPredicate {
    match expr {
        // column = 'literal'
        Expr::BinOp {
            left,
            op: BinOperator::Eq,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (Expr::Column(col), Expr::StringLit(val))
            | (Expr::StringLit(val), Expr::Column(col)) => CompiledPredicate::EqString {
                column: col.clone(),
                value: val.clone(),
            },
            (Expr::Column(col), Expr::NumberLit(val))
            | (Expr::NumberLit(val), Expr::Column(col)) => CompiledPredicate::EqNumber {
                column: col.clone(),
                value: *val,
            },
            _ => CompiledPredicate::Interpreted,
        },
        // column > number
        Expr::BinOp {
            left,
            op: BinOperator::Gt,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (Expr::Column(col), Expr::NumberLit(val)) => CompiledPredicate::GtNumber {
                column: col.clone(),
                value: *val,
            },
            _ => CompiledPredicate::Interpreted,
        },
        // column < number
        Expr::BinOp {
            left,
            op: BinOperator::Lt,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (Expr::Column(col), Expr::NumberLit(val)) => CompiledPredicate::LtNumber {
                column: col.clone(),
                value: *val,
            },
            _ => CompiledPredicate::Interpreted,
        },
        // column >= low AND column <= high (range predicate)
        Expr::BinOp {
            left,
            op: BinOperator::And,
            right,
        } => {
            if let (Some((col1, low)), Some((col2, high))) = (extract_gte(left), extract_lte(right))
            {
                if col1 == col2 {
                    return CompiledPredicate::RangeNumber {
                        column: col1,
                        low,
                        high,
                    };
                }
            }
            CompiledPredicate::Interpreted
        }
        // column IN ('a', 'b', 'c')
        Expr::InList {
            expr,
            list,
            negated: false,
        } => {
            if let Expr::Column(col) = expr.as_ref() {
                // Check if all list items are string literals
                let strings: Vec<String> = list
                    .iter()
                    .filter_map(|e| {
                        if let Expr::StringLit(s) = e {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if strings.len() == list.len() {
                    return CompiledPredicate::InStringSet {
                        column: col.clone(),
                        values: strings.into_iter().collect(),
                    };
                }
                // Check if all list items are number literals
                let numbers: Vec<i64> = list
                    .iter()
                    .filter_map(|e| {
                        if let Expr::NumberLit(n) = e {
                            Some(*n as i64)
                        } else {
                            None
                        }
                    })
                    .collect();
                if numbers.len() == list.len() {
                    return CompiledPredicate::InNumberSet {
                        column: col.clone(),
                        values: numbers.into_iter().collect(),
                    };
                }
            }
            CompiledPredicate::Interpreted
        }
        _ => CompiledPredicate::Interpreted,
    }
}

/// Extract (column, value) from `column >= number`.
fn extract_gte(expr: &Expr) -> Option<(String, f64)> {
    if let Expr::BinOp {
        left,
        op: BinOperator::GtEq,
        right,
    } = expr
    {
        if let (Expr::Column(col), Expr::NumberLit(val)) = (left.as_ref(), right.as_ref()) {
            return Some((col.clone(), *val));
        }
    }
    None
}

/// Extract (column, value) from `column <= number`.
fn extract_lte(expr: &Expr) -> Option<(String, f64)> {
    if let Expr::BinOp {
        left,
        op: BinOperator::LtEq,
        right,
    } = expr
    {
        if let (Expr::Column(col), Expr::NumberLit(val)) = (left.as_ref(), right.as_ref()) {
            return Some((col.clone(), *val));
        }
    }
    None
}

/// Fast field extraction from JSON bytes without full parsing.
/// Looks for `"field_name":` pattern and extracts the value.
fn extract_string_field(doc: &[u8], field: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_slice(doc).ok()?;
    match parsed.get(field)? {
        serde_json::Value::String(s) => Some(s.clone()),
        v => Some(v.to_string()),
    }
}

fn extract_number_field(doc: &[u8], field: &str) -> Option<f64> {
    let parsed: serde_json::Value = serde_json::from_slice(doc).ok()?;
    parsed.get(field)?.as_f64()
}

/// Evaluate a SqlValue from a compiled predicate match context.
pub fn sql_value_from_json_field(doc: &[u8], field: &str) -> SqlValue {
    let parsed: Option<serde_json::Value> = serde_json::from_slice(doc).ok();
    match parsed.and_then(|v| v.get(field).cloned()) {
        Some(serde_json::Value::String(s)) => SqlValue::Text(s),
        Some(serde_json::Value::Number(n)) => SqlValue::Number(n.as_f64().unwrap_or(0.0)),
        Some(serde_json::Value::Bool(b)) => SqlValue::Bool(b),
        Some(serde_json::Value::Null) | None => SqlValue::Null,
        Some(other) => SqlValue::Text(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_eq_string() {
        let expr = Expr::BinOp {
            left: Box::new(Expr::Column("name".to_string())),
            op: BinOperator::Eq,
            right: Box::new(Expr::StringLit("Alice".to_string())),
        };
        let compiled = try_compile_predicate(&expr);
        assert!(!compiled.is_interpreted());

        let doc = br#"{"name":"Alice","age":30}"#;
        assert!(compiled.matches(doc));

        let doc2 = br#"{"name":"Bob","age":25}"#;
        assert!(!compiled.matches(doc2));
    }

    #[test]
    fn compile_gt_number() {
        let expr = Expr::BinOp {
            left: Box::new(Expr::Column("age".to_string())),
            op: BinOperator::Gt,
            right: Box::new(Expr::NumberLit(25.0)),
        };
        let compiled = try_compile_predicate(&expr);
        assert!(!compiled.is_interpreted());

        let doc = br#"{"name":"Alice","age":30}"#;
        assert!(compiled.matches(doc));

        let doc2 = br#"{"name":"Bob","age":20}"#;
        assert!(!compiled.matches(doc2));
    }

    #[test]
    fn compile_in_string_set() {
        let expr = Expr::InList {
            expr: Box::new(Expr::Column("status".to_string())),
            list: vec![
                Expr::StringLit("active".to_string()),
                Expr::StringLit("pending".to_string()),
            ],
            negated: false,
        };
        let compiled = try_compile_predicate(&expr);
        assert!(!compiled.is_interpreted());

        let doc = br#"{"status":"active"}"#;
        assert!(compiled.matches(doc));

        let doc2 = br#"{"status":"closed"}"#;
        assert!(!compiled.matches(doc2));
    }
}
