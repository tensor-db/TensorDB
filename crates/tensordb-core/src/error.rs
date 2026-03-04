use std::num::TryFromIntError;

use thiserror::Error;

/// Stable, categorized error codes for SQL operations.
///
/// Code ranges:
/// - T1xxx: Syntax errors (parsing)
/// - T2xxx: Schema errors (table/column/index/view not found or duplicate)
/// - T3xxx: Constraint violations
/// - T4xxx: Execution errors (runtime failures)
/// - T6xxx: Auth/permission errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // Syntax (T1xxx)
    SyntaxError = 1001,
    UnexpectedToken = 1002,
    UnterminatedString = 1003,
    UnknownKeyword = 1004,
    TrailingTokens = 1005,
    // Schema (T2xxx)
    TableNotFound = 2001,
    ColumnNotFound = 2002,
    TableAlreadyExists = 2003,
    IndexNotFound = 2004,
    IndexAlreadyExists = 2005,
    ViewNotFound = 2006,
    TypeMismatch = 2007,
    // Constraint (T3xxx)
    NotNullViolation = 3001,
    UniqueViolation = 3002,
    PkViolation = 3003,
    // Execution (T4xxx)
    DivisionByZero = 4001,
    InvalidCast = 4002,
    QueryTimeout = 4003,
    MemoryLimitExceeded = 4004,
    ExecutionError = 4005,
    // Auth (T6xxx)
    PermissionDenied = 6001,
    AuthRequired = 6002,
}

impl ErrorCode {
    pub fn code_str(&self) -> String {
        format!("T{}", *self as u16)
    }

    pub fn category(&self) -> &'static str {
        let code = *self as u16;
        match code {
            1001..=1999 => "Syntax",
            2001..=2999 => "Schema",
            3001..=3999 => "Constraint",
            4001..=4999 => "Execution",
            6001..=6999 => "Auth",
            _ => "Unknown",
        }
    }
}

/// Structured SQL error with stable code, message, optional suggestion, and position.
#[derive(Debug, Clone)]
pub struct SqlError {
    pub code: ErrorCode,
    pub message: String,
    pub suggestion: Option<String>,
    pub position: Option<usize>,
}

impl std::fmt::Display for SqlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.code_str(), self.message)?;
        if let Some(ref hint) = self.suggestion {
            write!(f, "\n  Hint: {}", hint)?;
        }
        Ok(())
    }
}

impl SqlError {
    pub fn syntax(msg: impl Into<String>, position: Option<usize>) -> Self {
        Self {
            code: ErrorCode::SyntaxError,
            message: msg.into(),
            suggestion: None,
            position,
        }
    }

    pub fn table_not_found(name: &str, suggestion: Option<String>) -> Self {
        Self {
            code: ErrorCode::TableNotFound,
            message: format!("table {name} does not exist"),
            suggestion,
            position: None,
        }
    }

    pub fn column_not_found(col: &str, table: &str, suggestion: Option<String>) -> Self {
        Self {
            code: ErrorCode::ColumnNotFound,
            message: format!("column {col} does not exist on table {table}"),
            suggestion,
            position: None,
        }
    }

    pub fn table_already_exists(name: &str) -> Self {
        Self {
            code: ErrorCode::TableAlreadyExists,
            message: format!("table {name} already exists"),
            suggestion: None,
            position: None,
        }
    }

    pub fn index_not_found(name: &str) -> Self {
        Self {
            code: ErrorCode::IndexNotFound,
            message: format!("index {name} does not exist"),
            suggestion: None,
            position: None,
        }
    }

    pub fn index_already_exists(name: &str) -> Self {
        Self {
            code: ErrorCode::IndexAlreadyExists,
            message: format!("index {name} already exists"),
            suggestion: None,
            position: None,
        }
    }

    pub fn view_not_found(name: &str) -> Self {
        Self {
            code: ErrorCode::ViewNotFound,
            message: format!("view {name} does not exist"),
            suggestion: None,
            position: None,
        }
    }

    pub fn type_mismatch(msg: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::TypeMismatch,
            message: msg.into(),
            suggestion: None,
            position: None,
        }
    }

    pub fn not_null_violation(column: &str) -> Self {
        Self {
            code: ErrorCode::NotNullViolation,
            message: format!("NOT NULL constraint violated for column {column}"),
            suggestion: None,
            position: None,
        }
    }

    pub fn unique_violation(index: &str, value: &str) -> Self {
        Self {
            code: ErrorCode::UniqueViolation,
            message: format!(
                "unique constraint violated for index {index}: duplicate value '{value}'"
            ),
            suggestion: None,
            position: None,
        }
    }

    pub fn pk_violation(table: &str, pk: &str) -> Self {
        Self {
            code: ErrorCode::PkViolation,
            message: format!("primary key '{pk}' already exists in table {table}"),
            suggestion: None,
            position: None,
        }
    }

    pub fn query_timeout(elapsed_ms: u64, limit_ms: u64) -> Self {
        Self {
            code: ErrorCode::QueryTimeout,
            message: format!("query timed out after {elapsed_ms}ms (limit: {limit_ms}ms)"),
            suggestion: Some("Consider adding indexes or reducing result set size".to_string()),
            position: None,
        }
    }

    pub fn memory_limit(used: u64, limit: u64) -> Self {
        Self {
            code: ErrorCode::MemoryLimitExceeded,
            message: format!("query exceeded memory limit: {used} bytes used, {limit} byte limit"),
            suggestion: Some("Consider adding LIMIT clause or reducing result set".to_string()),
            position: None,
        }
    }

    pub fn permission_denied(msg: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::PermissionDenied,
            message: msg.into(),
            suggestion: None,
            position: None,
        }
    }

    pub fn execution(msg: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::ExecutionError,
            message: msg.into(),
            suggestion: None,
            position: None,
        }
    }
}

// Levenshtein distance for "Did you mean?" suggestions
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let mut prev = (0..=b.len()).collect::<Vec<_>>();
    let mut curr = vec![0; b.len() + 1];

    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca.eq_ignore_ascii_case(&cb) { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b.len()]
}

pub fn suggest_closest(input: &str, candidates: &[&str], max_distance: usize) -> Option<String> {
    candidates
        .iter()
        .map(|c| (*c, levenshtein(input, c)))
        .filter(|(_, d)| *d > 0 && *d <= max_distance)
        .min_by_key(|(_, d)| *d)
        .map(|(c, _)| c.to_string())
}

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("utf8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("serde json error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("int conversion error: {0}")]
    IntConv(#[from] TryFromIntError),
    #[error("invalid varint")]
    InvalidVarint,
    #[error("wal crc mismatch")]
    WalCrcMismatch,
    #[error("wal magic mismatch")]
    WalMagicMismatch,
    #[error("sstable format error: {0}")]
    SstableFormat(String),
    #[error("manifest format error: {0}")]
    ManifestFormat(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("sql parse error: {0}")]
    SqlParse(SqlError),
    #[error("sql execution error: {0}")]
    SqlExec(SqlError),
    #[error("not found")]
    NotFound,
    #[error("channel closed")]
    ChannelClosed,
    #[error("feature not enabled: {0}")]
    FeatureNotEnabled(String),
    #[error("LLM not available (no model loaded)")]
    LlmNotAvailable,
    #[error("LLM error: {0}")]
    LlmError(String),
    #[error("vector error: {0}")]
    VectorError(String),
}

/// Helper to create a `TensorError::SqlParse` with a default SyntaxError code.
pub fn sql_parse_err(msg: impl Into<String>) -> TensorError {
    TensorError::SqlParse(SqlError {
        code: ErrorCode::SyntaxError,
        message: msg.into(),
        suggestion: None,
        position: None,
    })
}

/// Helper to create a `TensorError::SqlExec` with a default ExecutionError code.
pub fn sql_exec_err(msg: impl Into<String>) -> TensorError {
    TensorError::SqlExec(SqlError {
        code: ErrorCode::ExecutionError,
        message: msg.into(),
        suggestion: None,
        position: None,
    })
}

pub type Result<T> = std::result::Result<T, TensorError>;
