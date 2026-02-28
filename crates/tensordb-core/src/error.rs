use std::num::TryFromIntError;

use thiserror::Error;

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
    SqlParse(String),
    #[error("sql execution error: {0}")]
    SqlExec(String),
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
}

pub type Result<T> = std::result::Result<T, TensorError>;
