use thiserror::Error;

pub type Result<T> = std::result::Result<T, GpError>;

#[derive(Error, Debug)]
pub enum GpError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("no embedding model installed")]
    NoModel,
    #[error("index not found or stale: {0}")]
    Index(String),
    #[error("model runtime error: {0}")]
    Model(String),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("training error: {0}")]
    Training(String),
    #[error("{0}")]
    Other(String),
}
