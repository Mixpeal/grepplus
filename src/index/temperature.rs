use crate::index::store::FileMeta;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileTemperature {
    /// Never embedded or no vectors on disk.
    Cold,
    /// Embedded and content hash matches source file.
    Hot,
    /// Embedded but source file changed since last embed.
    Cool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TemperatureStats {
    pub hot: usize,
    pub cold: usize,
    pub cool: usize,
    pub total_files: usize,
    pub embedded_chunks: usize,
}

pub fn file_content_hash(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(blake3::hash(&bytes).to_hex().to_string())
}

/// Resolve live temperature from on-disk meta + source file.
pub fn resolve_temperature(meta: &FileMeta, source: &Path, has_vectors: bool) -> FileTemperature {
    if !has_vectors {
        return FileTemperature::Cold;
    }
    match file_content_hash(source) {
        Some(h) if h == meta.content_hash => FileTemperature::Hot,
        Some(_) => FileTemperature::Cool,
        None => FileTemperature::Cool,
    }
}

impl FileTemperature {
    pub fn as_str(self) -> &'static str {
        match self {
            FileTemperature::Cold => "cold",
            FileTemperature::Hot => "hot",
            FileTemperature::Cool => "cool",
        }
    }
}
