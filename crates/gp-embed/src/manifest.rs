use gp_core::error::{GpError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Onnx,
}

impl ModelFormat {
    pub fn as_str(self) -> &'static str {
        "onnx"
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "onnx" => Some(Self::Onnx),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    Mean,
    Cls,
    Last,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelManifest {
    pub id: String,
    #[serde(default)]
    pub revision: String,
    #[serde(default)]
    pub repo: String,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default = "default_model_file")]
    pub model_file: String,
    #[serde(default = "default_tokenizer")]
    pub tokenizer_file: String,
    #[serde(default)]
    pub extra_files: Vec<String>,
    #[serde(default)]
    pub quant: Option<String>,
    /// Catalog id or repo slug this install belongs to (for grouping variants).
    #[serde(default)]
    pub base_id: String,
    #[serde(default = "default_native_dim")]
    pub native_dim: usize,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default = "default_pool")]
    pub pooling: String,
    #[serde(default)]
    pub max_batch: usize,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub size_mb: u32,
}

fn default_format() -> String {
    "onnx".into()
}

fn default_model_file() -> String {
    "model.onnx".into()
}

fn default_tokenizer() -> String {
    "tokenizer.json".into()
}

fn default_native_dim() -> usize {
    384
}

fn default_max_len() -> usize {
    512
}

fn default_pool() -> String {
    "mean".into()
}

impl ModelManifest {
    pub fn read(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut manifest: ModelManifest = serde_json::from_str(&raw).map_err(GpError::Serde)?;
        let dir = path.parent().unwrap_or(Path::new("."));
        if !dir.join(&manifest.model_file).exists() {
            for candidate in [
                "onnx/model_q4f16.onnx",
                "onnx/model_quantized.onnx",
                "onnx/model.onnx",
                "model.onnx",
            ] {
                if dir.join(candidate).exists() {
                    manifest.model_file = candidate.into();
                    break;
                }
            }
        }
        Ok(manifest)
    }

    pub fn format_kind(&self) -> ModelFormat {
        ModelFormat::parse(&self.format).unwrap_or(ModelFormat::Onnx)
    }

    pub fn pooling_mode(&self) -> Pooling {
        match self.pooling.to_lowercase().as_str() {
            "cls" => Pooling::Cls,
            "last" => Pooling::Last,
            _ => Pooling::Mean,
        }
    }

    pub fn effective_max_batch(&self) -> usize {
        if self.max_batch > 0 {
            return self.max_batch;
        }
        match self.pooling.to_lowercase().as_str() {
            "last" => 1,
            _ => 32,
        }
    }

    pub fn weights_exist(&self, dir: &Path) -> bool {
        if !dir.join(&self.model_file).exists() {
            return false;
        }
        for file in &self.extra_files {
            if !dir.join(file).exists() {
                return false;
            }
        }
        if self.format_kind() == ModelFormat::Onnx
            && !self.tokenizer_file.is_empty()
            && !dir.join(&self.tokenizer_file).exists()
        {
            return false;
        }
        true
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}
