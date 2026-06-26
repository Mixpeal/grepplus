use gp_core::error::{GpError, Result};
use gp_core::config::Config;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const CATALOG_JSON: &str = include_str!("../../../models/catalog.json");

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogEntry {
    pub id: String,
    pub repo: String,
    pub revision: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub sha256_model: String,
    #[serde(default)]
    pub sha256_tokenizer: String,
    pub model_file: String,
    #[serde(default = "default_tokenizer")]
    pub tokenizer_file: String,
    #[serde(default)]
    pub extra_files: Vec<String>,
    #[serde(default = "default_format")]
    pub format: String,
    pub native_dim: usize,
    pub max_len: usize,
    #[serde(default = "default_pool")]
    pub pooling: String,
    #[serde(default)]
    pub max_batch: usize,
    pub size_mb: u32,
    pub description: String,
}

fn default_tokenizer() -> String {
    "tokenizer.json".into()
}

fn default_format() -> String {
    "onnx".into()
}

fn default_pool() -> String {
    "mean".into()
}

impl CatalogEntry {
    pub fn sha256_for(&self, file: &str) -> Option<&str> {
        if file == self.model_file {
            if !self.sha256_model.is_empty() {
                return Some(&self.sha256_model);
            }
        } else if file == self.tokenizer_file {
            if !self.sha256_tokenizer.is_empty() {
                return Some(&self.sha256_tokenizer);
            }
        }
        if !self.sha256.is_empty() {
            return Some(&self.sha256);
        }
        None
    }

    pub fn format_kind(&self) -> crate::manifest::ModelFormat {
        crate::manifest::ModelFormat::parse(&self.format)
            .unwrap_or(crate::manifest::ModelFormat::Onnx)
    }

    pub fn to_manifest(&self) -> crate::manifest::ModelManifest {
        crate::manifest::ModelManifest {
            id: self.id.clone(),
            revision: self.revision.clone(),
            repo: self.repo.clone(),
            format: self.format.clone(),
            sha256: self.sha256_model.clone(),
            model_file: self.model_file.clone(),
            tokenizer_file: self.tokenizer_file.clone(),
            extra_files: self.extra_files.clone(),
            quant: None,
            base_id: self.id.clone(),
            native_dim: self.native_dim,
            max_len: self.max_len,
            pooling: self.pooling.clone(),
            max_batch: self.max_batch,
            description: self.description.clone(),
            size_mb: self.size_mb,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Catalog {
    pub models: Vec<CatalogEntry>,
}

impl Catalog {
    pub fn builtin() -> Result<Self> {
        serde_json::from_str(CATALOG_JSON).map_err(GpError::Serde)
    }

    pub fn user_catalog_path() -> PathBuf {
        Config::global_config_dir().join("catalog.json")
    }

    pub fn load_user() -> Result<Self> {
        let path = Self::user_catalog_path();
        if !path.exists() {
            return Ok(Self { models: vec![] });
        }
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(GpError::Serde)
    }

    pub fn merged() -> Result<Self> {
        let mut builtin = Self::builtin()?;
        let user = Self::load_user()?;
        for entry in user.models {
            if let Some(existing) = builtin.models.iter_mut().find(|m| m.id == entry.id) {
                *existing = entry;
            } else {
                builtin.models.push(entry);
            }
        }
        Ok(builtin)
    }

    pub fn save_user_entry(entry: &CatalogEntry) -> Result<()> {
        let mut user = Self::load_user()?;
        if let Some(existing) = user.models.iter_mut().find(|m| m.id == entry.id) {
            *existing = entry.clone();
        } else {
            user.models.push(entry.clone());
        }
        let dir = Config::global_config_dir();
        std::fs::create_dir_all(&dir)?;
        let path = Self::user_catalog_path();
        std::fs::write(path, serde_json::to_string_pretty(&user)?)?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&CatalogEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn list(&self) -> &[CatalogEntry] {
        &self.models
    }

    pub fn default_id(&self) -> Option<&str> {
        self.models.first().map(|m| m.id.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_loads() {
        let cat = Catalog::builtin().unwrap();
        assert!(!cat.models.is_empty());
        assert_eq!(cat.default_id(), Some("qwen3-embedding-0.6b"));
    }
}
