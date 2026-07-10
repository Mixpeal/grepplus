use crate::core::error::Result;
use crate::core::types::ChunkRef;
use crate::index::StoredChunk;
use blake3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchSig {
    pub path: String,
    pub sig: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunkRecord {
    pub chunk_ref: ChunkRef,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMeta {
    pub path: String,
    pub content_hash: String,
    pub chunk_count: usize,
    #[serde(default = "default_temperature")]
    pub temperature: String,
}

fn default_temperature() -> String {
    "cold".into()
}

/// Write per-file index artifacts under `files/<hash>/`.
pub fn write_file_artifacts(
    root: &Path,
    repo: &Path,
    chunks: &[StoredChunk],
    mark_hot: bool,
) -> Result<usize> {
    let files_dir = root.join("files");
    std::fs::create_dir_all(&files_dir)?;

    let mut by_file: HashMap<PathBuf, Vec<&StoredChunk>> = HashMap::new();
    for chunk in chunks {
        by_file
            .entry(chunk.chunk_ref.file.clone())
            .or_default()
            .push(chunk);
    }

    for (file_path, file_chunks) in &by_file {
        let rel = file_path
            .strip_prefix(repo)
            .unwrap_or(file_path)
            .to_string_lossy()
            .replace('\\', "/");
        let hash = path_hash(&rel);
        let dir = files_dir.join(&hash);
        std::fs::create_dir_all(&dir)?;

        let content_hash =
            crate::index::temperature::file_content_hash(file_path).unwrap_or_else(|| {
                file_chunks
                    .first()
                    .map(|c| blake3::hash(c.text.as_bytes()).to_hex().to_string())
                    .unwrap_or_default()
            });

        let records: Vec<StoredChunkRecord> = file_chunks
            .iter()
            .map(|c| StoredChunkRecord {
                chunk_ref: c.chunk_ref.clone(),
                text: c.text.clone(),
            })
            .collect();

        std::fs::write(
            dir.join("chunks.json"),
            serde_json::to_string_pretty(&records)?,
        )?;

        if mark_hot {
            let codes: Vec<_> = file_chunks.iter().map(|c| c.code.clone()).collect();
            std::fs::write(dir.join("vectors.pq4.json"), serde_json::to_string(&codes)?)?;
        }

        let meta = FileMeta {
            path: rel,
            content_hash,
            chunk_count: file_chunks.len(),
            temperature: if mark_hot { "hot" } else { "cold" }.into(),
        };
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    }

    Ok(by_file.len())
}

/// Persist MinHash file signatures for SketchBeam warm-start.
pub fn write_sketch_index(root: &Path, file_sigs: &HashMap<PathBuf, [u32; 64]>) -> Result<()> {
    let sketch_dir = root.join("sketch");
    std::fs::create_dir_all(&sketch_dir)?;
    let entries: Vec<SketchSig> = file_sigs
        .iter()
        .map(|(p, sig)| SketchSig {
            path: p.to_string_lossy().into_owned(),
            sig: sig.to_vec(),
        })
        .collect();
    std::fs::write(
        sketch_dir.join("file_index.json"),
        serde_json::to_string_pretty(&entries)?,
    )?;
    Ok(())
}

pub fn path_hash(relative_path: &str) -> String {
    blake3::hash(relative_path.as_bytes()).to_hex().to_string()[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_hash_stable() {
        assert_eq!(path_hash("src/main.rs"), path_hash("src/main.rs"));
    }
}
