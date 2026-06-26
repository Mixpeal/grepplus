use gp_core::error::{GpError, Result};
use gp_core::traits::ProjectionBackend;
use gp_core::types::ChunkRef;
use gp_pq4::BaselineQ4;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod ann;
mod cache;
mod files;
mod jit;
mod store;
mod temperature;
mod watch;

pub use ann::{load_graph, save_graph, AnnGraph};

pub use cache::{
    corpus_cache_key, index_cache_root, index_path_for, legacy_index_exists, legacy_index_path,
    purge_expired, touch_access, CacheMeta, LEGACY_INDEX_DIR,
};
pub use jit::{candidate_beam, ensure_sketch_shell, sketch_for_repo};
pub use watch::watch_repo;
pub use store::{
    path_hash, write_file_artifacts, write_sketch_index, FileMeta, StoredChunkRecord,
};
pub use temperature::{FileTemperature, TemperatureStats};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManifest {
    pub version: u32,
    pub model_id: String,
    pub projection: String,
    pub dim: usize,
    pub chunk_count: usize,
    pub file_count: usize,
    #[serde(default)]
    pub sketch_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    pub chunk_ref: ChunkRef,
    pub text: String,
    pub code: gp_core::traits::Q4Code,
}

pub struct Index {
    pub root: PathBuf,
    pub manifest: IndexManifest,
    pub chunks: Vec<StoredChunk>,
    pub repo: PathBuf,
}

impl Index {
    pub fn index_path(repo: &Path) -> PathBuf {
        index_path_for(repo)
    }

    pub fn exists(repo: &Path) -> bool {
        Self::index_path(repo).join("manifest.json").exists()
    }

    pub fn purge(repo: &Path) -> Result<()> {
        let root = Self::index_path(repo);
        if root.is_dir() {
            std::fs::remove_dir_all(&root)
                .map_err(|e| GpError::Index(e.to_string()))?;
        }
        Ok(())
    }

    pub fn warn_legacy_index(repo: &Path) {
        if legacy_index_exists(repo) {
            eprintln!(
                "note: legacy in-repo index at {} — indexes now live in {}",
                legacy_index_path(repo).display(),
                index_cache_root().display()
            );
        }
    }

    pub fn open(repo: &Path) -> Result<Self> {
        let root = Self::index_path(repo);
        let manifest_path = root.join("manifest.json");
        if !manifest_path.exists() {
            return Err(GpError::Index(format!(
                "no index at {}",
                root.display()
            )));
        }
        touch_access(&root, repo)?;
        let manifest: IndexManifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
        let chunks_path = root.join("chunks.json");
        let chunks: Vec<StoredChunk> = if chunks_path.exists() && !manifest.sketch_only {
            serde_json::from_str(&std::fs::read_to_string(&chunks_path)?)?
        } else {
            vec![]
        };
        Ok(Self {
            root,
            manifest,
            chunks,
            repo: repo.to_path_buf(),
        })
    }

    pub fn temperature_stats(&self) -> Result<TemperatureStats> {
        files::temperature_stats(&self.root, &self.repo)
    }

    /// Tier-1 shell: sketch + per-file chunk text, no embeddings (all COLD).
    pub fn build_sketch_only(
        repo: &Path,
        model_id: &str,
        dim: usize,
        projection: &str,
    ) -> Result<Self> {
        Self::build_inner(repo, model_id, dim, projection, None, true)
    }

    /// Full warm index: sketch + embed all chunks (all HOT). Same as `build`.
    pub fn build(
        repo: &Path,
        model_id: &str,
        dim: usize,
        projection: &str,
        vectors: Option<&[Vec<f32>]>,
    ) -> Result<Self> {
        Self::build_inner(repo, model_id, dim, projection, vectors, false)
    }

    fn build_inner(
        repo: &Path,
        model_id: &str,
        dim: usize,
        projection: &str,
        vectors: Option<&[Vec<f32>]>,
        force_sketch_only: bool,
    ) -> Result<Self> {
        let root = Self::index_path(repo);
        std::fs::create_dir_all(&root)?;
        touch_access(&root, repo)?;

        let sketch_only = force_sketch_only || vectors.is_none();
        let (backend, stored_projection) = if sketch_only {
            (
                Box::new(BaselineQ4 { proj_dim: dim }) as Box<dyn ProjectionBackend>,
                normalize_projection_name(projection),
            )
        } else {
            build_projection_backend(projection, dim, vectors, &root)?
        };

        let sketch = gp_sketch::SketchBeam::build(vec![repo.to_path_buf()])?;
        let mark_hot = !sketch_only;

        let mut stored = Vec::new();
        for (i, chunk) in sketch.chunks.iter().enumerate() {
            let vec = vectors
                .and_then(|v| v.get(i))
                .cloned()
                .unwrap_or_else(|| vec![0.0; dim]);
            let code = if mark_hot {
                backend.project(&vec)
            } else {
                gp_core::traits::Q4Code {
                    bytes: vec![],
                    dim: 0,
                    scale: 0.0,
                    bias: 0.0,
                }
            };
            stored.push(StoredChunk {
                chunk_ref: chunk.chunk_ref.clone(),
                text: chunk.text.clone(),
                code,
            });
        }

        let manifest = IndexManifest {
            version: 2,
            model_id: model_id.into(),
            projection: stored_projection,
            dim,
            chunk_count: stored.len(),
            file_count: sketch.file_count(),
            sketch_only,
        };

        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;

        if mark_hot {
            std::fs::write(
                root.join("chunks.json"),
                serde_json::to_string_pretty(&stored)?,
            )?;
        } else if root.join("chunks.json").exists() {
            let _ = std::fs::remove_file(root.join("chunks.json"));
        }

        store::write_file_artifacts(&root, repo, &stored, mark_hot)?;
        store::write_sketch_index(&root, sketch.file_sigs())?;

        Ok(Self {
            root,
            manifest,
            chunks: if mark_hot { stored } else { vec![] },
            repo: repo.to_path_buf(),
        })
    }

    /// Load all per-file chunk records from cache (for JIT reheat).
    pub fn chunk_records_map(&self) -> Result<HashMap<String, Vec<StoredChunkRecord>>> {
        let mut map = HashMap::new();
        let files_dir = self.root.join("files");
        if !files_dir.is_dir() {
            return Ok(map);
        }
        for entry in std::fs::read_dir(&files_dir).map_err(|e| GpError::Index(e.to_string()))? {
            let entry = entry.map_err(|e| GpError::Index(e.to_string()))?;
            if !entry.path().is_dir() {
                continue;
            }
            let meta = files::read_file_meta(&entry.path())?;
            let records = files::read_chunk_records(&entry.path())?;
            map.insert(meta.path, records);
        }
        Ok(map)
    }

    /// JIT semantic path: score HOT files, batch-reheat COLD/COOL in candidate beam.
    pub fn jit_semantic_search(
        &self,
        query_vec: &[f32],
        backend: &dyn ProjectionBackend,
        candidates: &[ChunkRef],
        embed_fn: &mut dyn FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
        embed_budget: usize,
        reheat_file_cap: usize,
        cold_first_file_cap: usize,
        cold_first_embed_budget: usize,
        top_k: usize,
    ) -> Result<Vec<(StoredChunk, f32)>> {
        let stats = files::temperature_stats(&self.root, &self.repo)?;
        let index_mostly_cold = stats.hot == 0;
        files::jit_semantic_search(
            &self.root,
            &self.repo,
            query_vec,
            backend,
            candidates,
            embed_fn,
            embed_budget,
            reheat_file_cap,
            cold_first_file_cap,
            cold_first_embed_budget,
            index_mostly_cold,
            top_k,
        )
    }

    pub fn search_semantic(
        &self,
        query_vec: &[f32],
        backend: &dyn ProjectionBackend,
        top_k: usize,
    ) -> Vec<(StoredChunk, f32)> {
        self.search_semantic_filtered(query_vec, backend, None, top_k)
    }

    /// Legacy full scan (monolithic chunks.json) or per-file load when empty.
    pub fn search_semantic_filtered(
        &self,
        query_vec: &[f32],
        backend: &dyn ProjectionBackend,
        candidates: Option<&[ChunkRef]>,
        top_k: usize,
    ) -> Vec<(StoredChunk, f32)> {
        if !self.chunks.is_empty() {
            return self.search_in_memory(query_vec, backend, candidates, top_k);
        }
        self.search_per_file_hot(query_vec, backend, candidates, top_k)
            .unwrap_or_default()
    }

    fn search_in_memory(
        &self,
        query_vec: &[f32],
        backend: &dyn ProjectionBackend,
        candidates: Option<&[ChunkRef]>,
        top_k: usize,
    ) -> Vec<(StoredChunk, f32)> {
        let cand_set: Option<std::collections::HashSet<(String, u32, u32)>> =
            candidates.map(|cs| {
                cs.iter()
                    .map(|c| {
                        (
                            c.file.to_string_lossy().into_owned(),
                            c.start_line,
                            c.end_line,
                        )
                    })
                    .collect()
            });

        let mut scored: Vec<(StoredChunk, f32)> = self
            .chunks
            .iter()
            .filter(|c| {
                cand_set.as_ref().map_or(true, |set| {
                    set.contains(&(
                        c.chunk_ref.file.to_string_lossy().into_owned(),
                        c.chunk_ref.start_line,
                        c.chunk_ref.end_line,
                    ))
                })
            })
            .map(|c| {
                let score = backend.score(query_vec, &c.code);
                (c.clone(), score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        scored
    }

    fn search_per_file_hot(
        &self,
        query_vec: &[f32],
        backend: &dyn ProjectionBackend,
        candidates: Option<&[ChunkRef]>,
        top_k: usize,
    ) -> Result<Vec<(StoredChunk, f32)>> {
        let cand_set: Option<std::collections::HashSet<(String, u32, u32)>> =
            candidates.map(|cs| {
                cs.iter()
                    .map(|c| {
                        (
                            c.file.to_string_lossy().into_owned(),
                            c.start_line,
                            c.end_line,
                        )
                    })
                    .collect()
            });

        let files_dir = self.root.join("files");
        if !files_dir.is_dir() {
            return Ok(vec![]);
        }

        let mut scored = Vec::new();
        for entry in std::fs::read_dir(&files_dir).map_err(|e| GpError::Index(e.to_string()))? {
            let entry = entry.map_err(|e| GpError::Index(e.to_string()))?;
            if !entry.path().is_dir() {
                continue;
            }
            let meta = files::read_file_meta(&entry.path())?;
            if let Some(chunks) = files::load_hot_chunks(&self.root, &self.repo, &meta.path)? {
                for chunk in chunks {
                    let key = (
                        chunk.chunk_ref.file.to_string_lossy().into_owned(),
                        chunk.chunk_ref.start_line,
                        chunk.chunk_ref.end_line,
                    );
                    if cand_set.as_ref().map_or(true, |s| s.contains(&key)) {
                        let score = backend.score(query_vec, &chunk.code);
                        scored.push((chunk, score));
                    }
                }
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        Ok(scored)
    }
}

fn normalize_projection_name(projection: &str) -> String {
    match projection {
        "pq4" | "baseline" => "baseline".into(),
        other => other.into(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredPca {
    matrix: Vec<f32>,
    mean: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

fn build_projection_backend(
    projection: &str,
    dim: usize,
    vectors: Option<&[Vec<f32>]>,
    root: &Path,
) -> Result<(Box<dyn ProjectionBackend>, String)> {
    match projection {
        "pca" => {
            let samples = vectors.ok_or_else(|| {
                GpError::Index("pca projection requires embedded vectors".into())
            })?;
            if samples.is_empty() {
                return Err(GpError::Index("pca projection: no vectors".into()));
            }
            let pca = gp_pq4::PcaQ4::fit(samples, dim);
            let stored = StoredPca {
                matrix: pca.matrix().to_vec(),
                mean: pca.mean().to_vec(),
                out_dim: pca.out_dim(),
                in_dim: pca.in_dim(),
            };
            std::fs::write(
                root.join("pca.json"),
                serde_json::to_string_pretty(&stored)?,
            )?;
            Ok((Box::new(pca), "pca".into()))
        }
        "pq4" => {
            let path = root.join("pq4.json");
            if path.exists() {
                let model = gp_pq4::train::load_learned(&path)?;
                Ok((Box::new(model), "pq4".into()))
            } else {
                Ok((
                    Box::new(BaselineQ4 { proj_dim: dim }),
                    "baseline".into(),
                ))
            }
        }
        "baseline" => Ok((
            Box::new(BaselineQ4 { proj_dim: dim }),
            "baseline".into(),
        )),
        other => Err(GpError::Index(format!("unknown projection: {other}"))),
    }
}

pub fn load_projection_backend(
    projection: &str,
    dim: usize,
    root: &Path,
) -> Result<Box<dyn ProjectionBackend>> {
    match projection {
        "pca" => {
            let path = root.join("pca.json");
            let raw = std::fs::read_to_string(&path).map_err(|_| {
                GpError::Index(format!("missing pca.json at {}", path.display()))
            })?;
            let stored: StoredPca = serde_json::from_str(&raw)?;
            Ok(Box::new(gp_pq4::PcaQ4::from_parts(
                stored.matrix,
                stored.mean,
                stored.out_dim,
                stored.in_dim,
            )))
        }
        "pq4" => {
            let path = root.join("pq4.json");
            if path.exists() {
                Ok(Box::new(gp_pq4::train::load_learned(&path)?))
            } else {
                Ok(Box::new(BaselineQ4 { proj_dim: dim }))
            }
        }
        "baseline" => Ok(Box::new(BaselineQ4 { proj_dim: dim })),
        other => Err(GpError::Index(format!("unknown projection: {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn with_cache_dir<F: FnOnce()>(f: F) {
        crate::cache::with_isolated_cache(f);
    }

    #[test]
    fn build_sketch_shell_leaves_files_cold() {
        with_cache_dir(|| {
            let dir = TempDir::new().unwrap();
            let p = dir.path().join("main.rs");
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "fn main() {{}}").unwrap();

            let idx = Index::build_sketch_only(dir.path(), "test", 8, "baseline").unwrap();
            assert!(idx.manifest.sketch_only);
            let stats = idx.temperature_stats().unwrap();
            assert_eq!(stats.hot, 0);
            assert!(stats.cold >= 1);
        });
    }

    #[test]
    fn build_and_open_index() {
        with_cache_dir(|| {
            let dir = TempDir::new().unwrap();
            let p = dir.path().join("main.rs");
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "fn main() {{}}").unwrap();

            let idx = Index::build(dir.path(), "test-model", 8, "baseline", None).unwrap();
            assert!(idx.manifest.chunk_count >= 1);
            assert!(!idx.root.starts_with(dir.path()));

            let loaded = Index::open(dir.path()).unwrap();
            assert_eq!(loaded.manifest.model_id, "test-model");
        });
    }
}
