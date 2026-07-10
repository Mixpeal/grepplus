use crate::core::error::{GpError, Result};
use crate::core::types::ChunkRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod ann;
mod cache;
mod files;
mod jit;
mod store;
mod temperature;
mod vectors;
mod watch;

pub use vectors::VectorCodec;

pub use ann::{load_graph, save_graph, AnnGraph};

#[cfg(test)]
pub use cache::with_isolated_cache;
pub use cache::{
    corpus_cache_key, index_cache_root, index_path_for, legacy_index_exists, legacy_index_path,
    purge_expired, touch_access, CacheMeta, LEGACY_INDEX_DIR,
};
pub use jit::{candidate_beam, candidate_beam_mode, ensure_sketch_shell, sketch_for_repo};
pub use store::{path_hash, write_file_artifacts, write_sketch_index, FileMeta, StoredChunkRecord};
pub use temperature::{FileTemperature, TemperatureStats};
pub use watch::watch_repo;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManifest {
    pub version: u32,
    pub model_id: String,
    /// Legacy field; always `baseline` for indexes built after PQ4 removal.
    #[serde(default = "default_projection")]
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
    pub code: crate::core::traits::Q4Code,
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
            std::fs::remove_dir_all(&root).map_err(|e| GpError::Index(e.to_string()))?;
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
            return Err(GpError::Index(format!("no index at {}", root.display())));
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
    pub fn build_sketch_only(repo: &Path, model_id: &str, dim: usize) -> Result<Self> {
        Self::build_sketch_only_with_options(
            repo,
            model_id,
            dim,
            crate::chunk::ChunkConfig::default(),
            &[],
        )
    }

    pub fn build_sketch_only_with_options(
        repo: &Path,
        model_id: &str,
        dim: usize,
        chunk_cfg: crate::chunk::ChunkConfig,
        exclude: &[String],
    ) -> Result<Self> {
        let sketch = crate::sketch::SketchBeam::build_with_options(
            vec![repo.to_path_buf()],
            chunk_cfg,
            exclude,
        )?;
        Self::finish_build(repo, model_id, dim, None, true, false, 500, sketch)
    }

    /// Full warm index: sketch + embed all chunks (all HOT). Same as `build`.
    pub fn build(
        repo: &Path,
        model_id: &str,
        dim: usize,
        vectors: Option<&[Vec<f32>]>,
    ) -> Result<Self> {
        let sketch = crate::sketch::SketchBeam::build(vec![repo.to_path_buf()])?;
        Self::finish_build(repo, model_id, dim, vectors, false, true, 500, sketch)
    }

    pub fn build_with_options(
        repo: &Path,
        model_id: &str,
        dim: usize,
        vectors: Option<&[Vec<f32>]>,
        ann_enabled: bool,
        ann_min_chunks: usize,
        sketch: crate::sketch::SketchBeam,
    ) -> Result<Self> {
        Self::finish_build(
            repo,
            model_id,
            dim,
            vectors,
            false,
            ann_enabled,
            ann_min_chunks,
            sketch,
        )
    }

    fn finish_build(
        repo: &Path,
        model_id: &str,
        dim: usize,
        vectors: Option<&[Vec<f32>]>,
        force_sketch_only: bool,
        ann_enabled: bool,
        ann_min_chunks: usize,
        sketch: crate::sketch::SketchBeam,
    ) -> Result<Self> {
        let root = Self::index_path(repo);
        std::fs::create_dir_all(&root)?;
        touch_access(&root, repo)?;

        let sketch_only = force_sketch_only || vectors.is_none();
        let codec = VectorCodec::new(dim);
        let mark_hot = !sketch_only;

        let mut stored = Vec::new();
        for (i, chunk) in sketch.chunks.iter().enumerate() {
            let vec = vectors
                .and_then(|v| v.get(i))
                .cloned()
                .unwrap_or_else(|| vec![0.0; dim]);
            let code = if mark_hot {
                codec.project(&vec)
            } else {
                crate::core::traits::Q4Code {
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
            projection: default_projection(),
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

        if mark_hot && ann_enabled && stored.len() >= ann_min_chunks {
            let codes: Vec<crate::core::traits::Q4Code> =
                stored.iter().map(|c| c.code.clone()).collect();
            let samples: Vec<Vec<f32>> = vectors
                .map(|v| v.to_vec())
                .unwrap_or_else(|| vec![vec![0.0; dim]; stored.len()]);
            let graph = ann::AnnGraph::build(&codes, &codec, &samples);
            let _ = ann::save_graph(&root.join("ann.json"), &graph);
        }

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
        codec: &VectorCodec,
        candidates: &[ChunkRef],
        embed_fn: &mut dyn FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
        embed_budget: usize,
        reheat_file_cap: usize,
        cold_first_file_cap: usize,
        cold_first_embed_budget: usize,
        top_k: usize,
        embed_dim: usize,
        embed_stats: Option<&crate::core::embed_stats::EmbedStatsCell>,
    ) -> Result<Vec<(StoredChunk, f32)>> {
        let stats = files::temperature_stats(&self.root, &self.repo)?;
        let index_mostly_cold = stats.hot == 0;
        files::jit_semantic_search(
            &self.root,
            &self.repo,
            query_vec,
            codec,
            candidates,
            embed_fn,
            embed_budget,
            reheat_file_cap,
            cold_first_file_cap,
            cold_first_embed_budget,
            index_mostly_cold,
            top_k,
            embed_dim,
            embed_stats,
        )
    }

    pub fn search_semantic(
        &self,
        query_vec: &[f32],
        codec: &VectorCodec,
        top_k: usize,
    ) -> Vec<(StoredChunk, f32)> {
        self.search_semantic_filtered(query_vec, codec, None, top_k)
    }

    /// Legacy full scan (monolithic chunks.json) or per-file load when empty.
    pub fn search_semantic_filtered(
        &self,
        query_vec: &[f32],
        codec: &VectorCodec,
        candidates: Option<&[ChunkRef]>,
        top_k: usize,
    ) -> Vec<(StoredChunk, f32)> {
        if !self.chunks.is_empty() {
            return self.search_in_memory(query_vec, codec, candidates, top_k);
        }
        self.search_per_file_hot(query_vec, codec, candidates, top_k)
            .unwrap_or_default()
    }

    /// Filtered semantic search, optionally expanded via ANN graph shortlist.
    pub fn search_semantic_with_ann(
        &self,
        query_vec: &[f32],
        codec: &VectorCodec,
        candidates: Option<&[ChunkRef]>,
        top_k: usize,
        ann_enabled: bool,
    ) -> Vec<(StoredChunk, f32)> {
        let mut base = self.search_semantic_filtered(query_vec, codec, candidates, top_k);
        if !ann_enabled || self.chunks.is_empty() {
            return base;
        }
        let ann_path = self.root.join("ann.json");
        let graph = match load_graph(&ann_path) {
            Ok(g) => g,
            Err(_) => {
                tracing::debug!("ann enabled but ann.json missing at {}", ann_path.display());
                return base;
            }
        };
        let codes: Vec<_> = self.chunks.iter().map(|c| c.code.clone()).collect();
        let ann_hits = graph.search(query_vec, &codes, codec, top_k);
        for (idx, score) in ann_hits {
            if let Some(chunk) = self.chunks.get(idx) {
                let key = (
                    chunk.chunk_ref.file.to_string_lossy().into_owned(),
                    chunk.chunk_ref.start_line,
                    chunk.chunk_ref.end_line,
                );
                if let Some((_, existing)) = base.iter_mut().find(|(c, _)| {
                    (
                        c.chunk_ref.file.to_string_lossy().into_owned(),
                        c.chunk_ref.start_line,
                        c.chunk_ref.end_line,
                    ) == key
                }) {
                    if score > *existing {
                        *existing = score;
                    }
                } else {
                    base.push((chunk.clone(), score));
                }
            }
        }
        base.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        base.truncate(top_k);
        base
    }

    fn search_in_memory(
        &self,
        query_vec: &[f32],
        codec: &VectorCodec,
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
                cand_set.as_ref().is_none_or(|set| {
                    set.contains(&(
                        c.chunk_ref.file.to_string_lossy().into_owned(),
                        c.chunk_ref.start_line,
                        c.chunk_ref.end_line,
                    ))
                })
            })
            .map(|c| {
                let score = codec.score(query_vec, &c.code);
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
        codec: &VectorCodec,
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
                    if cand_set.as_ref().is_none_or(|s| s.contains(&key)) {
                        let score = codec.score(query_vec, &chunk.code);
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

fn default_projection() -> String {
    "baseline".into()
}

/// Vector codec for the index embedding dimension.
pub fn vector_codec(dim: usize) -> VectorCodec {
    VectorCodec::new(dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn with_cache_dir<F: FnOnce()>(f: F) {
        crate::index::cache::with_isolated_cache(f);
    }

    #[test]
    fn build_sketch_shell_leaves_files_cold() {
        with_cache_dir(|| {
            let dir = TempDir::new().unwrap();
            let p = dir.path().join("main.rs");
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "fn main() {{}}").unwrap();

            let idx = Index::build_sketch_only(dir.path(), "test", 8).unwrap();
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

            let idx = Index::build(dir.path(), "test-model", 8, None).unwrap();
            assert!(idx.manifest.chunk_count >= 1);
            assert!(!idx.root.starts_with(dir.path()));

            let loaded = Index::open(dir.path()).unwrap();
            assert_eq!(loaded.manifest.model_id, "test-model");
        });
    }

    #[test]
    fn ann_graph_used_in_semantic_search() {
        with_cache_dir(|| {
            let dir = TempDir::new().unwrap();
            for i in 0..12 {
                let p = dir.path().join(format!("f{i}.rs"));
                let mut f = std::fs::File::create(&p).unwrap();
                writeln!(f, "fn item_{i}() {{ let x = {i}; }}").unwrap();
            }
            let dim = 8;
            let sketch = crate::sketch::SketchBeam::build(vec![dir.path().to_path_buf()]).unwrap();
            let vectors: Vec<Vec<f32>> = sketch
                .chunks
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut v = vec![0.0; dim];
                    v[i % dim] = 1.0;
                    v
                })
                .collect();
            let idx = Index::build_with_options(
                dir.path(),
                "test-ann",
                dim,
                Some(&vectors),
                true,
                1, // force ANN even for small indexes
                sketch,
            )
            .unwrap();
            assert!(idx.root.join("ann.json").exists());
            let codec = VectorCodec::new(dim);
            let q = vectors[0].clone();
            let hits = idx.search_semantic_with_ann(&q, &codec, None, 5, true);
            assert!(!hits.is_empty());
        });
    }
}
