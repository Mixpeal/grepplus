mod bm25;
mod minhash;

use bm25::Bm25;
use gp_chunk::{chunk_file, Chunk, ChunkConfig};
use gp_core::error::{GpError, Result};
use gp_core::traits::PreFocus;
use gp_core::types::ChunkRef;
use ignore::WalkBuilder;
use minhash::{jaccard, MinHasher};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct SketchBeam {
    pub roots: Vec<PathBuf>,
    pub chunk_cfg: ChunkConfig,
    pub chunks: Vec<Chunk>,
    file_sigs: HashMap<PathBuf, [u32; minhash::NUM_HASHES]>,
    bm25: Bm25,
    minhash: MinHasher,
}

impl SketchBeam {
    pub fn file_count(&self) -> usize {
        self.file_sigs.len()
    }

    pub fn file_sigs(&self) -> &HashMap<PathBuf, [u32; minhash::NUM_HASHES]> {
        &self.file_sigs
    }
}

impl SketchBeam {
    pub fn build(roots: Vec<PathBuf>) -> Result<Self> {
        let chunk_cfg = ChunkConfig::default();
        let minhash = MinHasher::new();
        let mut chunks = Vec::new();
        let mut file_sigs: HashMap<PathBuf, [u32; minhash::NUM_HASHES]> = HashMap::new();

        let root = roots.first().cloned().unwrap_or_else(|| PathBuf::from("."));
        let walker = WalkBuilder::new(&root).standard_filters(true).build();
        for entry in walker.flatten() {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            if is_binary(path) {
                continue;
            }
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let file_chunks = chunk_file(path, &content, &chunk_cfg);
            if file_chunks.is_empty() {
                continue;
            }
            let combined: String = file_chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
            file_sigs.insert(path.to_path_buf(), minhash.signature(&combined));
            chunks.extend(file_chunks);
        }

        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let bm25 = Bm25::build(&texts);

        Ok(Self {
            roots,
            chunk_cfg,
            chunks,
            file_sigs,
            bm25,
            minhash,
        })
    }

    /// Rebuild sketch structures from a persisted index (no repo walk).
    pub fn load_from_index(index_root: &Path, repo: &Path) -> Result<Self> {
        #[derive(Deserialize)]
        struct SigEntry {
            path: String,
            sig: Vec<u32>,
        }
        #[derive(Deserialize)]
        struct ChunkRecord {
            chunk_ref: ChunkRef,
            text: String,
        }

        let sketch_path = index_root.join("sketch").join("file_index.json");
        let raw = std::fs::read_to_string(&sketch_path)
            .map_err(|e| GpError::Index(format!("{sketch_path:?}: {e}")))?;
        let entries: Vec<SigEntry> = serde_json::from_str(&raw)?;

        let mut file_sigs: HashMap<PathBuf, [u32; minhash::NUM_HASHES]> = HashMap::new();
        for e in entries {
            let mut sig = [0u32; minhash::NUM_HASHES];
            for (i, v) in e.sig.iter().take(minhash::NUM_HASHES).enumerate() {
                sig[i] = *v;
            }
            file_sigs.insert(PathBuf::from(e.path), sig);
        }

        let files_dir = index_root.join("files");
        let mut chunks = Vec::new();
        if files_dir.is_dir() {
            for entry in std::fs::read_dir(&files_dir)
                .map_err(|e| GpError::Index(e.to_string()))?
                .flatten()
            {
                let chunks_path = entry.path().join("chunks.json");
                if !chunks_path.exists() {
                    continue;
                }
                let raw = std::fs::read_to_string(&chunks_path)
                    .map_err(|e| GpError::Index(e.to_string()))?;
                let records: Vec<ChunkRecord> = serde_json::from_str(&raw)?;
                for r in records {
                    chunks.push(Chunk {
                        chunk_ref: r.chunk_ref,
                        text: r.text,
                    });
                }
            }
        }

        if chunks.is_empty() {
            return Self::build(vec![repo.to_path_buf()]);
        }

        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let bm25 = Bm25::build(&texts);
        Ok(Self {
            roots: vec![repo.to_path_buf()],
            chunk_cfg: ChunkConfig::default(),
            chunks,
            file_sigs,
            bm25,
            minhash: MinHasher::new(),
        })
    }
}

impl PreFocus for SketchBeam {
    fn sketch_beam(&self, query: &str, beam_width: usize, cap: usize) -> Result<Vec<ChunkRef>> {
        let qsig = self.minhash.signature(query);

        let mut file_scores: Vec<(PathBuf, f32)> = self
            .file_sigs
            .iter()
            .map(|(path, sig)| (path.clone(), jaccard(&qsig, sig)))
            .collect();
        file_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        file_scores.truncate(beam_width);

        let candidate_files: std::collections::HashSet<_> =
            file_scores.iter().map(|(p, _)| p.clone()).collect();

        let candidate_ids: Vec<u32> = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| candidate_files.contains(&c.chunk_ref.file))
            .map(|(i, _)| i as u32)
            .collect();

        let bm25_hits = self.bm25.search(query, cap, Some(&candidate_ids));
        let out: Vec<ChunkRef> = bm25_hits
            .into_iter()
            .filter_map(|(id, _)| self.chunks.get(id as usize).map(|c| c.chunk_ref.clone()))
            .collect();
        Ok(out)
    }
}

fn is_binary(path: &Path) -> bool {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return true,
    };
    bytes.iter().take(8192).any(|&b| b == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn sketch_beam_finds_paraphrase() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("resilience.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "// graceful degradation when upstream fails").unwrap();
        writeln!(f, "fn circuit_breaker() {{").unwrap();
        writeln!(f, "    fallback_handler();").unwrap();
        writeln!(f, "}}").unwrap();

        let beam = SketchBeam::build(vec![dir.path().to_path_buf()]).unwrap();
        let chunks = beam
            .sketch_beam("graceful degradation when upstream fails", 10, 5)
            .unwrap();
        assert!(!chunks.is_empty());
    }
}
