use gp_core::traits::{GrepEngine, GrepOptions};
use gp_core::types::{ChunkRef, GrepHit, RetrievalSource, ScoredChunk};
use gp_core::{query, Result};

use crate::{ParallelGrep, RipgrepEngine};

/// Resolved lexical backend.
pub enum GrepBackend {
    Parallel(ParallelGrep),
    Ripgrep(RipgrepEngine),
}

impl GrepBackend {
    pub fn search(&self, pattern: &str, opts: &GrepOptions) -> Result<Vec<GrepHit>> {
        match self {
            Self::Parallel(g) => g.search(pattern, opts),
            Self::Ripgrep(g) => g.search(pattern, opts),
        }
    }
}

/// CLI / default grep: always in-process unless config is `ripgrep`.
pub fn resolve_cli_backend(backend: &str) -> GrepBackend {
    if backend == "ripgrep" {
        if let Ok(rg) = RipgrepEngine::discover() {
            return GrepBackend::Ripgrep(rg);
        }
    }
    GrepBackend::Parallel(ParallelGrep)
}

/// Hybrid exact channel: `ripgrep` or `auto` uses external rg when available.
pub fn resolve_exact_backend(backend: &str) -> GrepBackend {
    match backend {
        "ripgrep" => resolve_cli_backend("ripgrep"),
        "auto" => {
            if let Ok(rg) = RipgrepEngine::discover() {
                GrepBackend::Ripgrep(rg)
            } else {
                GrepBackend::Parallel(ParallelGrep)
            }
        }
        _ => GrepBackend::Parallel(ParallelGrep),
    }
}

pub fn hits_to_chunks(hits: &[GrepHit]) -> Vec<ChunkRef> {
    hits.iter()
        .map(|h| ChunkRef {
            file: h.file.clone(),
            chunk_id: h.line_no,
            start_line: h.line_no,
            end_line: h.line_no,
            byte_start: h.byte_offset,
            byte_end: h.byte_offset + h.line.len() as u64,
        })
        .collect()
}

pub fn exact_pattern(query: &str) -> (String, bool) {
    let stripped = query::strip_quotes(query.trim());
    if query::is_literal_query(query) || query::is_quoted(query.trim()) {
        (stripped.to_string(), true)
    } else {
        (query.to_string(), false)
    }
}

pub fn exact_grep_scored(
    query: &str,
    roots: &[std::path::PathBuf],
    cap: usize,
    backend: &GrepBackend,
) -> Vec<ScoredChunk> {
    let (pattern, fixed) = exact_pattern(query);
    if pattern.is_empty() {
        return vec![];
    }
    let opts = GrepOptions {
        roots: roots.to_vec(),
        fixed_string: fixed,
        max_results: Some(cap),
        ..Default::default()
    };
    let hits = backend.search(&pattern, &opts).unwrap_or_default();
    hits_to_scored(&hits, RetrievalSource::Grep)
}

fn hits_to_scored(hits: &[GrepHit], source: RetrievalSource) -> Vec<ScoredChunk> {
    hits_to_chunks(hits)
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| ScoredChunk {
            chunk,
            score: 1.0 / (i as f32 + 1.0),
            source,
            preview: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn exact_finds_literal() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("auth.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "fn handleSessionRefresh() {{").unwrap();

        let backend = GrepBackend::Parallel(ParallelGrep);
        let hits = exact_grep_scored(
            "handleSessionRefresh",
            &[dir.path().to_path_buf()],
            10,
            &backend,
        );
        assert!(!hits.is_empty());
    }
}
