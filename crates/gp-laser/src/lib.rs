mod expand;

use expand::Expander;
use gp_chunk::{chunk_file, ChunkConfig};
use gp_core::error::Result;
use gp_core::traits::{GrepEngine, GrepOptions, LaserFocus};
use gp_core::types::{ChunkRef, GrepHit};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

pub struct Laser<G: GrepEngine> {
    pub grep: G,
    pub expander: Expander,
    pub roots: Vec<PathBuf>,
    pub chunk_cfg: ChunkConfig,
}

impl<G: GrepEngine> Laser<G> {
    pub fn new(grep: G, roots: Vec<PathBuf>) -> Self {
        Self {
            grep,
            expander: Expander::builtin(),
            roots,
            chunk_cfg: ChunkConfig::default(),
        }
    }
}

impl<G: GrepEngine> LaserFocus for Laser<G> {
    fn focus(&self, query: &str, cap: usize) -> Result<Vec<ChunkRef>> {
        let terms = self.expander.expand(query);
        if terms.is_empty() {
            return Ok(vec![]);
        }
        let alt = terms
            .iter()
            .map(|t| regex::escape(t))
            .collect::<Vec<_>>()
            .join("|");
        let pattern = format!(r"(?i)\b({alt})\b");
        let opts = GrepOptions {
            roots: self.roots.clone(),
            max_results: Some(cap * 4),
            ..Default::default()
        };
        let hits = self.grep.search(&pattern, &opts)?;
        let mut chunks = map_hits_to_chunks(&hits, &self.chunk_cfg);
        chunks.truncate(cap);
        Ok(chunks)
    }
}

fn map_hits_to_chunks(hits: &[GrepHit], cfg: &ChunkConfig) -> Vec<ChunkRef> {
    let mut by_file: HashMap<PathBuf, Vec<&GrepHit>> = HashMap::new();
    for hit in hits {
        by_file.entry(hit.file.clone()).or_default().push(hit);
    }

    let mut scored: BTreeMap<(String, u32, u32), (ChunkRef, usize)> = BTreeMap::new();

    for (path, file_hits) in by_file {
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let chunks = chunk_file(&path, &content, cfg);
        for hit in file_hits {
            for chunk in &chunks {
                if hit.line_no >= chunk.chunk_ref.start_line
                    && hit.line_no <= chunk.chunk_ref.end_line
                {
                    let key = (
                        path.to_string_lossy().into_owned(),
                        chunk.chunk_ref.start_line,
                        chunk.chunk_ref.end_line,
                    );
                    scored
                        .entry(key)
                        .and_modify(|(_, count)| *count += 1)
                        .or_insert((chunk.chunk_ref.clone(), 1));
                }
            }
        }
    }

    let mut out: Vec<(ChunkRef, usize)> = scored.into_values().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out.into_iter().map(|(c, _)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_grep::ParallelGrep;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn laser_finds_chunk_for_query() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("auth.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "fn login() {{").unwrap();
        writeln!(f, "    validate_token();").unwrap();
        writeln!(f, "}}").unwrap();

        let laser = Laser::new(ParallelGrep, vec![dir.path().to_path_buf()]);
        let chunks = laser.focus("auth login", 10).unwrap();
        assert!(!chunks.is_empty());
    }
}
