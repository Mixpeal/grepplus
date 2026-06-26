use gp_core::traits::Fusion;
use gp_core::types::{ChunkRef, RetrievalSource, ScoredChunk};
use std::collections::{HashMap, HashSet};

pub struct RrfFusion {
    pub k: f32,
}

/// Score floor for verbatim exact-match hits pinned at rank 0.
pub const EXACT_PIN_SCORE: f32 = 1_000.0;

impl Default for RrfFusion {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

fn chunk_key(c: &ChunkRef) -> (String, u32, u32) {
    (
        c.file.to_string_lossy().into_owned(),
        c.start_line,
        c.end_line,
    )
}

impl RrfFusion {
    /// Fuse expanded laser hits with verbatim exact-match hits (dual lexical channel).
    pub fn fuse_lexical(&self, exact: Vec<ScoredChunk>, laser: Vec<ScoredChunk>) -> Vec<ScoredChunk> {
        let mut exact = exact;
        // Exact channel gets 2x RRF weight via duplicate rank-0 pass.
        let boosted_exact: Vec<ScoredChunk> = exact
            .iter()
            .enumerate()
            .map(|(i, hit)| ScoredChunk {
                chunk: hit.chunk.clone(),
                score: 2.0 / (self.k + i as f32 + 1.0),
                source: hit.source,
                preview: hit.preview.clone(),
            })
            .collect();
        exact.extend(boosted_exact);
        self.fuse(exact, laser)
    }

    /// RRF lexical + semantic, then pin exact-match chunks at the top.
    pub fn fuse_hybrid(
        &self,
        lexical: Vec<ScoredChunk>,
        semantic: Vec<ScoredChunk>,
        exact_pins: Vec<ScoredChunk>,
    ) -> Vec<ScoredChunk> {
        let pins = exact_pins.clone();
        let fused = if lexical.is_empty() {
            if semantic.is_empty() {
                exact_pins
            } else if exact_pins.is_empty() {
                semantic
            } else {
                self.fuse(exact_pins, semantic)
            }
        } else if semantic.is_empty() {
            if exact_pins.is_empty() {
                lexical
            } else {
                self.fuse_lexical(exact_pins, lexical)
            }
        } else {
            let inner = self.fuse(lexical, semantic);
            if exact_pins.is_empty() {
                inner
            } else {
                self.fuse(exact_pins, inner)
            }
        };

        pin_exact_hits(fused, &pins)
    }
}

impl Fusion for RrfFusion {
    fn fuse(&self, grep: Vec<ScoredChunk>, semantic: Vec<ScoredChunk>) -> Vec<ScoredChunk> {
        let mut scores: HashMap<(String, u32, u32), ScoredChunk> = HashMap::new();

        for (rank, hit) in grep.into_iter().enumerate() {
            let key = chunk_key(&hit.chunk);
            let rrf = 1.0 / (self.k + rank as f32 + 1.0);
            scores
                .entry(key)
                .and_modify(|e| {
                    e.score += rrf;
                    e.source = RetrievalSource::Fused;
                })
                .or_insert(ScoredChunk {
                    chunk: hit.chunk,
                    score: rrf,
                    source: RetrievalSource::Grep,
                    preview: hit.preview,
                });
        }

        for (rank, hit) in semantic.into_iter().enumerate() {
            let key = chunk_key(&hit.chunk);
            let rrf = 1.0 / (self.k + rank as f32 + 1.0);
            scores
                .entry(key)
                .and_modify(|e| {
                    e.score += rrf;
                    e.source = RetrievalSource::Fused;
                    if e.preview.is_none() {
                        e.preview = hit.preview.clone();
                    }
                })
                .or_insert(ScoredChunk {
                    chunk: hit.chunk,
                    score: rrf,
                    source: hit.source,
                    preview: hit.preview,
                });
        }

        let mut out: Vec<_> = scores.into_values().collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

fn pin_exact_hits(ranked: Vec<ScoredChunk>, exact_pins: &[ScoredChunk]) -> Vec<ScoredChunk> {
    if exact_pins.is_empty() {
        return ranked;
    }

    let mut pinned: Vec<ScoredChunk> = Vec::new();
    let mut rest: Vec<ScoredChunk> = Vec::new();
    let mut pinned_keys: HashSet<(String, u32, u32)> = HashSet::new();

    for mut hit in ranked {
        if exact_pins.iter().any(|pin| chunks_overlap(&hit.chunk, &pin.chunk)) {
            let key = chunk_key(&hit.chunk);
            if pinned_keys.insert(key.clone()) {
                hit.score = EXACT_PIN_SCORE;
                hit.source = RetrievalSource::Grep;
                pinned.push(hit);
            }
        } else {
            rest.push(hit);
        }
    }

    pinned.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pinned.extend(rest);
    pinned
}

fn chunks_overlap(a: &ChunkRef, b: &ChunkRef) -> bool {
    a.file == b.file && a.start_line <= b.end_line && a.end_line >= b.start_line
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_core::types::ChunkRef;
    use std::path::PathBuf;

    fn chunk(line: u32) -> ChunkRef {
        ChunkRef {
            file: PathBuf::from("a.rs"),
            chunk_id: line,
            start_line: line,
            end_line: line + 5,
            byte_start: 0,
            byte_end: 100,
        }
    }

    fn scored(line: u32, score: f32, source: RetrievalSource) -> ScoredChunk {
        ScoredChunk {
            chunk: chunk(line),
            score,
            source,
            preview: None,
        }
    }

    #[test]
    fn rrf_merges_lists() {
        let fusion = RrfFusion::default();
        let grep = vec![scored(1, 1.0, RetrievalSource::Grep)];
        let semantic = vec![scored(2, 0.9, RetrievalSource::Pq4)];
        let out = fusion.fuse(grep, semantic);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn exact_pins_surface_first() {
        let fusion = RrfFusion::default();
        let exact = vec![scored(42, 1.0, RetrievalSource::Grep)];
        let laser = vec![scored(40, 1.0, RetrievalSource::Laser)];
        let semantic = vec![scored(2, 0.9, RetrievalSource::Pq4)];
        let lexical = fusion.fuse_lexical(exact.clone(), laser);
        let out = fusion.fuse_hybrid(lexical, semantic, exact);
        assert!(out[0].score >= EXACT_PIN_SCORE);
        assert!(out[0].chunk.start_line <= 42 && out[0].chunk.end_line >= 42);
    }
}
