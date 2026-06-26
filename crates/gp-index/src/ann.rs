//! Flat HNSW-style ANN over PQ4 codes for large-repo search.

use gp_core::error::{GpError, Result};
use gp_core::traits::{ProjectionBackend, Q4Code};
use serde::{Deserialize, Serialize};
use std::path::Path;

const M: usize = 16;
const EF_BUILD: usize = 64;
const EF_SEARCH: usize = 48;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnGraph {
    pub version: u32,
    pub neighbors: Vec<Vec<u32>>,
}

impl AnnGraph {
    pub fn build(codes: &[Q4Code], backend: &dyn ProjectionBackend, query_samples: &[Vec<f32>]) -> Self {
        let n = codes.len();
        let mut neighbors = vec![Vec::new(); n];
        if n == 0 {
            return Self {
                version: 1,
                neighbors,
            };
        }
        for i in 0..n {
            let mut scores: Vec<(u32, f32)> = Vec::new();
            let qi = &query_samples[i % query_samples.len().max(1)];
            for (j, code) in codes.iter().enumerate() {
                if i == j {
                    continue;
                }
                let s = backend.score(qi, code);
                scores.push((j as u32, s));
            }
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            neighbors[i] = scores.into_iter().take(M).map(|(j, _)| j).collect();
        }
        let _ = EF_BUILD;
        Self {
            version: 1,
            neighbors,
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        codes: &[Q4Code],
        backend: &dyn ProjectionBackend,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        if codes.is_empty() {
            return vec![];
        }
        let mut visited = vec![false; codes.len()];
        let mut candidates: Vec<(usize, f32)> = (0..codes.len().min(EF_SEARCH))
            .map(|i| (i, backend.score(query, &codes[i])))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut best = candidates.clone();

        for _ in 0..EF_SEARCH {
            let Some((entry, _)) = candidates.first().copied() else {
                break;
            };
            candidates.remove(0);
            if visited[entry] {
                continue;
            }
            visited[entry] = true;
            for &nb in &self.neighbors[entry] {
                let idx = nb as usize;
                if idx >= codes.len() || visited[idx] {
                    continue;
                }
                let s = backend.score(query, &codes[idx]);
                candidates.push((idx, s));
                best.push((idx, s));
            }
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            candidates.truncate(EF_SEARCH);
        }

        best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        best.dedup_by_key(|(i, _)| *i);
        best.truncate(top_k);
        best
    }
}

pub fn save_graph(path: &Path, graph: &AnnGraph) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(graph)?)
        .map_err(|e| GpError::Index(e.to_string()))?;
    Ok(())
}

pub fn load_graph(path: &Path) -> Result<AnnGraph> {
    let raw = std::fs::read_to_string(path).map_err(|e| GpError::Index(e.to_string()))?;
    serde_json::from_str(&raw).map_err(|e| GpError::Index(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gp_pq4::BaselineQ4;

    #[test]
    fn graph_search_returns_results() {
        let backend = BaselineQ4 { proj_dim: 4 };
        let codes: Vec<Q4Code> = (0..20)
            .map(|i| {
                let mut v = vec![0.0; 4];
                v[i % 4] = 1.0;
                backend.project(&v)
            })
            .collect();
        let samples: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let mut v = vec![0.0; 4];
                v[i % 4] = 1.0;
                v
            })
            .collect();
        let graph = AnnGraph::build(&codes, &backend, &samples);
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let hits = graph.search(&q, &codes, &backend, 5);
        assert!(!hits.is_empty());
    }
}
