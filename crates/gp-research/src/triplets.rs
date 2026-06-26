use gp_core::error::{GpError, Result};
use gp_core::traits::{Embedder, PreFocus};
use gp_eval::{load_queries, EvalQuery};
use gp_index::Index;
use gp_pq4::train::{save_learned, train_learned_pq4, Triplet, TrainConfig};
use std::path::{Path, PathBuf};

/// Train learned PQ4 projection from eval triplets and save to index root.
pub fn train_pq4(
    corpus: &Path,
    queries_path: &Path,
    embedder: &dyn Embedder,
    out_dim: usize,
) -> Result<PathBuf> {
    let triplets = build_triplets(corpus, queries_path, embedder)?;
    let model = train_learned_pq4(&triplets, &TrainConfig { out_dim, ..Default::default() })?;
    let index_root = Index::index_path(corpus);
    std::fs::create_dir_all(&index_root)?;
    let path = index_root.join("pq4.json");
    save_learned(&path, &model)?;
    Ok(path)
}

fn build_triplets(
    corpus: &Path,
    queries_path: &Path,
    embedder: &dyn Embedder,
) -> Result<Vec<Triplet>> {
    let queries = load_queries(queries_path)?;
    let sketch = gp_sketch::SketchBeam::build(vec![corpus.to_path_buf()])?;
    let chunk_cfg = gp_chunk::ChunkConfig::default();
    let mut triplets = Vec::new();

    for q in &queries {
        if let Some(t) = triplet_for_query(corpus, embedder, &sketch, &chunk_cfg, q)? {
            triplets.push(t);
        }
    }
    if triplets.is_empty() {
        return Err(GpError::Training(
            "no triplets built — check corpus paths and oracles".into(),
        ));
    }
    Ok(triplets)
}

fn triplet_for_query(
    corpus: &Path,
    embedder: &dyn Embedder,
    sketch: &gp_sketch::SketchBeam,
    chunk_cfg: &gp_chunk::ChunkConfig,
    q: &EvalQuery,
) -> Result<Option<Triplet>> {
    let q_vec = embedder.embed_query(&q.query)?;
    let positive = match positive_embedding(corpus, embedder, chunk_cfg, q)? {
        Some(v) => v,
        None => return Ok(None),
    };
    let negative = match negative_embedding(embedder, sketch, q)? {
        Some(v) => v,
        None => return Ok(None),
    };
    Ok(Some(Triplet {
        anchor: q_vec,
        positive,
        negative,
    }))
}

fn positive_embedding(
    corpus: &Path,
    embedder: &dyn Embedder,
    chunk_cfg: &gp_chunk::ChunkConfig,
    q: &EvalQuery,
) -> Result<Option<Vec<f32>>> {
    for oracle in &q.oracles {
        let file = corpus.join(&oracle.file);
        let content = std::fs::read_to_string(&file).map_err(|e| GpError::Io(e))?;
        let chunks = gp_chunk::chunk_file(&file, &content, chunk_cfg);
        for c in chunks {
            if c.chunk_ref.start_line <= oracle.end_line
                && c.chunk_ref.end_line >= oracle.start_line
            {
                let emb = embedder.embed(&[c.text])?;
                return Ok(emb.into_iter().next());
            }
        }
    }
    Ok(None)
}

fn negative_embedding(
    embedder: &dyn Embedder,
    sketch: &gp_sketch::SketchBeam,
    q: &EvalQuery,
) -> Result<Option<Vec<f32>>> {
    let candidates = sketch.sketch_beam(&q.query, 20, 10)?;
    for cand in candidates {
        let is_oracle = q.oracles.iter().any(|o| {
            cand.file.ends_with(&o.file)
                && cand.start_line <= o.end_line
                && cand.end_line >= o.start_line
        });
        if is_oracle {
            continue;
        }
        if let Some(ch) = sketch.chunks.iter().find(|c| c.chunk_ref == cand) {
            let emb = embedder.embed(&[ch.text.clone()])?;
            return Ok(emb.into_iter().next());
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_queries_path_errors() {
        let dir = tempfile::TempDir::new().expect("tmpdir");
        let qpath = dir.path().join("empty.jsonl");
        std::fs::write(&qpath, "").expect("write");
        struct NoEmbed;
        impl Embedder for NoEmbed {
            fn embed(&self, _: &[String]) -> Result<Vec<Vec<f32>>> {
                Ok(vec![])
            }
            fn embed_query(&self, _: &str) -> Result<Vec<f32>> {
                Ok(vec![])
            }
            fn dim(&self) -> usize {
                8
            }
            fn model_id(&self) -> &str {
                "noop"
            }
        }
        let err = build_triplets(dir.path(), &qpath, &NoEmbed).unwrap_err();
        assert!(matches!(err, GpError::Training(_)));
    }
}
