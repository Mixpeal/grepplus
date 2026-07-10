use crate::core::error::Result;
use crate::sketch::SketchBeam;
use std::path::Path;

/// Build sketch shell only (Tier 1): no embeddings, all files COLD.
pub fn ensure_sketch_shell(repo: &Path, model_id: &str, dim: usize) -> Result<crate::index::Index> {
    if crate::index::Index::exists(repo) {
        return crate::index::Index::open(repo);
    }
    crate::index::Index::build_sketch_only(repo, model_id, dim)
}

/// Load sketch beam from index cache when possible, else walk the repo.
pub fn sketch_for_repo(repo: &Path) -> Result<SketchBeam> {
    let index_root = crate::index::Index::index_path(repo);
    if index_root.join("sketch").join("file_index.json").exists() {
        SketchBeam::load_from_index(&index_root, repo)
    } else {
        SketchBeam::build(vec![repo.to_path_buf()])
    }
}

pub fn candidate_beam(
    repo: &Path,
    query: &str,
    beam_width: usize,
    cap: usize,
) -> Result<Vec<crate::core::types::ChunkRef>> {
    candidate_beam_mode(repo, query, beam_width, cap, "beam")
}

pub fn candidate_beam_mode(
    repo: &Path,
    query: &str,
    beam_width: usize,
    cap: usize,
    sketch_mode: &str,
) -> Result<Vec<crate::core::types::ChunkRef>> {
    let beam = sketch_for_repo(repo)?;
    beam.sketch_beam_mode(query, beam_width, cap, sketch_mode)
}
