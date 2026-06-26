use gp_core::error::{GpError, Result};
use gp_core::query;
use gp_core::traits::{Embedder, LaserFocus};
use gp_core::types::{ChunkRef, RetrievalSource, Route, ScoredChunk};
use gp_fusion::RrfFusion;
use gp_grep::{exact_grep_scored, resolve_exact_backend, ParallelGrep};
use gp_index::{candidate_beam, ensure_sketch_shell, load_projection_backend, Index};
use gp_laser::Laser;
use gp_sketch::SketchBeam;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct IndexBuildOptions {
    pub model_id: String,
    pub dim: usize,
    pub projection: String,
    pub sketch_only: bool,
}

/// Build sketch shell or full warm index.
pub fn build_index(
    repo: &Path,
    embedder: Option<&dyn Embedder>,
    opts: &IndexBuildOptions,
) -> Result<Index> {
    if opts.sketch_only {
        return Index::build_sketch_only(repo, &opts.model_id, opts.dim, &opts.projection);
    }

    let sketch = SketchBeam::build(vec![repo.to_path_buf()])?;
    let vectors = if let Some(emb) = embedder {
        let texts: Vec<String> = sketch.chunks.iter().map(|c| c.text.clone()).collect();
        if texts.is_empty() {
            None
        } else {
            Some(emb.embed(&texts)?)
        }
    } else {
        None
    };

    Index::build(
        repo,
        &opts.model_id,
        opts.dim,
        &opts.projection,
        vectors.as_deref(),
    )
}

pub struct SearchOptions {
    pub route: Route,
    pub laser_cap: usize,
    pub sketch_beam: usize,
    pub top_k: usize,
    pub dim: usize,
    pub projection: String,
    pub jit_enabled: bool,
    pub jit_embed_budget: usize,
    pub jit_reheat_file_cap: usize,
    pub jit_cold_first_file_cap: usize,
    pub jit_cold_first_embed_budget: usize,
    pub model_id: String,
    /// `parallel`, `ripgrep`, or `auto` — see `GrepCfg`.
    pub grep_backend: String,
}

impl SearchOptions {
    pub fn from_config(cfg: &gp_core::config::Config, route: Route) -> Self {
        Self {
            route,
            laser_cap: cfg.search.laser_candidate_cap,
            sketch_beam: cfg.search.sketch_beam_width,
            top_k: 20,
            dim: cfg.embedder.dim,
            projection: cfg.index.projection.clone(),
            jit_enabled: cfg.search.jit_enabled,
            jit_embed_budget: cfg.search.jit_embed_budget,
            jit_reheat_file_cap: cfg.search.jit_reheat_file_cap,
            jit_cold_first_file_cap: cfg.search.jit_cold_first_file_cap,
            jit_cold_first_embed_budget: cfg.search.jit_cold_first_embed_budget,
            model_id: cfg.embedder.active.clone(),
            grep_backend: cfg.grep.backend.clone(),
        }
    }
}

pub fn hybrid_search(
    query: &str,
    paths: &[PathBuf],
    embedder: Option<&dyn Embedder>,
    opts: &SearchOptions,
) -> Result<Vec<ScoredChunk>> {
    let repo = paths.first().cloned().unwrap_or_else(|| PathBuf::from("."));

    match opts.route {
        Route::Grep => return Ok(grep_route_scored(query, paths, opts)),
        Route::Prefocus => return prefocus_search(query, &repo, embedder, opts),
        Route::Semantic | Route::Hybrid => {}
    }

    let exact_backend = resolve_exact_backend(&opts.grep_backend);
    let exact = exact_grep_scored(query, paths, opts.laser_cap, &exact_backend);
    let laser = laser_scored(query, paths, opts.laser_cap);
    let fusion = RrfFusion::default();
    let lexical = fusion.fuse_lexical(exact.clone(), laser);

    let mut semantic = Vec::new();
    if let Some(emb) = embedder {
        semantic = semantic_scored(query, &repo, paths, emb, opts)?;
    }

    if opts.route == Route::Semantic && !semantic.is_empty() {
        let mut out = fusion.fuse_hybrid(vec![], semantic, exact);
        out.truncate(opts.top_k);
        return Ok(out);
    }

    if lexical.is_empty() && semantic.is_empty() {
        if let Ok(prefocus) = prefocus_search(query, &repo, embedder, opts) {
            if !prefocus.is_empty() {
                return Ok(prefocus);
            }
        }
    }

    let mut fused = fusion.fuse_hybrid(lexical, semantic, exact);
    fused.truncate(opts.top_k);
    Ok(fused)
}

/// Literal / regex grep fast path — no embedding, exact scan first.
fn grep_route_scored(query: &str, paths: &[PathBuf], opts: &SearchOptions) -> Vec<ScoredChunk> {
    let exact_backend = resolve_exact_backend(&opts.grep_backend);
    let exact = exact_grep_scored(query, paths, opts.laser_cap, &exact_backend);
    if query::is_literal_query(query) || query::is_quoted(query.trim()) {
        return exact;
    }
    if !exact.is_empty() {
        return exact;
    }
    laser_scored(query, paths, opts.laser_cap)
}

fn laser_scored(query: &str, paths: &[PathBuf], cap: usize) -> Vec<ScoredChunk> {
    let grep = ParallelGrep;
    let laser = Laser::new(grep, paths.to_vec());
    laser
        .focus(query, cap)
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| ScoredChunk {
            chunk,
            score: 1.0 / (i as f32 + 1.0),
            source: RetrievalSource::Laser,
            preview: None,
        })
        .collect()
}

fn merge_candidates(a: &[ChunkRef], b: &[ChunkRef]) -> Vec<ChunkRef> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for c in a.iter().chain(b.iter()) {
        let key = (
            c.file.to_string_lossy().into_owned(),
            c.start_line,
            c.end_line,
        );
        if seen.insert(key) {
            out.push(c.clone());
        }
    }
    out
}

fn gather_candidates(
    query: &str,
    paths: &[PathBuf],
    repo: &Path,
    opts: &SearchOptions,
) -> Vec<ChunkRef> {
    let exact_backend = resolve_exact_backend(&opts.grep_backend);
    let exact: Vec<ChunkRef> = exact_grep_scored(query, paths, opts.laser_cap, &exact_backend)
        .into_iter()
        .map(|s| s.chunk)
        .collect();
    let laser: Vec<ChunkRef> = laser_scored(query, paths, opts.laser_cap)
        .into_iter()
        .map(|s| s.chunk)
        .collect();
    let sketch = candidate_beam(repo, query, opts.sketch_beam, opts.laser_cap).unwrap_or_default();
    merge_candidates(&merge_candidates(&exact, &laser), &sketch)
}

fn semantic_scored(
    query: &str,
    repo: &Path,
    paths: &[PathBuf],
    embedder: &dyn Embedder,
    opts: &SearchOptions,
) -> Result<Vec<ScoredChunk>> {
    ensure_sketch_shell(repo, &opts.model_id, opts.dim, &opts.projection)?;
    let index = Index::open(repo)?;
    if index.manifest.dim != opts.dim {
        return Err(GpError::Index(format!(
            "index dim {} != config dim {}",
            index.manifest.dim, opts.dim
        )));
    }

    let candidates = gather_candidates(query, paths, repo, opts);
    let query_vec = embedder.embed_query(query)?;
    let backend = load_projection_backend(&index.manifest.projection, opts.dim, &index.root)?;

    let hits = if opts.jit_enabled && index.chunks.is_empty() {
        let mut embed_fn = |texts: &[String]| embedder.embed(texts);
        index.jit_semantic_search(
            &query_vec,
            backend.as_ref(),
            &candidates,
            &mut embed_fn,
            opts.jit_embed_budget,
            opts.jit_reheat_file_cap,
            opts.jit_cold_first_file_cap,
            opts.jit_cold_first_embed_budget,
            opts.top_k,
        )?
    } else {
        index.search_semantic_filtered(
            &query_vec,
            backend.as_ref(),
            Some(&candidates),
            opts.top_k,
        )
    };

    Ok(hits
        .into_iter()
        .map(|(c, score)| ScoredChunk {
            chunk: c.chunk_ref,
            score,
            source: RetrievalSource::Pq4,
            preview: Some(c.text.chars().take(160).collect()),
        })
        .collect())
}

fn prefocus_search(
    query: &str,
    repo: &Path,
    embedder: Option<&dyn Embedder>,
    opts: &SearchOptions,
) -> Result<Vec<ScoredChunk>> {
    let candidates = candidate_beam(repo, query, opts.sketch_beam, opts.laser_cap)?;

    if candidates.is_empty() {
        return Ok(vec![]);
    }

    if let Some(emb) = embedder {
        ensure_sketch_shell(repo, &opts.model_id, opts.dim, &opts.projection)?;
        let index = Index::open(repo)?;
        let query_vec = emb.embed_query(query)?;
        let backend = load_projection_backend(&index.manifest.projection, opts.dim, &index.root)?;

        let hits = if opts.jit_enabled && index.chunks.is_empty() {
            let mut embed_fn = |texts: &[String]| emb.embed(texts);
            index.jit_semantic_search(
                &query_vec,
                backend.as_ref(),
                &candidates,
                &mut embed_fn,
                opts.jit_embed_budget,
                opts.jit_reheat_file_cap,
                opts.jit_cold_first_file_cap,
                opts.jit_cold_first_embed_budget,
                opts.top_k,
            )?
        } else {
            index.search_semantic_filtered(
                &query_vec,
                backend.as_ref(),
                Some(&candidates),
                opts.top_k,
            )
        };

        if !hits.is_empty() {
            return Ok(hits
                .into_iter()
                .map(|(c, score)| ScoredChunk {
                    chunk: c.chunk_ref,
                    score,
                    source: RetrievalSource::Sketch,
                    preview: Some(c.text.chars().take(160).collect()),
                })
                .collect());
        }
    }

    Ok(candidates
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| ScoredChunk {
            chunk,
            score: 1.0 / (i as f32 + 1.0),
            source: RetrievalSource::Sketch,
            preview: None,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    struct MockEmbedder {
        dim: usize,
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0; self.dim];
                    for (i, b) in t.bytes().enumerate().take(self.dim) {
                        v[i % self.dim] += b as f32 / 255.0;
                    }
                    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-8 {
                        for x in &mut v {
                            *x /= norm;
                        }
                    }
                    v
                })
                .collect())
        }

        fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
            self.embed(&[query.to_string()]).map(|v| v[0].clone())
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn model_id(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn jit_search_cold_start() {
        let cache = TempDir::new().unwrap();
        std::env::set_var("GREPPLUS_CACHE_DIR", cache.path());

        let dir = TempDir::new().unwrap();
        let p = dir.path().join("auth.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "fn handleSessionRefresh() {{").unwrap();
        writeln!(f, "    validate_token();").unwrap();
        writeln!(f, "}}").unwrap();

        let emb = MockEmbedder { dim: 8 };
        let search_opts = SearchOptions {
            route: Route::Semantic,
            laser_cap: 10,
            sketch_beam: 5,
            top_k: 5,
            dim: 8,
            projection: "baseline".into(),
            jit_enabled: true,
            jit_embed_budget: 64,
            jit_reheat_file_cap: 16,
            jit_cold_first_file_cap: 6,
            jit_cold_first_embed_budget: 18,
            model_id: "mock".into(),
            grep_backend: "parallel".into(),
        };
        let hits = hybrid_search(
            "handleSessionRefresh",
            &[dir.path().to_path_buf()],
            Some(&emb),
            &search_opts,
        )
        .unwrap();
        assert!(!hits.is_empty());

        std::env::remove_var("GREPPLUS_CACHE_DIR");
    }

    #[test]
    fn literal_grep_route_uses_exact() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("auth.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "fn handleSessionRefresh() {{").unwrap();

        let search_opts = SearchOptions {
            route: Route::Grep,
            laser_cap: 10,
            sketch_beam: 5,
            top_k: 5,
            dim: 8,
            projection: "baseline".into(),
            jit_enabled: false,
            jit_embed_budget: 64,
            jit_reheat_file_cap: 16,
            jit_cold_first_file_cap: 6,
            jit_cold_first_embed_budget: 18,
            model_id: "mock".into(),
            grep_backend: "parallel".into(),
        };
        let hits = hybrid_search(
            "handleSessionRefresh",
            &[dir.path().to_path_buf()],
            None,
            &search_opts,
        )
        .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].source, RetrievalSource::Grep);
    }

    #[test]
    fn full_warm_index_still_works() {
        let cache = TempDir::new().unwrap();
        std::env::set_var("GREPPLUS_CACHE_DIR", cache.path());

        let dir = TempDir::new().unwrap();
        let p = dir.path().join("auth.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "fn handleSessionRefresh() {{").unwrap();

        let emb = MockEmbedder { dim: 8 };
        let opts = IndexBuildOptions {
            model_id: "mock".into(),
            dim: 8,
            projection: "baseline".into(),
            sketch_only: false,
        };
        build_index(dir.path(), Some(&emb), &opts).unwrap();

        let search_opts = SearchOptions {
            route: Route::Semantic,
            laser_cap: 10,
            sketch_beam: 5,
            top_k: 5,
            dim: 8,
            projection: "baseline".into(),
            jit_enabled: true,
            jit_embed_budget: 64,
            jit_reheat_file_cap: 16,
            jit_cold_first_file_cap: 6,
            jit_cold_first_embed_budget: 18,
            model_id: "mock".into(),
            grep_backend: "parallel".into(),
        };
        let hits = hybrid_search(
            "handleSessionRefresh",
            &[dir.path().to_path_buf()],
            Some(&emb),
            &search_opts,
        )
        .unwrap();
        assert!(!hits.is_empty());

        std::env::remove_var("GREPPLUS_CACHE_DIR");
    }
}
