use crate::core::error::{GpError, Result};
use crate::core::query;
use crate::core::traits::{Embedder, LaserFocus};
use crate::core::types::{ChunkRef, RetrievalSource, Route, ScoredChunk};
use crate::fusion::RrfFusion;
use crate::grep::{exact_grep_scored, resolve_exact_backend, ParallelGrep};
use crate::index::{candidate_beam_mode, ensure_sketch_shell, vector_codec, Index};
use crate::laser::Laser;
use crate::sketch::SketchBeam;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct IndexBuildOptions {
    pub model_id: String,
    pub dim: usize,
    pub sketch_only: bool,
    pub chunk_mode: String,
    pub exclude: Vec<String>,
    pub ann_enabled: bool,
    pub ann_min_chunks: usize,
}

impl IndexBuildOptions {
    pub fn from_config(cfg: &crate::core::config::Config, sketch_only: bool) -> Self {
        Self {
            model_id: cfg.embedder.active.clone(),
            dim: cfg.embedder.dim,
            sketch_only,
            chunk_mode: cfg.index.chunk_mode.clone(),
            exclude: cfg.index.exclude.clone(),
            ann_enabled: cfg.search.ann_enabled,
            ann_min_chunks: cfg.index.ann_min_chunks,
        }
    }
}

/// Build sketch shell or full warm index.
pub fn build_index(
    repo: &Path,
    embedder: Option<&dyn Embedder>,
    opts: &IndexBuildOptions,
) -> Result<Index> {
    let chunk_cfg = crate::chunk::ChunkConfig::from_mode(&opts.chunk_mode);
    if opts.sketch_only {
        return Index::build_sketch_only_with_options(
            repo,
            &opts.model_id,
            opts.dim,
            chunk_cfg,
            &opts.exclude,
        );
    }

    let sketch =
        SketchBeam::build_with_options(vec![repo.to_path_buf()], chunk_cfg, &opts.exclude)?;
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

    Index::build_with_options(
        repo,
        &opts.model_id,
        opts.dim,
        vectors.as_deref(),
        opts.ann_enabled,
        opts.ann_min_chunks,
        sketch,
    )
}

pub struct SearchOptions {
    pub route: Route,
    pub laser_cap: usize,
    pub sketch_beam: usize,
    pub top_k: usize,
    pub dim: usize,
    pub jit_enabled: bool,
    pub jit_embed_budget: usize,
    pub jit_reheat_file_cap: usize,
    pub jit_cold_first_file_cap: usize,
    pub jit_cold_first_embed_budget: usize,
    pub model_id: String,
    /// `parallel`, `ripgrep`, or `auto` — see `GrepCfg`.
    pub grep_backend: String,
    /// Sketch mode: `beam`, `minhash`, or `bm25`.
    pub sketch_mode: String,
    pub fusion: String,
    pub ann_enabled: bool,
    pub chunk_mode: String,
    pub exclude: Vec<String>,
    /// Optional JIT embed byte accounting (eval harness).
    pub embed_stats: Option<crate::core::embed_stats::EmbedStatsCell>,
}

impl SearchOptions {
    pub fn from_config(cfg: &crate::core::config::Config, route: Route) -> Self {
        Self {
            route,
            laser_cap: cfg.search.laser_candidate_cap,
            sketch_beam: cfg.search.sketch_beam_width,
            top_k: 20,
            dim: cfg.embedder.dim,
            jit_enabled: cfg.search.jit_enabled,
            jit_embed_budget: cfg.search.jit_embed_budget,
            jit_reheat_file_cap: cfg.search.jit_reheat_file_cap,
            jit_cold_first_file_cap: cfg.search.jit_cold_first_file_cap,
            jit_cold_first_embed_budget: cfg.search.jit_cold_first_embed_budget,
            model_id: cfg.embedder.active.clone(),
            grep_backend: cfg.grep.backend.clone(),
            sketch_mode: cfg.index.sketch.clone(),
            fusion: cfg.search.fusion.clone(),
            ann_enabled: cfg.search.ann_enabled,
            chunk_mode: cfg.index.chunk_mode.clone(),
            exclude: cfg.index.exclude.clone(),
            embed_stats: None,
        }
    }
}

fn make_fusion(name: &str) -> Result<RrfFusion> {
    match name.to_ascii_lowercase().as_str() {
        "rrf" => Ok(RrfFusion::default()),
        other => Err(GpError::Config(format!(
            "unknown search.fusion `{other}` (supported: rrf)"
        ))),
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
    let laser = laser_scored(query, paths, opts);
    let fusion = make_fusion(&opts.fusion)?;
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
    laser_scored(query, paths, opts)
}

fn laser_scored(query: &str, paths: &[PathBuf], opts: &SearchOptions) -> Vec<ScoredChunk> {
    let grep = ParallelGrep;
    let exclude = crate::core::exclude_to_globs(&opts.exclude);
    let laser = Laser::with_options(
        grep,
        paths.to_vec(),
        crate::chunk::ChunkConfig::from_mode(&opts.chunk_mode),
        exclude,
    );
    laser
        .focus(query, opts.laser_cap)
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
    let laser: Vec<ChunkRef> = laser_scored(query, paths, opts)
        .into_iter()
        .map(|s| s.chunk)
        .collect();
    let sketch = candidate_beam_mode(
        repo,
        query,
        opts.sketch_beam,
        opts.laser_cap,
        &opts.sketch_mode,
    )
    .unwrap_or_default();
    merge_candidates(&merge_candidates(&exact, &laser), &sketch)
}

fn semantic_scored(
    query: &str,
    repo: &Path,
    paths: &[PathBuf],
    embedder: &dyn Embedder,
    opts: &SearchOptions,
) -> Result<Vec<ScoredChunk>> {
    ensure_sketch_shell(repo, &opts.model_id, opts.dim)?;
    let index = Index::open(repo)?;
    if index.manifest.dim != opts.dim {
        return Err(GpError::Index(format!(
            "index dim {} != config dim {}",
            index.manifest.dim, opts.dim
        )));
    }

    let candidates = gather_candidates(query, paths, repo, opts);
    let query_vec = embedder.embed_query(query)?;
    let codec = vector_codec(opts.dim);

    let hits = if opts.jit_enabled && index.chunks.is_empty() {
        let mut embed_fn = |texts: &[String]| embedder.embed(texts);
        index.jit_semantic_search(
            &query_vec,
            &codec,
            &candidates,
            &mut embed_fn,
            opts.jit_embed_budget,
            opts.jit_reheat_file_cap,
            opts.jit_cold_first_file_cap,
            opts.jit_cold_first_embed_budget,
            opts.top_k,
            opts.dim,
            opts.embed_stats.as_ref(),
        )?
    } else {
        index.search_semantic_with_ann(
            &query_vec,
            &codec,
            Some(&candidates),
            opts.top_k,
            opts.ann_enabled,
        )
    };

    Ok(hits
        .into_iter()
        .map(|(c, score)| ScoredChunk {
            chunk: c.chunk_ref,
            score,
            source: RetrievalSource::Vector,
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
    let candidates = candidate_beam_mode(
        repo,
        query,
        opts.sketch_beam,
        opts.laser_cap,
        &opts.sketch_mode,
    )?;

    if candidates.is_empty() {
        return Ok(vec![]);
    }

    if let Some(emb) = embedder {
        ensure_sketch_shell(repo, &opts.model_id, opts.dim)?;
        let index = Index::open(repo)?;
        let query_vec = emb.embed_query(query)?;
        let codec = vector_codec(opts.dim);

        let hits = if opts.jit_enabled && index.chunks.is_empty() {
            let mut embed_fn = |texts: &[String]| emb.embed(texts);
            index.jit_semantic_search(
                &query_vec,
                &codec,
                &candidates,
                &mut embed_fn,
                opts.jit_embed_budget,
                opts.jit_reheat_file_cap,
                opts.jit_cold_first_file_cap,
                opts.jit_cold_first_embed_budget,
                opts.top_k,
                opts.dim,
                opts.embed_stats.as_ref(),
            )?
        } else {
            index.search_semantic_with_ann(
                &query_vec,
                &codec,
                Some(&candidates),
                opts.top_k,
                opts.ann_enabled,
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
        crate::index::with_isolated_cache(|| {
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
                jit_enabled: true,
                jit_embed_budget: 64,
                jit_reheat_file_cap: 16,
                jit_cold_first_file_cap: 6,
                jit_cold_first_embed_budget: 18,
                model_id: "mock".into(),
                grep_backend: "parallel".into(),
                sketch_mode: "beam".into(),
                fusion: "rrf".into(),
                ann_enabled: true,
                chunk_mode: "line".into(),
                exclude: vec![],
                embed_stats: None,
            };
            let hits = hybrid_search(
                "handleSessionRefresh",
                &[dir.path().to_path_buf()],
                Some(&emb),
                &search_opts,
            )
            .unwrap();
            assert!(!hits.is_empty());
        });
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
            jit_enabled: false,
            jit_embed_budget: 64,
            jit_reheat_file_cap: 16,
            jit_cold_first_file_cap: 6,
            jit_cold_first_embed_budget: 18,
            model_id: "mock".into(),
            grep_backend: "parallel".into(),
            sketch_mode: "beam".into(),
            fusion: "rrf".into(),
            ann_enabled: true,
            chunk_mode: "line".into(),
            exclude: vec![],
            embed_stats: None,
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
        crate::index::with_isolated_cache(|| {
            let dir = TempDir::new().unwrap();
            let p = dir.path().join("auth.rs");
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "fn handleSessionRefresh() {{").unwrap();

            let emb = MockEmbedder { dim: 8 };
            let opts = IndexBuildOptions {
                model_id: "mock".into(),
                dim: 8,
                sketch_only: false,
                chunk_mode: "line".into(),
                exclude: vec![],
                ann_enabled: true,
                ann_min_chunks: 500,
            };
            build_index(dir.path(), Some(&emb), &opts).unwrap();

            let search_opts = SearchOptions {
                route: Route::Semantic,
                laser_cap: 10,
                sketch_beam: 5,
                top_k: 5,
                dim: 8,
                jit_enabled: true,
                jit_embed_budget: 64,
                jit_reheat_file_cap: 16,
                jit_cold_first_file_cap: 6,
                jit_cold_first_embed_budget: 18,
                model_id: "mock".into(),
                grep_backend: "parallel".into(),
                sketch_mode: "beam".into(),
                fusion: "rrf".into(),
                ann_enabled: true,
                chunk_mode: "line".into(),
                exclude: vec![],
                embed_stats: None,
            };
            let hits = hybrid_search(
                "handleSessionRefresh",
                &[dir.path().to_path_buf()],
                Some(&emb),
                &search_opts,
            )
            .unwrap();
            assert!(!hits.is_empty());
        });
    }
}
