use crate::core::config::Config;
use crate::core::embed_stats::EmbedStatsCell;
use crate::core::error::Result;
use crate::core::traits::{
    CategoryMetrics, EvalHarness, EvalMetrics, EvalMode, GrepEngine, GrepOptions, LaserFocus,
    QueryEmbedStats, Router,
};
use crate::core::types::{ChunkRef, RepoMeta, Route};
use crate::embed::{resolve_embedder, EnsureOptions};
use crate::grep::{ParallelGrep, RipgrepEngine, UnixGrepEngine};
use crate::router::{resolve_router, FeatureRouter, HeuristicRouter};
use crate::search::{build_index, hybrid_search, IndexBuildOptions, SearchOptions};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Deserialize)]
pub struct EvalQuery {
    pub id: String,
    pub query: String,
    pub category: String,
    pub oracles: Vec<OracleLocation>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OracleLocation {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// Load eval queries from a JSONL file.
pub fn load_queries(path: &std::path::Path) -> Result<Vec<EvalQuery>> {
    let raw = std::fs::read_to_string(path)?;
    let mut queries = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        queries.push(serde_json::from_str(line)?);
    }
    Ok(queries)
}

#[derive(Debug, Clone, Default)]
pub struct HarnessOverrides {
    pub jit_embed_budget: Option<usize>,
    pub jit_reheat_file_cap: Option<usize>,
    pub router_mode: Option<String>,
}

pub struct AgentCodeHarness {
    pub corpus: PathBuf,
    pub queries_path: PathBuf,
    pub config: Option<Config>,
    pub ensure_index: bool,
    pub warm_index: bool,
    pub isolate_modes: bool,
    pub yes_download: bool,
    pub filter_category: Option<String>,
    pub filter_laser_miss: bool,
    pub overrides: HarnessOverrides,
}

impl AgentCodeHarness {
    pub fn new(corpus: impl Into<PathBuf>, queries_path: impl Into<PathBuf>) -> Self {
        Self {
            corpus: corpus.into(),
            queries_path: queries_path.into(),
            config: None,
            ensure_index: false,
            warm_index: false,
            isolate_modes: false,
            yes_download: false,
            filter_category: None,
            filter_laser_miss: false,
            overrides: HarnessOverrides::default(),
        }
    }

    pub fn with_config(mut self, cfg: Config) -> Self {
        self.config = Some(cfg);
        self
    }

    pub fn ensure_index(mut self, yes: bool) -> Self {
        self.ensure_index = yes;
        self
    }

    pub fn warm_index(mut self, yes: bool) -> Self {
        self.warm_index = yes;
        self
    }

    pub fn isolate_modes(mut self, yes: bool) -> Self {
        self.isolate_modes = yes;
        self
    }

    pub fn ensure_model(mut self, yes: bool) -> Self {
        self.yes_download = yes;
        self
    }

    pub fn filter_category(mut self, cat: Option<String>) -> Self {
        self.filter_category = cat;
        self
    }

    pub fn filter_laser_miss(mut self, yes: bool) -> Self {
        self.filter_laser_miss = yes;
        self
    }

    pub fn overrides(mut self, o: HarnessOverrides) -> Self {
        self.overrides = o;
        self
    }

    fn load_queries(&self) -> Result<Vec<EvalQuery>> {
        load_queries(&self.queries_path)
    }

    fn filter_queries(&self, queries: Vec<EvalQuery>) -> Result<Vec<EvalQuery>> {
        let mut out = queries;
        if let Some(ref cat) = self.filter_category {
            out.retain(|q| q.category == *cat);
        }
        if self.filter_laser_miss {
            out.retain(|q| {
                self.laser_search(&q.query)
                    .map(|r| recall_at_k(&r, &q.oracles, 10) == 0.0)
                    .unwrap_or(false)
            });
        }
        Ok(out)
    }

    fn unix_grep_search(&self, query: &str) -> Result<Vec<ChunkRef>> {
        let grep = UnixGrepEngine::discover()?;
        let opts = GrepOptions {
            roots: vec![self.corpus.clone()],
            max_results: Some(100),
            ..Default::default()
        };
        let hits = grep.search(query, &opts)?;
        Ok(hits_to_chunks(&hits))
    }

    fn ripgrep_search(&self, query: &str) -> Result<Vec<ChunkRef>> {
        let rg = RipgrepEngine::discover()?;
        let opts = GrepOptions {
            roots: vec![self.corpus.clone()],
            max_results: Some(100),
            ..Default::default()
        };
        let hits = rg.search(query, &opts)?;
        Ok(hits_to_chunks(&hits))
    }

    fn parallel_grep_search(&self, query: &str) -> Result<Vec<ChunkRef>> {
        let grep = ParallelGrep;
        let opts = GrepOptions {
            roots: vec![self.corpus.clone()],
            max_results: Some(100),
            ..Default::default()
        };
        let hits = grep.search(query, &opts)?;
        Ok(hits_to_chunks(&hits))
    }

    fn ensure_corpus_index(&self) -> Result<Option<Arc<dyn crate::core::traits::Embedder>>> {
        let cfg = match &self.config {
            Some(c) => c.clone(),
            None => return Ok(None),
        };
        let mut cfg = cfg;
        let opts = EnsureOptions::for_required_semantic(self.yes_download);
        let embedder = resolve_embedder(&mut cfg, &opts).ok().flatten();
        if self.ensure_index && embedder.is_some() {
            if self.warm_index {
                if !crate::index::Index::exists(&self.corpus) {
                    let opts = IndexBuildOptions {
                        model_id: cfg.embedder.active.clone(),
                        dim: cfg.embedder.dim,
                        sketch_only: false,
                        chunk_mode: "line".into(),
                        exclude: vec![],
                        ann_enabled: true,
                        ann_min_chunks: 500,
                    };
                    build_index(&self.corpus, embedder.as_deref(), &opts)?;
                }
            } else if !crate::index::Index::exists(&self.corpus) {
                let opts = IndexBuildOptions {
                    model_id: cfg.embedder.active.clone(),
                    dim: cfg.embedder.dim,
                    sketch_only: true,
                    chunk_mode: "line".into(),
                    exclude: vec![],
                    ann_enabled: true,
                    ann_min_chunks: 500,
                };
                build_index(&self.corpus, embedder.as_deref(), &opts)?;
            }
        }
        Ok(embedder)
    }

    fn laser_search(&self, query: &str) -> Result<Vec<ChunkRef>> {
        let laser = crate::laser::Laser::new(ParallelGrep, vec![self.corpus.clone()]);
        laser.focus(query, 100)
    }

    fn repo_meta(&self, embedder: Option<&dyn crate::core::traits::Embedder>) -> RepoMeta {
        let index_warm = self.warm_index || crate::index::Index::exists(&self.corpus);
        RepoMeta {
            has_model: embedder.is_some(),
            index_warm,
            ..Default::default()
        }
    }

    fn route_for_mode(&self, mode: EvalMode, query: &str, meta: &RepoMeta) -> Route {
        match mode {
            EvalMode::Grep | EvalMode::Ripgrep | EvalMode::Laser | EvalMode::FixedGrep => {
                Route::Grep
            }
            EvalMode::Vector | EvalMode::Jit => Route::Semantic,
            EvalMode::Hybrid | EvalMode::FixedHybrid => Route::Hybrid,
            EvalMode::Prefocus => Route::Prefocus,
            EvalMode::RouterHeuristic => HeuristicRouter.route(query, meta).route,
            EvalMode::RouterFeature => FeatureRouter.route(query, meta).route,
            EvalMode::RouterLearned => {
                if let Some(cfg) = &self.config {
                    if let Ok(router) = resolve_router(cfg) {
                        return router.route(query, meta).route;
                    }
                }
                HeuristicRouter.route(query, meta).route
            }
        }
    }

    fn oracle_route(&self, query: &EvalQuery) -> Route {
        if self
            .laser_search(&query.query)
            .map(|r| recall_at_k(&r, &query.oracles, 10) > 0.0)
            .unwrap_or(false)
        {
            return Route::Grep;
        }
        if self
            .integrated_search(&query.query, EvalMode::Hybrid, None, &EmbedStatsCell::new())
            .map(|r| recall_at_k(&r, &query.oracles, 10) > 0.0)
            .unwrap_or(false)
        {
            return Route::Hybrid;
        }
        Route::Prefocus
    }

    fn integrated_search(
        &self,
        query: &str,
        mode: EvalMode,
        embedder: Option<&dyn crate::core::traits::Embedder>,
        embed_stats: &EmbedStatsCell,
    ) -> Result<Vec<ChunkRef>> {
        if matches!(mode, EvalMode::Ripgrep) {
            return self.ripgrep_search(query);
        }
        if matches!(mode, EvalMode::Grep | EvalMode::FixedGrep) {
            return self.unix_grep_search(query);
        }

        let cfg = match &self.config {
            Some(c) => c,
            None => {
                return match mode {
                    EvalMode::Laser => self.laser_search(query),
                    EvalMode::Grep | EvalMode::Ripgrep | EvalMode::FixedGrep => unreachable!(),
                    EvalMode::Vector
                    | EvalMode::Hybrid
                    | EvalMode::Jit
                    | EvalMode::Prefocus
                    | EvalMode::FixedHybrid
                    | EvalMode::RouterHeuristic
                    | EvalMode::RouterFeature
                    | EvalMode::RouterLearned => self.parallel_grep_search(query),
                };
            }
        };

        let meta = self.repo_meta(embedder);
        let route = self.route_for_mode(mode, query, &meta);
        let mut search_opts = SearchOptions::from_config(cfg, route);
        if let Some(b) = self.overrides.jit_embed_budget {
            search_opts.jit_embed_budget = b;
        }
        if let Some(c) = self.overrides.jit_reheat_file_cap {
            search_opts.jit_reheat_file_cap = c;
        }
        search_opts.embed_stats = Some(embed_stats.clone());

        if self.warm_index && mode != EvalMode::Jit {
            search_opts.jit_enabled = false;
        }
        if mode == EvalMode::Jit {
            search_opts.jit_enabled = true;
        }

        let scored = hybrid_search(
            query,
            std::slice::from_ref(&self.corpus),
            embedder,
            &search_opts,
        )?;
        Ok(scored.into_iter().map(|s| s.chunk).collect())
    }

    fn run_query(
        &self,
        query: &EvalQuery,
        mode: EvalMode,
        embedder: Option<&dyn crate::core::traits::Embedder>,
        embed_stats: &EmbedStatsCell,
    ) -> Result<(Vec<ChunkRef>, f32, Option<Route>)> {
        let start = Instant::now();
        embed_stats.reset();
        let results = self.integrated_search(&query.query, mode, embedder, embed_stats)?;
        let ms = start.elapsed().as_secs_f32() * 1000.0;
        let chosen_route = if is_router_mode(mode) {
            Some(self.route_for_mode(mode, &query.query, &self.repo_meta(embedder)))
        } else {
            None
        };
        Ok((results, ms, chosen_route))
    }
    /// Evaluate a single query (for trace generation / debugging).
    pub fn eval_single_query(&self, query: &EvalQuery, mode: EvalMode) -> Result<(f32, f32)> {
        let embedder = self.ensure_corpus_index()?;
        let stats = EmbedStatsCell::new();
        let (results, _latency, _) = self.run_query(query, mode, embedder.as_deref(), &stats)?;
        let recall = recall_at_k(&results, &query.oracles, 10);
        let mrr = mrr(&results, &query.oracles);
        Ok((recall, mrr))
    }
}

fn is_router_mode(mode: EvalMode) -> bool {
    matches!(
        mode,
        EvalMode::RouterHeuristic
            | EvalMode::RouterFeature
            | EvalMode::RouterLearned
            | EvalMode::FixedGrep
            | EvalMode::FixedHybrid
    )
}

impl EvalHarness for AgentCodeHarness {
    fn run(&self, mode: EvalMode, _query_set: &str) -> Result<EvalMetrics> {
        let queries = self.filter_queries(self.load_queries()?)?;
        let embedder = self.ensure_corpus_index()?;
        let mut per_category: BTreeMap<String, CategoryMetrics> = BTreeMap::new();
        let mut total_recall = 0.0f32;
        let mut total_mrr = 0.0f32;
        let mut total_hit = 0.0f32;
        let mut total_latency = 0.0f32;
        let mut cold_latency = 0.0f32;
        let mut warm_latency_sum = 0.0f32;
        let mut warm_count = 0usize;
        let mut cumulative_embed_bytes = 0u64;
        let mut per_query_stats: Vec<QueryEmbedStats> = Vec::new();
        let mut route_hits = 0usize;
        let mut route_total = 0usize;
        let session_stats = EmbedStatsCell::new();

        for (i, q) in queries.iter().enumerate() {
            let query_stats = EmbedStatsCell::new();
            let (results, latency, chosen_route) =
                self.run_query(q, mode, embedder.as_deref(), &query_stats)?;
            let snap = query_stats.snapshot();
            cumulative_embed_bytes += snap.bytes_embedded as u64;
            per_query_stats.push(QueryEmbedStats {
                query_id: q.id.clone(),
                chunks_embedded: snap.chunks_embedded,
                bytes_embedded: snap.bytes_embedded,
            });

            if i == 0 {
                cold_latency = latency;
            } else {
                warm_latency_sum += latency;
                warm_count += 1;
            }
            total_latency += latency;

            let recall = recall_at_k(&results, &q.oracles, 10);
            let mrr = mrr(&results, &q.oracles);
            let hit = if recall > 0.0 { 1.0 } else { 0.0 };

            total_recall += recall;
            total_mrr += mrr;
            total_hit += hit;

            if let Some(chosen) = chosen_route {
                let oracle = self.oracle_route(q);
                route_total += 1;
                if routes_match(chosen, oracle) {
                    route_hits += 1;
                }
            }

            let cat = per_category.entry(q.category.clone()).or_default();
            cat.n += 1;
            cat.recall_at_10 += recall;
            cat.mrr += mrr;
        }

        let _ = session_stats;

        let n = queries.len().max(1) as f32;
        for cat in per_category.values_mut() {
            let cn = cat.n.max(1) as f32;
            cat.recall_at_10 /= cn;
            cat.mrr /= cn;
        }

        Ok(EvalMetrics {
            recall_at_10: total_recall / n,
            mrr: total_mrr / n,
            hit_rate: total_hit / n,
            mean_latency_ms: total_latency / n,
            cold_latency_ms: cold_latency,
            warm_latency_ms: if warm_count > 0 {
                warm_latency_sum / warm_count as f32
            } else {
                0.0
            },
            cumulative_embed_bytes,
            per_query: per_query_stats,
            per_category,
            route_accuracy: if route_total > 0 {
                Some(route_hits as f32 / route_total as f32)
            } else {
                None
            },
        })
    }
}

fn routes_match(a: Route, b: Route) -> bool {
    a == b
        || (matches!(a, Route::Grep) && matches!(b, Route::Grep))
        || (matches!(a, Route::Semantic | Route::Hybrid | Route::Prefocus)
            && matches!(b, Route::Semantic | Route::Hybrid | Route::Prefocus))
}

pub fn eval_mode_label(mode: EvalMode) -> &'static str {
    match mode {
        EvalMode::Grep => "grep",
        EvalMode::Ripgrep => "ripgrep",
        EvalMode::Laser => "laser",
        EvalMode::Vector => "vector",
        EvalMode::Hybrid => "hybrid",
        EvalMode::Jit => "jit",
        EvalMode::Prefocus => "prefocus",
        EvalMode::FixedGrep => "fixed-grep",
        EvalMode::FixedHybrid => "fixed-hybrid",
        EvalMode::RouterHeuristic => "router-heuristic",
        EvalMode::RouterFeature => "router-feature",
        EvalMode::RouterLearned => "router-learned",
    }
}

pub fn compare_modes(
    harness: &AgentCodeHarness,
    modes: &[EvalMode],
) -> Result<BTreeMap<String, EvalMetrics>> {
    let mut out = BTreeMap::new();
    for mode in modes {
        if harness.isolate_modes {
            crate::index::Index::purge(&harness.corpus)?;
            harness.ensure_corpus_index()?;
        }
        let key = eval_mode_label(*mode).to_string();
        out.insert(key, harness.run(*mode, "")?);
    }
    Ok(out)
}

#[derive(Debug, serde::Serialize)]
pub struct EvalCompareJson {
    pub modes: BTreeMap<String, EvalMetrics>,
}

/// Serialize compare results for plotting pipelines (`--format json`).
pub fn results_to_json(results: &BTreeMap<String, EvalMetrics>) -> Result<String> {
    let payload = EvalCompareJson {
        modes: results.clone(),
    };
    Ok(serde_json::to_string_pretty(&payload)?)
}

/// Markdown-friendly table for benchmark reports.
pub fn format_report(results: &BTreeMap<String, EvalMetrics>) -> String {
    let mut lines = vec![
        "| mode | recall@10 | mrr | hit_rate | cold_ms | warm_ms | mean_ms | embed_mb |"
            .to_string(),
        "|------|-----------|-----|----------|---------|---------|---------|----------|"
            .to_string(),
    ];
    for (mode, m) in results {
        lines.push(format!(
            "| {mode} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1} | {:.1} | {:.2} |",
            m.recall_at_10,
            m.mrr,
            m.hit_rate,
            m.cold_latency_ms,
            m.warm_latency_ms,
            m.mean_latency_ms,
            m.cumulative_embed_bytes as f64 / 1_048_576.0
        ));
    }
    lines.join("\n")
}

fn hits_to_chunks(hits: &[crate::core::types::GrepHit]) -> Vec<ChunkRef> {
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

fn recall_at_k(results: &[ChunkRef], oracles: &[OracleLocation], k: usize) -> f32 {
    if oracles.is_empty() {
        return 0.0;
    }
    let top = results.iter().take(k);
    let hits = oracles
        .iter()
        .filter(|o| {
            top.clone().any(|r| {
                r.file.ends_with(&o.file)
                    && r.start_line <= o.end_line
                    && r.end_line >= o.start_line
            })
        })
        .count();
    hits as f32 / oracles.len() as f32
}

fn mrr(results: &[ChunkRef], oracles: &[OracleLocation]) -> f32 {
    for (rank, r) in results.iter().enumerate() {
        for o in oracles {
            if r.file.ends_with(&o.file) && r.start_line <= o.end_line && r.end_line >= o.start_line
            {
                return 1.0 / (rank as f32 + 1.0);
            }
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_computes() {
        let results = vec![ChunkRef {
            file: PathBuf::from("src/auth.rs"),
            chunk_id: 0,
            start_line: 10,
            end_line: 20,
            byte_start: 0,
            byte_end: 100,
        }];
        let oracles = vec![OracleLocation {
            file: "auth.rs".into(),
            start_line: 12,
            end_line: 15,
        }];
        assert!(recall_at_k(&results, &oracles, 10) > 0.0);
    }
}
