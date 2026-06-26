use gp_core::config::Config;
use gp_core::error::Result;
use gp_core::traits::{CategoryMetrics, EvalHarness, EvalMetrics, EvalMode, GrepEngine, GrepOptions, LaserFocus};
use gp_core::types::{ChunkRef, Route};
use gp_embed::{resolve_embedder, EnsureOptions};
use gp_grep::{ParallelGrep, RipgrepEngine, UnixGrepEngine};
use gp_search::{build_index, hybrid_search, IndexBuildOptions, SearchOptions};
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

pub struct AgentCodeHarness {
    pub corpus: PathBuf,
    pub queries_path: PathBuf,
    pub config: Option<Config>,
    pub ensure_index: bool,
    pub warm_index: bool,
    pub isolate_modes: bool,
    pub yes_download: bool,
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

    fn load_queries(&self) -> Result<Vec<EvalQuery>> {
        load_queries(&self.queries_path)
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

    fn ensure_corpus_index(&self) -> Result<Option<Arc<dyn gp_core::traits::Embedder>>> {
        let cfg = match &self.config {
            Some(c) => c.clone(),
            None => return Ok(None),
        };
        let mut cfg = cfg;
        let opts = EnsureOptions::for_required_semantic(self.yes_download);
        let embedder = resolve_embedder(&mut cfg, &opts).ok().flatten();
        if self.ensure_index && embedder.is_some() {
            if self.warm_index {
                if !gp_index::Index::exists(&self.corpus) {
                    let opts = IndexBuildOptions {
                        model_id: cfg.embedder.active.clone(),
                        dim: cfg.embedder.dim,
                        projection: cfg.index.projection.clone(),
                        sketch_only: false,
                    };
                    build_index(&self.corpus, embedder.as_deref(), &opts)?;
                }
            } else if !gp_index::Index::exists(&self.corpus) {
                let opts = IndexBuildOptions {
                    model_id: cfg.embedder.active.clone(),
                    dim: cfg.embedder.dim,
                    projection: cfg.index.projection.clone(),
                    sketch_only: true,
                };
                build_index(&self.corpus, embedder.as_deref(), &opts)?;
            }
        }
        Ok(embedder)
    }

    fn laser_search(&self, query: &str) -> Result<Vec<ChunkRef>> {
        let laser = gp_laser::Laser::new(ParallelGrep, vec![self.corpus.clone()]);
        Ok(laser.focus(query, 100)?)
    }

    fn integrated_search(
        &self,
        query: &str,
        mode: EvalMode,
        embedder: Option<&dyn gp_core::traits::Embedder>,
    ) -> Result<Vec<ChunkRef>> {
        if matches!(mode, EvalMode::Ripgrep) {
            return self.ripgrep_search(query);
        }
        if matches!(mode, EvalMode::Grep) {
            return self.unix_grep_search(query);
        }

        let cfg = match &self.config {
            Some(c) => c,
            None => {
                return match mode {
                    EvalMode::Laser => self.laser_search(query),
                    EvalMode::Grep | EvalMode::Ripgrep => unreachable!(),
                    EvalMode::Vector | EvalMode::Hybrid | EvalMode::Jit => {
                        self.parallel_grep_search(query)
                    }
                };
            }
        };

        let route = match mode {
            EvalMode::Grep | EvalMode::Ripgrep => unreachable!(),
            EvalMode::Laser => Route::Grep,
            EvalMode::Vector => Route::Semantic,
            EvalMode::Hybrid => Route::Hybrid,
            EvalMode::Jit => Route::Semantic,
        };
        let mut search_opts = SearchOptions::from_config(cfg, route);
        if self.warm_index && mode != EvalMode::Jit {
            search_opts.jit_enabled = false;
        }
        if mode == EvalMode::Jit {
            search_opts.jit_enabled = true;
        }
        let scored = hybrid_search(
            query,
            &[self.corpus.clone()],
            embedder,
            &search_opts,
        )?;
        Ok(scored.into_iter().map(|s| s.chunk).collect())
    }

    fn run_query(
        &self,
        query: &EvalQuery,
        mode: EvalMode,
        embedder: Option<&dyn gp_core::traits::Embedder>,
    ) -> Result<(Vec<ChunkRef>, f32)> {
        let start = Instant::now();
        let results = self.integrated_search(&query.query, mode, embedder)?;
        let ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok((results, ms))
    }
}

impl EvalHarness for AgentCodeHarness {
    fn run(&self, mode: EvalMode, _query_set: &str) -> Result<EvalMetrics> {
        let queries = self.load_queries()?;
        let embedder = self.ensure_corpus_index()?;
        let mut per_category: BTreeMap<String, CategoryMetrics> = BTreeMap::new();
        let mut total_recall = 0.0f32;
        let mut total_mrr = 0.0f32;
        let mut total_success = 0.0f32;
        let mut total_latency = 0.0f32;
        let mut cold_latency = 0.0f32;
        let mut warm_latency_sum = 0.0f32;
        let mut warm_count = 0usize;

        for (i, q) in queries.iter().enumerate() {
            let (results, latency) = self.run_query(q, mode, embedder.as_deref())?;
            if i == 0 {
                cold_latency = latency;
            } else {
                warm_latency_sum += latency;
                warm_count += 1;
            }
            total_latency += latency;

            let recall = recall_at_k(&results, &q.oracles, 10);
            let mrr = mrr(&results, &q.oracles);
            let success = if recall > 0.0 { 1.0 } else { 0.0 };

            total_recall += recall;
            total_mrr += mrr;
            total_success += success;

            let cat = per_category.entry(q.category.clone()).or_default();
            cat.n += 1;
            cat.recall_at_10 += recall;
            cat.mrr += mrr;
        }

        let n = queries.len().max(1) as f32;
        for cat in per_category.values_mut() {
            let cn = cat.n.max(1) as f32;
            cat.recall_at_10 /= cn;
            cat.mrr /= cn;
        }

        Ok(EvalMetrics {
            recall_at_10: total_recall / n,
            mrr: total_mrr / n,
            success_rate: total_success / n,
            mean_latency_ms: total_latency / n,
            cold_latency_ms: cold_latency,
            warm_latency_ms: if warm_count > 0 {
                warm_latency_sum / warm_count as f32
            } else {
                0.0
            },
            per_category,
        })
    }
}

pub fn eval_mode_label(mode: EvalMode) -> &'static str {
    match mode {
        EvalMode::Grep => "grep",
        EvalMode::Ripgrep => "ripgrep",
        EvalMode::Laser => "laser",
        EvalMode::Vector => "vector",
        EvalMode::Hybrid => "hybrid",
        EvalMode::Jit => "jit",
    }
}

pub fn compare_modes(
    harness: &AgentCodeHarness,
    modes: &[EvalMode],
) -> Result<BTreeMap<String, EvalMetrics>> {
    let mut out = BTreeMap::new();
    for mode in modes {
        if harness.isolate_modes {
            gp_index::Index::purge(&harness.corpus)?;
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
        "| mode | recall@10 | mrr | cold_ms | warm_ms | mean_ms |".to_string(),
        "|------|-----------|-----|---------|---------|---------|".to_string(),
    ];
    for (mode, m) in results {
        lines.push(format!(
            "| {mode} | {:.3} | {:.3} | {:.1} | {:.1} | {:.1} |",
            m.recall_at_10, m.mrr, m.cold_latency_ms, m.warm_latency_ms, m.mean_latency_ms
        ));
    }
    lines.join("\n")
}

fn hits_to_chunks(hits: &[gp_core::types::GrepHit]) -> Vec<ChunkRef> {
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
            if r.file.ends_with(&o.file)
                && r.start_line <= o.end_line
                && r.end_line >= o.start_line
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
