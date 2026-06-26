use crate::error::Result;
use crate::types::*;

/// Track 3. Decides where a query goes.
pub trait Router: Send + Sync {
    fn route(&self, query: &str, meta: &RepoMeta) -> RouteDecision;
}

/// Lexical search. Index-free. (gp-grep)
pub trait GrepEngine: Send + Sync {
    fn search(&self, pattern: &str, opts: &GrepOptions) -> Result<Vec<GrepHit>>;
}

/// Laser focus: shrink corpus → candidate chunks via expanded lexical search.
pub trait LaserFocus: Send + Sync {
    fn focus(&self, query: &str, cap: usize) -> Result<Vec<ChunkRef>>;
}

/// Track 4. Semantic pre-focus when laser returns empty.
pub trait PreFocus: Send + Sync {
    fn sketch_beam(
        &self,
        query: &str,
        beam_width: usize,
        cap: usize,
    ) -> Result<Vec<ChunkRef>>;
}

/// Embedding model. (gp-embed)
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
    fn dim(&self) -> usize;
    fn model_id(&self) -> &str;
}

/// Track 2. Quantized embedding code stored in the index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Q4Code {
    pub bytes: Vec<u8>,
    pub dim: u16,
    pub scale: f32,
    pub bias: f32,
}

/// Fuse lexical + semantic hits into one ranked list.
pub trait Fusion: Send + Sync {
    fn fuse(&self, grep: Vec<ScoredChunk>, semantic: Vec<ScoredChunk>) -> Vec<ScoredChunk>;
}

/// Track 1. Eval harness contract.
pub trait EvalHarness {
    fn run(&self, mode: EvalMode, query_set: &str) -> Result<EvalMetrics>;
}

#[derive(Debug, Clone, Default)]
pub struct GrepOptions {
    pub case_insensitive: bool,
    pub fixed_string: bool,
    pub roots: Vec<std::path::PathBuf>,
    pub include_globs: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub context_lines: usize,
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    /// POSIX/BSD/GNU `/usr/bin/grep` subprocess baseline.
    Grep,
    Ripgrep,
    Laser,
    Vector,
    Hybrid,
    /// JIT: sketch shell + temperature-aware embed (no full warm index required).
    Jit,
    /// Track 4: SketchBeam pre-focus path only.
    Prefocus,
    /// Track 3: always route to grep.
    FixedGrep,
    /// Track 3: always route to hybrid.
    FixedHybrid,
    /// Track 3: heuristic router per query.
    RouterHeuristic,
    /// Track 3: feature router per query.
    RouterFeature,
    /// Track 3: learned router per query.
    RouterLearned,
}

/// Per-query JIT embed accounting (fp32 bytes = chunks × dim × 4).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct QueryEmbedStats {
    pub query_id: String,
    pub chunks_embedded: usize,
    pub bytes_embedded: usize,
}

/// Session-level embed accounting for JIT economics curves.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct EmbedAccounting {
    pub chunks_embedded: usize,
    pub bytes_embedded: usize,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct EvalMetrics {
    pub recall_at_10: f32,
    pub mrr: f32,
    /// Fraction of queries with recall > 0 (retrieval proxy).
    #[serde(alias = "success_rate")]
    pub hit_rate: f32,
    pub mean_latency_ms: f32,
    /// Latency of the first query (cold JIT / cache miss proxy).
    pub cold_latency_ms: f32,
    /// Mean latency excluding the first query (session-warm proxy).
    pub warm_latency_ms: f32,
    /// Cumulative fp32 embed bytes across the eval session.
    pub cumulative_embed_bytes: u64,
    /// Per-query embed stats (JIT economics).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub per_query: Vec<QueryEmbedStats>,
    pub per_category: std::collections::BTreeMap<String, CategoryMetrics>,
    /// Router ablation: fraction of queries where chosen route matched oracle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_accuracy: Option<f32>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CategoryMetrics {
    pub recall_at_10: f32,
    pub mrr: f32,
    pub n: usize,
}
