use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A pointer to a region of source. The universal currency of the system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRef {
    pub file: PathBuf,
    pub chunk_id: u32,
    pub start_line: u32,
    pub end_line: u32,
    pub byte_start: u64,
    pub byte_end: u64,
}

/// A scored result. `score` is comparable only within a single retriever's
/// output unless normalized by Fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChunk {
    pub chunk: ChunkRef,
    pub score: f32,
    pub source: RetrievalSource,
    /// Optional matched text/preview for display.
    pub preview: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalSource {
    Grep,
    Laser,
    Sketch,
    #[serde(alias = "Pq4")]
    Vector,
    Fused,
}

/// A raw grep hit (line-level). Distinct from ChunkRef so the grep engine
/// stays index-free.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrepHit {
    pub file: PathBuf,
    pub line_no: u32,
    pub byte_offset: u64,
    pub line: String,
    pub match_start: usize,
    pub match_end: usize,
}

/// Router output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub route: Route,
    pub confidence: f32,
    pub rationale: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    Grep,
    Semantic,
    Hybrid,
    Prefocus,
}

/// Metadata about a repo, passed to the router.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepoMeta {
    pub languages: Vec<String>,
    pub file_count: usize,
    pub index_warm: bool,
    pub has_model: bool,
}
