use crate::core::error::{GpError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub embedder: EmbedderCfg,
    #[serde(default)]
    pub index: IndexCfg,
    #[serde(default)]
    pub router: RouterCfg,
    #[serde(default)]
    pub search: SearchCfg,
    #[serde(default)]
    pub grep: GrepCfg,
    #[serde(default)]
    pub research: ResearchCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedderCfg {
    pub active: String,
    pub dim: usize,
    pub query_instruct: String,
}

impl Default for EmbedderCfg {
    fn default() -> Self {
        Self {
            active: "qwen3-embedding-0.6b".into(),
            dim: 256,
            query_instruct: "Given a code search query, retrieve relevant source passages".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexCfg {
    pub auto_ensure: bool,
    pub sketch: String,
    pub exclude: Vec<String>,
    /// Drop cached indexes not accessed within this many days (stored under ~/.grepplus/cache/).
    pub cache_ttl_days: u32,
    /// `line` or `ast` chunking strategy.
    pub chunk_mode: String,
    /// Build ANN graph when chunk count exceeds this threshold.
    pub ann_min_chunks: usize,
}

impl Default for IndexCfg {
    fn default() -> Self {
        Self {
            auto_ensure: false,
            sketch: "beam".into(),
            exclude: vec!["node_modules".into(), "target".into(), ".git".into()],
            cache_ttl_days: 7,
            chunk_mode: "line".into(),
            ann_min_chunks: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RouterCfg {
    pub mode: String,
    pub contrib_traces: bool,
    /// Path to learned router weights (relative to ~/.grepplus/ or absolute).
    pub model_path: String,
}

impl Default for RouterCfg {
    fn default() -> Self {
        Self {
            mode: "heuristic".into(),
            contrib_traces: false,
            model_path: "router/model.json".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchCfg {
    pub fusion: String,
    pub laser_candidate_cap: usize,
    pub sketch_beam_width: usize,
    /// JIT embed: max chunk embeds per query (reheat cold/cool files in beam).
    pub jit_embed_budget: usize,
    /// Max cold/cool files to reheat per query (beam order).
    pub jit_reheat_file_cap: usize,
    /// On a mostly-cold index, cap first-query reheat (faster cold@1).
    pub jit_cold_first_file_cap: usize,
    pub jit_cold_first_embed_budget: usize,
    /// Use temperature-aware JIT search (skip HOT, reheat COLD).
    pub jit_enabled: bool,
    /// Use ANN graph when index has ann/ built.
    pub ann_enabled: bool,
}

impl Default for SearchCfg {
    fn default() -> Self {
        Self {
            fusion: "rrf".into(),
            laser_candidate_cap: 500,
            sketch_beam_width: 50,
            jit_embed_budget: 64,
            jit_reheat_file_cap: 16,
            jit_cold_first_file_cap: 6,
            jit_cold_first_embed_budget: 18,
            jit_enabled: true,
            ann_enabled: true,
        }
    }
}

/// Lexical search backend selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GrepCfg {
    /// `parallel` (default, in-process), `ripgrep` (external rg), or `auto`
    /// (parallel for CLI default grep; ripgrep for hybrid exact channel when available).
    pub backend: String,
}

impl Default for GrepCfg {
    fn default() -> Self {
        Self {
            backend: "parallel".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ResearchCfg {
    pub eval_corpus: String,
}

impl Default for ResearchCfg {
    fn default() -> Self {
        Self {
            eval_corpus: "./eval/agentcode".into(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let mut cfg = Self::default();
        if let Some(global) = global_config_path() {
            if global.exists() {
                merge_file(&mut cfg, &global)?;
            }
        }
        let local = PathBuf::from(".grepplus.toml");
        if local.exists() {
            merge_file(&mut cfg, &local)?;
        }
        Ok(cfg)
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let mut cfg = Self::default();
        merge_file(&mut cfg, path)?;
        Ok(cfg)
    }

    pub fn models_dir() -> PathBuf {
        home_dir()
            .map(|h| h.join(".grepplus").join("models"))
            .unwrap_or_else(|| PathBuf::from(".grepplus/models"))
    }

    pub fn global_config_dir() -> PathBuf {
        home_dir()
            .map(|h| h.join(".grepplus"))
            .unwrap_or_else(|| PathBuf::from(".grepplus"))
    }

    pub fn global_config_path() -> PathBuf {
        Self::global_config_dir().join("config.toml")
    }

    /// User-level cache root (`~/.grepplus/cache/`). Override with `GREPPLUS_CACHE_DIR`.
    pub fn cache_dir() -> PathBuf {
        std::env::var_os("GREPPLUS_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| Self::global_config_dir().join("cache"))
    }

    /// Update `embedder.active` in the global config file.
    pub fn set_active_embedder(id: &str) -> Result<()> {
        let mut cfg = Self::load().unwrap_or_default();
        cfg.embedder.active = id.to_string();
        cfg.save_global()
    }

    pub fn save_global(&self) -> Result<()> {
        let dir = Self::global_config_dir();
        std::fs::create_dir_all(&dir)?;
        let path = Self::global_config_path();
        let body = toml::to_string_pretty(self).map_err(|e| GpError::Config(e.to_string()))?;
        std::fs::write(path, body)?;
        Ok(())
    }
}

/// Partial overlay so missing keys keep existing defaults (deep-merge per section).
#[derive(Debug, Default, Deserialize)]
struct ConfigOverlay {
    #[serde(default)]
    embedder: Option<EmbedderOverlay>,
    #[serde(default)]
    index: Option<IndexOverlay>,
    #[serde(default)]
    router: Option<RouterOverlay>,
    #[serde(default)]
    search: Option<SearchOverlay>,
    #[serde(default)]
    grep: Option<GrepOverlay>,
    #[serde(default)]
    research: Option<ResearchOverlay>,
}

#[derive(Debug, Default, Deserialize)]
struct EmbedderOverlay {
    active: Option<String>,
    dim: Option<usize>,
    query_instruct: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct IndexOverlay {
    auto_ensure: Option<bool>,
    sketch: Option<String>,
    exclude: Option<Vec<String>>,
    cache_ttl_days: Option<u32>,
    chunk_mode: Option<String>,
    ann_min_chunks: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct RouterOverlay {
    mode: Option<String>,
    contrib_traces: Option<bool>,
    model_path: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct SearchOverlay {
    fusion: Option<String>,
    laser_candidate_cap: Option<usize>,
    sketch_beam_width: Option<usize>,
    jit_embed_budget: Option<usize>,
    jit_reheat_file_cap: Option<usize>,
    jit_cold_first_file_cap: Option<usize>,
    jit_cold_first_embed_budget: Option<usize>,
    jit_enabled: Option<bool>,
    ann_enabled: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct GrepOverlay {
    backend: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ResearchOverlay {
    eval_corpus: Option<String>,
}

fn merge_file(cfg: &mut Config, path: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(path)?;
    let overlay: ConfigOverlay =
        toml::from_str(&raw).map_err(|e| GpError::Config(format!("{}: {e}", path.display())))?;
    if let Some(e) = overlay.embedder {
        if let Some(v) = e.active {
            cfg.embedder.active = v;
        }
        if let Some(v) = e.dim {
            cfg.embedder.dim = v;
        }
        if let Some(v) = e.query_instruct {
            cfg.embedder.query_instruct = v;
        }
    }
    if let Some(i) = overlay.index {
        if let Some(v) = i.auto_ensure {
            cfg.index.auto_ensure = v;
        }
        if let Some(v) = i.sketch {
            cfg.index.sketch = v;
        }
        if let Some(v) = i.exclude {
            cfg.index.exclude = v;
        }
        if let Some(v) = i.cache_ttl_days {
            cfg.index.cache_ttl_days = v;
        }
        if let Some(v) = i.chunk_mode {
            cfg.index.chunk_mode = v;
        }
        if let Some(v) = i.ann_min_chunks {
            cfg.index.ann_min_chunks = v;
        }
    }
    if let Some(r) = overlay.router {
        if let Some(v) = r.mode {
            cfg.router.mode = v;
        }
        if let Some(v) = r.contrib_traces {
            cfg.router.contrib_traces = v;
        }
        if let Some(v) = r.model_path {
            cfg.router.model_path = v;
        }
    }
    if let Some(s) = overlay.search {
        if let Some(v) = s.fusion {
            cfg.search.fusion = v;
        }
        if let Some(v) = s.laser_candidate_cap {
            cfg.search.laser_candidate_cap = v;
        }
        if let Some(v) = s.sketch_beam_width {
            cfg.search.sketch_beam_width = v;
        }
        if let Some(v) = s.jit_embed_budget {
            cfg.search.jit_embed_budget = v;
        }
        if let Some(v) = s.jit_reheat_file_cap {
            cfg.search.jit_reheat_file_cap = v;
        }
        if let Some(v) = s.jit_cold_first_file_cap {
            cfg.search.jit_cold_first_file_cap = v;
        }
        if let Some(v) = s.jit_cold_first_embed_budget {
            cfg.search.jit_cold_first_embed_budget = v;
        }
        if let Some(v) = s.jit_enabled {
            cfg.search.jit_enabled = v;
        }
        if let Some(v) = s.ann_enabled {
            cfg.search.ann_enabled = v;
        }
    }
    if let Some(g) = overlay.grep {
        if let Some(v) = g.backend {
            cfg.grep.backend = v;
        }
    }
    if let Some(r) = overlay.research {
        if let Some(v) = r.eval_corpus {
            cfg.research.eval_corpus = v;
        }
    }
    Ok(())
}

/// Turn config exclude names into globs for grep backends.
pub fn exclude_to_globs(exclude: &[String]) -> Vec<String> {
    exclude
        .iter()
        .map(|name| {
            if name.contains('*') || name.contains('/') {
                name.clone()
            } else {
                format!("**/{name}/**")
            }
        })
        .collect()
}

fn global_config_path() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".grepplus").join("config.toml"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let cfg = Config::default();
        assert_eq!(cfg.embedder.active, "qwen3-embedding-0.6b");
        assert_eq!(cfg.search.laser_candidate_cap, 500);
        assert_eq!(cfg.search.fusion, "rrf");
    }

    #[test]
    fn partial_toml_deep_merges_search_section() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("partial.toml");
        std::fs::write(&path, "[search]\nfusion = \"rrf\"\n").unwrap();
        let mut cfg = Config::default();
        merge_file(&mut cfg, &path).unwrap();
        assert_eq!(cfg.search.fusion, "rrf");
        assert_eq!(cfg.search.laser_candidate_cap, 500);
        assert_eq!(cfg.embedder.dim, 256);
    }
}
