//! Local route traces for router training (Track 3).

use gp_core::error::{GpError, Result};
use gp_core::types::Route;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTrace {
    pub query: String,
    pub route: String,
    pub latency_ms: f32,
    pub success: Option<bool>,
}

pub fn traces_dir() -> PathBuf {
    gp_core::config::Config::global_config_dir().join("traces")
}

pub fn append_trace(trace: &RouteTrace) -> Result<()> {
    let dir = traces_dir();
    std::fs::create_dir_all(&dir).map_err(|e| GpError::Io(e))?;
    let path = dir.join("routes.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| GpError::Io(e))?;
    let line = serde_json::to_string(trace)?;
    writeln!(file, "{line}").map_err(|e| GpError::Io(e))?;
    Ok(())
}

pub fn load_traces(path: &Path) -> Result<Vec<RouteTrace>> {
    let raw = std::fs::read_to_string(path).map_err(|e| GpError::Io(e))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line)?);
    }
    Ok(out)
}

pub fn route_label(route: Route) -> &'static str {
    match route {
        Route::Grep => "grep",
        Route::Semantic => "semantic",
        Route::Hybrid => "hybrid",
        Route::Prefocus => "prefocus",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_trace_line() {
        let t = RouteTrace {
            query: "test".into(),
            route: "grep".into(),
            latency_ms: 1.0,
            success: Some(true),
        };
        let line = serde_json::to_string(&t).expect("json");
        let parsed: RouteTrace = serde_json::from_str(&line).expect("parse");
        assert_eq!(parsed.query, "test");
    }
}
