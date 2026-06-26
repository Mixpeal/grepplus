use gp_core::error::{GpError, Result};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn init_runtime() {
    INIT.call_once(|| {
        let _ = ort::init().with_name("grepplus").commit();
    });
}

fn try_open_session(model_path: &Path, level: GraphOptimizationLevel) -> Result<Session> {
    Session::builder()
        .map_err(|e| GpError::Model(e.to_string()))?
        .with_optimization_level(level)
        .map_err(|e| GpError::Model(e.to_string()))?
        .commit_from_file(model_path)
        .map_err(|e| GpError::Model(e.to_string()))
}

/// Some community ONNX exports (e.g. FP16 BERT-style) break under full ORT graph fusion.
pub fn open_session(model_path: &Path) -> Result<Session> {
    init_runtime();
    if let Ok(session) = try_open_session(model_path, GraphOptimizationLevel::Level1) {
        return Ok(session);
    }
    try_open_session(model_path, GraphOptimizationLevel::Disable)
}
