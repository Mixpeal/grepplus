use gp_core::traits::EvalMode;
use gp_core::types::Route;
use gp_eval::{load_queries, AgentCodeHarness};
use gp_router::{route_label, traces_dir, RouteTrace};
use std::path::{Path, PathBuf};

/// Generate synthetic oracle-route traces from eval suite (one row per query).
pub fn generate_router_traces(
    harness: &AgentCodeHarness,
    output: Option<&Path>,
) -> Result<PathBuf, gp_core::error::GpError> {
    let queries = load_queries(&harness.queries_path)?;
    let out = output
        .map(PathBuf::from)
        .unwrap_or_else(|| traces_dir().join("agentcode_routes.jsonl"));

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(gp_core::error::GpError::Io)?;
    }

    let mut lines = Vec::new();
    for q in &queries {
        let (laser_recall, _) = harness.eval_single_query(q, EvalMode::Laser)?;
        let (hybrid_recall, _) = harness.eval_single_query(q, EvalMode::Hybrid)?;

        let route = if laser_recall > 0.0 {
            Route::Grep
        } else if hybrid_recall > 0.0 {
            Route::Hybrid
        } else {
            Route::Prefocus
        };

        let trace = RouteTrace {
            query: q.query.clone(),
            route: route_label(route).to_string(),
            latency_ms: 0.0,
            success: Some(hybrid_recall > 0.0),
        };
        lines.push(
            serde_json::to_string(&trace)
                .map_err(|e| gp_core::error::GpError::Config(e.to_string()))?,
        );
    }

    std::fs::write(&out, lines.join("\n") + "\n").map_err(gp_core::error::GpError::Io)?;
    Ok(out)
}
