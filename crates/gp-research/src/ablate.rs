use gp_core::traits::{EvalHarness, EvalMode};
use gp_eval::AgentCodeHarness;
use gp_index::Index;
use gp_search::{build_index, IndexBuildOptions};
use std::collections::BTreeMap;

use crate::triplets::train_pq4;

/// Ablation: baseline vs pca vs learned pq4 on eval suite.
pub fn ablate_projections(
    harness: &AgentCodeHarness,
    projections: &[&str],
) -> Result<BTreeMap<String, gp_core::traits::EvalMetrics>, gp_core::error::GpError> {
    let mut out = BTreeMap::new();
    let corpus = harness.corpus.clone();
    let cfg = harness
        .config
        .clone()
        .ok_or_else(|| gp_core::error::GpError::Config("harness missing config".into()))?;

    for proj in projections {
        Index::purge(&corpus)?;
        let mut cfg = cfg.clone();
        let embedder = gp_embed::resolve_embedder(
            &mut cfg,
            &gp_embed::EnsureOptions::default(),
        )?;
        let Some(emb) = embedder.as_deref() else {
            return Err(gp_core::error::GpError::NoModel);
        };
        if *proj == "pq4" {
            train_pq4(&corpus, &harness.queries_path, emb, cfg.embedder.dim)?;
        }
        let opts = IndexBuildOptions {
            model_id: cfg.embedder.active.clone(),
            dim: cfg.embedder.dim,
            projection: (*proj).into(),
            sketch_only: false,
        };
        build_index(&corpus, Some(emb), &opts)?;
        let key = format!("proj-{proj}");
        out.insert(key, harness.run(EvalMode::Vector, "")?);
    }
    Ok(out)
}

pub fn format_ablation(results: &BTreeMap<String, gp_core::traits::EvalMetrics>) -> String {
    let mut lines = vec![
        "| projection | recall@10 | mrr | mean_ms |".to_string(),
        "|------------|-----------|-----|---------|".to_string(),
    ];
    for (proj, m) in results {
        lines.push(format!(
            "| {proj} | {:.3} | {:.3} | {:.1} |",
            m.recall_at_10, m.mrr, m.mean_latency_ms
        ));
    }
    lines.join("\n")
}
