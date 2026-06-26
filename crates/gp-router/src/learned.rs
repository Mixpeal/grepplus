//! Learned linear router (Track 3 phase C).

use crate::features::{QueryFeatures, QueryFeatures as Qf};
use gp_core::error::{GpError, Result};
use gp_core::traits::Router;
use gp_core::types::{RepoMeta, Route, RouteDecision};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedRouterModel {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub labels: Vec<String>,
}

impl LearnedRouterModel {
    pub fn predict_route(&self, features: &QueryFeatures) -> Route {
        let x = features.to_vector();
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, w) in self.weights.iter().enumerate() {
            let mut s = self.bias.get(i).copied().unwrap_or(0.0);
            for (j, &xj) in x.iter().enumerate() {
                s += w.get(j).copied().unwrap_or(0.0) * xj;
            }
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        label_to_route(self.labels.get(best).map(|s| s.as_str()).unwrap_or("grep"))
    }
}

pub struct LearnedRouter {
    model: LearnedRouterModel,
}

impl LearnedRouter {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| GpError::Io(e))?;
        let model: LearnedRouterModel = serde_json::from_str(&raw)?;
        Ok(Self { model })
    }
}

impl Router for LearnedRouter {
    fn route(&self, query: &str, meta: &RepoMeta) -> RouteDecision {
        let features = QueryFeatures::extract(query, meta);
        let route = self.model.predict_route(&features);
        RouteDecision {
            route,
            confidence: 0.75,
            rationale: vec!["learned router".into()],
        }
    }
}

/// Train a simple one-vs-rest linear model from labeled traces.
pub fn train_from_traces(traces: &[crate::trace::RouteTrace]) -> Result<LearnedRouterModel> {
    if traces.is_empty() {
        return Err(GpError::Training("no traces".into()));
    }
    let labels = vec![
        "grep".into(),
        "semantic".into(),
        "hybrid".into(),
        "prefocus".into(),
    ];
    let dim = Qf::DIM;
    let mut weights = vec![vec![0.0f32; dim]; labels.len()];
    let mut bias = vec![0.0f32; labels.len()];
    let lr = 0.01f32;

    for t in traces {
        let weight = if t.success == Some(false) { 0.25 } else { 1.0 };
        let route = label_to_route(&t.route);
        let meta = RepoMeta {
            has_model: true,
            index_warm: true,
            ..Default::default()
        };
        let f = QueryFeatures::extract(&t.query, &meta);
        let x = f.to_vector();
        let target = route_index(route);
        for (i, w) in weights.iter_mut().enumerate() {
            let y = if i == target { 1.0 } else { 0.0 };
            let mut score = bias[i];
            for (j, &xj) in x.iter().enumerate() {
                score += w[j] * xj;
            }
            let pred = 1.0 / (1.0 + (-score).exp());
            let err = pred - y;
            bias[i] -= lr * err * weight;
            for (j, wj) in w.iter_mut().enumerate() {
                *wj -= lr * err * x[j] * weight;
            }
        }
    }

    Ok(LearnedRouterModel {
        weights,
        bias,
        labels,
    })
}

pub fn save_model(path: &Path, model: &LearnedRouterModel) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| GpError::Io(e))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(model)?).map_err(|e| GpError::Io(e))?;
    Ok(())
}

fn route_index(route: Route) -> usize {
    match route {
        Route::Grep => 0,
        Route::Semantic => 1,
        Route::Hybrid => 2,
        Route::Prefocus => 3,
    }
}

fn label_to_route(label: &str) -> Route {
    match label {
        "semantic" => Route::Semantic,
        "hybrid" => Route::Hybrid,
        "prefocus" => Route::Prefocus,
        _ => Route::Grep,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::RouteTrace;

    #[test]
    fn trains_from_traces() {
        let traces = vec![
            RouteTrace {
                query: "handleFoo".into(),
                route: "grep".into(),
                latency_ms: 1.0,
                success: Some(true),
            },
            RouteTrace {
                query: "where is auth logic".into(),
                route: "hybrid".into(),
                latency_ms: 2.0,
                success: Some(true),
            },
        ];
        let model = train_from_traces(&traces).expect("train");
        let symbol = model.predict_route(&QueryFeatures::extract(
            "handleFoo",
            &RepoMeta::default(),
        ));
        let nl = model.predict_route(&QueryFeatures::extract(
            "where is auth logic",
            &RepoMeta {
                has_model: true,
                index_warm: true,
                ..Default::default()
            },
        ));
        assert!(matches!(symbol, Route::Grep | Route::Hybrid));
        assert!(matches!(nl, Route::Semantic | Route::Hybrid | Route::Prefocus));
    }
}
