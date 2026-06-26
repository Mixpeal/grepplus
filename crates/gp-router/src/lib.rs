mod features;
mod learned;
mod trace;

use features::feature_route;
use gp_core::config::Config;
use gp_core::error::Result;
use gp_core::query;
use gp_core::traits::Router;
use gp_core::types::{RepoMeta, Route, RouteDecision};
use learned::LearnedRouter;
use std::path::Path;

pub use features::QueryFeatures;
pub use learned::{save_model, train_from_traces as train_router};
pub use trace::{append_trace, load_traces, route_label, traces_dir, RouteTrace};

pub struct HeuristicRouter;

impl Router for HeuristicRouter {
    fn route(&self, query: &str, meta: &RepoMeta) -> RouteDecision {
        let mut rationale = Vec::new();

        if query::has_regex_metacharacters(query) {
            rationale.push("regex metacharacters present".into());
            return RouteDecision {
                route: Route::Grep,
                confidence: 0.9,
                rationale,
            };
        }

        if query::is_literal_query(query) {
            rationale.push("literal identifier — grep fast path".into());
            return RouteDecision {
                route: Route::Grep,
                confidence: 0.92,
                rationale,
            };
        }

        if query::is_quoted(query) {
            rationale.push("quoted string".into());
            return RouteDecision {
                route: Route::Grep,
                confidence: 0.8,
                rationale,
            };
        }

        if query::has_natural_language_cue(query) {
            rationale.push("natural language query".into());
            let route = if meta.index_warm && meta.has_model {
                Route::Hybrid
            } else if meta.has_model {
                Route::Semantic
            } else {
                Route::Prefocus
            };
            return RouteDecision {
                route,
                confidence: 0.7,
                rationale,
            };
        }

        if query.split_whitespace().count() >= 2 && meta.has_model {
            rationale.push("multi-word query with model available".into());
            return RouteDecision {
                route: Route::Hybrid,
                confidence: 0.6,
                rationale,
            };
        }

        rationale.push("default to grep".into());
        RouteDecision {
            route: Route::Grep,
            confidence: 0.5,
            rationale,
        }
    }
}

pub struct FeatureRouter;

impl Router for FeatureRouter {
    fn route(&self, query: &str, meta: &RepoMeta) -> RouteDecision {
        let features = QueryFeatures::extract(query, meta);
        let route = feature_route(&features);
        RouteDecision {
            route,
            confidence: 0.65,
            rationale: vec!["feature router".into()],
        }
    }
}

/// Resolve router implementation from config.
pub fn resolve_router(cfg: &Config) -> Result<Box<dyn Router>> {
    match cfg.router.mode.as_str() {
        "learned" => {
            let path = router_model_path(cfg);
            if path.exists() {
                Ok(Box::new(LearnedRouter::load(&path)?))
            } else {
                Ok(Box::new(FeatureRouter))
            }
        }
        "feature" => Ok(Box::new(FeatureRouter)),
        _ => Ok(Box::new(HeuristicRouter)),
    }
}

pub fn router_model_path(cfg: &Config) -> std::path::PathBuf {
    let p = Path::new(&cfg.router.model_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        Config::global_config_dir().join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_symbol_to_grep() {
        let r = HeuristicRouter;
        let d = r.route("handleSessionRefresh", &RepoMeta::default());
        assert_eq!(d.route, Route::Grep);
    }

    #[test]
    fn routes_nl_to_semantic_when_model_available() {
        let r = HeuristicRouter;
        let d = r.route(
            "where is retry logic",
            &RepoMeta {
                has_model: true,
                ..Default::default()
            },
        );
        assert!(matches!(d.route, Route::Semantic | Route::Hybrid | Route::Prefocus));
    }

    #[test]
    fn feature_router_differs_on_nl() {
        let r = FeatureRouter;
        let d = r.route(
            "where is retry logic",
            &RepoMeta {
                has_model: true,
                ..Default::default()
            },
        );
        assert_ne!(d.route, Route::Grep);
    }
}
