//! Query feature extraction for routing (Track 3 phase B).

use crate::core::query;
use crate::core::types::{RepoMeta, Route};

#[derive(Debug, Clone)]
pub struct QueryFeatures {
    pub word_count: f32,
    pub char_len: f32,
    pub has_camel_case: f32,
    pub has_regex: f32,
    pub has_nl_cue: f32,
    pub is_literal: f32,
    pub has_model: f32,
    pub index_warm: f32,
}

impl QueryFeatures {
    pub const DIM: usize = 8;

    pub fn extract(query: &str, meta: &RepoMeta) -> Self {
        Self {
            word_count: query.split_whitespace().count() as f32,
            char_len: query.len() as f32,
            has_camel_case: if query::has_camel_case(query.trim()) {
                1.0
            } else {
                0.0
            },
            has_regex: if query::has_regex_metacharacters(query) {
                1.0
            } else {
                0.0
            },
            has_nl_cue: if query::has_natural_language_cue(query) {
                1.0
            } else {
                0.0
            },
            is_literal: if query::is_literal_query(query) {
                1.0
            } else {
                0.0
            },
            has_model: if meta.has_model { 1.0 } else { 0.0 },
            index_warm: if meta.index_warm { 1.0 } else { 0.0 },
        }
    }

    pub fn to_vector(&self) -> [f32; Self::DIM] {
        [
            self.word_count,
            self.char_len,
            self.has_camel_case,
            self.has_regex,
            self.has_nl_cue,
            self.is_literal,
            self.has_model,
            self.index_warm,
        ]
    }
}

/// Map feature vector to route via fixed weights (phase B).
pub fn feature_route(features: &QueryFeatures) -> Route {
    if features.has_regex > 0.5 || features.is_literal > 0.5 {
        return Route::Grep;
    }
    if features.has_nl_cue > 0.5 {
        return if features.index_warm > 0.5 {
            Route::Hybrid
        } else if features.has_model > 0.5 {
            Route::Semantic
        } else {
            Route::Prefocus
        };
    }
    if features.word_count >= 2.0 && features.has_model > 0.5 {
        return Route::Hybrid;
    }
    Route::Grep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_features_route_grep() {
        let f = QueryFeatures::extract("handleSessionRefresh", &RepoMeta::default());
        assert_eq!(feature_route(&f), Route::Grep);
    }

    #[test]
    fn nl_features_route_semantic() {
        let f = QueryFeatures::extract(
            "where is retry logic",
            &RepoMeta {
                has_model: true,
                ..Default::default()
            },
        );
        assert!(matches!(feature_route(&f), Route::Semantic | Route::Hybrid));
    }
}
