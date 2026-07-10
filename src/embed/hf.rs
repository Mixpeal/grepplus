use crate::core::error::{GpError, Result};
use serde::Deserialize;
use std::collections::HashSet;

const HF_API: &str = "https://huggingface.co/api";

#[derive(Debug, Clone)]
pub struct HfFile {
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfModelInfo {
    #[allow(dead_code)]
    pub id: Option<String>,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingCheck {
    Ok,
    Warn(String),
    Reject(String),
}

pub struct HfClient {
    http: reqwest::blocking::Client,
}

impl HfClient {
    pub fn new() -> Result<Self> {
        let mut builder = reqwest::blocking::Client::builder().user_agent("grepplus/0.1");
        if let Ok(token) = std::env::var("GREPPLUS_HF_TOKEN").or_else(|_| std::env::var("HF_TOKEN"))
        {
            if !token.is_empty() {
                builder = builder.default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::AUTHORIZATION,
                        format!("Bearer {token}")
                            .parse::<reqwest::header::HeaderValue>()
                            .map_err(|e| GpError::Model(e.to_string()))?,
                    );
                    headers
                });
            }
        }
        Ok(Self {
            http: builder.build().map_err(|e| GpError::Model(e.to_string()))?,
        })
    }

    pub fn model_info(&self, repo: &str) -> Result<HfModelInfo> {
        let url = format!("{HF_API}/models/{repo}");
        let resp = self
            .http
            .get(&url)
            .send()
            .map_err(|e| GpError::Model(format!("HF API {repo}: {e}")))?;
        if !resp.status().is_success() {
            return Err(GpError::Model(format!(
                "HF API {repo}: HTTP {}",
                resp.status()
            )));
        }
        resp.json().map_err(|e| GpError::Model(e.to_string()))
    }

    pub fn list_tree(&self, repo: &str, revision: &str) -> Result<Vec<HfFile>> {
        let url = format!("{HF_API}/models/{repo}/tree/{revision}?recursive=1");
        let resp = self
            .http
            .get(&url)
            .send()
            .map_err(|e| GpError::Model(format!("HF tree {repo}: {e}")))?;
        if !resp.status().is_success() {
            return Err(GpError::Model(format!(
                "HF tree {repo}@{revision}: HTTP {}",
                resp.status()
            )));
        }
        let nodes: Vec<TreeNode> = resp.json().map_err(|e| GpError::Model(e.to_string()))?;
        Ok(nodes
            .into_iter()
            .filter(|n| n.r#type == "file")
            .map(|n| HfFile {
                path: n.path,
                size: n.size.unwrap_or(0),
            })
            .collect())
    }

    pub fn search_repos(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let url = format!(
            "{HF_API}/models?search={}&limit={limit}",
            urlencoding(query)
        );
        let resp = self
            .http
            .get(&url)
            .send()
            .map_err(|e| GpError::Model(format!("HF search: {e}")))?;
        if !resp.status().is_success() {
            return Ok(vec![]);
        }
        let models: Vec<SearchHit> = resp.json().map_err(|e| GpError::Model(e.to_string()))?;
        Ok(models.into_iter().filter_map(|m| m.id).collect())
    }

    pub fn fetch_json<T: for<'de> Deserialize<'de>>(
        &self,
        repo: &str,
        revision: &str,
        file: &str,
    ) -> Result<T> {
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");
        let resp = self
            .http
            .get(&url)
            .send()
            .map_err(|e| GpError::Model(format!("fetch {file}: {e}")))?;
        if !resp.status().is_success() {
            return Err(GpError::Model(format!(
                "fetch {file} from {repo}: HTTP {}",
                resp.status()
            )));
        }
        resp.json().map_err(|e| GpError::Model(e.to_string()))
    }
}

#[derive(Debug, Deserialize)]
struct TreeNode {
    path: String,
    r#type: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct SearchHit {
    id: Option<String>,
}

fn urlencoding(s: &str) -> String {
    s.replace(' ', "+")
}

const EMBED_PIPELINE: &[&str] = &["feature-extraction", "sentence-similarity"];
const EMBED_TAGS: &[&str] = &[
    "sentence-transformers",
    "text-embedding",
    "text-embeddings-inference",
    "feature-extraction",
    "sentence-similarity",
];
const REJECT_PIPELINE: &[&str] = &[
    "text-generation",
    "image-classification",
    "automatic-speech-recognition",
    "text-to-image",
    "object-detection",
];

pub fn classify_embedding_model(info: &HfModelInfo, force: bool) -> Result<EmbeddingCheck> {
    if force {
        return Ok(EmbeddingCheck::Warn(
            "skipping embedding model check (--force)".into(),
        ));
    }

    let tags: HashSet<String> = info.tags.iter().map(|t| t.to_lowercase()).collect();
    let has_embed_tag = EMBED_TAGS.iter().any(|t| tags.contains(&t.to_lowercase()));
    let pipeline = info
        .pipeline_tag
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();

    if tags.iter().any(|t| t.contains("cross-encoder")) {
        return Ok(EmbeddingCheck::Warn(
            "cross-encoder reranker — not a bi-encoder index model".into(),
        ));
    }

    if EMBED_PIPELINE.iter().any(|p| pipeline == *p) || has_embed_tag {
        return Ok(EmbeddingCheck::Ok);
    }

    if REJECT_PIPELINE.iter().any(|p| pipeline == *p) && !has_embed_tag {
        return Ok(EmbeddingCheck::Reject(format!(
            "pipeline_tag={pipeline} does not look like an embedding model"
        )));
    }

    if pipeline.is_empty() && !has_embed_tag {
        return Ok(EmbeddingCheck::Warn(
            "no feature-extraction tag — proceeding because repo has runnable weights".into(),
        ));
    }

    Ok(EmbeddingCheck::Ok)
}

pub fn repo_basename(repo: &str) -> String {
    repo.rsplit('/').next().unwrap_or(repo).to_string()
}

pub fn local_id_from_repo(repo: &str, quant: Option<&str>) -> String {
    let base = repo_basename(repo).to_lowercase().replace(['/', ' '], "-");
    match quant {
        Some(q) if !q.is_empty() => install_local_id(&base, q, None),
        _ => base,
    }
}

/// Local install id for a base model + ONNX quant variant.
/// When `quant` matches the catalog default, returns `base_id` unchanged (backward compatible).
pub fn install_local_id(base_id: &str, quant: &str, default_quant: Option<&str>) -> String {
    let q = quant.to_lowercase().replace('.', "_");
    if default_quant.is_some_and(|d| d.eq_ignore_ascii_case(quant) || d == q) {
        return base_id.to_string();
    }
    if base_id.ends_with(&format!("-{q}")) {
        return base_id.to_string();
    }
    format!("{base_id}-{q}")
}

pub fn infer_base_id(id: &str, quant: Option<&str>) -> String {
    if let Some(q) = quant {
        let suffix = format!("-{}", q.to_lowercase().replace('.', "_"));
        if id.ends_with(&suffix) && id.len() > suffix.len() {
            return id[..id.len() - suffix.len()].to_string();
        }
    }
    id.to_string()
}

pub fn find_mirror_repos(client: &HfClient, repo: &str) -> Result<Vec<String>> {
    let basename = repo_basename(repo);
    let mut candidates = Vec::new();
    for query in [format!("{basename} ONNX"), format!("{basename} onnx")] {
        for id in client.search_repos(&query, 12)? {
            if id == repo {
                continue;
            }
            let lower = id.to_lowercase();
            if !lower.contains(&basename.to_lowercase()) {
                continue;
            }
            if lower.contains("onnx") && !lower.contains("gguf") && !candidates.contains(&id) {
                candidates.push(id);
            }
        }
    }
    candidates.sort_by_key(|id| mirror_rank(id));
    Ok(candidates)
}

fn mirror_rank(id: &str) -> i32 {
    let lower = id.to_lowercase();
    if lower.starts_with("onnx-community/") {
        return 0;
    }
    if lower.contains("-onnx") || lower.contains("/onnx") {
        return 1;
    }
    if lower.contains("-gguf") || lower.contains("/gguf") {
        return 2;
    }
    10
}

#[derive(Debug, Deserialize)]
pub struct HfConfig {
    pub hidden_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub architectures: Vec<String>,
}

pub fn infer_pooling(config: &HfConfig) -> &'static str {
    let arch = config
        .architectures
        .first()
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    if arch.contains("causal") || arch.contains("qwen") || arch.contains("llama") {
        "last"
    } else {
        "mean"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_local_id_default_quant_uses_base() {
        assert_eq!(
            install_local_id("harrier-oss-v1-0.6b", "model_q4f16", Some("model_q4f16")),
            "harrier-oss-v1-0.6b"
        );
    }

    #[test]
    fn install_local_id_other_quant_suffixes() {
        assert_eq!(
            install_local_id("harrier-oss-v1-0.6b", "model_fp16", Some("model_q4f16")),
            "harrier-oss-v1-0.6b-model_fp16"
        );
    }

    #[test]
    fn local_id_slug() {
        assert_eq!(
            local_id_from_repo("microsoft/harrier-oss-v1-0.6b", Some("Q4_K_M")),
            "harrier-oss-v1-0.6b-q4_k_m"
        );
    }

    #[test]
    fn classify_rejects_llm() {
        let info = HfModelInfo {
            id: Some("meta-llama/Llama-3.2-1B".into()),
            pipeline_tag: Some("text-generation".into()),
            tags: vec!["text-generation".into()],
        };
        match classify_embedding_model(&info, false).unwrap() {
            EmbeddingCheck::Reject(_) => {}
            other => panic!("expected reject, got {other:?}"),
        }
    }
}
