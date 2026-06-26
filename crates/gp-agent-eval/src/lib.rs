//! Minimal agent-loop eval: grepplus serve + path-picking agent.

use gp_core::types::ScoredChunk;
use gp_eval::load_queries;
use serde::Serialize;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct AgentTaskResult {
    pub query_id: String,
    pub retriever: String,
    pub delivery: String,
    pub success: bool,
    pub latency_ms: f32,
    pub picked_path: Option<String>,
    pub oracle_files: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentEvalMetrics {
    pub retriever: String,
    pub delivery: String,
    pub task_success_rate: f32,
    pub mean_latency_ms: f32,
    pub n: usize,
    pub tasks: Vec<AgentTaskResult>,
}

#[derive(Debug, Serialize)]
pub struct AgentEvalBundle {
    pub results: Vec<AgentEvalMetrics>,
}

#[derive(Debug, serde::Deserialize)]
struct SearchResponse {
    route: String,
    hits: Vec<ScoredChunk>,
}

fn hit_file(hit: &ScoredChunk) -> String {
    hit.chunk.file.to_string_lossy().into_owned()
}

/// Run factorial agent eval against a live or spawned grepplus serve instance.
pub fn run_agent_eval(
    cfg: &gp_core::config::Config,
    corpus: &Path,
    suite: &Path,
    retrievers: &str,
    serve_addr: &str,
    ensure_index: bool,
    warm_index: bool,
    yes_download: bool,
) -> Result<AgentEvalBundle, gp_core::error::GpError> {
    let _ = cfg;
    let queries = load_queries(suite)?;
    let addr: std::net::SocketAddr = serve_addr
        .parse()
        .map_err(|e| gp_core::error::GpError::Config(format!("invalid serve addr: {e}")))?;

    let child = maybe_spawn_server(addr, ensure_index, warm_index, yes_download)?;
    wait_for_health(addr)?;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| gp_core::error::GpError::Config(e.to_string()))?;

    let deliveries = ["bullets", "json"];
    let mut results = Vec::new();

    for retriever in retrievers.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        for delivery in deliveries {
            let mut tasks = Vec::new();
            let mut successes = 0usize;
            let mut latency_sum = 0f32;

            for q in &queries {
                let start = Instant::now();
                let route = match retriever {
                    "grep" => "grep",
                    "hybrid" => "hybrid",
                    other => {
                        return Err(gp_core::error::GpError::Config(format!(
                            "unknown retriever: {other}"
                        )))
                    }
                };

                let resp = client
                    .post(format!("http://{addr}/search"))
                    .json(&serde_json::json!({
                        "query": q.query,
                        "path": corpus.to_string_lossy(),
                        "route": route,
                    }))
                    .send()
                    .map_err(|e| gp_core::error::GpError::Config(e.to_string()))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().unwrap_or_default();
                    return Err(gp_core::error::GpError::Config(format!(
                        "search failed ({status}): {body}"
                    )));
                }

                let body: SearchResponse = resp.json().map_err(|e| {
                    gp_core::error::GpError::Config(format!("decode search response: {e}"))
                })?;

                let picked = agent_pick_path(&body.hits, delivery, &q.query);
                let oracle_files: Vec<String> = q.oracles.iter().map(|o| o.file.clone()).collect();
                let success = picked
                    .as_ref()
                    .map(|p| oracle_files.iter().any(|o| path_matches(p, o)))
                    .unwrap_or(false);

                if success {
                    successes += 1;
                }
                let ms = start.elapsed().as_secs_f32() * 1000.0;
                latency_sum += ms;

                tasks.push(AgentTaskResult {
                    query_id: q.id.clone(),
                    retriever: retriever.to_string(),
                    delivery: delivery.to_string(),
                    success,
                    latency_ms: ms,
                    picked_path: picked,
                    oracle_files,
                });
            }

            let n = queries.len().max(1);
            results.push(AgentEvalMetrics {
                retriever: retriever.to_string(),
                delivery: delivery.to_string(),
                task_success_rate: successes as f32 / n as f32,
                mean_latency_ms: latency_sum / n as f32,
                n: queries.len(),
                tasks,
            });
        }
    }

    if let Some(mut proc) = child {
        let _ = proc.kill();
        let _ = proc.wait();
    }

    Ok(AgentEvalBundle { results })
}

fn maybe_spawn_server(
    addr: std::net::SocketAddr,
    ensure_index: bool,
    warm_index: bool,
    yes_download: bool,
) -> Result<Option<std::process::Child>, gp_core::error::GpError> {
    if health_ok(addr) {
        return Ok(None);
    }

    let bin = std::env::var("GREPPLUS_BIN").unwrap_or_else(|_| "grepplus".into());
    let mut cmd = std::process::Command::new(bin);
    cmd.args(["serve", "--bind", &addr.to_string(), "--no-reload-config"]);
    if ensure_index {
        cmd.arg("--ensure-index");
    }
    if warm_index {
        cmd.arg("--warm-index");
    }
    if yes_download {
        cmd.arg("--yes-download");
    }
    let child = cmd
        .spawn()
        .map_err(|e| gp_core::error::GpError::Config(format!("spawn serve: {e}")))?;

    Ok(Some(child))
}

fn health_ok(addr: std::net::SocketAddr) -> bool {
    reqwest::blocking::Client::new()
        .get(format!("http://{addr}/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn wait_for_health(addr: std::net::SocketAddr) -> Result<(), gp_core::error::GpError> {
    for _ in 0..60 {
        if health_ok(addr) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(gp_core::error::GpError::Config(
        "serve health check timed out".into(),
    ))
}

fn agent_pick_path(hits: &[ScoredChunk], delivery: &str, query: &str) -> Option<String> {
    if hits.is_empty() {
        return None;
    }

    if std::env::var("GREPPLUS_AGENT_MOCK").ok().as_deref() == Some("1") {
        return Some(hit_file(&hits[0]));
    }

    if let Some(path) = llm_pick_path(hits, delivery, query) {
        return Some(path);
    }

    Some(hit_file(&hits[0]))
}

fn llm_pick_path(hits: &[ScoredChunk], delivery: &str, query: &str) -> Option<String> {
    let prompt = format_agent_prompt(hits, delivery, query);

    if let Ok(url) = std::env::var("OLLAMA_HOST") {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2:3b".into());
        if let Ok(answer) = ollama_complete(&url, &model, &prompt) {
            return extract_path_from_answer(&answer, hits);
        }
    }

    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        if let Ok(answer) = openai_complete(&key, &model, &prompt) {
            return extract_path_from_answer(&answer, hits);
        }
    }

    None
}

fn format_agent_prompt(hits: &[ScoredChunk], delivery: &str, query: &str) -> String {
    let mut s = format!(
        "You are a coding agent. Given search results, reply with ONLY the relative file path that best answers the query.\nQuery: {query}\n\nResults:\n"
    );
    if delivery == "json" {
        s.push_str(&serde_json::to_string(hits).unwrap_or_default());
    } else {
        for (i, h) in hits.iter().take(5).enumerate() {
            s.push_str(&format!(
                "{}. {}:{}-{} (score={:.3})\n{}\n",
                i + 1,
                hit_file(h),
                h.chunk.start_line,
                h.chunk.end_line,
                h.score,
                h.preview.as_deref().unwrap_or("")
            ));
        }
    }
    s.push_str("\nReply with the file path only.");
    s
}

fn extract_path_from_answer(answer: &str, hits: &[ScoredChunk]) -> Option<String> {
    let trimmed = answer.trim();
    for h in hits {
        let file = hit_file(h);
        if trimmed.contains(&file) {
            return Some(file);
        }
    }
    let line = trimmed.lines().next()?.trim();
    if line.contains('/') || line.ends_with(".rs") || line.ends_with(".ts") {
        return Some(line.to_string());
    }
    None
}

fn ollama_complete(host: &str, model: &str, prompt: &str) -> Result<String, String> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{host}/api/generate"))
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
        }))
        .send()
        .map_err(|e| e.to_string())?;
    let v: serde_json::Value = resp.json().map_err(|e| e.to_string())?;
    Ok(v["response"].as_str().unwrap_or("").to_string())
}

fn openai_complete(key: &str, model: &str, prompt: &str) -> Result<String, String> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(key)
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }))
        .send()
        .map_err(|e| e.to_string())?;
    let v: serde_json::Value = resp.json().map_err(|e| e.to_string())?;
    Ok(v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string())
}

fn path_matches(picked: &str, oracle: &str) -> bool {
    picked.ends_with(oracle) || oracle.ends_with(picked) || picked.contains(oracle)
}

pub fn format_agent_report(bundle: &AgentEvalBundle) -> String {
    let mut lines = vec![
        "| retriever | delivery | task_success | mean_ms | n |".to_string(),
        "|-----------|----------|--------------|---------|---|".to_string(),
    ];
    for r in &bundle.results {
        lines.push(format!(
            "| {} | {} | {:.3} | {:.1} | {} |",
            r.retriever, r.delivery, r.task_success_rate, r.mean_latency_ms, r.n
        ));
    }
    lines.join("\n")
}
