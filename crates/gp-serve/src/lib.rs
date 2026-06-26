//! HTTP daemon for agent integrations (`grepplus serve`).

use axum::extract::{Request, State};
use axum::http::{header::AUTHORIZATION, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use gp_core::config::Config;
use gp_core::error::GpError;
use gp_core::traits::Embedder;
use gp_core::types::{Route, ScoredChunk};
use gp_embed::{is_installed, require_embedder, EnsureOptions};
use gp_index::Index;
use gp_router::{resolve_router, route_label};
use gp_search::{build_index, hybrid_search, IndexBuildOptions, SearchOptions};
use notify::{RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use tower_http::cors::{Any, CorsLayer};

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub yes_download: bool,
    pub ensure_index: bool,
    pub warm_index: bool,
    pub auth_token: Option<String>,
    pub cors: bool,
    pub reload_config: bool,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            yes_download: false,
            ensure_index: false,
            warm_index: false,
            auth_token: std::env::var("GREPPLUS_SERVE_TOKEN").ok(),
            cors: true,
            reload_config: true,
        }
    }
}

#[derive(Clone)]
struct AppState {
    cfg: Arc<RwLock<Config>>,
    embedder: Arc<Mutex<Option<Arc<dyn Embedder>>>>,
    yes_download: bool,
    ensure_index: bool,
    warm_index: bool,
    auth_token: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub path: PathBuf,
    #[serde(default)]
    pub route: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub route: String,
    pub hits: Vec<ScoredChunk>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub model_loaded: bool,
    pub auth_required: bool,
}

pub async fn run_server(
    cfg: Config,
    addr: SocketAddr,
    opts: ServeOptions,
) -> gp_core::error::Result<()> {
    let state = AppState {
        cfg: Arc::new(RwLock::new(cfg)),
        embedder: Arc::new(Mutex::new(None)),
        yes_download: opts.yes_download,
        ensure_index: opts.ensure_index,
        warm_index: opts.warm_index,
        auth_token: opts.auth_token.clone(),
    };

    if opts.reload_config {
        spawn_config_watcher(Arc::clone(&state.cfg));
    }

    let search = Router::new()
        .route("/search", post(search_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    let mut app = Router::new()
        .route("/health", get(health_handler))
        .merge(search)
        .with_state(state);

    if opts.cors {
        app = app.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(Any),
        );
    }

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| GpError::Other(e.to_string()))?;

    print_startup_banner(addr, &opts);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| GpError::Other(e.to_string()))?;
    Ok(())
}

fn print_startup_banner(addr: SocketAddr, opts: &ServeOptions) {
    eprintln!("grep+ serve v{VERSION} listening on http://{addr}");
    eprintln!("  GET  /health");
    eprintln!("  POST /search  (JSON: query, path, optional route)");
    if opts.auth_token.is_some() {
        eprintln!("  auth: Bearer token required on /search (GREPPLUS_SERVE_TOKEN / --token)");
    } else {
        eprintln!("  auth: disabled (set --token or GREPPLUS_SERVE_TOKEN to enable)");
    }
    if opts.cors {
        eprintln!("  cors: enabled");
    }
    if opts.ensure_index {
        eprintln!(
            "  index: ensure sketch shell per search{}",
            if opts.warm_index {
                " (warm index when missing)"
            } else {
                ""
            }
        );
    }
    if opts.reload_config {
        eprintln!(
            "  config: hot-reload from {}",
            Config::global_config_path().display()
        );
    }
    eprintln!("  model: lazy load (skipped for route=grep)");
    eprintln!("  Ctrl+C to shut down gracefully");
}

async fn shutdown_signal() {
    if tokio::signal::ctrl_c().await.is_ok() {
        eprintln!("\ngrep+ serve shutting down gracefully...");
    }
}

fn spawn_config_watcher(cfg: Arc<RwLock<Config>>) {
    let path = Config::global_config_path();
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = match notify::recommended_watcher(tx) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("grep+ serve: config watch disabled ({e})");
                return;
            }
        };
        if let Some(parent) = path.parent() {
            if let Err(e) = watcher.watch(parent, RecursiveMode::NonRecursive) {
                eprintln!("grep+ serve: config watch disabled ({e})");
                return;
            }
        }
        while rx.recv().is_ok() {
            match Config::load() {
                Ok(new_cfg) => {
                    if let Ok(mut guard) = cfg.write() {
                        *guard = new_cfg;
                        eprintln!("grep+ serve: reloaded {}", path.display());
                    }
                }
                Err(e) => eprintln!("grep+ serve: config reload failed: {e}"),
            }
        }
    });
}

async fn auth_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    if let Some(expected) = &state.auth_token {
        let authorized = req
            .headers()
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v == format!("Bearer {expected}"));
        if !authorized {
            return (
                StatusCode::UNAUTHORIZED,
                "missing or invalid Authorization: Bearer <token>",
            )
                .into_response();
        }
    }
    next.run(req).await
}

async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let model_loaded = state
        .embedder
        .lock()
        .map(|g| g.is_some())
        .unwrap_or(false);
    Json(HealthResponse {
        status: "ok",
        version: VERSION,
        model_loaded,
        auth_required: state.auth_token.is_some(),
    })
}

async fn search_handler(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let cfg_guard = state
        .cfg
        .read()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "config lock poisoned".into()))?;
    let active_model = cfg_guard.embedder.active.clone();
    let has_model = is_installed(&active_model)
        || state
            .embedder
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false);

    let route = if let Some(r) = req.route.as_deref() {
        parse_route(r)
    } else {
        let router = resolve_router(&cfg_guard)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let meta = gp_core::types::RepoMeta {
            has_model,
            index_warm: Index::exists(&req.path),
            ..Default::default()
        };
        router.route(&req.query, &meta).route
    };
    drop(cfg_guard);

    let embedder = resolve_embedder_for_route(&state, route)
        .map_err(|e| map_embedder_error(e))?;

    if state.ensure_index && route_needs_model(route) {
        ensure_repo_index(&state, &req.path, embedder.as_deref())
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    let cfg_guard = state
        .cfg
        .read()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "config lock poisoned".into()))?;
    let search_opts = SearchOptions::from_config(&cfg_guard, route);
    drop(cfg_guard);

    let hits = hybrid_search(
        &req.query,
        &[req.path.clone()],
        embedder.as_deref(),
        &search_opts,
    )
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(SearchResponse {
        route: route_label(route).to_string(),
        hits,
    }))
}

fn ensure_repo_index(
    state: &AppState,
    repo: &std::path::Path,
    embedder: Option<&dyn Embedder>,
) -> gp_core::error::Result<()> {
    let cfg = state
        .cfg
        .read()
        .map_err(|_| GpError::Other("config lock poisoned".into()))?;
    if state.warm_index {
        if !Index::exists(repo) {
            let opts = IndexBuildOptions {
                model_id: cfg.embedder.active.clone(),
                dim: cfg.embedder.dim,
                sketch_only: false,
            };
            build_index(repo, embedder, &opts)?;
        }
    } else {
        let _ = gp_index::ensure_sketch_shell(
            repo,
            &cfg.embedder.active,
            cfg.embedder.dim,
        );
    }
    Ok(())
}

fn resolve_embedder_for_route(
    state: &AppState,
    route: Route,
) -> gp_core::error::Result<Option<Arc<dyn Embedder>>> {
    if !route_needs_model(route) {
        return Ok(None);
    }

    {
        let guard = state
            .embedder
            .lock()
            .map_err(|_| GpError::Other("embedder lock poisoned".into()))?;
        if let Some(e) = guard.as_ref() {
            return Ok(Some(Arc::clone(e)));
        }
    }

    let mut cfg = state
        .cfg
        .write()
        .map_err(|_| GpError::Other("config lock poisoned".into()))?;
    let opts = EnsureOptions::for_required_semantic(state.yes_download);
    let embedder = require_embedder(&mut cfg, &opts)?;
    let embedder = Arc::from(embedder);

    let mut guard = state
        .embedder
        .lock()
        .map_err(|_| GpError::Other("embedder lock poisoned".into()))?;
    *guard = Some(Arc::clone(&embedder));
    Ok(Some(embedder))
}

fn map_embedder_error(err: GpError) -> (StatusCode, String) {
    match err {
        GpError::NoModel => (
            StatusCode::SERVICE_UNAVAILABLE,
            "embedding model required — run `grepplus models install` or start serve with --yes-download"
                .into(),
        ),
        other => (StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
    }
}

fn route_needs_model(route: Route) -> bool {
    matches!(route, Route::Semantic | Route::Hybrid | Route::Prefocus)
}

fn parse_route(s: &str) -> Route {
    match s {
        "semantic" => Route::Semantic,
        "hybrid" => Route::Hybrid,
        "prefocus" => Route::Prefocus,
        _ => Route::Grep,
    }
}
