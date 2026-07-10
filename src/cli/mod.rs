use crate::core::config::Config;
use crate::core::error::GpError;
use crate::core::traits::{EvalHarness, EvalMode, GrepOptions};
use crate::core::types::{RepoMeta, Route, ScoredChunk};
use crate::embed::{
    default_pull_opts, interactive_pick, is_installed, print_models_list, pull_model, remove_model,
    require_embedder, set_active_model, EnsureOptions,
};
use crate::eval::{
    compare_modes, format_report, results_to_json, AgentCodeHarness, HarnessOverrides,
};
use crate::grep::resolve_cli_backend;
use crate::index::{purge_expired, Index};
use crate::router::{append_trace, resolve_router, route_label, RouteTrace};
use crate::search::{build_index, hybrid_search, IndexBuildOptions, SearchOptions};
use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod welcome;

#[derive(Clone, Copy, ValueEnum, Debug, Default)]
enum EvalFormat {
    #[default]
    Table,
    Json,
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum RouteArg {
    Grep,
    Semantic,
    Hybrid,
    Prefocus,
}

impl From<RouteArg> for Route {
    fn from(r: RouteArg) -> Self {
        match r {
            RouteArg::Grep => Route::Grep,
            RouteArg::Semantic => Route::Semantic,
            RouteArg::Hybrid => Route::Hybrid,
            RouteArg::Prefocus => Route::Prefocus,
        }
    }
}

#[derive(Parser)]
#[command(name = "grepplus", version, about = "grep+ hybrid search CLI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    pattern: Option<String>,
    #[arg(default_value = ".")]
    paths: Vec<PathBuf>,

    #[arg(short = 'i', long)]
    ignore_case: bool,
    #[arg(short = 'F', long = "fixed-strings")]
    fixed_string: bool,
    #[arg(short = 'n', long, default_value_t = true)]
    line_numbers: bool,

    #[arg(long)]
    semantic: bool,
    #[arg(long)]
    hybrid: bool,
    #[arg(long)]
    prefocus: bool,
    #[arg(long, value_enum)]
    route: Option<RouteArg>,
    #[arg(long)]
    route_debug: bool,
    #[arg(long)]
    yes_download: bool,
    #[arg(long)]
    ensure_model: bool,
    #[arg(long)]
    ensure_index: bool,
    #[arg(long)]
    warm_index: bool,
    #[arg(long)]
    local_traces: bool,
    #[arg(short = 'q', long)]
    quiet: bool,
    #[arg(short = 'g', long = "glob", value_name = "GLOB")]
    globs: Vec<String>,
    #[arg(short = 'l', long = "files-with-matches")]
    files_with_matches: bool,
    #[arg(short = 'c', long = "count")]
    count: bool,
    #[arg(long = "max-count", value_name = "NUM")]
    max_count: Option<usize>,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage embedding models (list, pull, install, use)
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Build and manage the search index (sketch, warm embed, watch)
    Index {
        paths: Vec<PathBuf>,
        #[arg(long)]
        status: bool,
        #[arg(long)]
        sketch_only: bool,
        #[arg(long)]
        yes_download: bool,
        #[arg(long)]
        ensure_model: bool,
        #[arg(long)]
        purge: bool,
        #[arg(long)]
        watch: bool,
    },
    /// Start HTTP daemon for agent integrations (/search, /health)
    Serve {
        #[arg(long, default_value = "127.0.0.1:9470")]
        bind: String,
        #[arg(long)]
        yes_download: bool,
        #[arg(long)]
        ensure_index: bool,
        #[arg(long)]
        warm_index: bool,
        /// Require `Authorization: Bearer <token>` on /search (also GREPPLUS_SERVE_TOKEN)
        #[arg(long)]
        token: Option<String>,
        /// Disable cross-origin headers (CORS enabled by default)
        #[arg(long)]
        no_cors: bool,
        /// Disable hot-reload of ~/.grepplus/config.toml
        #[arg(long)]
        no_reload_config: bool,
        /// Allow non-loopback bind without a bearer token
        #[arg(long)]
        allow_unauthenticated: bool,
    },
    /// Research utilities (router traces, etc.)
    #[command(hide = true)]
    Research {
        #[command(subcommand)]
        cmd: ResearchCmd,
    },
    /// Train the learned query router from local search traces
    #[command(hide = true)]
    Router {
        #[command(subcommand)]
        cmd: RouterCmd,
    },
    /// Run retrieval benchmarks (grep, ripgrep, laser, vector, hybrid, jit)
    #[command(hide = true)]
    Eval {
        #[command(subcommand)]
        cmd: EvalCmd,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    List,
    Pull {
        target: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long)]
        quant: Option<String>,
        #[arg(long)]
        as_id: Option<String>,
        /// Skip quant picker; install the recommended ONNX variant
        #[arg(long)]
        yes_download: bool,
        /// Include unquantized full-precision ONNX in the quant picker
        #[arg(long)]
        include_full: bool,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        pin: bool,
        #[arg(long)]
        set_active: bool,
    },
    Use {
        id: Option<String>,
    },
    Install,
    Remove {
        id: String,
    },
}

#[derive(Subcommand)]
enum ResearchCmd {
    Router {
        #[command(subcommand)]
        cmd: ResearchRouterCmd,
    },
}

#[derive(Subcommand)]
enum ResearchRouterCmd {
    Traces {
        corpus: PathBuf,
        #[arg(long, default_value = "eval/agentcode/queries.jsonl")]
        suite: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        yes_download: bool,
    },
}

#[derive(Subcommand)]
enum RouterCmd {
    Train {
        #[arg(long, default_value = "traces/routes.jsonl")]
        traces: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum EvalCmd {
    Run {
        corpus: PathBuf,
        #[arg(long, default_value = "eval/agentcode/queries.jsonl")]
        suite: PathBuf,
        #[arg(long, default_value = "laser")]
        mode: String,
        #[arg(long)]
        ensure_index: bool,
        #[arg(long)]
        warm_index: bool,
        #[arg(long)]
        isolate_modes: bool,
        #[arg(long)]
        yes_download: bool,
        #[arg(long)]
        filter_category: Option<String>,
        #[arg(long)]
        filter_laser_miss: bool,
        #[arg(long)]
        jit_embed_budget: Option<usize>,
        #[arg(long)]
        jit_reheat_file_cap: Option<usize>,
    },
    Compare {
        corpus: PathBuf,
        #[arg(long, default_value = "eval/agentcode/queries.jsonl")]
        suite: PathBuf,
        #[arg(long, default_value = "grep,ripgrep,laser,vector,hybrid,jit")]
        modes: String,
        #[arg(long)]
        ensure_index: bool,
        #[arg(long)]
        warm_index: bool,
        #[arg(long)]
        isolate_modes: bool,
        #[arg(long)]
        yes_download: bool,
        #[arg(long, value_enum, default_value_t = EvalFormat::Table)]
        format: EvalFormat,
        #[arg(long)]
        filter_category: Option<String>,
        #[arg(long)]
        filter_laser_miss: bool,
        #[arg(long)]
        jit_embed_budget: Option<usize>,
        #[arg(long)]
        jit_reheat_file_cap: Option<usize>,
    },
    Report {
        corpus: PathBuf,
        #[arg(long, default_value = "eval/agentcode/queries.jsonl")]
        suite: PathBuf,
        #[arg(long, default_value = "grep,ripgrep,laser,vector,hybrid,jit")]
        modes: String,
        #[arg(long)]
        ensure_index: bool,
        #[arg(long)]
        warm_index: bool,
        #[arg(long)]
        isolate_modes: bool,
        #[arg(long)]
        yes_download: bool,
        #[arg(long, value_enum, default_value_t = EvalFormat::Table)]
        format: EvalFormat,
        #[arg(long)]
        filter_category: Option<String>,
        #[arg(long)]
        filter_laser_miss: bool,
        #[arg(long)]
        jit_embed_budget: Option<usize>,
        #[arg(long)]
        jit_reheat_file_cap: Option<usize>,
    },
    Agent {
        corpus: PathBuf,
        #[arg(long, default_value = "eval/agentcode/queries.jsonl")]
        suite: PathBuf,
        #[arg(long, default_value = "grep,hybrid")]
        retrievers: String,
        #[arg(long, default_value = "127.0.0.1:9470")]
        serve_addr: String,
        #[arg(long)]
        ensure_index: bool,
        #[arg(long)]
        warm_index: bool,
        #[arg(long)]
        yes_download: bool,
        #[arg(long, value_enum, default_value_t = EvalFormat::Table)]
        format: EvalFormat,
    },
}

pub fn run() {
    if print_root_help_if_requested() {
        return;
    }
    if let Err(err) = main_inner() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
    // ONNX Runtime teardown can segfault on macOS when processes exit in quick
    // succession (ort 2.0 RC); skip destructors after successful CLI runs.
    // Revisit when ort ships a stable release without this teardown bug.
    std::process::exit(0);
}

/// Root `--help` mixes search args + subcommands; clap omits subcommand descriptions unless we render explicitly.
fn print_root_help_if_requested() -> bool {
    const SUBCOMMANDS: &[&str] = &[
        "models", "index", "research", "router", "serve", "eval", "help",
    ];

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.iter().any(|a| a == "--help" || a == "-h") {
        return false;
    }
    if args.iter().any(|a| SUBCOMMANDS.contains(&a.as_str())) {
        return false;
    }

    print!("{}", Cli::command().render_help());
    true
}

fn main_inner() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,ort=warn")),
        )
        .init();
    let cli = Cli::parse();
    let mut cfg = Config::load().context("load config")?;
    let _ = purge_expired(cfg.index.cache_ttl_days);

    if let Some(cmd) = cli.command {
        return dispatch_command(cmd, &mut cfg);
    }

    if cli.pattern.is_none() {
        welcome::print_welcome(&cfg);
        return Ok(());
    }

    let pattern = cli.pattern.expect("pattern checked above");

    run_search(
        &pattern,
        &cli.paths,
        &mut cfg,
        SearchFlags {
            ignore_case: cli.ignore_case,
            fixed_string: cli.fixed_string,
            line_numbers: cli.line_numbers,
            semantic: cli.semantic,
            hybrid: cli.hybrid,
            prefocus: cli.prefocus,
            route: cli.route.map(Route::from),
            route_debug: cli.route_debug,
            yes_download: cli.yes_download,
            ensure_model: cli.ensure_model,
            ensure_index: cli.ensure_index,
            warm_index: cli.warm_index,
            local_traces: cli.local_traces,
            quiet: cli.quiet,
            globs: cli.globs,
            files_with_matches: cli.files_with_matches,
            count: cli.count,
            max_count: cli.max_count,
        },
    )
}

struct SearchFlags {
    ignore_case: bool,
    fixed_string: bool,
    line_numbers: bool,
    semantic: bool,
    hybrid: bool,
    prefocus: bool,
    route: Option<Route>,
    route_debug: bool,
    yes_download: bool,
    ensure_model: bool,
    ensure_index: bool,
    warm_index: bool,
    local_traces: bool,
    quiet: bool,
    globs: Vec<String>,
    files_with_matches: bool,
    count: bool,
    max_count: Option<usize>,
}

fn needs_model(route: Route) -> bool {
    matches!(route, Route::Semantic | Route::Hybrid | Route::Prefocus)
}

fn dispatch_command(cmd: Commands, cfg: &mut Config) -> Result<()> {
    match cmd {
        Commands::Models { cmd } => match cmd {
            ModelsCmd::List => {
                print_models_list(&cfg.embedder.active).context("list models")?;
            }
            ModelsCmd::Pull {
                target,
                revision,
                quant,
                as_id,
                yes_download,
                include_full,
                force,
                pin,
                set_active,
            } => {
                let mut opts = default_pull_opts(&target);
                opts.revision = revision;
                opts.quant = quant;
                opts.as_id = as_id;
                opts.non_interactive = yes_download;
                opts.include_full = include_full;
                opts.force = force;
                opts.pin_catalog = pin;
                let path = pull_model(&opts).context("pull model")?;
                let id = path.file_name().and_then(|s| s.to_str()).unwrap_or("model");
                if set_active {
                    set_active_model(id)?;
                    cfg.embedder.active = id.to_string();
                    println!("installed and activated {id} at {}", path.display());
                } else {
                    println!("model installed at {}", path.display());
                }
            }
            ModelsCmd::Use { id } => {
                let picked = match id {
                    Some(id) => {
                        set_active_model(&id).context("set active model")?;
                        id
                    }
                    None => interactive_pick(cfg).context("pick model")?,
                };
                cfg.embedder.active = picked.clone();
                println!("active model set to {picked}");
            }
            ModelsCmd::Install => {
                let id = interactive_pick(cfg).context("model setup")?;
                cfg.embedder.active = id.clone();
                println!("active model set to {id}");
            }
            ModelsCmd::Remove { id } => {
                remove_model(&id).context("remove model")?;
                if cfg.embedder.active == id {
                    eprintln!("warning: removed active model {id}");
                }
                println!("removed model {id}");
            }
        },
        Commands::Index {
            paths,
            status,
            sketch_only,
            yes_download,
            ensure_model: ensure_model_flag,
            purge,
            watch,
        } => {
            let repo = paths.first().cloned().unwrap_or_else(|| PathBuf::from("."));

            if watch {
                return crate::index::watch_repo(&repo).context("index watch");
            }

            if purge {
                Index::purge(&repo)?;
                println!("purged index cache for {}", repo.display());
                if paths.len() == 1 && !status && !sketch_only {
                    return Ok(());
                }
            }

            Index::warn_legacy_index(&repo);

            if status {
                if Index::exists(&repo) {
                    let idx = Index::open(&repo)?;
                    let stats = idx.temperature_stats().unwrap_or_default();
                    println!(
                        "index: {} chunks, {} files, model={}, sketch_only={}",
                        idx.manifest.chunk_count,
                        idx.manifest.file_count,
                        idx.manifest.model_id,
                        idx.manifest.sketch_only
                    );
                    println!(
                        "temperature: hot={} cold={} cool={} (of {} files)",
                        stats.hot, stats.cold, stats.cool, stats.total_files
                    );
                } else {
                    println!("no index at {}", Index::index_path(&repo).display());
                }
                return Ok(());
            }

            let embedder = if sketch_only {
                None
            } else {
                let opts = EnsureOptions::for_required_semantic(yes_download || ensure_model_flag);
                Some(require_embedder(cfg, &opts).context("resolve embedder")?)
            };

            let opts = IndexBuildOptions::from_config(cfg, sketch_only);
            let idx = build_index(&repo, embedder.as_deref(), &opts).context("build index")?;
            println!(
                "indexed {} chunks from {} files → {}",
                idx.manifest.chunk_count,
                idx.manifest.file_count,
                idx.root.display()
            );
        }
        Commands::Research { cmd } => match cmd {
            ResearchCmd::Router { cmd } => match cmd {
                ResearchRouterCmd::Traces {
                    corpus,
                    suite,
                    output,
                    yes_download,
                } => {
                    let harness = build_eval_harness(
                        cfg,
                        corpus,
                        suite,
                        true,
                        true,
                        false,
                        yes_download,
                        None,
                        false,
                        None,
                        None,
                    );
                    let path = crate::research::generate_router_traces(&harness, output.as_deref())
                        .context("router traces")?;
                    println!("wrote router traces to {}", path.display());
                }
            },
        },
        Commands::Router { cmd } => match cmd {
            RouterCmd::Train { traces, output } => {
                let path = if traces.is_absolute() {
                    traces
                } else {
                    Config::global_config_dir().join(traces)
                };
                let items = crate::router::load_traces(&path).context("load traces")?;
                let model = crate::router::train_router(&items).context("train router")?;
                let out = output.unwrap_or_else(|| crate::router::router_model_path(cfg));
                crate::router::save_model(&out, &model).context("save router model")?;
                println!("router model saved to {}", out.display());
            }
        },
        Commands::Serve {
            bind,
            yes_download,
            ensure_index,
            warm_index,
            token,
            no_cors,
            no_reload_config,
            allow_unauthenticated,
        } => {
            let addr: std::net::SocketAddr = bind.parse().context("invalid bind address")?;
            let auth_token = token.or_else(|| std::env::var("GREPPLUS_SERVE_TOKEN").ok());
            let opts = crate::serve::ServeOptions {
                yes_download,
                ensure_index,
                warm_index,
                auth_token,
                cors: !no_cors,
                reload_config: !no_reload_config,
                allow_unauthenticated,
            };
            let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
            rt.block_on(crate::serve::run_server(cfg.clone(), addr, opts))
                .context("serve")?;
        }
        Commands::Eval { cmd } => match cmd {
            EvalCmd::Run {
                corpus,
                suite,
                mode,
                ensure_index,
                warm_index,
                isolate_modes: _,
                yes_download,
                filter_category,
                filter_laser_miss,
                jit_embed_budget,
                jit_reheat_file_cap,
            } => {
                let harness = build_eval_harness(
                    cfg,
                    corpus,
                    suite,
                    ensure_index,
                    warm_index,
                    false,
                    yes_download,
                    filter_category,
                    filter_laser_miss,
                    jit_embed_budget,
                    jit_reheat_file_cap,
                );
                let eval_mode = parse_mode(&mode)?;
                let metrics = harness.run(eval_mode, "").context("eval run")?;
                print_metrics(&mode, &metrics);
            }
            EvalCmd::Compare {
                corpus,
                suite,
                modes,
                ensure_index,
                warm_index,
                isolate_modes,
                yes_download,
                format,
                filter_category,
                filter_laser_miss,
                jit_embed_budget,
                jit_reheat_file_cap,
            } => {
                let harness = build_eval_harness(
                    cfg,
                    corpus,
                    suite,
                    ensure_index,
                    warm_index,
                    isolate_modes,
                    yes_download,
                    filter_category,
                    filter_laser_miss,
                    jit_embed_budget,
                    jit_reheat_file_cap,
                );
                let parsed: Result<Vec<EvalMode>> =
                    modes.split(',').map(|m| parse_mode(m.trim())).collect();
                let results = compare_modes(&harness, &parsed?).context("eval compare")?;
                match format {
                    EvalFormat::Json => println!("{}", results_to_json(&results)?),
                    EvalFormat::Table => {
                        for (mode, metrics) in results {
                            print_metrics(&mode, &metrics);
                            println!();
                        }
                    }
                }
            }
            EvalCmd::Report {
                corpus,
                suite,
                modes,
                ensure_index,
                warm_index,
                isolate_modes,
                yes_download,
                format,
                filter_category,
                filter_laser_miss,
                jit_embed_budget,
                jit_reheat_file_cap,
            } => {
                let harness = build_eval_harness(
                    cfg,
                    corpus,
                    suite,
                    ensure_index,
                    warm_index,
                    isolate_modes,
                    yes_download,
                    filter_category,
                    filter_laser_miss,
                    jit_embed_budget,
                    jit_reheat_file_cap,
                );
                let parsed: Result<Vec<EvalMode>> =
                    modes.split(',').map(|m| parse_mode(m.trim())).collect();
                let results = compare_modes(&harness, &parsed?).context("eval report")?;
                match format {
                    EvalFormat::Json => println!("{}", results_to_json(&results)?),
                    EvalFormat::Table => {
                        println!("{}", format_report(&results));
                        println!();
                        for (mode, metrics) in &results {
                            print_metrics(mode, metrics);
                            println!();
                        }
                    }
                }
            }
            EvalCmd::Agent {
                corpus,
                suite,
                retrievers,
                serve_addr,
                ensure_index,
                warm_index,
                yes_download,
                format,
            } => {
                let results = crate::agent_eval::run_agent_eval(
                    cfg,
                    &corpus,
                    &suite,
                    &retrievers,
                    &serve_addr,
                    ensure_index,
                    warm_index,
                    yes_download,
                )
                .context("eval agent")?;
                match format {
                    EvalFormat::Json => println!("{}", serde_json::to_string_pretty(&results)?),
                    EvalFormat::Table => {
                        println!("{}", crate::agent_eval::format_agent_report(&results))
                    }
                }
            }
        },
    }
    Ok(())
}

fn run_search(
    pattern: &str,
    paths: &[PathBuf],
    cfg: &mut Config,
    flags: SearchFlags,
) -> Result<()> {
    let repo = paths.first().cloned().unwrap_or_else(|| PathBuf::from("."));
    let meta = RepoMeta {
        index_warm: Index::exists(&repo),
        has_model: is_installed(&cfg.embedder.active),
        ..Default::default()
    };

    let router = resolve_router(cfg).context("resolve router")?;
    let mut decision = router.route(pattern, &meta);

    if flags.semantic {
        decision.route = Route::Semantic;
    }
    if flags.hybrid {
        decision.route = Route::Hybrid;
    }
    if flags.prefocus {
        decision.route = Route::Prefocus;
    }
    if let Some(r) = flags.route {
        decision.route = r;
    }

    if flags.route_debug {
        println!(
            "route={:?} confidence={:.2}",
            decision.route, decision.confidence
        );
        for r in &decision.rationale {
            println!("  - {r}");
        }
    }

    if decision.route == Route::Grep {
        return run_grep(pattern, paths, &flags, cfg);
    }

    let force_semantic = flags.semantic || flags.hybrid || flags.prefocus || flags.route.is_some();

    let embedder = if needs_model(decision.route) {
        let opts = EnsureOptions::for_required_semantic(flags.yes_download || flags.ensure_model);
        match require_embedder(cfg, &opts) {
            Ok(e) => Some(e),
            Err(GpError::NoModel) if !force_semantic => {
                eprintln!("No embedding model selected; falling back to grep.");
                return run_grep(pattern, paths, &flags, cfg);
            }
            Err(GpError::NoModel) => {
                anyhow::bail!(
                    "Semantic search requires an embedding model.\n\
                     Run: grepplus models install\n\
                     Or:  grepplus --semantic --yes-download ..."
                );
            }
            Err(e) => return Err(e.into()),
        }
    } else {
        None
    };

    if (flags.ensure_index || cfg.index.auto_ensure) && embedder.is_some() {
        let sketch_only = !flags.warm_index;
        if sketch_only {
            let _ =
                crate::index::ensure_sketch_shell(&repo, &cfg.embedder.active, cfg.embedder.dim);
        } else if !Index::exists(&repo) {
            let opts = IndexBuildOptions::from_config(cfg, false);
            build_index(&repo, embedder.as_deref(), &opts).context("build index")?;
        }
    }

    let start = std::time::Instant::now();
    let search_opts = SearchOptions::from_config(cfg, decision.route);
    let results =
        hybrid_search(pattern, paths, embedder.as_deref(), &search_opts).context("search")?;
    let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

    if flags.local_traces || cfg.router.contrib_traces {
        let _ = append_trace(&RouteTrace {
            query: pattern.to_string(),
            route: route_label(decision.route).to_string(),
            latency_ms: elapsed_ms,
            success: None,
        });
    }

    if results.is_empty() {
        if !flags.quiet {
            eprintln!("note: no hybrid/semantic hits; falling back to grep");
        }
        run_grep(pattern, paths, &flags, cfg)?;
    } else {
        print_scored(&results, flags.line_numbers);
    }
    Ok(())
}

fn run_grep(pattern: &str, paths: &[PathBuf], flags: &SearchFlags, cfg: &Config) -> Result<()> {
    let backend = resolve_cli_backend(&cfg.grep.backend);
    let opts = GrepOptions {
        case_insensitive: flags.ignore_case,
        fixed_string: flags.fixed_string,
        roots: paths.to_vec(),
        include_globs: flags.globs.clone(),
        exclude_globs: crate::core::exclude_to_globs(&cfg.index.exclude),
        max_results: flags.max_count,
        files_with_matches: flags.files_with_matches,
        count_only: flags.count,
        ..Default::default()
    };
    let hits = backend.search(pattern, &opts).context("grep search")?;
    if flags.count {
        let mut by_file = std::collections::BTreeMap::<String, usize>::new();
        for hit in &hits {
            *by_file.entry(hit.file.display().to_string()).or_default() += 1;
        }
        for (file, n) in by_file {
            println!("{file}:{n}");
        }
        return Ok(());
    }
    if flags.files_with_matches {
        let mut seen = std::collections::BTreeSet::new();
        for hit in hits {
            let f = hit.file.display().to_string();
            if seen.insert(f.clone()) {
                println!("{f}");
            }
        }
        return Ok(());
    }
    for hit in hits {
        if flags.line_numbers {
            print!("{}:{}:", hit.file.display(), hit.line_no);
        }
        println!("{}", hit.line);
    }
    Ok(())
}

fn print_scored(hits: &[ScoredChunk], line_numbers: bool) {
    for hit in hits {
        if line_numbers {
            print!(
                "{}:{}-{} ",
                hit.chunk.file.display(),
                hit.chunk.start_line,
                hit.chunk.end_line
            );
        }
        if let Some(p) = &hit.preview {
            println!("{p}");
        } else {
            println!("[score={:.4} {:?}]", hit.score, hit.source);
        }
    }
}

fn build_eval_harness(
    cfg: &Config,
    corpus: PathBuf,
    suite: PathBuf,
    ensure_index: bool,
    warm_index: bool,
    isolate_modes: bool,
    yes_download: bool,
    filter_category: Option<String>,
    filter_laser_miss: bool,
    jit_embed_budget: Option<usize>,
    jit_reheat_file_cap: Option<usize>,
) -> AgentCodeHarness {
    AgentCodeHarness::new(corpus, suite)
        .with_config(cfg.clone())
        .ensure_index(ensure_index)
        .warm_index(warm_index)
        .isolate_modes(isolate_modes)
        .ensure_model(yes_download)
        .filter_category(filter_category)
        .filter_laser_miss(filter_laser_miss)
        .overrides(HarnessOverrides {
            jit_embed_budget,
            jit_reheat_file_cap,
            router_mode: None,
        })
}

fn parse_mode(s: &str) -> Result<EvalMode> {
    match s {
        "grep" | "unix-grep" | "posix-grep" => Ok(EvalMode::Grep),
        "ripgrep" | "rg" => Ok(EvalMode::Ripgrep),
        "laser" => Ok(EvalMode::Laser),
        "vector" | "semantic" => Ok(EvalMode::Vector),
        "hybrid" => Ok(EvalMode::Hybrid),
        "jit" | "progressive" => Ok(EvalMode::Jit),
        "prefocus" => Ok(EvalMode::Prefocus),
        "fixed-grep" | "always-grep" => Ok(EvalMode::FixedGrep),
        "fixed-hybrid" | "always-hybrid" => Ok(EvalMode::FixedHybrid),
        "router-heuristic" | "auto-heuristic" => Ok(EvalMode::RouterHeuristic),
        "router-feature" | "auto-feature" => Ok(EvalMode::RouterFeature),
        "router-learned" | "auto-learned" => Ok(EvalMode::RouterLearned),
        other => anyhow::bail!(
            "unknown eval mode: {other} (use grep, ripgrep, laser, vector, hybrid, jit, prefocus, fixed-grep, fixed-hybrid, router-heuristic, router-feature, router-learned)"
        ),
    }
}

fn print_metrics(mode: &str, m: &crate::core::traits::EvalMetrics) {
    println!("mode={mode}");
    println!("  recall@10: {:.3}", m.recall_at_10);
    println!("  mrr:       {:.3}", m.mrr);
    println!("  hit_rate:  {:.3}", m.hit_rate);
    if let Some(ra) = m.route_accuracy {
        println!("  route_acc: {:.3}", ra);
    }
    println!(
        "  embed_mb:  {:.3}",
        m.cumulative_embed_bytes as f64 / 1_048_576.0
    );
    println!("  cold@1:    {:.1} ms", m.cold_latency_ms);
    println!("  warm@2+:   {:.1} ms", m.warm_latency_ms);
    println!("  mean:      {:.1} ms", m.mean_latency_ms);
    for (cat, cm) in &m.per_category {
        println!(
            "  [{cat}] n={} recall@10={:.3} mrr={:.3}",
            cm.n, cm.recall_at_10, cm.mrr
        );
    }
}
