//! Welcome banner when `grepplus` is run with no pattern or subcommand.

use gp_core::config::Config;
use gp_embed::{is_installed, list_installed, Catalog};

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn print_welcome(cfg: &Config) {
    let installed = list_installed().unwrap_or_default();
    let active = &cfg.embedder.active;
    let active_installed = is_installed(active);

    eprintln!("grep+ v{VERSION} — hybrid lexical + semantic search");
    eprintln!();

    eprintln!("Model");
    if active_installed {
        eprintln!("  active:    {active} (installed)");
    } else if installed.is_empty() {
        eprintln!("  active:    {active} (not installed)");
        eprintln!("  installed: none");
    } else {
        eprintln!("  active:    {active} (not installed — pick one below)");
        eprintln!(
            "  installed: {}",
            installed.join(", ")
        );
    }
    if !installed.is_empty() && !active_installed {
        eprintln!("  tip:       grepplus models use {active}");
    }
    eprintln!();

    eprintln!("Quick start");
    eprintln!("  grepplus paymentWebhook ./src           literal search (no model)");
    eprintln!("  grepplus --hybrid \"retry logic\" ./src   hybrid semantic + grep");
    eprintln!("  grepplus models install                 pick a model (↑/↓ + Enter)");
    eprintln!("  grepplus index ./src --ensure-model     build warm index");
    eprintln!("  grepplus --help                         full usage");
    eprintln!();

    eprintln!("Commands");
    eprintln!("  models    Manage embedding models (list, pull, install, use)");
    eprintln!("  index     Build and manage the search index");
    eprintln!("  serve     HTTP daemon for agents (/search, /health)");
    eprintln!();

    if let Ok(catalog) = Catalog::builtin() {
        if let Some(default) = catalog.default_id() {
            if installed.is_empty() {
                eprintln!(
                    "Semantic search needs a model. Run `grepplus models install` (~{} MB default: {default}).",
                    catalog
                        .get(default)
                        .map(|m| m.size_mb)
                        .unwrap_or(0)
                );
            }
        }
    }
}
