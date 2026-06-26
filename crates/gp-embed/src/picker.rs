use crate::catalog::Catalog;
use crate::download::{is_installed, list_installed, model_dir};
use crate::hf::infer_base_id;
use crate::manifest::ModelManifest;
use crate::pull::{default_pull_opts, install_model, pull_model};
use console::style;
use gp_core::config::Config;
use gp_core::error::{GpError, Result};
use inquire::error::InquireError;
use inquire::{Select, Text};
use std::collections::{BTreeMap, HashSet};
use std::io::{self, IsTerminal, Write};

#[derive(Debug, Clone)]
pub struct InstalledVariant {
    pub id: String,
    pub base_id: String,
    pub quant: String,
    pub repo: String,
    pub description: String,
    pub size_mb: u32,
}

pub fn list_installed_variants() -> Result<Vec<InstalledVariant>> {
    let catalog = Catalog::merged()?;
    let mut out = Vec::new();
    for id in list_installed()? {
        let manifest = ModelManifest::read(&model_dir(&id).join("manifest.json"))?;
        let base_id = if manifest.base_id.is_empty() {
            infer_base_id(&id, manifest.quant.as_deref())
        } else {
            manifest.base_id.clone()
        };
        let quant = manifest
            .quant
            .clone()
            .unwrap_or_else(|| "unknown".into());
        let size_mb = if manifest.size_mb > 0 {
            manifest.size_mb
        } else {
            catalog
                .get(&base_id)
                .map(|e| e.size_mb)
                .filter(|_| id == base_id)
                .unwrap_or(0)
        };
        out.push(InstalledVariant {
            id,
            base_id,
            quant,
            repo: manifest.repo,
            description: manifest.description,
            size_mb,
        });
    }
    out.sort_by(|a, b| a.base_id.cmp(&b.base_id).then(a.id.cmp(&b.id)));
    Ok(out)
}

fn count_variants_for_base(base_id: &str, installed: &[InstalledVariant]) -> usize {
    installed
        .iter()
        .filter(|v| v.base_id == base_id)
        .count()
}

const ID_WIDTH: usize = 26;
const HF_EMBED_MODELS_URL: &str =
    "https://huggingface.co/models?pipeline_tag=feature-extraction&search=onnx&sort=downloads";

#[derive(Debug, Clone)]
enum ModelPickAction {
    Catalog(String),
    /// Activate an installed model family not in the catalog (e.g. from `models pull`).
    OtherFamily(String),
    HuggingFace,
    Skip,
}

enum PickFlow {
    Installed(String),
    Retry,
    Skip,
}

fn is_output_tty() -> bool {
    io::stdout().is_terminal()
}

/// Print installed models grouped by base id / family.
pub fn print_models_list(active: &str) -> Result<()> {
    let catalog = Catalog::merged()?;
    let variants = list_installed_variants()?;

    if is_output_tty() {
        println!();
        println!("{}", style("Installed models").bold().underlined());
    } else {
        println!("Installed models");
    }
    println!();

    if variants.is_empty() {
        if is_output_tty() {
            println!("{}", style("  (none)").dim());
        } else {
            println!("  (none)");
        }
        println!();
        print_install_hint(&catalog)?;
        return Ok(());
    }

    let mut groups: BTreeMap<String, Vec<InstalledVariant>> = BTreeMap::new();
    for v in variants {
        groups.entry(v.base_id.clone()).or_default().push(v);
    }

    for (base_id, group) in &groups {
        let repo = group.first().map(|v| v.repo.as_str()).unwrap_or("");
        if is_output_tty() {
            println!(
                "{}  {}",
                style(base_id).bold(),
                style(repo).dim()
            );
        } else {
            println!("{base_id}  {repo}");
        }
        for v in group {
            print_installed_variant_line(v, active);
        }
        println!();
    }

    print_install_hint(&catalog)?;
    Ok(())
}

fn print_installed_variant_line(v: &InstalledVariant, active: &str) {
    let is_active = v.id == active;
    let size = if v.size_mb > 0 {
        format!("~{} MB", v.size_mb)
    } else {
        "       ".to_string()
    };
    let tags = if is_active { " [active]" } else { "" };

    if !is_output_tty() {
        println!(
            "    {:<16} {:<26} {:>8}{tags}",
            v.quant, v.id, size
        );
        return;
    }

    let quant = style(format!("{:<16}", v.quant)).dim().to_string();
    let id = if is_active {
        style(format!("{:<26}", v.id)).cyan().bold().to_string()
    } else {
        format!("{:<26}", v.id)
    };
    let size_styled = style(format!("{:>8}", size)).dim().to_string();
    let active_styled = if is_active {
        style(" [active]").cyan().bold().to_string()
    } else {
        String::new()
    };
    println!("    {quant} {id} {size_styled}{active_styled}");
}

fn print_install_hint(catalog: &Catalog) -> Result<()> {
    let hint = if let Some(default) = catalog.default_id() {
        let size = catalog
            .get(default)
            .map(|m| m.size_mb)
            .unwrap_or(0);
        if size > 0 {
            format!(
                "Run `grepplus models install` to browse recommended models (~{size} MB default: {default})."
            )
        } else {
            format!("Run `grepplus models install` to browse recommended models (default: {default}).")
        }
    } else {
        "Run `grepplus models install` to browse recommended models.".to_string()
    };
    if is_output_tty() {
        println!("{}", style(hint).dim());
    } else {
        println!("{hint}");
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct EnsureOptions {
    /// Auto-pull default model without prompting.
    pub yes_download: bool,
    /// Allow interactive picker when TTY.
    pub interactive: bool,
    /// When true, missing model is an error (not `Ok(None)` from `resolve_embedder`).
    pub require: bool,
    /// Explicit model id (CLI flag or `GREPPLUS_MODEL` env).
    pub model_id: Option<String>,
}

impl EnsureOptions {
    pub fn from_env_and_flags(yes_download: bool) -> Self {
        let model_id = std::env::var("GREPPLUS_MODEL").ok();
        Self {
            yes_download,
            interactive: true,
            require: false,
            model_id,
        }
    }

    /// Semantic / index / serve paths: prompt on TTY, error if still missing.
    pub fn for_required_semantic(yes_download: bool) -> Self {
        Self {
            yes_download,
            interactive: !yes_download,
            require: true,
            model_id: std::env::var("GREPPLUS_MODEL").ok(),
        }
    }
}

fn is_interactive_tty() -> bool {
    io::stdin().is_terminal() && io::stderr().is_terminal()
}

/// Ensure a usable embedding model is installed. Returns the model id selected.
pub fn ensure_model(cfg: &Config, opts: &EnsureOptions) -> Result<String> {
    if let Some(id) = &opts.model_id {
        if !is_installed(id) {
            let mut pull_opts = default_pull_opts(id);
            pull_opts.non_interactive = opts.yes_download;
            install_model(id, &pull_opts)?;
        }
        return Ok(id.clone());
    }

    if is_installed(&cfg.embedder.active) {
        return Ok(cfg.embedder.active.clone());
    }

    let installed = list_installed()?;
    if installed.len() == 1 {
        return Ok(installed[0].clone());
    }
    if !installed.is_empty() && is_installed(&cfg.embedder.active) {
        return Ok(cfg.embedder.active.clone());
    }

    if opts.yes_download {
        let id = default_model_id(cfg)?;
        if !is_installed(&id) {
            let mut pull_opts = default_pull_opts(&id);
            pull_opts.non_interactive = true;
            install_model(&id, &pull_opts)?;
        }
        return Ok(id);
    }

    if opts.interactive && is_interactive_tty() {
        return interactive_pick(cfg);
    }

    Err(GpError::NoModel)
}

fn default_model_id(cfg: &Config) -> Result<String> {
    let catalog = Catalog::merged()?;
    if is_installed(&cfg.embedder.active) {
        return Ok(cfg.embedder.active.clone());
    }
    catalog
        .default_id()
        .map(str::to_string)
        .ok_or(GpError::NoModel)
}

struct ModelChoice {
    action: ModelPickAction,
    label: String,
}

fn model_choices(catalog: &Catalog, installed: &[InstalledVariant]) -> Vec<ModelChoice> {
    let default_id = catalog.default_id();
    let catalog_ids: HashSet<&str> = catalog.list().iter().map(|m| m.id.as_str()).collect();

    let mut choices: Vec<ModelChoice> = catalog
        .list()
        .iter()
        .map(|m| {
            let is_default = default_id == Some(m.id.as_str());
            let n = count_variants_for_base(&m.id, installed);
            let suffix = if n > 0 {
                format!(" ({n} installed)")
            } else {
                String::new()
            };
            let default_tag = if is_default { " [default]" } else { "" };
            ModelChoice {
                action: ModelPickAction::Catalog(m.id.clone()),
                label: format!(
                    "{:<id_w$} from ~{:>3} MB  {desc}{default_tag}{suffix}",
                    m.id,
                    m.size_mb,
                    desc = m.description,
                    id_w = ID_WIDTH,
                ),
            }
        })
        .collect();

    let mut other_by_base: BTreeMap<String, Vec<&InstalledVariant>> = BTreeMap::new();
    for v in installed {
        if !catalog_ids.contains(v.base_id.as_str()) {
            other_by_base
                .entry(v.base_id.clone())
                .or_default()
                .push(v);
        }
    }
    for (base_id, group) in other_by_base {
        let n = group.len();
        let repo = group.first().map(|v| v.repo.as_str()).unwrap_or("");
        let size_mb = group.iter().map(|v| v.size_mb).max().unwrap_or(0);
        let suffix = if n > 1 {
            format!(" ({n} installed)")
        } else {
            " [installed]".to_string()
        };
        choices.push(ModelChoice {
            action: ModelPickAction::OtherFamily(base_id.clone()),
            label: format!(
                "{:<id_w$} from ~{:>3} MB  {repo}{suffix}",
                base_id,
                size_mb,
                id_w = ID_WIDTH,
            ),
        });
    }

    choices.push(ModelChoice {
        action: ModelPickAction::HuggingFace,
        label: "Find more on Hugging Face…".into(),
    });
    choices.push(ModelChoice {
        action: ModelPickAction::Skip,
        label: "Skip for now".into(),
    });
    choices
}

pub fn interactive_pick(_cfg: &Config) -> Result<String> {
    loop {
        let catalog = Catalog::merged()?;
        let installed = list_installed_variants()?;
        let choices = model_choices(&catalog, &installed);

        let flow = if is_interactive_tty() {
            interactive_pick_inquire(&choices)?
        } else {
            interactive_pick_stdin(&choices)?
        };

        match flow {
            PickFlow::Installed(id) => return Ok(id),
            PickFlow::Retry => continue,
            PickFlow::Skip => return Err(GpError::NoModel),
        }
    }
}

fn interactive_pick_inquire(choices: &[ModelChoice]) -> Result<PickFlow> {
    let labels: Vec<&str> = choices.iter().map(|c| c.label.as_str()).collect();
    let page_size = labels.len().max(7);
    let selection = Select::new("Choose a model", labels)
        .with_page_size(page_size)
        .with_help_message(
            "↑/↓ navigate — recommended catalog, other installed, Hugging Face, or skip",
        )
        .prompt();

    match selection {
        Ok(label) => {
            let choice = choices
                .iter()
                .find(|c| c.label == label)
                .ok_or_else(|| GpError::Config("model picker: invalid selection".into()))?;
            handle_pick_action(&choice.action)
        }
        Err(InquireError::OperationCanceled) => Ok(PickFlow::Skip),
        Err(e) => Err(GpError::Config(format!("model picker: {e}"))),
    }
}

fn interactive_pick_stdin(choices: &[ModelChoice]) -> Result<PickFlow> {
    eprintln!();
    eprintln!("grep+ — choose an embedding model for semantic search");
    eprintln!("(Pure grep works without a model. You can add or change models anytime.)");
    eprintln!();
    for (i, c) in choices.iter().enumerate() {
        eprintln!("  {}. {}", i + 1, c.label);
    }
    eprint!("Selection [1]: ");
    io::stderr().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let choice = line.trim();
    let picked = if choice.is_empty() {
        &choices[0]
    } else if let Ok(n) = choice.parse::<usize>() {
        choices
            .get(n.saturating_sub(1))
            .ok_or_else(|| GpError::Config(format!("invalid selection: {choice}")))?
    } else if let Some(c) = choices.iter().find(|c| {
        matches!(&c.action, ModelPickAction::Catalog(id) if id == choice)
            || matches!(&c.action, ModelPickAction::OtherFamily(id) if id == choice)
    }) {
        c
    } else {
        return Err(GpError::Config(format!("invalid selection: {choice}")));
    };
    handle_pick_action(&picked.action)
}

fn handle_pick_action(action: &ModelPickAction) -> Result<PickFlow> {
    match action {
        ModelPickAction::Catalog(id) => {
            let id = install_and_activate_catalog(id)?;
            Ok(PickFlow::Installed(id))
        }
        ModelPickAction::OtherFamily(base_id) => activate_other_family(base_id),
        ModelPickAction::HuggingFace => browse_hf_install(),
        ModelPickAction::Skip => Ok(PickFlow::Skip),
    }
}

fn activate_other_family(base_id: &str) -> Result<PickFlow> {
    let installed = list_installed_variants()?;
    let variants: Vec<_> = installed
        .iter()
        .filter(|v| v.base_id == base_id)
        .collect();
    if variants.is_empty() {
        return Err(GpError::Config(format!("no installed variants for {base_id}")));
    }
    if variants.len() == 1 {
        set_active_model(&variants[0].id)?;
        return Ok(PickFlow::Installed(variants[0].id.clone()));
    }
    if !is_interactive_tty() {
        return Err(GpError::Config(format!(
            "pass a variant id, e.g. `grepplus models use {}`",
            variants[0].id
        )));
    }

    let labels: Vec<String> = variants
        .iter()
        .map(|v| format_use_choice(v, ""))
        .collect();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let selection = Select::new("Choose installed variant", label_refs)
        .with_help_message("Multiple quants installed for this model")
        .prompt();

    match selection {
        Ok(label) => {
            let idx = labels.iter().position(|l| l == &label).unwrap_or(0);
            let id = variants[idx].id.clone();
            set_active_model(&id)?;
            Ok(PickFlow::Installed(id))
        }
        Err(InquireError::OperationCanceled) => Ok(PickFlow::Retry),
        Err(e) => Err(GpError::Config(format!("variant picker: {e}"))),
    }
}

fn install_and_activate_catalog(catalog_id: &str) -> Result<String> {
    let mut pull_opts = default_pull_opts(catalog_id);
    pull_opts.non_interactive = false;
    let path = install_model(catalog_id, &pull_opts)?;
    let id = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(catalog_id);
    set_active_model(id)?;
    Ok(id.to_string())
}

fn browse_hf_install() -> Result<PickFlow> {
    open_hf_models_page();
    eprintln!();
    eprintln!("Browse ONNX embedding models at:");
    eprintln!("  {HF_EMBED_MODELS_URL}");
    eprintln!();

    let repo = Text::new("Paste Hugging Face repo (org/name)")
        .with_help_message("Needs an ONNX export — Enter alone returns to the model list")
        .prompt();

    let Ok(repo) = repo else {
        return Ok(PickFlow::Retry);
    };
    let repo = repo.trim();
    if repo.is_empty() {
        return Ok(PickFlow::Retry);
    }
    if !repo.contains('/') || repo.starts_with('/') || repo.ends_with('/') {
        return Err(GpError::Config(format!(
            "invalid repo {repo:?} — use org/name, e.g. intfloat/e5-small-v2"
        )));
    }

    let mut opts = default_pull_opts(repo);
    opts.non_interactive = false;
    let path = pull_model(&opts)?;
    let id = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(repo);
    set_active_model(id)?;
    Ok(PickFlow::Installed(id.to_string()))
}

fn open_hf_models_page() {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open")
        .arg(HF_EMBED_MODELS_URL)
        .spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open")
        .arg(HF_EMBED_MODELS_URL)
        .spawn();
    #[cfg(windows)]
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", "", HF_EMBED_MODELS_URL])
        .spawn();
}

/// Pick an installed variant to activate (`models use` with no id).
pub fn interactive_use_pick(active: &str) -> Result<String> {
    let variants = list_installed_variants()?;
    if variants.is_empty() {
        return Err(GpError::Config(
            "no models installed — run `grepplus models install`".into(),
        ));
    }
    if variants.len() == 1 {
        return Ok(variants[0].id.clone());
    }
    if !is_interactive_tty() {
        return Err(GpError::Config(
            "pass a model id (e.g. `grepplus models use harrier-oss-v1-0.6b-model_fp16`)".into(),
        ));
    }

    let labels: Vec<String> = variants
        .iter()
        .map(|v| format_use_choice(v, active))
        .collect();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let selection = Select::new("Choose active model", label_refs)
        .with_help_message("Grouped by model family — pick a specific quant/install id")
        .prompt();

    match selection {
        Ok(label) => {
            let idx = labels.iter().position(|l| l == &label).unwrap_or(0);
            Ok(variants[idx].id.clone())
        }
        Err(InquireError::OperationCanceled) => {
            Err(GpError::Config("model selection cancelled".into()))
        }
        Err(e) => Err(GpError::Config(format!("model picker: {e}"))),
    }
}

fn format_use_choice(v: &InstalledVariant, active: &str) -> String {
    let active_tag = if v.id == active { " [active]" } else { "" };
    let size = if v.size_mb > 0 {
        format!(" ~{} MB", v.size_mb)
    } else {
        String::new()
    };
    format!(
        "[{}] {} · {}{}{}",
        v.base_id, v.quant, v.id, size, active_tag
    )
}

pub fn set_active_model(id: &str) -> Result<()> {
    let catalog = Catalog::merged()?;
    if catalog.get(id).is_none() && !is_installed(id) {
        return Err(GpError::Config(format!("unknown model id: {id}")));
    }
    Config::set_active_embedder(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_from_catalog() {
        let cfg = Config::default();
        let id = default_model_id(&cfg).unwrap();
        assert_eq!(id, "qwen3-embedding-0.6b");
    }
}
