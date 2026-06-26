use crate::catalog::{Catalog, CatalogEntry};
use crate::download::{download_hf_path, is_installed, model_dir};
use crate::hf::{
    classify_embedding_model, find_mirror_repos, infer_pooling, install_local_id,
    local_id_from_repo, EmbeddingCheck, HfClient, HfConfig,
};
use crate::manifest::ModelManifest;
use crate::progress::with_spinner;
use crate::variants::{
    discover_onnx_variants, pick_mirror_repo, pick_onnx_variant, quant_label_from_path,
    OnnxVariant, OnnxVariantPick, VariantPickInstallCtx,
};
use gp_core::config::Config;
use gp_core::error::{GpError, Result};
use std::io::IsTerminal;
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct PullOptions {
    pub target: String,
    pub revision: String,
    pub quant: Option<String>,
    pub as_id: Option<String>,
    /// Skip quant picker; use the recommended (first-ranked) ONNX variant.
    pub non_interactive: bool,
    /// Show unquantized full-precision ONNX (~multi-GB) in the quant picker.
    pub include_full: bool,
    pub force: bool,
    pub pin_catalog: bool,
}

pub fn install_model(id: &str, opts: &PullOptions) -> Result<PathBuf> {
    let catalog = Catalog::merged()?;
    let entry = catalog
        .get(id)
        .ok_or_else(|| GpError::Config(format!("unknown model id: {id}")))?
        .clone();
    install_onnx(
        &entry.repo,
        &entry.revision,
        &entry.id,
        Some(&entry),
        opts,
    )
}

pub fn pull_model(opts: &PullOptions) -> Result<PathBuf> {
    let catalog = Catalog::merged()?;
    if !opts.target.contains('/') {
        if catalog.get(&opts.target).is_some() {
            return install_model(&opts.target, opts);
        }
        return Err(GpError::Config(format!(
            "unknown model id: {} (use org/repo or a catalog id)",
            opts.target
        )));
    }

    let client = HfClient::new()?;
    let repo = opts.target.clone();
    let revision = opts.revision.clone();

    let info = with_spinner("Checking Hugging Face model", || client.model_info(&repo))?;
    match classify_embedding_model(&info, opts.force)? {
        EmbeddingCheck::Ok => {}
        EmbeddingCheck::Warn(msg) => eprintln!("note: {msg}"),
        EmbeddingCheck::Reject(msg) if !opts.force => {
            return Err(GpError::Model(format!(
                "{msg} (use --force to override)"
            )));
        }
        EmbeddingCheck::Reject(_) => {}
    }

    let base_id = local_id_from_repo(&repo, None);

    install_onnx(&repo, &revision, &base_id, None, opts)
}

fn install_onnx(
    repo: &str,
    revision: &str,
    base_id: &str,
    catalog_entry: Option<&CatalogEntry>,
    opts: &PullOptions,
) -> Result<PathBuf> {
    let client = HfClient::new()?;
    let mut pull_repo = repo.to_string();
    let mut files = with_spinner("Pulling quantizations", || {
        client.list_tree(&pull_repo, revision)
    })?;
    let mut variants = discover_onnx_variants(&files);

    let mirrors: Option<Vec<String>> = if variants.is_empty() {
        if catalog_entry.is_some() {
            return Err(GpError::Model(format!(
                "{repo} has no ONNX weights (catalog entry may be stale)"
            )));
        }
        let found: Vec<String> = with_spinner("Searching ONNX exports", || {
            find_mirror_repos(&client, repo)
        })?
            .into_iter()
            .filter(|id| {
                let lower = id.to_lowercase();
                lower.contains("onnx") && !lower.contains("gguf")
            })
            .collect();
        if found.is_empty() {
            return Err(GpError::Model(format!(
                "{repo} has no ONNX weights — grep+ needs an ONNX export \
                 (safetensors/GGUF alone are not supported)"
            )));
        }
        Some(found)
    } else {
        None
    };

    let allow_back = mirrors.as_ref().is_some_and(|m| m.len() > 1);
    let default_quant = catalog_entry.map(|e| quant_label_from_path(&e.model_file));
    let install_ctx = VariantPickInstallCtx {
        base_id,
        default_quant: default_quant.as_deref(),
    };
    let variant = if let Some(mirror_list) = mirrors {
        loop {
            pull_repo = pick_mirror_repo(&mirror_list, opts.non_interactive)?;
            files = with_spinner("Pulling quantizations", || {
                client.list_tree(&pull_repo, revision)
            })?;
            variants = discover_onnx_variants(&files);
            if variants.is_empty() {
                return Err(GpError::Model(format!(
                    "mirror {pull_repo} also has no ONNX weights"
                )));
            }
            match pick_onnx_variant(
                &variants,
                opts.quant.as_deref(),
                opts.non_interactive,
                opts.include_full,
                allow_back,
                Some(&install_ctx),
            )? {
                OnnxVariantPick::Selected(v) => break v,
                OnnxVariantPick::ChangeExport => continue,
            }
        }
    } else {
        match pick_onnx_variant(
            &variants,
            opts.quant.as_deref(),
            opts.non_interactive,
            opts.include_full,
            false,
            Some(&install_ctx),
        )? {
            OnnxVariantPick::Selected(v) => v,
            OnnxVariantPick::ChangeExport => {
                return Err(GpError::Config("unexpected back in single-repo pull".into()))
            }
        }
    };

    let local_id = if let Some(explicit) = &opts.as_id {
        explicit.clone()
    } else {
        install_local_id(base_id, &variant.label, default_quant.as_deref())
    };

    if is_installed(&local_id) && !opts.force {
        eprintln!("note: {local_id} already installed");
        return Ok(model_dir(&local_id));
    }

    write_install(
        &client,
        &pull_repo,
        revision,
        base_id,
        &local_id,
        catalog_entry,
        variant,
        &files,
        opts,
    )
}

fn write_install(
    client: &HfClient,
    repo: &str,
    revision: &str,
    base_id: &str,
    local_id: &str,
    catalog_entry: Option<&CatalogEntry>,
    variant: &OnnxVariant,
    files: &[crate::hf::HfFile],
    opts: &PullOptions,
) -> Result<PathBuf> {
    let dest = model_dir(local_id);
    std::fs::create_dir_all(&dest)?;

    let mut to_fetch = vec![variant.model_file.clone()];
    to_fetch.extend(variant.extra_files.clone());

    let tokenizer_file = if files.iter().any(|f| f.path == "tokenizer.json") {
        if !to_fetch.contains(&"tokenizer.json".to_string()) {
            to_fetch.push("tokenizer.json".into());
        }
        "tokenizer.json".to_string()
    } else {
        return Err(GpError::Model(format!(
            "{repo} missing tokenizer.json required for ONNX models"
        )));
    };

    for file in &to_fetch {
        let sha = catalog_entry.and_then(|e| e.sha256_for(file));
        download_hf_path(repo, revision, file, &dest, sha)?;
    }

    let (native_dim, max_len, pooling) = if let Some(entry) = catalog_entry {
        (entry.native_dim, entry.max_len, entry.pooling.as_str())
    } else {
        infer_model_meta(client, repo, revision)?
    };

    let description = catalog_entry
        .map(|e| e.description.clone())
        .unwrap_or_else(|| format!("Pulled from {repo}"));

    let manifest = ModelManifest {
        id: local_id.to_string(),
        revision: revision.to_string(),
        repo: repo.to_string(),
        format: "onnx".into(),
        sha256: String::new(),
        model_file: variant.model_file.clone(),
        tokenizer_file,
        extra_files: variant.extra_files.clone(),
        quant: Some(variant.label.clone()),
        base_id: base_id.to_string(),
        native_dim,
        max_len,
        pooling: pooling.to_string(),
        max_batch: if pooling == "last" { 1 } else { 0 },
        description: format!("{description} [{}]", variant.label),
        size_mb: (variant.total_bytes / 1024 / 1024).max(1) as u32,
    };
    manifest.write(&dest.join("manifest.json"))?;

    if !opts.non_interactive {
        if let Err(err) = with_spinner("Verifying install", || smoke_test(local_id)) {
            let _ = std::fs::remove_dir_all(&dest);
            return Err(GpError::Model(format!(
                "{err}\n\
                 install rolled back — try another quant, e.g. \
                 `grepplus models pull {base_id} --quant model_q4f16`"
            )));
        }
    }

    if opts.pin_catalog {
        let entry = CatalogEntry {
            id: local_id.to_string(),
            repo: repo.to_string(),
            revision: revision.to_string(),
            sha256: String::new(),
            sha256_model: String::new(),
            sha256_tokenizer: String::new(),
            model_file: manifest.model_file,
            tokenizer_file: manifest.tokenizer_file,
            extra_files: manifest.extra_files,
            format: "onnx".into(),
            native_dim,
            max_len,
            pooling: pooling.to_string(),
            max_batch: manifest.max_batch,
            size_mb: (variant.total_bytes / 1024 / 1024).max(1) as u32,
            description: manifest.description,
        };
        Catalog::save_user_entry(&entry)?;
    }

    Ok(dest)
}

fn infer_model_meta(
    client: &HfClient,
    repo: &str,
    revision: &str,
) -> Result<(usize, usize, &'static str)> {
    if let Ok(cfg) = client.fetch_json::<HfConfig>(repo, revision, "config.json") {
        let dim = cfg.hidden_size.unwrap_or(384);
        let max_len = cfg.max_position_embeddings.unwrap_or(512).min(8192);
        let pooling = infer_pooling(&cfg);
        return Ok((dim, max_len, pooling));
    }
    Ok((384, 512, "mean"))
}

fn smoke_test(model_id: &str) -> Result<()> {
    let cfg = Config::default();
    let _ = crate::load_embedder(&cfg, model_id)?;
    Ok(())
}

/// Default pull/install options for interactive TTY sessions.
pub fn default_pull_opts(target: impl Into<String>) -> PullOptions {
    PullOptions {
        target: target.into(),
        revision: "main".into(),
        non_interactive: !std::io::stdin().is_terminal(),
        ..Default::default()
    }
}
