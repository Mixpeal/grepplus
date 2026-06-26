mod catalog;
mod download;
mod hf;
mod manifest;
mod onnx;
mod ort_session;
mod picker;
mod progress;
mod pull;
mod util;
mod variants;

pub use catalog::{Catalog, CatalogEntry};
pub use download::{
    download_model, is_installed, list_installed, model_dir, remove_model,
};
pub use manifest::{ModelManifest, Pooling};
pub use onnx::OnnxEmbedder;
pub use picker::{
    ensure_model, interactive_pick, interactive_use_pick, list_installed_variants, print_models_list,
    set_active_model, EnsureOptions,
};
pub use pull::{default_pull_opts, install_model, pull_model, PullOptions};

use gp_core::config::Config;
use gp_core::error::{GpError, Result};
use gp_core::traits::Embedder;
use std::sync::Arc;

pub fn load_embedder(cfg: &Config, model_id: &str) -> Result<Arc<dyn Embedder>> {
    if !is_installed(model_id) {
        return Err(GpError::NoModel);
    }
    let dir = model_dir(model_id);
    let embedder = OnnxEmbedder::load(
        &dir,
        cfg.embedder.dim,
        cfg.embedder.query_instruct.clone(),
    )?;
    Ok(Arc::new(embedder))
}

pub fn load_active_embedder(cfg: &Config) -> Result<Arc<dyn Embedder>> {
    load_embedder(cfg, &cfg.embedder.active)
}

pub fn resolve_embedder(cfg: &mut Config, opts: &EnsureOptions) -> Result<Option<Arc<dyn Embedder>>> {
    match ensure_model(cfg, opts) {
        Ok(id) => {
            cfg.embedder.active = id.clone();
            Ok(Some(load_embedder(cfg, &id)?))
        }
        Err(GpError::NoModel) if !opts.require => Ok(None),
        Err(e) => Err(e),
    }
}

pub fn require_embedder(cfg: &mut Config, opts: &EnsureOptions) -> Result<Arc<dyn Embedder>> {
    let mut opts = opts.clone();
    opts.require = true;
    resolve_embedder(cfg, &opts)?.ok_or(GpError::NoModel)
}
