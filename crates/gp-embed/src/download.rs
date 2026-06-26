use crate::catalog::CatalogEntry;
use crate::manifest::ModelManifest;
use gp_core::error::{GpError, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn model_dir(id: &str) -> PathBuf {
    gp_core::config::Config::models_dir().join(id)
}

pub fn is_installed(id: &str) -> bool {
    let dir = model_dir(id);
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        return false;
    }
    match ModelManifest::read(&manifest_path) {
        Ok(m) => m.weights_exist(&dir),
        Err(_) => false,
    }
}

pub fn list_installed() -> Result<Vec<String>> {
    let dir = gp_core::config::Config::models_dir();
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            if is_installed(name) {
                out.push(name.to_string());
            }
        }
    }
    out.sort();
    Ok(out)
}

pub fn remove_model(id: &str) -> Result<()> {
    let dir = model_dir(id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(())
}

pub fn download_model(id: &str) -> Result<PathBuf> {
    let opts = crate::pull::default_pull_opts(id);
    crate::pull::install_model(id, &opts)
}

fn ensure_parent(dest: &Path, file: &str) -> Result<()> {
    if let Some(parent) = Path::new(file).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(dest.join(parent))?;
        }
    }
    Ok(())
}

pub fn download_hf_path(
    repo: &str,
    revision: &str,
    file: &str,
    dest: &Path,
    expected_sha: Option<&str>,
) -> Result<()> {
    let out_path = dest.join(file);
    if out_path.exists() {
        if let Some(expected) = expected_sha {
            if !expected.is_empty() {
                verify_sha256(&out_path, expected)?;
            }
        }
        return Ok(());
    }

    ensure_parent(dest, file)?;

    let url = format!(
        "https://huggingface.co/{repo}/resolve/{revision}/{}",
        file.replace('%', "%25")
    );

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
    let client = builder
        .build()
        .map_err(|e| GpError::Model(e.to_string()))?;

    let mut resp = client
        .get(&url)
        .send()
        .map_err(|e| GpError::Model(format!("download failed for {file}: {e}")))?;

    if !resp.status().is_success() {
        return Err(GpError::Model(format!(
            "download failed for {file} from {url}: HTTP {}",
            resp.status()
        )));
    }

    let total = resp.content_length();
    let bar = ProgressBar::new(total.unwrap_or(0));
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {bytes}/{total_bytes}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
    );
    bar.set_message(format!("downloading {file}"));

    let temp_path = dest.join(format!(".{}.part", file.replace('/', "_")));
    let mut out = File::create(&temp_path).map_err(|e| {
        GpError::Io(std::io::Error::new(
            e.kind(),
            format!("create temp file {}: {e}", temp_path.display()),
        ))
    })?;
    let mut downloaded = 0u64;
    let mut buf = [0u8; 8192];
    loop {
        let n = resp
            .read(&mut buf)
            .map_err(|e| GpError::Model(e.to_string()))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        downloaded += n as u64;
        bar.set_position(downloaded);
    }
    bar.finish_with_message(format!("downloaded {file}"));

    if let Some(expected) = expected_sha {
        if !expected.is_empty() {
            verify_sha256(&temp_path, expected)?;
        }
    }

    std::fs::rename(&temp_path, &out_path).map_err(|e| {
        GpError::Io(std::io::Error::new(
            e.kind(),
            format!(
                "rename {} → {}: {e}",
                temp_path.display(),
                out_path.display()
            ),
        ))
    })?;
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hex::encode(hasher.finalize());
    if !digest.eq_ignore_ascii_case(expected) {
        return Err(GpError::Model(format!(
            "sha256 mismatch for {}: expected {expected}, got {digest}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use tempfile::TempDir;

    #[test]
    fn verify_sha256_matches() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("t.bin");
        std::fs::write(&p, b"hello").unwrap();
        let hash = hex::encode(Sha256::digest(b"hello"));
        verify_sha256(&p, &hash).unwrap();
    }
}
