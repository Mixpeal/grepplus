//! Watch corpus files and mark index entries COOL on change.

use crate::files::{file_dir, read_file_meta, write_file_meta};
use crate::temperature::FileTemperature;
use gp_core::error::{GpError, Result};
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;

/// Block and watch `repo` until interrupted; marks changed files COOL.
pub fn watch_repo(repo: &Path) -> Result<()> {
    let index_root = crate::Index::index_path(repo);
    if !index_root.join("manifest.json").exists() {
        return Err(GpError::Index(format!(
            "no index for {} — run grepplus index first",
            repo.display()
        )));
    }

    let (tx, rx) = channel();
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(ev) = res {
                let _ = tx.send(ev);
            }
        },
        Config::default().with_poll_interval(Duration::from_secs(1)),
    )
    .map_err(|e| GpError::Other(e.to_string()))?;

    watcher
        .watch(repo, RecursiveMode::Recursive)
        .map_err(|e| GpError::Other(e.to_string()))?;

    eprintln!("watching {} (Ctrl+C to stop)", repo.display());
    loop {
        match rx.recv() {
            Ok(event) => {
                if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                    for p in event.paths {
                        mark_cool_if_indexed(&index_root, repo, &p)?;
                    }
                }
            }
            Err(_) => break,
        }
    }
    Ok(())
}

fn mark_cool_if_indexed(index_root: &Path, repo: &Path, changed: &Path) -> Result<()> {
    let rel = changed
        .strip_prefix(repo)
        .unwrap_or(changed)
        .to_string_lossy()
        .replace('\\', "/");
    let dir = file_dir(index_root, &rel);
    if !dir.join("meta.json").exists() {
        return Ok(());
    }
    let mut meta = read_file_meta(&dir)?;
    if meta.temperature == FileTemperature::Hot.as_str() {
        meta.temperature = FileTemperature::Cool.as_str().into();
        write_file_meta(&dir, &meta)?;
        eprintln!("marked COOL: {rel}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cool_mark_updates_meta() {
        crate::cache::with_isolated_cache(|| {
            let repo = tempfile::TempDir::new().expect("repo");
            let file = repo.path().join("src/a.rs");
            std::fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
            std::fs::write(&file, "fn a() {}").expect("write");
            let idx =
                crate::Index::build_sketch_only(repo.path(), "t", 8, "baseline").expect("build");
            let dir = file_dir(&idx.root, "src/a.rs");
            let mut meta = read_file_meta(&dir).expect("meta");
            meta.temperature = FileTemperature::Hot.as_str().into();
            write_file_meta(&dir, &meta).expect("write meta");
            std::fs::write(&file, "fn a() { println!(); }").expect("rewrite");
            mark_cool_if_indexed(&idx.root, repo.path(), &file).expect("mark");
            let meta = read_file_meta(&dir).expect("meta2");
            assert_eq!(meta.temperature, "cool");
        });
    }
}
