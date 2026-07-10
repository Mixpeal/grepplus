use crate::core::config::Config;
use crate::core::error::{GpError, Result};
use blake3;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Legacy in-repo directory name (pre–user-cache layout).
pub const LEGACY_INDEX_DIR: &str = ".grepplus";

const CACHE_META: &str = "cache.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMeta {
    pub corpus: String,
    pub created_at: u64,
    pub accessed_at: u64,
}

/// Root for all corpus indexes: `~/.grepplus/cache/index/`.
pub fn index_cache_root() -> PathBuf {
    Config::cache_dir().join("index")
}

/// Stable key for a corpus path (canonical absolute path → blake3 prefix).
pub fn corpus_cache_key(repo: &Path) -> String {
    let path = fs::canonicalize(repo).unwrap_or_else(|_| repo.to_path_buf());
    let normalized = path.to_string_lossy().replace('\\', "/");
    blake3::hash(normalized.as_bytes()).to_hex().to_string()[..16].to_string()
}

/// On-disk index directory for `repo` (never inside the project tree).
pub fn index_path_for(repo: &Path) -> PathBuf {
    index_cache_root().join(corpus_cache_key(repo))
}

pub fn legacy_index_path(repo: &Path) -> PathBuf {
    repo.join(LEGACY_INDEX_DIR)
}

/// Serialize tests that mutate `GREPPLUS_CACHE_DIR` (parallel-safe).
#[cfg(test)]
pub fn with_isolated_cache<F: FnOnce()>(f: F) {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cache = tempfile::TempDir::new().expect("temp cache dir");
    std::env::set_var("GREPPLUS_CACHE_DIR", cache.path());
    f();
    std::env::remove_var("GREPPLUS_CACHE_DIR");
}

pub fn legacy_index_exists(repo: &Path) -> bool {
    legacy_index_path(repo).join("manifest.json").exists()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn meta_path(root: &Path) -> PathBuf {
    root.join(CACHE_META)
}

pub fn read_cache_meta(root: &Path) -> Option<CacheMeta> {
    let raw = fs::read_to_string(meta_path(root)).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn write_cache_meta(root: &Path, corpus: &Path) -> Result<()> {
    fs::create_dir_all(root)?;
    let corpus_str = fs::canonicalize(corpus)
        .unwrap_or_else(|_| corpus.to_path_buf())
        .to_string_lossy()
        .replace('\\', "/");
    let now = now_secs();
    let meta = read_cache_meta(root).unwrap_or(CacheMeta {
        corpus: corpus_str.clone(),
        created_at: now,
        accessed_at: now,
    });
    let meta = CacheMeta {
        corpus: corpus_str,
        created_at: meta.created_at,
        accessed_at: now,
    };
    fs::write(meta_path(root), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

pub fn touch_access(root: &Path, corpus: &Path) -> Result<()> {
    write_cache_meta(root, corpus)
}

/// Remove index caches whose last access is older than `ttl_days`. Returns dirs removed.
pub fn purge_expired(ttl_days: u32) -> Result<usize> {
    purge_expired_in(&index_cache_root(), ttl_days)
}

pub fn purge_expired_in(root: &Path, ttl_days: u32) -> Result<usize> {
    if ttl_days == 0 {
        return Ok(0);
    }
    if !root.is_dir() {
        return Ok(0);
    }
    let ttl_secs = u64::from(ttl_days) * 86_400;
    let cutoff = now_secs().saturating_sub(ttl_secs);
    let mut removed = 0usize;

    for entry in fs::read_dir(root).map_err(|e| GpError::Index(e.to_string()))? {
        let entry = entry.map_err(|e| GpError::Index(e.to_string()))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let accessed = read_cache_meta(&path)
            .map(|m| m.accessed_at)
            .or_else(|| {
                fs::metadata(path.join("manifest.json"))
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
            })
            .unwrap_or(0);
        if accessed < cutoff {
            fs::remove_dir_all(&path).map_err(|e| GpError::Index(e.to_string()))?;
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn corpus_key_stable_for_same_path() {
        let dir = TempDir::new().unwrap();
        let a = corpus_cache_key(dir.path());
        let b = corpus_cache_key(dir.path());
        assert_eq!(a, b);
    }

    #[test]
    fn purge_removes_stale_entries() {
        let cache = TempDir::new().unwrap();
        let index_root = cache.path().join("index");
        let stale = index_root.join("deadbeefdeadbeef");
        fs::create_dir_all(&stale).unwrap();
        let meta = CacheMeta {
            corpus: "/tmp/old".into(),
            created_at: 0,
            accessed_at: 0,
        };
        fs::write(meta_path(&stale), serde_json::to_string(&meta).unwrap()).unwrap();

        let removed = purge_expired_in(&index_root, 7).unwrap();
        assert_eq!(removed, 1);
        assert!(!stale.exists());
    }

    #[test]
    fn index_path_not_under_corpus() {
        with_isolated_cache(|| {
            let corpus = TempDir::new().unwrap();
            let idx = index_path_for(corpus.path());
            assert!(!idx.starts_with(corpus.path()));
        });
    }
}
