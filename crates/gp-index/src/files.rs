use crate::store::{path_hash, FileMeta, StoredChunkRecord};
use crate::temperature::{resolve_temperature, FileTemperature, TemperatureStats};
use crate::StoredChunk;
use gp_core::error::{GpError, Result};
use crate::vectors::VectorCodec;
use gp_core::traits::Q4Code;
use gp_core::types::ChunkRef;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub fn files_root(index_root: &Path) -> PathBuf {
    index_root.join("files")
}

pub fn file_dir(index_root: &Path, rel_path: &str) -> PathBuf {
    files_root(index_root).join(path_hash(rel_path))
}

pub fn read_file_meta(dir: &Path) -> Result<FileMeta> {
    let raw = std::fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| GpError::Index(e.to_string()))?;
    serde_json::from_str(&raw).map_err(|e| GpError::Index(e.to_string()))
}

pub fn write_file_meta(dir: &Path, meta: &FileMeta) -> Result<()> {
    std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

pub fn read_chunk_records(dir: &Path) -> Result<Vec<StoredChunkRecord>> {
    let path = dir.join("chunks.json");
    if !path.exists() {
        return Ok(vec![]);
    }
    let raw = std::fs::read_to_string(&path).map_err(|e| GpError::Index(e.to_string()))?;
    serde_json::from_str(&raw).map_err(|e| GpError::Index(e.to_string()))
}

pub fn write_chunk_records(dir: &Path, records: &[StoredChunkRecord]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("chunks.json"), serde_json::to_string_pretty(records)?)?;
    Ok(())
}

pub fn read_vectors(dir: &Path) -> Result<Vec<Q4Code>> {
    let path = dir.join("vectors.pq4.json");
    if !path.exists() {
        return Ok(vec![]);
    }
    let raw = std::fs::read_to_string(&path).map_err(|e| GpError::Index(e.to_string()))?;
    serde_json::from_str(&raw).map_err(|e| GpError::Index(e.to_string()))
}

pub fn write_vectors(dir: &Path, codes: &[Q4Code]) -> Result<()> {
    std::fs::write(dir.join("vectors.pq4.json"), serde_json::to_string(codes)?)?;
    Ok(())
}

pub fn has_vectors(dir: &Path) -> bool {
    dir.join("vectors.pq4.json").exists()
}

pub fn rel_path_for(repo: &Path, file: &Path) -> String {
    file.strip_prefix(repo)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/")
}

pub fn temperature_stats(index_root: &Path, repo: &Path) -> Result<TemperatureStats> {
    let root = files_root(index_root);
    if !root.is_dir() {
        return Ok(TemperatureStats::default());
    }
    let mut stats = TemperatureStats::default();
    for entry in std::fs::read_dir(&root).map_err(|e| GpError::Index(e.to_string()))? {
        let entry = entry.map_err(|e| GpError::Index(e.to_string()))?;
        if !entry.path().is_dir() {
            continue;
        }
        let meta = read_file_meta(&entry.path())?;
        let source = repo.join(&meta.path);
        let temp = resolve_temperature(&meta, &source, has_vectors(&entry.path()));
        stats.total_files += 1;
        stats.embedded_chunks += meta.chunk_count;
        match temp {
            FileTemperature::Hot => stats.hot += 1,
            FileTemperature::Cold => stats.cold += 1,
            FileTemperature::Cool => stats.cool += 1,
        }
    }
    Ok(stats)
}

/// Load HOT chunks for a file from cache (skip embed).
pub fn load_hot_chunks(
    index_root: &Path,
    repo: &Path,
    rel: &str,
) -> Result<Option<Vec<StoredChunk>>> {
    let dir = file_dir(index_root, rel);
    if !dir.is_dir() {
        return Ok(None);
    }
    let meta = read_file_meta(&dir)?;
    let source = repo.join(rel);
    if resolve_temperature(&meta, &source, has_vectors(&dir)) != FileTemperature::Hot {
        return Ok(None);
    }
    let records = read_chunk_records(&dir)?;
    let codes = read_vectors(&dir)?;
    if records.len() != codes.len() {
        return Err(GpError::Index(format!(
            "chunk/vector mismatch for {rel}"
        )));
    }
    let out: Vec<StoredChunk> = records
        .into_iter()
        .zip(codes)
        .map(|(r, code)| StoredChunk {
            chunk_ref: r.chunk_ref,
            text: r.text,
            code,
        })
        .collect();
    Ok(Some(out))
}

fn chunk_key(c: &ChunkRef) -> (String, u32, u32) {
    (
        c.file.to_string_lossy().into_owned(),
        c.start_line,
        c.end_line,
    )
}

fn matches_candidates(chunk: &ChunkRef, cand: &HashSet<(String, u32, u32)>) -> bool {
    cand.contains(&chunk_key(chunk))
}

struct ReheatPlan {
    rel: String,
    records: Vec<StoredChunkRecord>,
    text_start: usize,
}

/// JIT semantic search: score HOT files from cache, batch-reheat COLD/COOL files in beam.
pub fn jit_semantic_search(
    index_root: &Path,
    repo: &Path,
    query_vec: &[f32],
    codec: &VectorCodec,
    candidates: &[ChunkRef],
    embed_fn: &mut dyn FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
    embed_budget: usize,
    reheat_file_cap: usize,
    cold_first_file_cap: usize,
    cold_first_embed_budget: usize,
    index_mostly_cold: bool,
    top_k: usize,
    embed_dim: usize,
    embed_stats: Option<&gp_core::embed_stats::EmbedStatsCell>,
) -> Result<Vec<(StoredChunk, f32)>> {
    if candidates.is_empty() {
        return Ok(vec![]);
    }

    let cand_set: HashSet<_> = candidates.iter().map(chunk_key).collect();

    // Unique files touched by candidates, preserve first-seen order (beam rank).
    let mut file_order: Vec<String> = Vec::new();
    let mut seen_files = HashSet::new();
    for c in candidates {
        let rel = rel_path_for(repo, &c.file);
        if seen_files.insert(rel.clone()) {
            file_order.push(rel);
        }
    }

    let mut scored: Vec<(StoredChunk, f32)> = Vec::new();
    let mut reheat_queue: Vec<String> = Vec::new();

    for rel in &file_order {
        let dir = file_dir(index_root, rel);
        let source = repo.join(rel);
        let temp = if dir.join("meta.json").exists() {
            let meta = read_file_meta(&dir)?;
            resolve_temperature(&meta, &source, has_vectors(&dir))
        } else {
            FileTemperature::Cold
        };

        if temp == FileTemperature::Hot {
            if let Some(chunks) = load_hot_chunks(index_root, repo, rel)? {
                for chunk in chunks {
                    if matches_candidates(&chunk.chunk_ref, &cand_set) {
                        let score = codec.score(query_vec, &chunk.code);
                        scored.push((chunk, score));
                    }
                }
            }
            continue;
        }

        reheat_queue.push(rel.clone());
    }

    if reheat_queue.is_empty() || embed_budget == 0 {
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        return Ok(scored);
    }

    let use_cold_limits = index_mostly_cold && scored.is_empty();
    let effective_file_cap = if use_cold_limits {
        reheat_file_cap.min(cold_first_file_cap)
    } else {
        reheat_file_cap
    };
    let effective_budget = if use_cold_limits {
        embed_budget.min(cold_first_embed_budget)
    } else {
        embed_budget
    };

    // Plan batch reheat: lazy-read chunk records only for cold files in beam order.
    let mut plans: Vec<ReheatPlan> = Vec::new();
    let mut batch_texts: Vec<String> = Vec::new();
    let mut chunks_planned = 0usize;

    for rel in reheat_queue.into_iter().take(effective_file_cap) {
        if chunks_planned >= effective_budget {
            break;
        }
        let dir = file_dir(index_root, &rel);
        let records = read_chunk_records(&dir)?;
        if records.is_empty() {
            continue;
        }
        let room = effective_budget.saturating_sub(chunks_planned);
        if records.len() > room {
            continue;
        }
        let text_start = batch_texts.len();
        for r in &records {
            batch_texts.push(r.text.clone());
        }
        chunks_planned += records.len();
        plans.push(ReheatPlan {
            rel,
            records,
            text_start,
        });
    }

    if plans.is_empty() {
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        return Ok(scored);
    }

    let vectors = embed_fn(&batch_texts)?;
    if let Some(stats) = embed_stats {
        stats.record_chunks(batch_texts.len(), embed_dim);
    }

    for plan in plans {
        let n = plan.records.len();
        let slice = &vectors[plan.text_start..plan.text_start + n];
        let codes: Vec<Q4Code> = slice.iter().map(|v| codec.project(v)).collect();

        let dir = file_dir(index_root, &plan.rel);
        std::fs::create_dir_all(&dir)?;
        write_chunk_records(&dir, &plan.records)?;
        write_vectors(&dir, &codes)?;

        let content_hash = file_content_hash_for_repo(repo, &plan.rel);
        let meta = FileMeta {
            path: plan.rel.clone(),
            content_hash,
            chunk_count: plan.records.len(),
            temperature: FileTemperature::Hot.as_str().into(),
        };
        write_file_meta(&dir, &meta)?;

        for (record, code) in plan.records.into_iter().zip(codes) {
            let chunk = StoredChunk {
                chunk_ref: record.chunk_ref,
                text: record.text,
                code,
            };
            if matches_candidates(&chunk.chunk_ref, &cand_set) {
                let score = codec.score(query_vec, &chunk.code);
                scored.push((chunk, score));
            }
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(top_k);
    Ok(scored)
}

fn file_content_hash_for_repo(repo: &Path, rel: &str) -> String {
    crate::temperature::file_content_hash(&repo.join(rel)).unwrap_or_default()
}
