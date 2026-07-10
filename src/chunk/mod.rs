use crate::core::types::ChunkRef;
use std::path::Path;

mod ast;

use ast::chunk_ast_boundaries;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkMode {
    Line,
    Ast,
}

impl ChunkMode {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "ast" => ChunkMode::Ast,
            _ => ChunkMode::Line,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ChunkMode::Line => "line",
            ChunkMode::Ast => "ast",
        }
    }
}

pub struct ChunkConfig {
    pub max_lines: usize,
    pub overlap_lines: usize,
    pub mode: ChunkMode,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_lines: 40,
            overlap_lines: 8,
            mode: ChunkMode::Line,
        }
    }
}

impl ChunkConfig {
    pub fn from_mode(mode: &str) -> Self {
        Self {
            mode: ChunkMode::parse(mode),
            ..Self::default()
        }
    }
}

pub struct Chunk {
    pub chunk_ref: ChunkRef,
    pub text: String,
}

/// Split file content using the configured chunking strategy.
pub fn chunk_file(path: &Path, content: &str, cfg: &ChunkConfig) -> Vec<Chunk> {
    match cfg.mode {
        ChunkMode::Line => chunk_line_windows(path, content, cfg),
        ChunkMode::Ast => chunk_ast_boundaries(path, content, cfg),
    }
}

pub(crate) fn chunk_line_windows(path: &Path, content: &str, cfg: &ChunkConfig) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return vec![];
    }
    let step = cfg.max_lines.saturating_sub(cfg.overlap_lines).max(1);
    let mut out = Vec::new();
    let mut chunk_id = 0u32;
    let mut start = 0usize;

    let line_byte_offsets = line_offsets(content);

    while start < lines.len() {
        let end = (start + cfg.max_lines).min(lines.len());
        let text = lines[start..end].join("\n");
        let byte_start = line_byte_offsets[start];
        let byte_end = if end < line_byte_offsets.len() {
            line_byte_offsets[end]
        } else {
            content.len() as u64
        };

        out.push(Chunk {
            chunk_ref: ChunkRef {
                file: path.to_path_buf(),
                chunk_id,
                start_line: (start + 1) as u32,
                end_line: end as u32,
                byte_start,
                byte_end,
            },
            text,
        });
        chunk_id += 1;
        if end == lines.len() {
            break;
        }
        start += step;
    }
    out
}

pub(crate) fn line_offsets(content: &str) -> Vec<u64> {
    let mut offsets = vec![0u64];
    let mut acc = 0u64;
    for line in content.split_inclusive('\n') {
        acc += line.len() as u64;
        offsets.push(acc);
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_no_chunks() {
        let chunks = chunk_file(Path::new("a.rs"), "", &ChunkConfig::default());
        assert!(chunks.is_empty());
    }

    #[test]
    fn short_file_one_chunk() {
        let content = "fn main() {}\n";
        let chunks = chunk_file(Path::new("main.rs"), content, &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_ref.start_line, 1);
        assert_eq!(chunks[0].chunk_ref.end_line, 1);
    }

    #[test]
    fn long_file_overlapping_chunks() {
        let content: String = (0..100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let cfg = ChunkConfig {
            max_lines: 40,
            overlap_lines: 8,
            mode: ChunkMode::Line,
        };
        let chunks = chunk_file(Path::new("big.rs"), &content, &cfg);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].chunk_ref.start_line, 1);
        assert_eq!(chunks[0].chunk_ref.end_line, 40);
        assert_eq!(chunks[1].chunk_ref.start_line, 33);
    }
}
