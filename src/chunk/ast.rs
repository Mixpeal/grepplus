//! Function-boundary chunking (Phase 3 AST-aware path without tree-sitter bindings).

use crate::chunk::{line_offsets, Chunk, ChunkConfig};
use crate::core::types::ChunkRef;
use regex::Regex;
use std::path::Path;
use std::sync::LazyLock;

static FN_START: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(?:pub\s+)?(?:async\s+)?fn\s+\w+").expect("fn regex"));

static TYPE_START: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(?:pub\s+)?(?:struct|enum|impl|trait|class|interface|function)\s+\w+")
        .expect("type regex")
});

/// Split content at top-level function/type boundaries, then cap span size.
pub fn chunk_ast_boundaries(path: &Path, content: &str, cfg: &ChunkConfig) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return vec![];
    }

    let offsets = line_offsets(content);
    let mut boundaries: Vec<usize> = vec![0];
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            continue;
        }
        if FN_START.is_match(line) || TYPE_START.is_match(line) {
            boundaries.push(i);
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries.push(lines.len());

    let mut out = Vec::new();
    let mut chunk_id = 0u32;
    for w in boundaries.windows(2) {
        let start = w[0];
        let end = w[1].min(start + cfg.max_lines);
        if start >= lines.len() {
            continue;
        }
        let end = end.max(start + 1).min(lines.len());
        let text = lines[start..end].join("\n");
        if text.trim().is_empty() {
            continue;
        }
        let byte_start = offsets[start];
        let byte_end = if end < offsets.len() {
            offsets[end]
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
    }
    if out.is_empty() {
        return crate::chunk::chunk_line_windows(path, content, cfg);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_function_boundaries() {
        let content = "fn a() {}\nfn b() {\n  let x = 1;\n}\n";
        let chunks = chunk_ast_boundaries(Path::new("t.rs"), content, &ChunkConfig::default());
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].chunk_ref.start_line, 1);
        assert_eq!(chunks[1].chunk_ref.start_line, 2);
    }
}
