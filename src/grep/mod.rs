mod lexical;
mod ripgrep;
mod unix_grep;

pub use lexical::{
    exact_grep_scored, exact_pattern, hits_to_chunks, resolve_cli_backend, resolve_exact_backend,
    GrepBackend,
};
pub use ripgrep::{resolve_ripgrep_binary, RipgrepEngine};
pub use unix_grep::{resolve_grep_binary, UnixGrepEngine};

use crate::core::{
    error::Result,
    traits::{GrepEngine, GrepOptions},
    types::GrepHit,
};
use ignore::{overrides::OverrideBuilder, WalkBuilder, WalkState};
use memchr::memmem;
use regex::bytes::Regex;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct ParallelGrep;

enum Matcher {
    Literal {
        finder: memmem::Finder<'static>,
        needle_len: usize,
    },
    Regex(Regex),
}

impl Matcher {
    fn build(pattern: &str, opts: &GrepOptions) -> Result<Self> {
        if opts.fixed_string {
            if opts.case_insensitive {
                let re = regex::bytes::RegexBuilder::new(&regex::escape(pattern))
                    .case_insensitive(true)
                    .build()?;
                return Ok(Matcher::Regex(re));
            }
            let finder = memmem::Finder::new(pattern.as_bytes()).into_owned();
            Ok(Matcher::Literal {
                finder,
                needle_len: pattern.len(),
            })
        } else {
            let re = regex::bytes::RegexBuilder::new(pattern)
                .case_insensitive(opts.case_insensitive)
                .build()?;
            Ok(Matcher::Regex(re))
        }
    }

    fn find(&self, line: &[u8]) -> Option<(usize, usize)> {
        match self {
            Matcher::Literal { finder, needle_len } => {
                finder.find(line).map(|s| (s, s + needle_len))
            }
            Matcher::Regex(re) => re.find(line).map(|m| (m.start(), m.end())),
        }
    }
}

impl GrepEngine for ParallelGrep {
    fn search(&self, pattern: &str, opts: &GrepOptions) -> Result<Vec<GrepHit>> {
        let matcher = Arc::new(Matcher::build(pattern, opts)?);
        let results = Arc::new(Mutex::new(Vec::<GrepHit>::new()));

        let roots = if opts.roots.is_empty() {
            vec![std::path::PathBuf::from(".")]
        } else {
            opts.roots.clone()
        };

        let mut builder = WalkBuilder::new(&roots[0]);
        for r in &roots[1..] {
            builder.add(r);
        }
        builder.standard_filters(true).threads(num_cpus::get());

        if !opts.include_globs.is_empty() || !opts.exclude_globs.is_empty() {
            let mut overrides = OverrideBuilder::new(&roots[0]);
            for g in &opts.include_globs {
                overrides
                    .add(g)
                    .map_err(|e| crate::core::error::GpError::Other(e.to_string()))?;
            }
            for g in &opts.exclude_globs {
                overrides
                    .add(&format!("!{g}"))
                    .map_err(|e| crate::core::error::GpError::Other(e.to_string()))?;
            }
            builder.overrides(
                overrides
                    .build()
                    .map_err(|e| crate::core::error::GpError::Other(e.to_string()))?,
            );
        }

        let max = opts.max_results;
        let context = opts.context_lines;
        let walker = builder.build_parallel();

        walker.run(|| {
            let matcher = Arc::clone(&matcher);
            let results = Arc::clone(&results);
            Box::new(move |entry| {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => return WalkState::Continue,
                };
                if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    return WalkState::Continue;
                }
                let path = entry.path();
                if let Ok(hits) = scan_file(path, &matcher, context) {
                    if !hits.is_empty() {
                        let mut guard = results.lock().unwrap();
                        guard.extend(hits);
                        if let Some(m) = max {
                            if guard.len() >= m {
                                return WalkState::Quit;
                            }
                        }
                    }
                }
                WalkState::Continue
            })
        });

        let mut out = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        if let Some(m) = max {
            out.truncate(m);
        }
        Ok(out)
    }
}

fn scan_file(path: &Path, matcher: &Matcher, context_lines: usize) -> Result<Vec<GrepHit>> {
    let bytes = std::fs::read(path)?;
    if bytes.iter().take(8192).any(|&b| b == 0) {
        return Ok(vec![]);
    }

    let mut hits = Vec::new();
    let mut line_no: u32 = 0;
    let mut offset: u64 = 0;
    let mut recent: Vec<(u32, u64, Vec<u8>)> = Vec::new();

    for line in bytes.split_inclusive(|&b| b == b'\n') {
        line_no += 1;
        let trimmed = strip_newline(line);
        recent.push((line_no, offset, trimmed.to_vec()));
        if recent.len() > context_lines + 1 {
            recent.remove(0);
        }

        if let Some((ms, me)) = matcher.find(trimmed) {
            if context_lines > 0 {
                for (ctx_line_no, ctx_offset, ctx_line) in &recent[..recent.len().saturating_sub(1)]
                {
                    hits.push(GrepHit {
                        file: path.to_path_buf(),
                        line_no: *ctx_line_no,
                        byte_offset: *ctx_offset,
                        line: String::from_utf8_lossy(ctx_line).into_owned(),
                        match_start: 0,
                        match_end: 0,
                    });
                }
            }
            hits.push(GrepHit {
                file: path.to_path_buf(),
                line_no,
                byte_offset: offset,
                line: String::from_utf8_lossy(trimmed).into_owned(),
                match_start: ms,
                match_end: me,
            });
        }
        offset += line.len() as u64;
    }
    Ok(hits)
}

fn strip_newline(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    if end > 0 && line[end - 1] == b'\n' {
        end -= 1;
    }
    if end > 0 && line[end - 1] == b'\r' {
        end -= 1;
    }
    &line[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::GrepOptions;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_fixture(dir: &TempDir) -> std::path::PathBuf {
        let p = dir.path().join("sample.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "fn handleSessionRefresh() {{").unwrap();
        writeln!(f, "    // retry logic").unwrap();
        writeln!(f, "}}").unwrap();
        p
    }

    #[test]
    fn literal_match() {
        let dir = TempDir::new().unwrap();
        write_fixture(&dir);
        let grep = ParallelGrep;
        let hits = grep
            .search(
                "handleSessionRefresh",
                &GrepOptions {
                    roots: vec![dir.path().to_path_buf()],
                    fixed_string: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn max_results_honored() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("many.txt");
        let mut f = std::fs::File::create(&p).unwrap();
        for _ in 0..20 {
            writeln!(f, "needle").unwrap();
        }
        let grep = ParallelGrep;
        let hits = grep
            .search(
                "needle",
                &GrepOptions {
                    roots: vec![dir.path().to_path_buf()],
                    fixed_string: true,
                    max_results: Some(5),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(hits.len(), 5);
    }
}
