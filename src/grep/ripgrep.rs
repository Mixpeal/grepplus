use crate::core::{
    error::{GpError, Result},
    traits::{GrepEngine, GrepOptions},
    types::GrepHit,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// External [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) subprocess backend.
pub struct RipgrepEngine {
    binary: PathBuf,
}

impl RipgrepEngine {
    /// Resolve the `rg` binary: `RIPGREP` env, then `PATH`.
    pub fn discover() -> Result<Self> {
        let binary = resolve_ripgrep_binary()?;
        Ok(Self { binary })
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }
}

impl GrepEngine for RipgrepEngine {
    fn search(&self, pattern: &str, opts: &GrepOptions) -> Result<Vec<GrepHit>> {
        if opts.roots.is_empty() {
            return Ok(vec![]);
        }

        let mut cmd = Command::new(&self.binary);
        cmd.arg("--json")
            .arg("--line-number")
            .arg("--no-heading")
            .stdin(Stdio::null());

        if opts.case_insensitive {
            cmd.arg("-i");
        }
        if opts.fixed_string {
            cmd.arg("-F");
        }
        if let Some(max) = opts.max_results {
            cmd.arg("--max-count").arg(max.to_string());
        }
        for glob in &opts.include_globs {
            cmd.arg("-g").arg(glob);
        }
        for glob in &opts.exclude_globs {
            cmd.arg("-g").arg(format!("!{glob}"));
        }

        cmd.arg("--").arg(pattern);
        for root in &opts.roots {
            cmd.arg(root);
        }

        let output = cmd.output().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                GpError::Other(format!(
                    "ripgrep binary not found at {} — install rg (brew install ripgrep) \
                     or set RIPGREP=/path/to/rg",
                    self.binary.display()
                ))
            } else {
                GpError::Io(e)
            }
        })?;

        if !output.status.success() && output.status.code() != Some(1) {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(GpError::Other(format!(
                "ripgrep failed (exit {:?}): {stderr}",
                output.status.code()
            )));
        }

        parse_json_matches(&output.stdout)
    }
}

/// Returns the resolved `rg` path and verifies it runs.
pub fn resolve_ripgrep_binary() -> Result<PathBuf> {
    let binary = if let Ok(path) = std::env::var("RIPGREP") {
        PathBuf::from(path)
    } else if let Ok(path) = std::env::var("GREPPLUS_RIPGREP") {
        PathBuf::from(path)
    } else {
        PathBuf::from("rg")
    };

    let output = Command::new(&binary)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                GpError::Other(
                    "ripgrep (rg) not found — install: brew install ripgrep, \
                     cargo install ripgrep, or set RIPGREP=/path/to/rg"
                        .into(),
                )
            } else {
                GpError::Io(e)
            }
        })?;

    if !output.status.success() {
        return Err(GpError::Other(format!(
            "ripgrep at {} failed --version",
            binary.display()
        )));
    }
    Ok(binary)
}

#[derive(Debug, Deserialize)]
struct JsonLine {
    #[serde(rename = "type")]
    kind: String,
    data: Option<JsonMatchData>,
}

#[derive(Debug, Deserialize)]
struct JsonMatchData {
    path: JsonText,
    lines: JsonText,
    line_number: u32,
    absolute_offset: u64,
    submatches: Option<Vec<JsonSubmatch>>,
}

#[derive(Debug, Deserialize)]
struct JsonText {
    text: String,
}

#[derive(Debug, Deserialize)]
struct JsonSubmatch {
    start: usize,
    end: usize,
}

fn parse_json_matches(stdout: &[u8]) -> Result<Vec<GrepHit>> {
    let mut hits = Vec::new();
    for line in stdout.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let row: JsonLine = match serde_json::from_slice(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if row.kind != "match" {
            continue;
        }
        let Some(data) = row.data else {
            continue;
        };
        let (match_start, match_end) = data
            .submatches
            .as_ref()
            .and_then(|s| s.first())
            .map(|m| (m.start, m.end))
            .unwrap_or((0, 0));

        hits.push(GrepHit {
            file: PathBuf::from(data.path.text),
            line_no: data.line_number,
            byte_offset: data.absolute_offset,
            line: data.lines.text,
            match_start,
            match_end,
        });
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn rg_available() -> Option<RipgrepEngine> {
        RipgrepEngine::discover().ok()
    }

    #[test]
    fn literal_match_via_rg() {
        let Some(rg) = rg_available() else {
            eprintln!("skipping: rg not installed");
            return;
        };
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("sample.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "fn handleSessionRefresh() {{").unwrap();

        let hits = rg
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
    fn max_results_honored_via_rg() {
        let Some(rg) = rg_available() else {
            eprintln!("skipping: rg not installed");
            return;
        };
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("many.txt");
        let mut f = std::fs::File::create(&p).unwrap();
        for _ in 0..20 {
            writeln!(f, "needle").unwrap();
        }
        let hits = rg
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
