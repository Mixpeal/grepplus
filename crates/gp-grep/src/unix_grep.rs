use gp_core::{
    error::{GpError, Result},
    traits::{GrepEngine, GrepOptions},
    types::GrepHit,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// External POSIX/BSD/GNU `grep` subprocess backend (`/usr/bin/grep` or `GREP` env).
pub struct UnixGrepEngine {
    binary: PathBuf,
}

impl UnixGrepEngine {
    /// Resolve the `grep` binary: `GREP` / `GREPPLUS_GREP` env, then `PATH`.
    pub fn discover() -> Result<Self> {
        let binary = resolve_grep_binary()?;
        Ok(Self { binary })
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }
}

impl GrepEngine for UnixGrepEngine {
    fn search(&self, pattern: &str, opts: &GrepOptions) -> Result<Vec<GrepHit>> {
        if opts.roots.is_empty() {
            return Ok(vec![]);
        }

        let mut cmd = Command::new(&self.binary);
        cmd.arg("-r")
            .arg("-n")
            .arg("-H")
            .arg("-I")
            .stdin(Stdio::null());

        if opts.case_insensitive {
            cmd.arg("-i");
        }
        if opts.fixed_string {
            cmd.arg("-F");
        } else {
            cmd.arg("-E");
        }
        if let Some(max) = opts.max_results {
            cmd.arg("-m").arg(max.to_string());
        }

        cmd.arg("--").arg(pattern);
        for root in &opts.roots {
            cmd.arg(root);
        }

        let output = cmd.output().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                GpError::Other(format!(
                    "grep binary not found at {} — install grep or set GREP=/path/to/grep",
                    self.binary.display()
                ))
            } else {
                GpError::Io(e)
            }
        })?;

        if !output.status.success() && output.status.code() != Some(1) {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(GpError::Other(format!(
                "grep failed (exit {:?}): {stderr}",
                output.status.code()
            )));
        }

        parse_grep_output(&output.stdout, opts.max_results)
    }
}

/// Returns the resolved `grep` path and verifies it runs.
pub fn resolve_grep_binary() -> Result<PathBuf> {
    let binary = if let Ok(path) = std::env::var("GREP") {
        PathBuf::from(path)
    } else if let Ok(path) = std::env::var("GREPPLUS_GREP") {
        PathBuf::from(path)
    } else {
        PathBuf::from("grep")
    };

    let output = Command::new(&binary)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                GpError::Other(
                    "grep not found — set GREP=/path/to/grep (POSIX grep required)".into(),
                )
            } else {
                GpError::Io(e)
            }
        })?;

    if !output.status.success() {
        return Err(GpError::Other(format!(
            "grep at {} failed --version",
            binary.display()
        )));
    }
    Ok(binary)
}

fn parse_grep_output(stdout: &[u8], max_total: Option<usize>) -> Result<Vec<GrepHit>> {
    let mut hits = Vec::new();
    for line in stdout.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let Some((file, rest)) = split_path_line(line) else {
            continue;
        };
        let Some((line_no, content)) = split_line_number(rest) else {
            continue;
        };
        let line_str = String::from_utf8_lossy(content).into_owned();
        let (match_start, match_end) = (0, 0usize);

        hits.push(GrepHit {
            file: PathBuf::from(file),
            line_no,
            byte_offset: 0,
            line: line_str,
            match_start,
            match_end,
        });

        if let Some(max) = max_total {
            if hits.len() >= max {
                break;
            }
        }
    }
    Ok(hits)
}

/// Split `path:lineno:content` handling Windows drive letters minimally.
fn split_path_line(line: &[u8]) -> Option<(&str, &[u8])> {
    let text = std::str::from_utf8(line).ok()?;
    if let Some(idx) = text.find(':') {
        let (path, _rest) = text.split_at(idx);
        if path.len() == 1 && path.as_bytes()[0].is_ascii_alphabetic() {
            let rest = &line[idx + 1..];
            return split_after_first_colon(rest);
        }
        return Some((path, &line[idx + 1..]));
    }
    None
}

fn split_after_first_colon(rest: &[u8]) -> Option<(&str, &[u8])> {
    let text = std::str::from_utf8(rest).ok()?;
    let idx = text.find(':')?;
    let path = std::str::from_utf8(&rest[..idx]).ok()?;
    Some((path, &rest[idx + 1..]))
}

fn split_line_number(rest: &[u8]) -> Option<(u32, &[u8])> {
    let text = std::str::from_utf8(rest).ok()?;
    let idx = text.find(':')?;
    let line_no: u32 = text[..idx].parse().ok()?;
    Some((line_no, &rest[idx + 1..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn grep_available() -> Option<UnixGrepEngine> {
        UnixGrepEngine::discover().ok()
    }

    #[test]
    fn literal_match_via_grep() {
        let Some(grep) = grep_available() else {
            eprintln!("skipping: grep not installed");
            return;
        };
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("sample.rs");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "fn handleSessionRefresh() {{").unwrap();

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
    fn max_results_honored_via_grep() {
        let Some(grep) = grep_available() else {
            eprintln!("skipping: grep not installed");
            return;
        };
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("many.txt");
        let mut f = std::fs::File::create(&p).unwrap();
        for _ in 0..20 {
            writeln!(f, "needle").unwrap();
        }
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
