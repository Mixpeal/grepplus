use grepplus::core::traits::{GrepEngine, GrepOptions};
use grepplus::grep::ParallelGrep;
use std::io::Write;
use tempfile::TempDir;

#[test]
fn literal_match() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("sample.rs");
    let mut f = std::fs::File::create(&p).unwrap();
    writeln!(f, "fn handleSessionRefresh() {{").unwrap();
    writeln!(f, "    // retry logic").unwrap();
    writeln!(f, "}}").unwrap();

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
fn case_insensitive_literal() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("sample.rs");
    std::fs::write(&p, "Hello World\n").unwrap();
    let grep = ParallelGrep;
    let hits = grep
        .search(
            "hello",
            &GrepOptions {
                roots: vec![dir.path().to_path_buf()],
                fixed_string: true,
                case_insensitive: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn regex_metacharacters() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("sample.rs");
    std::fs::write(&p, "foo123bar\n").unwrap();
    let grep = ParallelGrep;
    let hits = grep
        .search(
            r"foo\d+bar",
            &GrepOptions {
                roots: vec![dir.path().to_path_buf()],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn binary_file_skipped() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("binary.bin");
    std::fs::write(&p, b"needle\0binary").unwrap();
    let grep = ParallelGrep;
    let hits = grep
        .search(
            "needle",
            &GrepOptions {
                roots: vec![dir.path().to_path_buf()],
                fixed_string: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn gitignore_respected() {
    let dir = TempDir::new().unwrap();
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output()
        .ok();
    std::fs::write(dir.path().join(".gitignore"), "ignored/\n").unwrap();
    std::fs::create_dir(dir.path().join("ignored")).unwrap();
    std::fs::write(dir.path().join("ignored/hidden.txt"), "secret\n").unwrap();
    std::fs::write(dir.path().join("visible.txt"), "secret\n").unwrap();

    let grep = ParallelGrep;
    let hits = grep
        .search(
            "secret",
            &GrepOptions {
                roots: vec![dir.path().to_path_buf()],
                fixed_string: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].file.ends_with("visible.txt"));
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
