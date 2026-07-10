//! CLI smoke tests (no network / no model required).

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_grepplus"))
}

#[test]
fn help_prints_usage() {
    let out = bin().arg("--help").output().expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("grepplus") || stdout.contains("grep+"));
}

#[test]
fn literal_grep_route() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn paymentWebhook() {}\n").unwrap();
    let out = bin()
        .args(["--route", "grep", "-F", "paymentWebhook"])
        .arg(dir.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("paymentWebhook"));
}

#[test]
fn models_list_exits_zero() {
    let out = bin().args(["models", "list"]).output().expect("run");
    assert!(out.status.success());
}
