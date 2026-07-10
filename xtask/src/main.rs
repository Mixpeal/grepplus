//! CI smoke: mini eval compare on agentcode mini.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // xtask's manifest dir is `xtask/`; workspace root is the parent.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask is a workspace member")
        .to_path_buf();
    let grepplus = root.join("target/release/grepplus");
    if !grepplus.exists() {
        let status = Command::new("cargo")
            .args(["build", "--release", "-p", "grepplus"])
            .current_dir(&root)
            .status()
            .expect("cargo build");
        assert!(status.success());
    }

    let corpus = root.join("eval/agentcode/repos/mini");
    let suite = root.join("eval/agentcode/queries.jsonl");

    let eval = Command::new(&grepplus)
        .args([
            "eval",
            "compare",
            corpus.to_str().unwrap(),
            "--suite",
            suite.to_str().unwrap(),
            "--modes",
            "laser,hybrid",
            "--ensure-index",
        ])
        .current_dir(&root)
        .status()
        .expect("eval compare");
    assert!(eval.success(), "eval compare failed");

    println!("xtask smoke: OK");
}
