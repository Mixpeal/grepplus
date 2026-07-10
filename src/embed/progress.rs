use crate::core::error::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, IsTerminal};
use std::time::Duration;

/// Run a slow step with a stderr spinner on TTY (e.g. Hugging Face tree scans).
pub fn with_spinner<T, F>(message: &str, work: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    if !io::stderr().is_terminal() {
        eprintln!("{message}...");
        return work();
    }

    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    bar.set_message(message.to_string());
    bar.enable_steady_tick(Duration::from_millis(80));

    let result = work();
    bar.finish_and_clear();
    result
}
