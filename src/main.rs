use std::io::IsTerminal;

use anyhow::Result;
use machinery_doc_query::cli::open_cli;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Warnings (blank pages, files without text) go to stderr so stdout stays
    // clean for results. RUST_LOG=debug etc. changes the level.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_target(false)
        .without_time()
        .init();

    open_cli().await
}
