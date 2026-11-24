// Runs the real binary the way a user does: arguments and environment in, text and an
// exit code out. The tests about a Qdrant that is down need no Docker. The others share
// the throwaway container of tests/common, so each of them uses collections of its own.

mod common;

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use common::{client, qdrant_url};
use machinery_doc_query::vector_store::qdrant::QdrantStore;
use predicates::prelude::*;
use qdrant_client::qdrant::ScrollPointsBuilder;
use tempfile::TempDir;

// Size of the vectors the hash provider makes in these tests.
const HASH_SIZE: u16 = 8;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pdfs")
}

// The binary with exactly the settings given here. It runs in an empty folder with an
// empty environment, so neither a .env file nor the variables of whoever runs the
// tests can change what it does. The Azure settings are placeholders: the config asks
// for them even when the hash provider is chosen.
fn machinery(folder: &TempDir, qdrant_url: &str, collection: &str) -> Command {
    let mut command = assert_cmd::cargo::cargo_bin_cmd!("machinery-doc-query");
    command
        .current_dir(folder.path())
        .env_clear()
        .env(
            "AZURE_OPENAI_ENDPOINT",
            "https://placeholder.openai.azure.com",
        )
        .env("AZURE_OPENAI_API_KEY", "placeholder")
        .env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "placeholder")
        .env("AZURE_OPENAI_CHAT_DEPLOYMENT", "placeholder")
        .env("QDRANT_URL", qdrant_url)
        .env("QDRANT_COLLECTION", collection)
        .env("EMBEDDING_PROVIDER", "hash")
        .env("EMBEDDING_DIMENSIONS", HASH_SIZE.to_string());
    command
}

// Ingests tests/fixtures/pdfs with the real binary: two readable manuals (3 chunks in
// all) and one PDF without text, which is skipped.
fn ingest_fixtures(folder: &TempDir, collection: &str) {
    machinery(folder, qdrant_url(), collection)
        .args(["ingest", "--input-dir"])
        .arg(fixtures_dir())
        .assert()
        .success();
}

// The text of one chunk of the collection, read straight from Qdrant.
async fn a_stored_text(collection: &str) -> String {
    let response = client()
        .scroll(
            ScrollPointsBuilder::new(collection)
                .limit(1)
                .with_payload(true),
        )
        .await
        .expect("scroll the collection");
    response.result[0]
        .payload
        .get("text")
        .and_then(|value| value.as_str())
        .expect("the point has a text")
        .to_string()
}

// ---- Qdrant is down (no Docker needed) ----

// Nothing listens on port 1. The advice is the first thing printed, stdout stays empty
// (the client's own "Failed to obtain server version" line used to land there) and the
// exit code is 1.
#[test]
fn query_with_qdrant_down_says_so_and_prints_nothing_to_stdout() {
    let folder = tempfile::tempdir().unwrap();

    machinery(&folder, "http://127.0.0.1:1", "docs")
        .args(["query", "how do I drain the boiler?"])
        .assert()
        .code(1)
        .stdout("")
        .stderr(
            predicate::str::starts_with("Error: Cannot reach Qdrant at http://127.0.0.1:1")
                .and(predicate::str::contains("Is Docker running?")),
        );
}

// ingest asks before it reads a PDF or pays for an embedding, so there is no summary.
#[test]
fn ingest_with_qdrant_down_says_so_before_doing_any_work() {
    let folder = tempfile::tempdir().unwrap();

    machinery(&folder, "http://127.0.0.1:1", "docs")
        .args(["ingest", "--input-dir"])
        .arg(fixtures_dir())
        .assert()
        .code(1)
        .stdout("")
        .stderr(
            predicate::str::starts_with("Error: Cannot reach Qdrant at http://127.0.0.1:1")
                .and(predicate::str::contains("Is Docker running?")),
        );
}

// ---- against the throwaway Qdrant ----

// The stored text is the question, so it has the vector of the chunk it came from (the
// hash provider gives equal texts equal vectors) and that chunk comes first with a
// score of 1. The output is a header naming the provider, and then the results with
// nothing in between (the question is not echoed back).
#[tokio::test]
async fn query_prints_the_stored_chunk_first_under_a_header_naming_the_provider() {
    let folder = tempfile::tempdir().unwrap();
    ingest_fixtures(&folder, "cli_exact");
    let text = a_stored_text("cli_exact").await;
    let first_line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .expect("a chunk has text");

    let assert = machinery(&folder, qdrant_url(), "cli_exact")
        .args(["query", text.as_str()])
        .assert()
        .success()
        .stderr("");

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let mut lines = stdout.lines().filter(|line| !line.is_empty());
    assert_eq!(lines.next(), Some("Provider: hash"));
    let header = lines.next().expect("a first result");
    assert!(
        header.starts_with("1. ") && header.ends_with(", score 1.00"),
        "got {header:?}"
    );
    let indented = format!("   {first_line}");
    assert_eq!(lines.next(), Some(indented.as_str()));
}

// --top-k reaches the search: with 3 passages stored, asking for 1 prints one.
#[tokio::test]
async fn query_prints_no_more_passages_than_top_k_asks_for() {
    let folder = tempfile::tempdir().unwrap();
    ingest_fixtures(&folder, "cli_top_k");

    machinery(&folder, qdrant_url(), "cli_top_k")
        .args(["query", "how do I drain the boiler?", "--top-k", "1"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\n1. ").and(predicate::str::contains("\n2. ").not()));
}

// The advice has to be the headline of the error, not the last line of a "Caused by" list.
#[test]
fn a_collection_that_was_never_ingested_is_explained_in_the_first_line_of_the_error() {
    let folder = tempfile::tempdir().unwrap();

    machinery(&folder, qdrant_url(), "cli_never_ingested")
        .args(["query", "how do I drain the boiler?"])
        .assert()
        .code(1)
        .stdout("")
        .stderr(predicate::str::starts_with(
            "Error: Collection \"cli_never_ingested\" does not exist. Run ingest first.",
        ));
}

// Nothing found is a normal answer: exit code 0 and a sentence, not an error.
#[tokio::test]
async fn an_empty_collection_is_a_normal_answer() {
    let folder = tempfile::tempdir().unwrap();
    QdrantStore::connect(qdrant_url(), None)
        .unwrap()
        .ensure_collection("cli_empty", u64::from(HASH_SIZE), false)
        .await
        .unwrap();

    machinery(&folder, qdrant_url(), "cli_empty")
        .args(["query", "how do I drain the boiler?"])
        .assert()
        .success()
        .stdout(
            predicate::str::starts_with("Provider: hash")
                .and(predicate::str::contains("No matching passages found.")),
        )
        .stderr("");
}
