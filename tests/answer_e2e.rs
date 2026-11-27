// Runs the real binary the way a user does, with Azure OpenAI as the provider. A mock server
// plays Azure: it makes the embeddings with the hash provider (equal texts get equal vectors,
// so a question that is the text of a stored chunk finds that chunk first) and answers the
// chat requests with whatever a test has prepared. Qdrant is the throwaway container of
// tests/common, so each test uses collections of its own. These tests need Docker: run them
// in the foreground, one test run at a time.

mod common;

use std::{
    future::Future,
    path::{Path, PathBuf},
    task::{Context, Poll, Waker},
};

use assert_cmd::Command;
use common::{client, qdrant_url};
use machinery_doc_query::{
    ai::{hash_provider::HashProvider, provider::EmbeddingProvider},
    chunk::{PAGE_KEY, SOURCE_KEY, TEXT_KEY},
    vector_store::qdrant::QdrantStore,
};
use predicates::prelude::*;
use qdrant_client::qdrant::ScrollPointsBuilder;
use serde_json::{Value, json};
use tempfile::TempDir;
use wiremock::{
    Mock, MockServer, Request, Respond, ResponseTemplate,
    matchers::{header, method, path, query_param},
};

// Size of the vectors the hash provider makes in these tests.
const HASH_SIZE: u16 = 8;
const API_KEY: &str = "test-key";
const API_VERSION: &str = "2024-08-01-preview";
const EMBEDDINGS_PATH: &str = "/openai/deployments/emb-e2e/embeddings";
const CHAT_PATH: &str = "/openai/deployments/chat-e2e/chat/completions";

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pdfs")
}

// ---- the mock Azure ----

// The vectors of the hash provider. It never waits for anything, so one poll is enough, and
// a mock cannot block on a future anyway.
fn embed_now(texts: &[String]) -> Vec<Vec<f32>> {
    let provider = HashProvider {
        dimensions: HASH_SIZE,
    };
    let mut embedding = std::pin::pin!(provider.embed(texts));
    match embedding
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(vectors) => vectors.expect("the hash provider cannot fail"),
        Poll::Pending => unreachable!("the hash provider does not wait for anything"),
    }
}

// The embeddings API: one vector per input text, from the hash provider.
struct HashEmbeddings;

impl Respond for HashEmbeddings {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().expect("the request body is JSON");
        let texts: Vec<String> = match &body["input"] {
            Value::String(text) => vec![text.clone()],
            Value::Array(texts) => texts
                .iter()
                .map(|text| text.as_str().expect("an input is a text").to_owned())
                .collect(),
            other => panic!("unexpected input: {other}"),
        };
        let data: Vec<Value> = embed_now(&texts)
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| {
                json!({"object": "embedding", "index": index, "embedding": embedding})
            })
            .collect();
        ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "model": "text-embedding-3-small",
            "data": data,
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        }))
    }
}

// A chat completion with every field the client insists on. `message` is the part that varies.
fn completion(message: Value) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "chatcmpl-e2e",
        "object": "chat.completion",
        "created": 1_700_000_000_u32,
        "model": "gpt-4o-2024-08-06",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }))
}

// The model follows the schema: its `content` is the answer, as JSON text.
fn model_says(answer: &Value) -> ResponseTemplate {
    completion(json!({"role": "assistant", "content": answer.to_string()}))
}

fn model_refuses(why: &str) -> ResponseTemplate {
    completion(json!({"role": "assistant", "content": null, "refusal": why}))
}

fn azure_rejects(status: u16, code: &str, message: &str) -> ResponseTemplate {
    ResponseTemplate::new(status)
        .set_body_json(json!({"error": {"code": code, "message": message}}))
}

// The mock Azure: the embeddings come from the hash provider and every chat request gets
// `chat` as its answer. `chat_requests` is how many of them the test expects. It is checked
// when the server is dropped at the end of the test, so 0 means "never asked".
async fn azure(chat: ResponseTemplate, chat_requests: u64) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(EMBEDDINGS_PATH))
        .and(query_param("api-version", API_VERSION))
        .and(header("api-key", API_KEY))
        .respond_with(HashEmbeddings)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .and(query_param("api-version", API_VERSION))
        .and(header("api-key", API_KEY))
        .respond_with(chat)
        .expect(chat_requests)
        .mount(&server)
        .await;
    server
}

// ---- the binary, and what is stored ----

// The binary with exactly the settings given here. It runs in an empty folder with an empty
// environment, so neither a .env file nor the variables of whoever runs the tests can change
// what it does. Azure is the provider, and the mock is Azure.
fn machinery(folder: &TempDir, azure: &MockServer, collection: &str) -> Command {
    let mut command = assert_cmd::cargo::cargo_bin_cmd!("machinery-doc-query");
    command
        .current_dir(folder.path())
        .env_clear()
        .env("AZURE_OPENAI_ENDPOINT", azure.uri())
        .env("AZURE_OPENAI_API_KEY", API_KEY)
        .env("AZURE_OPENAI_API_VERSION", API_VERSION)
        .env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "emb-e2e")
        .env("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-e2e")
        .env("QDRANT_URL", qdrant_url())
        .env("QDRANT_COLLECTION", collection)
        .env("EMBEDDING_PROVIDER", "azure")
        .env("EMBEDDING_DIMENSIONS", HASH_SIZE.to_string());
    command
}

// Ingests tests/fixtures/pdfs through the mock: two readable manuals (3 chunks in all) and
// one PDF without text, which is skipped.
fn ingest_fixtures(folder: &TempDir, azure: &MockServer, collection: &str) {
    machinery(folder, azure, collection)
        .args(["ingest", "--input-dir"])
        .arg(fixtures_dir())
        .assert()
        .success();
}

// One chunk of the collection, read straight from Qdrant: where it is from, and its text.
struct StoredChunk {
    file: String,
    page: i64,
    text: String,
}

async fn a_stored_chunk(collection: &str) -> StoredChunk {
    let response = client()
        .scroll(
            ScrollPointsBuilder::new(collection)
                .limit(1)
                .with_payload(true),
        )
        .await
        .expect("scroll the collection");
    let payload = &response.result[0].payload;
    StoredChunk {
        file: payload
            .get(SOURCE_KEY)
            .and_then(|value| value.as_str())
            .expect("the point has a file")
            .to_string(),
        page: payload
            .get(PAGE_KEY)
            .and_then(|value| value.as_integer())
            .expect("the point has a page"),
        text: payload
            .get(TEXT_KEY)
            .and_then(|value| value.as_str())
            .expect("the point has a text")
            .to_string(),
    }
}

// What the model says in the tests that need a procedure. Passage 1 is the chunk the question
// is the text of, because it has the same vector and so comes first.
fn drain_answer() -> Value {
    json!({
        "summary": "Drain the boiler.",
        "unable_to_answer": false,
        "steps": [
            {
                "instruction": "Open the drain valve.",
                "warnings": ["WARNING: scalding water."],
                "source_indices": [1],
            },
            {
                "instruction": "Close the drain valve.",
                "warnings": [],
                "source_indices": [1],
            },
        ],
    })
}

// ---- an answer ----

// The PRD's success criterion: one output holds the step, its warning and the file the step
// comes from. The answer is all of stdout (no header, no note), and stderr stays empty.
#[tokio::test]
async fn an_answer_shows_the_step_its_warning_and_the_source_file_together() {
    let folder = tempfile::tempdir().unwrap();
    let azure = azure(model_says(&drain_answer()), 1).await;
    ingest_fixtures(&folder, &azure, "answer_happy");
    let stored = a_stored_chunk("answer_happy").await;

    machinery(&folder, &azure, "answer_happy")
        .args(["query", stored.text.as_str()])
        .assert()
        .success()
        .stdout(
            predicate::str::starts_with("Drain the boiler.")
                .and(predicate::str::contains("Open the drain valve."))
                .and(predicate::str::contains("WARNING: scalding water."))
                .and(predicate::str::contains(format!(
                    "Sources:\n[1] {}, page {}",
                    stored.file, stored.page
                ))),
        )
        .stderr("");
}

// Nothing is sent to the model that was not retrieved, and everything that was retrieved is
// sent: the question's own chunk first (it is passage 1, the one the model cites), then the
// other two. The reply is asked for in the shape of the schema.
#[tokio::test]
async fn the_model_is_sent_every_retrieved_passage_and_the_schema() {
    let folder = tempfile::tempdir().unwrap();
    let azure = azure(model_says(&drain_answer()), 1).await;
    ingest_fixtures(&folder, &azure, "answer_request");
    let stored = a_stored_chunk("answer_request").await;

    machinery(&folder, &azure, "answer_request")
        .args(["query", stored.text.as_str()])
        .assert()
        .success();

    let requests = azure.received_requests().await.unwrap();
    let chat: Vec<&Request> = requests
        .iter()
        .filter(|request| request.url.path() == CHAT_PATH)
        .collect();
    assert_eq!(chat.len(), 1);
    let body: Value = chat[0].body_json().unwrap();
    let user = body["messages"][1]["content"].as_str().unwrap();
    assert!(
        user.contains(&format!("[1] {}, page {}", stored.file, stored.page)),
        "got: {user}"
    );
    assert!(
        user.contains("[2] ") && user.contains("[3] "),
        "got: {user}"
    );
    assert!(!user.contains("[4] "), "got: {user}");
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(
        body["response_format"]["json_schema"]["name"],
        "procedure_answer"
    );
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    // Reasoning models reject these two.
    assert!(body.get("temperature").is_none() && body.get("max_tokens").is_none());
}

// Saying that it cannot answer is an answer: the reason is printed and the exit code is 0.
#[tokio::test]
async fn a_model_that_cannot_answer_exits_0_with_the_reason() {
    let folder = tempfile::tempdir().unwrap();
    let reply = json!({
        "summary": "The passages do not cover the pump.",
        "unable_to_answer": true,
        "steps": [],
    });
    let azure = azure(model_says(&reply), 1).await;
    ingest_fixtures(&folder, &azure, "answer_unable");
    let stored = a_stored_chunk("answer_unable").await;

    machinery(&folder, &azure, "answer_unable")
        .args(["query", stored.text.as_str()])
        .assert()
        .success()
        .stdout("Unable to answer: The passages do not cover the pump.\n")
        .stderr("");
}

// ---- a request that fails ----

// A refusal is not an answer: nothing on stdout, the refusal on stderr and exit code 1.
#[tokio::test]
async fn a_refusal_exits_1_and_says_so() {
    let folder = tempfile::tempdir().unwrap();
    let azure = azure(model_refuses("I can't help with that."), 1).await;
    ingest_fixtures(&folder, &azure, "answer_refused");
    let stored = a_stored_chunk("answer_refused").await;

    machinery(&folder, &azure, "answer_refused")
        .args(["query", stored.text.as_str()])
        .assert()
        .code(1)
        .stdout("")
        .stderr(
            predicate::str::starts_with("Error: ")
                .and(predicate::str::contains("the model refused to answer"))
                .and(predicate::str::contains("I can't help with that.")),
        );
}

// A wrong key is Azure's answer to the chat request, and the error keeps what Azure said, which
// is also the hint: its own text names the key and the endpoint. Asking again would not help, so
// the request is made once.
#[tokio::test]
async fn a_rejected_key_exits_1_and_keeps_what_azure_said() {
    let folder = tempfile::tempdir().unwrap();
    let rejected = azure_rejects(
        401,
        "401",
        "Access denied due to invalid subscription key or wrong API endpoint.",
    );
    let azure = azure(rejected, 1).await;
    ingest_fixtures(&folder, &azure, "answer_401");
    let stored = a_stored_chunk("answer_401").await;

    machinery(&folder, &azure, "answer_401")
        .args(["query", stored.text.as_str()])
        .assert()
        .code(1)
        .stdout("")
        .stderr(
            predicate::str::starts_with("Error: ")
                .and(predicate::str::contains("Access denied"))
                .and(predicate::str::contains("401")),
        );
}

// ---- no request at all ----

// --retrieval-only is for debugging and for saving the chat request: it is never made.
#[tokio::test]
async fn retrieval_only_prints_the_passages_and_never_asks_the_model() {
    let folder = tempfile::tempdir().unwrap();
    let azure = azure(model_says(&drain_answer()), 0).await;
    ingest_fixtures(&folder, &azure, "answer_retrieval_only");
    let stored = a_stored_chunk("answer_retrieval_only").await;

    machinery(&folder, &azure, "answer_retrieval_only")
        .args(["query", stored.text.as_str(), "--retrieval-only"])
        .assert()
        .success()
        .stdout(predicate::str::starts_with("Provider: azure\n1. "))
        .stderr("");
}

// Nothing found is a normal result, and there is nothing to ask the model about.
#[tokio::test]
async fn an_empty_collection_is_an_unable_answer_and_the_model_is_never_asked() {
    let folder = tempfile::tempdir().unwrap();
    let azure = azure(model_says(&drain_answer()), 0).await;
    QdrantStore::connect(qdrant_url(), None)
        .unwrap()
        .ensure_collection("answer_empty", u64::from(HASH_SIZE), false)
        .await
        .unwrap();

    machinery(&folder, &azure, "answer_empty")
        .args(["query", "how do I drain the boiler?"])
        .assert()
        .success()
        .stdout(predicate::str::starts_with("Unable to answer: "))
        .stderr("");
}
