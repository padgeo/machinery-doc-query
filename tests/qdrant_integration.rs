// Runs the vector store and the ingest pipeline against a real Qdrant in Docker.
// The tests share one throwaway container (see tests/common), so every test uses
// collections of its own, and nothing here touches the Qdrant of docker-compose.yml.
// Docker has to be running.

mod common;

use std::path::{Path, PathBuf};

use common::{client, qdrant_url};
use machinery_doc_query::{
    ai::{
        hash_provider::HashProvider,
        provider::{EmbeddingProvider, ProviderError},
    },
    chunk::{Chunk, ChunkOptions},
    ingest, query,
    vector_store::qdrant::{QdrantStore, QdrantStoreError},
};
use qdrant_client::{
    Qdrant,
    qdrant::{
        CountPointsBuilder, ScrollPointsBuilder, point_id::PointIdOptions, vector_output::Vector,
    },
};

// Size of the hand-made vectors in the tests below.
const SIZE: u64 = 4;
// Size of the vectors HashProvider makes when a test ingests the fixture PDFs.
const HASH_SIZE: u16 = 8;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pdfs")
}

fn chunk(file: &str, chunk_index: usize, text: &str) -> Chunk {
    Chunk {
        source_file: PathBuf::from(file),
        page_number: 1,
        chunk_index,
        text: text.to_string(),
    }
}

/// One point as it is stored in Qdrant.
#[derive(Debug, PartialEq)]
struct StoredPoint {
    id: String,
    source_file: String,
    page_number: i64,
    chunk_index: i64,
    text: String,
    vector: Vec<f32>,
}

/// Everything a collection holds, read straight from Qdrant (not through the code
/// under test) and sorted by point id.
async fn stored(client: &Qdrant, collection: &str) -> Vec<StoredPoint> {
    let response = client
        .scroll(
            ScrollPointsBuilder::new(collection)
                .limit(100)
                .with_payload(true)
                .with_vectors(true),
        )
        .await
        .expect("scroll the collection");

    let mut points: Vec<StoredPoint> = response
        .result
        .into_iter()
        .map(|point| {
            let id = match point.id.and_then(|id| id.point_id_options) {
                Some(PointIdOptions::Uuid(uuid)) => uuid,
                other => panic!("expected a UUID point id, got {other:?}"),
            };
            let text = |key: &str| {
                point
                    .payload
                    .get(key)
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| panic!("payload field {key} is missing"))
            };
            let number = |key: &str| {
                point
                    .payload
                    .get(key)
                    .and_then(|value| value.as_integer())
                    .unwrap_or_else(|| panic!("payload field {key} is missing"))
            };
            let vector = match point.vectors.and_then(|vectors| vectors.get_vector()) {
                Some(Vector::Dense(dense)) => dense.data,
                other => panic!("expected a dense vector, got {other:?}"),
            };
            StoredPoint {
                id,
                source_file: text("source_file"),
                page_number: number("page_number"),
                chunk_index: number("chunk_index"),
                text: text("text"),
                vector,
            }
        })
        .collect();
    points.sort_by(|a, b| a.id.cmp(&b.id));
    points
}

async fn count(client: &Qdrant, collection: &str) -> u64 {
    client
        .count(CountPointsBuilder::new(collection).exact(true))
        .await
        .expect("count the points")
        .result
        .expect("a count")
        .count
}

/// Ingests tests/fixtures/pdfs: two readable manuals (3 chunks in all) and one PDF
/// without text, which is skipped.
async fn ingest_fixtures(
    store: &QdrantStore,
    collection: &str,
    provider: &HashProvider,
    batch_size: usize,
) {
    ingest::run(
        &fixtures_dir(),
        store,
        collection,
        provider,
        batch_size,
        &ChunkOptions::default(),
        false,
    )
    .await
    .expect("ingest the fixtures");
}

// An embedding provider that is down: every call fails.
struct FailingProvider;

impl EmbeddingProvider for FailingProvider {
    fn embedded_dimensions(&self) -> u16 {
        SIZE as u16
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, ProviderError> {
        Err(ProviderError::RequestFailed("provider is down".to_string()))
    }
}

// ---- the collection ----

#[tokio::test]
async fn ensure_collection_creates_the_collection_and_can_be_repeated() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();

    store
        .ensure_collection("repeated", SIZE, false)
        .await
        .unwrap();
    store
        .ensure_collection("repeated", SIZE, false)
        .await
        .unwrap();

    assert!(client().collection_exists("repeated").await.unwrap());
}

#[tokio::test]
async fn ensure_collection_refuses_a_different_vector_size() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    store
        .ensure_collection("size_checked", SIZE, false)
        .await
        .unwrap();

    let err = store
        .ensure_collection("size_checked", 8, false)
        .await
        .unwrap_err();

    assert!(
        matches!(
            err,
            QdrantStoreError::VectorSizeMismatch {
                expected: 8,
                actual: 4
            }
        ),
        "got {err:?}"
    );
}

#[tokio::test]
async fn ensure_collection_with_recreate_drops_what_was_stored() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    store
        .ensure_collection("recreated", SIZE, false)
        .await
        .unwrap();
    store
        .upsert_chunks(
            "recreated",
            &[chunk("a.pdf", 0, "text")],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .await
        .unwrap();
    assert_eq!(count(&client(), "recreated").await, 1);

    store
        .ensure_collection("recreated", SIZE, true)
        .await
        .unwrap();

    assert_eq!(count(&client(), "recreated").await, 0);
}

// ---- storing and finding chunks ----

#[tokio::test]
async fn upserted_chunks_keep_their_payload_and_their_vector() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    store
        .ensure_collection("round_trip", SIZE, false)
        .await
        .unwrap();
    let chunks = [
        chunk("boiler.pdf", 0, "Open the drain valve."),
        chunk("boiler.pdf", 1, "WARNING: scalding water."),
    ];
    let embeddings = [vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

    store
        .upsert_chunks("round_trip", &chunks, &embeddings)
        .await
        .unwrap();

    let points = stored(&client(), "round_trip").await;
    assert_eq!(points.len(), 2);
    // each chunk is found again with its own text and its own vector
    for (chunk, embedding) in chunks.iter().zip(&embeddings) {
        let point = points
            .iter()
            .find(|p| p.text == chunk.text)
            .expect("the chunk was stored");
        assert_eq!(point.source_file, "boiler.pdf");
        assert_eq!(point.page_number, 1);
        assert_eq!(point.chunk_index, chunk.chunk_index as i64);
        assert_eq!(&point.vector, embedding);
    }
}

// Point ids come from the file name and the chunk index, so ingesting a chunk again
// replaces it. This is what keeps a second ingest from duplicating everything.
#[tokio::test]
async fn upserting_the_same_chunk_again_replaces_it_in_place() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    store
        .ensure_collection("replaced", SIZE, false)
        .await
        .unwrap();
    store
        .upsert_chunks(
            "replaced",
            &[chunk("a.pdf", 0, "old text")],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .await
        .unwrap();

    store
        .upsert_chunks(
            "replaced",
            &[chunk("a.pdf", 0, "new text")],
            &[vec![0.0, 1.0, 0.0, 0.0]],
        )
        .await
        .unwrap();

    let points = stored(&client(), "replaced").await;
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].text, "new text");
    assert_eq!(points[0].vector, vec![0.0, 1.0, 0.0, 0.0]);
}

#[tokio::test]
async fn search_returns_at_most_top_k_results() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    store.ensure_collection("top_k", SIZE, false).await.unwrap();
    let chunks = [
        chunk("a.pdf", 0, "one"),
        chunk("a.pdf", 1, "two"),
        chunk("a.pdf", 2, "three"),
    ];
    let embeddings = [
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    store
        .upsert_chunks("top_k", &chunks, &embeddings)
        .await
        .unwrap();
    let question = vec![1.0, 0.0, 0.0, 0.0];

    let two = store.search("top_k", question.clone(), 2).await.unwrap();
    let all = store.search("top_k", question, 10).await.unwrap();

    assert_eq!(two.len(), 2);
    assert_eq!(all.len(), 3);
}

// search fails loudly: a missing collection or a vector of the wrong size must never
// look like "no matching passages". The query command checks both before it searches,
// so this is what is left if the collection changes in between. The wording is
// Qdrant's own and nothing here depends on it.
#[tokio::test]
async fn searching_a_missing_collection_is_an_error() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();

    let err = store
        .search("no_such_collection", vec![1.0, 0.0, 0.0, 0.0], 3)
        .await
        .err()
        .expect("searching a missing collection fails");

    assert!(
        matches!(err, QdrantStoreError::QdrantProcessError(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn a_query_vector_of_the_wrong_size_is_an_error() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    store
        .ensure_collection("wrong_query_size", SIZE, false)
        .await
        .unwrap();

    let err = store
        .search("wrong_query_size", vec![1.0, 0.0, 0.0], 3)
        .await
        .err()
        .expect("a vector of the wrong size is refused");

    assert!(
        matches!(err, QdrantStoreError::QdrantProcessError(_)),
        "got {err:?}"
    );
}

// ---- the ingest pipeline on a real collection ----

// The batches used to be paired with the wrong chunks: with more than one batch,
// the first chunks were overwritten with the vectors of later batches and the rest
// were never stored. Ingesting one chunk at a time has to give exactly what one big
// batch gives.
#[tokio::test]
async fn ingesting_in_tiny_batches_stores_every_chunk_with_its_own_vector() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    let provider = HashProvider {
        dimensions: HASH_SIZE,
    };

    ingest_fixtures(&store, "one_batch", &provider, 96).await;
    ingest_fixtures(&store, "tiny_batches", &provider, 1).await;

    let client = client();
    let reference = stored(&client, "one_batch").await;
    let tiny = stored(&client, "tiny_batches").await;
    assert_eq!(reference.len(), 3, "the two readable manuals hold 3 chunks");
    assert_eq!(tiny, reference);
}

#[tokio::test]
async fn ingesting_twice_does_not_duplicate_points() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    let provider = HashProvider {
        dimensions: HASH_SIZE,
    };

    ingest_fixtures(&store, "ingested_twice", &provider, 2).await;
    let first = stored(&client(), "ingested_twice").await;
    ingest_fixtures(&store, "ingested_twice", &provider, 2).await;
    let second = stored(&client(), "ingested_twice").await;

    assert_eq!(first.len(), 3);
    assert_eq!(second, first);
}

// ensure_collection with recreate deletes the old collection. Embedding can fail
// (bad key, outage), and that must happen before the old collection is touched.
#[tokio::test]
async fn a_failing_provider_leaves_an_existing_collection_alone_even_with_recreate() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    store.ensure_collection("kept", SIZE, false).await.unwrap();
    store
        .upsert_chunks(
            "kept",
            &[chunk("old.pdf", 0, "what was already stored")],
            &[vec![1.0, 0.0, 0.0, 0.0]],
        )
        .await
        .unwrap();

    let err = ingest::run(
        &fixtures_dir(),
        &store,
        "kept",
        &FailingProvider,
        96,
        &ChunkOptions::default(),
        true,
    )
    .await
    .unwrap_err();

    let rendered = format!("{err:#}");
    assert!(rendered.contains("Embed failed"), "got: {rendered}");
    let points = stored(&client(), "kept").await;
    assert_eq!(points.len(), 1, "the old collection was dropped");
    assert_eq!(points[0].text, "what was already stored");
}

// ---- the query path on a real collection ----

// HashProvider gives equal texts equal vectors, so a question that is the exact text
// of a stored chunk has that chunk's vector, and the chunk has to come first with a
// similarity of 1. That runs the whole path: embed, check the collection, search and
// convert what comes back. top_k (3) and the vector size (HASH_SIZE) differ, so using
// one for the other cannot go unnoticed.
#[tokio::test]
async fn a_question_equal_to_a_stored_text_finds_that_chunk_first() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    let provider = HashProvider {
        dimensions: HASH_SIZE,
    };
    ingest_fixtures(&store, "query_round_trip", &provider, 96).await;
    let texts: Vec<String> = stored(&client(), "query_round_trip")
        .await
        .into_iter()
        .map(|point| point.text)
        .collect();
    assert_eq!(texts.len(), 3, "the two readable manuals hold 3 chunks");

    for text in &texts {
        let found = query::run(text, &store, "query_round_trip", &provider, 3)
            .await
            .unwrap();

        assert_eq!(found.len(), 3);
        assert_eq!(&found[0].chunk.text, text);
        assert!(found[0].score >= 0.999, "score was {}", found[0].score);
    }
}

// top_k is the number of passages the user asked for, and run() has to hand it on.
#[tokio::test]
async fn a_query_returns_as_many_passages_as_top_k_asks_for() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    let provider = HashProvider {
        dimensions: HASH_SIZE,
    };
    ingest_fixtures(&store, "query_top_k", &provider, 96).await;

    for top_k in 1..=3_u64 {
        let found = query::run(
            "how do I drain the boiler?",
            &store,
            "query_top_k",
            &provider,
            top_k,
        )
        .await
        .unwrap();

        assert_eq!(found.len() as u64, top_k, "top_k was {top_k}");
    }
}

// query only reads. A collection that was never ingested has to be reported, and
// asking about it must not create it, which ensure_collection would.
#[tokio::test]
async fn querying_a_collection_that_was_never_ingested_says_to_ingest_and_creates_nothing() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    let provider = HashProvider {
        dimensions: HASH_SIZE,
    };

    let err = query::run(
        "how do I drain the boiler?",
        &store,
        "no_such_manuals",
        &provider,
        3,
    )
    .await
    .err()
    .expect("a missing collection is an error");

    let message = err.to_string();
    assert!(
        message.contains("no_such_manuals")
            && message.contains("does not exist")
            && message.contains("ingest"),
        "got {err:#}"
    );
    assert!(
        !client().collection_exists("no_such_manuals").await.unwrap(),
        "the query created the collection"
    );
}

// The collection holds 4-number vectors and this provider makes 3: both sizes have to
// be named, the right way round.
#[tokio::test]
async fn querying_with_a_provider_of_another_size_names_both_sizes() {
    let store = QdrantStore::connect(qdrant_url(), None).unwrap();
    store
        .ensure_collection("holds_four", SIZE, false)
        .await
        .unwrap();
    let provider = HashProvider { dimensions: 3 };

    let err = query::run(
        "how do I drain the boiler?",
        &store,
        "holds_four",
        &provider,
        3,
    )
    .await
    .err()
    .expect("a provider of another size is refused");

    let message = err.to_string();
    assert!(
        message.contains("holds 4-number vectors") && message.contains("makes 3"),
        "got {err:#}"
    );
}
