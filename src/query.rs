use anyhow::{Context, Result, anyhow, bail};

use crate::{
    ai::provider::EmbeddingProvider,
    chunk::RetrievedChunk,
    vector_store::qdrant::{QdrantStore, QdrantStoreError},
};

pub async fn embed_question<P: EmbeddingProvider>(
    provider: &P,
    question: &str,
) -> Result<Vec<f32>> {
    if question.trim().is_empty() {
        bail!("Question was empty")
    }

    let mut vectors = provider
        .embed(&[question.to_owned()])
        .await
        .context("Embed failed")?;

    if vectors.len() != 1 {
        bail!("Embed result should have len 1, but was {}", vectors.len());
    }
    let vector = vectors.remove(0);
    let expected = usize::from(provider.embedded_dimensions());
    if vector.len() != expected {
        bail!(
            "the question's vector has {} numbers, expected {expected}",
            vector.len()
        );
    }
    Ok(vector)
}

pub async fn run<P: EmbeddingProvider>(
    question: &str,
    store: &QdrantStore,
    collection: &str,
    provider: &P,
    top_k: u64,
) -> Result<Vec<RetrievedChunk>> {
    if top_k == 0 {
        bail!("top_k must be larger than 0")
    }

    let embedded_question = embed_question(provider, question).await?;

    store
        .verify_collection(collection, u64::from(provider.embedded_dimensions()))
        .await
        .map_err(|err| explain_collection_error(err, collection))?;

    store
        .search(collection, embedded_question, top_k)
        .await
        .with_context(|| format!("Failed to search collection {}", collection))
}

fn explain_collection_error(err: QdrantStoreError, collection: &str) -> anyhow::Error {
    match err {
        QdrantStoreError::CollectionDoesntExist(_) => {
            anyhow!("Collection \"{collection}\" does not exist. Run ingest first.")
        }
        // expected is the size this provider makes, actual is the size the collection holds.
        QdrantStoreError::VectorSizeMismatch { expected, actual } => anyhow!(
            "The collection holds {actual}-number vectors but this provider makes {expected}. Use the provider that ingested the documents."
        ),
        other => {
            anyhow::Error::new(other).context(format!("Failed to check collection {collection}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::ai::{hash_provider::HashProvider, provider::ProviderError};

    // A provider that always gives the same answer, so each test decides what comes
    // back: vectors, or the message of a request that failed. A failed request also
    // names the texts it was asked to embed, so a test can see what reached the provider.
    struct StubProvider {
        dimensions: u16,
        answer: Result<Vec<Vec<f32>>, String>,
    }

    impl EmbeddingProvider for StubProvider {
        fn embedded_dimensions(&self) -> u16 {
            self.dimensions
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProviderError> {
            self.answer.clone().map_err(|message| {
                ProviderError::RequestFailed(format!("{message}, asked to embed {texts:?}"))
            })
        }
    }

    fn answering(dimensions: u16, vectors: Vec<Vec<f32>>) -> StubProvider {
        StubProvider {
            dimensions,
            answer: Ok(vectors),
        }
    }

    fn failing(message: &str) -> StubProvider {
        StubProvider {
            dimensions: 4,
            answer: Err(message.to_owned()),
        }
    }

    // Nothing listens on port 1, so any call to Qdrant (the collection check, the
    // search) fails with a connection error. An error from anywhere else shows that
    // Qdrant was never contacted.
    fn unreachable_store() -> QdrantStore {
        QdrantStore::connect("http://127.0.0.1:1", None).unwrap()
    }

    // ---- embed_question ----

    // The provider is a failing one: if it were asked, the error would be its own.
    #[rstest]
    #[case::empty("")]
    #[case::spaces("   ")]
    #[case::tabs_and_newlines("\n\t")]
    #[tokio::test]
    async fn a_blank_question_is_refused_before_the_provider_is_asked(#[case] question: &str) {
        let err = embed_question(&failing("provider was asked"), question)
            .await
            .unwrap_err();

        assert!(err.to_string().contains("empty"), "got {err:#}");
    }

    #[tokio::test]
    async fn a_good_answer_is_returned_as_it_is() {
        let provider = HashProvider { dimensions: 8 };
        let question = "how do I drain the boiler?";

        let vector = embed_question(&provider, question).await.unwrap();

        // HashProvider gives the same text the same vector, so this is the vector of
        // the question itself.
        let direct = provider
            .embed(&[question.to_owned()])
            .await
            .unwrap()
            .remove(0);
        assert_eq!(vector, direct);
    }

    // With no vector, remove(0) would panic. With two, it would quietly keep the first.
    #[rstest]
    #[case::none(0)]
    #[case::two(2)]
    #[tokio::test]
    async fn the_wrong_number_of_vectors_is_refused(#[case] count: usize) {
        let provider = answering(4, vec![vec![0.0; 4]; count]);

        let err = embed_question(&provider, "question").await.unwrap_err();

        assert!(
            err.to_string().contains(&format!("len 1, but was {count}")),
            "got {err:#}"
        );
    }

    // A vector of the wrong size would be refused by Qdrant with a far less helpful message.
    #[rstest]
    #[case::too_short(3)]
    #[case::too_long(5)]
    #[case::empty(0)]
    #[tokio::test]
    async fn a_vector_of_the_wrong_size_is_refused(#[case] size: usize) {
        let provider = answering(4, vec![vec![0.0; size]]);

        let err = embed_question(&provider, "question").await.unwrap_err();

        assert!(
            err.to_string()
                .contains(&format!("{size} numbers, expected 4")),
            "got {err:#}"
        );
    }

    // ---- run ----

    // Asking for no results can never work. The provider is a failing one: if it were
    // asked, the error would be its own instead of this one.
    #[tokio::test]
    async fn zero_results_are_refused_before_the_provider_is_asked() {
        let err = run(
            "question",
            &unreachable_store(),
            "boiler_docs",
            &failing("provider was asked"),
            0,
        )
        .await
        .err()
        .unwrap();

        assert!(err.to_string().contains("top_k"), "got {err:#}");
    }

    // The error keeps the provider's own message, and Qdrant is not contacted: the
    // collection check would have failed with an error that names the collection.
    #[tokio::test]
    async fn a_failing_provider_stops_the_query_before_qdrant_is_contacted() {
        let err = run(
            "question",
            &unreachable_store(),
            "boiler_docs",
            &failing("provider is down"),
            5,
        )
        .await
        .err()
        .unwrap();

        let chain = format!("{err:#}");
        assert!(
            chain.contains("Embed failed") && chain.contains("provider is down"),
            "got {chain}"
        );
        assert!(!chain.contains("boiler_docs"), "got {chain}");
    }

    // run() takes two texts, the question and the collection name, and nothing but the
    // types stops one being used for the other. The failing provider reports the text
    // it was asked to embed, and that has to be the question.
    #[tokio::test]
    async fn the_question_is_what_gets_embedded() {
        let err = run(
            "how do I drain the boiler?",
            &unreachable_store(),
            "boiler_docs",
            &failing("provider is down"),
            5,
        )
        .await
        .err()
        .unwrap();

        let chain = format!("{err:#}");
        assert!(chain.contains("how do I drain the boiler?"), "got {chain}");
        assert!(!chain.contains("boiler_docs"), "got {chain}");
    }

    // With Qdrant down, the collection check is the first call that fails. The error
    // has to name the collection and must not claim that it is missing, because
    // running ingest would not help.
    #[tokio::test]
    async fn qdrant_being_down_is_not_mistaken_for_a_missing_collection() {
        let err = run(
            "question",
            &unreachable_store(),
            "boiler_docs",
            &answering(4, vec![vec![0.0; 4]]),
            5,
        )
        .await
        .err()
        .unwrap();

        let chain = format!("{err:#}");
        assert!(chain.contains("boiler_docs"), "got {chain}");
        assert!(!chain.contains("does not exist"), "got {chain}");
    }

    // ---- explain_collection_error ----

    #[test]
    fn a_missing_collection_is_explained() {
        let err = explain_collection_error(
            QdrantStoreError::CollectionDoesntExist("boiler_docs".to_owned()),
            "boiler_docs",
        );

        assert_eq!(
            err.to_string(),
            "Collection \"boiler_docs\" does not exist. Run ingest first."
        );
    }

    // expected is the size this provider makes, actual is the size the collection holds.
    #[test]
    fn a_size_mismatch_names_both_sizes_the_right_way_round() {
        let err = explain_collection_error(
            QdrantStoreError::VectorSizeMismatch {
                expected: 3,
                actual: 4,
            },
            "boiler_docs",
        );

        assert_eq!(
            err.to_string(),
            "The collection holds 4-number vectors but this provider makes 3. Use the provider that ingested the documents."
        );
    }

    // Anything else is not advice, but it still has to say what was being done and
    // keep the cause.
    #[test]
    fn any_other_error_names_the_collection_and_keeps_its_cause() {
        let err = explain_collection_error(
            QdrantStoreError::InputError {
                msg: "the cause".to_owned(),
            },
            "boiler_docs",
        );

        let chain = format!("{err:#}");
        assert!(chain.contains("boiler_docs"), "got {chain}");
        assert!(chain.contains("the cause"), "got {chain}");
    }
}
