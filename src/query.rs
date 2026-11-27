use anyhow::{Context, Result, anyhow, bail};

use crate::{
    ai::{
        prompt::build_prompt,
        provider::{ChatProvider, EmbeddingProvider},
        schema::{SCHEMA_NAME, answer_schema, parse_answer},
    },
    answer::{Answer, check_citations},
    chunk::RetrievedChunk,
    vector_store::qdrant::{QdrantStore, QdrantStoreError},
};

const NO_PASSAGES_RESPONSE: &str = "No passages were found to answer from.";

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

/// Asks the model for a procedure that answers `question` from `chunks`, then checks the
/// citations in its reply against those same chunks.
///
/// A model that says it cannot answer gives `Answer::Unable`, which is not an error. An
/// error is a request that failed (it was refused, cut off, filtered, got no answer or
/// could not be sent) or a reply that does not match the schema, and that error shows how
/// the reply starts. With no chunks there is nothing to answer from, so the model is not
/// asked at all.
pub async fn answer_from_chunks<C: ChatProvider>(
    question: &str,
    chunks: &[RetrievedChunk],
    chat: &C,
) -> Result<Answer> {
    if chunks.is_empty() {
        return Ok(Answer::Unable {
            reason: NO_PASSAGES_RESPONSE.to_owned(),
        });
    }

    let prompt = build_prompt(question, chunks);
    let reply_string = chat
        .complete(&prompt, SCHEMA_NAME, &answer_schema())
        .await
        .context("Chat error")?;
    let parsed_reply = parse_answer(&reply_string)?;
    Ok(check_citations(parsed_reply, chunks))
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
    use std::{path::PathBuf, sync::Mutex};

    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        ai::{
            hash_provider::HashProvider,
            prompt::{Prompt, build_prompt},
            provider::ProviderError,
            schema::{SCHEMA_NAME, answer_schema},
        },
        answer::{AnswerStep, Source},
        chunk::Chunk,
    };

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

    // ---- answer_from_chunks ----

    // What a chat model was asked: the prompt and the schema that came with it.
    struct Asked {
        prompt: Prompt,
        schema_name: String,
        schema: Value,
    }

    // A chat model that gives the one reply it was prepared with and remembers each time it
    // was asked. A second question gets NoAnswer, so a request that should not happen shows.
    struct StubChat {
        reply: Mutex<Option<Result<String, ProviderError>>>,
        asked: Mutex<Vec<Asked>>,
    }

    impl ChatProvider for StubChat {
        async fn complete(
            &self,
            prompt: &Prompt,
            schema_name: &str,
            schema: &Value,
        ) -> Result<String, ProviderError> {
            self.asked.lock().unwrap().push(Asked {
                prompt: prompt.clone(),
                schema_name: schema_name.to_owned(),
                schema: schema.clone(),
            });
            self.reply
                .lock()
                .unwrap()
                .take()
                .unwrap_or(Err(ProviderError::NoAnswer))
        }
    }

    fn replies_with(raw: &str) -> StubChat {
        StubChat {
            reply: Mutex::new(Some(Ok(raw.to_owned()))),
            asked: Mutex::new(Vec::new()),
        }
    }

    fn fails_with(error: ProviderError) -> StubChat {
        StubChat {
            reply: Mutex::new(Some(Err(error))),
            asked: Mutex::new(Vec::new()),
        }
    }

    fn passage(file: &str, page: usize, text: &str) -> RetrievedChunk {
        RetrievedChunk {
            chunk: Chunk {
                source_file: PathBuf::from(file),
                page_number: page,
                chunk_index: 0,
                text: text.to_owned(),
            },
            score: 0.5,
        }
    }

    // Passage 1 is a.pdf page 1, passage 2 is b.pdf page 2, passage 3 is c.pdf page 3.
    fn three_passages() -> Vec<RetrievedChunk> {
        vec![
            passage("a.pdf", 1, "Open the valve."),
            passage("b.pdf", 2, "Wear gloves."),
            passage("c.pdf", 3, "Close the valve."),
        ]
    }

    fn source(file: &str, page: usize) -> Source {
        Source {
            file: PathBuf::from(file),
            page,
        }
    }

    fn step(instruction: &str, warnings: &[&str], source_indices: &[i64]) -> Value {
        json!({
            "instruction": instruction,
            "warnings": warnings,
            "source_indices": source_indices,
        })
    }

    // The reply of a model that follows the schema and gives these steps.
    fn procedure(steps: Value) -> String {
        json!({
            "summary": "Drain the boiler.",
            "unable_to_answer": false,
            "steps": steps,
        })
        .to_string()
    }

    // The model is sent what was retrieved, in the order its numbers count from, with the
    // schema it has to follow. One question is one request.
    #[tokio::test]
    async fn the_model_is_asked_once_with_the_prompt_and_the_schema() {
        let chunks = three_passages();
        let chat = replies_with(&procedure(json!([step("Do it.", &[], &[1])])));

        answer_from_chunks("how do I drain the boiler?", &chunks, &chat)
            .await
            .unwrap();

        let asked = chat.asked.lock().unwrap();
        assert_eq!(asked.len(), 1);
        assert_eq!(
            asked[0].prompt,
            build_prompt("how do I drain the boiler?", &chunks)
        );
        assert_eq!(asked[0].schema_name, SCHEMA_NAME);
        assert_eq!(asked[0].schema, answer_schema());
    }

    // The numbers in the answer count the passages in the prompt, so the sources have to be
    // the files and pages of those passages. Here the model cites passage 2, then 3 and 1.
    #[tokio::test]
    async fn the_sources_of_a_procedure_are_the_chunks_that_were_sent() {
        let chunks = three_passages();
        let reply = procedure(json!([
            step("Open the valve.", &["WARNING: hot."], &[2]),
            step("Close it.", &[], &[3, 1]),
        ]));

        let answer = answer_from_chunks("question", &chunks, &replies_with(&reply))
            .await
            .unwrap();

        assert_eq!(
            answer,
            Answer::Steps {
                summary: "Drain the boiler.".to_string(),
                steps: vec![
                    AnswerStep {
                        instruction: "Open the valve.".to_string(),
                        warnings: vec!["WARNING: hot.".to_string()],
                        sources: vec![1],
                    },
                    AnswerStep {
                        instruction: "Close it.".to_string(),
                        warnings: vec![],
                        sources: vec![2, 3],
                    },
                ],
                sources: vec![source("b.pdf", 2), source("c.pdf", 3), source("a.pdf", 1)],
            }
        );
    }

    // A number that points at no passage is the model's mistake. It is dropped, and its step
    // stays without a source: the answer is not turned into an error.
    #[tokio::test]
    async fn a_number_the_model_made_up_is_dropped_without_failing_the_answer() {
        let chunks = three_passages();
        let reply = procedure(json!([
            step("Open the valve.", &[], &[1, 9]),
            step("Close it.", &[], &[9]),
        ]));

        let answer = answer_from_chunks("question", &chunks, &replies_with(&reply))
            .await
            .unwrap();

        match answer {
            Answer::Steps { steps, sources, .. } => {
                assert_eq!(sources, vec![source("a.pdf", 1)]);
                assert_eq!(steps[0].sources, vec![1]);
                assert!(steps[1].sources.is_empty());
            }
            other => panic!("expected a procedure, got {other:?}"),
        }
    }

    // Saying that it cannot answer is an answer, not a failure: the reader is told why.
    #[tokio::test]
    async fn a_model_that_cannot_answer_gives_an_unable_answer_and_not_an_error() {
        let reply = json!({
            "summary": "The passages do not cover the pump.",
            "unable_to_answer": true,
            "steps": [],
        })
        .to_string();

        let answer = answer_from_chunks("question", &three_passages(), &replies_with(&reply))
            .await
            .unwrap();

        assert_eq!(
            answer,
            Answer::Unable {
                reason: "The passages do not cover the pump.".to_string()
            }
        );
    }

    // No steps and no word about it is still no procedure. What the model wrote instead is
    // prose that nothing was cited for, so it is not shown as the reason.
    #[tokio::test]
    async fn a_reply_with_no_steps_is_an_unable_answer_too() {
        let reply = json!({
            "summary": "Just open the valve.",
            "unable_to_answer": false,
            "steps": [],
        })
        .to_string();

        let answer = answer_from_chunks("question", &three_passages(), &replies_with(&reply))
            .await
            .unwrap();

        match answer {
            Answer::Unable { reason } => {
                assert!(!reason.trim().is_empty());
                assert!(!reason.contains("open the valve"), "got: {reason}");
            }
            other => panic!("expected an unable answer, got {other:?}"),
        }
    }

    // Every way a request can fail is an error that keeps the provider's own words. A
    // refusal is not an "unable", and a cut-off answer must not be read as a short one.
    // The request is not repeated here: the provider already retries what is worth it.
    #[rstest]
    #[case::refused(ProviderError::Refused("I cannot help with that.".to_string()))]
    #[case::cut_off(ProviderError::Truncated)]
    #[case::filtered(ProviderError::Filtered("violence".to_string()))]
    #[case::no_answer(ProviderError::NoAnswer)]
    #[case::request_failed(ProviderError::RequestFailed("connection reset".to_string()))]
    #[tokio::test]
    async fn a_failed_request_is_an_error_that_keeps_the_providers_message(
        #[case] error: ProviderError,
    ) {
        let expected = error.to_string();
        let chat = fails_with(error);

        let err = answer_from_chunks("question", &three_passages(), &chat)
            .await
            .unwrap_err();

        let chain = format!("{err:#}");
        assert!(chain.contains(&expected), "got {chain}");
        assert_eq!(chat.asked.lock().unwrap().len(), 1);
    }

    // A reply that is not what the schema asks for is an error, not an unable answer, and
    // the error shows how the reply starts, so that it can be seen what the model wrote.
    #[rstest]
    #[case::not_json("I'm sorry, but I cannot help with that.")]
    #[case::wrong_type(r#"{"summary": 5, "unable_to_answer": false, "steps": []}"#)]
    #[case::missing_field(r#"{"summary": "Drain it.", "steps": []}"#)]
    #[case::unknown_field(
        r#"{"summary": "x", "unable_to_answer": false, "steps": [], "extra": 1}"#
    )]
    #[case::cut_off(r#"{"summary": "Drain the boi"#)]
    #[tokio::test]
    async fn a_reply_that_does_not_match_the_schema_is_an_error_that_shows_its_start(
        #[case] raw: &str,
    ) {
        let err = answer_from_chunks("question", &three_passages(), &replies_with(raw))
            .await
            .unwrap_err();

        let chain = format!("{err:#}");
        let start: String = raw.chars().take(12).collect();
        assert!(chain.contains(&start), "got {chain}");
    }

    // The reply can be pages long, and only the start of it is quoted.
    #[tokio::test]
    async fn a_long_malformed_reply_is_cut_in_the_error() {
        let raw = format!("Sorry. {}", "blah ".repeat(2_000));

        let err = answer_from_chunks("question", &three_passages(), &replies_with(&raw))
            .await
            .unwrap_err();

        let chain = format!("{err:#}");
        assert!(chain.contains("Sorry."), "got {chain}");
        assert!(chain.len() < 1_000, "{} characters: {chain}", chain.len());
    }

    // With nothing retrieved there is nothing to answer from. Asking anyway would cost a
    // request and invite an answer from general knowledge, so the model is not asked.
    #[tokio::test]
    async fn with_no_chunks_the_model_is_not_asked() {
        let chat = replies_with(&procedure(json!([step("Do it.", &[], &[1])])));

        let answer = answer_from_chunks("question", &[], &chat).await.unwrap();

        assert!(chat.asked.lock().unwrap().is_empty());
        match answer {
            Answer::Unable { reason } => assert!(!reason.trim().is_empty()),
            other => panic!("expected an unable answer, got {other:?}"),
        }
    }

    // One passage is enough to answer from: only an empty list means there is nothing.
    #[tokio::test]
    async fn a_single_chunk_is_enough_to_ask_the_model() {
        let chunks = vec![passage("a.pdf", 1, "Open the valve.")];
        let reply = procedure(json!([step("Open the valve.", &[], &[1])]));

        let answer = answer_from_chunks("question", &chunks, &replies_with(&reply))
            .await
            .unwrap();

        match answer {
            Answer::Steps { sources, .. } => assert_eq!(sources, vec![source("a.pdf", 1)]),
            other => panic!("expected a procedure, got {other:?}"),
        }
    }
}
