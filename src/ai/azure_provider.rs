use std::time::Duration;

use async_openai::{
    Client,
    config::AzureConfig,
    error::OpenAIError::{self, Reqwest},
    types::embeddings::{CreateEmbeddingRequestArgs, CreateEmbeddingResponse, EmbeddingInput},
};
use thiserror::Error;
use tokio::time::{error::Elapsed, timeout};

use crate::{
    ai::{
        provider::{EmbeddingProvider, ProviderError},
        retry::{RetryPolicy, retry},
    },
    config::AppConfig,
};

/// How long one embedding request may take before it counts as timed out.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_mins(2);

#[derive(Debug, Error)]
pub enum RequestError {
    #[error("Request timed out: {0}")]
    Timeout(Elapsed),
    #[error("Request failed: {0}")]
    Function(OpenAIError),
}

pub struct AzureEmbeddingSettings {
    endpoint: String,
    api_key: String,
    api_version: String,
    deployment: String,
}

impl From<&AppConfig> for AzureEmbeddingSettings {
    fn from(value: &AppConfig) -> Self {
        Self {
            endpoint: value.azure_openai_endpoint.clone(),
            api_key: value.azure_openai_api_key.clone(),
            api_version: value.azure_openai_api_version.clone(),
            deployment: value.azure_openai_embedding_deployment.clone(),
        }
    }
}

pub struct AzureOpenAiProvider {
    settings: AzureEmbeddingSettings,
    dimensions: u16,
    batch_size: u16,
    client: Client<AzureConfig>,
    retry_policy: RetryPolicy,
    request_timeout: Duration,
}

impl AzureOpenAiProvider {
    pub fn new(settings: AzureEmbeddingSettings, dimensions: u16, batch_size: u16) -> Self {
        Self {
            client: Self::connect_client(
                &settings.endpoint.clone(),
                &settings.api_key.clone(),
                &settings.api_version.clone(),
                &settings.deployment.clone(),
            ),
            settings,
            dimensions,
            batch_size: batch_size.max(1), // place lower limit to prevent panic
            retry_policy: RetryPolicy::default(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }

    /// Replaces how often, and how long between attempts, a timed out or dropped request
    /// is tried again.
    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Replaces how long one request may take before it counts as timed out.
    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }

    pub fn connect_client(
        endpoint: &str,
        api_key: &str,
        api_version: &str,
        deployment: &str,
    ) -> Client<AzureConfig> {
        Client::with_config(
            AzureConfig::new()
                .with_api_base(endpoint.trim_end_matches('/'))
                .with_api_key(api_key)
                .with_api_version(api_version)
                .with_deployment_id(deployment),
        )
    }

    pub async fn embed_batch(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>, ProviderError> {
        let response = retry(
            &self.retry_policy,
            AzureOpenAiProvider::is_transient,
            || self.attempt_request(batch),
        )
        .await
        .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        tracing::debug!(tokens = response.usage.total_tokens, batch = batch.len());
        Self::into_ordered_vectors(response, batch.len(), self.dimensions.into())
    }

    async fn attempt_request(
        &self,
        batch: &[&str],
    ) -> Result<CreateEmbeddingResponse, RequestError> {
        timeout(self.request_timeout, self.request_once(batch))
            .await
            .map_err(RequestError::Timeout)?
            .map_err(RequestError::Function)
    }

    /// Decides whether `retry` should try a failed request again.
    ///
    /// async-openai already retries 429, 5xx and connect errors itself: 3 retries
    /// (4 requests in total) with exponential backoff, honouring `Retry-After`.
    /// Retrying those here too would multiply the requests, so this only covers
    /// what the client leaves alone:
    /// - timeouts, because the client has none (ours is in `attempt_request`)
    /// - other transport errors, e.g. the server hanging up without answering or
    ///   cutting the response short
    ///
    /// Anything else (400, 401, 403, 404, malformed JSON) would fail the same way
    /// again, so it is reported straight away.
    fn is_transient(e: &RequestError) -> bool {
        match e {
            RequestError::Timeout(_) => true,
            RequestError::Function(Reqwest(e)) => !e.is_connect(),
            _ => false,
        }
    }

    async fn request_once(&self, batch: &[&str]) -> Result<CreateEmbeddingResponse, OpenAIError> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.settings.deployment)
            .input(EmbeddingInput::StringArray(
                batch.iter().copied().map(String::from).collect(),
            ))
            .build()?;
        self.client.embeddings().create(request).await
    }

    fn into_ordered_vectors(
        mut response: CreateEmbeddingResponse,
        expected: usize,
        dims: usize,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        let data = &mut response.data;
        if data.len() != expected {
            return Err(ProviderError::CountMismatch {
                expected,
                got: response.data.len(),
            });
        }

        data.sort_by_key(|e| e.index);
        if data
            .iter()
            .enumerate()
            .any(|(i, e)| usize::try_from(e.index).ok() != Some(i))
        {
            return Err(ProviderError::RequestFailed(
                "embedding indexes are not 0..n".to_string(),
            ));
        }
        if let Some(mismatch) = data.iter().find(|e| e.embedding.len() != dims) {
            return Err(ProviderError::DimensionMismatch {
                expected: dims,
                got: mismatch.embedding.len(),
            });
        }

        Ok(response.data.into_iter().map(|e| e.embedding).collect())
    }
}

impl EmbeddingProvider for AzureOpenAiProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ProviderError> {
        if texts.is_empty() {
            let empty: Vec<Vec<f32>> = vec![];
            return Ok(empty);
        }
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size.into()) {
            vectors.extend(self.embed_batch(batch).await?);
        }
        debug_assert_eq!(vectors.len(), texts.len());
        Ok(vectors)
    }

    fn embedded_dimensions(&self) -> u16 {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use async_openai::types::embeddings::{Embedding, EmbeddingUsage};
    use rstest::rstest;
    use serde_json::{Value, json};
    use std::time::Instant;
    use url::Url;
    use wiremock::{
        Mock, MockServer, Request, Respond, ResponseTemplate,
        matchers::{body_partial_json, header, method, path, query_param},
    };

    use super::*;

    const API_KEY: &str = "test-key";
    const API_VERSION: &str = "2024-08-01-preview";
    const DEPLOYMENT: &str = "emb-prod";
    const EMBEDDINGS_PATH: &str = "/openai/deployments/emb-prod/embeddings";

    fn provider_at(endpoint: String, dimensions: u16, batch_size: u16) -> AzureOpenAiProvider {
        AzureOpenAiProvider::new(
            AzureEmbeddingSettings {
                endpoint,
                api_key: API_KEY.to_string(),
                api_version: API_VERSION.to_string(),
                deployment: DEPLOYMENT.to_string(),
            },
            dimensions,
            batch_size,
        )
    }

    // "text-0", "text-1", ... The number in the text lets EchoEmbeddings tag
    // each vector, so a test can tell which text a vector belongs to.
    fn texts(count: usize) -> Vec<String> {
        (0..count).map(|i| format!("text-{i}")).collect()
    }

    fn strs(input: &[String]) -> Vec<&str> {
        input.iter().map(|s| s.as_str()).collect()
    }

    fn app_config() -> AppConfig {
        AppConfig {
            azure_openai_endpoint: "https://my-resource.openai.azure.com".to_string(),
            azure_openai_api_key: "secret-key".to_string(),
            azure_openai_api_version: "2024-08-01-preview".to_string(),
            azure_openai_embedding_deployment: "embeddings-prod".to_string(),
            azure_openai_chat_deployment: "chat-prod".to_string(),
            qdrant_url: Url::parse("http://localhost:6334").unwrap(),
            qdrant_api_key: None,
            qdrant_collection: "machinery_docs".to_string(),
            embedding_dimensions: 1536,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_attempts: 5,
        }
    }

    // A response as the crate would deserialize it, built from (index, vector) pairs.
    fn response(data: Vec<(u32, Vec<f32>)>) -> CreateEmbeddingResponse {
        CreateEmbeddingResponse {
            object: "list".to_string(),
            model: "text-embedding-3-small".to_string(),
            data: data
                .into_iter()
                .map(|(index, embedding)| Embedding {
                    index,
                    object: "embedding".to_string(),
                    embedding,
                })
                .collect(),
            usage: EmbeddingUsage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
        }
    }

    // ---- AzureEmbeddingSettings ----

    #[test]
    fn settings_use_the_embedding_deployment_not_the_chat_one() {
        let settings = AzureEmbeddingSettings::from(&app_config());

        assert_eq!(settings.endpoint, "https://my-resource.openai.azure.com");
        assert_eq!(settings.api_key, "secret-key");
        assert_eq!(settings.api_version, "2024-08-01-preview");
        assert_eq!(settings.deployment, "embeddings-prod");
    }

    #[test]
    fn embedded_dimensions_is_the_configured_size() {
        let provider = provider_at("https://my-resource.openai.azure.com".to_string(), 1536, 96);

        assert_eq!(provider.embedded_dimensions(), 1536);
    }

    // ---- into_ordered_vectors (pure: no server needed) ----

    #[test]
    fn a_response_already_in_order_passes_through() {
        let data = vec![(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])];

        let vectors = AzureOpenAiProvider::into_ordered_vectors(response(data), 2, 2).unwrap();

        assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    }

    #[test]
    fn a_shuffled_response_is_put_back_in_index_order() {
        let data = vec![
            (2, vec![2.0, 0.0]),
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
        ];

        let vectors = AzureOpenAiProvider::into_ordered_vectors(response(data), 3, 2).unwrap();

        assert_eq!(
            vectors,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]]
        );
    }

    #[test]
    fn a_fully_reversed_response_is_put_back_in_index_order() {
        let data: Vec<(u32, Vec<f32>)> = (0..20u16)
            .rev()
            .map(|i| (u32::from(i), vec![f32::from(i), 0.0]))
            .collect();

        let vectors = AzureOpenAiProvider::into_ordered_vectors(response(data), 20, 2).unwrap();

        let firsts: Vec<f32> = vectors.iter().map(|v| v[0]).collect();
        let expected: Vec<f32> = (0..20u16).map(f32::from).collect();
        assert_eq!(firsts, expected);
    }

    #[test]
    fn an_empty_response_for_an_empty_batch_is_fine() {
        let vectors = AzureOpenAiProvider::into_ordered_vectors(response(vec![]), 0, 4).unwrap();

        assert!(vectors.is_empty());
    }

    #[rstest]
    #[case::too_few(3, 2)]
    #[case::too_many(1, 2)]
    #[case::none_returned(2, 0)]
    fn a_wrong_number_of_vectors_is_a_count_mismatch(#[case] expected: usize, #[case] got: usize) {
        let data = (0..got)
            .map(|i| (u32::try_from(i).unwrap(), vec![0.0; 3]))
            .collect();

        let err =
            AzureOpenAiProvider::into_ordered_vectors(response(data), expected, 3).unwrap_err();

        assert!(
            matches!(err, ProviderError::CountMismatch { expected: e, got: g } if e == expected && g == got),
            "got {err:?}"
        );
    }

    #[test]
    fn a_vector_of_the_wrong_size_is_a_dimension_mismatch() {
        let data = vec![(0, vec![0.0; 3])];

        let err = AzureOpenAiProvider::into_ordered_vectors(response(data), 1, 4).unwrap_err();

        assert!(
            matches!(
                err,
                ProviderError::DimensionMismatch {
                    expected: 4,
                    got: 3
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_wrong_sized_vector_is_found_even_when_it_is_not_the_first() {
        let data = vec![(0, vec![0.0; 4]), (1, vec![0.0; 4]), (2, vec![0.0; 3])];

        let err = AzureOpenAiProvider::into_ordered_vectors(response(data), 3, 4).unwrap_err();

        assert!(
            matches!(
                err,
                ProviderError::DimensionMismatch {
                    expected: 4,
                    got: 3
                }
            ),
            "got {err:?}"
        );
    }

    // Indexes come from the network. Bad ones must be an error the user can
    // read, never a panic (and never silently mislabelled chunks, because the
    // caller zips vectors with chunks by position).
    #[rstest]
    #[case::duplicate(vec![0, 0])]
    #[case::gap(vec![0, 2])]
    #[case::not_starting_at_zero(vec![1, 2])]
    fn indexes_that_are_not_exactly_zero_to_n_are_an_error(#[case] indexes: Vec<u32>) {
        let data = indexes.iter().map(|&i| (i, vec![0.0; 3])).collect();

        let result = AzureOpenAiProvider::into_ordered_vectors(response(data), indexes.len(), 3);

        assert!(result.is_err(), "indexes {indexes:?} should be rejected");
    }

    // ---- embed / embed_batch against a mock Azure endpoint ----

    // Answers every embeddings request with one vector per input. Vector i of
    // a request is [n, 0, 0, ..] where n is the number in the input text
    // ("text-7" -> 7). With `shuffled`, the vectors come back in reverse order,
    // which the real API is allowed to do (that is what `index` is for).
    struct EchoEmbeddings {
        dimensions: usize,
        shuffled: bool,
    }

    impl Respond for EchoEmbeddings {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: Value = request.body_json().expect("the request body is JSON");
            let inputs = body["input"].as_array().expect("input is an array");

            let mut data: Vec<Value> = inputs
                .iter()
                .enumerate()
                .map(|(index, text)| {
                    let n: f32 = text
                        .as_str()
                        .unwrap()
                        .trim_start_matches("text-")
                        .parse()
                        .unwrap();
                    let mut embedding = vec![0.0_f32; self.dimensions];
                    embedding[0] = n;
                    json!({"object": "embedding", "index": index, "embedding": embedding})
                })
                .collect();
            if self.shuffled {
                data.reverse();
            }

            ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "model": "text-embedding-3-small",
                "data": data,
                "usage": {"prompt_tokens": 1, "total_tokens": 1}
            }))
        }
    }

    async fn echo_server(dimensions: usize, shuffled: bool) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .respond_with(EchoEmbeddings {
                dimensions,
                shuffled,
            })
            .mount(&server)
            .await;
        server
    }

    async fn input_sizes(server: &MockServer) -> Vec<usize> {
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|r| {
                r.body_json::<Value>().unwrap()["input"]
                    .as_array()
                    .unwrap()
                    .len()
            })
            .collect()
    }

    #[tokio::test]
    async fn embed_sends_one_request_in_the_azure_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .and(query_param("api-version", API_VERSION))
            .and(header("api-key", API_KEY))
            .and(body_partial_json(json!({"input": ["text-0", "text-1"]})))
            .respond_with(EchoEmbeddings {
                dimensions: 4,
                shuffled: false,
            })
            .expect(1)
            .mount(&server)
            .await;

        let vectors = provider_at(server.uri(), 4, 10)
            .embed(&strs(&texts(2)))
            .await
            .unwrap();

        assert_eq!(vectors.len(), 2);
    }

    #[tokio::test]
    async fn embed_splits_the_texts_into_batches() {
        let server = echo_server(4, false).await;

        provider_at(server.uri(), 4, 2)
            .embed(&strs(&texts(5)))
            .await
            .unwrap();

        assert_eq!(input_sizes(&server).await, vec![2, 2, 1]);
    }

    #[tokio::test]
    async fn embed_keeps_input_order_across_batches_when_the_server_shuffles() {
        let server = echo_server(4, true).await;

        let vectors = provider_at(server.uri(), 4, 2)
            .embed(&strs(&texts(5)))
            .await
            .unwrap();

        let firsts: Vec<f32> = vectors.iter().map(|v| v[0]).collect();
        assert_eq!(firsts, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[tokio::test]
    async fn embed_with_no_texts_sends_no_request() {
        let server = MockServer::start().await;

        let vectors = provider_at(server.uri(), 4, 10).embed(&[]).await.unwrap();

        assert!(vectors.is_empty());
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_batch_size_of_zero_is_treated_as_one() {
        let server = echo_server(4, false).await;

        let vectors = provider_at(server.uri(), 4, 0)
            .embed(&strs(&texts(2)))
            .await
            .unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!(input_sizes(&server).await, vec![1, 1]);
    }

    #[tokio::test]
    async fn a_trailing_slash_on_the_endpoint_does_not_double_the_slash() {
        let server = echo_server(4, false).await;

        provider_at(format!("{}/", server.uri()), 4, 10)
            .embed(&strs(&texts(1)))
            .await
            .unwrap();

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests[0].url.path(), EMBEDDINGS_PATH);
    }

    #[tokio::test]
    async fn an_http_error_becomes_request_failed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {"code": "401", "message": "Access denied: invalid subscription key"}
            })))
            .mount(&server)
            .await;

        let err = provider_at(server.uri(), 4, 10)
            .embed(&strs(&texts(1)))
            .await
            .unwrap_err();

        assert!(
            matches!(&err, ProviderError::RequestFailed(msg) if msg.contains("invalid subscription key")),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn vectors_of_the_wrong_size_from_the_server_are_rejected() {
        // the server answers with 4 numbers per vector, the provider expects 8
        let server = echo_server(4, false).await;

        let err = provider_at(server.uri(), 8, 10)
            .embed(&strs(&texts(2)))
            .await
            .unwrap_err();

        assert!(
            matches!(
                err,
                ProviderError::DimensionMismatch {
                    expected: 8,
                    got: 4
                }
            ),
            "got {err:?}"
        );
    }

    // embed must not hand back a partial result: one bad batch fails the call,
    // and no further batches are sent after it.
    #[tokio::test]
    async fn a_bad_response_in_a_later_batch_fails_the_whole_call() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .respond_with(EchoEmbeddings {
                dimensions: 4,
                shuffled: false,
            })
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [],
                "usage": {"prompt_tokens": 0, "total_tokens": 0}
            })))
            .mount(&server)
            .await;

        let err = provider_at(server.uri(), 4, 2)
            .embed(&strs(&texts(6)))
            .await
            .unwrap_err();

        assert!(
            matches!(
                err,
                ProviderError::CountMismatch {
                    expected: 2,
                    got: 0
                }
            ),
            "got {err:?}"
        );
        // batch 1 (ok) and batch 2 (bad); batch 3 is never sent
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    // ---- retry and timeout wiring ----

    // Shrinks the waits between attempts of our own retry layer so these tests finish
    // in milliseconds: 3 attempts, 1 ms to 5 ms apart. That layer only retries
    // timeouts and transport errors; HTTP errors are retried by async-openai itself.
    // The per-request timeout is left at its generous default so a slow test machine
    // cannot trigger it by accident.
    fn fast(provider: AzureOpenAiProvider) -> AzureOpenAiProvider {
        provider.with_retry_policy(RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        })
    }

    fn api_error(status: u16) -> ResponseTemplate {
        ResponseTemplate::new(status).set_body_json(json!({
            "error": {"code": status.to_string(), "message": "synthetic failure"}
        }))
    }

    #[tokio::test]
    async fn a_429_is_retried_and_the_next_answer_is_used() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .respond_with(api_error(429))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .respond_with(EchoEmbeddings {
                dimensions: 4,
                shuffled: false,
            })
            .mount(&server)
            .await;

        let vectors = fast(provider_at(server.uri(), 4, 10))
            .embed(&strs(&texts(2)))
            .await
            .unwrap();

        assert_eq!(vectors.len(), 2);
        // the 429, then async-openai's own retry that worked
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    // A 401 means the key is wrong. Asking again cannot fix that.
    #[tokio::test]
    async fn a_401_is_sent_once_and_not_retried() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(api_error(401))
            .mount(&server)
            .await;

        let err = fast(provider_at(server.uri(), 4, 10))
            .embed(&strs(&texts(1)))
            .await
            .unwrap_err();

        assert!(
            matches!(err, ProviderError::RequestFailed(_)),
            "got {err:?}"
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    // async-openai retries a 5xx itself: the first request plus 3 retries. Our own
    // layer leaves HTTP errors alone (see is_transient), so the two do not multiply.
    #[tokio::test]
    async fn a_persistent_503_is_given_up_on_after_the_clients_own_retries() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(api_error(503))
            .mount(&server)
            .await;

        let err = fast(provider_at(server.uri(), 4, 10))
            .embed(&strs(&texts(1)))
            .await
            .unwrap_err();

        assert!(
            matches!(err, ProviderError::RequestFailed(_)),
            "got {err:?}"
        );
        // 1 request + 3 retries by async-openai; ours would have made it 12
        assert_eq!(server.received_requests().await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn a_slow_server_hits_the_timeout_and_is_retried() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(1)))
            .mount(&server)
            .await;
        let provider =
            fast(provider_at(server.uri(), 4, 10)).with_request_timeout(Duration::from_millis(150));

        let err = provider.embed(&strs(&texts(1))).await.unwrap_err();

        assert!(
            matches!(&err, ProviderError::RequestFailed(msg) if msg.contains("timed out")),
            "got {err:?}"
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 3);
    }

    // Nothing listens on port 1, so every try is refused straight away. async-openai
    // tries again 3 times, waiting 100 ms, 200 ms and 400 ms; our own layer leaves
    // connect errors to it (see is_transient). The waiting is the only way to see
    // that the retries happened: a single refused try takes a millisecond or two.
    #[tokio::test]
    async fn connection_refused_is_retried_by_the_client_then_reported() {
        let provider = fast(provider_at("http://127.0.0.1:1".to_string(), 4, 10));

        let started = Instant::now();
        let err = provider.embed(&strs(&texts(1))).await.unwrap_err();

        assert!(
            matches!(err, ProviderError::RequestFailed(_)),
            "got {err:?}"
        );
        // a sleep never returns early, and the client sleeps 700 ms in total
        assert!(
            started.elapsed() >= Duration::from_millis(500),
            "the client did not retry the refused connection"
        );
    }

    // Our retry covers one batch, not the whole run: when batch 2 times out and needs
    // a second try, batch 1 is not sent again.
    #[tokio::test]
    async fn a_retry_resends_only_the_failed_batch() {
        let server = MockServer::start().await;
        let echo = || EchoEmbeddings {
            dimensions: 4,
            shuffled: false,
        };
        // answered in the order mounted: ok (batch 1), too slow (batch 2), then ok
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .respond_with(echo())
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(2)))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(EMBEDDINGS_PATH))
            .respond_with(echo())
            .mount(&server)
            .await;
        let provider =
            fast(provider_at(server.uri(), 4, 2)).with_request_timeout(Duration::from_millis(300));

        let vectors = provider.embed(&strs(&texts(4))).await.unwrap();

        let firsts: Vec<f32> = vectors.iter().map(|v| v[0]).collect();
        assert_eq!(firsts, vec![0.0, 1.0, 2.0, 3.0]);
        // batch 1 once, batch 2 twice (the timed out try, then the retry)
        assert_eq!(input_sizes(&server).await, vec![2, 2, 2]);
    }
}
