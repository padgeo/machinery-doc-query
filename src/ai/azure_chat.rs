use std::time::Duration;

use async_openai::{
    Client,
    config::AzureConfig,
    error::OpenAIError::{self, ApiError, Reqwest},
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs,
        CreateChatCompletionResponse, FinishReason, ResponseFormat, ResponseFormatJsonSchema,
    },
};
use serde_json::Value;
use tokio::time::{error::Elapsed, timeout};

use crate::{
    ai::{
        prompt::Prompt,
        provider::{ChatProvider, ProviderError},
        retry::{RetryPolicy, retry},
    },
    config::AppConfig,
};

#[derive(Debug, thiserror::Error)]
enum RequestError {
    #[error("request timed out: {0}")]
    Timeout(Elapsed),
    #[error("{0}")]
    Api(OpenAIError),
}

pub struct AzureChatSettings {
    endpoint: String,
    api_key: String,
    api_version: String,
    deployment: String,
}

// TODO: Change when PR6 is merged
impl From<&AppConfig> for AzureChatSettings {
    fn from(value: &AppConfig) -> Self {
        Self {
            endpoint: value.azure_openai_endpoint.clone(),
            api_key: value.azure_openai_api_key.clone(),
            api_version: value.azure_openai_api_version.clone(),
            deployment: value.azure_openai_chat_deployment.clone(),
        }
    }
}

pub struct AzureChatProvider {
    client: Client<AzureConfig>,
    deployment: String,
    retry_policy: RetryPolicy,
    request_timeout: Duration,
}

impl AzureChatProvider {
    // TODO: Change when PR6 is merged
    pub fn new(settings: AzureChatSettings) -> Self {
        let client = Client::with_config(
            AzureConfig::new()
                .with_api_base(settings.endpoint.trim_end_matches('/'))
                .with_api_key(settings.api_key.as_str())
                .with_api_version(settings.api_version.as_str())
                .with_deployment_id(settings.deployment.as_str()),
        );
        Self {
            client,
            deployment: settings.deployment,
            retry_policy: RetryPolicy {
                max_attempts: 5,
                base_delay: Duration::from_millis(2),
                max_delay: Duration::from_secs(2),
            },
            request_timeout: Duration::from_mins(2),
        }
    }

    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    async fn request_once(
        &self,
        prompt: &Prompt,
        schema_name: &str,
        schema: &Value,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessage::from(prompt.system.as_str()).into(),
            ChatCompletionRequestUserMessage::from(prompt.user.as_str()).into(),
        ];
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.deployment)
            .messages(messages)
            .response_format(ResponseFormat::JsonSchema {
                json_schema: ResponseFormatJsonSchema {
                    description: None,
                    name: schema_name.to_owned(),
                    schema: schema.clone(),
                    strict: Some(true),
                },
            })
            .build()?;
        self.client.chat().create(request).await
    }

    fn reply_text(response: CreateChatCompletionResponse) -> Result<String, ProviderError> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or(ProviderError::NoAnswer)?;
        if let Some(refusal) = choice
            .message
            .refusal
            .filter(|text| !text.trim().is_empty())
        {
            return Err(ProviderError::Refused(refusal));
        }
        match choice.finish_reason {
            Some(FinishReason::Length) => return Err(ProviderError::Truncated),
            Some(FinishReason::ContentFilter) => {
                return Err(ProviderError::Filtered(
                    "the answer was blocked by the content filter".to_owned(),
                ));
            }
            _ => {}
        }
        match choice.message.content {
            Some(text) if !text.trim().is_empty() => Ok(text),
            _ => Err(ProviderError::NoAnswer),
        }
    }

    fn into_provider_error(err: RequestError) -> ProviderError {
        match err {
            RequestError::Api(ApiError(api))
                if api.api_error.code.as_deref() == Some("content_filter") =>
            {
                ProviderError::Filtered(api.api_error.message)
            }
            other => ProviderError::RequestFailed(other.to_string()),
        }
    }

    async fn attempt(
        &self,
        prompt: &Prompt,
        schema_name: &str,
        schema: &Value,
    ) -> Result<CreateChatCompletionResponse, RequestError> {
        timeout(
            self.request_timeout,
            self.request_once(prompt, schema_name, schema),
        )
        .await
        .map_err(RequestError::Timeout)?
        .map_err(RequestError::Api)
    }

    fn is_transient(e: &RequestError) -> bool {
        match e {
            RequestError::Timeout(_) => true,
            RequestError::Api(Reqwest(e)) => !e.is_connect(),
            _ => false,
        }
    }
}

impl ChatProvider for AzureChatProvider {
    async fn complete(
        &self,
        prompt: &Prompt,
        schema_name: &str,
        schema: &Value,
    ) -> Result<String, ProviderError> {
        let response = retry(&self.retry_policy, Self::is_transient, || {
            self.attempt(prompt, schema_name, schema)
        })
        .await
        .map_err(AzureChatProvider::into_provider_error)?;
        AzureChatProvider::reply_text(response)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rstest::rstest;
    use serde_json::json;
    use url::Url;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path, query_param},
    };

    use super::*;
    use crate::ai::schema::{SCHEMA_NAME, answer_schema};

    const API_KEY: &str = "test-key";
    const API_VERSION: &str = "2024-08-01-preview";
    const DEPLOYMENT: &str = "chat-prod";
    const CHAT_PATH: &str = "/openai/deployments/chat-prod/chat/completions";

    // What the model puts in `content`: JSON, as text. The provider hands it back untouched.
    const ANSWER: &str =
        r#"{"summary": "Drain the boiler.", "unable_to_answer": false, "steps": []}"#;

    fn settings_at(endpoint: String) -> AzureChatSettings {
        AzureChatSettings {
            endpoint,
            api_key: API_KEY.to_string(),
            api_version: API_VERSION.to_string(),
            deployment: DEPLOYMENT.to_string(),
        }
    }

    fn provider_at(endpoint: String) -> AzureChatProvider {
        AzureChatProvider::new(settings_at(endpoint))
    }

    // Shrinks the waits between attempts of our own retry layer: 3 attempts, 1 ms to 5 ms
    // apart. That layer only retries timeouts and transport errors. HTTP errors are
    // retried by async-openai itself. The request timeout stays at its generous default,
    // so a slow test machine cannot trip it by accident.
    fn fast(provider: AzureChatProvider) -> AzureChatProvider {
        provider.with_retry_policy(RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        })
    }

    fn prompt() -> Prompt {
        Prompt {
            system: "SYSTEM RULES".to_string(),
            user: "USER MESSAGE".to_string(),
        }
    }

    async fn ask(provider: &AzureChatProvider) -> Result<String, ProviderError> {
        provider
            .complete(&prompt(), SCHEMA_NAME, &answer_schema())
            .await
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
        }
    }

    // A chat completion with every field the client insists on. `message` and
    // `finish_reason` are the parts that vary. A missing `finish_reason` is allowed.
    fn completion(message: Value, finish_reason: Option<&str>) -> Value {
        let mut choice = json!({"index": 0, "message": message});
        if let Some(reason) = finish_reason {
            choice["finish_reason"] = json!(reason);
        }
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000_u32,
            "model": "gpt-4o-2024-08-06",
            "choices": [choice],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })
    }

    fn reply(message: Value, finish_reason: Option<&str>) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(completion(message, finish_reason))
    }

    fn answering(content: &str) -> ResponseTemplate {
        reply(
            json!({"role": "assistant", "content": content}),
            Some("stop"),
        )
    }

    fn api_error(status: u16, code: &str, message: &str) -> ResponseTemplate {
        ResponseTemplate::new(status)
            .set_body_json(json!({"error": {"code": code, "message": message}}))
    }

    async fn server_replying(template: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(CHAT_PATH))
            .respond_with(template)
            .mount(&server)
            .await;
        server
    }

    async fn sent_bodies(server: &MockServer) -> Vec<Value> {
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|request| request.body_json::<Value>().unwrap())
            .collect()
    }

    async fn request_count(server: &MockServer) -> usize {
        server.received_requests().await.unwrap().len()
    }

    // ---- the settings ----

    #[test]
    fn settings_use_the_chat_deployment_not_the_embedding_one() {
        let settings = AzureChatSettings::from(&app_config());

        assert_eq!(settings.endpoint, "https://my-resource.openai.azure.com");
        assert_eq!(settings.api_key, "secret-key");
        assert_eq!(settings.api_version, "2024-08-01-preview");
        assert_eq!(settings.deployment, "chat-prod");
    }

    // ---- what is sent ----

    #[tokio::test]
    async fn complete_sends_one_request_in_the_azure_chat_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(CHAT_PATH))
            .and(query_param("api-version", API_VERSION))
            .and(header("api-key", API_KEY))
            .respond_with(answering(ANSWER))
            .expect(1)
            .mount(&server)
            .await;

        let answer = ask(&provider_at(server.uri())).await.unwrap();

        assert_eq!(answer, ANSWER);
    }

    #[tokio::test]
    async fn the_request_carries_both_messages_and_the_strict_schema_and_nothing_else() {
        let server = server_replying(answering(ANSWER)).await;

        ask(&provider_at(server.uri())).await.unwrap();

        let bodies = sent_bodies(&server).await;
        assert_eq!(bodies.len(), 1);
        let body = &bodies[0];
        assert_eq!(
            body["messages"],
            json!([
                {"role": "system", "content": "SYSTEM RULES"},
                {"role": "user", "content": "USER MESSAGE"}
            ])
        );
        assert_eq!(
            body["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME, "schema": answer_schema(), "strict": true}
            })
        );
        // Reasoning models reject `temperature` and `max_tokens`, and the default API
        // version does not know `max_completion_tokens`. Send none of them, or anything
        // else nobody asked for.
        let sent: BTreeSet<&str> = body
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let allowed = BTreeSet::from(["messages", "model", "response_format"]);
        assert!(
            sent.is_subset(&allowed),
            "unexpected keys in the request: {:?}",
            sent.difference(&allowed).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn a_trailing_slash_on_the_endpoint_does_not_double_the_slash() {
        let server = server_replying(answering(ANSWER)).await;

        ask(&provider_at(format!("{}/", server.uri())))
            .await
            .unwrap();

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests[0].url.path(), CHAT_PATH);
    }

    // ---- what comes back ----

    #[tokio::test]
    async fn the_answer_text_comes_back_exactly_as_the_model_wrote_it() {
        let text = "  {\"summary\": \"Heat to 30\u{b0}C\",\n \"unable_to_answer\": false, \"steps\": []}\n";
        let server = server_replying(answering(text)).await;

        let answer = ask(&provider_at(server.uri())).await.unwrap();

        assert_eq!(answer, text);
    }

    #[tokio::test]
    async fn a_reply_without_a_finish_reason_still_returns_its_text() {
        let server =
            server_replying(reply(json!({"role": "assistant", "content": ANSWER}), None)).await;

        assert_eq!(ask(&provider_at(server.uri())).await.unwrap(), ANSWER);
    }

    // Azure adds fields of its own to every reply, and none of them may get in the way.
    #[tokio::test]
    async fn the_fields_azure_adds_to_a_reply_are_ignored() {
        let mut body = completion(
            json!({"role": "assistant", "content": ANSWER}),
            Some("stop"),
        );
        body["prompt_filter_results"] = json!([{
            "prompt_index": 0,
            "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}}
        }]);
        body["choices"][0]["content_filter_results"] =
            json!({"hate": {"filtered": false, "severity": "safe"}});
        let server = server_replying(ResponseTemplate::new(200).set_body_json(body)).await;

        assert_eq!(ask(&provider_at(server.uri())).await.unwrap(), ANSWER);
    }

    // ---- replies that are not an answer ----

    #[tokio::test]
    async fn a_refusal_becomes_a_refused_error_with_the_models_reason() {
        let refusal = "I'm sorry, I can't help with that.";
        let server = server_replying(reply(
            json!({"role": "assistant", "content": null, "refusal": refusal}),
            Some("stop"),
        ))
        .await;

        let err = ask(&provider_at(server.uri())).await.unwrap_err();

        assert!(
            matches!(&err, ProviderError::Refused(reason) if reason.contains("can't help")),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains(refusal),
            "the user should see why: {err}"
        );
    }

    // The text is there, but it stops half way: it is not an answer, so it is not returned.
    #[tokio::test]
    async fn a_cut_off_answer_is_a_truncated_error_even_though_it_has_text() {
        let server = server_replying(reply(
            json!({"role": "assistant", "content": "{\"summary\": \"Drain the bo"}),
            Some("length"),
        ))
        .await;

        let err = ask(&provider_at(server.uri())).await.unwrap_err();

        assert!(matches!(err, ProviderError::Truncated), "got {err:?}");
    }

    #[tokio::test]
    async fn an_answer_stopped_by_the_content_filter_is_a_filtered_error() {
        let server = server_replying(reply(
            json!({"role": "assistant", "content": null}),
            Some("content_filter"),
        ))
        .await;

        let err = ask(&provider_at(server.uri())).await.unwrap_err();

        assert!(matches!(err, ProviderError::Filtered(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn a_reply_with_no_choices_is_a_no_answer_error() {
        let mut body = completion(
            json!({"role": "assistant", "content": ANSWER}),
            Some("stop"),
        );
        body["choices"] = json!([]);
        let server = server_replying(ResponseTemplate::new(200).set_body_json(body)).await;

        let err = ask(&provider_at(server.uri())).await.unwrap_err();

        assert!(matches!(err, ProviderError::NoAnswer), "got {err:?}");
    }

    #[rstest]
    #[case::no_content(json!({"role": "assistant", "content": null}))]
    #[case::empty_content(json!({"role": "assistant", "content": ""}))]
    #[case::blank_content(json!({"role": "assistant", "content": "  \n "}))]
    #[tokio::test]
    async fn an_empty_answer_is_a_no_answer_error(#[case] message: Value) {
        let server = server_replying(reply(message, Some("stop"))).await;

        let err = ask(&provider_at(server.uri())).await.unwrap_err();

        assert!(matches!(err, ProviderError::NoAnswer), "got {err:?}");
    }

    // ---- errors from Azure ----

    // Azure answers HTTP 400 with the code `content_filter` when it blocks the prompt itself.
    #[tokio::test]
    async fn a_400_content_filter_on_the_prompt_is_a_filtered_error_and_is_sent_once() {
        let server = server_replying(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "message": "The response was filtered due to the prompt triggering Azure OpenAI's content management policy.",
                "type": null,
                "param": "prompt",
                "code": "content_filter",
                "status": 400,
                "innererror": {"code": "ResponsibleAIPolicyViolation"}
            }
        })))
        .await;

        let err = ask(&provider_at(server.uri())).await.unwrap_err();

        assert!(
            matches!(&err, ProviderError::Filtered(why) if why.contains("content management policy")),
            "got {err:?}"
        );
        assert_eq!(request_count(&server).await, 1);
    }

    // Only the code `content_filter` means a filter. Any other 400 is a mistake in the request.
    #[tokio::test]
    async fn another_400_is_a_failed_request_not_a_filtered_one() {
        let server = server_replying(api_error(
            400,
            "unsupported_parameter",
            "Unsupported parameter: 'temperature'",
        ))
        .await;

        let err = ask(&provider_at(server.uri())).await.unwrap_err();

        assert!(
            matches!(&err, ProviderError::RequestFailed(why) if why.contains("Unsupported parameter")),
            "got {err:?}"
        );
        assert_eq!(request_count(&server).await, 1);
    }

    // A 401 means the key is wrong. Asking again cannot fix that.
    #[tokio::test]
    async fn a_401_is_sent_once_and_reported_as_a_failed_request() {
        let server = server_replying(api_error(
            401,
            "401",
            "Access denied: invalid subscription key",
        ))
        .await;

        let err = ask(&fast(provider_at(server.uri()))).await.unwrap_err();

        assert!(
            matches!(&err, ProviderError::RequestFailed(why) if why.contains("invalid subscription key")),
            "got {err:?}"
        );
        assert_eq!(request_count(&server).await, 1);
    }

    #[tokio::test]
    async fn the_api_key_never_appears_in_an_error() {
        let server = server_replying(api_error(401, "401", "Access denied")).await;
        let settings = AzureChatSettings {
            api_key: "very-secret-key".to_string(),
            ..settings_at(server.uri())
        };

        let err = ask(&AzureChatProvider::new(settings)).await.unwrap_err();

        let shown = format!("{err} {err:?}");
        assert!(
            !shown.contains("very-secret-key"),
            "the key leaked: {shown}"
        );
    }

    // The same variant carries the embedding errors, and its message says "embedding
    // request failed". A failed chat call must not blame the embeddings.
    #[tokio::test]
    async fn a_chat_failure_does_not_call_itself_an_embedding_failure() {
        let server = server_replying(api_error(401, "401", "Access denied")).await;

        let err = ask(&provider_at(server.uri())).await.unwrap_err();

        assert!(
            !err.to_string().to_lowercase().contains("embedding"),
            "got: {err}"
        );
    }

    // ---- retries and timeouts ----
    //
    // async-openai retries 429, 5xx and connect errors itself: the first request plus 3
    // retries, with a growing wait and honouring `Retry-After`. Our own layer only
    // retries what the client leaves alone: timeouts and other transport errors.

    #[tokio::test]
    async fn a_429_is_retried_by_the_client_and_the_next_answer_is_used() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(CHAT_PATH))
            .respond_with(api_error(429, "429", "Rate limit exceeded"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(CHAT_PATH))
            .respond_with(answering(ANSWER))
            .mount(&server)
            .await;

        let answer = ask(&provider_at(server.uri())).await.unwrap();

        assert_eq!(answer, ANSWER);
        // the 429, then the client's own retry that worked
        assert_eq!(request_count(&server).await, 2);
    }

    #[tokio::test]
    async fn a_persistent_503_is_given_up_on_after_the_clients_own_retries() {
        let server = server_replying(api_error(503, "503", "Service unavailable")).await;

        let err = ask(&fast(provider_at(server.uri()))).await.unwrap_err();

        assert!(
            matches!(err, ProviderError::RequestFailed(_)),
            "got {err:?}"
        );
        // 1 request + 3 retries by async-openai. Retrying here too would make it 12.
        assert_eq!(request_count(&server).await, 4);
    }

    #[tokio::test]
    async fn a_slow_server_hits_the_timeout_and_is_retried() {
        let server =
            server_replying(ResponseTemplate::new(200).set_delay(Duration::from_secs(1))).await;
        let provider =
            fast(provider_at(server.uri())).with_request_timeout(Duration::from_millis(150));

        let err = ask(&provider).await.unwrap_err();

        assert!(
            matches!(&err, ProviderError::RequestFailed(why) if why.contains("timed out")),
            "got {err:?}"
        );
        // fast() allows 3 attempts, and each one timed out
        assert_eq!(request_count(&server).await, 3);
    }

    // The request is built again for every attempt. A retry must ask the same question.
    #[tokio::test]
    async fn a_timed_out_attempt_is_retried_with_the_same_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(CHAT_PATH))
            .respond_with(answering(ANSWER).set_delay(Duration::from_secs(2)))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(CHAT_PATH))
            .respond_with(answering(ANSWER))
            .mount(&server)
            .await;
        let provider =
            fast(provider_at(server.uri())).with_request_timeout(Duration::from_millis(300));

        let answer = ask(&provider).await.unwrap();

        assert_eq!(answer, ANSWER);
        let bodies = sent_bodies(&server).await;
        assert_eq!(bodies.len(), 2);
        assert_eq!(bodies[0], bodies[1]);
    }
}
