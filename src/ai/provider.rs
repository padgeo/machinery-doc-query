use serde_json::Value;

use crate::ai::{azure_provider::AzureOpenAiProvider, hash_provider::HashProvider, prompt::Prompt};

pub trait EmbeddingProvider {
    fn embedded_dimensions(&self) -> u16;
    #[allow(async_fn_in_trait)]
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProviderError>;
}

pub trait ChatProvider {
    /// Asks the model for a reply that follows `schema` (named `schema_name`) and returns
    /// the reply text exactly as the model wrote it. Reading it is up to the caller.
    #[allow(async_fn_in_trait)]
    async fn complete(
        &self,
        prompt: &Prompt,
        schema_name: &str,
        schema: &Value,
    ) -> Result<String, ProviderError>;
}

#[derive(thiserror::Error, Debug)]
pub enum ProviderError {
    #[error("request to the model failed: {0}")]
    RequestFailed(String),
    #[error("count mismatch, expected {expected} but found {got}")]
    CountMismatch { expected: usize, got: usize },
    #[error("dimension mismatch, expected {expected} but found {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("the model refused to answer: {0}")]
    Refused(String),
    #[error("the answer was cut off before it was complete")]
    Truncated,
    #[error("Azure's content filter blocked the request or the answer: {0}")]
    Filtered(String),
    #[error("the model returned no answer")]
    NoAnswer,
}

pub enum Provider {
    Hash(HashProvider),
    // Boxed because of clippy wanring: clippy::large_enum_variant
    Azure(Box<AzureOpenAiProvider>),
}

impl EmbeddingProvider for Provider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProviderError> {
        match self {
            Provider::Hash(hash) => hash.embed(texts).await,
            Provider::Azure(azure) => azure.embed(texts).await,
        }
    }

    fn embedded_dimensions(&self) -> u16 {
        match self {
            Provider::Hash(hash) => hash.embedded_dimensions(),
            Provider::Azure(azure) => azure.embedded_dimensions(),
        }
    }
}
