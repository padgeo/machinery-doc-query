use crate::ai::{azure_provider::AzureOpenAiProvider, hash_provider::HashProvider};

pub trait EmbeddingProvider {
    fn embedded_dimensions(&self) -> u16;
    #[allow(async_fn_in_trait)]
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProviderError>;
}

#[derive(thiserror::Error, Debug)]
pub enum ProviderError {
    #[error("embedding request failed: {0}")]
    RequestFailed(String),
    #[error("count mismatch, expected {expected} but found {got}")]
    CountMismatch { expected: usize, got: usize },
    #[error("dimension mismatch, expected {expected} but found {got}")]
    DimensionMismatch { expected: usize, got: usize },
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
