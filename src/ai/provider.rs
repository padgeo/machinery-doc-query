pub trait EmbeddingProvider {
    fn embedded_dimensions(&self) -> usize;
    #[allow(async_fn_in_trait)]
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProviderError>;
}

#[derive(thiserror::Error, Debug)]
pub enum ProviderError {
    #[error("embedding request failed: {0}")]
    RequestFailed(String),
}
