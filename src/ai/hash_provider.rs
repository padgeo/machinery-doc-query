use std::hash::{DefaultHasher, Hash, Hasher};

use crate::ai::provider::{EmbeddingProvider, ProviderError};

pub struct HashProvider {
    pub dimensions: usize,
}

impl HashProvider {
    fn embed_one(text: &str, dimensions: usize) -> Vec<f32> {
        let mut vector: Vec<f32> = (0..dimensions)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                text.hash(&mut hasher);
                i.hash(&mut hasher);
                let bits = hasher.finish();
                (bits as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0 // -> roughly [-1.0, 1.0]
            })
            .collect();
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            vector.iter_mut().for_each(|x| *x /= magnitude);
        }
        vector
    }
}

impl EmbeddingProvider for HashProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ProviderError> {
        let embedded: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| HashProvider::embed_one(t, self.dimensions))
            .collect();
        Ok(embedded)
    }

    fn embedded_dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(dimensions: usize) -> HashProvider {
        HashProvider { dimensions }
    }

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[tokio::test]
    async fn returns_one_vector_per_text_with_the_requested_dimensions() {
        let vectors = provider(8).embed(&texts(&["a", "b", "c"])).await.unwrap();
        assert_eq!(vectors.len(), 3);
        assert!(vectors.iter().all(|v| v.len() == 8));
    }

    #[test]
    fn reports_its_configured_dimensions() {
        assert_eq!(provider(1536).embedded_dimensions(), 1536);
    }

    #[tokio::test]
    async fn same_text_always_gives_the_same_vector() {
        let p = provider(16);
        let first = p.embed(&texts(&["WARNING: hot surface"])).await.unwrap();
        let second = p.embed(&texts(&["WARNING: hot surface"])).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn different_texts_give_different_vectors() {
        let vectors = provider(16)
            .embed(&texts(&["drain the boiler", "replace the gasket"]))
            .await
            .unwrap();
        assert_ne!(vectors[0], vectors[1]);
    }

    #[tokio::test]
    async fn vectors_are_unit_length() {
        let vectors = provider(64).embed(&texts(&["one", "two"])).await.unwrap();
        for v in vectors {
            let magnitude = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((magnitude - 1.0).abs() < 1e-4, "magnitude was {magnitude}");
        }
    }

    // The EmbeddingProvider contract: embed(texts)[i] is the embedding of texts[i].
    // upsert_chunks pairs chunks with embeddings purely by position.
    #[tokio::test]
    async fn output_order_matches_input_order() {
        let p = provider(8);
        let batch = p.embed(&texts(&["first", "second"])).await.unwrap();
        let first_alone = p.embed(&texts(&["first"])).await.unwrap();
        let second_alone = p.embed(&texts(&["second"])).await.unwrap();
        assert_eq!(batch[0], first_alone[0]);
        assert_eq!(batch[1], second_alone[0]);
    }

    #[tokio::test]
    async fn empty_input_gives_empty_output() {
        assert!(provider(8).embed(&[]).await.unwrap().is_empty());
    }
}
