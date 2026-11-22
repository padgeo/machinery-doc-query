use qdrant_client::{
    Qdrant, QdrantError,
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, QueryPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder, vectors_config::Config,
    },
};
use uuid::Uuid;

use crate::chunk::{Chunk, RetrievedChunk, RetrievedChunkError};

// Arbitrary value, no real source
const NAMESPACE: Uuid = Uuid::from_u128(0x6f8f_2a10_9c3e_4b7a_8d2f_1a0e_5c6b_7d8e);

pub struct QdrantStore {
    client: Qdrant,
}

#[derive(thiserror::Error, Debug)]
pub enum QdrantStoreError {
    #[error("Qdrant build error occurred: {0:?}")]
    QdrantProcessError(#[source] QdrantError),
    #[error("Qdrant vector size mismatch found: expected {expected} but found {actual}")]
    VectorSizeMismatch { expected: u64, actual: u64 },
    #[error("Qdrant input error: {msg}")]
    InputError { msg: String },
    #[error("Qdrant returned a point that cannot be read: {0}")]
    MalformedPoint(#[from] RetrievedChunkError),
}

impl QdrantStore {
    pub fn connect(url: &str, api_key: Option<&str>) -> Result<Self, QdrantStoreError> {
        let mut builder = Qdrant::from_url(url);
        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }
        let qdrant_client = builder
            .build()
            .map_err(QdrantStoreError::QdrantProcessError)?;
        Ok(Self {
            client: qdrant_client,
        })
    }

    pub async fn ensure_collection(
        &self,
        name: &str,
        vector_size: u64,
        recreate: bool,
    ) -> Result<(), QdrantStoreError> {
        let exists = self
            .client
            .collection_exists(name)
            .await
            .map_err(QdrantStoreError::QdrantProcessError)?;

        if exists && recreate {
            self.client
                .delete_collection(name)
                .await
                .map_err(QdrantStoreError::QdrantProcessError)?;
        }

        if !exists || recreate {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(name)
                        .vectors_config(VectorParamsBuilder::new(vector_size, Distance::Cosine)),
                )
                .await
                .map_err(QdrantStoreError::QdrantProcessError)?;
            return Ok(());
        }

        // existing collection, not recreating: validate instead of assuming
        let info = self
            .client
            .collection_info(name)
            .await
            .map_err(QdrantStoreError::QdrantProcessError)?;

        let info_vector_size: Option<u64> = match info
            .result
            .and_then(|res| res.config)
            .and_then(|config| config.params)
            .and_then(|params| params.vectors_config)
            .and_then(|vec_config| vec_config.config)
        {
            Some(Config::Params(vec_params)) => Some(vec_params.size),
            Some(Config::ParamsMap(map)) => map.map.values().next().map(|v| v.size),
            _ => None,
        };

        if let Some(actual) = info_vector_size
            && actual != vector_size
        {
            return Err(QdrantStoreError::VectorSizeMismatch {
                expected: vector_size,
                actual,
            });
        }
        Ok(())
    }

    fn point_id(source_file: &str, chunk_index: usize) -> Uuid {
        Uuid::new_v5(
            &NAMESPACE,
            format!("{source_file}#{chunk_index}").as_bytes(),
        )
    }

    pub async fn upsert_chunks(
        &self,
        collection: &str,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<(), QdrantStoreError> {
        if chunks.len() != embeddings.len() {
            return Err(QdrantStoreError::InputError {
                msg: format!(
                    "Size mismatch between chunks ({}) and embeddings ({})",
                    chunks.len(),
                    embeddings.len()
                ),
            });
        }
        let points: Result<Vec<PointStruct>, QdrantStoreError> = chunks
            .iter()
            .zip(embeddings)
            .map(|(c, vector)| {
                let source_file_string =
                    c.source_file
                        .to_str()
                        .ok_or_else(|| QdrantStoreError::InputError {
                            msg: format!(
                                "Invalid source file path received from chunk {}: {:?}",
                                c.chunk_index, c.source_file
                            ),
                        })?;
                Ok(PointStruct::new(
                    Self::point_id(source_file_string, c.chunk_index),
                    vector.clone(),
                    [
                        ("source_file", source_file_string.into()),
                        ("page_number", (c.page_number as i64).into()),
                        ("chunk_index", (c.chunk_index as i64).into()),
                        ("text", c.text.clone().into()),
                    ],
                ))
            })
            .collect();

        self.client
            .upsert_points_chunked(
                UpsertPointsBuilder::new(collection, points?).wait(true),
                100,
            )
            .await
            .map_err(QdrantStoreError::QdrantProcessError)?;
        Ok(())
    }

    pub async fn search(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        top_k: u64,
    ) -> Result<Vec<RetrievedChunk>, QdrantStoreError> {
        let response = self
            .client
            .query(
                QueryPointsBuilder::new(collection)
                    .query(query_vector)
                    .limit(top_k)
                    .with_payload(true),
            )
            .await
            .map_err(QdrantStoreError::QdrantProcessError)?;
        let chunks = response
            .result
            .into_iter()
            .map(RetrievedChunk::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::path::PathBuf;

    #[test]
    fn point_id_is_deterministic() {
        assert_eq!(
            QdrantStore::point_id("boiler_x200_manual.pdf", 3),
            QdrantStore::point_id("boiler_x200_manual.pdf", 3)
        );
    }

    #[test]
    fn point_id_differs_by_chunk_index() {
        assert_ne!(
            QdrantStore::point_id("boiler_x200_manual.pdf", 0),
            QdrantStore::point_id("boiler_x200_manual.pdf", 1)
        );
    }

    #[test]
    fn point_id_differs_by_source_file() {
        assert_ne!(
            QdrantStore::point_id("boiler_x200_manual.pdf", 0),
            QdrantStore::point_id("pump_y50_manual.pdf", 0)
        );
    }

    #[test]
    fn point_id_is_a_version_5_uuid() {
        assert_eq!(QdrantStore::point_id("a.pdf", 0).get_version_num(), 5);
    }

    // Pins NAMESPACE and the "{source_file}#{chunk_index}" format. Changing either
    // re-derives every ID, so points that are already stored would be orphaned
    // instead of upserted in place on the next ingest.
    #[test]
    fn point_id_matches_known_values() {
        assert_eq!(
            QdrantStore::point_id("boiler_x200_manual.pdf", 0).to_string(),
            "900d8c16-add4-540e-a684-f68e54c51f3c"
        );
        assert_eq!(
            QdrantStore::point_id("boiler_x200_manual.pdf", 1).to_string(),
            "7d1fe917-8285-5cdb-af0c-882c63ed18df"
        );
        assert_eq!(
            QdrantStore::point_id("pump_y50_manual.pdf", 0).to_string(),
            "6e47ec0a-1a82-5c8e-9d08-0b5caac7469f"
        );
    }

    #[test]
    fn vector_size_mismatch_message_names_both_sizes() {
        let err = QdrantStoreError::VectorSizeMismatch {
            expected: 1536,
            actual: 1500,
        };
        assert_eq!(
            err.to_string(),
            "Qdrant vector size mismatch found: expected 1536 but found 1500"
        );
    }

    // ---- upsert_chunks: chunks and embeddings are paired by position ----

    fn chunk(chunk_index: usize) -> Chunk {
        Chunk {
            source_file: PathBuf::from("manual.pdf"),
            page_number: 1,
            chunk_index,
            text: format!("chunk {chunk_index}"),
        }
    }

    // zip() silently drops the longer side, so a different number of chunks and
    // embeddings would store the wrong vector with the wrong text. It has to fail
    // before anything is sent. Nothing listens on port 1: a request would fail with
    // a connection error, so getting an InputError proves none was made.
    #[rstest]
    #[case::fewer_embeddings(2, 1)]
    #[case::more_embeddings(1, 2)]
    #[case::no_embeddings(2, 0)]
    #[tokio::test]
    async fn upsert_chunks_rejects_a_different_number_of_embeddings(
        #[case] chunk_count: usize,
        #[case] embedding_count: usize,
    ) {
        let store = QdrantStore::connect("http://127.0.0.1:1", None).unwrap();
        let chunks: Vec<Chunk> = (0..chunk_count).map(chunk).collect();
        let embeddings = vec![vec![0.0_f32; 4]; embedding_count];

        let err = store
            .upsert_chunks("unused", &chunks, &embeddings)
            .await
            .unwrap_err();

        let expected = format!("chunks ({chunk_count}) and embeddings ({embedding_count})");
        assert!(
            matches!(&err, QdrantStoreError::InputError { msg } if msg.contains(&expected)),
            "got {err:?}"
        );
    }

    // The guard must not turn valid input away: with matching counts the call gets
    // as far as the network, which is refused here.
    #[tokio::test]
    async fn upsert_chunks_with_matching_counts_gets_past_the_guard() {
        let store = QdrantStore::connect("http://127.0.0.1:1", None).unwrap();
        let chunks = vec![chunk(0), chunk(1)];
        let embeddings = vec![vec![0.0_f32; 4]; 2];

        let err = store
            .upsert_chunks("unused", &chunks, &embeddings)
            .await
            .unwrap_err();

        assert!(
            matches!(err, QdrantStoreError::QdrantProcessError(_)),
            "got {err:?}"
        );
    }
}
