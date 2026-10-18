use std::fmt::Debug;

use crate::cli::ConnectionArgs;
use url::Url;

pub struct AppConfig {
    azure_openai_endpoint: String,
    #[allow(dead_code)] // will be used in the future
    azure_openai_api_key: String,
    azure_openai_api_version: String,
    azure_openai_embedding_deployment: String,
    azure_openai_chat_deployment: String,
    pub qdrant_url: Url,
    pub qdrant_api_key: Option<String>,
    pub qdrant_collection: String,
    pub embedding_dimensions: u16,
}

impl Debug for AppConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppConfig")
            .field("azure_openai_endpoint", &self.azure_openai_endpoint)
            .field("azure_openai_api_key", &"[REDACTED]")
            .field("azure_openai_api_version", &self.azure_openai_api_version)
            .field(
                "azure_openai_embedding_deployment",
                &self.azure_openai_embedding_deployment,
            )
            .field(
                "azure_openai_chat_deployment",
                &self.azure_openai_chat_deployment,
            )
            .field("qdrant_url", &self.qdrant_url)
            .field("qdrant_api_key", &"[REDACTED]")
            .field("qdrant_collection", &self.qdrant_collection)
            .field("embedding_dimensions", &self.embedding_dimensions)
            .finish()
    }
}

impl AppConfig {
    pub fn try_from(cargs: ConnectionArgs) -> Result<Self, ConfigError> {
        let azure_openai_endpoint = cargs.endpoint.ok_or(ConfigError::Missing {
            field: "azure_openai_endpoint",
            env_var: "AZURE_OPENAI_ENDPOINT",
            flag: "endpoint",
        })?;

        let azure_openai_api_key = cargs.api_key.ok_or(ConfigError::Missing {
            field: "azure_openai_api_key",
            env_var: "AZURE_OPENAI_API_KEY",
            flag: "api_key",
        })?;

        let azure_openai_embedding_deployment =
            cargs.embedding_deployment.ok_or(ConfigError::Missing {
                field: "azure_openai_embedding_deployment",
                env_var: "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                flag: "embedding_deployment",
            })?;

        let azure_openai_chat_deployment = cargs.chat_deployment.ok_or(ConfigError::Missing {
            field: "azure_openai_chat_deployment",
            env_var: "AZURE_OPENAI_CHAT_DEPLOYMENT",
            flag: "chat_deployment",
        })?;

        let qdrant_url = url::Url::parse(&cargs.qdrant_url)
            .map_err(|e| ConfigError::InvalidQdrantUrl(cargs.qdrant_url.clone(), e))?;

        Ok(AppConfig {
            azure_openai_endpoint,
            azure_openai_api_key,
            azure_openai_api_version: cargs.api_version,
            azure_openai_embedding_deployment,
            azure_openai_chat_deployment,
            qdrant_url,
            qdrant_api_key: cargs.qdrant_api_key,
            qdrant_collection: cargs.qdrant_collection,
            embedding_dimensions: cargs.embedding_dimensions,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ConfigError {
    #[error(
        "missing {field}: set the {env_var} environment variable, add it to .env or pass --{flag}"
    )]
    Missing {
        field: &'static str,
        env_var: &'static str,
        flag: &'static str,
    },
    #[error("invalid Qdrant URL {0:?}")]
    InvalidQdrantUrl(String, #[source] url::ParseError),
    #[error("ConfigError occurred: {0}")]
    Other(String),
}

/*
 * TEST SUITE
 */

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_connection_args() -> ConnectionArgs {
        ConnectionArgs {
            endpoint: Some("https://my-resource.openai.azure.com".to_string()),
            api_key: Some("secret-key".to_string()),
            api_version: "2024-08-01-preview".to_string(),
            embedding_deployment: Some("embeddings-prod".to_string()),
            chat_deployment: Some("chat-prod".to_string()),
            qdrant_url: "http://localhost:6334".to_string(),
            qdrant_api_key: None,
            qdrant_collection: "machinery_docs".to_string(),
            embedding_dimensions: 1536,
        }
    }

    #[test]
    fn valid_args_produce_ok_config() {
        let config = AppConfig::try_from(valid_connection_args()).expect("should build config");

        assert_eq!(
            config.azure_openai_endpoint,
            "https://my-resource.openai.azure.com"
        );
        assert_eq!(config.azure_openai_api_key, "secret-key");
        assert_eq!(config.azure_openai_api_version, "2024-08-01-preview");
        assert_eq!(config.azure_openai_embedding_deployment, "embeddings-prod");
        assert_eq!(config.azure_openai_chat_deployment, "chat-prod");
        assert_eq!(
            config.qdrant_url,
            Url::parse("http://localhost:6334").unwrap()
        );
        assert_eq!(config.qdrant_api_key, None);
        assert_eq!(config.qdrant_collection, "machinery_docs");
        assert_eq!(config.embedding_dimensions, 1536);
    }

    #[test]
    fn qdrant_api_key_is_optional() {
        let args = ConnectionArgs {
            qdrant_api_key: Some("qdrant-secret".to_string()),
            ..valid_connection_args()
        };

        let config = AppConfig::try_from(args).expect("should build config");
        assert_eq!(config.qdrant_api_key, Some("qdrant-secret".to_string()));
    }

    #[test]
    fn missing_endpoint_reports_actionable_error() {
        let args = ConnectionArgs {
            endpoint: None,
            ..valid_connection_args()
        };

        let err = AppConfig::try_from(args).expect_err("should fail without endpoint");

        assert!(matches!(
            err,
            ConfigError::Missing {
                field: "azure_openai_endpoint",
                env_var: "AZURE_OPENAI_ENDPOINT",
                flag: "endpoint",
            }
        ));
        assert_eq!(
            err.to_string(),
            "missing azure_openai_endpoint: set the AZURE_OPENAI_ENDPOINT environment variable, add it to .env or pass --endpoint"
        );
    }

    #[test]
    fn missing_api_key_reports_actionable_error() {
        let args = ConnectionArgs {
            api_key: None,
            ..valid_connection_args()
        };

        let err = AppConfig::try_from(args).expect_err("should fail without api key");

        assert!(matches!(
            err,
            ConfigError::Missing {
                field: "azure_openai_api_key",
                env_var: "AZURE_OPENAI_API_KEY",
                flag: "api_key",
            }
        ));
    }

    #[test]
    fn missing_embedding_deployment_reports_actionable_error() {
        let args = ConnectionArgs {
            embedding_deployment: None,
            ..valid_connection_args()
        };

        let err = AppConfig::try_from(args).expect_err("should fail without embedding deployment");

        assert!(matches!(
            err,
            ConfigError::Missing {
                field: "azure_openai_embedding_deployment",
                env_var: "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                flag: "embedding_deployment",
            }
        ));
    }

    #[test]
    fn missing_chat_deployment_reports_actionable_error() {
        let args = ConnectionArgs {
            chat_deployment: None,
            ..valid_connection_args()
        };

        let err = AppConfig::try_from(args).expect_err("should fail without chat deployment");

        assert!(matches!(
            err,
            ConfigError::Missing {
                field: "azure_openai_chat_deployment",
                env_var: "AZURE_OPENAI_CHAT_DEPLOYMENT",
                flag: "chat_deployment",
            }
        ));
    }

    #[test]
    fn invalid_qdrant_url_is_rejected() {
        let args = ConnectionArgs {
            qdrant_url: "not a url".to_string(),
            ..valid_connection_args()
        };

        let err = AppConfig::try_from(args).expect_err("should fail on malformed qdrant url");

        match err {
            ConfigError::InvalidQdrantUrl(raw, _source) => assert_eq!(raw, "not a url"),
            other => panic!("expected InvalidQdrantUrl, got {other:?}"),
        }
    }

    #[test]
    fn debug_output_never_contains_either_api_key() {
        let args = ConnectionArgs {
            qdrant_api_key: Some("qdrant-secret".to_string()),
            ..valid_connection_args()
        };
        let config = AppConfig::try_from(args).unwrap();

        for rendered in [format!("{config:?}"), format!("{config:#?}")] {
            assert!(
                !rendered.contains("secret-key"),
                "azure api key leaked: {rendered}"
            );
            assert!(
                !rendered.contains("qdrant-secret"),
                "qdrant api key leaked: {rendered}"
            );
            assert!(rendered.contains("[REDACTED]"));
        }
    }

    #[test]
    fn debug_output_still_shows_the_non_secret_settings() {
        let config = AppConfig::try_from(valid_connection_args()).unwrap();
        let rendered = format!("{config:?}");

        assert!(rendered.contains("https://my-resource.openai.azure.com"));
        assert!(rendered.contains("embeddings-prod"));
        assert!(rendered.contains("machinery_docs"));
        assert!(rendered.contains("1536"));
    }
}
