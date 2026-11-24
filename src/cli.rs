use crate::{
    ai::{
        azure_provider::{AzureEmbeddingSettings, AzureOpenAiProvider, DEFAULT_REQUEST_TIMEOUT},
        hash_provider::HashProvider,
        provider::Provider,
        retry::RetryPolicy,
    },
    chunk::ChunkOptions,
    config::AppConfig,
    vector_store::qdrant::QdrantStore,
};
use anyhow::{Context, Error, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(ValueEnum, Clone, Debug)]
pub enum ProviderKind {
    Azure,
    Hash,
}

#[derive(Parser, Debug)]
#[command(name = "machinery-doc-query")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Ingest {
        #[arg(short, long)]
        input_dir: PathBuf,
        #[arg(short = 's', long, default_value_t = 1500)]
        chunk_size: u16,
        #[arg(short = 'o', long, default_value_t = 200)]
        chunk_overlap: u16,
        #[arg(short, long)]
        recreate_collection: bool,
        #[arg(short, long, default_value_t = 96)]
        batch_size: u16,
        #[arg(short, long)]
        dry_run: bool,
        #[command(flatten)]
        connection: ConnectionArgs,
    },
    Query {
        // positional - no #[arg], clap treats an unattributed field as one
        question: String,
        #[arg(short, long, default_value_t = 5)]
        top_k: u16,
        #[command(flatten)]
        connection: ConnectionArgs,
    },
    Doctor {
        #[command(flatten)]
        connection: ConnectionArgs,
    },
}

#[derive(Args, Debug)]
pub struct ConnectionArgs {
    #[arg(long, env = "AZURE_OPENAI_ENDPOINT")]
    pub endpoint: Option<String>,
    #[arg(long, env = "AZURE_OPENAI_API_KEY")]
    pub api_key: Option<String>,
    #[arg(
        long,
        env = "AZURE_OPENAI_API_VERSION",
        default_value_t = String::from("2024-08-01-preview")
    )]
    pub api_version: String,
    #[arg(long, env = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")]
    pub embedding_deployment: Option<String>,
    #[arg(long, env = "AZURE_OPENAI_CHAT_DEPLOYMENT")]
    pub chat_deployment: Option<String>,
    #[arg(
        long,
        env = "QDRANT_URL",
        default_value_t = String::from("http://localhost:6334")
    )]
    pub qdrant_url: String,
    #[arg(long, env = "QDRANT_API_KEY")]
    pub qdrant_api_key: Option<String>,
    #[arg(
        long,
        env = "QDRANT_COLLECTION",
        default_value_t = String::from("machinery_docs")
    )]
    pub qdrant_collection: String,
    #[arg(long, env = "EMBEDDING_DIMENSIONS", default_value_t = 1536)]
    pub embedding_dimensions: u16,
    #[arg(long, env = "EMBEDDING_PROVIDER", value_enum, default_value_t = ProviderKind::Azure)]
    pub provider_kind: ProviderKind,
    /// Seconds one embedding request may take before it counts as timed out (azure provider)
    #[arg(
        long,
        env = "EMBEDDING_REQUEST_TIMEOUT_SECS",
        default_value_t = DEFAULT_REQUEST_TIMEOUT.as_secs(),
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub request_timeout_secs: u64,
    /// How many times a timed out or dropped embedding request is tried (azure provider)
    #[arg(
        long,
        env = "EMBEDDING_MAX_ATTEMPTS",
        default_value_t = RetryPolicy::default().max_attempts,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub max_attempts: u32,
}

pub async fn open_cli() -> Result<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    match cli.command {
        Commands::Ingest {
            input_dir,
            chunk_size,
            chunk_overlap,
            recreate_collection,
            batch_size,
            dry_run,
            connection,
        } => {
            process_ingest(
                input_dir,
                chunk_size,
                chunk_overlap,
                recreate_collection,
                batch_size,
                dry_run,
                connection,
            )
            .await
        }
        Commands::Query {
            question,
            top_k,
            connection,
        } => process_query(question, top_k, connection),
        Commands::Doctor { connection } => process_doctor(connection),
    }
}

async fn process_ingest(
    input_dir: PathBuf,
    chunk_size: u16,
    chunk_overlap: u16,
    recreate_collection: bool,
    batch_size: u16,
    // this will be used in the future
    #[allow(unused)] dry_run: bool,
    connection: ConnectionArgs,
) -> Result<(), Error> {
    let provider_kind = connection.provider_kind.clone();
    let app_config = AppConfig::try_from(connection)?;
    let store = QdrantStore::connect(
        app_config.qdrant_url.as_str(),
        app_config.qdrant_api_key.as_deref(),
    )
    .context("QdrantStore error occurred")?;
    let collection = &app_config.qdrant_collection;
    // clippy note: converting `u16` to `usize` cannot fail
    let chunk_size = usize::from(chunk_size);
    let overlap = usize::from(chunk_overlap);
    let chunk_options = ChunkOptions {
        chunk_size,
        overlap,
    };
    let recreate = recreate_collection;

    let provider = build_provider(&provider_kind, &app_config, batch_size);
    let summary = crate::ingest::run::<Provider>(
        &input_dir,
        &store,
        collection,
        &provider,
        batch_size.into(),
        &chunk_options,
        recreate,
    )
    .await
    .context("Ingest Error")?;
    println!("{:?}", summary);
    summary.ensure_ingested_something()?;
    Ok(())
}

fn build_provider(
    provider_kind: &ProviderKind,
    app_config: &AppConfig,
    batch_size: u16,
) -> Provider {
    match provider_kind {
        ProviderKind::Azure => Provider::Azure(Box::new(
            AzureOpenAiProvider::new(
                AzureEmbeddingSettings::from(app_config),
                app_config.embedding_dimensions,
                batch_size,
            )
            .with_request_timeout(app_config.request_timeout)
            .with_retry_policy(RetryPolicy {
                max_attempts: app_config.max_attempts,
                ..Default::default()
            }),
        )),
        ProviderKind::Hash => Provider::Hash(HashProvider {
            dimensions: app_config.embedding_dimensions,
        }),
    }
}

fn process_query(question: String, top_k: u16, connection: ConnectionArgs) -> Result<()> {
    let app_config = AppConfig::try_from(connection)?;
    println!("{:?}", app_config);
    println!("'query' was called");
    drop(question);
    let _ = top_k;
    Ok(())
}

fn process_doctor(connection: ConnectionArgs) -> Result<()> {
    let app_config = AppConfig::try_from(connection)?;
    println!("{:?}", app_config);
    println!("'doctor' was called");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::provider::EmbeddingProvider;
    use clap::{CommandFactory, error::ErrorKind};
    use rstest::rstest;
    use std::time::Duration;
    use url::Url;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    // Note: the connection flags also read real environment variables
    // (QDRANT_URL etc.), so these tests only assert on non-env arguments.
    fn parse(args: &[&str]) -> std::result::Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("machinery-doc-query").chain(args.iter().copied()))
    }

    #[test]
    fn a_subcommand_is_required() {
        assert!(parse(&[]).is_err());
    }

    #[test]
    fn unknown_flags_are_rejected() {
        let err = parse(&["ingest", "--input-dir", "pdfs", "--bogus"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    }

    #[test]
    fn ingest_requires_an_input_dir() {
        let err = parse(&["ingest"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn ingest_defaults_match_the_planned_chunking_settings() {
        let cli = parse(&["ingest", "--input-dir", "pdfs"]).unwrap();
        let Commands::Ingest {
            input_dir,
            chunk_size,
            chunk_overlap,
            recreate_collection,
            batch_size,
            dry_run,
            ..
        } = cli.command
        else {
            panic!("expected the ingest command");
        };

        assert_eq!(input_dir, PathBuf::from("pdfs"));
        assert_eq!(chunk_size, 1500);
        assert_eq!(chunk_overlap, 200);
        assert_eq!(batch_size, 96);
        assert!(!recreate_collection);
        assert!(!dry_run);
    }

    #[test]
    fn ingest_accepts_the_short_flags() {
        let cli = parse(&[
            "ingest", "-i", "pdfs", "-s", "800", "-o", "100", "-b", "32", "-r", "-d",
        ])
        .unwrap();
        let Commands::Ingest {
            input_dir,
            chunk_size,
            chunk_overlap,
            recreate_collection,
            batch_size,
            dry_run,
            ..
        } = cli.command
        else {
            panic!("expected the ingest command");
        };

        assert_eq!(input_dir, PathBuf::from("pdfs"));
        assert_eq!(chunk_size, 800);
        assert_eq!(chunk_overlap, 100);
        assert_eq!(batch_size, 32);
        assert!(recreate_collection);
        assert!(dry_run);
    }

    #[test]
    fn ingest_rejects_a_chunk_size_that_does_not_fit_in_u16() {
        let err = parse(&["ingest", "-i", "pdfs", "--chunk-size", "70000"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn query_takes_the_question_as_a_positional_argument() {
        let cli = parse(&["query", "how do I drain the boiler?"]).unwrap();
        let Commands::Query {
            question, top_k, ..
        } = cli.command
        else {
            panic!("expected the query command");
        };

        assert_eq!(question, "how do I drain the boiler?");
        assert_eq!(top_k, 5);
    }

    #[test]
    fn query_requires_a_question() {
        let err = parse(&["query"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn query_top_k_can_be_overridden() {
        let cli = parse(&["query", "a question", "--top-k", "3"]).unwrap();
        let Commands::Query { top_k, .. } = cli.command else {
            panic!("expected the query command");
        };
        assert_eq!(top_k, 3);
    }

    #[test]
    fn doctor_needs_no_arguments() {
        let cli = parse(&["doctor"]).unwrap();
        assert!(matches!(cli.command, Commands::Doctor { .. }));
    }

    // ---- embedding provider ----

    #[test]
    fn the_provider_defaults_to_azure() {
        // Read the declared default instead of parsing: parsing would also pick
        // up EMBEDDING_PROVIDER from the environment of whoever runs the tests.
        let command = Cli::command();
        let ingest = command.find_subcommand("ingest").unwrap();
        let provider = ingest
            .get_arguments()
            .find(|arg| arg.get_id().as_str() == "provider_kind")
            .expect("the provider flag exists");

        let defaults: Vec<String> = provider
            .get_default_values()
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect();

        assert_eq!(defaults, vec!["azure"]);
    }

    // The flag beats the environment variable, so these two do not depend on
    // what is set where the tests run.
    #[test]
    fn the_provider_flag_accepts_azure() {
        let cli = parse(&["ingest", "-i", "pdfs", "--provider-kind", "azure"]).unwrap();
        let Commands::Ingest { connection, .. } = cli.command else {
            panic!("expected the ingest command");
        };

        assert!(matches!(connection.provider_kind, ProviderKind::Azure));
    }

    // query has to embed the question with the same provider that ingested the
    // documents, so every command takes the flag.
    #[rstest]
    #[case::ingest(&["ingest", "-i", "pdfs"])]
    #[case::query(&["query", "a question"])]
    #[case::doctor(&["doctor"])]
    fn every_command_accepts_the_provider_flag(#[case] args: &[&str]) {
        let mut args = args.to_vec();
        args.extend(["--provider-kind", "hash"]);

        let cli = parse(&args).unwrap();

        let connection = match cli.command {
            Commands::Ingest { connection, .. }
            | Commands::Query { connection, .. }
            | Commands::Doctor { connection } => connection,
        };
        assert!(matches!(connection.provider_kind, ProviderKind::Hash));
    }

    #[test]
    fn an_unknown_provider_is_rejected_and_the_choices_are_listed() {
        let err = parse(&["ingest", "-i", "pdfs", "--provider-kind", "ollama"]).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::InvalidValue);
        let message = err.to_string();
        assert!(
            message.contains("azure") && message.contains("hash"),
            "the error should list the valid choices, got: {message}"
        );
    }

    // ---- embedding request settings ----

    #[test]
    fn the_request_settings_default_to_the_providers_own_defaults() {
        // Declared defaults, not a parse: parsing would also read EMBEDDING_* variables
        // from the environment of whoever runs the tests.
        let command = Cli::command();
        let ingest = command.find_subcommand("ingest").unwrap();
        let default_of = |id: &str| -> Vec<String> {
            ingest
                .get_arguments()
                .find(|arg| arg.get_id().as_str() == id)
                .expect("the flag exists")
                .get_default_values()
                .iter()
                .map(|value| value.to_string_lossy().into_owned())
                .collect()
        };

        assert_eq!(
            default_of("request_timeout_secs"),
            vec![DEFAULT_REQUEST_TIMEOUT.as_secs().to_string()]
        );
        assert_eq!(
            default_of("max_attempts"),
            vec![RetryPolicy::default().max_attempts.to_string()]
        );
    }

    // The flags beat the environment variables, so this does not depend on what is set
    // where the tests run. Every command takes them: query embeds with the same provider.
    #[rstest]
    #[case::ingest(&["ingest", "-i", "pdfs"])]
    #[case::query(&["query", "a question"])]
    #[case::doctor(&["doctor"])]
    fn every_command_accepts_the_request_settings(#[case] args: &[&str]) {
        let mut args = args.to_vec();
        args.extend(["--request-timeout-secs", "30", "--max-attempts", "3"]);

        let cli = parse(&args).unwrap();

        let connection = match cli.command {
            Commands::Ingest { connection, .. }
            | Commands::Query { connection, .. }
            | Commands::Doctor { connection } => connection,
        };
        assert_eq!(connection.request_timeout_secs, 30);
        assert_eq!(connection.max_attempts, 3);
    }

    // A timeout of 0 would fail every request at once, and 0 attempts means nothing.
    #[rstest]
    #[case::zero_timeout("--request-timeout-secs", "0")]
    #[case::zero_attempts("--max-attempts", "0")]
    #[case::timeout_not_a_number("--request-timeout-secs", "soon")]
    #[case::attempts_not_a_number("--max-attempts", "many")]
    fn an_unusable_request_setting_is_rejected(#[case] flag: &str, #[case] value: &str) {
        let err = parse(&["ingest", "-i", "pdfs", flag, value]).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::ValueValidation);
    }

    // ---- build_provider ----

    fn app_config(endpoint: &str) -> AppConfig {
        AppConfig {
            azure_openai_endpoint: endpoint.to_string(),
            azure_openai_api_key: "test-key".to_string(),
            azure_openai_api_version: "2024-08-01-preview".to_string(),
            azure_openai_embedding_deployment: "emb-prod".to_string(),
            azure_openai_chat_deployment: "chat-prod".to_string(),
            qdrant_url: Url::parse("http://localhost:6334").unwrap(),
            qdrant_api_key: None,
            qdrant_collection: "machinery_docs".to_string(),
            embedding_dimensions: 8,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_attempts: 5,
        }
    }

    #[test]
    fn build_provider_picks_the_provider_that_was_asked_for() {
        let config = app_config("https://my-resource.openai.azure.com");

        let hash = build_provider(&ProviderKind::Hash, &config, 10);
        let azure = build_provider(&ProviderKind::Azure, &config, 10);

        assert!(matches!(hash, Provider::Hash(_)));
        assert!(matches!(azure, Provider::Azure(_)));
    }

    // The settings have to get from the config into the provider. A server that never
    // answers, a 150 ms timeout and 2 attempts: exactly 2 requests, then "timed out".
    #[tokio::test]
    async fn build_provider_gives_the_azure_provider_the_configured_timeout_and_attempts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(2)))
            .mount(&server)
            .await;
        let config = AppConfig {
            request_timeout: Duration::from_millis(150),
            max_attempts: 2,
            ..app_config(&server.uri())
        };
        let provider = build_provider(&ProviderKind::Azure, &config, 10);

        let err = provider.embed(&["some text"]).await.unwrap_err();

        assert!(err.to_string().contains("timed out"), "got: {err}");
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }
}
