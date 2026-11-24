use crate::{
    ai::{
        azure_provider::{AzureEmbeddingSettings, AzureOpenAiProvider},
        hash_provider::HashProvider,
        provider::Provider,
    },
    chunk::ChunkOptions,
    config::AppConfig,
    ingest, query, render,
    vector_store::qdrant::QdrantStore,
};
use anyhow::{Context, Error, Result};
use clap::{Args, Parser, Subcommand, ValueEnum, value_parser};
use std::{fmt::Display, path::PathBuf};

const DEFAULT_BATCH_SIZE: u16 = 96;

#[derive(ValueEnum, Clone, Debug)]
pub enum ProviderKind {
    Azure,
    Hash,
}

impl Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderKind::Azure => f.write_str("azure"),
            ProviderKind::Hash => f.write_str("hash"),
        }
    }
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
        #[arg(short, long, default_value_t = DEFAULT_BATCH_SIZE)]
        batch_size: u16,
        #[arg(short, long)]
        dry_run: bool,
        #[command(flatten)]
        connection: ConnectionArgs,
    },
    Query {
        #[arg(value_parser = parse_question)]
        question: String,
        #[arg(short, long, default_value_t = 5, value_parser = value_parser!(u16).range(1..))]
        top_k: u16,
        #[command(flatten)]
        connection: ConnectionArgs,
    },
    Doctor {
        #[command(flatten)]
        connection: ConnectionArgs,
    },
}

// A question with nothing in it is a usage error (the usage text, exit code 2), not a
// failure after connecting to Qdrant. The question is kept exactly as it was typed.
fn parse_question(raw: &str) -> Result<String, String> {
    if raw.trim().is_empty() {
        Err("the question must not be blank".to_owned())
    } else {
        Ok(raw.to_owned())
    }
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
        } => process_query(question, top_k, connection).await,
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
    let store = connect_to_qdrant(&app_config).await?;
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
    let summary = ingest::run::<Provider>(
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

async fn process_query(question: String, top_k: u16, connection: ConnectionArgs) -> Result<()> {
    let provider_kind = connection.provider_kind.clone();
    let app_config = AppConfig::try_from(connection)?;
    let store = connect_to_qdrant(&app_config).await?;
    let provider = build_provider(&provider_kind, &app_config, DEFAULT_BATCH_SIZE);
    let collection = &app_config.qdrant_collection;
    let chunks = query::run(&question, &store, collection, &provider, top_k as u64).await?;
    println!("Provider: {provider_kind}");
    println!("{}", render::render_chunks(&chunks));
    Ok(())
}

fn process_doctor(connection: ConnectionArgs) -> Result<()> {
    let app_config = AppConfig::try_from(connection)?;
    println!("{:?}", app_config);
    println!("'doctor' was called");
    Ok(())
}

fn build_provider(
    provider_kind: &ProviderKind,
    app_config: &AppConfig,
    batch_size: u16,
) -> Provider {
    match provider_kind {
        ProviderKind::Azure => Provider::Azure(Box::new(AzureOpenAiProvider::new(
            AzureEmbeddingSettings::from(app_config),
            app_config.embedding_dimensions,
            batch_size,
        ))),
        ProviderKind::Hash => Provider::Hash(HashProvider {
            dimensions: app_config.embedding_dimensions,
        }),
    }
}

// Connects to Qdrant and asks it to answer, so that a stopped container is reported
// first, before anything is parsed, embedded or searched.
async fn connect_to_qdrant(app_config: &AppConfig) -> Result<QdrantStore> {
    let url = app_config.qdrant_url.as_str();
    let store = QdrantStore::connect(url, app_config.qdrant_api_key.as_deref())?;
    store
        .health_check()
        .await
        .with_context(|| format!("Cannot reach Qdrant at {url}. Is Docker running?"))?;
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::provider::EmbeddingProvider;
    use clap::{CommandFactory, error::ErrorKind};
    use rstest::rstest;

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

    // ---- input that clap refuses ----

    // query::run refuses a top_k of 0 as well, and Qdrant would, but a usage error comes
    // first: it says what is wrong with the command line before anything is connected.
    // 65536 is the first number that does not fit the u16 of the field.
    #[rstest]
    #[case::zero(&["--top-k", "0"])]
    #[case::zero_with_the_short_flag(&["-t", "0"])]
    #[case::too_large_for_a_u16(&["--top-k", "65536"])]
    #[case::not_a_number(&["--top-k", "many"])]
    fn query_rejects_a_top_k_outside_1_to_65535(#[case] top_k_args: &[&str]) {
        let mut args = vec!["query", "a question"];
        args.extend(top_k_args);

        let err = parse(&args).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::ValueValidation);
    }

    // Both ends of the range are allowed. The value parser has to produce the type of the
    // field (u16): one of another type makes reading the field panic.
    #[rstest]
    #[case::smallest("1", 1)]
    #[case::largest("65535", 65535)]
    fn query_accepts_a_top_k_from_1_to_65535(#[case] given: &str, #[case] expected: u16) {
        let cli = parse(&["query", "a question", "--top-k", given]).unwrap();

        let Commands::Query { top_k, .. } = cli.command else {
            panic!("expected the query command");
        };
        assert_eq!(top_k, expected);
    }

    #[rstest]
    #[case::empty("")]
    #[case::spaces("   ")]
    #[case::tabs_and_newlines("\n\t")]
    fn query_rejects_a_blank_question(#[case] question: &str) {
        let err = parse(&["query", question]).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::ValueValidation);
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

    // ---- showing and building the provider ----

    // The header of a query names the provider the way the flag spells it, so it can be
    // compared with what was typed when the documents were ingested.
    #[test]
    fn the_provider_is_shown_the_way_the_flag_spells_it() {
        for kind in ProviderKind::value_variants() {
            let flag_value = kind
                .to_possible_value()
                .expect("every provider can be chosen with the flag");

            assert_eq!(kind.to_string(), flag_value.get_name());
        }
    }

    // The provider kind and config of a query for the given kind, made from flags (which
    // beat the environment) the way the commands make theirs.
    fn kind_and_config(provider_kind: &str) -> (ProviderKind, AppConfig) {
        let cli = parse(&[
            "query",
            "a question",
            "--provider-kind",
            provider_kind,
            "--endpoint",
            "https://placeholder.openai.azure.com",
            "--api-key",
            "placeholder",
            "--embedding-deployment",
            "placeholder",
            "--chat-deployment",
            "placeholder",
            "--embedding-dimensions",
            "8",
        ])
        .unwrap();
        let Commands::Query { connection, .. } = cli.command else {
            panic!("expected the query command");
        };
        let kind = connection.provider_kind.clone();
        (kind, AppConfig::try_from(connection).unwrap())
    }

    // Both commands get their provider from build_provider, and the vectors it makes have
    // to be as long as the config says: that is the size the collection is created with.
    #[test]
    fn build_provider_makes_a_hash_provider_with_the_configured_dimensions() {
        let (kind, config) = kind_and_config("hash");

        let provider = build_provider(&kind, &config, DEFAULT_BATCH_SIZE);

        assert!(matches!(provider, Provider::Hash(_)));
        assert_eq!(provider.embedded_dimensions(), 8);
    }

    #[test]
    fn build_provider_makes_an_azure_provider_with_the_configured_dimensions() {
        let (kind, config) = kind_and_config("azure");

        let provider = build_provider(&kind, &config, DEFAULT_BATCH_SIZE);

        assert!(matches!(provider, Provider::Azure(_)));
        assert_eq!(provider.embedded_dimensions(), 8);
    }
}
