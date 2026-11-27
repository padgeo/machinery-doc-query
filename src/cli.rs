use crate::{
    ai::{
        azure_chat::{AzureChatProvider, AzureChatSettings},
        azure_provider::{AzureEmbeddingSettings, AzureOpenAiProvider},
        hash_provider::HashProvider,
        provider::{ChatProvider, Provider},
    },
    chunk::{ChunkOptions, RetrievedChunk},
    config::AppConfig,
    ingest,
    query::{answer_from_chunks, run},
    render::{render_answer, render_chunks},
    vector_store::qdrant::QdrantStore,
};
use anyhow::{Context, Error, Result};
use clap::{Args, Parser, Subcommand, ValueEnum, value_parser};
use std::{fmt::Display, path::PathBuf};

const DEFAULT_BATCH_SIZE: u16 = 96;
const HASH_CHAT_ERR_MSG: &str =
    "Note: the hash provider has no chat model, so the passages are shown instead of an answer.";

#[derive(ValueEnum, Clone, Debug)]
pub enum ProviderKind {
    /// Embeddings and answers from Azure OpenAI.
    Azure,
    /// Vectors made from a hash of the text, with no model: for tests and offline use. There
    /// is no chat model either, so a query prints the passages only.
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

/// Answers questions about machinery manuals with step-by-step procedures, the warnings that
/// belong to each step and the pages they come from.
#[derive(Parser, Debug)]
#[command(name = "machinery-doc-query")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Read the PDF files in a folder and store their text in the vector database, ready to
    /// be queried.
    Ingest {
        /// Folder with the PDF files to ingest. Its subfolders are searched too.
        #[arg(short, long)]
        input_dir: PathBuf,
        /// Largest size of a passage, in bytes of text.
        #[arg(short = 's', long, default_value_t = 1500)]
        chunk_size: u16,
        /// How many bytes at the end of a passage are repeated at the start of the next one.
        #[arg(short = 'o', long, default_value_t = 200)]
        chunk_overlap: u16,
        /// Delete the collection first and build it again, instead of adding to it.
        #[arg(short, long)]
        recreate_collection: bool,
        /// How many passages are sent to the embedding provider in one request.
        #[arg(short, long, default_value_t = DEFAULT_BATCH_SIZE)]
        batch_size: u16,
        /// Not implemented yet: meant to show what would be ingested without storing anything.
        #[arg(short, long)]
        dry_run: bool,
        #[command(flatten)]
        connection: ConnectionArgs,
    },
    /// Answer a question from the ingested manuals: the steps, the warnings that belong to
    /// them and the pages they come from.
    Query {
        /// The question to answer, for example "how do I drain the boiler?".
        #[arg(value_parser = parse_question)]
        question: String,
        /// How many passages to retrieve for the answer.
        #[arg(short, long, default_value_t = 5, value_parser = value_parser!(u16).range(1..))]
        top_k: u16,
        /// Print the passages that were found instead of asking the model for an answer.
        #[arg(long)]
        retrieval_only: bool,
        #[command(flatten)]
        connection: ConnectionArgs,
    },
    /// Show the settings the other commands would use. The keys are hidden.
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
    /// Azure OpenAI endpoint, for example https://my-resource.openai.azure.com.
    #[arg(long, env = "AZURE_OPENAI_ENDPOINT")]
    pub endpoint: Option<String>,
    /// Azure OpenAI API key.
    #[arg(long, env = "AZURE_OPENAI_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Azure OpenAI API version.
    #[arg(
        long,
        env = "AZURE_OPENAI_API_VERSION",
        default_value_t = String::from("2024-08-01-preview")
    )]
    pub api_version: String,
    /// Azure OpenAI deployment that makes the embeddings.
    #[arg(long, env = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")]
    pub embedding_deployment: Option<String>,
    /// Azure OpenAI deployment that writes the answers.
    #[arg(long, env = "AZURE_OPENAI_CHAT_DEPLOYMENT")]
    pub chat_deployment: Option<String>,
    /// Address of the Qdrant server, on its gRPC port.
    #[arg(
        long,
        env = "QDRANT_URL",
        default_value_t = String::from("http://localhost:6334")
    )]
    pub qdrant_url: String,
    /// API key of the Qdrant server, if it asks for one.
    #[arg(long, env = "QDRANT_API_KEY", hide_env_values = true)]
    pub qdrant_api_key: Option<String>,
    /// Qdrant collection that holds the passages.
    #[arg(
        long,
        env = "QDRANT_COLLECTION",
        default_value_t = String::from("machinery_docs")
    )]
    pub qdrant_collection: String,
    /// Length of the vectors the embedding model makes. Use the same value to ingest and to
    /// query, because the collection is created with it.
    #[arg(long, env = "EMBEDDING_DIMENSIONS", default_value_t = 1536)]
    pub embedding_dimensions: u16,
    /// Which provider to use. Query with the provider that ingested the documents, because
    /// vectors from different providers cannot be compared.
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
            retrieval_only,
        } => process_query(question, top_k, connection, retrieval_only).await,
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

async fn process_query(
    question: String,
    top_k: u16,
    connection: ConnectionArgs,
    retrieval_only: bool,
) -> Result<()> {
    let provider_kind = connection.provider_kind.clone();
    let app_config = AppConfig::try_from(connection)?;
    let store = connect_to_qdrant(&app_config).await?;
    let provider = build_provider(&provider_kind, &app_config, DEFAULT_BATCH_SIZE);
    let collection = &app_config.qdrant_collection;
    let chunks = run(&question, &store, collection, &provider, top_k as u64).await?;
    let chat = AzureChatProvider::new(AzureChatSettings::from(&app_config));
    let output = query_output(&question, &chunks, &provider_kind, retrieval_only, &chat).await?;
    if let Some(note) = output.stderr {
        eprintln!("{}", note);
    }
    println!("{}", output.stdout);
    Ok(())
}

/// What a query prints: the text for stdout and, when there is something to explain, a note
/// for stderr. The note is kept off stdout so that stdout is the answer, or the passages,
/// alone.
#[derive(Debug, PartialEq)]
struct QueryOutput {
    stdout: String,
    stderr: Option<String>,
}

/// Decides what a query prints, once its passages are retrieved.
///
/// The model is asked, and its answer is the output, unless it cannot be or was not asked
/// for: the hash provider has no chat model (a note on stderr says so), and
/// `--retrieval-only` asks for the passages. The passages are printed the way the query
/// command did before there was a model: a header naming the provider, then the results.
async fn query_output<C: ChatProvider>(
    question: &str,
    chunks: &[RetrievedChunk],
    provider_kind: &ProviderKind,
    retrieval_only: bool,
    chat: &C,
) -> Result<QueryOutput> {
    match (retrieval_only, provider_kind) {
        (true, _) => Ok(QueryOutput {
            stdout: format!("Provider: {}\n{}", provider_kind, render_chunks(chunks)),
            stderr: None,
        }),
        (false, ProviderKind::Azure) => {
            let reply = answer_from_chunks(question, chunks, chat).await?;
            let render = render_answer(&reply);
            Ok(QueryOutput {
                stdout: render,
                stderr: None,
            })
        }
        (false, ProviderKind::Hash) => Ok(QueryOutput {
            stdout: format!("Provider: hash\n{}", render_chunks(chunks)),
            stderr: Some(HASH_CHAT_ERR_MSG.to_owned()),
        }),
    }
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
    use crate::{
        ai::{
            prompt::Prompt,
            provider::{EmbeddingProvider, ProviderError},
        },
        answer::{Answer, AnswerStep, Source},
        chunk::Chunk,
    };
    use clap::{CommandFactory, error::ErrorKind};
    use rstest::rstest;
    use serde_json::{Value, json};
    use std::sync::Mutex;

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

    // ---- --retrieval-only ----

    #[test]
    fn query_asks_the_model_unless_retrieval_only_is_given() {
        let cli = parse(&["query", "a question"]).unwrap();

        let Commands::Query { retrieval_only, .. } = cli.command else {
            panic!("expected the query command");
        };
        assert!(!retrieval_only);
    }

    #[test]
    fn query_accepts_retrieval_only() {
        let cli = parse(&["query", "a question", "--retrieval-only"]).unwrap();

        let Commands::Query { retrieval_only, .. } = cli.command else {
            panic!("expected the query command");
        };
        assert!(retrieval_only);
    }

    #[test]
    fn retrieval_only_goes_with_the_other_query_flags() {
        let cli = parse(&[
            "query",
            "--retrieval-only",
            "a question",
            "--top-k",
            "3",
            "--provider-kind",
            "hash",
        ])
        .unwrap();

        let Commands::Query {
            question,
            top_k,
            retrieval_only,
            connection,
        } = cli.command
        else {
            panic!("expected the query command");
        };
        assert_eq!(question, "a question");
        assert_eq!(top_k, 3);
        assert!(retrieval_only);
        assert!(matches!(connection.provider_kind, ProviderKind::Hash));
    }

    // The flag is about the model, and only query has one. ConnectionArgs is shared by all
    // three commands, so a flag declared there would be accepted by ingest and doctor too.
    #[rstest]
    #[case::ingest(&["ingest", "-i", "pdfs"])]
    #[case::doctor(&["doctor"])]
    fn only_query_has_the_retrieval_only_flag(#[case] args: &[&str]) {
        let mut args = args.to_vec();
        args.push("--retrieval-only");

        let err = parse(&args).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    }

    // --help is where a user learns what the flag does.
    #[test]
    fn the_retrieval_only_flag_is_explained_in_the_help() {
        let command = Cli::command();
        let query = command.find_subcommand("query").unwrap();
        let flag = query
            .get_arguments()
            .find(|arg| arg.get_id().as_str() == "retrieval_only")
            .expect("query has the flag");

        let help = flag.get_help().map(|help| help.to_string());

        assert!(help.is_some_and(|help| !help.trim().is_empty()));
    }

    // ---- what a query prints ----

    // A chat model that gives the one reply it was prepared with and counts how often it
    // was asked. A second question gets NoAnswer.
    struct StubChat {
        reply: Mutex<Option<Result<String, ProviderError>>>,
        asked: Mutex<usize>,
    }

    impl StubChat {
        fn times_asked(&self) -> usize {
            *self.asked.lock().unwrap()
        }
    }

    impl ChatProvider for StubChat {
        async fn complete(
            &self,
            _prompt: &Prompt,
            _schema_name: &str,
            _schema: &Value,
        ) -> Result<String, ProviderError> {
            *self.asked.lock().unwrap() += 1;
            self.reply
                .lock()
                .unwrap()
                .take()
                .unwrap_or(Err(ProviderError::NoAnswer))
        }
    }

    fn replies_with(raw: &str) -> StubChat {
        StubChat {
            reply: Mutex::new(Some(Ok(raw.to_owned()))),
            asked: Mutex::new(0),
        }
    }

    fn fails_with(error: ProviderError) -> StubChat {
        StubChat {
            reply: Mutex::new(Some(Err(error))),
            asked: Mutex::new(0),
        }
    }

    // A model that must not be asked: if it is, its failure ends the test.
    fn not_to_be_asked() -> StubChat {
        fails_with(ProviderError::RequestFailed(
            "the model was asked".to_string(),
        ))
    }

    fn passage(file: &str, page: usize, text: &str) -> RetrievedChunk {
        RetrievedChunk {
            chunk: Chunk {
                source_file: PathBuf::from(file),
                page_number: page,
                chunk_index: 0,
                text: text.to_owned(),
            },
            score: 0.5,
        }
    }

    // The reply of a model that follows the schema: one step, cited to passage 1.
    fn one_step_reply() -> String {
        json!({
            "summary": "Drain the boiler.",
            "unable_to_answer": false,
            "steps": [{
                "instruction": "Open the valve.",
                "warnings": ["WARNING: hot."],
                "source_indices": [1],
            }],
        })
        .to_string()
    }

    // The answer that reply makes when passage 1 is a.pdf page 1.
    fn one_step_answer() -> Answer {
        Answer::Steps {
            summary: "Drain the boiler.".to_string(),
            steps: vec![AnswerStep {
                instruction: "Open the valve.".to_string(),
                warnings: vec!["WARNING: hot.".to_string()],
                sources: vec![1],
            }],
            sources: vec![Source {
                file: PathBuf::from("a.pdf"),
                page: 1,
            }],
        }
    }

    // With a chat model, what is printed is the answer and nothing but the answer.
    #[tokio::test]
    async fn the_answer_is_what_a_query_prints_when_the_model_can_be_asked() {
        let chunks = vec![passage("a.pdf", 1, "Open the valve.")];
        let chat = replies_with(&one_step_reply());

        let output = query_output(
            "how do I drain the boiler?",
            &chunks,
            &ProviderKind::Azure,
            false,
            &chat,
        )
        .await
        .unwrap();

        assert_eq!(
            output,
            QueryOutput {
                stdout: render_answer(&one_step_answer()),
                stderr: None,
            }
        );
        assert_eq!(chat.times_asked(), 1);
    }

    // --retrieval-only prints the passages the way query did before there was a model, and
    // the model is not asked: that would cost a request nobody wants.
    #[tokio::test]
    async fn retrieval_only_prints_the_passages_without_asking_the_model() {
        let chunks = vec![passage("a.pdf", 1, "Open the valve.")];
        let chat = not_to_be_asked();

        let output = query_output("question", &chunks, &ProviderKind::Azure, true, &chat)
            .await
            .unwrap();

        assert_eq!(
            output,
            QueryOutput {
                stdout: format!("Provider: azure\n{}", render_chunks(&chunks)),
                stderr: None,
            }
        );
        assert_eq!(chat.times_asked(), 0);
    }

    // The hash provider only makes vectors, so there is no model to ask. The passages are
    // printed as with --retrieval-only, and a line on stderr says why there is no answer.
    // It is on stderr so that stdout stays the passages alone.
    #[tokio::test]
    async fn the_hash_provider_prints_the_passages_and_a_note_on_stderr() {
        let chunks = vec![passage("a.pdf", 1, "Open the valve.")];
        let chat = not_to_be_asked();

        let output = query_output("question", &chunks, &ProviderKind::Hash, false, &chat)
            .await
            .unwrap();

        assert_eq!(
            output.stdout,
            format!("Provider: hash\n{}", render_chunks(&chunks))
        );
        let note = output.stderr.expect("a note on stderr");
        assert!(note.contains("hash"), "got: {note:?}");
        assert!(
            !note.trim().is_empty() && !note.trim().contains('\n'),
            "got: {note:?}"
        );
        assert_eq!(chat.times_asked(), 0);
    }

    // Asking for the passages is explanation enough.
    #[tokio::test]
    async fn retrieval_only_with_the_hash_provider_needs_no_note() {
        let chunks = vec![passage("a.pdf", 1, "Open the valve.")];
        let chat = not_to_be_asked();

        let output = query_output("question", &chunks, &ProviderKind::Hash, true, &chat)
            .await
            .unwrap();

        assert_eq!(
            output,
            QueryOutput {
                stdout: format!("Provider: hash\n{}", render_chunks(&chunks)),
                stderr: None,
            }
        );
        assert_eq!(chat.times_asked(), 0);
    }

    // Saying that it cannot answer is an answer: it is printed, and the command succeeds.
    #[tokio::test]
    async fn a_model_that_cannot_answer_is_printed_as_the_output() {
        let chunks = vec![passage("a.pdf", 1, "Open the valve.")];
        let reply = json!({
            "summary": "The passages do not cover the pump.",
            "unable_to_answer": true,
            "steps": [],
        })
        .to_string();

        let output = query_output(
            "question",
            &chunks,
            &ProviderKind::Azure,
            false,
            &replies_with(&reply),
        )
        .await
        .unwrap();

        assert_eq!(
            output,
            QueryOutput {
                stdout: "Unable to answer: The passages do not cover the pump.".to_string(),
                stderr: None,
            }
        );
    }

    // A failed request is the command's error, so nothing is printed on stdout and the exit
    // code and stderr tell what happened.
    #[tokio::test]
    async fn a_failed_request_is_an_error_and_not_an_output() {
        let chunks = vec![passage("a.pdf", 1, "Open the valve.")];
        let chat = fails_with(ProviderError::Refused(
            "I cannot help with that.".to_string(),
        ));

        let err = query_output("question", &chunks, &ProviderKind::Azure, false, &chat)
            .await
            .unwrap_err();

        let chain = format!("{err:#}");
        assert!(chain.contains("I cannot help with that."), "got {chain}");
    }

    // Nothing found is a normal result. With a model it says that it cannot answer, and the
    // model is not asked about nothing.
    #[tokio::test]
    async fn with_no_passages_the_answer_says_it_cannot_answer_and_the_model_is_not_asked() {
        let chat = not_to_be_asked();

        let output = query_output("question", &[], &ProviderKind::Azure, false, &chat)
            .await
            .unwrap();

        assert!(
            output.stdout.starts_with("Unable to answer: "),
            "got: {}",
            output.stdout
        );
        assert_eq!(output.stderr, None);
        assert_eq!(chat.times_asked(), 0);
    }

    #[tokio::test]
    async fn with_no_passages_the_passages_output_is_the_usual_sentence() {
        let chat = not_to_be_asked();

        let output = query_output("question", &[], &ProviderKind::Hash, true, &chat)
            .await
            .unwrap();

        assert_eq!(output.stdout, "Provider: hash\nNo matching passages found.");
    }
}
