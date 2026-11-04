use crate::{
    ai::hash_provider::HashProvider, chunk::ChunkOptions, config::AppConfig, ingest,
    vector_store::qdrant::QdrantStore,
};
use anyhow::{Context, Error, Result};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

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
    // these will be used in the future
    #[allow(unused)] batch_size: u16,
    #[allow(unused)] dry_run: bool,
    connection: ConnectionArgs,
) -> Result<(), Error> {
    let app_config = AppConfig::try_from(connection)?;
    let store = QdrantStore::connect(
        app_config.qdrant_url.as_str(),
        app_config.qdrant_api_key.as_deref(),
    )
    .context("QdrantStore error occurred")?;
    let collection = app_config.qdrant_collection;
    // clippy note: converting `u16` to `usize` cannot fail
    let dimensions = usize::from(app_config.embedding_dimensions);
    let provider = HashProvider { dimensions };

    // clippy note: converting `u16` to `usize` cannot fail
    let chunk_size = usize::from(chunk_size);
    let overlap = usize::from(chunk_overlap);
    let chunk_options = ChunkOptions {
        chunk_size,
        overlap,
    };
    let recreate = recreate_collection;

    let summary = ingest::run::<HashProvider>(
        &input_dir,
        &store,
        &collection,
        &provider,
        &chunk_options,
        recreate,
    )
    .await
    .context("Ingest Error")?;
    println!("{:?}", summary);
    summary.ensure_ingested_something()?;
    Ok(())
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
    use clap::error::ErrorKind;

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
}
