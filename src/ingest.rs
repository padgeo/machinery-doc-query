use std::fmt::Debug;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use glob::glob;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::ai::provider::EmbeddingProvider;
use crate::chunk::{Chunk, ChunkOptions, chunk_pages};
use crate::pdf::extract::extract_pages;
use crate::vector_store::qdrant::QdrantStore;

type ExtractOutcome = Result<(PathBuf, Vec<String>), (PathBuf, String)>;

#[derive(Default)]
pub struct IngestSummary {
    pub succeeded: Vec<PathBuf>,
    // readable PDFs that contain no text at all (e.g. scans) - not an error, not ingested
    pub skipped: Vec<(PathBuf, String)>,
    pub failed: Vec<(PathBuf, String)>,
}

impl IngestSummary {
    pub fn new() -> Self {
        Self::default()
    }

    // A run is a success when at least one file was ingested, even if others
    // failed or were skipped. Nothing ingested at all is an error (exit code 1).
    pub fn ensure_ingested_something(&self) -> Result<()> {
        if self.succeeded.is_empty() {
            bail!(
                "no files were ingested ({} failed, {} skipped)",
                self.failed.len(),
                self.skipped.len()
            );
        }
        Ok(())
    }
}

impl Debug for IngestSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Ingest Summary:\n")?;
        f.write_str("---------------\n")?;
        f.write_str("SUCCESS:\n")?;
        let succeded_print = self
            .succeeded
            .iter()
            .map(|p| p.to_str().unwrap_or("NONE"))
            .collect::<Vec<&str>>()
            .join("\n");
        f.write_str(&succeded_print)?;
        f.write_str("\n---------------\n")?;
        f.write_str("SKIPPED:\n")?;
        let skipped_print = self
            .skipped
            .iter()
            .map(|(p, s)| format!("{}: {}", p.to_str().unwrap_or("NONE"), s))
            .collect::<Vec<String>>()
            .join("\n");
        f.write_str(&skipped_print)?;
        f.write_str("\n---------------\n")?;
        f.write_str("FAIL:\n")?;
        let failed_print = self
            .failed
            .iter()
            .map(|(p, s)| format!("{}: {}", p.to_str().unwrap_or("NONE"), s))
            .collect::<Vec<String>>()
            .join("\n");
        f.write_str(&failed_print)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum IngestError {
    #[error("Error occurring during ingest run: {0}")]
    RunError(String),
    #[error("Error occurred during PDF collection: {0}")]
    CollectionError(String),
}

pub async fn run<P: EmbeddingProvider>(
    input_dir: &PathBuf,
    store: &QdrantStore,
    collection: &str,
    provider: &P,
    chunk_options: &ChunkOptions,
    recreate: bool,
) -> Result<IngestSummary> {
    // Canonical form: "./pdfs", "pdfs/" and an absolute path all become the same
    // directory, so strip_prefix below always works and source_file (and with it
    // the Qdrant point IDs) is identical however --input-dir was spelled.
    let input_dir = input_dir
        .canonicalize()
        .with_context(|| format!("cannot read input directory {}", input_dir.display()))?;
    let all_pdfs = collect_pdfs(&input_dir).context("PDF collection failed")?;
    if all_pdfs.len() == 0 {
        return Ok(IngestSummary::new());
    }

    let (all_pages, summary) = extract_pdfs(all_pdfs).await;
    let all_pages = all_pages
        .into_iter()
        .map(|(p, pages)| {
            (
                p.strip_prefix(&input_dir).unwrap_or(&p).to_path_buf(),
                pages,
            )
        })
        .collect();
    let chunks = chunk_all_pages(all_pages, chunk_options);

    // Nothing to embed (every file was blank or unreadable): don't touch Qdrant.
    if chunks.is_empty() {
        warn!("no text could be extracted from any file, nothing to ingest");
        return Ok(summary);
    }

    let vector_size = u64::try_from(provider.embedded_dimensions()).context("Conversion error")?;
    store
        .ensure_collection(collection, vector_size, recreate)
        .await
        .context("Ensure collection failed")?;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = provider.embed(&texts).await.context("Embed failed")?;
    store
        .upsert_chunks(collection, &chunks, &embeddings)
        .await
        .context("Upsert chunks failed")?;

    Ok(summary)
}

fn collect_pdfs(input_dir: &PathBuf) -> Result<Vec<PathBuf>, IngestError> {
    match glob(&format!("{}/**/*.pdf", input_dir.display())) {
        Ok(files) => {
            let pdfs: Vec<PathBuf> = files.into_iter().filter_map(|f| f.ok()).collect();
            if pdfs.len() == 0 {
                return Err(IngestError::CollectionError(format!(
                    "No files found in dir {:?}",
                    input_dir
                )));
            }
            return Ok(pdfs);
        }

        Err(e) => Err(IngestError::CollectionError(e.to_string())),
    }
}

async fn extract_pdfs(pdfs: Vec<PathBuf>) -> (Vec<(PathBuf, Vec<String>)>, IngestSummary) {
    // The path is kept next to the handle so that even a panicking extraction
    // task can be reported by file name.
    let handles: Vec<(PathBuf, JoinHandle<ExtractOutcome>)> = pdfs
        .into_iter()
        .map(|pdf_path| {
            let reported_path = pdf_path.clone();
            let handle = tokio::task::spawn_blocking(move || match extract_pages(&pdf_path) {
                Ok(pages) => Ok((pdf_path, pages)),
                Err(e) => Err((pdf_path, e.to_string())),
            });
            (reported_path, handle)
        })
        .collect();

    let mut summary = IngestSummary::new();
    let mut all_pages = Vec::new();
    for (reported_path, handle) in handles {
        match handle.await {
            Ok(Ok((path, pages))) => {
                let blank_pages = pages.iter().filter(|p| p.trim().is_empty()).count();
                if blank_pages == pages.len() {
                    // pdf-extract has no OCR, so a scanned PDF extracts as blank pages
                    let reason =
                        "no extractable text (scanned or image-only PDF? OCR is not supported)"
                            .to_string();
                    warn!(file = %path.display(), "{reason}");
                    summary.skipped.push((path, reason));
                } else {
                    if blank_pages > 0 {
                        warn!(
                            file = %path.display(),
                            blank_pages,
                            total_pages = pages.len(),
                            "some pages have no extractable text"
                        );
                    }
                    summary.succeeded.push(path.clone());
                    all_pages.push((path, pages));
                }
            }
            Ok(Err((path, msg))) => summary.failed.push((path, msg)),
            Err(join_err) => summary.failed.push((
                reported_path,
                format!("extraction task panicked: {join_err}"),
            )),
        }
    }
    (all_pages, summary)
}

fn chunk_all_pages(all_pages: Vec<(PathBuf, Vec<String>)>, options: &ChunkOptions) -> Vec<Chunk> {
    all_pages
        .iter()
        .map(|(source_file, pages)| chunk_pages(source_file, &pages, options))
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::hash_provider::HashProvider;
    use std::collections::BTreeSet;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from("tests/fixtures/pdfs")
            .canonicalize()
            .expect("fixtures directory exists")
    }

    // Removes its directory on drop, so a failing assertion doesn't leave files behind.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!("mdq-{label}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            TempDir(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn file_names(paths: &[PathBuf]) -> BTreeSet<String> {
        paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn collect_pdfs_finds_every_fixture_pdf() {
        let pdfs = collect_pdfs(&fixtures_dir()).expect("fixtures contain pdfs");
        let expected: BTreeSet<String> = [
            "boiler_x200_manual.pdf",
            "no_text_layer.pdf",
            "pump_y50_manual.pdf",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(file_names(&pdfs), expected);
    }

    #[test]
    fn collect_pdfs_errors_when_the_directory_holds_no_pdfs() {
        let src = PathBuf::from("src").canonicalize().unwrap();
        assert!(collect_pdfs(&src).is_err());
    }

    #[test]
    fn collect_pdfs_errors_for_a_missing_directory() {
        assert!(collect_pdfs(&PathBuf::from("/definitely/not/a/real/dir")).is_err());
    }

    // Whatever collect_pdfs returns must be something extract_pdfs can actually read.
    // The fixture without any text is read fine but skipped, not counted as extracted.
    #[tokio::test]
    async fn every_collected_pdf_can_be_extracted() {
        let pdfs = collect_pdfs(&fixtures_dir()).unwrap();
        let (pages, summary) = extract_pdfs(pdfs).await;
        assert!(
            summary.failed.is_empty(),
            "extraction failed for: {:?}",
            summary.failed
        );
        assert_eq!(pages.len(), 2);
        assert_eq!(summary.skipped.len(), 1);
    }

    #[tokio::test]
    async fn a_pdf_without_text_is_skipped_not_succeeded() {
        let no_text = fixtures_dir().join("no_text_layer.pdf");

        let (pages, summary) = extract_pdfs(vec![no_text.clone()]).await;

        assert!(pages.is_empty());
        assert!(summary.succeeded.is_empty());
        assert!(summary.failed.is_empty());
        assert_eq!(summary.skipped.len(), 1);
        assert_eq!(summary.skipped[0].0, no_text);
        assert!(summary.skipped[0].1.contains("no extractable text"));
    }

    #[tokio::test]
    async fn corrupt_pdfs_are_reported_as_failed_and_the_rest_continue() {
        let dir = TempDir::new("corrupt");
        let zero_bytes = dir.0.join("zero_bytes.pdf");
        let not_a_pdf = dir.0.join("not_a_pdf.pdf");
        std::fs::write(&zero_bytes, b"").unwrap();
        std::fs::write(&not_a_pdf, b"this is not a pdf").unwrap();
        let good = fixtures_dir().join("boiler_x200_manual.pdf");

        let (pages, summary) =
            extract_pdfs(vec![zero_bytes.clone(), good.clone(), not_a_pdf.clone()]).await;

        assert_eq!(summary.succeeded, vec![good]);
        assert_eq!(pages.len(), 1);
        let failed: BTreeSet<&PathBuf> = summary.failed.iter().map(|(p, _)| p).collect();
        assert_eq!(failed, BTreeSet::from([&zero_bytes, &not_a_pdf]));
    }

    #[tokio::test]
    async fn one_unreadable_pdf_does_not_stop_the_others() {
        let good = fixtures_dir().join("boiler_x200_manual.pdf");
        let missing = fixtures_dir().join("does_not_exist.pdf");

        let (pages, summary) = extract_pdfs(vec![good.clone(), missing.clone()]).await;

        assert_eq!(pages.len(), 1);
        assert_eq!(summary.succeeded, vec![good]);
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].0, missing);
    }

    // chunk_index is per file: the point ID is "{source_file}#{chunk_index}", so each
    // file must start counting from 0 again.
    #[test]
    fn chunk_index_restarts_for_each_file() {
        let chunks = chunk_all_pages(
            vec![
                (PathBuf::from("a.pdf"), vec!["first file".to_string()]),
                (PathBuf::from("b.pdf"), vec!["second file".to_string()]),
            ],
            &ChunkOptions::default(),
        );

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].source_file, PathBuf::from("a.pdf"));
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].source_file, PathBuf::from("b.pdf"));
        assert_eq!(chunks[1].chunk_index, 0);
    }

    #[test]
    fn summary_has_a_section_for_each_outcome() {
        let summary = IngestSummary {
            succeeded: vec![PathBuf::from("a.pdf")],
            skipped: vec![(PathBuf::from("b.pdf"), "no text".to_string())],
            failed: vec![(PathBuf::from("c.pdf"), "boom".to_string())],
        };

        assert_eq!(
            format!("{summary:?}"),
            "Ingest Summary:\n---------------\nSUCCESS:\na.pdf\n---------------\nSKIPPED:\nb.pdf: no text\n---------------\nFAIL:\nc.pdf: boom"
        );
    }

    #[test]
    fn a_run_where_nothing_was_ingested_is_an_error() {
        let summary = IngestSummary {
            succeeded: vec![],
            skipped: vec![(PathBuf::from("scan.pdf"), "no text".to_string())],
            failed: vec![(PathBuf::from("bad.pdf"), "boom".to_string())],
        };

        let err = summary.ensure_ingested_something().unwrap_err();
        assert_eq!(
            err.to_string(),
            "no files were ingested (1 failed, 1 skipped)"
        );
    }

    #[test]
    fn one_ingested_file_is_enough_for_success() {
        let summary = IngestSummary {
            succeeded: vec![PathBuf::from("good.pdf")],
            skipped: vec![(PathBuf::from("scan.pdf"), "no text".to_string())],
            failed: vec![(PathBuf::from("bad.pdf"), "boom".to_string())],
        };

        assert!(summary.ensure_ingested_something().is_ok());
    }

    // Nothing listens on port 1, so any attempt to reach Qdrant would make the run fail.
    #[tokio::test]
    async fn run_never_touches_qdrant_when_every_file_is_skipped() {
        let dir = TempDir::new("only-scans");
        std::fs::copy(
            fixtures_dir().join("no_text_layer.pdf"),
            dir.0.join("scan.pdf"),
        )
        .unwrap();
        let store = QdrantStore::connect("http://127.0.0.1:1", None).unwrap();
        let provider = HashProvider { dimensions: 8 };

        let summary = run(
            &dir.0,
            &store,
            "unused",
            &provider,
            &ChunkOptions::default(),
            false,
        )
        .await
        .expect("a directory of scans is not an error by itself");

        assert!(summary.succeeded.is_empty());
        assert_eq!(summary.skipped.len(), 1);
    }

    #[tokio::test]
    async fn run_names_the_directory_when_it_cannot_be_read() {
        let store = QdrantStore::connect("http://127.0.0.1:1", None).unwrap();
        let provider = HashProvider { dimensions: 8 };
        let missing = PathBuf::from("/definitely/not/a/real/dir");

        let err = run(
            &missing,
            &store,
            "unused",
            &provider,
            &ChunkOptions::default(),
            false,
        )
        .await
        .unwrap_err();

        assert!(format!("{err:#}").contains("/definitely/not/a/real/dir"));
    }
}
