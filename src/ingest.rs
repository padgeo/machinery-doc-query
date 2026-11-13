use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use glob::glob;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::ai::provider::EmbeddingProvider;
use crate::chunk::{Chunk, ChunkOptions, chunk_pages};
use crate::pdf::extract::extract_pages;
use crate::vector_store::qdrant::QdrantStore;

type ExtractHandle = (PathBuf, JoinHandle<ExtractOutcome>);
type ExtractOutcome = Result<ExtractResultOk, (PathBuf, String)>;
type ExtractResultOk = (PathBuf, Vec<String>);

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
    input_dir: &Path,
    store: &QdrantStore,
    collection: &str,
    provider: &P,
    batch_size: usize,
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
    if all_pdfs.is_empty() {
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
    for batch in chunks.chunks(batch_size.max(1)) {
        let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let embeddings = provider.embed(&texts).await.context("Embed failed")?;
        store
            .upsert_chunks(collection, batch, &embeddings)
            .await
            .context("Upsert chunks failed")?;
    }
    Ok(summary)
}

fn collect_pdfs(input_dir: &PathBuf) -> Result<Vec<PathBuf>, IngestError> {
    match glob(&format!("{}/**/*.pdf", input_dir.display())) {
        Ok(files) => {
            let pdfs: Vec<PathBuf> = files.into_iter().filter_map(|f| f.ok()).collect();
            if pdfs.is_empty() {
                return Err(IngestError::CollectionError(format!(
                    "No files found in dir {:?}",
                    input_dir
                )));
            }
            Ok(pdfs)
        }

        Err(e) => Err(IngestError::CollectionError(e.to_string())),
    }
}

async fn extract_pdfs(pdfs: Vec<PathBuf>) -> (Vec<ExtractResultOk>, IngestSummary) {
    // Parsing is CPU- and memory-bound, and spawn_blocking alone would run up to
    // 512 at once. A permit per core keeps memory and threads flat.
    let limit = std::thread::available_parallelism().map_or(4, |n| n.get());
    let permits = Arc::new(Semaphore::new(limit));
    let mut summary = IngestSummary::new();
    let handles = extraction_handles(pdfs, &mut summary, &permits).await;
    let all_pages = perform_extraction(handles, &mut summary).await;
    (all_pages, summary)
}

async fn extraction_handles(
    paths: Vec<PathBuf>,
    summary: &mut IngestSummary,
    permits: &Arc<Semaphore>,
) -> Vec<ExtractHandle> {
    let mut handles: Vec<ExtractHandle> = Vec::with_capacity(paths.len());
    for pdf_path in paths {
        let Ok(permit) = permits.clone().acquire_owned().await else {
            summary
                .failed
                .push((pdf_path, "extraction limiter was closed".to_string()));
            continue;
        };

        let reported_path = pdf_path.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let _permit = permit; // released when the extraction finishes or panics
            match extract_pages(&pdf_path) {
                Ok(pages) => Ok((pdf_path, pages)),
                Err(e) => Err((pdf_path, e.to_string())),
            }
        });
        handles.push((reported_path, handle));
    }
    handles
}

async fn perform_extraction(
    handles: Vec<ExtractHandle>,
    summary: &mut IngestSummary,
) -> Vec<ExtractResultOk> {
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
    all_pages
}

fn chunk_all_pages(all_pages: Vec<ExtractResultOk>, options: &ChunkOptions) -> Vec<Chunk> {
    all_pages
        .iter()
        .flat_map(|(source_file, pages)| {
            let pages_ref: Vec<&str> = pages.iter().map(String::as_str).collect();
            chunk_pages(source_file, &pages_ref, options)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ai::hash_provider::HashProvider,
        pdf::gen_fixture_pdfs::{BOILER, NO_TEXT, PUMP, TestPdfs},
    };
    use rstest::{fixture, rstest};
    use std::collections::BTreeSet;

    #[fixture]
    fn generated_test_pdfs() -> TestPdfs {
        TestPdfs::generate()
    }

    #[rstest]
    fn collect_pdfs_finds_every_generated_pdf(generated_test_pdfs: TestPdfs) {
        let pdfs = collect_pdfs(&generated_test_pdfs.dir()).expect("the directory contains pdfs");

        let found: BTreeSet<String> = pdfs
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        let expected: BTreeSet<String> = [BOILER, PUMP, NO_TEXT]
            .iter()
            .map(|name| name.to_string())
            .collect();
        assert_eq!(found, expected);
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
    #[rstest]
    async fn every_collected_pdf_can_be_extracted(generated_test_pdfs: TestPdfs) {
        let pdfs = collect_pdfs(&generated_test_pdfs.dir()).unwrap();
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
    #[rstest]
    async fn a_pdf_without_text_is_skipped_not_succeeded(generated_test_pdfs: TestPdfs) {
        let no_text = generated_test_pdfs.path(NO_TEXT);

        let (pages, summary) = extract_pdfs(vec![no_text.clone()]).await;

        assert!(pages.is_empty());
        assert!(summary.succeeded.is_empty());
        assert!(summary.failed.is_empty());
        assert_eq!(summary.skipped.len(), 1);
        assert_eq!(summary.skipped[0].0, no_text);
        assert!(summary.skipped[0].1.contains("no extractable text"));
    }

    #[tokio::test]
    #[rstest]
    async fn corrupt_pdfs_are_reported_as_failed_and_the_rest_continue(
        generated_test_pdfs: TestPdfs,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let zero_bytes = dir.path().join("zero_bytes.pdf");
        let not_a_pdf = dir.path().join("not_a_pdf.pdf");
        std::fs::write(&zero_bytes, b"").unwrap();
        std::fs::write(&not_a_pdf, b"this is not a pdf").unwrap();
        let good = generated_test_pdfs.path(BOILER);

        let (pages, summary) =
            extract_pdfs(vec![zero_bytes.clone(), good.clone(), not_a_pdf.clone()]).await;

        assert_eq!(summary.succeeded, vec![good]);
        assert_eq!(pages.len(), 1);
        let failed: BTreeSet<&PathBuf> = summary.failed.iter().map(|(p, _)| p).collect();
        assert_eq!(failed, BTreeSet::from([&zero_bytes, &not_a_pdf]));
    }

    #[tokio::test]
    #[rstest]
    async fn one_unreadable_pdf_does_not_stop_the_others(generated_test_pdfs: TestPdfs) {
        let good = generated_test_pdfs.path(BOILER);
        let missing = generated_test_pdfs.dir().join("does_not_exist.pdf");

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
    #[rstest]
    async fn run_never_touches_qdrant_when_every_file_is_skipped(generated_test_pdfs: TestPdfs) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::copy(
            generated_test_pdfs.path(NO_TEXT),
            dir.path().join("scan.pdf"),
        )
        .unwrap();
        let store = QdrantStore::connect("http://127.0.0.1:1", None).unwrap();
        let provider = HashProvider { dimensions: 8 };

        let summary = run(
            dir.path(),
            &store,
            "unused",
            &provider,
            3,
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
            1,
            &ChunkOptions::default(),
            false,
        )
        .await
        .unwrap_err();

        assert!(format!("{err:#}").contains("/definitely/not/a/real/dir"));
    }

    // A handle that is already running a closure which returns `outcome`, like the real ones.
    fn extraction_returning(path: &str, outcome: ExtractOutcome) -> ExtractHandle {
        (
            PathBuf::from(path),
            tokio::task::spawn_blocking(move || outcome),
        )
    }

    // acquire only fails on a closed semaphore. That has to fail the file, not panic.
    #[tokio::test]
    async fn a_closed_limiter_fails_the_files_instead_of_panicking() {
        let permits = Arc::new(Semaphore::new(1));
        permits.close();
        let mut summary = IngestSummary::new();

        let handles = extraction_handles(
            vec![PathBuf::from("never_read.pdf")],
            &mut summary,
            &permits,
        )
        .await;

        assert!(handles.is_empty());
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].0, PathBuf::from("never_read.pdf"));
        assert!(summary.failed[0].1.contains("limiter was closed"));
    }

    // The permit lives inside the extraction closure, so every way a file can end
    // (extracted, no text, unreadable) has to give it back.
    #[tokio::test]
    #[rstest]
    async fn every_permit_is_handed_back_once_the_extractions_finish(
        generated_test_pdfs: TestPdfs,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let not_a_pdf = dir.path().join("not_a_pdf.pdf");
        std::fs::write(&not_a_pdf, b"this is not a pdf").unwrap();
        let pdfs = vec![
            generated_test_pdfs.path(BOILER),
            generated_test_pdfs.path(NO_TEXT),
            not_a_pdf,
        ];
        let permits = Arc::new(Semaphore::new(3));
        let mut summary = IngestSummary::new();

        let handles = extraction_handles(pdfs, &mut summary, &permits).await;
        perform_extraction(handles, &mut summary).await;

        assert_eq!(permits.available_permits(), 3);
    }

    // With one permit every file waits for the previous one to hand its permit back, so a
    // permit that is never released would hang here. The timeout turns that hang into a failure.
    #[tokio::test]
    #[rstest]
    async fn a_limit_of_one_still_extracts_every_file(generated_test_pdfs: TestPdfs) {
        let dir = tempfile::tempdir().unwrap();
        let not_a_pdf = dir.path().join("not_a_pdf.pdf");
        std::fs::write(&not_a_pdf, b"this is not a pdf").unwrap();
        let pdfs = vec![
            generated_test_pdfs.path(BOILER),
            not_a_pdf.clone(),
            generated_test_pdfs.path(PUMP),
        ];
        let permits = Arc::new(Semaphore::new(1));
        let mut summary = IngestSummary::new();

        let pages = tokio::time::timeout(std::time::Duration::from_secs(20), async {
            let handles = extraction_handles(pdfs, &mut summary, &permits).await;
            perform_extraction(handles, &mut summary).await
        })
        .await
        .expect("extraction hung: a permit was never handed back");

        assert_eq!(pages.len(), 2);
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].0, not_a_pdf);
    }

    // A panic inside the parser reaches us as a JoinError. It has to come out as a failed
    // file with its name, and the other files must carry on.
    #[tokio::test]
    async fn a_panicking_extraction_is_reported_by_file_name() {
        let boom: ExtractHandle = (
            PathBuf::from("boom.pdf"),
            tokio::task::spawn_blocking(|| -> ExtractOutcome { panic!("parser blew up") }),
        );
        let fine = extraction_returning(
            "fine.pdf",
            Ok((PathBuf::from("fine.pdf"), vec!["some text".to_string()])),
        );
        let mut summary = IngestSummary::new();

        let pages = perform_extraction(vec![boom, fine], &mut summary).await;

        assert_eq!(pages.len(), 1);
        assert_eq!(summary.succeeded, vec![PathBuf::from("fine.pdf")]);
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].0, PathBuf::from("boom.pdf"));
        assert!(summary.failed[0].1.contains("extraction task panicked"));
        assert!(summary.failed[0].1.contains("parser blew up"));
    }

    #[tokio::test]
    async fn results_keep_the_input_order_even_when_the_first_file_finishes_last() {
        let slow: ExtractHandle = (
            PathBuf::from("slow.pdf"),
            tokio::task::spawn_blocking(|| -> ExtractOutcome {
                std::thread::sleep(std::time::Duration::from_millis(150));
                Ok((PathBuf::from("slow.pdf"), vec!["some text".to_string()]))
            }),
        );
        let fast = extraction_returning(
            "fast.pdf",
            Ok((PathBuf::from("fast.pdf"), vec!["some text".to_string()])),
        );
        let mut summary = IngestSummary::new();

        let pages = perform_extraction(vec![slow, fast], &mut summary).await;

        let order: Vec<PathBuf> = pages.into_iter().map(|(path, _)| path).collect();
        assert_eq!(
            order,
            vec![PathBuf::from("slow.pdf"), PathBuf::from("fast.pdf")]
        );
        assert_eq!(summary.succeeded, order);
    }

    // A blank page in the middle stays in place, so the pages after it keep their real numbers.
    #[tokio::test]
    async fn a_pdf_with_a_blank_page_is_kept_with_every_page_in_place() {
        let partly_blank = extraction_returning(
            "partly_blank.pdf",
            Ok((
                PathBuf::from("partly_blank.pdf"),
                vec![
                    "first".to_string(),
                    "  \n ".to_string(),
                    "third".to_string(),
                ],
            )),
        );
        let mut summary = IngestSummary::new();

        let pages = perform_extraction(vec![partly_blank], &mut summary).await;

        assert_eq!(summary.succeeded, vec![PathBuf::from("partly_blank.pdf")]);
        assert!(summary.skipped.is_empty());
        assert!(summary.failed.is_empty());
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].1.len(), 3);
    }
}
