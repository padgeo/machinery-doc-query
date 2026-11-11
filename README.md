# machinery-doc-query

## Scaling and roadmap

### How ingestion scales today

- **Extraction is bounded.** At most one PDF per CPU core is parsed at a time, so memory and thread count stay flat however many PDFs a directory holds. On a 24-core machine, 240 synthetic 4.5 MB PDFs took 0.8 s and 348 MB with the limit, against 1.1 s and 1.3 GB without it, and the extracted text was identical.
- **Embedding and storing are batched.** Chunks are embedded and upserted `--batch-size` at a time (default 96), so only one batch of vectors is in memory, progress is visible per batch, and a failure keeps everything stored before it. Re-running `ingest` is safe: point IDs are derived from the file and chunk index, so existing points are updated in place.
- **A bad file doesn't stop the run.** Every PDF ends up as succeeded, skipped (no extractable text) or failed. The exit code is 1 only when nothing was ingested.

### Known limits

- The extracted text of the whole directory is held in memory until it has been chunked.
- Re-running `ingest` embeds already-ingested files again (embeddings cost money). There is no skip-unchanged logic yet.
- If a document gets shorter, or the chunk settings change, points from the previous run can stay in the collection. `ingest --recreate-collection` rebuilds it.
- No OCR: PDFs without a text layer (scans) are skipped and reported.

### Roadmap

Planned after the core `ingest` and `query` commands are complete:

1. **Streaming ingest.** Take each document through fetch, extract, chunk, embed and store as soon as it is ready, with bounded concurrency per stage, instead of finishing each stage for every file first.
2. **Incremental ingest.** Replace a document's old points when it changes, and skip documents that haven't.
3. **Remote sources.** Start with Azure Blob Storage, because it keeps documents inside the Azure tenant as the project brief requires. Other sources such as Google Drive need a data-residency check first.

```text
today    all PDFs -> extract (bounded) -> chunk everything -> embed + store, batch by batch
planned  each document -> fetch -> extract -> chunk -> embed + store -> next document
```

Streaming isn't built yet on purpose. The project scope is a local directory, the batch pipeline handles that scale comfortably (numbers above), and a streaming design adds concurrency complexity that is best added together with the first remote source. The detailed reasoning is in [plan.md](plan.md#post-m7-roadmap-ingest-scaling-and-remote-sources).
