# machinery-doc-query

## Scaling and roadmap

### How ingestion scales today

- **Extraction is bounded.** At most one PDF per CPU core is parsed at a time, so memory and thread count stay flat however many PDFs a directory holds. On a 24-core machine, 240 synthetic 4.5 MB PDFs took 0.8 s and 348 MB with the limit, against 1.1 s and 1.3 GB without it, and the extracted text was identical.
- **Embedding and storing are batched.** Chunks are embedded and upserted `--batch-size` at a time (default 96), so only one batch of vectors is in memory and a failure keeps everything stored before it. Re-running `ingest` is safe: point IDs are derived from the file and chunk index, so existing points are updated in place.
- **Embedding requests are bounded and retried.** Each request to Azure OpenAI has a 60 s timeout. The OpenAI client already retries 429, 5xx and connection errors (3 retries with backoff, honouring `Retry-After`). The provider adds up to 5 attempts for what the client leaves alone: timeouts and other transport errors, such as a connection dropped mid-response. Errors that a retry cannot fix (400, 401, ...) fail straight away.
- **`--recreate-collection` survives a failing provider.** The first batch is embedded before the old collection is deleted, so a bad key or an outage leaves the old collection intact.
- **A bad file doesn't stop the run.** Every PDF ends up as succeeded, skipped (no extractable text) or failed. The exit code is 1 only when nothing was ingested.

### Known limits

- The extracted text of the whole directory is held in memory until it has been chunked.
- Re-running `ingest` embeds already-ingested files again (embeddings cost money). There is no skip-unchanged logic yet.
- If a document gets shorter, or the chunk settings change, points from the previous run can stay in the collection. `ingest --recreate-collection` rebuilds it.
- No OCR: PDFs without a text layer (scans) are skipped and reported.
- `--provider-kind hash` (`EMBEDDING_PROVIDER=hash`) runs the pipeline without an Azure key, but its vectors carry no meaning, so searching a collection built with it gives meaningless results.

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
