# Fexillon Machinery Doc Query CLI — Implementation Plan

## Context

Fexillon technicians currently search hundreds of machinery PDFs by hand to find procedures, risking missed safety/business-impact warnings. This project builds a Rust CLI that ingests a directory of PDFs into a local Qdrant vector store (via Azure OpenAI embeddings) and answers natural-language procedure questions with step-by-step, source-cited, inline-warning answers generated via Azure OpenAI — all data staying inside Fexillon's Azure tenant and local Docker infrastructure.

**Current repo state:** greenfield. `Cargo.toml` has the 6 PRD-specified dependencies declared (no feature flags set yet); `src/main.rs` is the unmodified `cargo new` "Hello, world!" stub; nothing else exists (no config, no Docker, no tests, no CI). Only a 1-line `README.md` is actually committed to git — `Cargo.toml`, `Cargo.lock`, `.gitignore`, and `src/` are untracked, on-disk work.

**Decisions confirmed with the user:**
- Add well-justified supporting crates beyond the PRD's 6 (error handling, config, retries, testing) rather than hand-rolling everything.
- Include a GitHub Actions CI workflow running the full test suite (Docker is available by default on GitHub-hosted runners, so the testcontainers-based Qdrant tests work unmodified).

Everything below reflects those two decisions and is otherwise based on verified behavior of the exact pinned crate versions (checked against docs.rs/crates.io/Microsoft Learn — not assumed).

---

## Architecture

### Module layout

Lib+bin split — `src/lib.rs` holds all real logic (testable without spawning a process); `src/main.rs` is a thin shell that parses CLI args, initializes tracing, dispatches, and maps errors to exit codes.

```
Cargo.toml
docker-compose.yml
.env.example
.github/workflows/ci.yml
src/
  main.rs
  lib.rs                   — `pub mod` wiring
  cli.rs                   — clap derive: Cli, Commands{Ingest,Query,Doctor}, ConnectionArgs (flattened)
  config.rs                — AppConfig, ConfigError, validation, redacted Debug impl
  chunk.rs                 — Chunk, RetrievedChunk, chunking algorithm, ChunkOptions
  pdf/
    mod.rs
    extract.rs              — wraps pdf-extract, PdfExtractionError
  vector_store/
    mod.rs
    qdrant.rs                — QdrantStore (connect / create-or-validate collection / upsert / search), deterministic point-ID derivation, QdrantStoreError
  ai/
    mod.rs
    provider.rs               — EmbeddingProvider + ChatProvider traits (native async-fn-in-trait), ProviderError
    azure_openai.rs            — AzureOpenAiProvider (two Client<AzureConfig> instances: embedding + chat deployments), retry/backoff wrapper
    schema.rs                  — StructuredAnswer/Step/Source types + the JSON Schema constant for response_format
    prompt.rs                  — system/user message construction
  ingest.rs                 — orchestration: walk dir → extract → chunk → batch-embed → upsert; IngestSummary (partial-failure tolerant)
  query.rs                   — orchestration: embed question → Qdrant search → build prompt → chat → StructuredAnswer
  render.rs                   — StructuredAnswer → CLI text (pure function, no AI/network knowledge)
  doctor.rs                    — connectivity-check command
tests/
  fixtures/pdfs/                — checked-in tiny generated PDFs
  fixtures/azure_openai/         — canned JSON response bodies for wiremock
  common/mod.rs                   — shared helpers: spawn Qdrant testcontainer, spawn wiremock server, build Command with env
  qdrant_integration.rs
  azure_openai_mock.rs
  cli_e2e.rs
examples/
  gen_fixture_pdfs.rs          — one-off generator for tests/fixtures/pdfs/*.pdf (not run by `cargo test`)
```

**Error handling:** modules below the orchestration layer (`config`, `pdf::extract`, `vector_store::qdrant`, `ai::provider`) use `thiserror`-derived enums with precise, matchable variants. `ingest.rs`/`query.rs`/`doctor.rs`/`main.rs` work in `anyhow::Result<T>` with `.context(...)`. `main()` prints `eprintln!("Error: {e:#}")` (full context chain) and exits 1 — no panics/backtraces for expected failure modes.

**Why `vector_store::qdrant` is a concrete struct but `ai::provider` is a trait:** the PRD's swappability requirement is scoped explicitly to the AI provider ("only Azure OpenAI is implemented now" but "architecture should allow future swappability"); Qdrant-via-Docker is a hard constraint, not something meant to flex. Both provider traits use native `async fn` in traits (stable in current Rust, no `async-trait` crate needed) since the concrete provider is chosen once at the composition root in `main.rs` — no `dyn` trait objects needed.

### Config & secrets

Precedence, entirely via clap's native mechanisms (no custom precedence code): `.env` file (via `dotenvy::dotenv().ok()`, non-fatal if absent) → populates process env → clap's `env` feature reads env vars as flag fallbacks → explicit CLI flags win.

Connection settings live in one flattened `ConnectionArgs` struct shared by `ingest`/`query`/`doctor` (endpoint, API key, API version, embedding deployment, chat deployment, Qdrant URL/API key/collection, embedding dimensions). Required fields are `Option<String>` at the clap layer on purpose — a separate `AppConfig::try_from(ConnectionArgs) -> Result<AppConfig, ConfigError>` validation step produces actionable errors ("set AZURE_OPENAI_API_KEY, add it to .env, or pass --azure-openai-api-key") instead of clap's generic "required argument missing." `url` crate validates the two URLs at config-load time. `AppConfig` gets a hand-written `Debug` impl redacting both API keys.

`AZURE_OPENAI_API_VERSION` defaults to `2024-08-01-preview` — verified as the first version supporting `response_format: json_schema`, which the response-generation design (below) depends on.

`.env.example` is checked in (documents every var, including that Azure "deployment name" ≠ model name); `.env` itself is gitignored.

### Chunking

Input comes from `pdf_extract::extract_text_by_pages` (page-tagged, needed for citation). Default **1500 chars/chunk, 200 char overlap**, character- not token-based (avoids a BPE-tokenizer dependency; 1500 chars ≈ 300–400 tokens, well under embedding input limits). Boundary heuristic scans backward from the target cut point for a paragraph break, then a sentence end, else hard-cuts — specifically so a `WARNING:`/`CAUTION:` line doesn't get severed from the step it modifies. Cuts happen on `char` boundaries (never byte offsets) so multi-byte characters (`°`, `µ`) can't cause a panic. Blank/image-only pages are skipped and logged via `tracing::warn!` rather than silently producing zero chunks.

Payload metadata per chunk: `source_file` (path relative to `--input-dir`, for portability), `page_number`, `chunk_index` (monotonic per file), `text`, `ingested_at`.

### Qdrant collection schema

- Vector size: config-driven, default **1536**, paired with **`text-embedding-3-small`** as the recommended embedding deployment (better quality than legacy `ada-002` at the same dimensionality/cost; `-3-large`'s 3072 dims is unjustified for this scope). Nothing in the code hardcodes 1536.
- Distance metric: `Cosine` (standard for OpenAI-family embeddings).
- **Point IDs are deterministic, not random:** `Uuid::new_v5(NAMESPACE, "{source_file}#{chunk_index}")`. This is required, not stylistic — Qdrant only accepts u64 or UUID-format point IDs, so an arbitrary hash string would be rejected. Deterministic IDs mean re-running `ingest` on the same directory **upserts in place instead of duplicating points**.
- Collection creation is idempotent: create if absent; if present, validate its vector size matches configured `embedding_dimensions` and fail with a clear `VectorSizeMismatch` error before attempting any upsert.
- Known, deliberately-unsolved limitation: changing chunk size/overlap between runs can change a file's chunk count and orphan stale points from a previous run. `ingest --recreate-collection` (drop + recreate) is the documented escape hatch rather than building point-diffing/GC.
- Batched upsert (100 points/call), not one call per point.

### AI response generation

**Decision: Azure OpenAI structured outputs (`response_format: json_schema`, `strict: true`), not function/tool calling.** Verified available in `async-openai` 0.41.3 (`types::chat::ResponseFormatJsonSchema`) and Azure OpenAI (API version ≥ `2024-08-01-preview`, `gpt-4o-2024-08-06`+/`gpt-4.1`/`gpt-5` family). Reasoning:
1. We want the model's answer to *already be* the structured object — tool calling is semantically for "the model decides to invoke an action," the wrong fit here.
2. Strict mode uses constrained decoding: the response is guaranteed schema-valid, eliminating brittle prompt-engineered-Markdown parsing.
3. Modeling `warnings: string[]` as a field **on each step object** (not a separate top-level list) makes "warnings woven into the relevant step" — the PRD's explicit requirement — a structural guarantee the renderer just reads off.
4. Tradeoff: Azure's strict-mode schema subset forbids `additionalProperties: true` (must be `false` everywhere), requires every property in `required`, and forbids string/number/array validation keywords. The schema is designed around these constraints from the start, with a unit test (below) guarding against future violations.

Schema shape (`ai/schema.rs`, a `serde_json::Value` constant — no `schemars` needed at this size): top-level `{ summary, unable_to_answer, steps[], sources[] }`; each step has `{ step_number, instruction, warnings[], source_indices[] }`; each source has `{ index, source_file, page_number }`.

`unable_to_answer` (with `steps: []`) is a deliberate safety feature — the system prompt instructs the model to use it rather than hallucinate a procedure when retrieved chunks don't cover the question, directly serving the PRD's concern about missing critical warnings via a confidently-wrong answer. The renderer never trusts the model's `step_number` for ordering — always uses array order and renumbers locally.

Two `Client<AzureConfig>` instances are required (not one) — `AzureConfig` binds one deployment ID per client and Azure's URL scheme is per-deployment, so `AzureOpenAiProvider` holds a separate `embedding_client` and `chat_client`, both built from the same endpoint/key/api-version with different `.with_deployment_id(...)`.

Prompt construction (`ai/prompt.rs`) is a pure function of `(question, &[RetrievedChunk])`, numbering the chunks in the user message (`[1] (file.pdf, p.12): "..."`) — the numbers `source_indices`/`sources.index` refer back to, so `query.rs` can defensively drop any indices the model returns that don't map to a chunk actually sent.

### CLI commands

| Command | Key args | Purpose |
|---|---|---|
| `ingest` | `--input-dir` (required), `[ConnectionArgs]`, `--chunk-size` (1500), `--chunk-overlap` (200), `--recreate-collection`, `--batch-size` (96), `--dry-run` | Batch ingestion pipeline |
| `query` | `question` (positional), `[ConnectionArgs]`, `--top-k` (5) | Semantic search + generated answer |
| `doctor` | `[ConnectionArgs]` | Validates config, then checks Qdrant + Azure OpenAI connectivity with minimal calls |

`doctor` is the one addition beyond the PRD's required two — justified because it's cheap (reuses the same connection-establishing code `ingest`/`query` already need) and directly useful onboarding tooling ("is Docker up? is my `.env` right?"), and the PRD itself says "at minimum two commands." Easy to cut if it turns out unwanted.

### Docker Compose (local dev)

`qdrant/qdrant:v1.19.0` (confirmed to exist on Docker Hub, matches the pinned client version), ports 6333 (REST/dashboard) and 6334 (gRPC — what `qdrant-client` actually talks to), persisted via a named volume.

### CI (GitHub Actions)

`.github/workflows/ci.yml` running `cargo fmt --check`, `cargo clippy`, and `cargo test` (unit + integration, including the testcontainers-based Qdrant tests) on push/PR. GitHub-hosted runners have Docker available by default, so no special setup is needed for testcontainers. No real Azure credentials are ever used in CI — all AI calls in tests go through `wiremock`.

---

## Testing strategy

**Unit tests** (co-located `#[cfg(test)]`):
- `chunk.rs` — boundary/overlap math, paragraph/sentence-boundary heuristic keeps a `WARNING:` line intact, UTF-8 multi-byte safety, empty input, chunk/page indexing across multiple pages.
- `config.rs` — every required field missing individually → correct `ConfigError` variant; malformed URLs; defaults. `AppConfig::try_from` takes an explicit struct rather than reading `std::env` directly, which is what keeps these tests deterministic and parallel-safe.
- `cli.rs` — `Cli::try_parse_from([...])` for flag/default/rejection coverage without spawning a subprocess.
- `pdf/extract.rs` — against checked-in fixtures: known substrings, correct page boundaries, missing file → `IoError`, a no-text-layer fixture returns empty text rather than erroring (turns the "no OCR" limitation into tested, documented behavior).
- `ai/prompt.rs` — chunk numbering/inclusion correctness, plus a schema self-check test walking the JSON Schema constant asserting `additionalProperties: false` and full `required` coverage everywhere (regression guard against violating Azure's strict-mode subset).
- `ai/schema.rs` — deserializes canned JSON (zero/multiple warnings, `unable_to_answer`, multiple sources) into `StructuredAnswer`.
- `render.rs` — exact-string assertions for representative `StructuredAnswer` fixtures.
- `vector_store/qdrant.rs` (non-networked parts) — deterministic point-ID derivation properties.

**Qdrant integration tests** (`tests/qdrant_integration.rs`) — `testcontainers` crate with a hand-configured `GenericImage::new("qdrant/qdrant", "v1.19.0")` (no ready-made `testcontainers-modules` Qdrant module exists). One shared container per test binary (lazy `OnceCell`), each test uses its own collection. Hand-crafted fake `Vec<f32>` embeddings — this layer never touches Azure. Covers: idempotent collection creation, upsert-then-search, re-upsert-updates-in-place (proves the re-ingestion design), vector-size-mismatch error, top-k respected.

**Azure OpenAI integration tests** (`tests/azure_openai_mock.rs`) — `wiremock`, pointed to via `AzureConfig::with_api_base(&mock_server.uri())`; never hits real Azure. Fixtures include a normal embeddings response, a chat-completion response whose `content` is a JSON-encoded string matching our schema (the nested-JSON-in-a-string detail that's easy to get wrong), a refusal variant, a malformed-content variant, and 429/500 fixtures to test the `backoff` retry wrapper. One test asserts the actual outgoing request included `response_format: json_schema` and the Azure `api-key` header — testing that structured outputs are invoked correctly, not just that responses parse.

**End-to-end CLI tests** (`tests/cli_e2e.rs`) — `assert_cmd` + `predicates` against the real compiled binary, wired to the testcontainers Qdrant + wiremock Azure. Covers: `ingest` happy path (verified independently against Qdrant), `ingest` run twice → point count unchanged, `query` happy path asserting step text **+ warning text + source filename** together (the single test tying the whole pipeline to the PRD's actual success criteria), no-match query → graceful message not a crash, missing/invalid config → clean non-zero exit, `--help` smoke test, `doctor` happy/failure paths, and a partial-failure ingest (one good + one corrupt file → good file still ingested, bad file reported, exit code 0).

**Partial-failure design:** `ingest.rs` collects an `IngestSummary { succeeded, failed }` rather than aborting on the first bad PDF. Exit code is 0 if at least one file succeeded, 1 only if zero succeeded or a hard infra error occurred. PDF extraction runs inside `tokio::task::spawn_blocking` since `pdf-extract` is synchronous/CPU-bound.

**Fixture PDFs**: `examples/gen_fixture_pdfs.rs` (uses `printpdf`, run manually via `cargo run --example gen_fixture_pdfs`, never by `cargo test`) generates small PDFs once; the resulting files are checked into `tests/fixtures/pdfs/` for deterministic tests. Three fixtures: a 2-page manual with `WARNING:`/`CAUTION:` lines embedded at specific steps, a second single-page manual with distinct content (tests multi-document citation doesn't cross-contaminate), and a no-extractable-text (image-only) fixture.

---

## Dependencies

```toml
[dependencies]
clap = { version = "4.6.6", features = ["derive", "env"] }
tokio = { version = "1.53.1", features = ["macros", "rt-multi-thread"] }
serde = { version = "1.0.229", features = ["derive"] }
pdf-extract = "0.12.0"
async-openai = { version = "0.41.3", features = ["chat-completion", "embedding"] }
qdrant-client = { version = "1.19.0", features = ["uuid"] }
serde_json = "1.0.151"
uuid = { version = "1.26.0", features = ["v5", "serde"] }
dotenvy = "0.15.7"
url = "2.5.8"
anyhow = "1.0.104"
thiserror = "2.0.20"
backoff = { version = "0.4.0", features = ["tokio"] }
tracing = "0.1.44"
tracing-subscriber = { version = "0.3.23", features = ["env-filter"] }

[dev-dependencies]
wiremock = "0.6.5"
testcontainers = "0.28.0"
assert_cmd = "2.2.2"
predicates = "3.1.4"
printpdf = "0.12.7"
```

`async-openai`'s default features already include `rustls`; it ships with **no** chat/embedding endpoints enabled by default, so `chat-completion`/`embedding` must be explicit. `clap`'s default features don't include `derive`/`env` either — both required.

---

## Build order

Each milestone compiles and has passing tests — no big-bang integration at the end.

1. **M0 — Scaffolding**: full `Cargo.toml` with feature flags above, `.env.example`, `docker-compose.yml`, empty `src/lib.rs`/`main.rs` skeleton, commit the already-generated `Cargo.lock` (application binaries should pin exact transitive versions in git). `.github/workflows/ci.yml` added here too, even though there's nothing to test yet beyond `cargo check`.
   *Verify: `cargo check` passes; CI workflow runs green on an empty test suite.*

2. **M1 — CLI + config**: `cli.rs` + `config.rs` fully real; subcommand bodies still stubs.
   *Tests: all `ConfigError` variants, `Cli::try_parse_from` flag/default coverage, first E2E test (missing config → clean non-zero exit).*

3. **M2 — PDF extraction + chunking**: `pdf/extract.rs` + `chunk.rs`; fixture PDFs generated and checked in here.
   *Tests: extraction + chunking unit tests against fixtures, including the no-text-layer case.*

4. **M3 — Qdrant + ingest plumbing**: `vector_store/qdrant.rs` real; `ingest.rs` wired to `Commands::Ingest`, but using a `#[cfg(test)]`-only fake `EmbeddingProvider` (deterministic hash-derived vector) so Qdrant correctness is tested independently of Azure. `ai::provider` traits introduced now (small, cheap) even though the real Azure impl lands in M4.
   *Tests: Qdrant integration tests via testcontainers; `ingest::run()` tested as a library call.*

5. **M4 — Azure embeddings**: `ai/azure_openai.rs` embeddings half only (chat half deferred to M6); `backoff` retry wrapper; real `AzureOpenAiProvider` replaces the fake at the composition root.
   *Tests: wiremock embeddings tests, 429-retry test; first true `assert_cmd` E2E `ingest` test (real binary + testcontainers Qdrant + wiremock embeddings).*

6. **M5 — Retrieval-only query**: `query.rs` embeds the question, searches Qdrant, prints raw ranked chunks (no LLM generation yet) — a deliberate stepping stone that validates retrieval quality independent of generation.
   *Tests: ranking/top-k correctness, E2E test asserting raw retrieved text appears for a matching question.*

7. **M6 — Full answer generation**: `ai/azure_openai.rs` chat half + `ai/schema.rs` + `ai/prompt.rs` + `render.rs`; `query.rs` updated to replace the M5 raw dump with the real structured, formatted answer.
   *Tests: wiremock chat tests (happy/refusal/malformed/request-shape), prompt/schema/render unit tests, and the E2E test asserting steps + inline warning + source filename together — first point the PRD's full success criteria are demonstrable end-to-end.*

8. **M7 — Polish**: `doctor.rs`; harden the E2E suite (re-ingestion idempotency, multi-doc citation disambiguation, partial-failure ingest, config-error paths, `--help` smoke); error-message pass; README with setup instructions (Docker, Azure env vars, example commands).
   *Verify: full test suite green locally and in CI.*

Explicitly deferred past M7 (don't build unless raised later): colored terminal output, progress bars, payload filtering by source document, streaming chat responses, automatic embedding-dimension detection.

---

## Risks / assumptions (documented, not blockers)

- **No OCR** (`pdf-extract`) — scanned/image-only PDFs yield empty text; assumed text-layer PDFs per the PRD's own stack choice. Mitigated by making this a tested, documented behavior rather than a silent failure.
- **Azure rate limits** during large batch ingestion — mitigated by request batching (96 chunks/call) plus `backoff`-based retry on 429/5xx only (never retrying 4xx config errors like 401).
- **`edition = "2024"`** needs a recent toolchain (MSRV ~1.85); installed `rustc` is well above this, but a `rust-toolchain.toml` is worth adding so other machines/CI don't drift onto an incompatible older toolchain.
- **Structured outputs require a specific Azure deployment** (`gpt-4o-2024-08-06`+/`gpt-4.1`/`gpt-5` family) and API version ≥ `2024-08-01-preview`. If Fexillon's real Azure resource has an older model deployed, chat calls fail immediately — `doctor` catches this early via a minimal smoke request.
- **Point-count drift across re-ingestion with different chunk parameters** — `--recreate-collection` is the documented escape hatch rather than building point-diffing/GC.
- **No authentication layer**, by PRD design — the CLI relies on host/network-level trust; worth stating explicitly since "sensitive documents" could otherwise be conflated with "the CLI enforces access control."

---

## Verification (end-to-end, once implemented)

1. `docker compose up -d` to start local Qdrant.
2. Copy `.env.example` → `.env`, fill in real Azure OpenAI endpoint/key/deployment names.
3. `cargo run -- doctor` — confirms both Qdrant and Azure OpenAI are reachable before doing anything else.
4. `cargo run -- ingest --input-dir ./sample_pdfs` against a small real (or fixture) PDF set — check reported success/failure counts.
5. `cargo run -- query "How do I clean the sensor housing on the X200?"` — confirm the response has step-by-step structure, an inline warning where the source document has one, and a correct source citation.
6. `cargo test` locally — full unit + Qdrant-testcontainer + wiremock-Azure + CLI-E2E suite green.
7. Push to a branch and confirm the GitHub Actions workflow passes.
