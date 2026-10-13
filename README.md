# AtlasLex — Moroccan Commercial Law RAG

A Python CLI for asking questions about legal PDFs with source references. It preserves the original French legal retrieval pipeline and runs embeddings and reranking locally on CPU. Gemini handles query rewriting and grounded answers.

## How it works

PDF extraction → article parsing → overlapping chunks → local embeddings/Qdrant + BM25 → reciprocal rank fusion → cross-encoder reranking → grouped article evidence → Gemini answer with checked source references.

Indexing does not call Gemini. The CLI keeps models loaded across questions; each question is independent. See [the architecture](docs/architecture.svg).

## Requirements

- Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Docker and Docker Compose.
- A Gemini API key and internet access for initial model downloads and Gemini requests.
- Disk space for the two model downloads and the local Qdrant volume; CPU inference can take time.

## Installation

Replace `<repository-url>` with this project's GitHub clone URL after publication:

```bash
git clone https://github.com/imdataboy/AtlasLex.git
cd atlaslex
uv sync --locked
docker compose up -d
uv run atlaslex
```

In the existing checkout, start at `uv sync --locked`. Select **1 — Set Gemini API Key** to enter a key with hidden input. It is saved to a Git-ignored `.env` with restricted permissions. Configuration settings are listed in `.env.example`; an API key being configured does not verify its validity or quota.

The credential removed from the historical notebook must be revoked/rotated by its owner, including anywhere it appears in prior Git history. Do not reuse it.

Qdrant binds to localhost, persists in a named Docker volume, and defaults to port 6333. Configure `QDRANT_HOST`, `QDRANT_PORT`, and `QDRANT_COLLECTION` in `.env` if needed. From another working directory, use `uv run --project /absolute/path/to/atlaslex atlaslex`. Data defaults to the checkout, or to `ATLASLEX_HOME` when set. Do not run simultaneous indexing sessions against the same collection.

Run the focused tests with `uv run pytest`; optionally run the Docker-backed check with `ATLASLEX_INTEGRATION=1 uv run pytest -m integration`. Tests mock Gemini and never spend API quota. Historical dataset utilities are documented in [examples](examples/README.md).

## Adding legal PDFs

Place PDFs in `data/raw/` (the supplied `cdc.pdf` is already there), then select **2 — Add / Index Legal PDFs**. Index all PDFs there, or enter a custom PDF/folder path. Subfolders are scanned too. Newly added personal PDFs and generated artifacts are Git-ignored.

Article-aware parsing is designed primarily for the supplied Moroccan Commercial Code. Other layouts use conservative article recognition or page-text fallback, with an explicit notice and no invented article numbers. Image-only scans require external OCR. A failed or empty document leaves the existing index intact; remove or fix it and retry.

Duplicate PDFs are detected by content hash. Unchanged documents reuse their embeddings. **Rebuild existing index** rereads registered sources and `data/raw/`, requests confirmation, and removes missing sources. Keep custom source files available. A rebuild prepares a complete replacement, verifies it, then switches the Qdrant alias; it does not delete the active index first. If no usable PDFs remain, the existing index is retained.

Qdrant and `data/processed/` form one knowledge base: keep both when moving or backing it up. BM25 is built from the active snapshot's `chunks.json`. Use a new collection name for an unrelated existing Qdrant collection; AtlasLex will not overwrite it.

## Asking questions

Select **3 — Start Chatting**, ask a question in French, and review the cited document, article and page references. Type `exit` (any case) to return to the main menu. The menu continues until **5 — Exit**. **4 — Check System Status** checks Qdrant, the corpus, and model cache files without downloading models or calling Gemini.

Chat lists the indexed filenames. The bundled `cdc.pdf` is the Commercial Code: a broad question such as “Quels sont mes droits comme client ?” can retrieve unrelated commercial contexts rather than a complete answer. Ask about a specific situation, or index the relevant legal PDF. An insufficient-evidence response means the retrieved passages do not support an answer; a citation rejection means the generated answer failed the source checks. For a question covered by the bundled corpus, try “Dans quels deux cas un commerçant ou une personne morale est-il radié d'office ?”.

`USE_QUERY_REWRITE=false` disables the rewriting call. `DENSE_TOP_K`, `SPARSE_TOP_K`, and `FINAL_TOP_K` control retrieval (defaults: 10, 10, 3). `MIN_RERANK_SCORE=0.1` is a configurable heuristic for abstention, not a calibrated confidence score. Rate/quota errors leave the chat open. Citation checks reject unknown article/source IDs; they cannot prove that every generated legal statement follows from the source.

If Google denies the **project's access** with a 403 error, open [Google Cloud Console](https://console.cloud.google.com/) for the project linked to your key and check for a restriction or appeal notice; contact Google support if none appears. Creating another key in that same project does not resolve a project restriction. A **reported-as-leaked key** is a separate error: revoke it and configure a replacement with Option 1. AtlasLex distinguishes these cases without displaying credentials or raw server responses.

## Models

- Embeddings: `antoinelouis/french-me5-base` (original `query:` / `passage:` prefixes, normalized vectors).
- Reranking: `antoinelouis/crossencoder-me5-base-mmarcoFR`.
- Rewriting and answers: `gemini-3-flash-preview`, configurable with `GEMINI_MODEL`.

The local models download automatically on first use and use the Hugging Face cache on subsequent runs. Their revisions are pinned in `config.py` to preserve the original weights. No GPU or paid embedding API is required. Dependencies are locked to a compatible Python 3.11 environment. Model availability and Gemini quotas remain external dependencies.

## Limitations

The supplied Code de commerce identifies a consolidation dated **December 19, 2019**; current legislation is not guaranteed. Its source URL and redistribution terms were not recorded in the original repository; verify provenance before redistributing the PDF. The MIT license covers project code, not the source documents or third-party models.

Answers depend on indexed documents, extraction quality and Gemini. Parsing arbitrary legal layouts, OCR, conversational memory and fine-tuning are outside scope. No measured legal-answer accuracy is claimed. This is an educational portfolio application, not a substitute for professional legal advice.
