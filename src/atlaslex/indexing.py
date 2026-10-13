"""Build a complete Qdrant/BM25 snapshot, then atomically switch its alias.

The Qdrant alias identifies the matching on-disk snapshot. Failed preparation or
upload leaves the previous alias and its sparse corpus usable. No LLM is called.
"""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models as qm

from .config import EMBED_MODEL, EMBED_REVISION, PIPELINE_VERSION, Settings
from .models import Models
from .preparation.chunking import articles_to_chunks, canonical_chunk_id
from .preparation.extraction import CDC_SHA256, document_hash, extract_pages
from .preparation.parsing import parse_articles


class IndexErrorBase(RuntimeError):
    """An actionable index error safe to show in the CLI."""


def discover_pdfs(path: Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError("That file or folder does not exist.")
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError("Select a PDF file or a folder containing PDFs.")
        return [path]
    return sorted(
        p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
    )


class IndexManager:
    def __init__(
        self, settings: Settings, models: Models | None = None, client=None, progress=print
    ):
        self.settings = settings
        self.models = models or Models(progress)
        self.client = (
            client
            if client is not None
            else QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=5,
                check_compatibility=False,
            )
        )
        self.progress = progress

    @property
    def snapshot_dir(self):
        return self.settings.processed_dir / self.settings.collection

    def active_collection(self) -> str | None:
        aliases = self.client.get_aliases().aliases
        return next(
            (a.collection_name for a in aliases if a.alias_name == self.settings.collection), None
        )

    def load(self) -> tuple[str | None, dict, list[dict]]:
        active = self.active_collection()
        if not active:
            if self.client.collection_exists(self.settings.collection):
                raise IndexErrorBase(
                    "This collection is not an AtlasLex snapshot. Select a different "
                    "QDRANT_COLLECTION; existing data has not been changed."
                )
            return None, {"documents": {}}, []
        directory = self.snapshot_dir / active
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            raw_chunks = (directory / "chunks.json").read_bytes()
            if hashlib.sha256(raw_chunks).hexdigest() != manifest["chunks_sha256"]:
                raise ValueError("checksum")
            chunks = json.loads(raw_chunks)
            ids = [canonical_chunk_id(c["chunk_id"]) for c in chunks]
            if len(set(ids)) != len(chunks):
                raise ValueError("duplicate IDs")
            if self.client.count(active, exact=True).count != len(chunks):
                raise ValueError("vector count")
        except (OSError, ValueError, KeyError, TypeError):
            raise IndexErrorBase(
                "The vector and local corpus snapshot are inconsistent. Restore the "
                "data/processed directory or rebuild from your source PDFs."
            ) from None
        return active, manifest, chunks

    def source_paths(self) -> list[Path]:
        try:
            _, manifest, _ = self.load()
        except IndexErrorBase:
            self.progress("Local manifest is unavailable. Rebuilding from data/raw/ only.")
            manifest = {"documents": {}}
        paths = {Path(p) for d in manifest["documents"].values() for p in d["paths"]}
        if self.settings.raw_dir.exists():
            paths.update(discover_pdfs(self.settings.raw_dir))
        missing = [p for p in paths if not p.exists()]
        for p in sorted(missing):
            self.progress(f"Source no longer present; will remove on rebuild: {p.name}")
        return sorted(paths - set(missing))

    def _fingerprint(self):
        return {
            "pipeline": PIPELINE_VERSION,
            "embedding_model": EMBED_MODEL,
            "embedding_revision": EMBED_REVISION,
            "max_chars": self.settings.max_chunk_chars,
            "overlap": self.settings.overlap_chars,
        }

    def _check_dimensions(self, name, dimension):
        config = self.client.get_collection(name).config.params.vectors
        if (
            not isinstance(config, qm.VectorParams)
            or config.size != dimension
            or config.distance != qm.Distance.COSINE
        ):
            raise IndexErrorBase(
                "Incompatible embedding dimensions or distance. Rebuild the index."
            )

    def index(self, paths: list[Path], *, rebuild: bool = False) -> dict:
        try:
            old, manifest, existing = self.load()
        except IndexErrorBase:
            if not rebuild:
                raise
            old = self.active_collection()
            if not old and self.client.collection_exists(self.settings.collection):
                raise IndexErrorBase(
                    "Cannot replace a collection not managed by AtlasLex."
                ) from None
            manifest, existing = {"documents": {}}, []
        fingerprint = self._fingerprint()
        if old and not rebuild and manifest.get("fingerprint") != fingerprint:
            raise IndexErrorBase("Processing configuration changed. Choose Rebuild existing index.")
        documents = (
            {}
            if rebuild
            else {k: dict(v, paths=list(v["paths"])) for k, v in manifest["documents"].items()}
        )
        # A changed file replaces its old content; a duplicate copy remains one document.
        incoming = {}
        for path in sorted({Path(p).expanduser().resolve() for p in paths}):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                raise IndexErrorBase(f"Source is not an existing PDF: {path.name}")
            sha = document_hash(path)
            incoming.setdefault(sha, []).append(path)
            for prior in list(documents):
                if prior != sha and str(path) in documents[prior]["paths"]:
                    documents[prior]["paths"].remove(str(path))
                    if not documents[prior]["paths"]:
                        del documents[prior]
        chunks = [] if rebuild else [c for c in existing if c["document_id"] in documents]
        warnings, processed, skipped = [], 0, 0
        for index, (sha, copies) in enumerate(incoming.items(), 1):
            if sha in documents:
                documents[sha]["paths"] = sorted(
                    set(documents[sha]["paths"]) | {str(p) for p in copies}
                )
                skipped += 1
                self.progress(f"[{index}/{len(incoming)}] Unchanged document: {copies[0].name}")
                continue
            path = copies[0]
            self.progress(f"[{index}/{len(incoming)}] Extracting: {path.name}")
            try:
                pages = extract_pages(path)
                articles, notes = parse_articles(
                    pages, sha, path.name, specialized=sha == CDC_SHA256
                )
                if not articles:
                    raise IndexErrorBase(
                        f"{path.name}: no extractable text. Scanned PDFs require OCR outside AtlasLex."
                    )
                self.progress("Parsing articles and generating chunks...")
                prepared = articles_to_chunks(
                    articles,
                    max_chars=self.settings.max_chunk_chars,
                    overlap=self.settings.overlap_chars,
                    token_count=self.models.token_count,
                    max_tokens=self.models.max_tokens,
                )
            except IndexErrorBase:
                raise
            except Exception:
                raise IndexErrorBase(
                    f"Could not process {path.name}. The previous index is unchanged."
                ) from None
            if not prepared:
                raise IndexErrorBase(f"{path.name}: no usable text chunks were produced.")
            warnings.extend(notes)
            documents[sha] = {
                "source_document": path.name,
                "paths": [str(p) for p in copies],
                "article_count": sum(a.article_number is not None for a in articles),
                "chunk_count": len(prepared),
            }
            chunks.extend(prepared)
            processed += 1
        summary = {
            "documents": len(documents),
            "articles": sum(d["article_count"] for d in documents.values()),
            "chunks": len(chunks),
            "processed": processed,
            "skipped": skipped,
            "warnings": warnings,
        }
        if not chunks:
            raise IndexErrorBase("No extractable documents. The existing index has been kept.")
        if old and not rebuild and documents == manifest["documents"]:
            return summary
        dimension = self.models.dimension
        if old and not rebuild:
            self._check_dimensions(old, dimension)
        stage = f"{self.settings.collection}_{uuid.uuid4().hex}"
        directory = self.snapshot_dir / stage
        directory.mkdir(parents=True)
        published = False
        try:
            self.client.create_collection(
                stage, vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE)
            )
            self._check_dimensions(stage, dimension)
            existing_by_id = {c["chunk_id"]: c for c in existing}
            self.progress(f"Generating embeddings and indexing {len(chunks)} chunks...")
            for start in range(0, len(chunks), 64):
                batch = chunks[start : start + 64]
                reuse_ids = [
                    c["chunk_id"]
                    for c in batch
                    if old
                    and manifest.get("vector_size") == dimension
                    and c["chunk_id"] in existing_by_id
                    and existing_by_id[c["chunk_id"]]["prefixed_text"] == c["prefixed_text"]
                    and manifest.get("fingerprint", {}).get("embedding_model") == EMBED_MODEL
                    and manifest.get("fingerprint", {}).get("embedding_revision") == EMBED_REVISION
                ]
                cached = (
                    {
                        canonical_chunk_id(p.id): p.vector
                        for p in self.client.retrieve(
                            old, ids=reuse_ids, with_vectors=True, with_payload=False
                        )
                    }
                    if reuse_ids
                    else {}
                )
                needed = [c for c in batch if c["chunk_id"] not in cached]
                if needed:
                    vectors = self.models.embed_passages([c["prefixed_text"] for c in needed])
                    if len(vectors) != len(needed):
                        raise IndexErrorBase("Embedding count does not match chunk count.")
                    cached.update({c["chunk_id"]: v for c, v in zip(needed, vectors)})
                if any(not isinstance(v, list) or len(v) != dimension for v in cached.values()):
                    raise IndexErrorBase("Embedding dimensions do not match the collection.")
                self.client.upsert(
                    stage,
                    points=[
                        qm.PointStruct(id=c["chunk_id"], vector=cached[c["chunk_id"]], payload=c)
                        for c in batch
                    ],
                    wait=True,
                )
                self.progress(f"Indexed {min(start + 64, len(chunks))}/{len(chunks)} chunks")
            if self.client.count(stage, exact=True).count != len(chunks):
                raise IndexErrorBase("Vector count verification failed; previous index retained.")
            # chunks.json is the canonical sparse corpus; BM25 builds from it.
            data = json.dumps(chunks, ensure_ascii=False, indent=2).encode()
            (directory / "chunks.json").write_bytes(data)
            snapshot = {
                "documents": documents,
                "fingerprint": fingerprint,
                "vector_size": dimension,
                "chunks_sha256": hashlib.sha256(data).hexdigest(),
            }
            (directory / "manifest.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2)
            )
            from .retrieval import build_bm25

            build_bm25(chunks)  # Validate sparse indexing before publishing either representation.
            actions = []
            if old:
                actions.append(
                    qm.DeleteAliasOperation(
                        delete_alias=qm.DeleteAlias(alias_name=self.settings.collection)
                    )
                )
            actions.append(
                qm.CreateAliasOperation(
                    create_alias=qm.CreateAlias(
                        collection_name=stage, alias_name=self.settings.collection
                    )
                )
            )
            self.client.update_collection_aliases(change_aliases_operations=actions)
            published = True
        except Exception:
            # A timeout during alias switching has an uncertain outcome. Only
            # discard staging after verifying it is NOT the active snapshot.
            try:
                if self.active_collection() != stage:
                    self.client.delete_collection(stage)
                    shutil.rmtree(directory)
            except Exception:
                pass
            raise
        if published and old:
            try:
                self.client.delete_collection(old)
                shutil.rmtree(self.snapshot_dir / old, ignore_errors=True)
            except Exception:
                warnings.append("New index is active; cleanup of the previous snapshot failed.")
        return summary

    def close(self):
        self.client.close()
