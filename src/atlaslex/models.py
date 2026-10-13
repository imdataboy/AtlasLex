"""Lazy, CPU-only loading of the original embedding and cross-encoder models."""

from functools import cached_property
from pathlib import Path

from .config import EMBED_MODEL, EMBED_REVISION, RERANK_MODEL, RERANK_REVISION


class Models:
    def __init__(self, progress=print):
        self.progress = progress

    @cached_property
    def embedding(self):
        from sentence_transformers import SentenceTransformer

        self.progress(f"Loading embedding model on CPU: {EMBED_MODEL}")
        return SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION, device="cpu")

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder

        self.progress(f"Loading reranking model on CPU: {RERANK_MODEL}")
        return CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, device="cpu")

    @property
    def dimension(self):
        return self.embedding.get_sentence_embedding_dimension()

    @property
    def max_tokens(self):
        return self.embedding.max_seq_length

    def token_count(self, text):
        # This check deliberately measures before splitting. Suppress the
        # tokenizer's warning about lengths that chunking is about to reduce.
        return len(self.embedding.tokenizer.encode(text, truncation=False, verbose=False))

    def embed_passages(self, texts):
        return self.embedding.encode(
            [f"passage: {t}" for t in texts], normalize_embeddings=True, show_progress_bar=True
        ).tolist()

    def embed_query(self, text):
        return self.embedding.encode(
            [f"query: {text}"], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()

    def rerank(self, query, chunks):
        if not chunks:
            return []
        pairs = [(query, c["prefixed_text"]) for c in chunks]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        return sorted(
            [dict(c, _rerank_score=float(s)) for c, s in zip(chunks, scores)],
            key=lambda c: c["_rerank_score"],
            reverse=True,
        )


def model_cache_status(model_name: str) -> str:
    """Inspect the cache only: no network calls or model instantiation."""
    from huggingface_hub import try_to_load_from_cache

    try:
        revision = {EMBED_MODEL: EMBED_REVISION, RERANK_MODEL: RERANK_REVISION}[model_name]
        weights = try_to_load_from_cache(model_name, "model.safetensors", revision=revision)
        config = try_to_load_from_cache(model_name, "config.json", revision=revision)
        if all(isinstance(p, str) and Path(p).is_file() for p in [weights, config]):
            return "Cached (loads on first use)"
    except (OSError, ValueError):
        pass
    return "Not cached (downloads on first use)"
