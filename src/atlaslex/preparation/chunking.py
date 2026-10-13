"""Overlapping article chunks with stable, document-scoped UUIDs."""

import uuid
from dataclasses import asdict
from typing import Callable

from .parsing import Article


def canonical_chunk_id(value) -> str:
    return str(uuid.UUID(str(value)))


def split_text(text: str, limit: int = 1800, overlap: int = 200) -> list[str]:
    if limit < 1 or not 0 <= overlap < limit:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            for separator in ["\n\n", "\n", ". ", " "]:
                cut = text.rfind(separator, start + limit // 2, end)
                if cut >= 0:
                    end = cut + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def articles_to_chunks(
    articles: list[Article],
    *,
    max_chars: int = 1800,
    overlap: int = 200,
    token_count: Callable[[str], int] | None = None,
    max_tokens: int = 512,
) -> list[dict]:
    result = []
    lookup = {(a.document_id, a.article_number): a for a in articles if a.article_number}
    for article in articles:
        label = f"Article {article.article_number}" if article.article_number else "Extrait"
        prefix = f"[{article.context_header}]\n\n{label}\n\n"
        # Empty/repealed articles can still have a substantive amendment note.
        body = article.text or "\n".join(article.footnotes)
        limit = max_chars
        while True:
            parts = split_text(body, limit, min(overlap, limit // 4))
            if token_count is None or all(
                token_count("passage: " + prefix + t) <= max_tokens for t in parts
            ):
                break
            limit = int(limit * 0.8)
            if limit < 40:
                raise ValueError("Document heading exceeds the embedding model's context limit.")
        for index, text in enumerate(parts):
            cid = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"atlaslex:{article.document_id}:{article.section_id}:{index}:{prefix}{text}",
                )
            )
            payload = asdict(article)
            payload.update(
                chunk_id=cid,
                sub_chunk_index=index,
                total_sub_chunks=len(parts),
                text=text,
                prefixed_text=prefix + text,
                context_header=article.context_header,
                page_number=article.page_numbers[0],
                char_count=len(text),
                cross_ref_summaries=[
                    {"article": ref, "text": lookup[article.document_id, ref].text[:200]}
                    for ref in article.cross_refs
                    if (article.document_id, ref) in lookup
                ],
            )
            result.append(payload)
    return result
