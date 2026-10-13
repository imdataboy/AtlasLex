import uuid

import pytest

from atlaslex.preparation.chunking import articles_to_chunks, canonical_chunk_id, split_text
from atlaslex.preparation.parsing import Article


def test_empty_and_short_articles():
    assert articles_to_chunks([]) == []
    chunks = articles_to_chunks([Article("h", "d.pdf", "article-1", "1", "Texte court.", [5])])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Texte court."
    assert chunks[0]["page_numbers"] == [5]


def test_paragraph_overlap_never_exceeds_limit():
    chunks = split_text("a" * 1700 + "\n\n" + "b" * 1700)
    assert max(map(len, chunks)) <= 1800
    assert chunks[-1].endswith("b" * 100)


def test_tail_is_never_dropped():
    chunks = split_text("A" * 1800 + "UNIQUE-END", 1800, 200)
    assert chunks[-1].endswith("UNIQUE-END")


def test_global_ids_and_canonical_uuid():
    chunks = articles_to_chunks(
        [Article(h, "a.pdf", "article-1", "1", "Same.", [1]) for h in ["hash1", "hash2"]]
    )
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]
    cid = chunks[0]["chunk_id"]
    assert canonical_chunk_id(uuid.UUID(cid).hex) == cid


def test_token_limit_counts_prefix():
    article = Article("h", "d.pdf", "a-1", "1", "mot " * 700, [1], {"livre": "Titre long"})
    chunks = articles_to_chunks([article], token_count=lambda s: len(s.split()), max_tokens=100)
    assert all(len(("passage: " + c["prefixed_text"]).split()) <= 100 for c in chunks)


def test_bad_sizes_rejected():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)
