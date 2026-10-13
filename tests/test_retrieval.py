import uuid

import pytest

from atlaslex.retrieval import Retriever, dedup_by_article, reciprocal_rank_fusion


def test_rrf_merges_both_uuid_representations():
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    fused = reciprocal_rank_fusion(
        [{"chunk_id": second}, {"chunk_id": first}], [{"chunk_id": uuid.UUID(first).hex}]
    )
    assert fused == [first, second]


def test_long_article_dedup_keeps_complementary_evidence(sample_chunk):
    first = dict(sample_chunk, text="First relevant text OVERLAP", sub_chunk_index=0)
    second = dict(
        sample_chunk,
        chunk_id=str(uuid.uuid4()),
        text="OVERLAP second relevant text",
        sub_chunk_index=1,
        page_numbers=[2],
    )
    result = dedup_by_article([first, second], 1)
    assert len(result) == 1
    assert result[0]["text"] == "First relevant text OVERLAP second relevant text"
    assert result[0]["page_numbers"] == [1, 2]


def test_identical_article_numbers_in_different_documents_survive(sample_chunk):
    other = dict(sample_chunk, document_id="doc2", chunk_id=str(uuid.uuid4()))
    assert len(dedup_by_article([sample_chunk, other], 2)) == 2


def test_top_k_not_hard_capped_at_three(manager):
    path = manager.settings.raw_dir / "a.pdf"
    # Fixture extraction adds Article 1, followed by these explicit headers.
    path.write_text("Règle 1.\n" + "\n".join(f"Article {n}\nRègle {n}." for n in range(2, 8)))
    manager.index([path])
    retriever = Retriever(manager)
    assert len(retriever.search("règle", top_k=5)) == 5
    with pytest.raises(ValueError):
        retriever.search("règle", top_k=0)


def test_empty_index_and_no_reliable_evidence(manager):
    assert Retriever(manager).search("commerce") == []
    path = manager.settings.raw_dir / "a.pdf"
    path.write_text("Commerce.")
    manager.index([path])
    manager.models.rerank = lambda query, chunks: [dict(c, _rerank_score=0.01) for c in chunks]
    assert Retriever(manager).search("unrelated question") == []


def test_missing_dense_lookup_is_skipped_safely(manager):
    from types import SimpleNamespace

    path = manager.settings.raw_dir / "a.pdf"
    path.write_text("Commerce.")
    manager.index([path])
    retriever = Retriever(manager)
    manager.client.query_points = lambda *a, **kw: SimpleNamespace(
        points=[SimpleNamespace(id=str(uuid.uuid4()), score=0.9)]
    )
    assert retriever.search("zzzznotincorpus") == []
