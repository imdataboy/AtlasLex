from pathlib import Path
from unittest.mock import patch

import pytest

from atlaslex.indexing import IndexErrorBase, discover_pdfs


def pdf(manager, name, text="Une règle du commerce."):
    path = manager.settings.raw_dir / name
    path.write_text(text)
    return path


def test_multiple_documents_and_consistent_ids(manager):
    paths = [pdf(manager, "a.pdf"), pdf(manager, "b.pdf", "Une autre règle.")]
    result = manager.index(paths)
    name, _, chunks = manager.load()
    points, _ = manager.client.scroll(name, limit=100)
    assert result["documents"] == 2
    assert {str(p.id) for p in points} == {c["chunk_id"] for c in chunks}
    assert len({c["document_id"] for c in chunks}) == 2
    assert {c["article_number"] for c in chunks} == {"1"}


def test_duplicate_and_unchanged_documents_reuse_vectors(manager):
    path = pdf(manager, "a.pdf")
    manager.index([path])
    first = manager.active_collection()
    count = len(manager.models.embedded)
    manager.index([path])
    assert manager.active_collection() == first
    duplicate = pdf(manager, "copy.pdf")
    manager.index([duplicate])
    assert len(manager.models.embedded) == count
    assert len(manager.load()[1]["documents"]) == 1
    assert len(manager.load()[2]) == 1


def test_changed_file_removes_obsolete_vectors(manager):
    path = pdf(manager, "a.pdf")
    manager.index([path])
    old, _, chunks = manager.load()
    old_ids = {c["chunk_id"] for c in chunks}
    path.write_text("La nouvelle disposition remplace le texte précédent.")
    manager.index([path])
    active, _, chunks = manager.load()
    assert active != old
    assert not manager.client.collection_exists(old)
    assert not (old_ids & {c["chunk_id"] for c in chunks})


def test_rebuild_removes_missing_source(manager):
    first, second = pdf(manager, "a.pdf"), pdf(manager, "b.pdf", "Autre règle.")
    manager.index([first, second])
    first.unlink()
    paths = manager.source_paths()
    assert first not in paths
    result = manager.index(paths, rebuild=True)
    assert result["documents"] == 1
    assert {c["source_document"] for c in manager.load()[2]} == {"b.pdf"}


def test_failed_upload_keeps_active_qdrant_and_bm25(manager):
    path = pdf(manager, "a.pdf")
    manager.index([path])
    old, manifest, chunks = manager.load()
    path.write_text("Changed content.")
    with patch.object(manager.client, "upsert", side_effect=RuntimeError("upload failed")):
        with pytest.raises(RuntimeError):
            manager.index([path], rebuild=True)
    assert manager.load() == (old, manifest, chunks)


def test_failed_alias_switch_keeps_previous_snapshot(manager):
    path = pdf(manager, "a.pdf")
    manager.index([path])
    old = manager.load()
    path.write_text("New content.")
    with patch.object(
        manager.client, "update_collection_aliases", side_effect=RuntimeError("offline")
    ):
        with pytest.raises(RuntimeError):
            manager.index([path], rebuild=True)
    assert manager.load() == old


def test_empty_or_failed_extraction_does_not_destroy_index(manager, monkeypatch):
    path = pdf(manager, "a.pdf")
    manager.index([path])
    old = manager.load()
    monkeypatch.setattr("atlaslex.indexing.extract_pages", lambda _: [])
    with pytest.raises(IndexErrorBase, match="no extractable text"):
        manager.index([path], rebuild=True)
    assert manager.load() == old


def test_corrupt_local_corpus_can_be_rebuilt(manager):
    path = pdf(manager, "a.pdf")
    manager.index([path])
    (manager.snapshot_dir / manager.active_collection() / "chunks.json").write_text("[]")
    with pytest.raises(IndexErrorBase, match="inconsistent"):
        manager.load()
    manager.index([path], rebuild=True)
    assert len(manager.load()[2]) == 1


def test_wrong_vector_dimensions_fail_before_publication(manager):
    path = pdf(manager, "a.pdf")
    manager.models.embed_passages = lambda texts: [[1.0, 2.0] for _ in texts]
    with pytest.raises(IndexErrorBase, match="dimensions"):
        manager.index([path])
    assert manager.active_collection() is None


def test_discovery_handles_missing_and_uppercase_pdf(tmp_path):
    with pytest.raises(ValueError):
        discover_pdfs(tmp_path / "missing")
    assert discover_pdfs(tmp_path) == []
    (tmp_path / "x.PDF").write_bytes(b"pdf")
    assert discover_pdfs(tmp_path) == [tmp_path / "x.PDF"]
    with pytest.raises(ValueError):
        discover_pdfs(Path(__file__))
