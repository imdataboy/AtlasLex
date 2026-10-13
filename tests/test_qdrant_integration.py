import os
import uuid

import pytest
from qdrant_client import QdrantClient

from atlaslex.config import Settings
from atlaslex.indexing import IndexManager
from atlaslex.retrieval import Retriever
from conftest import FakeModels


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("ATLASLEX_INTEGRATION") != "1", reason="Opt-in Docker Qdrant test")
def test_real_qdrant_snapshot_and_retrieval(tmp_path, monkeypatch):
    from atlaslex.preparation.extraction import RawPage

    settings = Settings(home=tmp_path, collection="atlaslex_test_" + uuid.uuid4().hex)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=5)
    manager = IndexManager(settings, FakeModels(), client, progress=lambda _: None)
    path = tmp_path / "document.pdf"
    path.write_bytes(b"test PDF fixture; extraction is isolated below")
    monkeypatch.setattr(
        "atlaslex.indexing.extract_pages", lambda _: [RawPage(1, "", "Article 1\nCommerce.", [])]
    )
    try:
        manager.index([path])
        assert Retriever(manager).search("Commerce", 1)[0]["article_number"] == "1"
        first = manager.active_collection()
        manager.index([path], rebuild=True)
        assert manager.active_collection() != first
        assert not client.collection_exists(first)
        assert len(manager.load()[2]) == 1
    finally:
        for c in client.get_collections().collections:
            if c.name.startswith(settings.collection + "_"):
                client.delete_collection(c.name)
        manager.close()
