import hashlib
from pathlib import Path

import pdfplumber
import pytest
from qdrant_client import QdrantClient

from atlaslex.config import Settings
from atlaslex.indexing import IndexManager
from atlaslex.preparation.extraction import RawPage, extract_page


@pytest.fixture(scope="session")
def cdc_pdf():
    with pdfplumber.open(Path(__file__).parents[1] / "data/raw/cdc.pdf") as pdf:
        yield pdf


@pytest.fixture
def pdf_pages(cdc_pdf):
    return lambda *numbers: [extract_page(cdc_pdf.pages[n - 1], n, True) for n in numbers]


class FakeModels:
    dimension = 3
    max_tokens = 512

    def __init__(self):
        self.embedded = []

    def token_count(self, text):
        return len(text.split())

    def embed_passages(self, texts):
        self.embedded.extend(texts)
        return [[float(x + 1) for x in hashlib.sha256(t.encode()).digest()[:3]] for t in texts]

    def embed_query(self, text):
        return [1.0, 2.0, 3.0]

    def rerank(self, query, chunks):
        return [dict(c, _rerank_score=0.9) for c in chunks]


@pytest.fixture
def manager(tmp_path, monkeypatch):
    settings = Settings(home=tmp_path)
    settings.raw_dir.mkdir(parents=True)
    client = QdrantClient(":memory:")
    models = FakeModels()
    manager = IndexManager(settings, models=models, client=client, progress=lambda _: None)
    monkeypatch.setattr(
        "atlaslex.indexing.extract_pages",
        lambda p: [RawPage(1, "", "Article 1\n" + p.read_text(), [])],
    )
    yield manager
    manager.close()


@pytest.fixture
def sample_chunk():
    from atlaslex.preparation.chunking import articles_to_chunks
    from atlaslex.preparation.parsing import Article

    return articles_to_chunks(
        [Article("doc1", "one.pdf", "article-1", "1", "Une disposition juridique.", [1])]
    )[0]
