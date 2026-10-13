from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from atlaslex.config import Settings
from atlaslex.generation import (
    Gemini,
    GeminiError,
    INVALID_KEY_MESSAGE,
    LEAKED_KEY_MESSAGE,
    NO_EVIDENCE,
    PROJECT_DENIED_MESSAGE,
    QUOTA_MESSAGE,
    rag_query,
    safe_gemini_error,
    validate_citations,
)


def test_citations_must_match_retrieved_articles(sample_chunk):
    assert validate_citations("Selon l'article 1 [S1], une règle s'applique.", [sample_chunk]) == [
        sample_chunk
    ]
    for answer in [
        "Article 999 [S1]",
        "Article 1 [S2]",
        "Article 1",
        "Article 42-1 [S1]",
        "Article n° 999 [S1]",
        "Article numéro 999 [S1]",
        "Art. 999 [S1]",
    ]:
        with pytest.raises(ValueError):
            validate_citations(answer, [sample_chunk])


def test_article_is_bound_to_locally_cited_source(sample_chunk):
    other = dict(sample_chunk, article_number="2", document_id="doc2")
    with pytest.raises(ValueError):
        validate_citations("Article 1 [S2].\n\nArticle 2 [S1].", [sample_chunk, other])


def test_no_evidence_makes_no_generation_request():
    client = Mock()
    gemini = Gemini(Settings(), client=client)
    assert gemini.answer("question", []) == {"answer": NO_EVIDENCE, "sources": []}
    client.models.generate_content.assert_not_called()


def test_invalid_citation_answer_is_not_displayed(sample_chunk):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text="Article 999 [S1] inventé.")
    answer = Gemini(Settings(), client=client).answer("question", [sample_chunk])
    assert "999" not in answer["answer"]
    assert not answer["sources"]


@pytest.mark.parametrize(
    "response",
    [
        NO_EVIDENCE,
        NO_EVIDENCE.replace("'", "’"),
        NO_EVIDENCE.replace("'", "ʼ").rstrip("."),
        NO_EVIDENCE.replace(" ", "\n\u00a0"),
        f"**{NO_EVIDENCE}**",
        f'"{NO_EVIDENCE}"',
    ],
)
def test_abstention_typography_is_not_a_citation_error(response, sample_chunk, monkeypatch):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text=response)
    validator = Mock(side_effect=AssertionError("An abstention needs no citations."))
    monkeypatch.setattr("atlaslex.generation.validate_citations", validator)
    assert Gemini(Settings(), client=client).answer("question", [sample_chunk]) == {
        "answer": NO_EVIDENCE,
        "sources": [],
    }
    validator.assert_not_called()


def test_abstention_followed_by_a_claim_still_requires_validation(sample_chunk):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text=NO_EVIDENCE + " Cependant, l'article 999 [S1] prévoit un remboursement."
    )
    result = Gemini(Settings(), client=client).answer("question", [sample_chunk])
    assert result["answer"] != NO_EVIDENCE
    assert "références non vérifiables" in result["answer"]
    assert "999" not in result["answer"]
    assert not result["sources"]


def test_error_classification():
    class APIError(Exception):
        def __init__(self, code, text):
            self.code = code
            super().__init__(text)

    assert str(safe_gemini_error(APIError(429, "quota"))) == QUOTA_MESSAGE
    assert str(safe_gemini_error(APIError(400, "API key not valid"))) == INVALID_KEY_MESSAGE
    assert "internet" in str(safe_gemini_error(httpx.ConnectError("secret must not be printed")))
    assert "quota" not in str(safe_gemini_error(APIError(503, "down")))
    assert "secret" not in str(safe_gemini_error(RuntimeError("secret")))


@pytest.mark.parametrize(
    ("provider_message", "expected"),
    [
        ("Your project has been denied access. Please contact support.", PROJECT_DENIED_MESSAGE),
        ("Your API key was reported as leaked. Please use another API key.", LEAKED_KEY_MESSAGE),
        (
            "Permission denied for this resource.",
            "Gemini access denied. Check the API project's permissions and model access.",
        ),
    ],
)
def test_permission_errors_are_specific_and_keep_server_details_private(provider_message, expected):
    from google.genai.errors import ClientError

    client = Mock()
    client.models.generate_content.side_effect = ClientError(
        403,
        {
            "error": {
                "code": 403,
                "status": "PERMISSION_DENIED",
                "message": provider_message + " Credential: private-test-placeholder",
            }
        },
    )
    with pytest.raises(GeminiError) as caught:
        Gemini(Settings(), client=client).rewrite_query("question")
    assert str(caught.value) == expected
    assert "private-test-placeholder" not in str(caught.value)
    assert caught.value.__suppress_context__
    client.models.generate_content.assert_called_once()


def test_rewrite_configuration_is_effective():
    retriever, gemini = Mock(), Mock()
    retriever.settings = Settings(use_query_rewrite=False)
    retriever.search.return_value = []
    gemini.answer.return_value = {"answer": NO_EVIDENCE, "sources": []}
    rag_query("original", retriever, gemini, top_k=5)
    gemini.rewrite_query.assert_not_called()
    retriever.search.assert_called_once_with("original", top_k=5)
    retriever.settings.use_query_rewrite = True
    gemini.rewrite_query.return_value = "rewritten"
    rag_query("original", retriever, gemini)
    gemini.rewrite_query.assert_called_once_with("original")
