from atlaslex.preparation.extraction import extract_pages


def test_superscript_is_not_article_number(pdf_pages):
    page = pdf_pages(13)[0]
    assert "Article 30[fn:10]" in page.clean_text
    assert "Article 3010" not in page.clean_text
    assert len(page.footnotes) == 4
    assert any("00-71" in n for n in page.footnotes)
    assert any("de l’article 03" in n for n in page.footnotes)


def test_numbered_provisions_are_not_footnotes(pdf_pages):
    page = pdf_pages(242)[0]
    assert not page.footnotes
    for number in [748, 749, 750]:
        assert f"Article {number}" in page.clean_text
    assert "3– avoir souscrit" in page.clean_text
    assert "5– avoir procédé" in page.clean_text


def test_law_numbers_do_not_start_new_notes(pdf_pages):
    page = pdf_pages(177)[0]
    assert len(page.footnotes) == 1
    assert "1-18-26" in page.footnotes[0]


def test_quoted_article_in_continued_note_is_not_body(pdf_pages):
    page = pdf_pages(31)[0]
    assert "Article 6" not in page.clean_text
    assert any("Article 6" in note for note in page.footnotes)


def test_unstyled_glued_reference_uses_explicit_note(pdf_pages):
    page = pdf_pages(161)[0]
    assert "Article 503[fn:70]" in page.clean_text


def test_empty_page_does_not_crash():
    from atlaslex.preparation.extraction import extract_page

    class EmptyPage:
        chars = []

        def extract_text(self, **kwargs):
            return None

        def extract_text_lines(self, **kwargs):
            return []

    page = extract_page(EmptyPage(), 1)
    assert page.clean_text == ""
    assert page.footnotes == []


def test_invalid_file_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        extract_pages(tmp_path / "missing.pdf")
