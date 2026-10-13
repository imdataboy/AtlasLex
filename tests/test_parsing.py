from atlaslex.preparation.extraction import RawPage
from atlaslex.preparation.parsing import article_references, parse_articles


def parse(pages):
    return parse_articles(pages, "cdc", "cdc.pdf", specialized=True)[0]


def test_articles_748_to_750_and_page_continuation(pdf_pages):
    articles = {a.article_number: a for a in parse(pdf_pages(242, 243))}
    assert {"747", "748", "749", "750", "751"} <= articles.keys()
    assert articles["750"].page_numbers == [242, 243]
    assert "Le droit de vote" in articles["750"].text
    assert "Le droit de vote" not in articles["747"].text


def test_article_30_and_footnotes(pdf_pages):
    articles = {a.article_number: a for a in parse(pdf_pages(13))}
    assert "Toute inscription" in articles["30"].text
    assert "Toute inscription" not in articles["29"].text
    assert articles["30"].footnotes[0].startswith("[10]")


def test_repealed_articles_keep_actual_amendment_note(pdf_pages):
    articles = {a.article_number: a for a in parse(pdf_pages(52, 53))}
    for number in [132, 133, 134, 135, 138, 139, 140, 141, 142]:
        assert articles[str(number)].text == ""
        assert "abrogées" in articles[str(number)].footnotes[0]


def test_livre_v_and_multiline_heading(pdf_pages):
    articles = parse(pdf_pages(177))
    for article in articles:
        assert (
            article.hierarchy["livre"] == "Livre V: LES PROCÉDURES DES DIFFICULTÉS DE L’ENTREPRISE"
        )
        assert any(n.startswith("[75]") for n in article.footnotes)
        assert 177 in article.footnote_pages


def test_complete_supplied_corpus_has_all_numbered_and_compound_articles():
    from pathlib import Path
    from atlaslex.preparation.extraction import CDC_SHA256, extract_pages

    path = Path(__file__).parents[1] / "data/raw/cdc.pdf"
    articles, warnings = parse_articles(
        extract_pages(path), CDC_SHA256, path.name, specialized=True
    )
    assert not warnings
    assert len(articles) == 839
    numbers = {a.article_number for a in articles}
    assert set(map(str, range(1, 799))) <= numbers
    assert {"42-1", "78.1", "389 bis", "391 bis", "536 bis", "544-11"} <= numbers


def test_article_identifiers_and_short_provisions():
    pages = [RawPage(1, "", "Article 42-1\nTexte court.\nArticle 389 bis\nAutre texte.", [])]
    articles, warnings = parse_articles(pages, "doc", "a.pdf")
    assert [a.article_number for a in articles] == ["42-1", "389 bis"]
    assert not warnings


def test_decimal_article_identifiers(pdf_pages):
    articles = parse(pdf_pages(30, 31))
    assert {"78", "78.1", "78.2"} <= {a.article_number for a in articles}
    assert "6" not in {a.article_number for a in articles}


def test_unknown_layout_has_provenance_without_invented_articles():
    pages = [RawPage(7, "", "Texte juridique sans structure reconnue.", [])]
    articles, warnings = parse_articles(pages, "hash", "custom.pdf")
    assert warnings
    assert articles[0].article_number is None
    assert articles[0].page_numbers == [7]
    assert articles[0].source_document == "custom.pdf"


def test_partial_and_repeated_article_layout_falls_back():
    for text in ["Preamble.\nArticle 1\nBody", "Article 1\nBody\nArticle 1\nMore body"]:
        articles, warnings = parse_articles([RawPage(2, "", text, [])], "h", "x.pdf")
        assert warnings
        assert all(a.article_number is None for a in articles)
        assert articles[0].text == text


def test_empty_document():
    assert parse_articles([], "hash", "empty.pdf") == ([], [])


def test_cross_reference_ranges_and_suffixes():
    assert set(article_references("articles 10 à 15 et 18; article 42-1; article 389 bis")) == {
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "18",
        "42-1",
        "389 bis",
    }
