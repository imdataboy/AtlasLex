"""Article-aware parsing for French legal headings, with a page-text fallback."""

import re
from dataclasses import dataclass, field

from .extraction import RawPage

ARTICLE_ID = r"\d{1,4}(?:\s*[-–.]\s*\d+)*(?:\s+(?:bis|ter|quater))?"
ARTICLE_HEADER = re.compile(rf"^Article\s+({ARTICLE_ID})\s*((?:\[fn:\d+\]\s*)*)[.:]?\s*$", re.I)
HIERARCHY = re.compile(
    r"^(Sous[- ]section|Section|Chapitre|Titre|Livre)\s+"
    r"(PREMIER|PREMIÈRE|PREMIERE|[IVXLC]+|\d+)\b.*$",
    re.I,
)
LEVELS = ["livre", "titre", "chapitre", "section", "sous_section"]
REF_MARKER = re.compile(r"\[fn:(\d+)\]")


def normalize_article(value: str) -> str:
    return re.sub(r"\s*[-–]\s*", "-", " ".join(value.lower().split()))


def article_references(text: str) -> list[str]:
    """Recognize explicit article lists/ranges, without treating law numbers as articles."""
    refs = set()
    pattern = re.compile(
        rf"\b(?:articles?|arts?\.)\s+({ARTICLE_ID}(?:\s*(?:,|et|ou|à|a|au)\s*{ARTICLE_ID})*)", re.I
    )
    for match in pattern.finditer(text):
        span = match[1]
        refs.update(normalize_article(x.group()) for x in re.finditer(ARTICLE_ID, span, re.I))
        for first, last in re.findall(r"\b(\d+)\s+(?:à|au)\s+(\d+)\b", span):
            if 0 <= int(last) - int(first) <= 1000:
                refs.update(str(n) for n in range(int(first), int(last) + 1))
    return sorted(refs)


@dataclass
class Article:
    document_id: str
    source_document: str
    section_id: str
    article_number: str | None
    text: str
    page_numbers: list[int]
    hierarchy: dict[str, str] = field(default_factory=dict)
    footnotes: list[str] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)
    footnote_pages: list[int] = field(default_factory=list)

    @property
    def context_header(self) -> str:
        return " > ".join(
            self.hierarchy.get(level, "") for level in LEVELS if self.hierarchy.get(level)
        )


def _page_sections(pages: list[RawPage], document_id: str, filename: str) -> list[Article]:
    return [
        Article(
            document_id,
            filename,
            f"page-{p.page_number}",
            None,
            (p.raw_text or p.clean_text).strip(),
            [p.page_number],
            footnotes=p.footnotes,
        )
        for p in pages
        if p.clean_text.strip() or p.footnotes
    ]


def parse_articles(
    pages: list[RawPage],
    document_id: str,
    filename: str,
    *,
    specialized: bool = False,
) -> tuple[list[Article], list[str]]:
    """Return sections and explicit limitations. Never invent an article number.

    Repeated headings or uncovered preamble text in unknown layouts cause a
    conservative whole-document page fallback rather than misleading citations.
    """
    articles, warnings = [], []
    hierarchy = {level: "" for level in LEVELS}
    hierarchy_refs = {level: set() for level in LEVELS}
    current = None
    lines: list[str] = []
    references: set[str] = set()
    heading_level = None
    uncovered, duplicate = False, False
    seen = set()
    all_notes = [(p.page_number, n) for p in pages for n in p.footnotes]

    def finish():
        nonlocal current, lines, references
        if current is None:
            return
        raw = "\n".join(lines).strip()
        current.text = REF_MARKER.sub("", raw).strip()
        references.update(REF_MARKER.findall(raw))
        relevant = list(current.footnotes)
        for page, note in all_notes:
            marker = re.match(r"\[(\d+)\]", note)
            inline = page in current.page_numbers and marker and marker[1] in references
            # The supplied corpus lists repealed, empty articles and their
            # shared amendment note. Retain that actual source text too.
            explicit = (
                specialized
                and not current.text
                and current.article_number in article_references(note)
            )
            if (inline or explicit) and note not in relevant:
                relevant.append(note)
                current.footnote_pages.append(page)
        current.footnotes = relevant
        current.footnote_pages = sorted(set(current.footnote_pages))
        current.cross_refs = [
            r for r in article_references(current.text) if r != current.article_number
        ]
        if current.text or relevant:
            articles.append(current)
        current, lines, references = None, [], set()

    for page in pages:
        for line in page.clean_text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = ARTICLE_HEADER.fullmatch(line)
            heading = HIERARCHY.fullmatch(REF_MARKER.sub("", line))
            if match:
                finish()
                number = normalize_article(match[1])
                duplicate |= number in seen
                seen.add(number)
                current = Article(
                    document_id,
                    filename,
                    f"article-{number}",
                    number,
                    "",
                    [page.page_number],
                    dict(hierarchy),
                )
                context_refs = set().union(*hierarchy_refs.values())
                for note_page, note in all_notes:
                    marker = re.match(r"\[(\d+)\]", note)
                    if marker and marker[1] in context_refs:
                        current.footnotes.append(note)
                        current.footnote_pages.append(note_page)
                references = set(REF_MARKER.findall(match[2]))
                heading_level = None
            elif heading:
                finish()
                level = heading[1].lower().replace("-", "_").replace(" ", "_")
                hierarchy[level] = REF_MARKER.sub("", line).strip()
                hierarchy_refs[level] = set(REF_MARKER.findall(line))
                for child in LEVELS[LEVELS.index(level) + 1 :]:
                    hierarchy[child] = ""
                    hierarchy_refs[child] = set()
                heading_level = level
            elif heading_level and current is None:
                # Title continuation precedes the next article. Limit to short
                # heading lines; unknown prose triggers fallback for other PDFs.
                if len(line) <= 180 and not line.endswith((".", ";")):
                    hierarchy[heading_level] += " " + REF_MARKER.sub("", line).strip()
                    hierarchy_refs[heading_level].update(REF_MARKER.findall(line))
                else:
                    uncovered = True
                    heading_level = None
            elif current is not None:
                lines.append(line)
                if page.page_number not in current.page_numbers:
                    current.page_numbers.append(page.page_number)
            else:
                uncovered = True
    finish()
    uncertain_markers = not specialized and any(REF_MARKER.search(p.clean_text) for p in pages)
    if not articles or duplicate or (uncovered and not specialized) or uncertain_markers:
        fallback = _page_sections(pages, document_id, filename)
        if fallback:
            warnings.append(
                f"{filename}: article structure could not be established reliably; "
                "using page text without article references."
            )
        return fallback, warnings
    return articles, warnings
