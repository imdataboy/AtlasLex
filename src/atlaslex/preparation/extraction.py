"""Extract text without mistaking numbered provisions for footnotes.

The bundled Code de commerce has a known layout: 14pt body, 12pt notes,
and smaller raised footnote markers. That profile is selected by content hash,
never by filename. Other PDFs retain their text and use conservative parsing.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pdfplumber.utils import extract_text

CDC_SHA256 = "2122e76f8757e608e807a358006bcf4618ed44a96ab03d90ebe1ef49288e536c"
NOTE_START = re.compile(r"^(\d{1,3})(?:\s*[-–—]\s*|\s+)([A-Za-zÀ-ÿ«“‘].*)")


@dataclass
class RawPage:
    page_number: int
    raw_text: str
    clean_text: str
    footnotes: list[str]


def document_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deduplicate_bold(text: str) -> str:
    """Repair the doubled capital headings in the original PDF, conservatively."""
    lines = []
    for line in text.splitlines():
        words = line.split()
        letters = "".join(c for c in line if c.isalpha())
        if letters and letters.isupper() and words:
            doubled = [
                w
                for w in words
                if len(w) >= 4
                and len(w) % 2 == 0
                and all(w[i] == w[i + 1] for i in range(0, len(w), 2))
            ]
            if len(doubled) / len(words) >= 0.6:
                line = " ".join(w[::2] if w in doubled else w for w in words)
        lines.append(line)
    return "\n".join(lines)


def _marked_line(chars: list[dict]) -> str:
    """Replace only raised, smaller digit runs; ordinary legal numbers stay intact."""
    chars = sorted(chars, key=lambda c: c["x0"])
    main = [c for c in chars if c["text"].isalpha()]
    if not main:
        return extract_text(chars) or ""
    main_size = max(c["size"] for c in main)
    baseline = max(c["bottom"] for c in main if c["size"] >= main_size * 0.95)
    result, i = [], 0
    while i < len(chars):
        c = chars[i]
        if c["text"].isdigit() and c["size"] < main_size * 0.8 and c["bottom"] < baseline - 2:
            digits, start = c["text"], dict(c)
            i += 1
            while i < len(chars):
                nxt, prev = chars[i], chars[i - 1]
                if not (
                    nxt["text"].isdigit()
                    and abs(nxt["size"] - c["size"]) < 0.5
                    and nxt["x0"] - prev["x1"] < 3
                ):
                    break
                digits += nxt["text"]
                i += 1
            start["text"] = f"[fn:{digits}]"
            start["top"] = min(x["top"] for x in main)
            result.append(start)
        else:
            result.append(dict(c))
            i += 1
    return extract_text(result, x_tolerance=3, y_tolerance=3).strip()


def extract_page(page, page_number: int, specialized: bool = False) -> RawPage:
    raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    body, notes = [], []
    in_notes = False
    for line in page.extract_text_lines(return_chars=True):
        chars = list(line["chars"])
        # extract_text_lines attaches word characters only. Reinsert the PDF's
        # real spaces before reconstruction, including small gaps in note text.
        tops = [c["top"] for c in chars if c["text"].isalpha()]
        baseline_top = sorted(tops)[len(tops) // 2] if tops else line["top"]
        chars.extend(
            c
            for c in page.chars
            if c["text"] == " "
            and line["x0"] <= c["x0"] <= line["x1"]
            and abs(c["top"] - baseline_top) < 2
        )
        text = _marked_line(chars)
        if not text.strip():
            continue
        if specialized and re.fullmatch(r"\s*-\s*\d+\s*-\s*", text):
            continue
        sizes = [c["size"] for c in chars if c["text"].strip()]
        # Require the known smaller note font AND a note prefix. A body list
        # such as "3– avoir..." is 14pt and must never start a footnote block.
        is_small = bool(sizes) and max(sizes) <= 12.2
        match = NOTE_START.match(text) if specialized and is_small else None
        if match:
            in_notes = True
            notes.append(f"[{match[1]}] {match[2]}")
        elif specialized and is_small:
            if in_notes:
                notes[-1] += " " + text
            else:
                # Some footnotes continue across PDF pages without repeating
                # their number (including quoted articles from OTHER laws).
                notes.append(text)
                in_notes = True
        else:
            in_notes = False
            body.append(text)
    clean = deduplicate_bold("\n".join(body))
    if specialized:
        clean = re.sub(
            r"Royaume\s+du\s+Maroc.*?Direction\s+de\s+L[eé]gislation",
            "",
            clean,
            flags=re.I | re.S,
        )

        # One header in the source (503 + note 70) has no superscript styling.
        # Repair only when the note explicitly names the proposed article.
        def repair_header(match):
            full = match[1]
            for note in notes:
                marker = re.match(r"\[(\d+)\]", note)
                if not marker or not full.endswith(marker[1]):
                    continue
                number = full[: -len(marker[1])]
                if number and re.search(rf"\barticle\s+{re.escape(number)}\b", note, re.I):
                    return f"Article {number}[fn:{marker[1]}]"
            return match[0]

        clean = re.sub(r"^Article\s+(\d{4,})\s*$", repair_header, clean, flags=re.M)
    return RawPage(page_number, raw, clean.strip(), notes)


def extract_pages(path: Path, *, specialized: bool | None = None) -> list[RawPage]:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("Select an existing PDF file.")
    if specialized is None:
        specialized = document_hash(path) == CDC_SHA256
    pages = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            if specialized and (number <= 4 or number >= 260):
                continue
            extracted = extract_page(page, number, specialized)
            if specialized and extracted.footnotes and not extracted.footnotes[0].startswith("["):
                prior = next((p for p in reversed(pages) if p.footnotes), None)
                if prior:
                    prior.footnotes[-1] += " " + extracted.footnotes.pop(0)
            pages.append(extracted)
            page.close()
    return pages
