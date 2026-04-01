"""
Extract clean, minimal text from raw 8-K filing documents.

Goal: strip all HTML, EDGAR boilerplate, exhibits, and legal filler so that
only the material facts remain before sending to an LLM.
"""

import re
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    _SKIP = {"script", "style", "head"}
    _BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "td", "th"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def handle_entityref(self, name):
        _ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "nbsp": " ", "quot": '"'}
        if self._skip_depth == 0:
            self._parts.append(_ENTITIES.get(name, ""))

    def handle_charref(self, name):
        if self._skip_depth == 0:
            try:
                ch = chr(int(name[1:], 16) if name.startswith("x") else int(name))
                self._parts.append(ch)
            except (ValueError, OverflowError):
                pass

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(html: str) -> str:
    p = _HTMLStripper()
    p.feed(html)
    return p.get_text()


# ---------------------------------------------------------------------------
# EDGAR / legal boilerplate patterns to remove
# ---------------------------------------------------------------------------

_BOILERPLATE = [
    # Forward-looking / safe harbor blocks
    r"(?i)cautionary (note|statement)s? (regarding|about) forward.looking.*?(?=\n{2,})",
    r"(?i)safe harbor statement.*?(?=\n{2,})",
    r"(?i)this (communication|release|announcement) (may contain|contains) forward.looking.*?(?=\n{2,})",
    # SEC signature/cover-page boilerplate
    r"(?i)pursuant to the requirements of the securities exchange act.*?\n",
    r"(?i)SIGNATURES?\s*\n[-=_]*\n.*",
    # Table-of-contents / exhibit index lines
    r"(?i)exhibit\s+\d+[\.\d]*\s+[-–]\s+.*\n",
    r"(?i)^\s*\d+\.\d+\s+[-–].*\n",
    # Page-number lines
    r"(?m)^\s*-?\s*\d+\s*-?\s*$\n?",
]

_BOILERPLATE_RE = [re.compile(p, re.DOTALL | re.MULTILINE) for p in _BOILERPLATE]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_exhibit_text(raw: bytes, max_chars: int = 40_000) -> str:
    """
    Extract clean text from a standalone exhibit file (EX-99.1, EX-99.2, etc.).

    Unlike extract_filing_text(), this does NOT try to isolate an EDGAR
    document block — exhibit files are standalone HTML or plain-text pages.
    """
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    # Strip HTML
    if re.search(r"<html|<!doctype", text, re.IGNORECASE):
        text = _strip_html(text)
    else:
        text = re.sub(r"<[^>]+>", " ", text)

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove boilerplate
    for pat in _BOILERPLATE_RE:
        text = pat.sub("", text)

    text = text.strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated ...]"

    return text


def extract_filing_text(raw: bytes, max_chars: int = 30_000) -> str:
    """
    Convert raw filing bytes (HTML or SGML/text) to clean prose.

    Strategy:
      1. Isolate the primary 8-K document body (drop exhibits).
      2. Strip HTML tags.
      3. Remove boilerplate / legal filler.
      4. Extract just the Item sections if detectable (highest signal density).
      5. Truncate to max_chars.
    """
    # Decode
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    # --- Step 1: isolate primary document body ---
    # EDGAR wraps: <DOCUMENT>\n<TYPE>8-K\n...<TEXT>\n...\n</TEXT>\n</DOCUMENT>
    primary = re.search(
        r"<TYPE>8-K[^\n]*\n.*?<TEXT>\n?(.*?)</TEXT>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if primary:
        text = primary.group(1)
    else:
        # Fall back: just the first <DOCUMENT> block
        first_doc = re.search(
            r"<DOCUMENT>(.*?)</DOCUMENT>", text, re.DOTALL | re.IGNORECASE
        )
        if first_doc:
            text = first_doc.group(1)

    # --- Step 2: strip HTML ---
    if re.search(r"<html|<!doctype", text, re.IGNORECASE):
        text = _strip_html(text)
    else:
        # Plain-text SGML — remove remaining tags
        text = re.sub(r"<[^>]+>", " ", text)

    # --- Step 3: normalise whitespace ---
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # --- Step 4: remove boilerplate ---
    for pat in _BOILERPLATE_RE:
        text = pat.sub("", text)

    # --- Step 5: try to extract Item sections (densest signal) ---
    item_blocks = re.findall(
        r"(?:^|\n)((?:Item|ITEM)\s+\d+\.\d+[^\n]*\n(?:.|\n)*?)"
        r"(?=\n(?:Item|ITEM)\s+\d+\.\d+|\Z)",
        text,
    )
    if item_blocks:
        candidate = "\n\n".join(b.strip() for b in item_blocks)
        # Only use item extraction if it captured meaningful content
        if len(candidate) > 300:
            text = candidate

    text = text.strip()

    # --- Step 6: truncate ---
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated for length ...]"

    return text
