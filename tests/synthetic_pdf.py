"""A minimal PDF writer, so a parser's extraction path runs in CI.

data/raw is gitignored, so a test that opens the real source skips wherever the
tree is absent, which is every CI run. Five parsers in this plan reached
parse() only through such a skip, and a control that runs on one laptop is not
a control. This module writes an uncompressed PDF with a standard Type 1 font
and, when asked, ruled lines, which is enough for both
pdfplumber.extract_text() and pdfplumber.extract_tables() to read the content
back. Verified against pdfplumber 0.11.4. [measured 2026-08-19]

It adds no dependency. Building the bytes by hand is the point: a fixture
generator that needed reportlab would put a new package into the lint and test
environments to test a parser that reads PDFs it did not write.

Text is encoded cp1252 against an explicit /WinAnsiEncoding, so an en dash, a
curly apostrophe and the other punctuation real specification text carries
round-trip as themselves. A character outside cp1252 raises and names itself
rather than arriving in the fixture as a substitute glyph, because a fixture
that quietly differs from the string the test declared is a test asserting
something nobody wrote.
"""

from __future__ import annotations

from typing import Final

# (x, y_from_top, text)
TextRun = tuple[float, float, str]
# (x0, y0_from_top, x1, y1_from_top)
Rule = tuple[float, float, float, float]

# The one encoding a Type 1 base font can carry without an embedded /Differences
# array that also covers the punctuation specification PDFs use.
_TEXT_ENCODING: Final[str] = "cp1252"


def _escape(text: str) -> str:
    """Escape the three characters a PDF literal string reserves.

    Raises:
        ValueError: If a character cannot be written under /WinAnsiEncoding.
    """
    for position, character in enumerate(text):
        try:
            character.encode(_TEXT_ENCODING)
        except UnicodeEncodeError:
            raise ValueError(
                f"build_pdf: {character!r} (U+{ord(character):04X}) at position "
                f"{position} of {text!r} is outside {_TEXT_ENCODING}, which is "
                f"what /WinAnsiEncoding can spell. Written anyway it would "
                f"reach the fixture as a different glyph and the test would "
                f"assert on a string nobody wrote."
            ) from None
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(
    pages: list[list[TextRun]],
    rules: list[list[Rule]] | None = None,
    width: float = 612,
    height: float = 792,
) -> bytes:
    """Pages of (x, y_from_top, text) runs, with optional (x0, y0, x1, y1) rules.

    y is measured from the top of the page, because a fixture is easier to read
    that way than in PDF user space. Rules are what pdfplumber's default
    "lines" table strategy detects, so a table fixture must draw its own grid.

    Raises:
        ValueError: If the two arguments carry different page counts, or a run
            holds a character /WinAnsiEncoding cannot spell.
    """
    line_sets = rules if rules is not None else [[] for _ in pages]
    if len(line_sets) != len(pages):
        raise ValueError(
            f"build_pdf: {len(pages)} page(s) of text and {len(line_sets)} "
            f"of rules. They index together."
        )
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )
    contents: list[int] = []
    for runs, lines in zip(pages, line_sets):
        stream = ["0.5 w"]
        for x0, y0, x1, y1 in lines:
            stream.append(
                f"{x0:.2f} {height - y0:.2f} m {x1:.2f} {height - y1:.2f} l S"
            )
        stream.append("BT /F1 10 Tf")
        for x, y, text in runs:
            stream.append(
                f"1 0 0 1 {x:.2f} {height - y:.2f} Tm ({_escape(text)}) Tj"
            )
        stream.append("ET")
        payload = "\n".join(stream).encode(_TEXT_ENCODING)
        contents.append(add(
            b"<< /Length " + str(len(payload)).encode() + b" >>\nstream\n"
            + payload + b"\nendstream"
        ))

    # The /Pages object is written after every page, so its object number has
    # to be predicted here: each page dictionary names its parent. One font and
    # one content stream per page are already in `objects`, and one page
    # dictionary per page follows, so the next free number is this.
    pages_id = len(objects) + len(contents) + 1
    page_ids: list[int] = []
    for content_id in contents:
        page_ids.append(add(
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        ))
    pages_obj = add(
        b"<< /Type /Pages /Count " + str(len(page_ids)).encode() + b" /Kids ["
        + b" ".join(f"{p} 0 R".encode() for p in page_ids) + b"] >>"
    )
    if pages_obj != pages_id:
        raise ValueError(
            f"build_pdf: the /Pages object landed at {pages_obj} while every "
            f"page dictionary names {pages_id} as its parent. The object "
            f"layout changed and the prediction above did not follow."
        )
    catalog = add(b"<< /Type /Catalog /Pages " + str(pages_obj).encode() + b" 0 R >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{start}\n%%EOF\n"
    ).encode()
    return bytes(out)
