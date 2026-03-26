from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def normalize_whitespace(s: str | None) -> str:
    if s is None:
        return ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def strip_headers_and_footers(text: str) -> str:
    """
    Best-effort cleanup of common Appendix PP headers/footers that can break parsing.
    Adjust the patterns if you see false positives in your PDFs.
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        l = line.strip()
        if not l:
            cleaned.append("")
            continue
        if re.search(r"State Operations Manual", l, flags=re.I):
            continue
        if re.search(r"Appendix PP", l, flags=re.I):
            continue
        if re.match(r"Page \d+ of \d+", l):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_pdf_text(path: Path) -> str:
    """
    Extract PDF text using pdfminer.six when available, falling back to PyPDF2.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # pdfminer.six (higher quality extraction for many PDFs)
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        text = extract_text(str(path))
        if text and text.strip():
            return text
    except Exception:
        pass

    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError("Unable to extract PDF text. Install `pdfminer.six` or `PyPDF2`.") from e


def parse_by_positions(text: str) -> pd.DataFrame:
    """
    Parse Appendix PP text into (deficiency_tag, title, text) blocks by scanning for line-start F-tag headers.
    """
    text = normalize_whitespace(text)
    text = strip_headers_and_footers(text)
    text = text.replace("\r", "\n").replace("\u00A0", " ")

    headers = list(re.finditer(r"^\sF\s?(\d{3,4})\b", text, flags=re.MULTILINE))
    rows: list[dict] = []
    for i, m in enumerate(headers):
        tag_digits = m.group(1)
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        lines = block.split("\n", 1)
        header_line = lines[0]
        body = lines[1] if len(lines) > 1 else ""

        title = re.sub(r"^\sF\s?" + re.escape(tag_digits) + r"\b", "", header_line).strip()
        deficiency_tag = int(tag_digits)
        rows.append(
            {
                "deficiency_tag": deficiency_tag,
                "f_tag": f"F{tag_digits}",
                "title": normalize_whitespace(title),
                "text": normalize_whitespace(body.strip()),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Deduplicate by numeric tag (PDFs sometimes repeat headers); keep the longest body.
    df["len_text"] = df["text"].str.len()
    df = (
        df.sort_values(["deficiency_tag", "len_text"], ascending=[True, False])
        .drop_duplicates("deficiency_tag")
        .drop(columns=["len_text"])
        .reset_index(drop=True)
    )
    return df


def build_baseline_from_pdf(pdf_path: Path) -> pd.DataFrame:
    raw_text = extract_pdf_text(pdf_path)
    return parse_by_positions(raw_text)

