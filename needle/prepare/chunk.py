"""Markdown chunking for QA data preparation.

Splits Chinese/English markdown documents by heading hierarchy,
preserving parent heading context as breadcrumbs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


@dataclass
class Chunk:
    doc_path: str
    heading_path: str
    content: str
    char_count: int
    chunk_index: int


def chunk_markdown(
    text: str,
    doc_path: str,
    max_chunk_chars: int = 2500,
    min_chunk_chars: int = 200,
) -> list[Chunk]:
    sections = _split_by_headings(text)
    chunks: list[Chunk] = []
    idx = 0

    for heading_path, body in sections:
        body = body.strip()
        if len(body) < min_chunk_chars:
            continue

        if len(body) <= max_chunk_chars:
            chunks.append(Chunk(
                doc_path=doc_path,
                heading_path=heading_path,
                content=body,
                char_count=len(body),
                chunk_index=idx,
            ))
            idx += 1
        else:
            for part in _split_on_paragraphs(body, max_chunk_chars):
                if len(part) < min_chunk_chars:
                    continue
                chunks.append(Chunk(
                    doc_path=doc_path,
                    heading_path=heading_path,
                    content=part,
                    char_count=len(part),
                    chunk_index=idx,
                ))
                idx += 1

    return chunks


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    lines = text.split("\n")
    current_body: list[str] = []

    def _breadcrumb() -> str:
        return " > ".join(h for _, h in heading_stack) if heading_stack else "(root)"

    def _flush():
        body = "\n".join(current_body)
        if body.strip():
            sections.append((_breadcrumb(), body))
        current_body.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            _flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, f"{'#' * level} {title}"))
        else:
            current_body.append(line)

    _flush()
    return sections


def _split_on_paragraphs(text: str, max_chars: int) -> list[str]:
    paragraphs = re.split(r"\n\n+", text)
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) + 2 > max_chars and current:
            parts.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        parts.append("\n\n".join(current))

    return parts


def chunk_directory(
    docs_dir: str | Path,
    glob_pattern: str = "**/*.md",
    max_chunk_chars: int = 2500,
    min_chunk_chars: int = 200,
) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    all_chunks: list[Chunk] = []

    for path in sorted(docs_dir.glob(glob_pattern)):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(docs_dir))
        chunks = chunk_markdown(text, rel, max_chunk_chars, min_chunk_chars)
        all_chunks.extend(chunks)

    return all_chunks
