"""Tests for needle/prepare/chunk.py."""

import tempfile
import os
from pathlib import Path
from needle.prepare.chunk import chunk_markdown, chunk_directory, Chunk


def test_simple_headings():
    md = """# Title

Intro paragraph with enough text to pass minimum size threshold for chunking. Let's add more content to exceed two hundred characters. This paragraph contains technical details about the system architecture and configuration.

## Section A

Content of section A with technical details about the hook protocol and its implementation. This section covers the hook timeout configuration, multi-hook stability patterns, and the decision file protocol specification.

## Section B

Content of section B covering the MCP transport layer, configuration constraints, and boundary definitions for the platform adapter architecture. Additional details about cloud sandbox execution.
"""
    chunks = chunk_markdown(md, "test.md")
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_path == "test.md" for c in chunks)
    assert chunks[0].chunk_index == 0


def test_heading_breadcrumb():
    md = """# Root

## Parent

### Child

Content under child heading with enough text to be meaningful. This section documents the ACP protocol implementation details, JSON-RPC message format, and the stdio transport layer configuration for agent communication.
"""
    chunks = chunk_markdown(md, "test.md")
    child_chunks = [c for c in chunks if "Child" in c.heading_path]
    assert len(child_chunks) >= 1
    assert "Root" in child_chunks[0].heading_path or "Parent" in child_chunks[0].heading_path


def test_min_chunk_filter():
    md = """# Title

Short.

## Real Section

This section has enough content to pass the minimum threshold of 200 characters. It discusses the platform adapter pattern used across coco, trae, and claude implementations. The adapter layer normalizes CLI differences.
"""
    chunks = chunk_markdown(md, "test.md", min_chunk_chars=200)
    for c in chunks:
        assert c.char_count >= 200


def test_large_chunk_splits():
    long_para = "这是一段很长的中文技术文档。" * 200
    md = f"# Title\n\n{long_para}\n\n另一个段落的内容。" + "补充内容。" * 50
    chunks = chunk_markdown(md, "test.md", max_chunk_chars=2500)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_directory(tmp_path):
    (tmp_path / "a.md").write_text(
        "# Doc A\n\n" + "Content A. " * 50 + "\n\n## Section\n\n" + "Details. " * 50,
        encoding="utf-8",
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text(
        "# Doc B\n\n" + "Content B describes the hook protocol. " * 40,
        encoding="utf-8",
    )
    (tmp_path / "skip.txt").write_text("not markdown")

    chunks = chunk_directory(str(tmp_path))
    docs = set(c.doc_path for c in chunks)
    assert "a.md" in docs
    assert os.path.join("sub", "b.md") in docs or "sub/b.md" in docs
    assert all(c.doc_path.endswith(".md") for c in chunks)


def test_empty_directory(tmp_path):
    chunks = chunk_directory(str(tmp_path))
    assert chunks == []


def test_real_wiki_chunking():
    wiki = Path("/Users/luwei.will/go/src/code.byted.org/larkarch/harness/docs/wiki")
    if not wiki.exists():
        return
    chunks = chunk_directory(str(wiki))
    assert len(chunks) > 100
    assert len(set(c.doc_path for c in chunks)) > 30
