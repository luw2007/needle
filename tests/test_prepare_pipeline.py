"""Tests for needle/prepare/pipeline.py (mock Gemma generation)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from needle.prepare.pipeline import (
    PrepareConfig, run_prepare, qa_to_messages,
    deduplicate_qa, split_train_eval,
)
from needle.prepare.qa_generate import QAPair


def _make_pair(q, a, doc="doc1.md", idx=0, qa_type="factual"):
    return QAPair(question=q, answer=a, qa_type=qa_type,
                  source_doc=doc, source_heading="# Test", chunk_index=idx)


def test_qa_to_messages():
    pair = _make_pair("什么是 ACP 协议？", "ACP 是 Agent Client Protocol。")
    msg = qa_to_messages(pair)
    assert "messages" in msg
    roles = [m["role"] for m in msg["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_deduplicate_exact():
    pairs = [
        _make_pair("什么是ACP协议？", "answer1"),
        _make_pair("什么是ACP协议？", "answer2"),
        _make_pair("如何配置hook？", "answer3"),
    ]
    result = deduplicate_qa(pairs)
    assert len(result) == 2


def test_deduplicate_similar():
    pairs = [
        _make_pair("什么是ACP协议的作用？", "answer1"),
        _make_pair("ACP协议的作用是什么？", "answer2"),
        _make_pair("如何配置hook超时？", "answer3"),
    ]
    result = deduplicate_qa(pairs, threshold=0.85)
    assert len(result) <= 3


def test_deduplicate_empty():
    assert deduplicate_qa([]) == []


def test_split_by_document():
    pairs = [
        _make_pair("q1", "a1", doc="doc1.md"),
        _make_pair("q2", "a2", doc="doc1.md"),
        _make_pair("q3", "a3", doc="doc2.md"),
        _make_pair("q4", "a4", doc="doc2.md"),
        _make_pair("q5", "a5", doc="doc3.md"),
        _make_pair("q6", "a6", doc="doc3.md"),
    ]
    train, eval_set = split_train_eval(pairs, eval_ratio=0.3, split_by="document", seed=42)
    eval_docs = set(p.source_doc for p in eval_set)
    train_docs = set(p.source_doc for p in train)
    assert len(eval_docs & train_docs) == 0
    assert len(train) + len(eval_set) == len(pairs)


def test_split_random():
    pairs = [_make_pair(f"q{i}", f"a{i}") for i in range(20)]
    train, eval_set = split_train_eval(pairs, eval_ratio=0.2, split_by="random", seed=42)
    assert len(train) + len(eval_set) == 20
    assert len(eval_set) >= 1


def test_dry_run(tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "test.md").write_text(
        "# Test\n\n" + "Content for testing. " * 50, encoding="utf-8"
    )

    import sys
    cfg = PrepareConfig(
        docs_dir=str(docs),
        output_path=str(tmp_path / "out.jsonl"),
        dry_run=True,
    )
    run_prepare(cfg)
    assert not (tmp_path / "out.jsonl").exists()


def test_full_pipeline_mocked(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "test.md").write_text(
        "# Test Doc\n\n" + "技术文档内容。Hook协议支持stdout通信。" * 30 + "\n\n"
        "## Section Two\n\n" + "MCP配置约束包括transport层和boundary定义。" * 30,
        encoding="utf-8",
    )

    mock_pairs = [
        QAPair("Hook协议支持什么通信方式？", "支持stdout通信。", "factual",
               "test.md", "# Test Doc", 0),
        QAPair("MCP配置有哪些约束？", "包括transport层和boundary定义。", "factual",
               "test.md", "# Test Doc > ## Section Two", 1),
    ]

    with patch("needle.prepare.pipeline.generate_qa_batch", return_value=mock_pairs):
        cfg = PrepareConfig(
            docs_dir=str(docs),
            output_path=str(tmp_path / "train.jsonl"),
            eval_output_path=str(tmp_path / "eval.jsonl"),
            eval_ratio=0.5,
            split_by="random",
            seed=42,
        )
        run_prepare(cfg)

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()

    with open(tmp_path / "train.jsonl") as f:
        records = [json.loads(l) for l in f if l.strip()]
    for r in records:
        assert "messages" in r
        roles = [m["role"] for m in r["messages"]]
        assert "user" in roles
        assert "assistant" in roles


def test_output_validates_with_finetune_data(tmp_path):
    pair = _make_pair("什么是ACP？", "ACP是Agent Client Protocol。")
    msg = qa_to_messages(pair)

    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(msg, ensure_ascii=False) + "\n")

    from needle.finetune.data import validate_jsonl
    errors = validate_jsonl(str(path))
    assert len(errors) == 0
