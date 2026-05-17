"""Pipeline: chunk → generate QA → deduplicate → split → write JSONL."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from .chunk import chunk_directory
from .qa_generate import QAPair, generate_qa_batch


@dataclass
class PrepareConfig:
    docs_dir: str
    output_path: str = "data/qa_train.jsonl"
    eval_output_path: str | None = None
    model_id: str | None = None
    max_chunk_chars: int = 2500
    min_chunk_chars: int = 200
    max_tokens: int = 2048
    eval_ratio: float = 0.15
    split_by: str = "document"
    seed: int = 42
    dry_run: bool = False

SYSTEM_MSG = "你是一个技术助手。基于技术文档准确回答问题。"


def run_prepare(cfg: PrepareConfig) -> None:
    print(f"Scanning: {cfg.docs_dir}", file=sys.stderr)
    chunks = chunk_directory(
        cfg.docs_dir,
        max_chunk_chars=cfg.max_chunk_chars,
        min_chunk_chars=cfg.min_chunk_chars,
    )
    docs = set(c.doc_path for c in chunks)
    print(f"  {len(chunks)} chunks from {len(docs)} documents", file=sys.stderr)

    if cfg.dry_run:
        _print_dry_run_stats(chunks)
        return

    print("Generating QA pairs...", file=sys.stderr)
    pairs = generate_qa_batch(chunks, model_id=cfg.model_id, max_tokens=cfg.max_tokens)
    print(f"  Raw: {len(pairs)} QA pairs", file=sys.stderr)

    pairs = deduplicate_qa(pairs)
    print(f"  After dedup: {len(pairs)} QA pairs", file=sys.stderr)

    train, eval_set = split_train_eval(
        pairs, eval_ratio=cfg.eval_ratio, split_by=cfg.split_by, seed=cfg.seed
    )
    print(f"  Train: {len(train)}, Eval: {len(eval_set)}", file=sys.stderr)

    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train, out)
    print(f"  Written: {out}", file=sys.stderr)

    eval_path = cfg.eval_output_path
    if eval_path is None:
        eval_path = str(out.with_stem(out.stem + "_eval"))
    eval_out = Path(eval_path)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(eval_set, eval_out)
    print(f"  Written: {eval_out}", file=sys.stderr)

    print(f"\nDone. {len(train)} train + {len(eval_set)} eval QA pairs.", file=sys.stderr)


def qa_to_messages(qa: QAPair) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": qa.question},
            {"role": "assistant", "content": qa.answer},
        ]
    }


def deduplicate_qa(pairs: list[QAPair], threshold: float = 0.85) -> list[QAPair]:
    if not pairs:
        return []
    kept: list[QAPair] = []
    seen_bigrams: list[set[str]] = []

    for p in pairs:
        bg = _char_bigrams(p.question)
        if any(_jaccard(bg, s) >= threshold for s in seen_bigrams):
            continue
        kept.append(p)
        seen_bigrams.append(bg)

    return kept


def split_train_eval(
    pairs: list[QAPair],
    eval_ratio: float = 0.15,
    split_by: str = "document",
    seed: int = 42,
) -> tuple[list[QAPair], list[QAPair]]:
    rng = random.Random(seed)

    if split_by == "document":
        doc_groups: dict[str, list[QAPair]] = {}
        for p in pairs:
            doc_groups.setdefault(p.source_doc, []).append(p)

        doc_names = sorted(doc_groups.keys())
        rng.shuffle(doc_names)

        target_eval = max(1, int(len(pairs) * eval_ratio))
        eval_pairs: list[QAPair] = []
        eval_docs: set[str] = set()

        for doc in doc_names:
            if len(eval_pairs) >= target_eval:
                break
            eval_pairs.extend(doc_groups[doc])
            eval_docs.add(doc)

        train_pairs = [p for p in pairs if p.source_doc not in eval_docs]
        return train_pairs, eval_pairs
    else:
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
        return shuffled[:split_idx], shuffled[split_idx:]


def _write_jsonl(pairs: list[QAPair], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            record = qa_to_messages(p)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _char_bigrams(text: str) -> set[str]:
    text = text.replace(" ", "").replace("\n", "")
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) > 1 else {text}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _print_dry_run_stats(chunks) -> None:
    print("\n[dry-run] Chunk statistics:", file=sys.stderr)
    sizes = [c.char_count for c in chunks]
    docs = set(c.doc_path for c in chunks)
    print(f"  Documents: {len(docs)}", file=sys.stderr)
    print(f"  Chunks: {len(chunks)}", file=sys.stderr)
    print(f"  Char range: {min(sizes)}-{max(sizes)}", file=sys.stderr)
    print(f"  Avg chars: {sum(sizes) // len(sizes)}", file=sys.stderr)
    print(f"  Estimated QA pairs: {len(chunks) * 3}-{len(chunks) * 5}", file=sys.stderr)
    print("\n[dry-run] No QA generation performed.", file=sys.stderr)
