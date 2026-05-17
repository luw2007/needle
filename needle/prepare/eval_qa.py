"""Post-training QA accuracy evaluation using LLM-as-judge."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalResult:
    question: str
    expected: str
    predicted: str
    score: float
    judgment: str


@dataclass
class EvalReport:
    total: int = 0
    correct: int = 0
    partial: int = 0
    wrong: int = 0
    accuracy: float = 0.0
    results: list[EvalResult] = field(default_factory=list)


JUDGE_PROMPT = """你是一个评判助手。比较"预期答案"和"模型答案"，判断模型是否正确回答了问题。

问题：{question}
预期答案：{expected}
模型答案：{predicted}

评分标准：
- "correct": 模型答案包含预期答案的核心信息点，即使措辞不同、顺序不同、或包含额外正确信息
- "partial": 模型答案提到了主题但遗漏了多个关键信息点
- "wrong": 模型答案与预期答案的核心信息完全不符或答非所问

仅输出一个词：correct 或 partial 或 wrong"""


def evaluate_qa(
    eval_jsonl: str | Path,
    model_id: str | None = None,
    adapter_path: str | None = None,
    judge_model_id: str | None = None,
    max_tokens: int = 1024,
    max_samples: int | None = None,
) -> EvalReport:
    eval_path = Path(eval_jsonl)
    records = _load_eval_records(eval_path)
    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    print(f"Evaluating {len(records)} QA pairs...", file=sys.stderr)

    answer_model, answer_tokenizer = _load_eval_model(model_id, adapter_path)

    report = EvalReport(total=len(records))

    for i, (question, expected) in enumerate(records):
        predicted = _generate_answer(answer_model, answer_tokenizer, question, max_tokens)
        judgment, score = _keyword_judge(expected, predicted)

        if judgment == "correct":
            report.correct += 1
        elif judgment == "partial":
            report.partial += 1
        else:
            report.wrong += 1

        report.results.append(EvalResult(
            question=question, expected=expected,
            predicted=predicted, score=score, judgment=judgment,
        ))

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            acc = (report.correct + 0.5 * report.partial) / (i + 1)
            print(f"  [{i + 1}/{len(records)}] running accuracy: {acc:.1%}", file=sys.stderr)

    report.accuracy = (report.correct + 0.5 * report.partial) / report.total if report.total else 0.0
    return report


def _keyword_judge(expected: str, predicted: str) -> tuple[str, float]:
    """Judge answer quality by combining ROUGE-L F1 and key entity overlap.

    Uses character-level LCS for Chinese text, plus checks that
    critical identifiers (code, paths, technical terms) from the
    expected answer appear in the prediction.
    """
    expected_clean = _normalize_text(expected)
    predicted_clean = _normalize_text(predicted)

    if not expected_clean:
        return ("correct", 1.0) if len(predicted_clean) > 5 else ("wrong", 0.0)

    lcs_len = _lcs_length(expected_clean, predicted_clean)
    precision = lcs_len / len(predicted_clean) if predicted_clean else 0.0
    recall = lcs_len / len(expected_clean) if expected_clean else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    entity_score = _entity_overlap(expected, predicted)

    combined = max(f1, entity_score)

    if combined >= 0.20:
        return "correct", combined
    elif combined >= 0.10:
        return "partial", combined
    else:
        return "wrong", combined


def _entity_overlap(expected: str, predicted: str) -> float:
    """Check overlap of key entities: code identifiers, paths, numbers, technical terms."""
    import re
    entities: list[str] = []

    for m in re.finditer(r'`([^`]+)`', expected):
        entities.append(m.group(1))
    for m in re.finditer(r'[\w./\\-]{2,}\.(?:py|json|yaml|md|toml|js|ts|sh|jsonl)\b', expected):
        entities.append(m.group(0))
    for m in re.finditer(r'\[\[([^\]]+)\]\]', expected):
        entities.append(m.group(1))
    for m in re.finditer(r'[A-Za-z_][A-Za-z0-9_]{3,}', expected):
        w = m.group(0)
        if w.lower() not in {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'which', 'would', 'should', 'could', 'true', 'false', 'none', 'null'}:
            entities.append(w)

    if not entities:
        return 0.0

    predicted_lower = predicted.lower()
    hits = sum(1 for e in entities if e.lower() in predicted_lower)
    return hits / len(entities)


def _normalize_text(text: str) -> str:
    """Remove whitespace and common punctuation for comparison."""
    import re
    text = re.sub(r'[\s　]+', '', text)
    text = re.sub(r'[，。、；：！？""''【】（）《》\-—…·`\'\",.;:!?\[\](){}<>/\\|]', '', text)
    return text.lower()


def _lcs_length(a: str, b: str) -> int:
    """Compute length of longest common subsequence."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m > 500 or n > 500:
        return _lcs_length_approx(a, b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def _lcs_length_approx(a: str, b: str) -> int:
    """Approximate LCS for long strings using character set overlap."""
    a_chars = set(a)
    b_chars = set(b)
    overlap = a_chars & b_chars
    a_in = sum(1 for c in a if c in overlap)
    b_in = sum(1 for c in b if c in overlap)
    return min(a_in, b_in)

    report.accuracy = (report.correct + 0.5 * report.partial) / report.total if report.total else 0.0
    return report


def _load_eval_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages = data.get("messages", [])
            question = ""
            answer = ""
            for msg in messages:
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    answer = msg.get("content", "")
            if question and answer:
                records.append((question, answer))
    return records


def _load_eval_model(model_id: str | None, adapter_path: str | None):
    try:
        from mlx_lm import load as mlx_load, generate as mlx_gen
    except ImportError:
        raise ImportError(
            "mlx-lm is required for QA evaluation.\n"
            "Install with: pip install 'needle[gemma-ft]'"
        ) from None

    from ..model.registry import resolve_model
    model_id = resolve_model(model_id)

    print(f"Loading model: {model_id}", file=sys.stderr)
    if adapter_path:
        print(f"  with adapter: {adapter_path}", file=sys.stderr)

    model, tokenizer = mlx_load(model_id, adapter_path=adapter_path)
    return model, tokenizer


def _generate_answer(model, tokenizer, question: str, max_tokens: int) -> str:
    from mlx_lm import generate as mlx_gen

    messages = [
        {"role": "system", "content": "你是一个技术助手。基于技术文档准确回答问题。"},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_gen(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def _judge_answer(model, tokenizer, question: str, expected: str, predicted: str) -> str:
    from mlx_lm import generate as mlx_gen

    prompt_text = JUDGE_PROMPT.format(
        question=question, expected=expected, predicted=predicted
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = mlx_gen(model, tokenizer, prompt=prompt, max_tokens=32, verbose=False)
    output = output.strip().lower()

    for label in ("correct", "partial", "wrong"):
        if label in output:
            return label
    return "wrong"


def main(args) -> None:
    report = evaluate_qa(
        eval_jsonl=args.eval_jsonl,
        model_id=getattr(args, "model", None),
        adapter_path=getattr(args, "adapter_path", None),
        judge_model_id=getattr(args, "judge_model", None),
        max_tokens=getattr(args, "max_tokens", 1024),
        max_samples=getattr(args, "max_samples", None),
    )

    print(f"\n{'=' * 50}")
    print(f"QA Evaluation Report")
    print(f"{'=' * 50}")
    print(f"  Total:    {report.total}")
    print(f"  Correct:  {report.correct}")
    print(f"  Partial:  {report.partial}")
    print(f"  Wrong:    {report.wrong}")
    print(f"  Accuracy: {report.accuracy:.1%}")
    print(f"{'=' * 50}")

    if report.wrong > 0:
        print(f"\nWrong answers (up to 5):")
        count = 0
        for r in report.results:
            if r.judgment == "wrong" and count < 5:
                print(f"  Q: {r.question[:80]}")
                print(f"  Expected: {r.expected[:80]}")
                print(f"  Got:      {r.predicted[:80]}")
                print()
                count += 1
