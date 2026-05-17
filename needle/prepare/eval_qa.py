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

    judge_model, judge_processor = None, None
    try:
        from ..model.gemma import load_model as load_vlm_model
        judge_model, judge_processor = load_vlm_model()
        print("  Judge: Gemma model (mlx_vlm)", file=sys.stderr)
    except Exception:
        print("  Judge: fallback (ROUGE-L + entity)", file=sys.stderr)

    report = EvalReport(total=len(records))

    for i, (question, expected, system_ctx) in enumerate(records):
        predicted = _generate_answer(answer_model, answer_tokenizer, question, max_tokens, system_ctx)
        judgment, score = _keyword_judge(expected, predicted, judge_model, judge_processor)

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
            avg = sum(r.score for r in report.results) / (i + 1)
            print(f"  [{i + 1}/{len(records)}] avg score: {avg:.1f}, min: {min(r.score for r in report.results):.1f}", file=sys.stderr)

    all_scores = [r.score for r in report.results]
    report.accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return report


def _keyword_judge(expected: str, predicted: str, judge_model=None, judge_processor=None) -> tuple[str, float]:
    """Judge answer quality using Gemma model as scorer (0-100)."""
    if judge_model is not None and judge_processor is not None:
        score = _model_score(expected, predicted, judge_model, judge_processor)
    else:
        score = _fallback_score(expected, predicted)

    if score >= 70:
        return "correct", score
    elif score >= 40:
        return "partial", score
    else:
        return "wrong", score


def _model_score(expected: str, predicted: str, model, processor) -> float:
    """Use Gemma model to score answer correctness 0-100."""
    import re
    try:
        from mlx_vlm import generate as mlx_generate
    except ImportError:
        return _fallback_score(expected, predicted)

    prompt_text = (
        f"判断「模型答案」是否正确回答了问题。标准答案提供参考。\n"
        f"标准答案：{expected}\n模型答案：{predicted}\n"
        f"评分：100=完全正确，75=基本正确但有遗漏，50=部分正确，25=大部分错误，0=完全错误。\n"
        f"只回复一个整数。"
    )
    msgs = [{"role": "user", "content": prompt_text}]
    p = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    out = mlx_generate(model, processor, p, max_tokens=16, verbose=False)
    text = out.text if hasattr(out, "text") else str(out)

    m = re.search(r'\d+', text.strip())
    if m:
        val = int(m.group())
        return min(val, 100)
    return _fallback_score(expected, predicted)


def _fallback_score(expected: str, predicted: str) -> float:
    """Fallback: ROUGE-L + entity overlap when model judge unavailable."""
    expected_clean = _normalize_text(expected)
    predicted_clean = _normalize_text(predicted)
    if not expected_clean:
        return 100.0 if len(predicted_clean) > 5 else 0.0

    lcs_len = _lcs_length(expected_clean, predicted_clean)
    recall = lcs_len / len(expected_clean) if expected_clean else 0.0
    entity_score = _entity_overlap(expected, predicted)
    return (0.6 * recall + 0.4 * entity_score) * 100


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
    text = re.sub(r'[，。、；：！？“”‘’【】（）《》\-—…·`\'\",.;:!?\[\](){}<>/\\|]', '', text)
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


def _load_eval_records(path: Path) -> list[tuple[str, str, str]]:
    """Load eval records as (question, expected_answer, system_context) tuples."""
    records: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages = data.get("messages", [])
            question = ""
            answer = ""
            system_ctx = ""
            for msg in messages:
                if msg.get("role") == "system":
                    system_ctx = msg.get("content", "")
                elif msg.get("role") == "user":
                    question = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    answer = msg.get("content", "")
            if question and answer:
                records.append((question, answer, system_ctx))
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


def _generate_answer(model, tokenizer, question: str, max_tokens: int, system_ctx: str = "") -> str:
    from mlx_lm import generate as mlx_gen

    system_content = system_ctx if system_ctx else "你是一个技术助手。基于技术文档准确回答问题。"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_gen(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


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
    print(f"  Avg Score: {report.accuracy:.1f}/100")
    all_scores = [r.score for r in report.results]
    if all_scores:
        print(f"  Min Score: {min(all_scores):.1f}")
        print(f"  >= 70:     {sum(1 for s in all_scores if s >= 70)}/{len(all_scores)}")
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
