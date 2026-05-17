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
- "correct": 模型答案包含预期答案的核心信息，即使措辞不同
- "partial": 模型答案部分正确但遗漏关键信息
- "wrong": 模型答案错误或完全无关

仅输出一个词：correct 或 partial 或 wrong"""


def evaluate_qa(
    eval_jsonl: str | Path,
    model_id: str | None = None,
    judge_model_id: str | None = None,
    max_tokens: int = 1024,
    max_samples: int | None = None,
) -> EvalReport:
    from ..model.gemma import load_model

    eval_path = Path(eval_jsonl)
    records = _load_eval_records(eval_path)
    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    print(f"Evaluating {len(records)} QA pairs...", file=sys.stderr)

    answer_model, answer_proc = load_model(model_id)
    if judge_model_id and judge_model_id != model_id:
        judge_model, judge_proc = load_model(judge_model_id)
    else:
        judge_model, judge_proc = answer_model, answer_proc

    report = EvalReport(total=len(records))

    for i, (question, expected) in enumerate(records):
        predicted = _generate_answer(answer_model, answer_proc, question, max_tokens)
        judgment = _judge_answer(judge_model, judge_proc, question, expected, predicted)

        if judgment == "correct":
            score = 1.0
            report.correct += 1
        elif judgment == "partial":
            score = 0.5
            report.partial += 1
        else:
            score = 0.0
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


def _generate_answer(model, processor, question: str, max_tokens: int) -> str:
    try:
        from mlx_vlm import generate as mlx_generate
    except ImportError:
        raise ImportError(
            "mlx-vlm is required for QA evaluation.\n"
            "Install with: pip install 'needle[prepare]'"
        ) from None

    messages = [{"role": "user", "content": question}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = mlx_generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    return result.text if hasattr(result, 'text') else str(result)


def _judge_answer(model, processor, question: str, expected: str, predicted: str) -> str:
    try:
        from mlx_vlm import generate as mlx_generate
    except ImportError:
        raise ImportError(
            "mlx-vlm is required for QA evaluation.\n"
            "Install with: pip install 'needle[prepare]'"
        ) from None

    prompt_text = JUDGE_PROMPT.format(
        question=question, expected=expected, predicted=predicted
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = mlx_generate(model, processor, prompt, max_tokens=32, verbose=False)
    output = output.text if hasattr(output, 'text') else str(output)
    output = output.strip().lower()

    for label in ("correct", "partial", "wrong"):
        if label in output:
            return label
    return "wrong"


def main(args) -> None:
    report = evaluate_qa(
        eval_jsonl=args.eval_jsonl,
        model_id=getattr(args, "model", None),
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
