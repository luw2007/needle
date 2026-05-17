"""Generate QA pairs from document chunks using local Gemma model."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from .chunk import Chunk

QA_SYSTEM_PROMPT = """你是一个技术文档QA数据生成专家。根据给定的技术文档片段，生成高质量的问答对。

要求：
1. 生成3-5个问答对（视内容丰富度而定）
2. 问题类型需多样化：
   - 事实型：直接从文档中可找到答案
   - 推理型：需要理解上下文才能回答
   - 应用型：如何使用/配置某个功能
3. 答案必须忠实于文档内容，不编造信息
4. 答案应完整，包含关键细节
5. 使用中文

输出格式（严格JSON数组）：
[
  {"question": "问题1", "answer": "答案1", "type": "factual"},
  {"question": "问题2", "answer": "答案2", "type": "reasoning"}
]

仅输出JSON数组，不要其他文字。"""


@dataclass
class QAPair:
    question: str
    answer: str
    qa_type: str
    source_doc: str
    source_heading: str
    chunk_index: int


def generate_qa_from_chunk(
    chunk: Chunk,
    model_id: str | None = None,
    max_tokens: int = 2048,
    num_retries: int = 2,
) -> list[QAPair]:
    from ..model.gemma import load_model

    model, processor = load_model(model_id)

    user_content = (
        f"{QA_SYSTEM_PROMPT}\n\n---\n"
        f"文档路径: {chunk.heading_path}\n\n"
        f"{chunk.content}\n---\n\n"
        f"请根据以上文档片段生成问答对。"
    )

    messages = [{"role": "user", "content": user_content}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    for attempt in range(num_retries + 1):
        try:
            from mlx_vlm import generate as mlx_generate

            output = mlx_generate(
                model, processor, prompt, max_tokens=max_tokens, verbose=False
            )
            if hasattr(output, 'text'):
                output = output.text
        except ImportError:
            raise ImportError(
                "mlx-vlm is required for QA generation.\n"
                "Install with: pip install 'needle[prepare]'"
            ) from None

        pairs = _parse_qa_output(output, chunk)
        if pairs:
            return pairs

        if attempt < num_retries:
            print(f"  Retry {attempt + 1}/{num_retries} for chunk {chunk.chunk_index}",
                  file=sys.stderr)

    return []


def _parse_qa_output(text: str, chunk: Chunk) -> list[QAPair]:
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return []

    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    pairs: list[QAPair] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        t = str(item.get("type", "factual")).strip()
        if q and a:
            pairs.append(QAPair(
                question=q,
                answer=a,
                qa_type=t if t in ("factual", "reasoning", "application") else "factual",
                source_doc=chunk.doc_path,
                source_heading=chunk.heading_path,
                chunk_index=chunk.chunk_index,
            ))

    return pairs


def generate_qa_batch(
    chunks: list[Chunk],
    model_id: str | None = None,
    max_tokens: int = 2048,
    progress: bool = True,
) -> list[QAPair]:
    all_pairs: list[QAPair] = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if progress:
            print(f"  [{i + 1}/{total}] {chunk.doc_path} — {chunk.heading_path[:50]}",
                  file=sys.stderr)

        pairs = generate_qa_from_chunk(chunk, model_id=model_id, max_tokens=max_tokens)
        all_pairs.extend(pairs)

        if progress:
            print(f"    → {len(pairs)} QA pairs generated", file=sys.stderr)

    return all_pairs
