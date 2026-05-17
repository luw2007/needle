"""SFT data validation, conversion, and tokenization for Gemma fine-tuning.

Supports two formats:
  - messages-based JSONL: {"messages": [{"role": "user"|"assistant"|"tool", ...}]}
  - legacy needle JSONL: {"query": str, "tools": [...], "answers": [...]}

Validation is pure Python + json (no mlx-lm dependency).
Tokenization via Gemma's apply_chat_template requires transformers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_ROLES = {"user", "assistant", "tool", "system"}


class ValidationError:
    """Single validation issue for a JSONL record."""

    def __init__(self, line: int, message: str):
        self.line = line
        self.message = message

    def __repr__(self):
        return f"ValidationError(line={self.line}, message={self.message!r})"


def validate_message(msg: dict[str, Any], line: int, idx: int) -> list[ValidationError]:
    """Validate a single message dict within a conversation."""
    errors: list[ValidationError] = []
    if not isinstance(msg, dict):
        errors.append(ValidationError(line, f"messages[{idx}]: not a dict"))
        return errors

    role = msg.get("role")
    if role not in VALID_ROLES:
        errors.append(ValidationError(line, f"messages[{idx}]: invalid role {role!r}"))

    if role == "assistant":
        has_content = bool(msg.get("content"))
        has_tool_calls = bool(msg.get("tool_calls"))
        if not has_content and not has_tool_calls:
            errors.append(ValidationError(
                line, f"messages[{idx}]: assistant must have content or tool_calls"))
        if has_tool_calls:
            for tc_idx, tc in enumerate(msg["tool_calls"]):
                if not isinstance(tc, dict):
                    errors.append(ValidationError(
                        line, f"messages[{idx}].tool_calls[{tc_idx}]: not a dict"))
                    continue
                if "name" not in tc and "function" not in tc:
                    errors.append(ValidationError(
                        line, f"messages[{idx}].tool_calls[{tc_idx}]: missing name/function"))
    elif role in ("user", "system"):
        if not msg.get("content"):
            errors.append(ValidationError(
                line, f"messages[{idx}]: {role} message must have content"))
    elif role == "tool":
        if "content" not in msg:
            errors.append(ValidationError(
                line, f"messages[{idx}]: tool message must have content"))

    return errors


def validate_record(record: dict[str, Any], line: int) -> list[ValidationError]:
    """Validate a single messages-based JSONL record."""
    errors: list[ValidationError] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        errors.append(ValidationError(line, "record must have non-empty 'messages' list"))
        return errors

    for idx, msg in enumerate(messages):
        errors.extend(validate_message(msg, line, idx))

    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if "user" not in roles:
        errors.append(ValidationError(line, "conversation must contain at least one user message"))
    if "assistant" not in roles:
        errors.append(ValidationError(line, "conversation must contain at least one assistant message"))

    return errors


def validate_jsonl(path: str | Path) -> list[ValidationError]:
    """Validate all records in a messages-based JSONL file."""
    errors: list[ValidationError] = []
    path = Path(path)
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(ValidationError(line_num, f"invalid JSON: {e}"))
                continue
            errors.extend(validate_record(record, line_num))
    return errors


def _normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool_call to {name, arguments} form."""
    if "function" in tc:
        fn = tc["function"]
        return {"name": fn.get("name", ""), "arguments": fn.get("arguments", {})}
    args = tc.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    return {"name": tc.get("name", ""), "arguments": args}


def legacy_to_messages(record: dict[str, Any]) -> dict[str, list[dict]]:
    """Convert a legacy needle record to messages format.

    Legacy format: {"query": str, "tools": [...], "answers": [...]}
    Output: {"messages": [{"role": ..., "content": ...}, ...]}
    """
    messages: list[dict[str, Any]] = []

    tools = record.get("tools")
    if tools:
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except (json.JSONDecodeError, TypeError):
                tools = []
        tools_desc = json.dumps(tools, separators=(",", ":"))
        messages.append({"role": "system", "content": f"Available tools: {tools_desc}"})

    query = record.get("query", "")
    messages.append({"role": "user", "content": query})

    answers = record.get("answers")
    if answers:
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except (json.JSONDecodeError, TypeError):
                answers = []
        if isinstance(answers, list) and len(answers) > 0:
            tool_calls = [_normalize_tool_call(a) for a in answers if isinstance(a, dict)]
            if tool_calls:
                messages.append({"role": "assistant", "tool_calls": tool_calls, "content": ""})
            else:
                messages.append({"role": "assistant", "content": json.dumps(answers)})
        else:
            messages.append({"role": "assistant", "content": str(answers)})
    else:
        messages.append({"role": "assistant", "content": ""})

    return {"messages": messages}


def convert_legacy_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    """Convert a legacy needle JSONL file to messages-based JSONL.

    Returns the number of converted records.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            converted = legacy_to_messages(record)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def messages_to_training_pairs(
    path: str | Path,
    tokenizer=None,
    max_seq_len: int = 2048,
) -> list[dict[str, Any]]:
    """Convert messages JSONL to training input/target pairs via apply_chat_template.

    Each record produces one training example with keys:
      - input_ids: list[int] (full tokenized conversation)
      - labels: list[int] (masked except assistant turns)
      - length: int

    Requires a Gemma tokenizer with apply_chat_template support.
    If tokenizer is None, imports and loads the default Gemma tokenizer.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-3-4b-it", trust_remote_code=True
        )

    path = Path(path)
    pairs: list[dict[str, Any]] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if not messages:
                continue

            formatted = _format_messages_for_template(messages)

            full_ids = tokenizer.apply_chat_template(
                formatted, tokenize=True, add_generation_prompt=False
            )
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]

            labels = _build_labels(messages, tokenizer, full_ids, max_seq_len)

            pairs.append({
                "input_ids": full_ids,
                "labels": labels,
                "length": len(full_ids),
            })

    return pairs


def _format_messages_for_template(messages: list[dict]) -> list[dict]:
    """Format messages for Gemma's apply_chat_template.

    Converts tool_calls to text content and normalizes roles.
    """
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            role = "user"

        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_text = json.dumps(tool_calls, separators=(",", ":"))
            content = f"[TOOL_CALLS]{tc_text}" if not content else f"{content}\n[TOOL_CALLS]{tc_text}"

        if role == "system":
            role = "user"

        formatted.append({"role": role, "content": content})

    return _merge_consecutive_roles(formatted)


def _merge_consecutive_roles(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role (required by some templates)."""
    if not messages:
        return []
    merged = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg.copy())
    return merged


LABEL_IGNORE_INDEX = -100


def _build_labels(
    messages: list[dict],
    tokenizer,
    full_ids: list[int],
    max_seq_len: int,
) -> list[int]:
    """Build labels masking non-assistant tokens with LABEL_IGNORE_INDEX.

    Tokenizes assistant turns separately to find their positions in full_ids.
    """
    labels = [LABEL_IGNORE_INDEX] * len(full_ids)

    assistant_contents: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_text = json.dumps(tool_calls, separators=(",", ":"))
                content = f"[TOOL_CALLS]{tc_text}" if not content else f"{content}\n[TOOL_CALLS]{tc_text}"
            assistant_contents.append(content)

    for ac in assistant_contents:
        if not ac:
            continue
        ac_ids = tokenizer.encode(ac, add_special_tokens=False)
        start = _find_subsequence(full_ids, ac_ids)
        if start >= 0:
            end = min(start + len(ac_ids), len(full_ids))
            for i in range(start, end):
                labels[i] = full_ids[i]

    return labels


def _find_subsequence(seq: list[int], subseq: list[int]) -> int:
    """Find first occurrence of subseq in seq. Returns -1 if not found."""
    if not subseq:
        return -1
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            return i
    return -1
