"""Tests for needle.finetune.data — SFT data validation and conversion."""

import json
import tempfile
from pathlib import Path

import pytest

from needle.finetune.data import (
    LABEL_IGNORE_INDEX,
    ValidationError,
    convert_legacy_jsonl,
    legacy_to_messages,
    validate_jsonl,
    validate_record,
    _normalize_tool_call,
    _find_subsequence,
    _format_messages_for_template,
    _merge_consecutive_roles,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestValidateRecord:
    def test_valid_simple_conversation(self):
        record = {"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]}
        errors = validate_record(record, 1)
        assert errors == []

    def test_valid_with_tool_calls(self):
        record = {"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "tool_calls": [{"name": "get_weather", "arguments": {"city": "NYC"}}], "content": ""},
            {"role": "tool", "content": '{"temp": 20}'},
            {"role": "assistant", "content": "It's 20°C."},
        ]}
        errors = validate_record(record, 1)
        assert errors == []

    def test_missing_messages(self):
        errors = validate_record({}, 1)
        assert len(errors) == 1
        assert "non-empty" in errors[0].message

    def test_empty_messages(self):
        errors = validate_record({"messages": []}, 1)
        assert len(errors) == 1

    def test_invalid_role(self):
        record = {"messages": [
            {"role": "invalid", "content": "x"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]}
        errors = validate_record(record, 1)
        assert any("invalid role" in e.message for e in errors)

    def test_assistant_no_content_no_tool_calls(self):
        record = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant"},
        ]}
        errors = validate_record(record, 1)
        assert any("content or tool_calls" in e.message for e in errors)

    def test_user_no_content(self):
        record = {"messages": [
            {"role": "user"},
            {"role": "assistant", "content": "a"},
        ]}
        errors = validate_record(record, 1)
        assert any("must have content" in e.message for e in errors)

    def test_tool_call_missing_name(self):
        record = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"bad": True}], "content": ""},
        ]}
        errors = validate_record(record, 1)
        assert any("missing name/function" in e.message for e in errors)

    def test_tool_call_with_function_key(self):
        record = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "foo", "arguments": {}}}], "content": ""},
        ]}
        errors = validate_record(record, 1)
        assert errors == []

    def test_no_user_message(self):
        record = {"messages": [
            {"role": "assistant", "content": "hi"},
        ]}
        errors = validate_record(record, 1)
        assert any("user message" in e.message for e in errors)

    def test_no_assistant_message(self):
        record = {"messages": [
            {"role": "user", "content": "q"},
        ]}
        errors = validate_record(record, 1)
        assert any("assistant message" in e.message for e in errors)

    def test_system_role_valid(self):
        record = {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]}
        errors = validate_record(record, 1)
        assert errors == []


class TestValidateJsonl:
    def test_valid_file(self):
        errors = validate_jsonl(FIXTURES / "sft_messages.jsonl")
        assert errors == []

    def test_invalid_file(self):
        errors = validate_jsonl(FIXTURES / "sft_invalid.jsonl")
        assert len(errors) > 0
        line_nums = {e.line for e in errors}
        assert 2 in line_nums  # invalid JSON
        assert 1 in line_nums  # missing assistant


class TestNormalizeToolCall:
    def test_simple_format(self):
        tc = {"name": "foo", "arguments": {"a": 1}}
        result = _normalize_tool_call(tc)
        assert result == {"name": "foo", "arguments": {"a": 1}}

    def test_function_format(self):
        tc = {"function": {"name": "bar", "arguments": {"x": "y"}}}
        result = _normalize_tool_call(tc)
        assert result == {"name": "bar", "arguments": {"x": "y"}}

    def test_string_arguments(self):
        tc = {"name": "baz", "arguments": '{"k": "v"}'}
        result = _normalize_tool_call(tc)
        assert result == {"name": "baz", "arguments": {"k": "v"}}

    def test_invalid_string_arguments(self):
        tc = {"name": "baz", "arguments": "not json"}
        result = _normalize_tool_call(tc)
        assert result == {"name": "baz", "arguments": {}}


class TestLegacyToMessages:
    def test_basic_conversion(self):
        record = {
            "query": "What's the weather?",
            "tools": [{"name": "get_weather", "description": "Get weather", "parameters": {}}],
            "answers": [{"name": "get_weather", "arguments": {"city": "NYC"}}],
        }
        result = legacy_to_messages(record)
        msgs = result["messages"]
        assert any(m["role"] == "system" for m in msgs)
        assert any(m["role"] == "user" for m in msgs)
        assert any(m["role"] == "assistant" for m in msgs)
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["name"] == "get_weather"

    def test_string_tools_and_answers(self):
        record = {
            "query": "test",
            "tools": json.dumps([{"name": "foo", "parameters": {}}]),
            "answers": json.dumps([{"name": "foo", "arguments": {"x": 1}}]),
        }
        result = legacy_to_messages(record)
        msgs = result["messages"]
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert assistant_msg["tool_calls"][0]["name"] == "foo"

    def test_no_tools(self):
        record = {"query": "hello", "answers": []}
        result = legacy_to_messages(record)
        msgs = result["messages"]
        assert not any(m["role"] == "system" for m in msgs)

    def test_empty_answers(self):
        record = {"query": "hi", "tools": [], "answers": None}
        result = legacy_to_messages(record)
        msgs = result["messages"]
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "tool_calls" not in assistant_msg


class TestConvertLegacyJsonl:
    def test_convert_fixture(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name

        count = convert_legacy_jsonl(FIXTURES / "sft_legacy.jsonl", output_path)
        assert count == 3

        with open(output_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) == 3
        for r in records:
            assert "messages" in r
            errors = validate_record(r, 1)
            assert errors == [], f"Converted record has errors: {errors}"

        Path(output_path).unlink()

    def test_converted_records_have_correct_structure(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name

        convert_legacy_jsonl(FIXTURES / "sft_legacy.jsonl", output_path)

        with open(output_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        first = records[0]
        msgs = first["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "weather" in user_msg["content"].lower()

        Path(output_path).unlink()


class TestFormatMessages:
    def test_merge_consecutive_roles(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        merged = _merge_consecutive_roles(msgs)
        assert len(merged) == 2
        assert merged[0]["content"] == "a\nb"

    def test_format_tool_calls_to_text(self):
        msgs = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "tool_calls": [{"name": "foo", "arguments": {}}], "content": ""},
        ]
        formatted = _format_messages_for_template(msgs)
        assert all(m["role"] in ("user", "assistant") for m in formatted)
        assert "[TOOL_CALLS]" in formatted[-1]["content"]

    def test_system_mapped_to_user(self):
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        formatted = _format_messages_for_template(msgs)
        assert formatted[0]["role"] == "user"

    def test_tool_mapped_to_user(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"name": "f", "arguments": {}}], "content": ""},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        formatted = _format_messages_for_template(msgs)
        roles = [m["role"] for m in formatted]
        assert "tool" not in roles


class TestFindSubsequence:
    def test_found(self):
        assert _find_subsequence([1, 2, 3, 4, 5], [3, 4]) == 2

    def test_not_found(self):
        assert _find_subsequence([1, 2, 3], [4, 5]) == -1

    def test_empty_subseq(self):
        assert _find_subsequence([1, 2], []) == -1

    def test_at_start(self):
        assert _find_subsequence([1, 2, 3], [1, 2]) == 0

    def test_at_end(self):
        assert _find_subsequence([1, 2, 3], [2, 3]) == 1
