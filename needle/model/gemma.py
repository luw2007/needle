"""Gemma-4-E4B-it inference via mlx-vlm."""

import json
import re
import sys

from .registry import resolve_model

_cache = {}


def load_model(model_id=None):
    model_id = resolve_model(model_id)
    if model_id in _cache:
        return _cache[model_id]
    try:
        from mlx_vlm import load
    except ImportError:
        raise ImportError(
            "mlx-vlm is required for Gemma inference but not installed.\n"
            "Install it with: pip install 'needle[mlx]'"
        ) from None

    print(f"Loading model: {model_id}", file=sys.stderr)
    model, processor = load(model_id)
    _cache[model_id] = (model, processor)
    return model, processor


def _convert_tool(t):
    params = t.get("parameters", {})
    properties = {}
    required = []
    for k, v in params.items():
        prop = {"type": v.get("type", "string")}
        if "description" in v:
            prop["description"] = v["description"]
        properties[k] = prop
        if v.get("required", False):
            required.append(k)
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                **({"required": required} if required else {}),
            },
        },
    }


def convert_tools(tools_json):
    tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
    return [_convert_tool(t) for t in tools]


def build_prompt(processor, query, tools_json):
    openai_tools = convert_tools(tools_json)
    messages = [{"role": "user", "content": query}]
    return processor.tokenizer.apply_chat_template(
        messages, tools=openai_tools, tokenize=False, add_generation_prompt=True
    )


def parse_tool_calls(text):
    pattern = r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>"
    calls = []
    for match in re.finditer(pattern, text, re.DOTALL):
        name = match.group(1)
        args_body = match.group(2)
        # Parse key:<|"|>value<|"|> format into dict
        arguments = {}
        kv_pattern = r"(\w+):<\|\"\|>(.*?)<\|\"\|>"
        for kv in re.finditer(kv_pattern, args_body):
            arguments[kv.group(1)] = kv.group(2)
        if not arguments:
            # Fallback: try standard JSON
            try:
                arguments = json.loads("{" + args_body + "}")
            except json.JSONDecodeError:
                pass
        calls.append({"name": name, "arguments": arguments})
    return calls


def generate(query, tools="[]", model_id=None, max_tokens=512, stream=True):
    try:
        from mlx_vlm import generate as mlx_generate
    except ImportError:
        raise ImportError(
            "mlx-vlm is required for Gemma inference but not installed.\n"
            "Install it with: pip install 'needle[mlx]'"
        ) from None

    model, processor = load_model(model_id)
    prompt = build_prompt(processor, query, tools)

    result = mlx_generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    raw_text = result.text if hasattr(result, "text") else str(result)

    calls = parse_tool_calls(raw_text)
    output = json.dumps(calls, ensure_ascii=False)

    if stream:
        print(f"\n<tool_call>{output}")
        sys.stdout.flush()

    return output


def generate_no_stream(query, tools="[]", model_id=None, max_tokens=512):
    return generate(query, tools=tools, model_id=model_id, max_tokens=max_tokens, stream=False)
