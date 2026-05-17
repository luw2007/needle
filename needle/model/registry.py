"""Model profile registry — single source of truth for supported model aliases."""

import os

DEFAULT_MODEL_ALIAS = "gemma-4-e4b-it-4bit"

PROFILES = {
    "gemma-4-e4b-it-4bit": "mlx-community/gemma-4-e4b-it-4bit",
    "gemma-4-e4b-it": "mlx-community/gemma-4-e4b-it-4bit",
    "gemma-4-e4b": "mlx-community/gemma-4-e4b-it-4bit",
    "gemma-4-e12b-it-4bit": "mlx-community/gemma-4-e12b-it-4bit",
    "gemma-4-e12b-it": "mlx-community/gemma-4-e12b-it-4bit",
    "gemma-4-e12b": "mlx-community/gemma-4-e12b-it-4bit",
}


def resolve_model(model_id: str | None) -> str:
    """Resolve a model alias or raw HF repo to a loadable model ID.

    Resolution order:
      1. Explicit model_id argument (if provided)
      2. NEEDLE_MODEL environment variable
      3. DEFAULT_MODEL_ALIAS

    The resolved value is then looked up in PROFILES; if not found,
    it is returned as-is (assumed to be a raw HF repo path).
    """
    model_id = model_id or os.environ.get("NEEDLE_MODEL") or DEFAULT_MODEL_ALIAS
    return PROFILES.get(model_id, model_id)


def list_profiles() -> str:
    """Return a formatted string listing available model profiles."""
    lines = ["Available model profiles:"]
    for alias, repo in PROFILES.items():
        marker = " (default)" if alias == DEFAULT_MODEL_ALIAS else ""
        lines.append(f"  {alias:30s} -> {repo}{marker}")
    lines.append("")
    lines.append("Any HuggingFace repo ID is also accepted directly.")
    lines.append("Override default via NEEDLE_MODEL env var.")
    return "\n".join(lines)
