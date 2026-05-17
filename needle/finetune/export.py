"""Fuse LoRA adapters and export to HuggingFace-compatible safetensors."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExportPlan:
    adapter_path: str
    base_model_id: str
    output_dir: str
    repo_id: Optional[str]
    private: bool

    def summary(self) -> str:
        lines = [
            "Export plan:",
            f"  adapter_path : {self.adapter_path}",
            f"  base_model   : {self.base_model_id}",
            f"  output_dir   : {self.output_dir}",
            f"  hf_repo      : {self.repo_id or '(local only)'}",
            f"  private       : {self.private}",
        ]
        return "\n".join(lines)


def _require_mlx_lm():
    try:
        import mlx_lm  # noqa: F401
        return mlx_lm
    except ImportError:
        raise ImportError(
            "mlx-lm is required for LoRA export but not installed.\n"
            "Install it with: pip install mlx-lm"
        ) from None


def fuse_and_export(
    adapter_path: str,
    base_model_id: str,
    output_dir: str,
    *,
    de_quantize: bool = False,
) -> str:
    """Fuse LoRA adapter into base model and save as safetensors.

    Returns the output directory path.
    """
    mlx_lm = _require_mlx_lm()

    adapter_path = str(Path(adapter_path).resolve())
    output_dir = str(Path(output_dir).resolve())

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    os.makedirs(output_dir, exist_ok=True)

    from mlx_lm import load as mlx_load
    from mlx_lm.fuse import save, dequantize_model
    from mlx.utils import tree_unflatten

    model, tokenizer, config = mlx_load(
        base_model_id, adapter_path=adapter_path, return_config=True
    )

    fused_linears = [
        (n, m.fuse(dequantize=de_quantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    if de_quantize:
        model = dequantize_model(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)

    save(Path(output_dir), base_model_id, model, tokenizer, config, donate_model=False)

    print(f"Fused model saved to: {output_dir}")
    return output_dir


def _generate_model_card(
    base_model_id: str,
    adapter_path: str,
    repo_id: Optional[str] = None,
) -> str:
    """Generate a minimal model card for the fused model."""
    title = repo_id or Path(adapter_path).name
    return f"""---
library_name: mlx
base_model: {base_model_id}
tags:
- needle
- mlx
- lora
- gemma
---

# {title}

Fine-tuned with [needle](https://github.com/Cactus-Compute/needle) using MLX LoRA.

## Base model

`{base_model_id}`

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_id or title}")
response = generate(model, tokenizer, prompt="your prompt here", max_tokens=512)
```
"""


def push_to_hub(
    output_dir: str,
    repo_id: str,
    *,
    base_model_id: str = "",
    adapter_path: str = "",
    private: bool = False,
) -> str:
    """Upload fused model directory to HuggingFace Hub.

    Returns the repo URL.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    readme_path = os.path.join(output_dir, "README.md")
    if not os.path.exists(readme_path):
        card = _generate_model_card(base_model_id, adapter_path, repo_id)
        with open(readme_path, "w") as f:
            f.write(card)

    print(f"Uploading to {repo_id} ...")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Pushed to: {url}")
    return url


def export_gemma(
    adapter_path: str,
    base_model_id: str,
    *,
    output_dir: Optional[str] = None,
    repo_id: Optional[str] = None,
    private: bool = False,
    dry_run: bool = False,
    de_quantize: bool = False,
) -> Optional[str]:
    """End-to-end: fuse LoRA, optionally push to HF Hub.

    Returns the HF repo URL if pushed, the output directory if local-only,
    or None for dry-run.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(adapter_path), "fused_model"
        )

    plan = ExportPlan(
        adapter_path=adapter_path,
        base_model_id=base_model_id,
        output_dir=output_dir,
        repo_id=repo_id,
        private=private,
    )
    print(plan.summary())

    if dry_run:
        print("\n(dry-run — no files written)")
        return None

    fuse_and_export(
        adapter_path,
        base_model_id,
        output_dir,
        de_quantize=de_quantize,
    )

    if repo_id:
        return push_to_hub(
            output_dir,
            repo_id,
            base_model_id=base_model_id,
            adapter_path=adapter_path,
            private=private,
        )

    return output_dir


def main(args):
    """CLI entry point for `needle export-gemma`."""
    export_gemma(
        adapter_path=args.adapter_path,
        base_model_id=args.base_model,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        private=args.private,
        dry_run=args.dry_run,
        de_quantize=args.de_quantize,
    )
