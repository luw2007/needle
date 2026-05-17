from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoRAParams:
    rank: int = 8
    dropout: float = 0.0
    scale: float = 20.0
    num_layers: int = 16


@dataclass
class FinetuneConfig:
    model: str = "google/gemma-3-4b-it"
    data: str = ""
    train: bool = True
    fine_tune_type: str = "lora"
    optimizer: str = "adam"
    learning_rate: float = 1e-5
    lr_schedule: dict | None = None
    batch_size: int = 4
    iters: int = 1000
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 200
    save_every: int = 100
    max_seq_length: int = 2048
    adapter_path: str = "adapters"
    resume_adapter_file: str | None = None
    grad_checkpoint: bool = False
    grad_accumulation_steps: int = 1
    seed: int = 0
    test: bool = False
    test_batches: int = 500
    mask_prompt: bool = False
    lora: LoRAParams = field(default_factory=LoRAParams)
    dry_run: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.model:
            errors.append("model is required")
        if not self.data:
            errors.append("data is required (path to JSONL dir or HF dataset name)")
        if self.fine_tune_type not in ("lora", "dora", "full"):
            errors.append(f"fine_tune_type must be lora/dora/full, got {self.fine_tune_type}")
        if self.optimizer not in ("adam", "adamw", "muon", "sgd", "adafactor"):
            errors.append(f"unsupported optimizer: {self.optimizer}")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.iters < 1:
            errors.append("iters must be >= 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        if self.lora.rank < 1:
            errors.append("lora.rank must be >= 1")
        return errors

    def print_plan(self):
        print("=" * 60)
        print("Gemma Fine-tune Plan")
        print("=" * 60)
        print(f"  Model:            {self.model}")
        print(f"  Data:             {self.data}")
        print(f"  Fine-tune type:   {self.fine_tune_type}")
        print(f"  Optimizer:        {self.optimizer}")
        print(f"  Learning rate:    {self.learning_rate}")
        print(f"  Batch size:       {self.batch_size}")
        print(f"  Iterations:       {self.iters}")
        print(f"  Max seq length:   {self.max_seq_length}")
        print(f"  Adapter path:     {self.adapter_path}")
        print(f"  Grad checkpoint:  {self.grad_checkpoint}")
        print(f"  Grad accum steps: {self.grad_accumulation_steps}")
        if self.fine_tune_type in ("lora", "dora"):
            print(f"  LoRA rank:        {self.lora.rank}")
            print(f"  LoRA dropout:     {self.lora.dropout}")
            print(f"  LoRA scale:       {self.lora.scale}")
            print(f"  LoRA num_layers:  {self.lora.num_layers}")
        print(f"  Seed:             {self.seed}")
        print(f"  Mask prompt:      {self.mask_prompt}")
        print("=" * 60)


def load_config(config_path: str | None = None, overrides: dict[str, Any] | None = None) -> FinetuneConfig:
    raw: dict[str, Any] = {}
    if config_path:
        p = Path(config_path)
        if not p.exists():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        with open(p) as f:
            import yaml
            raw = yaml.safe_load(f) or {}

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                if isinstance(v, dict) and isinstance(raw.get(k), dict):
                    raw[k].update(v)
                else:
                    raw[k] = v

    lora_raw = raw.pop("lora", raw.pop("lora_parameters", None))
    lora = LoRAParams()
    if isinstance(lora_raw, dict):
        for k, v in lora_raw.items():
            if hasattr(lora, k):
                setattr(lora, k, type(getattr(lora, k))(v))

    cfg = FinetuneConfig(lora=lora)
    for k, v in raw.items():
        if hasattr(cfg, k) and k != "lora":
            default = getattr(cfg, k)
            if default is not None and not isinstance(v, type(default)):
                try:
                    v = type(default)(v)
                except (TypeError, ValueError):
                    pass
            setattr(cfg, k, v)

    return cfg
