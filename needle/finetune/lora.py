from __future__ import annotations

import sys
import types
from pathlib import Path

from .config import FinetuneConfig


def run_finetune(cfg: FinetuneConfig) -> None:
    errors = cfg.validate()
    if errors:
        for e in errors:
            print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)

    cfg.print_plan()

    if cfg.dry_run:
        print("\n[dry-run] Validation passed. Would start training with the above config.")
        return

    try:
        import mlx_lm
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError:
        print(
            "Error: mlx-lm is required for Gemma fine-tuning.\n"
            "Install with: pip install 'needle[gemma-ft]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from mlx_lm.lora import run as mlx_lora_run

    args = types.SimpleNamespace(
        model=cfg.model,
        data=cfg.data,
        train=cfg.train,
        fine_tune_type=cfg.fine_tune_type,
        optimizer=cfg.optimizer,
        optimizer_config={"adam": {}, "adamw": {}, "muon": {}, "sgd": {}, "adafactor": {}},
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        batch_size=cfg.batch_size,
        iters=cfg.iters,
        val_batches=cfg.val_batches,
        steps_per_report=cfg.steps_per_report,
        steps_per_eval=cfg.steps_per_eval,
        save_every=cfg.save_every,
        max_seq_length=cfg.max_seq_length,
        adapter_path=cfg.adapter_path,
        resume_adapter_file=cfg.resume_adapter_file,
        grad_checkpoint=cfg.grad_checkpoint,
        grad_accumulation_steps=cfg.grad_accumulation_steps,
        seed=cfg.seed,
        num_layers=cfg.lora.num_layers,
        lora_parameters={"rank": cfg.lora.rank, "dropout": cfg.lora.dropout, "scale": cfg.lora.scale},
        test=cfg.test,
        test_batches=cfg.test_batches,
        mask_prompt=cfg.mask_prompt,
        report_to=None,
        project_name=None,
        config=None,
        clear_cache_threshold=0,
    )

    mlx_lora_run(args)
