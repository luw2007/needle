"""Tests for needle.finetune.export — no network calls, no real model loading."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


class TestExportPlan(unittest.TestCase):
    def test_dry_run_prints_plan(self):
        from needle.finetune.export import export_gemma

        with tempfile.TemporaryDirectory() as tmp:
            adapter_path = os.path.join(tmp, "adapters")
            os.makedirs(adapter_path)

            result = export_gemma(
                adapter_path=adapter_path,
                base_model_id="mlx-community/gemma-3-4b-it-4bit",
                dry_run=True,
            )
            self.assertIsNone(result)

    def test_plan_summary_contains_fields(self):
        from needle.finetune.export import ExportPlan

        plan = ExportPlan(
            adapter_path="/tmp/adapter",
            base_model_id="mlx-community/gemma-3-4b-it-4bit",
            output_dir="/tmp/output",
            repo_id="user/my-model",
            private=True,
        )
        summary = plan.summary()
        self.assertIn("/tmp/adapter", summary)
        self.assertIn("mlx-community/gemma-3-4b-it-4bit", summary)
        self.assertIn("user/my-model", summary)
        self.assertIn("True", summary)


class TestRequireMlxLm(unittest.TestCase):
    def test_missing_mlx_lm_raises_clear_error(self):
        from needle.finetune.export import _require_mlx_lm

        with patch.dict(sys.modules, {"mlx_lm": None}):
            with self.assertRaises(ImportError) as ctx:
                _require_mlx_lm()
            self.assertIn("mlx-lm", str(ctx.exception))
            self.assertIn("pip install", str(ctx.exception))


class TestFuseAndExport(unittest.TestCase):
    @patch("mlx.utils.tree_unflatten", return_value=[])
    @patch("mlx_lm.fuse.save")
    @patch("mlx_lm.load")
    @patch("needle.finetune.export._require_mlx_lm")
    def test_fuse_called_with_correct_args(self, mock_require, mock_load, mock_save, mock_unflatten):
        from needle.finetune.export import fuse_and_export

        mock_model = MagicMock()
        mock_model.named_modules.return_value = []
        mock_tokenizer = MagicMock()
        mock_config = {}
        mock_load.return_value = (mock_model, mock_tokenizer, mock_config)

        with tempfile.TemporaryDirectory() as tmp:
            adapter_path = os.path.join(tmp, "adapters")
            os.makedirs(adapter_path)
            Path(adapter_path, "adapters.safetensors").touch()

            output_dir = os.path.join(tmp, "output")

            result = fuse_and_export(
                adapter_path=adapter_path,
                base_model_id="mlx-community/gemma-3-4b-it-4bit",
                output_dir=output_dir,
            )

            resolved_adapter = str(Path(adapter_path).resolve())
            resolved_output = str(Path(output_dir).resolve())
            self.assertEqual(result, resolved_output)
            mock_load.assert_called_once_with(
                "mlx-community/gemma-3-4b-it-4bit",
                adapter_path=resolved_adapter,
                return_config=True,
            )
            mock_save.assert_called_once()

    def test_missing_adapter_path_raises(self):
        from needle.finetune.export import fuse_and_export

        with patch("needle.finetune.export._require_mlx_lm"):
            with self.assertRaises(FileNotFoundError):
                fuse_and_export(
                    adapter_path="/nonexistent/path",
                    base_model_id="test-model",
                    output_dir="/tmp/out",
                )


class TestPushToHub(unittest.TestCase):
    @patch("huggingface_hub.HfApi")
    def test_push_creates_repo_and_uploads(self, MockHfApi):
        from needle.finetune.export import push_to_hub

        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model.safetensors").touch()

            result = push_to_hub(
                output_dir=tmp,
                repo_id="user/test-model",
                base_model_id="mlx-community/gemma-3-4b-it-4bit",
                adapter_path="/tmp/adapters",
                private=True,
            )

            self.assertEqual(result, "https://huggingface.co/user/test-model")
            mock_api.create_repo.assert_called_once_with(
                "user/test-model", repo_type="model", private=True, exist_ok=True
            )
            mock_api.upload_folder.assert_called_once_with(
                folder_path=tmp,
                repo_id="user/test-model",
                repo_type="model",
            )
            self.assertTrue(os.path.exists(os.path.join(tmp, "README.md")))

    @patch("huggingface_hub.HfApi")
    def test_existing_readme_not_overwritten(self, MockHfApi):
        from needle.finetune.export import push_to_hub

        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmp:
            readme = os.path.join(tmp, "README.md")
            with open(readme, "w") as f:
                f.write("custom card")

            push_to_hub(
                output_dir=tmp,
                repo_id="user/test-model",
                base_model_id="base",
                adapter_path="/tmp/a",
            )

            with open(readme) as f:
                self.assertEqual(f.read(), "custom card")


class TestModelCard(unittest.TestCase):
    def test_card_contains_base_model(self):
        from needle.finetune.export import _generate_model_card

        card = _generate_model_card(
            base_model_id="mlx-community/gemma-3-4b-it-4bit",
            adapter_path="/tmp/adapters",
            repo_id="user/my-finetuned",
        )
        self.assertIn("mlx-community/gemma-3-4b-it-4bit", card)
        self.assertIn("user/my-finetuned", card)
        self.assertIn("needle", card)
        self.assertIn("lora", card)


class TestExportGemmaEndToEnd(unittest.TestCase):
    @patch("needle.finetune.export.push_to_hub")
    @patch("needle.finetune.export.fuse_and_export")
    def test_full_flow_with_push(self, mock_fuse, mock_push):
        from needle.finetune.export import export_gemma

        with tempfile.TemporaryDirectory() as tmp:
            adapter_path = os.path.join(tmp, "adapters")
            os.makedirs(adapter_path)
            output_dir = os.path.join(tmp, "out")

            mock_fuse.return_value = output_dir
            mock_push.return_value = "https://huggingface.co/user/model"

            result = export_gemma(
                adapter_path=adapter_path,
                base_model_id="base-model",
                output_dir=output_dir,
                repo_id="user/model",
                private=True,
            )

        self.assertEqual(result, "https://huggingface.co/user/model")
        mock_fuse.assert_called_once()
        mock_push.assert_called_once_with(
            output_dir,
            "user/model",
            base_model_id="base-model",
            adapter_path=adapter_path,
            private=True,
        )

    @patch("needle.finetune.export.fuse_and_export")
    def test_local_only_returns_output_dir(self, mock_fuse):
        from needle.finetune.export import export_gemma

        with tempfile.TemporaryDirectory() as tmp:
            adapter_path = os.path.join(tmp, "adapters")
            os.makedirs(adapter_path)
            output_dir = os.path.join(tmp, "out")
            mock_fuse.return_value = output_dir

            result = export_gemma(
                adapter_path=adapter_path,
                base_model_id="base-model",
                output_dir=output_dir,
            )

        self.assertEqual(result, output_dir)


class TestCLISubcommand(unittest.TestCase):
    def test_export_gemma_in_argparser(self):
        """Verify export-gemma subcommand is registered in CLI."""
        import argparse
        # Re-parse to check argparse accepts the subcommand
        with patch.object(sys, "argv", [
            "needle", "export-gemma", "/tmp/adapters",
            "--base-model", "test-model",
            "--dry-run",
        ]):
            from needle.cli import main
            with patch("needle.finetune.export.main") as mock_main:
                main()
                mock_main.assert_called_once()
                called_args = mock_main.call_args[0][0]
                self.assertEqual(called_args.adapter_path, "/tmp/adapters")
                self.assertEqual(called_args.base_model, "test-model")
                self.assertTrue(called_args.dry_run)


if __name__ == "__main__":
    unittest.main()
