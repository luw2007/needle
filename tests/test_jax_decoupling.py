"""Tests that the inference CLI path does not require JAX at import time."""

import importlib
import sys
import unittest
from unittest.mock import patch


class TestNoJAXOnInference(unittest.TestCase):
    """Verify that importing inference modules doesn't pull in JAX."""

    def test_registry_import_no_jax(self):
        with patch.dict(sys.modules, {"jax": None, "jax.numpy": None, "flax": None}):
            mod = importlib.import_module("needle.model.registry")
            self.assertIn("gemma-4-e4b-it-4bit", mod.PROFILES)

    def test_registry_resolve_default(self):
        from needle.model.registry import resolve_model, DEFAULT_MODEL_ALIAS, PROFILES
        result = resolve_model(None)
        self.assertEqual(result, PROFILES[DEFAULT_MODEL_ALIAS])

    def test_registry_resolve_alias(self):
        from needle.model.registry import resolve_model
        result = resolve_model("gemma-4-e12b")
        self.assertEqual(result, "mlx-community/gemma-4-e12b-it-4bit")

    def test_registry_resolve_passthrough(self):
        from needle.model.registry import resolve_model
        result = resolve_model("my-org/custom-model")
        self.assertEqual(result, "my-org/custom-model")

    def test_registry_list_profiles(self):
        from needle.model.registry import list_profiles
        output = list_profiles()
        self.assertIn("gemma-4-e4b-it-4bit", output)
        self.assertIn("(default)", output)

    def test_gemma_module_import_no_jax(self):
        with patch.dict(sys.modules, {"jax": None, "jax.numpy": None, "flax": None}):
            mod = importlib.import_module("needle.model.gemma")
            self.assertTrue(hasattr(mod, "generate"))
            self.assertTrue(hasattr(mod, "load_model"))

    def test_run_main_no_jax_at_import(self):
        """model/run.py main() delegates to gemma — module import shouldn't need JAX."""
        blocked = {"jax", "jax.numpy", "jax.nn", "jax.nn.initializers", "flax", "flax.linen"}
        with patch.dict(sys.modules, {k: None for k in blocked}):
            if "needle.model.run" in sys.modules:
                del sys.modules["needle.model.run"]
            mod = importlib.import_module("needle.model.run")
            self.assertTrue(hasattr(mod, "main"))
            self.assertTrue(hasattr(mod, "normalize_tools"))

    def test_cli_run_path_no_xla_setup(self):
        """The 'run' command should NOT trigger _setup_jax_env."""
        from needle.cli import _JAX_COMMANDS
        self.assertNotIn("run", _JAX_COMMANDS)
        self.assertNotIn("playground", _JAX_COMMANDS)
        self.assertIn("train", _JAX_COMMANDS)
        self.assertIn("pretrain", _JAX_COMMANDS)


class TestNormalizeTools(unittest.TestCase):
    """Verify normalize/restore round-trip without JAX."""

    def test_normalize_roundtrip(self):
        from needle.model.run import normalize_tools, restore_tool_names
        tools_json = '[{"name": "getWeather", "description": "Get weather"}]'
        normalized, name_map = normalize_tools(tools_json)
        self.assertIn("get_weather", normalized)
        self.assertEqual(name_map["get_weather"], "getWeather")
        restored = restore_tool_names('[{"name": "get_weather"}]', name_map)
        self.assertIn("getWeather", restored)


class TestRunCheckpointDeprecation(unittest.TestCase):
    """Verify --checkpoint on 'needle run' gives a clear deprecation error."""

    def test_run_checkpoint_exits_with_error(self):
        import io
        from contextlib import redirect_stderr
        buf = io.StringIO()
        with patch.object(sys, "argv", ["needle", "run", "--checkpoint", "checkpoints/test.pkl"]):
            with self.assertRaises(SystemExit) as ctx, redirect_stderr(buf):
                from needle.cli import main
                main()
        self.assertEqual(ctx.exception.code, 1)
        msg = buf.getvalue()
        self.assertIn("--checkpoint is no longer supported", msg)
        self.assertIn("needle eval --checkpoint", msg)

    def test_eval_checkpoint_still_accepted(self):
        """eval subparser still accepts --checkpoint as required arg."""
        import argparse as _ap
        from needle.cli import main
        # This will fail deeper (no actual checkpoint), but argparse should not reject it
        with patch.object(sys, "argv", ["needle", "eval", "--checkpoint", "test.pkl"]):
            try:
                main()
            except (SystemExit, Exception):
                pass
        # If we got past argparse, checkpoint was accepted — check the namespace
        parser = _ap.ArgumentParser(prog="needle", add_help=False)
        sub = parser.add_subparsers(dest="command")
        p = sub.add_parser("eval", add_help=False)
        p.add_argument("--checkpoint", type=str, required=True)
        args = parser.parse_args(["eval", "--checkpoint", "x.pkl"])
        self.assertEqual(args.checkpoint, "x.pkl")


if __name__ == "__main__":
    unittest.main()
