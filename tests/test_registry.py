"""Tests for needle.model.registry."""

import importlib.util
import os

_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "needle", "model", "registry.py"
)
_spec = importlib.util.spec_from_file_location("needle.model.registry", _REGISTRY_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

resolve_model = _mod.resolve_model
list_profiles = _mod.list_profiles
PROFILES = _mod.PROFILES
DEFAULT_MODEL_ALIAS = _mod.DEFAULT_MODEL_ALIAS


def test_default_resolution():
    assert resolve_model(None) == "mlx-community/gemma-4-e4b-it-4bit"


def test_known_aliases():
    assert resolve_model("gemma-4-e4b-it-4bit") == "mlx-community/gemma-4-e4b-it-4bit"
    assert resolve_model("gemma-4-e4b-it") == "mlx-community/gemma-4-e4b-it-4bit"
    assert resolve_model("gemma-4-e4b") == "mlx-community/gemma-4-e4b-it-4bit"
    assert resolve_model("gemma-4-e12b-it-4bit") == "mlx-community/gemma-4-e12b-it-4bit"
    assert resolve_model("gemma-4-e12b-it") == "mlx-community/gemma-4-e12b-it-4bit"
    assert resolve_model("gemma-4-e12b") == "mlx-community/gemma-4-e12b-it-4bit"


def test_raw_hf_passthrough():
    assert resolve_model("org/custom-model") == "org/custom-model"
    assert resolve_model("mlx-community/some-other-model") == "mlx-community/some-other-model"


def test_env_var_fallback(monkeypatch):
    monkeypatch.setenv("NEEDLE_MODEL", "gemma-4-e12b")
    assert resolve_model(None) == "mlx-community/gemma-4-e12b-it-4bit"


def test_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("NEEDLE_MODEL", "gemma-4-e12b")
    assert resolve_model("gemma-4-e4b") == "mlx-community/gemma-4-e4b-it-4bit"


def test_env_var_raw_passthrough(monkeypatch):
    monkeypatch.setenv("NEEDLE_MODEL", "org/from-env")
    assert resolve_model(None) == "org/from-env"


def test_list_profiles_output():
    output = list_profiles()
    assert "gemma-4-e4b-it-4bit" in output
    assert "NEEDLE_MODEL" in output
    assert "(default)" in output


def test_default_alias_in_profiles():
    assert DEFAULT_MODEL_ALIAS in PROFILES
