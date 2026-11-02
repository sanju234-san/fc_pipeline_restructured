"""Wiring tests for the GRPO trainer (trl is stubbed — no torch needed).

``trl`` / ``transformers`` / ``datasets`` are optional research extras and are
not installed here, so ``run_grpo_training`` cannot actually train. To still
verify that the trainer is *wired correctly* (config mapping, reward adapter,
dataset hand-off, train() call, checkpoint save) we inject lightweight stub
modules into ``sys.modules``.

These tests do NOT train a model and do NOT prove learning happens; they prove
the integration points are correct. See test_outputs/grpo_audit.py for the
runtime findings.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Stub modules
# ---------------------------------------------------------------------------
class _StubGRPOConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, item):
        return self.kwargs.get(item)


class _StubGRPOTrainer:
    instances: list = []

    def __init__(self, model=None, args=None, train_dataset=None, reward_funcs=None,
                 processing_class=None, **kwargs):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.reward_funcs = reward_funcs
        self.processing_class = processing_class
        self.kwargs = kwargs
        self.trained = False
        self.saved_to: list[str] = []
        _StubGRPOTrainer.instances.append(self)

    def train(self):
        self.trained = True
        return {"train_loss": 0.0}

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")
        self.saved_to.append(path)


class _StubTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def _install_stubs(monkeypatch):
    trl = types.ModuleType("trl")
    trl.GRPOConfig = _StubGRPOConfig
    trl.GRPOTrainer = _StubGRPOTrainer
    monkeypatch.setitem(sys.modules, "trl", trl)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _StubTokenizer()
    )
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: object()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    datasets = types.ModuleType("datasets")

    class _Dataset(list):
        @classmethod
        def from_list(cls, records):
            return cls(records)

    datasets.Dataset = _Dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    _StubGRPOTrainer.instances = []
    return trl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_build_grpo_config_maps_group_size_to_num_generations(monkeypatch):
    _install_stubs(monkeypatch)
    from fc_pipeline.research.grpo_trainer import _build_grpo_config

    cfg = _build_grpo_config(
        output_dir="out", model_name="m", num_train_epochs=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=5e-6, max_new_tokens=128, group_size=8,
        logging_steps=10, save_steps=100, seed=1,
    )
    assert cfg.kwargs["num_generations"] == 8
    assert cfg.kwargs["output_dir"] == "out"
    assert cfg.kwargs["report_to"] == "none"


def test_reward_adapter_delegates_to_composite_reward(monkeypatch):
    _install_stubs(monkeypatch)
    from fc_pipeline.research.grpo_reward_fn import batch_composite_reward
    from fc_pipeline.research.grpo_trainer import _make_reward_fn

    fn = _make_reward_fn()
    prompts = ["Metric: PLI\nConnectivity value: 0.05"]
    completions = ['{"classification": "weak", "metrics": ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]}']
    out = fn(prompts, completions)
    assert out == batch_composite_reward(prompts, completions)
    assert isinstance(out[0], float)


def test_run_grpo_training_wires_trainer_and_saves(monkeypatch, tmp_path):
    _install_stubs(monkeypatch)
    from fc_pipeline.research.grpo_trainer import run_grpo_training

    out_dir = tmp_path / "grpo_run"
    run_grpo_training(
        model_name_or_path="stub-model",
        output_dir=str(out_dir),
        n_samples=12,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        group_size=4,
        save_steps=100,
    )

    assert len(_StubGRPOTrainer.instances) == 1
    trainer = _StubGRPOTrainer.instances[0]
    assert trainer.trained is True
    assert callable(trainer.reward_funcs)
    assert len(trainer.train_dataset) == 12
    # checkpoint is written to <output_dir>/final_model (+ tokenizer)
    final = out_dir / "final_model"
    assert (final / "config.json").exists()
    assert (final / "tokenizer_config.json").exists()


def test_check_deps_reports_all_missing(monkeypatch):
    for name in ("trl", "transformers", "datasets"):
        monkeypatch.setitem(sys.modules, name, None)
    from fc_pipeline.research.grpo_trainer import _check_deps

    # Setting a module to None makes `__import__` raise ImportError.
    with pytest.raises(ImportError) as ei:
        _check_deps()
    assert "trl" in str(ei.value)
