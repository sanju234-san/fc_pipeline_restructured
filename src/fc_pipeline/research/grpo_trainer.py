"""GRPO offline fine-tuning trainer for the EEG Supervisor LLM.

This module lives entirely inside ``fc_pipeline.research`` and is **never**
imported by the Supervisor, the LangGraph pipeline, or any deterministic node.
It is an **offline research tool** — run it from the command line or a notebook
to produce a fine-tuned model checkpoint, then update ``SUPERVISOR_LLM_MODEL``
in ``.env`` to point the live pipeline at the new model.

Dependencies (install via requirements-research.txt)::

    pip install -r requirements-research.txt
    # adds: trl>=0.8, transformers>=4.40, datasets>=2.18

Quick start::

    python -m fc_pipeline.research.grpo_trainer \\
        --model_name_or_path "Qwen/Qwen2-0.5B-Instruct" \\
        --output_dir ./outputs/grpo_run_001 \\
        --n_samples 400 \\
        --num_train_epochs 2

What it optimises
-----------------
Three reward signals (see ``grpo_reward_fn``):
  1. Metric classification accuracy (50% weight) — uses TAU_PHASE / TAU_ZEROLAG
  2. Valid metric set naming (30% weight) — must use only the 5 canonical names
  3. JSON format compliance (20% weight) — expects {classification, metrics}
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guard: trl must be importable
# ---------------------------------------------------------------------------
def _check_deps() -> None:
    missing = []
    for pkg in ("trl", "transformers", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"GRPO trainer requires: {', '.join(missing)}.\n"
            "Install them with:  pip install -r requirements-research.txt"
        )


# ---------------------------------------------------------------------------
# Reward adapter (trl GRPOTrainer expects a specific callable signature)
# ---------------------------------------------------------------------------

def _make_reward_fn():
    """Return a trl-compatible reward function closure."""
    from fc_pipeline.research.grpo_reward_fn import batch_composite_reward

    def reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs,  # trl may pass extra metadata
    ) -> List[float]:
        return batch_composite_reward(prompts, completions)

    return reward_fn


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

def _build_grpo_config(
    output_dir: str,
    model_name: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_new_tokens: int,
    group_size: int,
    logging_steps: int,
    save_steps: int,
    seed: int,
) -> Any:
    """Build and return a ``trl.GRPOConfig`` instance."""
    try:
        from trl import GRPOConfig
    except ImportError as exc:
        raise ImportError("trl is required: pip install trl>=0.8") from exc

    return GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_new_tokens=max_new_tokens,
        num_generations=group_size,        # G in GRPO — completions per prompt
        logging_steps=logging_steps,
        save_steps=save_steps,
        seed=seed,
        report_to="none",                  # disable W&B / HF Hub logging by default
        remove_unused_columns=False,
    )


# ---------------------------------------------------------------------------
# Main trainer entry point
# ---------------------------------------------------------------------------

def run_grpo_training(
    model_name_or_path: str,
    output_dir: str,
    n_samples: int = 400,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    max_new_tokens: int = 128,
    group_size: int = 4,
    logging_steps: int = 10,
    save_steps: int = 100,
    seed: int = 42,
    dataset_seed: int = 42,
) -> None:
    """Run the GRPO fine-tuning loop.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path.  A small instruction-tuned model
        (e.g. ``Qwen/Qwen2-0.5B-Instruct`` or ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``)
        works well for testing on a laptop without a GPU.
    output_dir : str
        Directory where checkpoints and the final model are saved.
    n_samples : int
        Number of synthetic EEG scenarios to generate.
    num_train_epochs : int
        Full passes over the dataset.
    per_device_train_batch_size : int
        Batch size per GPU/CPU device.
    gradient_accumulation_steps : int
        Accumulate gradients over this many steps before an optimizer step.
    learning_rate : float
        AdamW learning rate.
    max_new_tokens : int
        Maximum tokens the model may generate per completion.
    group_size : int
        Number of completions sampled per prompt in GRPO (``G`` in the paper).
    logging_steps : int
        Log metrics every N steps.
    save_steps : int
        Save a checkpoint every N steps.
    seed : int
        Trainer seed.
    dataset_seed : int
        Seed for the synthetic dataset generator.
    """
    _check_deps()

    from trl import GRPOTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from fc_pipeline.research.grpo_dataset import build_grpo_dataset

    logger.info("Building synthetic EEG scenario dataset (n=%d, seed=%d)", n_samples, dataset_seed)
    dataset = build_grpo_dataset(n_samples=n_samples, seed=dataset_seed, use_hf_datasets=True)
    logger.info("Dataset ready: %d records", len(dataset))

    logger.info("Loading tokenizer and model: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    config = _build_grpo_config(
        output_dir=output_dir,
        model_name=model_name_or_path,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_new_tokens=max_new_tokens,
        group_size=group_size,
        logging_steps=logging_steps,
        save_steps=save_steps,
        seed=seed,
    )

    reward_fn = _make_reward_fn()

    logger.info("Initialising GRPOTrainer …")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training — output dir: %s", output_dir)
    trainer.train()

    out_path = Path(output_dir) / "final_model"
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    logger.info("Training complete. Final model saved to: %s", out_path)

    print(
        f"\n[GRPO] Training complete.\n"
        f"Final model: {out_path}\n\n"
        "To use this model in the live pipeline, set in your .env:\n"
        f"  SUPERVISOR_LLM_MODEL=<your-serving-path-or-id>\n"
        "and point SUPERVISOR_LLM_ENDPOINT at a vLLM / Ollama server "
        "serving the fine-tuned checkpoint.\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning for the EEG Functional Connectivity Supervisor LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/grpo_run",
        help="Directory to save checkpoints and final model.",
    )
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=4,
                        help="Number of completions per prompt (G in GRPO paper).")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_seed", type=int, default=42)

    args = parser.parse_args()

    run_grpo_training(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_new_tokens=args.max_new_tokens,
        group_size=args.group_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        dataset_seed=args.dataset_seed,
    )


if __name__ == "__main__":
    _cli()
