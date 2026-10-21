"""GRPO audit: run the subset of the pipeline that does NOT need trl/transformers.

The project's GRPO stack is
  research/grpo_dataset.py    -> synthetic scenario generator (NumPy only)
  research/grpo_reward_fn.py  -> reward functions (NumPy only)
  research/grpo_trainer.py    -> trl GRPOTrainer wrapper (needs trl/transformers/datasets)

trl / transformers / datasets are NOT installed in this venv, so the trainer
itself cannot execute here.  This script therefore:
  * verifies the generator + reward functions end-to-end (they are runnable),
  * inspects shapes / metadata / duplicates / leakage,
  * probes the reward surface for inconsistencies and reward-hacking holes,
  * confirms the trainer fails *cleanly* with an actionable ImportError.

Run:
    .\.venv\Scripts\python.exe test_outputs\grpo_audit.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fc_pipeline.config.thresholds import TAU_PHASE, TAU_ZEROLAG  # noqa: E402
from fc_pipeline.research.grpo_dataset import (  # noqa: E402
    VALID_METRICS,
    build_grpo_dataset,
)
from fc_pipeline.research.grpo_reward_fn import (  # noqa: E402
    composite_reward,
    format_compliance_reward,
    metric_classification_reward,
    metric_set_reward,
)

SEP = "-" * 78


def hdr(t: str) -> None:
    print(f"\n{SEP}\n{t}\n{SEP}")


# ---------------------------------------------------------------------------
def audit_dataset() -> dict:
    hdr("A) SYNTHETIC DATASET GENERATION")
    ds = build_grpo_dataset(n_samples=300, seed=42, use_hf_datasets=False)
    print(f"records            : {len(ds)}")
    print(f"keys               : {sorted(ds[0].keys())}")
    print(f"prompt type        : {type(ds[0]['prompt']).__name__}")
    print(f"chosen type        : {type(ds[0]['chosen']).__name__}")

    # Duplicate analysis
    prompts = [r["prompt"] for r in ds]
    chosen = [r["chosen"] for r in ds]
    print(f"duplicate prompts  : {len(prompts) - len(set(prompts))}")
    print(f"distinct chosen    : {len(set(chosen))} (of {len(chosen)})")
    print(f"distinct prompt tpl: {len(set(re.sub(r'[0-9.]+', 'N', p) for p in prompts))}")

    # Label / metric validity vs thresholds
    bad_label, bad_metric, mismatch = 0, 0, 0
    label_counts: Counter = Counter()
    metric_subset_counts: Counter = Counter()
    for rec in ds:
        try:
            obj = json.loads(rec["chosen"])
        except json.JSONDecodeError:
            bad_label += 1
            continue
        label_counts[obj.get("classification")] += 1
        ms = obj.get("metrics", [])
        metric_subset_counts[tuple(ms)] += 1
        if any(m not in VALID_METRICS for m in ms):
            bad_metric += 1
        # Ground-truth check against the value embedded in the prompt
        m = re.search(r"connectivity value:\s*([0-9.]+)", rec["prompt"], re.IGNORECASE)
        if m:
            v = float(m.group(1))
            expect = (
                "zero_lag" if v >= TAU_ZEROLAG
                else "phase_coupled" if v >= TAU_PHASE
                else "weak"
            )
            if obj.get("classification") != expect:
                mismatch += 1

    print(f"invalid metric name: {bad_metric}")
    print(f"unparseable chosen : {bad_label}")
    print(f"label != threshold : {mismatch}")
    print(f"label distribution : {dict(label_counts)}")
    print(f"metric subsets used: {dict(metric_subset_counts)}")
    print(f"sample prompt      : {ds[0]['prompt'].splitlines()[4]!r} ...")
    print(f"sample chosen      : {ds[0]['chosen']}")

    return {
        "n": len(ds),
        "distinct_chosen": len(set(chosen)),
        "distinct_subsets": len(metric_subset_counts),
        "bad_metric": bad_metric,
        "mismatch": mismatch,
    }


# ---------------------------------------------------------------------------
def audit_rewards() -> None:
    hdr("B) REWARD FUNCTIONS")

    def prompt(value: float, metric: str = "PLI", requested: str = "all") -> str:
        return (
            "You are the Supervisor Agent for an EEG Functional Connectivity pipeline.\n\n"
            f"Metric: {metric}\nConnectivity value: {value:.4f}\n"
            f"Requested metrics: {requested}\n"
        )

    cases = [
        ("correct weak JSON",
         prompt(0.05), '{"classification": "weak", "metrics": ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]}'),
        ("correct zero_lag JSON",
         prompt(0.45), '{"classification": "zero_lag", "metrics": ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]}'),
        ("correct phase_coupled JSON",
         prompt(0.25), '{"classification": "phase_coupled", "metrics": ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]}'),
        ("WRONG label, full metrics",
         prompt(0.45), '{"classification": "weak", "metrics": ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]}'),
        ("correct label, SUBSET metrics",
         prompt(0.45), '{"classification": "zero_lag", "metrics": ["PLI"]}'),
        ("invented metric name",
         prompt(0.45), '{"classification": "zero_lag", "metrics": ["PLI", "MagicCoherence"]}'),
        ("valid labels but 'ambiguous'",
         prompt(0.45), '{"classification": "ambiguous", "metrics": ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]}'),
        ("non-JSON prose",
         prompt(0.45), "The link looks phase coupled, you should compute PLI."),
        ("empty",
         prompt(0.45), ""),
    ]

    print(f"{'case':38s} {'cls':>5s} {'set':>5s} {'fmt':>5s} {'composite':>9s}")
    for name, p, c in cases:
        rc = metric_classification_reward(p, c)
        rm = metric_set_reward(p, c)
        rf = format_compliance_reward(p, c)
        print(f"{name:38s} {rc:5.2f} {rm:5.2f} {rf:5.2f} {composite_reward(p, c):9.3f}")

    print("\n-- reward-hacking probes --")
    # A degenerate constant answer: always say zero_lag + all 5 metrics.
    const = '{"classification": "zero_lag", "metrics": ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]}'
    scores = [composite_reward(prompt(v), const) for v in (0.05, 0.25, 0.45, 0.9)]
    print(f"constant 'zero_lag + all 5' over values [0.05,0.25,0.45,0.9]: {[round(s,3) for s in scores]}")
    print("  -> a constant response scores >= 0.50 on every prompt (format+set free):",
          all(s >= 0.50 for s in scores))

    # Metric-name irrelevance: same value, different metric field -> identical reward.
    a = metric_classification_reward(prompt(0.45, metric="PLI"), const)
    b = metric_classification_reward(prompt(0.45, metric="Coherence"), const)
    print(f"metric field PLI vs Coherence (same value) -> same reward: {a == b} ({a})")


# ---------------------------------------------------------------------------
def audit_trainer() -> None:
    hdr("C) TRAINER (trl) AVAILABILITY")
    for pkg in ("torch", "trl", "transformers", "datasets"):
        try:
            __import__(pkg)
            print(f"  {pkg:14s}: INSTALLED")
        except ImportError as exc:
            print(f"  {pkg:14s}: MISSING ({exc})")

    from fc_pipeline.research import grpo_trainer

    try:
        grpo_trainer._check_deps()
        print("  _check_deps(): passed")
    except ImportError as exc:
        print(f"  _check_deps() -> ImportError: {str(exc).splitlines()[0]}")

    try:
        grpo_trainer.run_grpo_training(
            model_name_or_path="stub", output_dir="test_outputs/grpo_stub", n_samples=4
        )
        print("  run_grpo_training(): ran (unexpected here)")
    except ImportError as exc:
        print(f"  run_grpo_training() -> clean ImportError: {str(exc).splitlines()[0]}")
    except Exception as exc:  # pragma: no cover
        print(f"  run_grpo_training() -> {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
def audit_leakage() -> None:
    hdr("D) DATA LEAKAGE / EVALUATION")
    import inspect
    from fc_pipeline.research import grpo_dataset, grpo_trainer

    src = inspect.getsource(grpo_dataset) + inspect.getsource(grpo_trainer)
    has_split = bool(re.search(r"train_test_split|validation|eval_dataset|test_dataset", src))
    print(f"train/val/test split in GRPO code : {has_split}")
    print("real EEG data referenced by GRPO  :",
          bool(re.search(r"read_raw|read_epochs|\.fif|\.edf", src)))
    print("-> dataset is fully synthetic (NumPy draws); no real EEG is read,")
    print("   so there is no real-data train/test contamination. There is also no")
    print("   held-out split: GRPO trains and is evaluated on the same generator.")


def main() -> None:
    print(f"Thresholds: TAU_PHASE={TAU_PHASE}  TAU_ZEROLAG={TAU_ZEROLAG}")
    stats = audit_dataset()
    audit_rewards()
    audit_trainer()
    audit_leakage()
    hdr("SUMMARY")
    print(json.dumps(stats, indent=2))
    print("\n[audit] done")


if __name__ == "__main__":
    main()
