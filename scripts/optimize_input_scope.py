"""Offline: measure and optimise the DSPy input-scope classifier.

    python scripts/optimize_input_scope.py            # baseline vs optimised
    python scripts/optimize_input_scope.py --no-compile   # baseline only

Uses the same SUPERVISOR_LLM_* endpoint as the pipeline. Writes the optimised
program to outputs/dspy/input_scope.json. Does NOT change the live guardrail.
Note: with only 30 examples the held-out split is tiny - treat scores as a
smoke signal and grow fc_pipeline/research/dspy_scope.py::LABELLED first.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import dspy
from dotenv import load_dotenv

from fc_pipeline.research.dspy_scope import InputScopeClassifier, labelled_examples, scope_metric


def _lm() -> dspy.LM:
    endpoint = (os.getenv("SUPERVISOR_LLM_ENDPOINT") or "").strip().strip("\"'")
    model = (os.getenv("SUPERVISOR_LLM_MODEL") or "").strip().strip("\"'")
    key = os.getenv("SUPERVISOR_LLM_API_KEY") or os.getenv("GROQ_API_KEY") or "not-required"
    if not endpoint or not model:
        raise SystemExit("Set SUPERVISOR_LLM_ENDPOINT and SUPERVISOR_LLM_MODEL (see .env).")
    return dspy.LM(f"openai/{model}", api_base=endpoint, api_key=key, temperature=0.0)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dspy.configure(lm=_lm())
    data = labelled_examples()
    random.Random(args.seed).shuffle(data)
    cut = int(len(data) * 0.6)
    train, held = data[:cut], data[cut:]

    evaluate = dspy.Evaluate(devset=held, metric=scope_metric, display_progress=False)
    base = InputScopeClassifier()
    print("baseline held-out score:", evaluate(base))
    if args.no_compile:
        return

    tuned = dspy.BootstrapFewShot(metric=scope_metric, max_bootstrapped_demos=4).compile(base, trainset=train)
    print("optimised held-out score:", evaluate(tuned))
    out = Path("outputs/dspy")
    out.mkdir(parents=True, exist_ok=True)
    tuned.save(str(out / "input_scope.json"))
    print("saved:", out / "input_scope.json")


if __name__ == "__main__":
    main()
