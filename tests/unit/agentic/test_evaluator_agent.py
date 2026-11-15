"""Tests verifying VLM payload generation and response parsing with mock VLM."""

import json
from types import SimpleNamespace
import pandas as pd

from fc_pipeline.agentic.evaluator.agent import run_evaluator_agent
from fc_pipeline.schemas.enums import (
    EvidenceCategoryEnum,
    MetricEnum,
    OverallVerdictEnum,
    VisualTopologyReadEnum,
)
from fc_pipeline.schemas.evidence import PairEvidenceSynthesis
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand


class _MockVLM:
    def invoke(self, messages):
        payload = {
            "visual_observations": "Mock VLM observed plausible fronto-parietal network.",
            "recommendations_for_user": ["Verify volume conduction on unresolved pairs."],
            "pair_reads": [
                {
                    "pair": ["F3", "P3"],
                    "read": "plausible_neural_topology",
                    "note": "Long-range fronto-parietal link.",
                }
            ],
        }
        return SimpleNamespace(content=json.dumps(payload))


def test_run_evaluator_agent_end_to_end_with_mock_vlm(tmp_path):
    pli_csv = tmp_path / "pli_run42.csv"
    pd.DataFrame(
        [[0.0, 0.12, 0.35], [0.12, 0.0, 0.28], [0.35, 0.28, 0.0]],
        index=["F3", "P3", "O1"],
        columns=["F3", "P3", "O1"],
    ).to_csv(pli_csv)

    state = {
        "run_id": "run42",
        "plan": AnalysisPlan(
            metrics=[MetricEnum.PLI, MetricEnum.PLV],
            freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=12.0),
            channels=["F3", "P3", "O1"],
            condition="rest",
        ),
        "parameter_manifest": [],
        "metric_csv_paths": {"pli": str(pli_csv)},
        "heatmap_image_paths": {},
        "network_image_paths": {},
        "evidence_summary": [
            PairEvidenceSynthesis(
                channel_pair=["F3", "P3"],
                phase_robust_mean=0.12,
                zero_lag_mean=0.52,
                classification=EvidenceCategoryEnum.ZERO_LAG_UNRESOLVED,
            ),
            PairEvidenceSynthesis(
                channel_pair=["F3", "O1"],
                phase_robust_mean=0.35,
                zero_lag_mean=0.45,
                classification=EvidenceCategoryEnum.FULLY_CONVERGENT,
            ),
        ],
    }

    out = run_evaluator_agent(state, llm=_MockVLM(), output_dir=tmp_path)
    verdict = out["evaluation_verdict"]
    assert verdict.overall_verdict == OverallVerdictEnum.WARNING
    assert "pli" in verdict.per_metric_sanity
    assert verdict.per_metric_sanity["pli"].confidence == 1.0
    assert len(verdict.artifacts.top_unresolved_visual_reads) == 1
    top_read = verdict.artifacts.top_unresolved_visual_reads[0]
    assert top_read.evaluator_visual_read == VisualTopologyReadEnum.PLAUSIBLE_NEURAL_TOPOLOGY
    assert top_read.classification == EvidenceCategoryEnum.ZERO_LAG_UNRESOLVED
    assert out["evaluation_summary_csv_path"].endswith("evaluation_summary_run42.csv")
