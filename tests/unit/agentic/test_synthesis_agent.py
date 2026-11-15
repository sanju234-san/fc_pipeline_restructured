"""Tests verifying Markdown report transcription rules with mock LLM and Gate 2 verification."""

from pathlib import Path
from types import SimpleNamespace

from fc_pipeline.agentic.synthesis.agent import run_synthesis_agent
from fc_pipeline.gates.cli.gate2_prepublication import verify_report_prepublication
from fc_pipeline.schemas.enums import (
    EvidenceCategoryEnum,
    MetricEnum,
    OverallVerdictEnum,
    VisualTopologyReadEnum,
)
from fc_pipeline.schemas.evaluation import (
    ArtifactDetectionResult,
    ArtifactSignatures,
    EvaluationVerdict,
    SanityCheckFlags,
)
from fc_pipeline.schemas.evidence import PairEvidenceSynthesis
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand


class _MockSynthesisLLM:
    def invoke(self, messages):
        return SimpleNamespace(
            content="Executive summary: Alpha connectivity was computed on C3 and C4 during rest with PLI mean=0.2800."
        )


def _sample_synthesis_state():
    return {
        "run_id": "synth01",
        "raw_data_path": "outputs/sample_raw.fif",
        "plan": AnalysisPlan(
            metrics=[MetricEnum.PLI, MetricEnum.PLV],
            freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=12.0),
            channels=["C3", "C4"],
            condition="rest",
        ),
        "parameter_manifest": [
            ParameterManifestEntry(
                name="tau_phase",
                category="coupling_threshold",
                proposed_value="0.20",
                confidence=1.0,
                needs_human_input=False,
                risk_tier="elevated",
                human_approved_value="0.20",
                rationale="Default phase threshold",
            ),
            ParameterManifestEntry(
                name="tau_zerolag",
                category="coupling_threshold",
                proposed_value="0.35",
                confidence=1.0,
                needs_human_input=False,
                risk_tier="elevated",
                human_approved_value="0.35",
                rationale="Default zero-lag threshold",
            ),
        ],
        "bad_channels_dropped": [],
        "channel_plot_paths": {},
        "heatmap_image_paths": {},
        "network_image_paths": {},
        "numerical_summaries": {
            "pli": {"mean": 0.28, "std": 0.04, "median": 0.28, "min": 0.28, "max": 0.28, "sparsity": 0.0},
            "plv": {"mean": 0.45, "std": 0.05, "median": 0.45, "min": 0.45, "max": 0.45, "sparsity": 0.0},
        },
        "evidence_summary": [
            PairEvidenceSynthesis(
                channel_pair=["C3", "C4"],
                phase_robust_mean=0.28,
                zero_lag_mean=0.45,
                classification=EvidenceCategoryEnum.FULLY_CONVERGENT,
                evaluator_visual_read=VisualTopologyReadEnum.PLAUSIBLE_NEURAL_TOPOLOGY,
                evaluator_visual_note="Inter-hemispheric sensorimotor link.",
            )
        ],
        "evaluation_verdict": EvaluationVerdict(
            overall_verdict=OverallVerdictEnum.PASS,
            per_metric_sanity={
                "pli": SanityCheckFlags(
                    values_in_bounds=True,
                    matrix_is_symmetric=True,
                    is_sparse_or_flat=False,
                    confidence=1.0,
                )
            },
            artifacts=ArtifactSignatures(
                ocular_contamination=ArtifactDetectionResult(
                    detected=False, affected_channels=[], confidence=0.95, detail="Clear"
                ),
                muscle_contamination=ArtifactDetectionResult(
                    detected=False, affected_channels=[], confidence=0.95, detail="Clear"
                ),
                electrode_pop=ArtifactDetectionResult(
                    detected=False, affected_channels=[], confidence=0.95, detail="Clear"
                ),
                top_unresolved_visual_reads=[],
            ),
            visual_observations="Clean symmetric sensorimotor topology.",
            recommendations_for_user=["Proceed with regional interpretation."],
        ),
    }


def test_run_synthesis_agent_writes_verified_report(tmp_path: Path):
    state = _sample_synthesis_state()
    out = run_synthesis_agent(state, llm=_MockSynthesisLLM(), output_dir=tmp_path)
    report_path = out.get("final_report_path")
    assert report_path is not None
    assert Path(report_path).is_file()

    content = Path(report_path).read_text(encoding="utf-8")
    check = verify_report_prepublication(content, state)
    assert check["passed"] is True
    assert "0.2800" in content
    assert "0.4500" in content
    assert "FULLY CONVERGENT" in content
    assert "ZERO-LAG COUPLING, CAUSE UNRESOLVED" in content
