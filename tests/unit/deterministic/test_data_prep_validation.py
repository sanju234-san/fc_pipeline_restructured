"""Tests for deterministic Data Preparation validation (fail-closed behaviour).

Manifests are produced by the real Supervisor function
``compile_and_confirm_manifest`` and recording metadata by the real dataset
tools, so these tests also guard the contract between the Supervisor output and
Data Prep validation.
"""

import os
from pathlib import Path
from typing import List

import pytest
from pydantic import ValidationError

from fc_pipeline.agentic.supervisor.agent import compile_and_confirm_manifest
from fc_pipeline.agentic.supervisor.tools.dataset_conditions import get_dataset_conditions
from fc_pipeline.agentic.supervisor.tools.dataset_info import get_dataset_info
from fc_pipeline.deterministic.data_prep.models import (
    DataPrepInput,
    ValidatedDataPrepParams,
)
from fc_pipeline.deterministic.data_prep.validation import (
    REQUIRED_MANIFEST_PARAMETERS,
    DataPrepValidationError,
    build_safe_output_path,
    validate_data_prep_input,
    validate_plan_against_recording,
    validate_raw_data_path,
    validate_run_id,
)
from fc_pipeline.pipeline.graph import approve_gate_1, reject_gate_1
from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand

CONFS = {"frequency_band": 1.0, "channels": 1.0, "condition": 1.0}
GOOD_RUN_ID = "chainlit_20260921_062638_001"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _safe_symlink(link: Path, target: Path, target_is_directory: bool = False) -> None:
    """Create a symlink, or skip if the OS denies symlink creation privileges (e.g. non-admin Windows)."""
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks on Windows requires Developer Mode or administrator privileges")
        raise


def _plan(
    channels=("F3", "F4"),
    condition="rest",
    fmin=8.0,
    fmax=12.0,
    name="alpha",
    metrics=None,
) -> AnalysisPlan:
    kwargs = {}
    if metrics is not None:
        kwargs["metrics"] = metrics
    return AnalysisPlan(
        freq_band=FrequencyBand(name=name, fmin=fmin, fmax=fmax),
        channels=list(channels),
        condition=condition,
        **kwargs,
    )


def _manifest(plan=None, **kw) -> List[ParameterManifestEntry]:
    manifest, err = compile_and_confirm_manifest(plan or _plan(), CONFS, **kw)
    assert err is None
    return manifest


def _entry(manifest, name) -> ParameterManifestEntry:
    return next(e for e in manifest if e.name == name)


def _input(path, *, plan=None, manifest=None, run_id=GOOD_RUN_ID,
           gate=True, preflight=True) -> DataPrepInput:
    plan = plan or _plan()
    return DataPrepInput(
        raw_data_path=str(path),
        plan=plan,
        parameter_manifest=manifest if manifest is not None else _manifest(plan),
        run_id=run_id,
        gate_1_approved=gate,
        preflight_confirmed=preflight,
    )


def _with(inp: DataPrepInput, **updates) -> DataPrepInput:
    """Copy WITHOUT re-validation, to simulate a corrupted/bypassed input."""
    return inp.model_copy(update=updates)


def _code(exc_info) -> str:
    return exc_info.value.code


# --------------------------------------------------------------------------- #
# Gate 1 approval
# --------------------------------------------------------------------------- #

class TestApproval:
    def test_approved_returns_validated_params(self, synthetic_eeg_path):
        params = validate_data_prep_input(_input(synthetic_eeg_path))
        assert isinstance(params, ValidatedDataPrepParams)
        assert params.channels == ("F3", "F4")
        assert params.condition == "rest"
        assert (params.fmin, params.fmax) == (8.0, 12.0)
        assert params.reference_method == "average"
        assert params.bad_channel_variance_threshold == pytest.approx(1e-15)
        assert params.min_cycles == 3.0
        assert params.run_id == GOOD_RUN_ID
        assert params.raw_data_path == Path(synthetic_eeg_path).resolve()

    def test_gate_1_rejected_blocks(self, synthetic_eeg_path):
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, gate=False))
        assert _code(e) == "GATE_1_NOT_APPROVED"

    def test_preflight_false_blocks(self, synthetic_eeg_path):
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, preflight=False))
        assert _code(e) == "PREFLIGHT_NOT_CONFIRMED"

    def test_defaults_are_not_approved(self, synthetic_eeg_path):
        plan = _plan()
        inp = DataPrepInput(
            raw_data_path=str(synthetic_eeg_path),
            plan=plan,
            parameter_manifest=_manifest(plan),
            run_id=GOOD_RUN_ID,
        )
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "GATE_1_NOT_APPROVED"

    @pytest.mark.parametrize("value", ["yes", "true", "True", 1, "1", "on"])
    @pytest.mark.parametrize("field", ["gate_1_approved", "preflight_confirmed"])
    def test_truthy_non_bool_flags_are_rejected_by_the_model(
        self, synthetic_eeg_path, field, value
    ):
        # Lax pydantic bool coercion would have turned these into True.
        plan = _plan()
        with pytest.raises(ValidationError):
            DataPrepInput(
                raw_data_path=str(synthetic_eeg_path),
                plan=plan,
                parameter_manifest=_manifest(plan),
                run_id=GOOD_RUN_ID,
                **{field: value},
            )

    def test_non_bool_flag_smuggled_past_model_still_blocks(self, synthetic_eeg_path):
        inp = _with(_input(synthetic_eeg_path), gate_1_approved="yes")
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "GATE_1_NOT_APPROVED"

    def test_approval_is_checked_before_filesystem(self, tmp_path):
        inp = _input(tmp_path / "does_not_exist.fif", gate=False)
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "GATE_1_NOT_APPROVED"

    def test_validation_never_flips_flags(self, synthetic_eeg_path):
        inp = _input(synthetic_eeg_path, gate=False, preflight=False)
        with pytest.raises(DataPrepValidationError):
            validate_data_prep_input(inp)
        assert inp.gate_1_approved is False
        assert inp.preflight_confirmed is False

    def test_flags_from_real_gate_1_helpers(self, synthetic_eeg_path):
        state = {"plan": _plan()}
        ok = approve_gate_1(state)
        params = validate_data_prep_input(
            _input(synthetic_eeg_path, gate=ok["gate_1_approved"],
                   preflight=ok["preflight_confirmed"])
        )
        assert params.condition == "rest"

        no = reject_gate_1(state)
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(
                _input(synthetic_eeg_path, gate=no["gate_1_approved"],
                       preflight=no["preflight_confirmed"])
            )
        assert _code(e) == "GATE_1_NOT_APPROVED"

    def test_wrong_input_type_blocks(self):
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input({"gate_1_approved": True})  # type: ignore[arg-type]
        assert _code(e) == "INVALID_INPUT"


# --------------------------------------------------------------------------- #
# Plan validation
# --------------------------------------------------------------------------- #

class TestPlan:
    def test_missing_plan_blocked_at_model_level(self, synthetic_eeg_path):
        with pytest.raises(ValidationError):
            DataPrepInput(
                raw_data_path=str(synthetic_eeg_path), plan=None,  # type: ignore[arg-type]
                parameter_manifest=[], run_id=GOOD_RUN_ID,
                gate_1_approved=True, preflight_confirmed=True,
            )

    def test_missing_plan_blocked_when_model_bypassed(self, synthetic_eeg_path):
        inp = _with(_input(synthetic_eeg_path), plan=None)
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "MISSING_PLAN"

    def test_non_plan_object_blocked(self, synthetic_eeg_path):
        inp = _with(_input(synthetic_eeg_path), plan={"channels": ["F3", "F4"]})
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_PLAN"

    def test_plan_mutated_after_construction_is_revalidated(self, synthetic_eeg_path):
        plan = _plan()
        inp = _input(synthetic_eeg_path, plan=plan)
        plan.freq_band.fmin = -3.0  # pydantic does not validate assignment
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_PLAN"

    @pytest.mark.parametrize(
        "fmin,fmax",
        [(12.0, 8.0), (8.0, 8.0), (8.0, float("inf")), (float("inf"), float("inf"))],
    )
    def test_invalid_frequency_band(self, synthetic_eeg_path, fmin, fmax):
        plan = AnalysisPlan.model_construct(
            metrics=[MetricEnum.PLI],
            freq_band=FrequencyBand.model_construct(name="x", fmin=fmin, fmax=fmax),
            channels=["F3", "F4"], condition="rest",
        )
        inp = _with(_input(synthetic_eeg_path), plan=plan)
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_FREQUENCY_BAND"

    @pytest.mark.parametrize(
        "channels",
        [["F3"], [], ["F3", "F3"], ["F3", ""], ["F3", "   "]],
    )
    def test_invalid_channels(self, synthetic_eeg_path, channels):
        plan = AnalysisPlan.model_construct(
            metrics=[MetricEnum.PLI],
            freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=12.0),
            channels=channels, condition="rest",
        )
        inp = _with(_input(synthetic_eeg_path), plan=plan)
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_CHANNELS"

    @pytest.mark.parametrize("condition", ["", "   "])
    def test_missing_condition(self, synthetic_eeg_path, condition):
        plan = AnalysisPlan.model_construct(
            metrics=[MetricEnum.PLI],
            freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=12.0),
            channels=["F3", "F4"], condition=condition,
        )
        inp = _with(_input(synthetic_eeg_path), plan=plan)
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_CONDITION"

    def test_empty_metrics(self, synthetic_eeg_path):
        plan = AnalysisPlan.model_construct(
            metrics=[],
            freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=12.0),
            channels=["F3", "F4"], condition="rest",
        )
        inp = _with(_input(synthetic_eeg_path), plan=plan)
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_PLAN"


# --------------------------------------------------------------------------- #
# Manifest & human approval
# --------------------------------------------------------------------------- #

class TestManifest:
    def test_empty_manifest_blocks(self, synthetic_eeg_path):
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=[]))
        assert _code(e) == "MISSING_MANIFEST_PARAMETER"

    @pytest.mark.parametrize("name", REQUIRED_MANIFEST_PARAMETERS)
    def test_each_required_parameter_is_required(self, synthetic_eeg_path, name):
        manifest = [m for m in _manifest() if m.name != name]
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "MISSING_MANIFEST_PARAMETER"
        assert name in str(e.value)

    def test_duplicate_required_entry_blocks(self, synthetic_eeg_path):
        manifest = _manifest()
        manifest.append(_entry(manifest, "reference").model_copy())
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "DUPLICATE_MANIFEST_ENTRY"

    def test_non_manifest_entry_blocks(self, synthetic_eeg_path):
        inp = _with(_input(synthetic_eeg_path), parameter_manifest=[{"name": "reference"}])
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_MANIFEST"

    def test_manifest_not_a_list_blocks(self, synthetic_eeg_path):
        inp = _with(_input(synthetic_eeg_path), parameter_manifest="reference=average")
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_MANIFEST"

    @pytest.mark.parametrize(
        "name,bad",
        [
            ("channels", "F3, C3"),
            ("channels", "F4, F3"),  # order is part of the plan
            ("condition", "task"),
            ("condition", "REST"),
            ("freq_band", "alpha (13.0 - 30.0 Hz)"),
        ],
    )
    def test_manifest_disagreeing_with_plan_blocks(self, synthetic_eeg_path, name, bad):
        manifest = _manifest()
        _entry(manifest, name).proposed_value = bad
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "MANIFEST_PLAN_MISMATCH"

    def test_malformed_freq_band_string_blocks(self, synthetic_eeg_path):
        manifest = _manifest()
        _entry(manifest, "freq_band").proposed_value = "alpha"
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "INVALID_PARAMETER"

    def test_freq_band_rounding_of_supervisor_display_is_tolerated(self, synthetic_eeg_path):
        plan = _plan(fmin=8.04, fmax=12.06, name="custom")
        params = validate_data_prep_input(_input(synthetic_eeg_path, plan=plan))
        assert (params.fmin, params.fmax) == (8.04, 12.06)  # plan stays canonical

    # ---- human approval semantics -------------------------------------- #

    def test_needs_human_input_without_approval_blocks(self, synthetic_eeg_path):
        manifest = _manifest()
        _entry(manifest, "reference").needs_human_input = True
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "HUMAN_APPROVAL_REQUIRED"

    def test_low_confidence_scientific_axis_without_approval_blocks(self, synthetic_eeg_path):
        plan = _plan()
        manifest, _ = compile_and_confirm_manifest(
            plan, {"frequency_band": 1.0, "channels": 0.5, "condition": 1.0}
        )
        assert _entry(manifest, "channels").needs_human_input is True
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, plan=plan, manifest=manifest))
        assert _code(e) == "HUMAN_APPROVAL_REQUIRED"

    def test_needs_human_input_with_matching_approval_passes(self, synthetic_eeg_path):
        plan = _plan()
        manifest, _ = compile_and_confirm_manifest(
            plan, {"frequency_band": 1.0, "channels": 0.5, "condition": 1.0}
        )
        _entry(manifest, "channels").human_approved_value = "F3, F4"
        params = validate_data_prep_input(_input(synthetic_eeg_path, plan=plan, manifest=manifest))
        assert params.channels == ("F3", "F4")

    def test_approval_conflicting_with_plan_blocks(self, synthetic_eeg_path):
        plan = _plan()
        manifest, _ = compile_and_confirm_manifest(
            plan, {"frequency_band": 1.0, "channels": 0.5, "condition": 1.0}
        )
        _entry(manifest, "channels").human_approved_value = "C3, C4"
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, plan=plan, manifest=manifest))
        assert _code(e) == "MANIFEST_PLAN_MISMATCH"

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_approval_is_not_an_approval(self, synthetic_eeg_path, blank):
        manifest = _manifest()
        entry = _entry(manifest, "reference")
        entry.needs_human_input = True
        entry.human_approved_value = blank
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "INVALID_HUMAN_APPROVAL"

    def test_approved_value_wins_over_proposed(self, synthetic_eeg_path):
        manifest = _manifest()
        entry = _entry(manifest, "bad_channel_variance_threshold")
        entry.human_approved_value = "1e-12"
        params = validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert params.bad_channel_variance_threshold == pytest.approx(1e-12)

    def test_approved_value_that_is_invalid_is_not_bypassed_by_valid_proposal(
        self, synthetic_eeg_path
    ):
        manifest = _manifest()
        _entry(manifest, "min_cycles").human_approved_value = "lots"
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "INVALID_PARAMETER"

    def test_advisory_rows_needing_human_input_do_not_block(self, synthetic_eeg_path):
        # Real Supervisor advisories: too few trials + non-complementary metrics.
        plan = _plan(metrics=[MetricEnum.PLI])
        manifest = _manifest(plan, trial_count=1)
        advisory = {e.name: e for e in manifest if e.category == "advisory"}
        assert {"trial_adequacy", "cross_metric_synthesis"} <= set(advisory)
        assert all(e.needs_human_input and e.human_approved_value is None
                   for e in advisory.values())
        params = validate_data_prep_input(_input(synthetic_eeg_path, plan=plan, manifest=manifest))
        assert params.condition == "rest"

    def test_validation_does_not_mutate_or_infer_approvals(self, synthetic_eeg_path):
        manifest = _manifest()
        before = [m.model_dump() for m in manifest]
        validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert [m.model_dump() for m in manifest] == before
        assert all(m.human_approved_value is None for m in manifest)

    # ---- reference ------------------------------------------------------- #

    @pytest.mark.parametrize(
        "value",
        [
            "average", "Average", "  average  ", "CAR", "common average reference",
            "unreferenced (proposed: average)",  # verbatim get_dataset_info output
        ],
    )
    def test_supported_reference_values(self, synthetic_eeg_path, value):
        manifest = _manifest()
        _entry(manifest, "reference").proposed_value = value
        params = validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert params.reference_method == "average"

    @pytest.mark.parametrize(
        "value,code",
        [
            ("mastoid", "REFERENCE_METHOD_UNSUPPORTED"),
            ("bipolar", "REFERENCE_METHOD_UNSUPPORTED"),
            ("unreferenced (proposed: mastoid)", "REFERENCE_METHOD_UNSUPPORTED"),
            ("custom", "INVALID_REFERENCE"),
            ("unknown (proposed: average)", "INVALID_REFERENCE"),
            ("laplacian", "INVALID_REFERENCE"),
            ("average; import os", "INVALID_REFERENCE"),
            ("average average", "INVALID_REFERENCE"),
        ],
    )
    def test_unsupported_reference_values_fail(self, synthetic_eeg_path, value, code):
        manifest = _manifest()
        _entry(manifest, "reference").proposed_value = value
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == code

    def test_human_may_not_approve_an_unsupported_reference(self, synthetic_eeg_path):
        manifest = _manifest()
        _entry(manifest, "reference").human_approved_value = "mastoid"
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "REFERENCE_METHOD_UNSUPPORTED"

    def test_real_supervisor_reference_string_for_unreferenced_recording(
        self, synthetic_eeg_path
    ):
        info = get_dataset_info.invoke({"data_path": str(synthetic_eeg_path)})
        assert info["reference"].startswith("unreferenced")
        manifest = _manifest(discovered_reference=info["reference"])
        params = validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert params.reference_method == "average"

    # ---- numeric thresholds ---------------------------------------------- #

    @pytest.mark.parametrize("name", ["bad_channel_variance_threshold", "min_cycles"])
    @pytest.mark.parametrize(
        "bad",
        ["nan", "inf", "-inf", "-1", "0", "0.0", "abc", "1_0", "0.6 stricter threshold",
         "1e400", "0x10", "3;4", "--3", "C:\\Users\\john\\secret.fif"],
    )
    def test_invalid_numeric_parameters_fail(self, synthetic_eeg_path, name, bad):
        manifest = _manifest()
        _entry(manifest, name).human_approved_value = bad
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert _code(e) == "INVALID_PARAMETER"
        # Free-text human input must never be echoed back in an error.
        assert bad not in str(e.value)

    @pytest.mark.parametrize("good,expected", [("3", 3.0), ("2.5", 2.5), ("1e1", 10.0), (" 4 ", 4.0)])
    def test_valid_min_cycles(self, synthetic_eeg_path, good, expected):
        manifest = _manifest()
        _entry(manifest, "min_cycles").human_approved_value = good
        params = validate_data_prep_input(_input(synthetic_eeg_path, manifest=manifest))
        assert params.min_cycles == expected


# --------------------------------------------------------------------------- #
# Raw data path
# --------------------------------------------------------------------------- #

class TestRawDataPath:
    def test_missing_dataset(self, tmp_path):
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(tmp_path / "nope.fif"))
        assert _code(e) == "DATA_FILE_NOT_FOUND"

    def test_error_does_not_leak_path(self, tmp_path):
        secret = tmp_path / "patient_jane_doe_secret.fif"
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(_input(secret))
        assert "jane" not in str(e.value) and str(tmp_path) not in str(e.value)

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_missing_or_non_string_path(self, bad):
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(bad)
        assert _code(e) == "MISSING_DATA_PATH"

    def test_nul_byte_path(self):
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path("data\x00.fif")
        assert _code(e) == "INVALID_DATA_PATH"

    def test_overlong_path(self):
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path("a" * 5000 + ".fif")
        assert _code(e) == "INVALID_DATA_PATH"

    def test_directory_is_not_a_file(self, tmp_path):
        d = tmp_path / "folder.fif"
        d.mkdir()
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(str(d))
        assert _code(e) == "DATA_PATH_NOT_A_FILE"

    @pytest.mark.parametrize("fname", ["notes.txt", "model.pkl", "data.npy", "noext", "x.fif.bak"])
    def test_unsupported_file_types(self, tmp_path, fname):
        f = tmp_path / fname
        f.write_bytes(b"x")
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(str(f))
        assert _code(e) == "UNSUPPORTED_DATA_FORMAT"

    def test_suffix_check_is_case_insensitive(self, tmp_path):
        f = tmp_path / "REC.FIF"
        f.write_bytes(b"x")
        assert validate_raw_data_path(str(f)) == f.resolve()

    def test_path_traversal_component_is_rejected(self, tmp_path, synthetic_eeg_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        sneaky = f"{sub}/../{Path(synthetic_eeg_path).name}"
        # Even when the target exists, '..' is refused outright.
        (tmp_path / Path(synthetic_eeg_path).name).write_bytes(b"x")
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(sneaky)
        assert _code(e) == "INVALID_DATA_PATH"

    def test_home_shortcut_is_not_expanded(self, tmp_path):
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path("~/recording.fif")
        assert _code(e) == "DATA_FILE_NOT_FOUND"

    def test_symlink_to_disallowed_type_is_blocked(self, tmp_path):
        target = tmp_path / "secrets.txt"
        target.write_text("x")
        link = tmp_path / "looks_ok.fif"
        _safe_symlink(link, target)
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(str(link))
        assert _code(e) == "UNSUPPORTED_DATA_FORMAT"

    def test_allowed_roots_permit_inside(self, synthetic_eeg_path):
        root = Path(synthetic_eeg_path).parent
        assert validate_raw_data_path(str(synthetic_eeg_path), allowed_data_roots=[root])

    def test_allowed_roots_block_outside(self, tmp_path, synthetic_eeg_path):
        other = tmp_path / "elsewhere"
        other.mkdir()
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(str(synthetic_eeg_path), allowed_data_roots=[other])
        assert _code(e) == "DATA_PATH_OUTSIDE_ALLOWED_ROOTS"

    def test_empty_allowed_roots_blocks_everything(self, synthetic_eeg_path):
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(str(synthetic_eeg_path), allowed_data_roots=[])
        assert _code(e) == "DATA_PATH_OUTSIDE_ALLOWED_ROOTS"

    def test_symlink_escaping_allowed_root_is_blocked(self, tmp_path, synthetic_eeg_path):
        root = tmp_path / "data_root"
        root.mkdir()
        link = root / "escape.fif"
        _safe_symlink(link, synthetic_eeg_path)
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(str(link), allowed_data_roots=[root])
        assert _code(e) == "DATA_PATH_OUTSIDE_ALLOWED_ROOTS"

    def test_sibling_directory_with_shared_prefix_is_not_inside_root(self, tmp_path):
        root = tmp_path / "data"
        evil = tmp_path / "data_evil"
        root.mkdir()
        evil.mkdir()
        f = evil / "rec.fif"
        f.write_bytes(b"x")
        with pytest.raises(DataPrepValidationError) as e:
            validate_raw_data_path(str(f), allowed_data_roots=[root])
        assert _code(e) == "DATA_PATH_OUTSIDE_ALLOWED_ROOTS"

    def test_roots_are_enforced_via_main_entry_point(self, tmp_path, synthetic_eeg_path):
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(
                _input(synthetic_eeg_path), allowed_data_roots=[tmp_path / "other"]
            )
        assert _code(e) == "DATA_PATH_OUTSIDE_ALLOWED_ROOTS"


# --------------------------------------------------------------------------- #
# run_id and output path safety
# --------------------------------------------------------------------------- #

class TestRunIdAndOutputPaths:
    @pytest.mark.parametrize(
        "ok", [GOOD_RUN_ID, f"{GOOD_RUN_ID}_rev2", "playground_test_06", "default", "a", "A-b_9"]
    )
    def test_valid_run_ids(self, ok):
        assert validate_run_id(ok) == ok

    @pytest.mark.parametrize(
        "bad",
        ["", " ", "../evil", "..", ".", "a/b", "a\\b", ".hidden", "a.b", "run id",
         "run\n", "x" * 81, "-lead", "_lead", "é", "a\x00b", "C:\\x", "run;rm", None, 7],
    )
    def test_malformed_run_ids_are_rejected_not_rewritten(self, bad):
        with pytest.raises(DataPrepValidationError) as e:
            validate_run_id(bad)
        assert _code(e) == "INVALID_RUN_ID"

    def test_malformed_run_id_blocks_main_entry_point(self, synthetic_eeg_path):
        inp = _with(_input(synthetic_eeg_path), run_id="../../etc/cron.d/x")
        with pytest.raises(DataPrepValidationError) as e:
            validate_data_prep_input(inp)
        assert _code(e) == "INVALID_RUN_ID"

    def test_safe_output_path(self, tmp_path):
        p = build_safe_output_path(tmp_path, GOOD_RUN_ID, "channels_before", ".png")
        assert p == tmp_path.resolve() / f"channels_before_{GOOD_RUN_ID}.png"
        assert p.parent == tmp_path.resolve()

    @pytest.mark.parametrize("stem", ["", "../x", "a/b", "Upper", "1abc", "a b", "a.b", "x" * 41])
    def test_bad_stems(self, tmp_path, stem):
        with pytest.raises(DataPrepValidationError) as e:
            build_safe_output_path(tmp_path, GOOD_RUN_ID, stem, ".png")
        assert _code(e) == "INVALID_OUTPUT_PATH"

    @pytest.mark.parametrize("suffix", [".py", "png", "", ".png/../x", ".sh", ".PNG"])
    def test_bad_suffixes(self, tmp_path, suffix):
        with pytest.raises(DataPrepValidationError) as e:
            build_safe_output_path(tmp_path, GOOD_RUN_ID, "channels_before", suffix)
        assert _code(e) == "INVALID_OUTPUT_PATH"

    def test_bad_run_id_in_output_path(self, tmp_path):
        with pytest.raises(DataPrepValidationError) as e:
            build_safe_output_path(tmp_path, "../x", "channels_before", ".png")
        assert _code(e) == "INVALID_RUN_ID"

    def test_missing_output_dir(self, tmp_path):
        with pytest.raises(DataPrepValidationError) as e:
            build_safe_output_path(tmp_path / "nope", GOOD_RUN_ID, "channels_before", ".png")
        assert _code(e) == "INVALID_OUTPUT_PATH"

    def test_output_dir_with_dotdot(self, tmp_path):
        (tmp_path / "a").mkdir()
        with pytest.raises(DataPrepValidationError) as e:
            build_safe_output_path(f"{tmp_path}/a/..", GOOD_RUN_ID, "channels_before", ".png")
        assert _code(e) == "INVALID_OUTPUT_PATH"

    def test_output_dir_is_a_file(self, tmp_path):
        f = tmp_path / "file"
        f.write_text("x")
        with pytest.raises(DataPrepValidationError) as e:
            build_safe_output_path(f, GOOD_RUN_ID, "channels_before", ".png")
        assert _code(e) == "INVALID_OUTPUT_PATH"

    def test_refuses_to_write_through_symlink(self, tmp_path):
        victim = tmp_path / "victim.txt"
        victim.write_text("keep")
        link = tmp_path / f"channels_before_{GOOD_RUN_ID}.png"
        _safe_symlink(link, victim)
        with pytest.raises(DataPrepValidationError) as e:
            build_safe_output_path(tmp_path, GOOD_RUN_ID, "channels_before", ".png")
        assert _code(e) == "INVALID_OUTPUT_PATH"

    def test_symlinked_output_dir_resolves_and_stays_contained(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        _safe_symlink(link, real, target_is_directory=True)
        p = build_safe_output_path(link, GOOD_RUN_ID, "channels_after", ".png")
        assert p.parent == real.resolve()


# --------------------------------------------------------------------------- #
# Recording metadata validation (real dataset tools on the synthetic fixture)
# --------------------------------------------------------------------------- #

def _params(**overrides) -> ValidatedDataPrepParams:
    base = dict(
        raw_data_path=Path("/tmp/x.fif"), run_id=GOOD_RUN_ID,
        channels=("F3", "F4"), condition="rest", fmin=8.0, fmax=12.0,
        reference_method="average", bad_channel_variance_threshold=1e-15, min_cycles=3.0,
    )
    base.update(overrides)
    return ValidatedDataPrepParams(**base)


class TestRecordingMetadata:
    def test_fixture_metadata_passes_end_to_end(self, synthetic_eeg_path):
        info = get_dataset_info.invoke({"data_path": str(synthetic_eeg_path)})
        conds = get_dataset_conditions.invoke({"data_path": str(synthetic_eeg_path)})
        params = validate_data_prep_input(_input(synthetic_eeg_path))
        validate_plan_against_recording(
            params,
            sfreq=info["sfreq"],
            channel_names=info["available_channels"],
            condition_labels=conds["conditions"].keys(),
        )

    def test_band_below_nyquist_passes(self):
        validate_plan_against_recording(
            _params(fmax=124.9), sfreq=250.0, channel_names=["F3", "F4"], condition_labels=["rest"]
        )

    @pytest.mark.parametrize("fmax", [125.0, 130.0])
    def test_band_at_or_above_nyquist_fails_not_clipped(self, fmax):
        with pytest.raises(DataPrepValidationError) as e:
            validate_plan_against_recording(
                _params(fmax=fmax), sfreq=250.0, channel_names=["F3", "F4"],
                condition_labels=["rest"],
            )
        assert _code(e) == "FREQUENCY_EXCEEDS_NYQUIST"

    @pytest.mark.parametrize("sfreq", [0, -250.0, float("nan"), float("inf"), True, "250"])
    def test_invalid_sampling_frequency(self, sfreq):
        with pytest.raises(DataPrepValidationError) as e:
            validate_plan_against_recording(
                _params(), sfreq=sfreq, channel_names=["F3", "F4"], condition_labels=["rest"]
            )
        assert _code(e) == "INVALID_SAMPLING_FREQUENCY"

    def test_missing_channel_fails(self):
        with pytest.raises(DataPrepValidationError) as e:
            validate_plan_against_recording(
                _params(), sfreq=250.0, channel_names=["F3", "C3"], condition_labels=["rest"]
            )
        assert _code(e) == "CHANNELS_NOT_IN_RECORDING"
        assert "F4" in str(e.value)

    def test_channel_match_is_exact_case(self):
        with pytest.raises(DataPrepValidationError) as e:
            validate_plan_against_recording(
                _params(), sfreq=250.0, channel_names=["f3", "f4"], condition_labels=["rest"]
            )
        assert _code(e) == "CHANNELS_NOT_IN_RECORDING"

    def test_many_missing_channels_are_capped_in_message(self):
        chans = tuple(f"CH{i}" for i in range(40))
        with pytest.raises(DataPrepValidationError) as e:
            validate_plan_against_recording(
                _params(channels=chans), sfreq=250.0, channel_names=["F3"],
                condition_labels=["rest"],
            )
        assert "more)" in str(e.value) and len(str(e.value)) < 500

    @pytest.mark.parametrize("labels", [[], ["task"], ["REST"], ["rest "]])
    def test_condition_must_exist_exactly(self, labels):
        with pytest.raises(DataPrepValidationError) as e:
            validate_plan_against_recording(
                _params(), sfreq=250.0, channel_names=["F3", "F4"], condition_labels=labels
            )
        assert _code(e) == "CONDITION_NOT_IN_RECORDING"


# --------------------------------------------------------------------------- #
# Contract properties
# --------------------------------------------------------------------------- #

class TestValidatedParamsContract:
    def test_validated_params_are_immutable(self, synthetic_eeg_path):
        params = validate_data_prep_input(_input(synthetic_eeg_path))
        with pytest.raises(ValidationError):
            params.reference_method = "mastoid"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            params.fmax = 500.0  # type: ignore[misc]

    def test_reference_type_only_admits_allowlisted_methods(self):
        with pytest.raises(ValidationError):
            _params(reference_method="mastoid")

    def test_validation_module_imports_no_llm_agent_ui_or_mne_code(self):
        """Validation must stay deterministic: no LLM/agent/UI/signal-processing imports."""
        import ast

        import fc_pipeline.deterministic.data_prep.validation as mod

        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"mne", "langchain", "langchain_core", "langchain_openai", "langgraph",
                     "chainlit", "openai", "nemoguardrails", "requests", "httpx", "subprocess"}
        assert imported.isdisjoint(forbidden), imported & forbidden
