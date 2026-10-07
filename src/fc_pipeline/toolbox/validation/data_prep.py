"""Deterministic, fail-closed validation for the Data Preparation stage.

Canonical implementation in the centralized EEG Toolbox.
Backward-compatible re-export shim:
    fc_pipeline.deterministic.data_prep.validation

Nothing in this module calls an LLM, loads EEG samples, or approves anything.
Every check either passes or raises :class:`DataPrepValidationError`; no check
"repairs" a value. In particular this module never turns a falsy Gate 1 flag
into a truthy one and never infers a human approval that was not recorded.

Two layers:

1. :func:`validate_data_prep_input` (pre-load). Checks Gate 1 approval, run_id,
   AnalysisPlan, the parameter manifest and the raw-data path, and returns an
   immutable :class:`ValidatedDataPrepParams`. Downstream modules consume that
   object rather than re-reading manifest strings.
2. :func:`validate_plan_against_recording` (post-header-read). Pure function over
   primitive recording metadata (sfreq, channel labels, condition labels), so it
   is testable without MNE and keeps loading in ``loader.py``.

Also provides :func:`build_safe_output_path`, the single place output filenames
are constructed (used later by plotting/pipeline).

Manifest approval policy
------------------------
Data Prep consumes exactly the manifest rows in ``REQUIRED_MANIFEST_PARAMETERS``.
For each consumed row the effective value is:

* ``human_approved_value`` when present (must be non-blank), otherwise
* ``proposed_value`` -- but ONLY when the row does not set ``needs_human_input``.

A consumed row with ``needs_human_input=True`` and no ``human_approved_value``
blocks execution. Advisory/informational rows (e.g. ``trial_adequacy``) are not
consumed by Data Prep and therefore never block it.

Error messages deliberately never echo file paths or free-text human input.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import (
    Collection,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Leaf contracts (imported first, before any deterministic.* imports, to break
# the circular import between toolbox.validation ↔ deterministic.data_prep.*).
# These names continue to be re-exported from this module so existing
#   from fc_pipeline.toolbox.validation.data_prep import REQUIRED_MANIFEST_PARAMETERS
# consumers keep working with no code changes.
# ---------------------------------------------------------------------------
from fc_pipeline.schemas.data_prep_contracts import (
    DataPrepValidationError,
    REFERENCE_ALIASES,
    RECOGNISED_UNSUPPORTED_REFERENCES,
    REQUIRED_MANIFEST_PARAMETERS,
    SUPPORTED_OUTPUT_SUFFIXES,
    SUPPORTED_RAW_SUFFIXES,
    SUPPORTED_REFERENCE_METHODS,
)
from fc_pipeline.schemas.data_prep_models import (
    DataPrepInput,
    ReferenceMethod,
    ValidatedDataPrepParams,
)
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Private allowlists / helpers (remain here; not part of the leaf contract)
# --------------------------------------------------------------------------- #

# The canonical reference alias map now lives in schemas.data_prep_contracts as
# ``REFERENCE_ALIASES``. Keep a local private alias with the original name so
# this file's private functions don't need to be edited (zero-logic migration).
_REFERENCE_ALIASES: Mapping[str, str] = REFERENCE_ALIASES
_RECOGNISED_UNSUPPORTED_REFERENCES = RECOGNISED_UNSUPPORTED_REFERENCES

# run_id becomes part of output filenames: no separators, dots or whitespace.
# Rejected, never rewritten (rewriting could make two runs collide).
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
_OUTPUT_STEM_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,39}")

_MAX_PATH_CHARS = 4096
_MAX_REPORTED_LABELS = 10
_MAX_LABEL_CHARS = 32

# Strict decimal/scientific notation. float() alone is too permissive: it
# accepts "nan", "inf" and underscores ("1_0").
_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(rf"[+-]?{_NUM}")
_FREQ_BAND_RE = re.compile(
    rf"(?P<name>.*)\(\s*(?P<fmin>{_NUM})\s*[-\u2013]\s*(?P<fmax>{_NUM})\s*Hz\s*\)\s*"
)
# The Supervisor renders freq_band with one decimal place, so a proposed value
# can only be trusted to half that resolution.
_FREQ_DISPLAY_TOLERANCE_HZ = 0.05 + 1e-9

# Status string get_dataset_info yields for headerless-reference recordings; the
# manifest carries it verbatim, e.g. "unreferenced (proposed: average)".
_UNREFERENCED_PROPOSAL_RE = re.compile(r"unreferenced \(proposed: ([a-z ]+)\)")


# --------------------------------------------------------------------------- #
# Local helpers (error factory moved AFTER imports above so DataPrepValidationError
# is already populated from the leaf contract).
# --------------------------------------------------------------------------- #


def _fail(code: str, message: str) -> DataPrepValidationError:
    return DataPrepValidationError(code, message)


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #

def _parse_number(raw: str, param: str) -> float:
    text = raw.strip()
    if not _NUMBER_RE.fullmatch(text):
        raise _fail("INVALID_PARAMETER", f"'{param}' must be a plain decimal number.")
    value = float(text)
    if not math.isfinite(value):
        raise _fail("INVALID_PARAMETER", f"'{param}' must be finite.")
    return value


def _parse_positive(raw: str, param: str) -> float:
    value = _parse_number(raw, param)
    if value <= 0.0:
        raise _fail("INVALID_PARAMETER", f"'{param}' must be greater than zero.")
    return value


def _short(label: object) -> str:
    text = str(label)
    return text if len(text) <= _MAX_LABEL_CHARS else text[:_MAX_LABEL_CHARS] + "..."


def _format_labels(labels: Iterable[object]) -> str:
    items = [_short(x) for x in labels]
    shown = ", ".join(items[:_MAX_REPORTED_LABELS])
    extra = len(items) - _MAX_REPORTED_LABELS
    return shown + (f" (+{extra} more)" if extra > 0 else "")


# --------------------------------------------------------------------------- #
# Gate 1 approval
# --------------------------------------------------------------------------- #

def _require_gate_1_approval(data_prep_input: DataPrepInput) -> None:
    # `is not True`, not truthiness: only the real boolean True counts.
    if data_prep_input.gate_1_approved is not True:
        raise _fail(
            "GATE_1_NOT_APPROVED",
            "Gate 1 has not been approved by a human; Data Preparation will not run.",
        )
    if data_prep_input.preflight_confirmed is not True:
        raise _fail(
            "PREFLIGHT_NOT_CONFIRMED",
            "Preflight has not been confirmed; Data Preparation will not run.",
        )


# --------------------------------------------------------------------------- #
# run_id and output paths
# --------------------------------------------------------------------------- #

def validate_run_id(run_id: object) -> str:
    """Return ``run_id`` unchanged if it is safe as a filename component."""
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise _fail(
            "INVALID_RUN_ID",
            "run_id must be 1-80 characters of letters, digits, '_' or '-', "
            "starting with a letter or digit.",
        )
    return run_id


def build_safe_output_path(
    output_dir: PathLike, run_id: str, stem: str, suffix: str
) -> Path:
    """Build ``<output_dir>/<stem>_<run_id><suffix>`` with containment checks.

    ``output_dir`` must already exist (the caller creates it from trusted
    configuration). ``stem`` and ``suffix`` come from allowlists, ``run_id`` is
    validated, and the result must sit directly inside the resolved
    ``output_dir`` and must not be an existing symlink.
    """
    validate_run_id(run_id)
    if not isinstance(stem, str) or not _OUTPUT_STEM_PATTERN.fullmatch(stem):
        raise _fail("INVALID_OUTPUT_PATH", "Output file stem is not allowed.")
    if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        raise _fail("INVALID_OUTPUT_PATH", "Output file suffix is not allowed.")

    try:
        base = Path(output_dir)
    except TypeError:
        raise _fail("INVALID_OUTPUT_PATH", "Output directory is not a valid path.")
    if ".." in base.parts or "\x00" in str(base):
        raise _fail("INVALID_OUTPUT_PATH", "Output directory must not contain '..'.")
    try:
        base_resolved = base.resolve(strict=True)
    except (OSError, RuntimeError):
        raise _fail("INVALID_OUTPUT_PATH", "Output directory does not exist.")
    if not base_resolved.is_dir():
        raise _fail("INVALID_OUTPUT_PATH", "Output directory is not a directory.")

    target = base_resolved / f"{stem}_{run_id}{suffix}"
    if target.is_symlink():
        raise _fail("INVALID_OUTPUT_PATH", "Refusing to write through a symlink.")
    if target.resolve(strict=False).parent != base_resolved:
        raise _fail("INVALID_OUTPUT_PATH", "Output path escapes the output directory.")
    return target


# --------------------------------------------------------------------------- #
# Raw data path
# --------------------------------------------------------------------------- #

def validate_raw_data_path(
    raw_data_path: object,
    *,
    allowed_data_roots: Optional[Sequence[PathLike]] = None,
) -> Path:
    """Validate the raw EEG file location and return its resolved path.

    The path must come from trusted application state/configuration, never from
    an LLM. Checks: non-empty string, no NUL, no ``..`` component, exists, is a
    regular file, has an allowlisted suffix (checked on the symlink-resolved
    target), and -- when ``allowed_data_roots`` is given -- lies inside one of
    them after resolution. Without roots only these structural checks apply;
    the caller (graph wiring) is responsible for supplying trusted roots.
    """
    if not isinstance(raw_data_path, str) or not raw_data_path.strip():
        raise _fail("MISSING_DATA_PATH", "raw_data_path is missing.")
    if "\x00" in raw_data_path or len(raw_data_path) > _MAX_PATH_CHARS:
        raise _fail("INVALID_DATA_PATH", "raw_data_path is malformed.")

    candidate = Path(raw_data_path)  # deliberately no expanduser()
    if ".." in candidate.parts:
        raise _fail("INVALID_DATA_PATH", "raw_data_path must not contain '..'.")

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise _fail("DATA_FILE_NOT_FOUND", "The raw data file does not exist.")
    except (OSError, RuntimeError):
        raise _fail("INVALID_DATA_PATH", "raw_data_path could not be resolved.")

    if not resolved.is_file():
        raise _fail("DATA_PATH_NOT_A_FILE", "raw_data_path is not a regular file.")

    name = resolved.name.lower()
    if not any(name.endswith(sfx) for sfx in SUPPORTED_RAW_SUFFIXES):
        raise _fail(
            "UNSUPPORTED_DATA_FORMAT",
            "Unsupported raw file type. Supported: "
            + ", ".join(SUPPORTED_RAW_SUFFIXES) + ".",
        )

    if allowed_data_roots is not None:
        roots = []
        for root in allowed_data_roots:
            try:
                roots.append(Path(root).resolve(strict=False))
            except (OSError, RuntimeError, TypeError):
                continue
        if not any(resolved.is_relative_to(r) for r in roots):
            raise _fail(
                "DATA_PATH_OUTSIDE_ALLOWED_ROOTS",
                "raw_data_path is outside the permitted data directories.",
            )
    return resolved


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #

def _validate_plan(plan: object) -> AnalysisPlan:
    if plan is None:
        raise _fail("MISSING_PLAN", "No AnalysisPlan was provided.")
    if not isinstance(plan, AnalysisPlan):
        raise _fail("INVALID_PLAN", "plan is not an AnalysisPlan.")
    try:
        # Re-validate: pydantic does not validate on attribute assignment or on
        # model_construct(), so the instance itself is not trusted.
        plan = AnalysisPlan.model_validate(plan.model_dump())
    except Exception:
        raise _fail("INVALID_PLAN", "AnalysisPlan failed schema validation.")

    if not plan.metrics:
        raise _fail("INVALID_PLAN", "Plan has no metrics.")

    band = plan.freq_band
    if not (math.isfinite(band.fmin) and math.isfinite(band.fmax)):
        raise _fail("INVALID_FREQUENCY_BAND", "Frequency band bounds must be finite.")
    if band.fmin <= 0.0 or band.fmin >= band.fmax:
        raise _fail(
            "INVALID_FREQUENCY_BAND",
            "Frequency band requires 0 < fmin < fmax.",
        )

    if len(plan.channels) < 2:
        raise _fail(
            "INVALID_CHANNELS",
            "Connectivity analysis requires at least 2 channels in the plan.",
        )
    if any((not isinstance(c, str)) or not c.strip() for c in plan.channels):
        raise _fail("INVALID_CHANNELS", "Plan contains an empty channel label.")
    if len(set(plan.channels)) != len(plan.channels):
        raise _fail("INVALID_CHANNELS", "Plan contains duplicate channel labels.")

    if not plan.condition or not plan.condition.strip():
        raise _fail("INVALID_CONDITION", "Plan condition is empty.")
    return plan


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def _index_required_entries(
    manifest: object,
) -> Dict[str, ParameterManifestEntry]:
    if not isinstance(manifest, (list, tuple)):
        raise _fail("INVALID_MANIFEST", "parameter_manifest must be a list.")

    found: Dict[str, ParameterManifestEntry] = {}
    for entry in manifest:
        if not isinstance(entry, ParameterManifestEntry):
            raise _fail("INVALID_MANIFEST", "Manifest contains a non-manifest entry.")
        if entry.name in REQUIRED_MANIFEST_PARAMETERS:
            if entry.name in found:
                raise _fail(
                    "DUPLICATE_MANIFEST_ENTRY",
                    f"Manifest has more than one '{entry.name}' entry.",
                )
            found[entry.name] = entry

    missing = [n for n in REQUIRED_MANIFEST_PARAMETERS if n not in found]
    if missing:
        raise _fail(
            "MISSING_MANIFEST_PARAMETER",
            "Manifest is missing required parameter(s): " + ", ".join(missing) + ".",
        )
    return found


def _effective_value(entry: ParameterManifestEntry) -> str:
    """Resolve the value Data Prep may act on, honouring human approval."""
    approved = entry.human_approved_value
    if approved is not None:
        if not isinstance(approved, str) or not approved.strip():
            raise _fail(
                "INVALID_HUMAN_APPROVAL",
                f"Approved value for '{entry.name}' is blank or malformed.",
            )
        return approved.strip()

    if entry.needs_human_input:
        raise _fail(
            "HUMAN_APPROVAL_REQUIRED",
            f"'{entry.name}' requires an explicit human-approved value, "
            "but none was recorded.",
        )

    proposed = entry.proposed_value
    if not isinstance(proposed, str) or not proposed.strip():
        raise _fail(
            "MISSING_MANIFEST_VALUE", f"'{entry.name}' has no usable value."
        )
    return proposed.strip()


def _parse_reference(value: str) -> str:
    norm = " ".join(value.lower().split())
    wrapped = _UNREFERENCED_PROPOSAL_RE.fullmatch(norm)
    if wrapped:
        norm = wrapped.group(1).strip()

    if norm in _RECOGNISED_UNSUPPORTED_REFERENCES:
        raise _fail(
            "REFERENCE_METHOD_UNSUPPORTED",
            f"Reference '{norm}' is recognised but cannot be executed: the "
            "plan/manifest carry no electrode specification for it.",
        )
    method = _REFERENCE_ALIASES.get(norm)
    if method is None or method not in SUPPORTED_REFERENCE_METHODS:
        raise _fail(
            "INVALID_REFERENCE",
            "Reference method is not supported. Supported: "
            + ", ".join(sorted(SUPPORTED_REFERENCE_METHODS)) + ".",
        )
    return method


def _check_channels_match_plan(value: str, plan: AnalysisPlan) -> None:
    listed = [c.strip() for c in value.split(",")]
    if listed != list(plan.channels):
        raise _fail(
            "MANIFEST_PLAN_MISMATCH",
            "Manifest 'channels' does not match the AnalysisPlan.",
        )


def _check_condition_matches_plan(value: str, plan: AnalysisPlan) -> None:
    if value != plan.condition:
        raise _fail(
            "MANIFEST_PLAN_MISMATCH",
            "Manifest 'condition' does not match the AnalysisPlan.",
        )


def _check_freq_band_matches_plan(value: str, plan: AnalysisPlan) -> None:
    match = _FREQ_BAND_RE.fullmatch(value)
    if not match:
        raise _fail(
            "INVALID_PARAMETER",
            "Manifest 'freq_band' must look like 'name (fmin - fmax Hz)'.",
        )
    fmin, fmax = float(match.group("fmin")), float(match.group("fmax"))
    band = plan.freq_band
    if (
        abs(fmin - band.fmin) > _FREQ_DISPLAY_TOLERANCE_HZ
        or abs(fmax - band.fmax) > _FREQ_DISPLAY_TOLERANCE_HZ
    ):
        raise _fail(
            "MANIFEST_PLAN_MISMATCH",
            "Manifest 'freq_band' does not match the AnalysisPlan.",
        )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def validate_data_prep_input(
    data_prep_input: DataPrepInput,
    *,
    allowed_data_roots: Optional[Sequence[PathLike]] = None,
) -> ValidatedDataPrepParams:
    """Validate everything knowable before touching EEG samples.

    Raises :class:`DataPrepValidationError` on the first failure. The check
    order is deliberate: approvals first, filesystem last.
    """
    if not isinstance(data_prep_input, DataPrepInput):
        raise _fail("INVALID_INPUT", "Data Preparation input has the wrong type.")

    _require_gate_1_approval(data_prep_input)
    run_id = validate_run_id(data_prep_input.run_id)
    plan = _validate_plan(data_prep_input.plan)

    entries = _index_required_entries(data_prep_input.parameter_manifest)
    values = {name: _effective_value(entries[name]) for name in REQUIRED_MANIFEST_PARAMETERS}

    # Scientific axes: the plan is canonical, so a manifest/approval that
    # disagrees with it is a conflict to surface, never something to reconcile.
    _check_freq_band_matches_plan(values["freq_band"], plan)
    _check_channels_match_plan(values["channels"], plan)
    _check_condition_matches_plan(values["condition"], plan)

    reference = _parse_reference(values["reference"])
    variance_threshold = _parse_positive(
        values["bad_channel_variance_threshold"], "bad_channel_variance_threshold"
    )
    min_cycles = _parse_positive(values["min_cycles"], "min_cycles")

    data_path = validate_raw_data_path(
        data_prep_input.raw_data_path, allowed_data_roots=allowed_data_roots
    )

    return ValidatedDataPrepParams(
        raw_data_path=data_path,
        run_id=run_id,
        channels=tuple(plan.channels),
        condition=plan.condition,
        fmin=plan.freq_band.fmin,
        fmax=plan.freq_band.fmax,
        reference_method=reference,  # type: ignore[arg-type]  # allowlisted above
        bad_channel_variance_threshold=variance_threshold,
        min_cycles=min_cycles,
    )


def validate_plan_against_recording(
    params: ValidatedDataPrepParams,
    *,
    sfreq: float,
    channel_names: Collection[str],
    condition_labels: Collection[str],
) -> None:
    """Check validated parameters against recording metadata (no signal data).

    Nothing is adjusted: a band above Nyquist, a channel the recording does not
    have, or an absent condition fails rather than being clipped or substituted.
    Epoch-duration vs. ``min_cycles`` is checked in ``epoching.py`` because it
    needs the epoch definition.
    """
    if (
        isinstance(sfreq, bool)
        or not isinstance(sfreq, (int, float))
        or not math.isfinite(sfreq)
        or sfreq <= 0
    ):
        raise _fail("INVALID_SAMPLING_FREQUENCY", "Sampling frequency is invalid.")

    nyquist = sfreq / 2.0
    if params.fmax >= nyquist:
        raise _fail(
            "FREQUENCY_EXCEEDS_NYQUIST",
            f"Band upper edge {params.fmax:g} Hz must be below the Nyquist "
            f"frequency {nyquist:g} Hz.",
        )

    available = set(channel_names)
    missing = [c for c in params.channels if c not in available]
    if missing:
        raise _fail(
            "CHANNELS_NOT_IN_RECORDING",
            "Requested channel(s) not in the recording: " + _format_labels(missing) + ".",
        )

    if params.condition not in set(condition_labels):
        raise _fail(
            "CONDITION_NOT_IN_RECORDING",
            "Requested condition was not found in the recording's events.",
        )
