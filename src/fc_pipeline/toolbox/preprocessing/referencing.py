"""Re-referencing implementation (average, mastoid, bipolar).

Canonical implementation in the centralized EEG Toolbox.
Backward-compatible re-export shim:
    fc_pipeline.deterministic.data_prep.referencing
"""

from __future__ import annotations

import logging
from typing import Tuple

import mne

from fc_pipeline.schemas.data_prep_models import ValidatedDataPrepParams
from fc_pipeline.schemas.data_prep_contracts import DataPrepValidationError

logger = logging.getLogger(__name__)


def apply_reference(
    raw: mne.io.BaseRaw,
    params: ValidatedDataPrepParams,
) -> Tuple[mne.io.BaseRaw, str]:
    """Apply the validated reference method to the recording.

    Currently only ``"average"`` (Common Average Reference) is supported,
    matching ``ReferenceMethod = Literal["average"]`` in ``models.py``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        EEG recording after channel selection and bad-channel removal.
        Modified **in-place**.
    params : ValidatedDataPrepParams
        Must contain ``reference_method`` (only ``"average"`` accepted).

    Returns
    -------
    raw : mne.io.BaseRaw
        The same Raw object after re-referencing.
    applied_method : str
        The reference method that was applied (``"average"``).

    Raises
    ------
    DataPrepValidationError
        If the reference method is not supported at execution time.
    """
    method = params.reference_method

    if method != "average":
        # This should never happen if validation ran, but guard anyway.
        raise DataPrepValidationError(
            "REFERENCE_METHOD_UNSUPPORTED",
            f"Reference method '{method}' is not executable. "
            "Only 'average' (CAR) is currently supported.",
        )

    # Detect existing reference from the recording
    existing_ref = raw.info.get("custom_ref_applied", False)
    logger.info(
        "Applying average reference (CAR). custom_ref_applied was: %s",
        existing_ref,
    )

    # Apply average reference in-place (projection=False for immediate application)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    logger.info(
        "Average reference applied. Channels: %d",
        len(raw.ch_names),
    )

    return raw, "average"
