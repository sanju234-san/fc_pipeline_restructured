"""Data Preparation shared constants and the typed validation exception.

Leaf package: NO imports of ``fc_pipeline.toolbox.*`` or
``fc_pipeline.deterministic.*``. Used by both sides of the toolbox ↔
deterministic shim chain to break the circular import between
``toolbox.validation.data_prep`` and ``deterministic.data_prep.validation``.

Migration note (zero-logic change):
``SUPPORTED_RAW_SUFFIXES``, ``SUPPORTED_REFERENCE_METHODS``,
``REQUIRED_MANIFEST_PARAMETERS``, ``SUPPORTED_OUTPUT_SUFFIXES``, and
``DataPrepValidationError`` previously lived inside
``fc_pipeline.toolbox.validation.data_prep``. The module still re-exports them
from here so existing importers (``from fc_pipeline.toolbox.validation.data_prep
import REQUIRED_MANIFEST_PARAMETERS``) continue to work unchanged.

``_REFERENCE_ALIASES``, ``_RECOGNISED_UNSUPPORTED_REFERENCES``, ``_RUN_ID_PATTERN``
and other private module-level helpers remain in the toolbox validation module
(they are not needed outside it).
"""

from __future__ import annotations

from typing import FrozenSet, Mapping, Tuple
from typing import get_args

from fc_pipeline.schemas.data_prep_models import ReferenceMethod

# --------------------------------------------------------------------------- #
# Allowlists / constants
# --------------------------------------------------------------------------- #

# Raw formats MNE can read without executing pickled/arbitrary code. Compared
# case-insensitively against the *resolved* target file name.
SUPPORTED_RAW_SUFFIXES: Tuple[str, ...] = (
    ".fif.gz",
    ".fif",
    ".edf",
    ".bdf",
    ".set",
    ".vhdr",
)

# Single source of truth is the ReferenceMethod Literal in data_prep_models.py.
SUPPORTED_REFERENCE_METHODS: FrozenSet[str] = frozenset(get_args(ReferenceMethod))

# Recognised reference aliases: aliases map to a canonical ReferenceMethod
# value. This map is consumed by toolbox.validation.data_prep; keeping it at
# leaf scope means deterministic/preprocessing shims can also reference it.
REFERENCE_ALIASES: Mapping[str, str] = {
    "average": "average",
    "car": "average",
    "common average": "average",
    "common average reference": "average",
}

# Recognised methods that cannot be executed because the plan/manifest carry no
# electrode specification for them. Reported distinctly from "unknown".
RECOGNISED_UNSUPPORTED_REFERENCES: FrozenSet[str] = frozenset({"mastoid", "bipolar"})

# The manifest rows Data Prep consumes (see module docstring of
# toolbox.validation.data_prep).
REQUIRED_MANIFEST_PARAMETERS: Tuple[str, ...] = (
    "freq_band",
    "channels",
    "condition",
    "reference",
    "bad_channel_variance_threshold",
    "min_cycles",
)

# Output filenames Data Prep writing accepts (image artifacts + preprocessed
# MNE .fif epochs).
SUPPORTED_OUTPUT_SUFFIXES: Tuple[str, ...] = (".png", ".fif")


# --------------------------------------------------------------------------- #
# Error type
# --------------------------------------------------------------------------- #


class DataPrepValidationError(Exception):
    """A deterministic validation failure with a stable machine-readable code.

    ``str(exc)`` is ``"CODE: message"``, matching the ``data_prep_error`` style
    used elsewhere in the design (e.g. ``INSUFFICIENT_CHANNELS: ...``).
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")
