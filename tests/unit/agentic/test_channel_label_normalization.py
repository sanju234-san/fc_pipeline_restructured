"""Regression tests for EEG/EDF electrode-label normalization."""

import sys
import types

# This test file is intentionally dependency-light.  The project normally uses
# langchain_core.tools.tool; provide a tiny decorator fallback for this focused
# normalization test when the full LangChain stack is not installed.
if "langchain_core.tools" not in sys.modules:
    lc = types.ModuleType("langchain_core")
    tools = types.ModuleType("langchain_core.tools")
    tools.tool = lambda fn: fn
    sys.modules.setdefault("langchain_core", lc)
    sys.modules.setdefault("langchain_core.tools", tools)

from fc_pipeline.agentic.supervisor.tools.channel_selection import (  # noqa: E402
    normalize_channel_label,
    resolve_channel_selection,
)


def test_normalize_common_edf_decorations():
    assert normalize_channel_label("EEG C3-REF") == "C3"
    assert normalize_channel_label("C4.") == "C4"
    assert normalize_channel_label("Fc5.") == "FC5"
    assert normalize_channel_label("Fcz.") == "FCZ"


def test_user_c3_c4_matches_period_suffixed_edf_headers():
    channels = ["Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "C3.", "C1.", "Cz.", "C2.", "C4."]
    # With langchain installed the tool is a StructuredTool (use .invoke); the
    # dependency-light fallback decorator above leaves a plain callable.
    if hasattr(resolve_channel_selection, "invoke"):
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "C3, C4", "available_channels": channels}
        )
    else:
        result = resolve_channel_selection("C3, C4", channels)
    assert result["resolved_channels"] == ["C3.", "C4."]
    assert result["confidence"] == 1.0
    assert result["needs_human_input"] is False
    assert result["error"] is None
