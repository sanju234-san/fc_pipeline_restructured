"""Tests for ToolboxRegistry metadata, scoped LLM tool retrieval, and policy drift."""

import pytest
from fc_pipeline.toolbox.registry import ToolboxRegistry, toolbox_registry
from fc_pipeline.agentic.supervisor.action_policy import ACTION_RISK


def test_registry_metadata_fields_present_on_all_default_tools():
    """Verify that all default registered tools have the 4 new metadata fields populated."""
    tools = toolbox_registry.list_tools()
    assert len(tools) == 5, f"Expected 5 registered tools, got {len(tools)}"

    for meta in tools:
        assert hasattr(meta, "execution_mode")
        assert meta.execution_mode == "llm_callable"

        assert hasattr(meta, "pipeline_stage")
        assert meta.pipeline_stage in {"discovery", "resolution", "diagnostic"}

        assert hasattr(meta, "requires_prerequisites")
        assert isinstance(meta.requires_prerequisites, list)

        assert hasattr(meta, "state_changing")
        assert isinstance(meta.state_changing, bool)

    # Check specific tool configurations
    info_meta = toolbox_registry.get_metadata("get_dataset_info")
    assert info_meta.pipeline_stage == "discovery"
    assert info_meta.requires_prerequisites == []
    assert info_meta.state_changing is False

    cond_meta = toolbox_registry.get_metadata("get_dataset_conditions")
    assert cond_meta.pipeline_stage == "discovery"
    assert cond_meta.requires_prerequisites == []
    assert cond_meta.state_changing is False

    band_meta = toolbox_registry.get_metadata("resolve_frequency_band")
    assert band_meta.pipeline_stage == "resolution"
    assert band_meta.requires_prerequisites == ["get_dataset_info"]
    assert band_meta.state_changing is False

    chan_meta = toolbox_registry.get_metadata("resolve_channel_selection")
    assert chan_meta.pipeline_stage == "resolution"
    assert chan_meta.requires_prerequisites == ["get_dataset_info"]
    assert chan_meta.state_changing is False

    plot_meta = toolbox_registry.get_metadata("generate_dataset_overview_plot")
    assert plot_meta.pipeline_stage == "diagnostic"
    assert plot_meta.requires_prerequisites == ["get_dataset_info"]
    assert plot_meta.state_changing is True


def test_get_scoped_llm_tools_excludes_deterministic_tools():
    """Guarantee that tools with execution_mode='deterministic' are NEVER returned by get_scoped_llm_tools."""
    reg = ToolboxRegistry()

    def dummy_llm():
        pass

    def dummy_det():
        pass

    reg.register(
        name="tool_llm",
        category="test",
        description="LLM tool",
        callable_obj=dummy_llm,
        allowed_agents=["supervisor"],
        execution_mode="llm_callable",
        pipeline_stage="resolution",
    )
    reg.register(
        name="tool_deterministic",
        category="test",
        description="Deterministic tool",
        callable_obj=dummy_det,
        allowed_agents=["supervisor"],
        execution_mode="deterministic",
        pipeline_stage="resolution",
    )

    scoped = reg.get_scoped_llm_tools()
    assert dummy_llm in scoped
    assert dummy_det not in scoped

    # Even when explicitly filtering for the supervisor agent and resolution stage
    scoped_agent_stage = reg.get_scoped_llm_tools(agent="supervisor", stage="resolution")
    assert dummy_llm in scoped_agent_stage
    assert dummy_det not in scoped_agent_stage


def test_get_scoped_llm_tools_agent_and_stage_filtering():
    """Verify that get_scoped_llm_tools properly filters on agent and stage."""
    reg = ToolboxRegistry()

    def fn_sup_disc():
        pass

    def fn_sup_res():
        pass

    def fn_other_agent():
        pass

    reg.register(
        name="sup_discovery",
        category="dataset",
        description="Supervisor discovery",
        callable_obj=fn_sup_disc,
        allowed_agents=["supervisor"],
        execution_mode="llm_callable",
        pipeline_stage="discovery",
    )
    reg.register(
        name="sup_resolution",
        category="dataset",
        description="Supervisor resolution",
        callable_obj=fn_sup_res,
        allowed_agents=["supervisor"],
        execution_mode="llm_callable",
        pipeline_stage="resolution",
    )
    reg.register(
        name="other_tool",
        category="other",
        description="Other agent tool",
        callable_obj=fn_other_agent,
        allowed_agents=["other_agent"],
        execution_mode="llm_callable",
        pipeline_stage="discovery",
    )

    # Filter by agent
    sup_tools = reg.get_scoped_llm_tools(agent="supervisor")
    assert fn_sup_disc in sup_tools
    assert fn_sup_res in sup_tools
    assert fn_other_agent not in sup_tools

    # Filter by stage
    discovery_tools = reg.get_scoped_llm_tools(stage="discovery")
    assert fn_sup_disc in discovery_tools
    assert fn_other_agent in discovery_tools
    assert fn_sup_res not in discovery_tools

    # Filter by both agent and stage
    sup_disc = reg.get_scoped_llm_tools(agent="supervisor", stage="discovery")
    assert sup_disc == [fn_sup_disc]

    # Nonexistent agent
    empty = reg.get_scoped_llm_tools(agent="nonexistent")
    assert empty == []


def test_action_risk_policy_drift_check():
    """Drift test: Every tool registered with execution_mode='llm_callable' MUST have an entry in ACTION_RISK."""
    llm_tools = [
        meta.name
        for meta in toolbox_registry.list_tools()
        if meta.execution_mode == "llm_callable"
    ]
    assert len(llm_tools) > 0, "Expected at least one LLM-callable tool in registry"

    for tool_name in llm_tools:
        assert tool_name in ACTION_RISK, (
            f"Policy drift detected: Tool '{tool_name}' is registered as 'llm_callable' "
            f"in ToolboxRegistry, but has no risk tier entry in action_policy.ACTION_RISK."
        )
