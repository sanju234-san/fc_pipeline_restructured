"""Toolbox registry for centralized EEG tools.

Provides registration, lookup, metadata tracking, and agent permission gating
for all deterministic scientific tools across dataset, preprocessing,
connectivity, visualization, and validation categories.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolMetadata:
    """Metadata specification for a tool registered in the toolbox."""

    name: str
    category: str
    description: str
    callable: Any
    allowed_agents: List[str] = field(default_factory=list)
    execution_mode: str = "deterministic"
    pipeline_stage: str = "discovery"
    requires_prerequisites: List[str] = field(default_factory=list)
    state_changing: bool = False


class ToolboxRegistry:
    """Central registry of deterministic scientific tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolMetadata] = {}

    def register(
        self,
        name: str,
        category: str,
        description: str,
        callable_obj: Any,
        allowed_agents: Optional[List[str]] = None,
        execution_mode: str = "deterministic",
        pipeline_stage: str = "discovery",
        requires_prerequisites: Optional[List[str]] = None,
        state_changing: bool = False,
    ) -> None:
        """Registers a tool in the toolbox."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered in ToolboxRegistry.")
        self._tools[name] = ToolMetadata(
            name=name,
            category=category,
            description=description,
            callable=callable_obj,
            allowed_agents=allowed_agents or [],
            execution_mode=execution_mode,
            pipeline_stage=pipeline_stage,
            requires_prerequisites=requires_prerequisites or [],
            state_changing=state_changing,
        )

    def get_tool(self, name: str) -> Any:
        """Retrieves a registered tool callable by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in ToolboxRegistry.")
        return self._tools[name].callable

    def get_metadata(self, name: str) -> ToolMetadata:
        """Retrieves tool metadata by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in ToolboxRegistry.")
        return self._tools[name]

    def get_tools_by_agent(self, agent_name: str) -> List[Any]:
        """Returns all tool callables permitted for a given agent."""
        return [
            meta.callable
            for meta in self._tools.values()
            if agent_name in meta.allowed_agents or "*" in meta.allowed_agents
        ]

    def get_tools_by_category(self, category: str) -> List[Any]:
        """Returns all tool callables belonging to a scientific category."""
        return [
            meta.callable
            for meta in self._tools.values()
            if meta.category == category
        ]

    def get_scoped_llm_tools(
        self,
        agent: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> List[Any]:
        """Returns LLM-callable tool callables filtered by allowed agent and pipeline stage.

        Guarantee: Tools with execution_mode != 'llm_callable' (such as 'deterministic')
        can NEVER be returned by this method under any condition.
        """
        tools: List[Any] = []
        for meta in self._tools.values():
            if meta.execution_mode != "llm_callable":
                continue
            if agent is not None and (agent not in meta.allowed_agents and "*" not in meta.allowed_agents):
                continue
            if stage is not None and meta.pipeline_stage != stage:
                continue
            tools.append(meta.callable)
        return tools

    def list_tools(self) -> List[ToolMetadata]:
        """Lists all registered tools and their metadata."""
        return list(self._tools.values())


# Singleton registry instance
toolbox_registry = ToolboxRegistry()


def _initialize_default_registry() -> None:
    """Registers scientific tools with their default metadata from the toolbox."""
    from fc_pipeline.toolbox.dataset.channel_selection import resolve_channel_selection
    from fc_pipeline.toolbox.dataset.dataset_conditions import get_dataset_conditions
    from fc_pipeline.toolbox.dataset.dataset_info import get_dataset_info
    from fc_pipeline.toolbox.dataset.frequency_band import resolve_frequency_band
    from fc_pipeline.toolbox.visualization.overview_plot import generate_dataset_overview_plot

    toolbox_registry.register(
        name="get_dataset_info",
        category="dataset",
        description="Inspects raw EEG recording file via MNE and returns metadata.",
        callable_obj=get_dataset_info,
        allowed_agents=["supervisor"],
        execution_mode="llm_callable",
        pipeline_stage="discovery",
        requires_prerequisites=[],
        state_changing=False,
    )
    toolbox_registry.register(
        name="get_dataset_conditions",
        category="dataset",
        description="Extracts condition labels and event counts from EEG annotations.",
        callable_obj=get_dataset_conditions,
        allowed_agents=["supervisor"],
        execution_mode="llm_callable",
        pipeline_stage="discovery",
        requires_prerequisites=[],
        state_changing=False,
    )
    toolbox_registry.register(
        name="resolve_frequency_band",
        category="dataset",
        description="Resolves canonical band names and verifies against Nyquist and min cycles.",
        callable_obj=resolve_frequency_band,
        allowed_agents=["supervisor"],
        execution_mode="llm_callable",
        pipeline_stage="resolution",
        requires_prerequisites=["get_dataset_info"],
        state_changing=False,
    )
    toolbox_registry.register(
        name="resolve_channel_selection",
        category="dataset",
        description="Matches requested channels or regions against 10-20 montage in recording.",
        callable_obj=resolve_channel_selection,
        allowed_agents=["supervisor"],
        execution_mode="llm_callable",
        pipeline_stage="resolution",
        requires_prerequisites=["get_dataset_info"],
        state_changing=False,
    )
    toolbox_registry.register(
        name="generate_dataset_overview_plot",
        category="visualization",
        description="Generates diagnostic pre-analysis plot (time series + Welch PSD).",
        callable_obj=generate_dataset_overview_plot,
        allowed_agents=["supervisor"],
        execution_mode="llm_callable",
        pipeline_stage="diagnostic",
        requires_prerequisites=["get_dataset_info"],
        state_changing=True,
    )


# Automatically populate registry on module import
_initialize_default_registry()
