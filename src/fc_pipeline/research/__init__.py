"""Optional, isolated research helpers (ddgs web search, Deep Agents, DSPy).

Nothing in this package is imported by the Supervisor, the graph or Data Prep.
It never computes, resolves or describes dataset facts - that stays with the
deterministic toolbox. It exists for *methodological* questions only
("what does wPLI measure?") and for optimising LLM prompts offline.
"""
