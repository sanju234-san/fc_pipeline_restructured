"""Appendix A cosmetic constants, colormaps, DPI, and file templates."""

from pathlib import Path

# Visualization Constants
HEATMAP_COLORMAP: str = "viridis"
HEATMAP_VMIN: float = 0.0
HEATMAP_VMAX: float = 1.0
FIGURE_DPI: int = 150
FIGURE_SIZE_INCHES: tuple[int, int] = (10, 8)
NETWORK_GRAPH_LAYOUT: str = "circular"
NETWORK_DENSITY_THRESHOLD: float = 0.40  # Top 40% strongest connections

# Path & File Templates
OUTPUT_DIR: Path = Path("outputs")
LOG_DIR: Path = Path("logs")

REPORT_FILENAME_TEMPLATE: str = "report_{run_id}.md"
TRACE_FILENAME_TEMPLATE: str = "trace_{run_id}.json"
CONNECTIVITY_CSV_FILENAME_TEMPLATE: str = "{metric}_{run_id}.csv"
EVIDENCE_CSV_FILENAME_TEMPLATE: str = "evidence_summary_{run_id}.csv"
EVALUATION_CSV_FILENAME_TEMPLATE: str = "evaluation_summary_{run_id}.csv"
CONSOLIDATED_REPORT_CSV_FILENAME_TEMPLATE: str = "consolidated_connectivity_report_{run_id}.csv"
HEATMAP_FILENAME_TEMPLATE: str = "heatmap_{metric}_{run_id}.png"
NETWORK_FILENAME_TEMPLATE: str = "network_{metric}_{run_id}.png"

