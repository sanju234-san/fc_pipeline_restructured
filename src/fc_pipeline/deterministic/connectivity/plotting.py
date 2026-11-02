"""Viridis heatmap and circular network connectivity graph rendering.

Follows Appendix A and Section 6.3 of PIPELINE_PROTOTYPE_DESIGN.md.
Strictly deterministic visualization; zero LLM / VLM / MLflow imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for server / background threads
import matplotlib.pyplot as plt
import numpy as np

from fc_pipeline.config.constants import (
    FIGURE_DPI,
    FIGURE_SIZE_INCHES,
    HEATMAP_COLORMAP,
    HEATMAP_VMIN,
    HEATMAP_VMAX,
    NETWORK_DENSITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


def render_connectivity_heatmap(
    matrix: np.ndarray,
    channel_names: Sequence[str],
    metric_name: str,
    output_path: Path,
    title: Optional[str] = None,
    vmin: float = HEATMAP_VMIN,
    vmax: float = HEATMAP_VMAX,
    cmap: str = HEATMAP_COLORMAP,
    dpi: int = FIGURE_DPI,
    figsize: tuple[int, int] = FIGURE_SIZE_INCHES,
) -> str:
    """Renders a symmetric [0, 1] bounded connectivity heatmap with explicit channel labels.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric (N, N) connectivity matrix.
    channel_names : Sequence[str]
        List of channel label strings.
    metric_name : str
        Name of metric (e.g., 'pli', 'wpli', 'imcoh', 'plv', 'coh').
    output_path : Path
        Destination PNG path.
    title : str, optional
        Plot title.
    vmin, vmax : float
        Color scale limits. Defaults to 0.0 and 1.0.
    cmap : str
        Colormap name. Defaults to 'viridis'.
    dpi : int
        Figure DPI. Defaults to 150.
    figsize : tuple
        Figure size in inches. Defaults to (10, 8).

    Returns
    -------
    str
        Resolved path to the saved heatmap image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(channel_names)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(channel_names, rotation=90, fontsize=9)
    ax.set_yticklabels(channel_names, fontsize=9)

    ax.set_title(
        title or f"Functional Connectivity Heatmap: {metric_name.upper()} (Scale: [{vmin:.1f}, {vmax:.1f}])",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{metric_name.upper()} Magnitude", rotation=270, labelpad=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(output_path.resolve())


def render_network_graph(
    matrix: np.ndarray,
    channel_names: Sequence[str],
    metric_name: str,
    output_path: Path,
    density_threshold: float = NETWORK_DENSITY_THRESHOLD,
    title: Optional[str] = None,
    dpi: int = FIGURE_DPI,
    figsize: tuple[int, int] = FIGURE_SIZE_INCHES,
) -> str:
    """Renders a 2D circular network connectivity graph showing strongest edges.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric (N, N) connectivity matrix.
    channel_names : Sequence[str]
        List of channel label strings.
    metric_name : str
        Name of metric.
    output_path : Path
        Destination PNG path.
    density_threshold : float
        Proportion of strongest connections to display (default 0.40 = top 40%).
    title : str, optional
        Plot title.
    dpi : int
        Figure DPI. Defaults to 150.
    figsize : tuple
        Figure size in inches. Defaults to (10, 8).

    Returns
    -------
    str
        Resolved path to the saved network graph image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(channel_names)

    # Angles for circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Start at top (pi/2) and go clockwise
    angles = np.pi / 2.0 - angles
    x_nodes = np.cos(angles)
    y_nodes = np.sin(angles)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")

    # Extract all off-diagonal upper triangle edges
    edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, float(matrix[i, j])))

    edges.sort(key=lambda x: x[2], reverse=True)
    n_edges_to_draw = max(1, int(round(len(edges) * density_threshold))) if edges else 0
    top_edges = edges[:n_edges_to_draw]

    # Draw edges
    if top_edges:
        weights = [e[2] for e in top_edges]
        w_min = min(weights)
        w_max = max(weights) if max(weights) > min(weights) else min(weights) + 1e-6

        cmap = plt.get_cmap(HEATMAP_COLORMAP)

        for i, j, weight in top_edges:
            norm_w = (weight - w_min) / (w_max - w_min)
            color = cmap(norm_w)
            linewidth = 0.8 + 2.5 * norm_w
            alpha = 0.35 + 0.55 * norm_w
            ax.plot(
                [x_nodes[i], x_nodes[j]],
                [y_nodes[i], y_nodes[j]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=1,
            )

    # Draw nodes
    ax.scatter(x_nodes, y_nodes, s=350, c="#2c3e50", edgecolors="#1abc9c", linewidths=2.0, zorder=2)

    # Draw node labels slightly outside the circle
    label_offset = 1.15
    for idx, name in enumerate(channel_names):
        lx = label_offset * np.cos(angles[idx])
        ly = label_offset * np.sin(angles[idx])
        ax.text(
            lx,
            ly,
            name,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=3,
        )

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    plot_title = title or (
        f"Circular Network Graph: {metric_name.upper()} "
        f"(Top {int(density_threshold * 100)}% Strongest Edges)"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(output_path.resolve())


def render_all_connectivity_plots(
    matrices: Dict[str, np.ndarray],
    channel_names: Sequence[str],
    output_dir: Path,
    run_id: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Renders both heatmaps and circular network graphs for all computed metrics.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str]]
        (heatmap_image_paths, network_image_paths) mapping metric name to image path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmaps: Dict[str, str] = {}
    networks: Dict[str, str] = {}

    for metric_name, matrix in matrices.items():
        heatmap_path = output_dir / f"heatmap_{metric_name}_{run_id}.png"
        network_path = output_dir / f"network_{metric_name}_{run_id}.png"

        render_connectivity_heatmap(
            matrix=matrix,
            channel_names=channel_names,
            metric_name=metric_name,
            output_path=heatmap_path,
        )
        heatmaps[metric_name] = str(heatmap_path.resolve())

        render_network_graph(
            matrix=matrix,
            channel_names=channel_names,
            metric_name=metric_name,
            output_path=network_path,
        )
        networks[metric_name] = str(network_path.resolve())

    return heatmaps, networks
