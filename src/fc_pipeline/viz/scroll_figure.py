"""Real-signal, scrollable Plotly figure for the Chainlit EEG viewer (works headless)."""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[3]


def _newest(patterns):
    hits = []
    for base in (_ROOT / "outputs", Path.cwd() / "outputs"):
        for pat in patterns:
            hits += glob.glob(str(base / pat), recursive=True)
    hits = [h for h in set(hits) if os.path.isfile(h)]
    return max(hits, key=os.path.getmtime) if hits else None


def _load(path):
    import mne
    low = path.lower()
    bounds = []
    if low.endswith(".edf"):
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif low.endswith(".bdf"):
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    else:
        try:
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        except Exception:
            ep = mne.read_epochs(path, preload=True, verbose=False)
            data = ep.get_data()
            n, _, t = data.shape
            raw = mne.io.RawArray(np.concatenate(list(data), axis=1), ep.info.copy(), verbose=False)
            bounds = [i * t / ep.info["sfreq"] for i in range(1, n)]
    try:
        mne.datasets.eegbci.standardize(raw)
    except Exception:
        pass
    return raw, bounds


def build_scroll_figure(raw, bounds=(), n_vis=20, window_s=10.0, scale_uv=None,
                        max_seconds=60.0, title=""):
    import plotly.graph_objects as go
    raw = raw.copy().pick("eeg")
    sf = raw.info["sfreq"]
    stop = min(int(max_seconds * sf), raw.n_times)
    data, times = raw[:, :stop]
    data = data * 1e6
    stride = max(1, stop // 2500)              # keep the browser fast
    data, times = data[:, ::stride], times[::stride]
    if scale_uv is None:
        scale_uv = float(np.percentile(data.std(axis=1), 90)) * 4 or 1.0
    names, n_ch, sp = raw.ch_names, len(raw.ch_names), 1.0
    fig = go.Figure()
    for i in range(n_ch):
        fig.add_trace(go.Scatter(x=times, y=-i * sp + data[i] / scale_uv * sp, mode="lines",
                                 line=dict(width=1, color="black"), name=names[i],
                                 showlegend=False,
                                 hovertemplate=names[i] + " %{x:.2f}s<extra></extra>"))
    n_vis = min(n_vis, n_ch)
    yr = lambda s: [-(s + n_vis - 1) * sp - 0.7, -s * sp + 0.7]
    steps = [dict(method="relayout", label=names[s], args=[{"yaxis.range": yr(s)}])
             for s in range(0, n_ch - n_vis + 1)]
    shapes = [dict(type="line", x0=b, x1=b, yref="paper", y0=0, y1=1,
                   line=dict(color="red", width=1, dash="dot"))
              for b in bounds if b <= times[-1]]
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)), height=680,
        margin=dict(l=70, r=20, t=40, b=20), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="black"), dragmode="pan", shapes=shapes,
        xaxis=dict(range=[0, min(window_s, float(times[-1]))],
                   rangeslider=dict(visible=True, thickness=0.08), title="Time (s)"),
        yaxis=dict(range=yr(0), fixedrange=True, tickmode="array",
                   tickvals=[-i * sp for i in range(n_ch)], ticktext=names),
        sliders=[dict(active=0, steps=steps, x=0.0, len=1.0, y=-0.28,
                      currentvalue=dict(prefix="First visible channel: "))])
    return fig


def figure_for_plot(*args, **kwargs):
    """Return a scrollable figure for channels_before / channels_after plots, else None."""
    cands = list(args) + list(kwargs.values())
    plot = next((str(a) for a in cands if isinstance(a, (str, Path))
                 and str(a).lower().endswith((".png", ".jpg", ".jpeg", ".svg"))), None)
    if not plot:
        return None
    name = Path(plot).stem.lower()
    if "channels" not in name:
        return None
    if "after" in name:
        path = _newest(["preprocessed_epo*.fif", "**/preprocessed_epo*.fif"])
        title = "Channels After (preprocessed epochs, stitched; red dotted = epoch boundaries)"
    elif "before" in name:
        path = _newest(["uploaded_datasets/*.edf", "uploaded_datasets/*.bdf", "uploaded_datasets/*.fif"])
        title = "Channels Before (raw recording, unfiltered)"
    else:
        return None
    if not path:
        logger.warning("scroll viewer: no data file found for %s", plot)
        return None
    raw, bounds = _load(path)
    return build_scroll_figure(raw, bounds=bounds, title=title)
