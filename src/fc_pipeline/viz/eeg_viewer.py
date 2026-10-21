"""Interactive PyQt5 EEG plot and channel viewer with zoom, slide, and channel navigation.

This module is standalone: it is launched as a *subprocess* by ``viewer_launcher.py``
so that the Qt event loop never conflicts with Chainlit's asyncio event loop.

Capabilities:
- True vertical scrolling through ALL EEG channels in the loaded dataset
- Display of manageable channel slices (e.g. 12 channels per page)
- Right vertical scrollbar and mouse-wheel scrolling connected directly to EEG channels
- Horizontal time navigation via toolbar buttons (<<, <, >, >>), arrow keys, and scrollbar
- Amplitude gain zoom (Zoom +, Zoom -, Fit, 1:1)
- Left gutter labeling with bad/flagged channels ([BAD] in red) and plan channels (gold)
- Dynamic channel count handling (works with any number of channels)
- Seamless fallback to zoomable static image view for heatmaps or non-EEG plots
- Light theme (white canvas, dark text) matching the app's light diagnostic plots

Usage (as __main__)::

    python -m fc_pipeline.viz.eeg_viewer /path/to/channels_before.png "Channels Before"
    python -m fc_pipeline.viz.eeg_viewer /path/to/raw_dataset.edf "Raw Recording"
    python -m fc_pipeline.viz.eeg_viewer /path/to/plot.png "Channels Before" --data /path/to/data.edf

"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard: PyQt5 must be importable
# ---------------------------------------------------------------------------
try:
    from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
    from PyQt5.QtGui import (
        QPixmap,
        QColor,
        QPalette,
        QWheelEvent,
        QPainter,
        QPainterPath,
        QFont,
        QPen,
        QBrush,
    )
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QGraphicsScene,
        QGraphicsView,
        QGraphicsPixmapItem,
        QToolBar,
        QLabel,
        QStatusBar,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QSizePolicy,
        QScrollBar,
    )
except ImportError as exc:  # pragma: no cover
    print(
        f"[EEG Viewer] PyQt5 is not installed: {exc}\n"
        "Install it with:  pip install PyQt5>=5.15",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants & Theming
# ---------------------------------------------------------------------------
_ZOOM_STEP = 1.20
_ZOOM_MIN = 0.10
_ZOOM_MAX = 20.0
# --- Light theme palette ----------------------------------------------------
_BG = "#ffffff"           # canvas / window background
_PANEL_BG = "#eef1f5"     # gutters, toolbar, status bar, scrollbar track
_ACCENT = "#c0392b"       # buttons / highlights (dark red, legible on white)
_TEXT = "#1a1a2e"         # primary text
_GRID = "#e0e4ea"         # dotted grid inside channel lanes
_TICK = "#666666"         # time-axis tick text
_SEP = "#c9ced6"          # lane separator lines
_TRACE = "#0e7c86"        # normal channel trace (dark teal)
_TRACE_PLAN = "#b9770e"   # plan-selected channel (dark amber)
_BUTTON_STYLE = (
    f"QPushButton {{"
    f"  background-color: {_PANEL_BG}; color: {_TEXT};"
    f"  border: 1px solid {_ACCENT}; border-radius: 4px;"
    f"  padding: 4px 10px; font-size: 12px;"
    f"}}"
    f"QPushButton:hover {{ background-color: {_ACCENT}; color: #ffffff; }}"
)


# ---------------------------------------------------------------------------
# EEG Data Resolution & Loading
# ---------------------------------------------------------------------------

# Qt-free helpers live in ``eeg_probe`` so the launcher/preflight can use them
# without importing the Qt widget stack.  Re-exported here for compatibility
# with ``python -m fc_pipeline.viz.eeg_viewer`` and existing call sites.
from fc_pipeline.viz.eeg_probe import (  # noqa: E402  (kept after the Qt guard)
    probe_eeg_dataset,
    resolve_eeg_dataset,
)


def load_eeg_signals(
    data_path: Path,
    plot_path: Optional[Path] = None,
) -> Optional[Dict]:
    """Load continuous EEG signal arrays and montage metadata using MNE."""
    try:
        import mne
        import numpy as np
    except ImportError:
        logger.warning("MNE or NumPy not available to load raw EEG signals.")
        return None

    data_path = Path(data_path)
    if not data_path.exists():
        return None

    try:
        data: np.ndarray
        ch_names: List[str]
        sfreq: float
        bads: Set[str] = set()
        analysis_channels: Set[str] = set()
        data_type: str = "raw"

        # Attempt to extract plan metadata from run context if available
        if plot_path:
            m = re.search(r"channels_(?:before|after)_(.+)\.png", Path(plot_path).name)
            if m:
                run_id = m.group(1)
                for cand in [plot_path.parent / f"run_context_{run_id}.json", plot_path.parent.parent / f"run_context_{run_id}.json"]:
                    if cand.exists():
                        try:
                            with open(cand, "r", encoding="utf-8") as f:
                                ctx = json.load(f)
                            if ctx.get("bad_channels_dropped"):
                                bads.update(ctx["bad_channels_dropped"])
                            plan = ctx.get("plan") or {}
                            if plan.get("channels"):
                                analysis_channels.update(plan["channels"])
                        except Exception:
                            pass

        suffix = data_path.suffix.lower()
        if suffix in (".fif",):
            # Try reading as Epochs first, then Raw
            try:
                epochs = mne.read_epochs(data_path, preload=True, verbose=False)
                ch_names = list(epochs.ch_names)
                sfreq = float(epochs.info["sfreq"])
                ep_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
                # Concatenate along time for a continuous view
                n_epochs, n_ch, n_times = ep_data.shape
                data = ep_data.transpose(1, 0, 2).reshape(n_ch, -1)
                bads.update(epochs.info.get("bads", []))
                data_type = "epochs"
            except Exception:
                raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
                ch_names = list(raw.ch_names)
                sfreq = float(raw.info["sfreq"])
                data = raw.get_data()
                bads.update(raw.info.get("bads", []))
                data_type = "raw"
        else:
            raw = mne.io.read_raw(data_path, preload=True, verbose=False)
            ch_names = list(raw.ch_names)
            sfreq = float(raw.info["sfreq"])
            data = raw.get_data()
            bads.update(raw.info.get("bads", []))

        n_channels, n_samples = data.shape
        duration = float(n_samples / sfreq) if sfreq > 0 else 0.0

        # Compute robust baseline scaling (volts per deflection)
        good_rows = [i for i, ch in enumerate(ch_names) if ch not in bads]
        if good_rows:
            stds = np.std(data[good_rows], axis=1)
            peak_std = float(np.max(stds)) if stds.size > 0 else 20e-6
        else:
            peak_std = 20e-6
        base_scale = 3.0 * peak_std if peak_std > 0 else 50e-6

        return {
            "data": data,
            "ch_names": ch_names,
            "sfreq": sfreq,
            "n_samples": n_samples,
            "duration": duration,
            "bads": bads,
            "analysis_channels": analysis_channels,
            "base_scale": base_scale,
            "data_type": data_type,
        }

    except Exception as exc:
        logger.exception("Failed loading EEG signals from %s: %s", data_path, exc)
        return None


# ---------------------------------------------------------------------------
# Interactive EEG Signal Canvas
# ---------------------------------------------------------------------------

class EEGSignalCanvas(QWidget):
    """High-performance custom QWidget for multi-channel EEG signal visualization.

    Features:
    - Vertical channel scrolling through all EEG channels in loaded dataset
    - Horizontal timeline sliding
    - Mouse-wheel channel navigation
    - Click and drag pan
    - Left channel label gutter with Bad and Plan indicators
    - Responsive dark-themed waveforms and grid
    """

    channel_scrolled = pyqtSignal(int)
    time_scrolled = pyqtSignal(float)
    view_updated = pyqtSignal()
    visible_changed = pyqtSignal(int)

    def __init__(self, eeg_data: Dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._data: np.ndarray = eeg_data["data"]
        self._ch_names: List[str] = eeg_data["ch_names"]
        self._sfreq: float = eeg_data["sfreq"]
        self._n_samples: int = eeg_data["n_samples"]
        self._duration: float = eeg_data["duration"]
        self._bads: Set[str] = eeg_data["bads"]
        self._analysis_channels: Set[str] = eeg_data["analysis_channels"]
        self._base_scale: float = eeg_data["base_scale"]

        self._n_channels: int = len(self._ch_names)
        # Responsive visible-channel count: aim for ~56 px per channel row and
        # recompute whenever the widget is resized (see resizeEvent).  This
        # replaces the old hard-coded 12 so the viewer stays usable at any
        # window size while still showing a manageable number of channels.
        self._min_visible: int = 4
        self._target_row_height: int = 56
        self._n_visible: int = 12
        self._start_channel: int = 0
        self._start_time: float = 0.0
        self._window_duration: float = 10.0
        self._amplitude_scale: float = 1.0

        self._dragging: bool = False
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_start_time: float = 0.0
        self._drag_start_ch: int = 0

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setBackgroundRole(QPalette.Window)
        self.setMinimumHeight(160)
        self._recompute_visible()

    def _recompute_visible(self) -> None:
        """Pick how many channel rows fit the current widget height.

        Mirrors the layout maths in ``paintEvent`` (gutter/time-bar margins) so
        the row height stays readable; clamped to ``[min_visible, n_channels]``.
        Emits ``visible_changed`` when the value actually changes.
        """
        plot_h = max(10, self.height() - 24 - 12)
        rows = max(self._min_visible, int(plot_h / self._target_row_height))
        new_visible = max(self._min_visible, min(rows, max(1, self._n_channels)))
        if new_visible != self._n_visible:
            self._n_visible = new_visible
            self.visible_changed.emit(self._n_visible)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._recompute_visible()
        # A smaller viewport can push the current start past the new maximum.
        max_start = max(0, self._n_channels - self._n_visible)
        if self._start_channel > max_start:
            self._start_channel = max_start
            self.channel_scrolled.emit(self._start_channel)
        super().resizeEvent(event)


    # ------------------------------------------------------------------ properties
    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def n_visible(self) -> int:
        return self._n_visible

    @property
    def start_channel(self) -> int:
        return self._start_channel

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def window_duration(self) -> float:
        return self._window_duration

    @property
    def total_duration(self) -> float:
        return self._duration

    @property
    def amplitude_scale(self) -> float:
        return self._amplitude_scale

    # ------------------------------------------------------------------ navigation
    def set_start_channel(self, ch_idx: int) -> None:
        max_start = max(0, self._n_channels - self._n_visible)
        new_ch = max(0, min(ch_idx, max_start))
        if new_ch != self._start_channel:
            self._start_channel = new_ch
            self.channel_scrolled.emit(self._start_channel)
            self.view_updated.emit()
            self.update()

    def scroll_channels(self, delta: int) -> None:
        self.set_start_channel(self._start_channel + delta)

    def set_start_time(self, t: float) -> None:
        max_time = max(0.0, self._duration - self._window_duration)
        new_t = max(0.0, min(t, max_time))
        if abs(new_t - self._start_time) > 1e-4:
            self._start_time = new_t
            self.time_scrolled.emit(self._start_time)
            self.view_updated.emit()
            self.update()

    def step_time(self, delta_t: float) -> None:
        self.set_start_time(self._start_time + delta_t)

    def zoom_amplitude(self, factor: float) -> None:
        self._amplitude_scale = max(0.1, min(self._amplitude_scale * factor, 20.0))
        self.view_updated.emit()
        self.update()

    def set_amplitude_scale(self, scale: float) -> None:
        self._amplitude_scale = scale
        self.view_updated.emit()
        self.update()

    def fit_auto_scale(self) -> None:
        self._amplitude_scale = 1.0
        self._window_duration = min(10.0, self._duration)
        self.view_updated.emit()
        self.update()

    def reset_view(self) -> None:
        self.set_start_channel(0)
        self.set_start_time(0.0)
        self.set_amplitude_scale(1.0)
        self._window_duration = min(10.0, self._duration)
        self.view_updated.emit()
        self.update()

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:
        import numpy as np

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        gutter_w = 88
        time_bar_h = 24
        plot_w = max(10, w - gutter_w - 14)
        plot_h = max(10, h - time_bar_h - 12)

        # 1. Backgrounds
        painter.fillRect(0, 0, w, h, QColor(_BG))
        painter.fillRect(0, 0, gutter_w, h - time_bar_h, QColor(_PANEL_BG))
        painter.fillRect(0, h - time_bar_h, w, time_bar_h, QColor(_PANEL_BG))

        # Gutter divider
        painter.setPen(QPen(QColor(_ACCENT), 1))
        painter.drawLine(gutter_w, 0, gutter_w, h - time_bar_h)

        if self._n_channels == 0:
            painter.setPen(QColor(_TEXT))
            painter.drawText(w // 2 - 80, h // 2, "No EEG channels loaded.")
            return

        visible_count = min(self._n_visible, self._n_channels - self._start_channel)
        if visible_count <= 0:
            return

        ch_height = plot_h / max(1, self._n_visible)

        # 2. Time grid vertical lines
        t_start = self._start_time
        t_end = self._start_time + self._window_duration
        if self._window_duration > 0:
            grid_step = 1.0 if self._window_duration <= 15.0 else 2.0
            first_grid = int(t_start / grid_step) * grid_step
            cur_t = first_grid
            painter.setFont(QFont("Segoe UI", 9))
            while cur_t <= t_end + 1e-4:
                if cur_t >= t_start:
                    gx = gutter_w + ((cur_t - t_start) / self._window_duration) * plot_w
                    painter.setPen(QPen(QColor(_GRID), 1, Qt.DotLine))
                    painter.drawLine(int(gx), 0, int(gx), int(h - time_bar_h))
                    painter.setPen(QColor(_TICK))
                    painter.drawText(int(gx) - 16, int(h - 6), f"{cur_t:.1f}s")
                cur_t += grid_step

        # 3. Waveforms & Channel Labels
        s0 = max(0, int(t_start * self._sfreq))
        s1 = min(self._n_samples, int(t_end * self._sfreq))
        n_pts = s1 - s0

        for i in range(visible_count):
            ch_idx = self._start_channel + i
            ch_name = self._ch_names[ch_idx]
            is_bad = ch_name in self._bads
            is_selected = ch_name in self._analysis_channels
            y_base = 10 + (i + 0.5) * ch_height

            # Row divider
            painter.setPen(QPen(QColor(_SEP), 1, Qt.SolidLine))
            painter.drawLine(gutter_w, int(y_base + ch_height * 0.5), w, int(y_base + ch_height * 0.5))

            # Channel label in gutter
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold if is_selected else QFont.Normal))
            if is_bad:
                painter.setPen(QColor(_ACCENT))
                label_txt = f"{ch_name} ✕"
            elif is_selected:
                painter.setPen(QColor(_TRACE_PLAN))
                label_txt = f"★ {ch_name}"
            else:
                painter.setPen(QColor(_TEXT))
                label_txt = ch_name

            painter.drawText(8, int(y_base + 4), label_txt)

            # Signal waveform
            if n_pts > 1:
                raw_trace = self._data[ch_idx, s0:s1]
                # Downsample if very dense for smooth 60fps rendering
                step = max(1, n_pts // int(plot_w * 2)) if n_pts > plot_w * 2 else 1
                trace_sub = raw_trace[::step]
                step_n = len(trace_sub)

                x_coords = gutter_w + (np.linspace(0, 1, step_n) * plot_w)
                deflection = (trace_sub / self._base_scale) * self._amplitude_scale * (ch_height * 0.42)
                y_coords = y_base - deflection
                y_coords = np.clip(y_coords, y_base - ch_height * 0.92, y_base + ch_height * 0.92)

                path = QPainterPath()
                path.moveTo(float(x_coords[0]), float(y_coords[0]))
                for k in range(1, step_n):
                    path.lineTo(float(x_coords[k]), float(y_coords[k]))

                if is_bad:
                    pen = QPen(QColor(_ACCENT), 1.2)
                elif is_selected:
                    pen = QPen(QColor(_TRACE_PLAN), 1.3)
                else:
                    pen = QPen(QColor(_TRACE), 1.2)

                painter.setPen(pen)
                painter.drawPath(path)

    # ------------------------------------------------------------------ events
    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            # Ctrl + Wheel = amplitude scale
            self.zoom_amplitude(1.15 if delta > 0 else (1.0 / 1.15))
        elif (mods & Qt.ShiftModifier) or (mods & Qt.AltModifier):
            # Shift / Alt + Wheel = slide time
            self.step_time(-1.0 if delta > 0 else 1.0)
        else:
            # Plain wheel = scroll one channel per notch (fine-grained, so all
            # channels are reachable).  Use the Prev/Next buttons or PageUp/
            # PageDown for full-page jumps.
            self.scroll_channels(-1 if delta > 0 else 1)
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.pos()
            self._drag_start_time = self._start_time
            self._drag_start_ch = self._start_channel
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._drag_start_pos:
            dx = event.pos().x() - self._drag_start_pos.x()
            dy = event.pos().y() - self._drag_start_pos.y()
            # Horizontal drag = slide time
            plot_w = max(10, self.width() - 100)
            dt = -(dx / plot_w) * self._window_duration
            self.set_start_time(self._drag_start_time + dt)

            # Vertical drag = scroll channels
            ch_h = max(10, (self.height() - 40) / max(1, self._n_visible))
            d_ch = int(-dy / ch_h)
            self.set_start_channel(self._drag_start_ch + d_ch)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False
        event.accept()


# ---------------------------------------------------------------------------
# Fallback QGraphicsView for Static Images (Heatmaps, Networks)
# ---------------------------------------------------------------------------

class _ZoomableView(QGraphicsView):
    """QGraphicsView with scroll-wheel zoom and keyboard pan for static plots."""

    def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None) -> None:
        super().__init__(scene, parent)
        self._scale = 1.0
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(0x01)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QColor(_BG))

    def zoom_in(self) -> None:
        self._apply_zoom(_ZOOM_STEP)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / _ZOOM_STEP)

    def zoom_fit(self) -> None:
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        t = self.transform()
        self._scale = t.m11()

    def zoom_100(self) -> None:
        self.resetTransform()
        self._scale = 1.0

    def _apply_zoom(self, factor: float) -> None:
        new_scale = self._scale * factor
        if new_scale < _ZOOM_MIN or new_scale > _ZOOM_MAX:
            return
        self._scale = new_scale
        self.scale(factor, factor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self._apply_zoom(_ZOOM_STEP)
        else:
            self._apply_zoom(1.0 / _ZOOM_STEP)
        event.accept()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class EEGViewerWindow(QMainWindow):
    """Top-level window with full channel vertical scrolling & time sliding."""

    def __init__(
        self,
        target_path: str | Path,
        title: str = "EEG Viewer",
        explicit_data_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self._target_path = Path(target_path)
        self._title = title or self._target_path.stem
        self.resize(1280, 780)
        self.setMinimumSize(820, 420)

        # Attempt to resolve and load continuous EEG signals
        self._resolved_data = resolve_eeg_dataset(self._target_path, explicit_data_path)
        self._eeg_signals = (
            load_eeg_signals(self._resolved_data, self._target_path)
            if self._resolved_data
            else None
        )

        # Honest mode reporting: "interactive" means the real EEG channel canvas
        # is active; "static" means we degraded to a pan/zoom image view.
        if self._eeg_signals is not None:
            self.viewer_mode = "interactive"
            self.fallback_reason = ""
        else:
            self.viewer_mode = "static"
            self.fallback_reason = self._fallback_reason()

        self._canvas: Optional[EEGSignalCanvas] = None
        self._view: Optional[_ZoomableView] = None
        self._pixmap_item = None

        self._setup_style()
        self._build_ui()

        if self._canvas is not None:
            self._sync_scrollbars()
            self._canvas.setFocus()
        elif self._view is not None:
            self._load_fallback_image()
            self._view.zoom_fit()

        # The window title must never claim to be the interactive channel viewer
        # when only a static image could be shown (AGENTS/HITL honesty rule).
        if self.viewer_mode == "interactive":
            self.setWindowTitle(f"EEG Viewer - {self._title}")
        else:
            self.setWindowTitle(
                f"EEG Viewer (static image) - {self._title}"
            )
            logger.warning(
                "EEG viewer degraded to STATIC image for %s: %s",
                self._target_path.name, self.fallback_reason,
            )

        self._update_status()

    def _fallback_reason(self) -> str:
        """Human-readable reason the interactive channel canvas is unavailable."""
        if self._resolved_data is None:
            return (
                "no EEG data file could be resolved for this plot "
                "(no --data path supplied and no run_context_*.json next to the plot)"
            )
        return (
            f"EEG file {self._resolved_data} was found but could not be loaded "
            "as MNE Raw/Epochs"
        )

    def viewer_status(self) -> Dict:
        """Machine-readable status of this viewer instance (for launcher/tests)."""
        return {
            "mode": self.viewer_mode,
            "reason": self.fallback_reason,
            "data_path": str(self._resolved_data) if self._resolved_data else None,
            "data_type": (self._eeg_signals or {}).get("data_type") if self._eeg_signals else None,
            "n_channels": self._canvas.n_channels if self._canvas is not None else 0,
            "n_visible": self._canvas.n_visible if self._canvas is not None else 0,
        }


    # ------------------------------------------------------------------ style
    def _setup_style(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(_BG))
        palette.setColor(QPalette.WindowText, QColor(_TEXT))
        palette.setColor(QPalette.Base, QColor(_PANEL_BG))
        palette.setColor(QPalette.Text, QColor(_TEXT))
        palette.setColor(QPalette.Button, QColor(_PANEL_BG))
        palette.setColor(QPalette.ButtonText, QColor(_TEXT))
        self.setPalette(palette)
        self.setStyleSheet(
            f"QMainWindow, QWidget {{ background-color: {_BG}; color: {_TEXT};"
            f"  font-family: 'Segoe UI', Arial, sans-serif; }}"
            f"QToolBar {{ background-color: {_PANEL_BG}; border-bottom: 1px solid {_ACCENT};"
            f"  spacing: 6px; padding: 4px 8px; }}"
            f"QStatusBar {{ background-color: {_PANEL_BG}; color: {_TEXT}; font-size: 11px; }}"
            f"QScrollBar:horizontal {{ height: 14px; background: {_PANEL_BG}; margin: 0px; }}"
            f"QScrollBar::handle:horizontal {{ background: {_ACCENT}; border-radius: 6px; min-width: 30px; }}"
            f"QScrollBar:vertical {{ width: 14px; background: {_PANEL_BG}; margin: 0px; }}"
            f"QScrollBar::handle:vertical {{ background: {_ACCENT}; border-radius: 6px; min-height: 30px; }}"
        )

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        def _btn(label: str, tip: str, slot, *, big: bool = False) -> QPushButton:
            b = QPushButton(label)
            b.setToolTip(tip)
            style = _BUTTON_STYLE
            if big:
                style = (
                    f"QPushButton {{"
                    f"  background-color: {_ACCENT}; color: #ffffff;"
                    f"  border: 1px solid {_ACCENT}; border-radius: 4px;"
                    f"  padding: 5px 14px; font-size: 13px; font-weight: bold;"
                    f"}}"
                    f"QPushButton:hover {{ background-color: #96281b; }}"
                )
            b.setStyleSheet(style)
            b.clicked.connect(slot)
            return b

        # ── Amplitude zoom ─────────────────────────────────────────────
        toolbar.addWidget(_btn("Zoom +", "Increase amplitude gain / zoom in (+)", self._zoom_in))
        toolbar.addWidget(_btn("Zoom -", "Decrease amplitude gain / zoom out (-)", self._zoom_out))
        toolbar.addWidget(_btn("Fit", "Auto-scale gain (F)", self._zoom_fit))
        toolbar.addWidget(_btn("1:1", "Reset to 100% gain (0)", self._zoom_100))
        toolbar.addSeparator()

        # ── Channel scroll buttons (most important — shown prominently) ─
        if self._eeg_signals is not None:
            toolbar.addWidget(_btn(
                "⬆  Prev Channels",
                "Scroll UP to previous channel group  (Mouse Wheel Up / ↑ Arrow)",
                self._scroll_channels_up,
                big=True,
            ))
            self._ch_pos_label = QLabel("")
            self._ch_pos_label.setStyleSheet(
                f"color: {_TEXT}; font-size: 12px; font-weight: bold;"
                f" padding: 0 10px; border: 1px solid {_SEP}; border-radius: 4px;"
            )
            toolbar.addWidget(self._ch_pos_label)
            toolbar.addWidget(_btn(
                "⬇  Next Channels",
                "Scroll DOWN to next channel group  (Mouse Wheel Down / ↓ Arrow)",
                self._scroll_channels_down,
                big=True,
            ))
            toolbar.addSeparator()
        else:
            self._ch_pos_label = None  # type: ignore[assignment]

        # ── Time navigation ────────────────────────────────────────────
        toolbar.addWidget(_btn("<<", "Jump 5s earlier", self._slide_left_large))
        toolbar.addWidget(_btn("<", "Step 1s earlier (Left Arrow)", self._slide_left))
        toolbar.addWidget(_btn(">", "Step 1s later (Right Arrow)", self._slide_right))
        toolbar.addWidget(_btn(">>", "Jump 5s later", self._slide_right_large))
        toolbar.addSeparator()
        toolbar.addWidget(_btn("Reset", "Reset to Channel 1, Time 0.0s", self._reset_view))

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        self._zoom_label = QLabel("Gain: 100%")
        self._zoom_label.setStyleSheet(
            f"color: {_ACCENT}; font-weight: bold; font-size: 13px; padding-right: 12px;"
        )
        toolbar.addWidget(self._zoom_label)

        if self._eeg_signals is not None:
            # Main interactive EEG plotting layout
            mid_layout = QHBoxLayout()
            mid_layout.setContentsMargins(0, 0, 0, 0)
            mid_layout.setSpacing(0)

            self._canvas = EEGSignalCanvas(self._eeg_signals)
            mid_layout.addWidget(self._canvas, 1)

            # ── Vertical scrollbar (wider + more visible) ───────────────
            self._v_scrollbar = QScrollBar(Qt.Vertical)
            self._v_scrollbar.setRange(0, max(0, self._canvas.n_channels - self._canvas.n_visible))
            self._v_scrollbar.setPageStep(self._canvas.n_visible)
            self._v_scrollbar.setSingleStep(1)
            self._v_scrollbar.setFixedWidth(22)
            self._v_scrollbar.valueChanged.connect(self._on_v_scroll)
            mid_layout.addWidget(self._v_scrollbar, 0)

            root_layout.addLayout(mid_layout, 1)

            # ── Horizontal scrollbar ────────────────────────────────────
            self._h_scrollbar = QScrollBar(Qt.Horizontal)
            max_h = max(0, int((self._canvas.total_duration - self._canvas.window_duration) * 10))
            self._h_scrollbar.setRange(0, max_h)
            self._h_scrollbar.setPageStep(50)
            self._h_scrollbar.setSingleStep(10)
            self._h_scrollbar.valueChanged.connect(self._on_h_scroll)
            root_layout.addWidget(self._h_scrollbar, 0)

            # Sync canvas signals
            self._canvas.channel_scrolled.connect(self._sync_v_scrollbar)
            self._canvas.time_scrolled.connect(self._sync_h_scrollbar)
            self._canvas.view_updated.connect(self._update_status)
            # The responsive row count changes on resize: keep the scrollbar
            # range/page-step and the position badge in step with the canvas.
            self._canvas.visible_changed.connect(self._on_visible_changed)

        else:
            # Fallback static image view
            self._scene = QGraphicsScene()
            self._view = _ZoomableView(self._scene)
            root_layout.addWidget(self._view, 1)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

    # ------------------------------------------------------------------ scroll sync
    def _sync_scrollbars(self) -> None:
        if self._canvas is None or not hasattr(self, "_v_scrollbar"):
            return
        self._v_scrollbar.blockSignals(True)
        self._v_scrollbar.setRange(0, max(0, self._canvas.n_channels - self._canvas.n_visible))
        self._v_scrollbar.setPageStep(self._canvas.n_visible)
        self._v_scrollbar.setValue(self._canvas.start_channel)
        self._v_scrollbar.blockSignals(False)

        self._h_scrollbar.blockSignals(True)
        max_h = max(0, int((self._canvas.total_duration - self._canvas.window_duration) * 10))
        self._h_scrollbar.setRange(0, max_h)
        self._h_scrollbar.setValue(int(self._canvas.start_time * 10))
        self._h_scrollbar.blockSignals(False)

    def _on_visible_changed(self, _n_visible: int) -> None:
        """Canvas row count changed (window resize): resync range + status."""
        self._sync_scrollbars()
        self._update_status()

    def _on_v_scroll(self, val: int) -> None:
        if self._canvas is not None:
            self._canvas.set_start_channel(val)

    def _sync_v_scrollbar(self, val: int) -> None:
        if hasattr(self, "_v_scrollbar"):
            self._v_scrollbar.blockSignals(True)
            self._v_scrollbar.setValue(val)
            self._v_scrollbar.blockSignals(False)

    def _on_h_scroll(self, val: int) -> None:
        if self._canvas is not None:
            self._canvas.set_start_time(val / 10.0)

    def _sync_h_scrollbar(self, t: float) -> None:
        if hasattr(self, "_h_scrollbar"):
            self._h_scrollbar.blockSignals(True)
            self._h_scrollbar.setValue(int(t * 10))
            self._h_scrollbar.blockSignals(False)

    # ------------------------------------------------------------------ fallback
    def _load_fallback_image(self) -> None:
        if not self._target_path.exists():
            self._scene.addText(f"File not found:\n{self._target_path}")
            return
        pixmap = QPixmap(str(self._target_path))
        if pixmap.isNull():
            self._scene.addText("Could not load image.")
            return
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        # Make it unmistakable that this is NOT the interactive channel viewer.
        banner = self._scene.addText(
            "STATIC IMAGE — channel scrolling is unavailable here.\n"
            f"Reason: {self.fallback_reason}"
        )
        banner.setDefaultTextColor(QColor(_ACCENT))
        banner.setFont(QFont("Segoe UI", 9, QFont.Bold))
        banner.setPos(8, 4)

    # ------------------------------------------------------------------ slots
    def _scroll_channels_up(self) -> None:
        """Scroll backward (up) by one full page of channels."""
        if self._canvas is not None:
            self._canvas.scroll_channels(-self._canvas.n_visible)
            self._sync_scrollbars()
            self._update_status()

    def _scroll_channels_down(self) -> None:
        """Scroll forward (down) by one full page of channels."""
        if self._canvas is not None:
            self._canvas.scroll_channels(self._canvas.n_visible)
            self._sync_scrollbars()
            self._update_status()

    def _zoom_in(self) -> None:
        if self._canvas is not None:
            self._canvas.zoom_amplitude(1.2)
        elif self._view is not None:
            self._view.zoom_in()
        self._update_status()

    def _zoom_out(self) -> None:
        if self._canvas is not None:
            self._canvas.zoom_amplitude(1.0 / 1.2)
        elif self._view is not None:
            self._view.zoom_out()
        self._update_status()

    def _zoom_fit(self) -> None:
        if self._canvas is not None:
            self._canvas.fit_auto_scale()
        elif self._view is not None:
            self._view.zoom_fit()
        self._update_status()

    def _zoom_100(self) -> None:
        if self._canvas is not None:
            self._canvas.set_amplitude_scale(1.0)
        elif self._view is not None:
            self._view.zoom_100()
        self._update_status()

    def _reset_view(self) -> None:
        if self._canvas is not None:
            self._canvas.reset_view()
            self._sync_scrollbars()
        elif self._view is not None:
            self._view.zoom_fit()
        self._update_status()

    def _slide_left(self) -> None:
        if self._canvas is not None:
            self._canvas.step_time(-1.0)
        elif self._view is not None:
            hbar = self._view.horizontalScrollBar()
            hbar.setValue(hbar.value() - 80)

    def _slide_right(self) -> None:
        if self._canvas is not None:
            self._canvas.step_time(1.0)
        elif self._view is not None:
            hbar = self._view.horizontalScrollBar()
            hbar.setValue(hbar.value() + 80)

    def _slide_left_large(self) -> None:
        if self._canvas is not None:
            self._canvas.step_time(-5.0)
        elif self._view is not None:
            hbar = self._view.horizontalScrollBar()
            hbar.setValue(hbar.value() - 320)

    def _slide_right_large(self) -> None:
        if self._canvas is not None:
            self._canvas.step_time(5.0)
        elif self._view is not None:
            hbar = self._view.horizontalScrollBar()
            hbar.setValue(hbar.value() + 320)

    def _update_status(self) -> None:
        if self._canvas is not None:
            c = self._canvas
            start = c.start_channel
            end = min(c.n_channels, start + c.n_visible)
            ch_names = c._ch_names
            ch_sub = f"{ch_names[start]} … {ch_names[end - 1]}" if end > start else ""
            t0 = c.start_time
            t1 = min(c.total_duration, t0 + c.window_duration)
            gain_pct = int(c.amplitude_scale * 100)
            self._zoom_label.setText(f"Gain: {gain_pct}%")
            # Update the channel position badge
            if self._ch_pos_label is not None:
                self._ch_pos_label.setText(f" Ch {start + 1}–{end}  /  {c.n_channels} ")
            self._status.showMessage(
                f"Channels {start + 1}–{end} of {c.n_channels} ({ch_sub})  |  "
                f"Time: {t0:.1f}s – {t1:.1f}s of {c.total_duration:.1f}s ({c._sfreq:g} Hz)  |  "
                "Mouse Wheel = scroll channels  |  Shift+Wheel = slide time  |  "
                "⬆⬇ buttons or drag right scrollbar = navigate channels"
            )
        elif self._view is not None:
            scale = self._view._scale
            pct = int(scale * 100)
            self._zoom_label.setText(f"{pct}%")
            self._status.showMessage(
                f"STATIC IMAGE — channel scrolling unavailable. "
                f"{self.fallback_reason}  |  Zoom: {pct}%  |  "
                "Scroll wheel = zoom  |  Drag = pan"
            )

    # ------------------------------------------------------------------ key events
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if self._canvas is not None:
            if key == Qt.Key_Up:
                self._canvas.scroll_channels(-1)
            elif key == Qt.Key_Down:
                self._canvas.scroll_channels(1)
            elif key == Qt.Key_PageUp:
                self._canvas.scroll_channels(-self._canvas.n_visible)
            elif key == Qt.Key_PageDown:
                self._canvas.scroll_channels(self._canvas.n_visible)
            elif key == Qt.Key_Left:
                self._slide_left()
            elif key == Qt.Key_Right:
                self._slide_right()
            elif key in (Qt.Key_Plus, Qt.Key_Equal):
                self._zoom_in()
            elif key == Qt.Key_Minus:
                self._zoom_out()
            elif key == Qt.Key_F:
                self._zoom_fit()
            elif key == Qt.Key_0:
                self._zoom_100()
            elif key == Qt.Key_Home:
                self._reset_view()
            else:
                super().keyPressEvent(event)
        else:
            if key in (Qt.Key_Plus, Qt.Key_Equal):
                self._zoom_in()
            elif key == Qt.Key_Minus:
                self._zoom_out()
            elif key == Qt.Key_0:
                self._zoom_100()
            elif key == Qt.Key_F:
                self._zoom_fit()
            elif key == Qt.Key_Left:
                self._slide_left()
            elif key == Qt.Key_Right:
                self._slide_right()
            else:
                super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="PyQt5 Interactive EEG Viewer")
    parser.add_argument("target_path", help="Path to plot PNG or EEG data file")
    parser.add_argument("title", nargs="?", default="", help="Window title")
    parser.add_argument("--data", dest="data_path", default=None, help="Explicit path to raw/preprocessed EEG data file")
    parser.add_argument(
        "--log-file",
        dest="log_file",
        default=None,
        help="Write diagnostics (including any static-image fallback) to this file.",
    )
    parser.add_argument(
        "--status-file",
        dest="status_file",
        default=None,
        help="Write a JSON status file (mode / n_channels / data_type / reason) on startup.",
    )
    args = parser.parse_args()

    title = args.title or Path(args.target_path).stem

    # Diagnostics: never discard them — a silent failure or fallback is exactly
    # what made the scrolling bug invisible before.
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    app = QApplication(sys.argv)
    app.setApplicationName("EEG Viewer")

    # Publish a "loading" stub first so the launcher can distinguish "still
    # loading the recording" from "viewer failed to start at all".
    if args.status_file:
        try:
            _sp = Path(args.status_file)
            _sp.parent.mkdir(parents=True, exist_ok=True)
            _sp.write_text(json.dumps({"mode": "loading"}), encoding="utf-8")
        except Exception:
            pass

    window = EEGViewerWindow(
        target_path=args.target_path,
        title=title,
        explicit_data_path=args.data_path,
    )
    status = window.viewer_status()
    logger.info("EEG viewer started: %s", status)

    if args.status_file:
        try:
            status_path = Path(args.status_file)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not write status file %s: %s", args.status_file, exc)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
