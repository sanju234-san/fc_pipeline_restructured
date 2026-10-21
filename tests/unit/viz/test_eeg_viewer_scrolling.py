"""Offscreen-Qt tests for the interactive EEG channel viewer.

These tests exercise the *real* ``EEGViewerWindow`` against a synthetic
32-channel MNE ``.fif``. They assert the two things that the original
scrolling bug got wrong:

1. When real EEG data is available, the window activates the interactive
   ``EEGSignalCanvas`` (per-channel rows) — NOT the static image fallback —
   and its channel navigation genuinely changes which channels are shown
   (mouse wheel, vertical scrollbar, Prev/Next buttons).
2. When no EEG data can be resolved, the window degrades to a pan/zoom
   **static** image and *says so* (title, banner, status bar) instead of
   silently mislabelling itself "Interactive Viewer".

Run::

    pytest tests/unit/viz -v

These tests need PyQt5; the suite skips cleanly if it is not installed. Qt
runs with the ``offscreen`` platform plugin, so no display is required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

import numpy as np  # noqa: E402
import mne  # noqa: E402

from PyQt5.QtCore import QPoint, QPointF, QSize, Qt  # noqa: E402
from PyQt5.QtGui import QResizeEvent, QWheelEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fc_pipeline.viz.eeg_probe import probe_eeg_dataset, resolve_eeg_dataset  # noqa: E402
from fc_pipeline.viz.eeg_viewer import EEGViewerWindow  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def fif32(tmp_path_factory) -> Path:
    """A 32-channel, 250 Hz, 30 s synthetic raw recording (real MNE object).

    30 s (rather than 10 s) so the default 10 s view window is smaller than the
    recording, making horizontal time navigation observable.
    """
    tmp = tmp_path_factory.mktemp("viz_eeg")
    path = tmp / "synth_raw.fif"  # "raw.fif" suffix keeps MNE quiet
    names = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
        "FC1", "FC2", "CP1", "CP2", "FT9", "FT10", "TP9", "TP10",
        "F1", "F2", "C1", "C2",
    ]
    sfreq = 250.0
    n_times = int(30.0 * sfreq)
    rng = np.random.default_rng(7)
    data = rng.standard_normal((len(names), n_times)) * 1e-6
    info = mne.create_info(ch_names=names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=[0.0], duration=[30.0], description=["rest"]))
    raw.save(str(path), overwrite=True, verbose=False)
    return path


@pytest.fixture
def png_only(tmp_path) -> Path:
    """A tiny valid PNG with no accompanying run_context JSON."""
    import base64

    path = tmp_path / "channels_before_run123.png"
    path.write_bytes(
        base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            b"YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
    )
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interactive_window(qapp, fif_path: Path) -> EEGViewerWindow:
    return EEGViewerWindow(
        target_path=str(fif_path),
        title="Channels Before",
        explicit_data_path=str(fif_path),
    )


def _resize_canvas(canvas, qapp, width: int, height: int) -> None:
    """Deterministically deliver a resize of ``width`` x ``height`` to a canvas."""
    canvas.resize(width, height)
    qapp.processEvents()
    ev = QResizeEvent(QSize(width, height), QSize(width, height))
    canvas.resizeEvent(ev)


def _wheel(canvas, notches: int, modifiers: Qt.KeyboardModifiers = Qt.NoModifier) -> None:
    """Deliver a real QWheelEvent (``notches`` * 120 eighths-of-a-degree)."""
    ev = QWheelEvent(
        QPointF(10.0, 10.0),
        QPointF(10.0, 10.0),
        QPoint(0, 0),
        QPoint(0, notches * 120),
        Qt.NoButton,
        modifiers,
        Qt.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(ev)


# ---------------------------------------------------------------------------
# Data resolution / probing
# ---------------------------------------------------------------------------
def test_probe_reads_channels_and_type(fif32):
    info = probe_eeg_dataset(fif32)
    assert info is not None
    assert info["n_channels"] == 32
    assert info["sfreq"] == pytest.approx(250.0)
    assert info["data_type"] == "raw"
    assert info["ch_names"][0] == "Fp1"


def test_probe_of_png_returns_none(png_only):
    assert probe_eeg_dataset(png_only) is None


def test_resolve_prefers_explicit_data(png_only, fif32):
    assert resolve_eeg_dataset(png_only, fif32) == fif32


def test_resolve_returns_none_without_data_or_run_context(png_only):
    assert resolve_eeg_dataset(png_only, None) is None


# ---------------------------------------------------------------------------
# Interactive mode: real data -> channel canvas
# ---------------------------------------------------------------------------
def test_real_data_activates_interactive_canvas(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    assert win.viewer_mode == "interactive"
    assert win._canvas is not None
    assert win._view is None  # NOT the static image fallback
    assert win._canvas.n_channels == 32
    assert "static" not in win.windowTitle().lower()
    status = win.viewer_status()
    assert status["mode"] == "interactive"
    assert status["n_channels"] == 32
    assert status["data_type"] == "raw"


def test_wheel_scrolling_changes_displayed_channels(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    canvas = win._canvas
    canvas.resize(1200, 480)
    canvas._recompute_visible()

    # Force a small window so only a subset of channels fits.
    _resize_canvas(canvas, qapp, 1200, 320)
    assert canvas.n_visible < canvas.n_channels

    start0 = canvas.start_channel
    first_name0 = canvas._ch_names[start0]

    # One notch down -> exactly one channel further.
    _wheel(canvas, -1)
    assert canvas.start_channel == start0 + 1
    assert canvas._ch_names[canvas.start_channel] != first_name0

    # One notch up -> back where we started.
    _wheel(canvas, 1)
    assert canvas.start_channel == start0

    # The status bar must reflect the new channel range.
    msg = win._status.currentMessage()
    assert f"of {canvas.n_channels}" in msg
    assert canvas._ch_names[canvas.start_channel] in msg


def test_wheel_is_clamped_to_channel_range(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    canvas = win._canvas
    _resize_canvas(canvas, qapp, 1200, 320)

    for _ in range(canvas.n_channels + 5):
        _wheel(canvas, -1)
    assert canvas.start_channel == canvas.n_channels - canvas.n_visible

    for _ in range(canvas.n_channels + 5):
        _wheel(canvas, 1)
    assert canvas.start_channel == 0


def test_vertical_scrollbar_is_tied_to_channels(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    canvas = win._canvas
    _resize_canvas(canvas, qapp, 1200, 320)

    sb = win._v_scrollbar
    assert sb.maximum() == canvas.n_channels - canvas.n_visible
    assert sb.pageStep() == canvas.n_visible
    assert sb.minimum() == 0

    # Dragging the scrollbar moves the visible channel window.
    sb.setValue(sb.maximum())
    assert canvas.start_channel == canvas.n_channels - canvas.n_visible

    # ...and the last channel is now on screen.
    end = canvas.start_channel + canvas.n_visible
    assert canvas._ch_names[end - 1] == canvas._ch_names[-1]


def test_wheel_keeps_scrollbar_thumb_in_sync(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    canvas = win._canvas
    _resize_canvas(canvas, qapp, 1200, 320)

    _wheel(canvas, -1)
    assert win._v_scrollbar.value() == canvas.start_channel


def test_prev_next_buttons_page_through_all_channels(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    canvas = win._canvas
    _resize_canvas(canvas, qapp, 1200, 320)

    win._scroll_channels_down()
    assert canvas.start_channel == canvas.n_visible
    win._scroll_channels_up()
    assert canvas.start_channel == 0

    # Page to the very end and confirm the final channel is reachable.
    for _ in range(canvas.n_channels):
        win._scroll_channels_down()
    assert canvas.start_channel == canvas.n_channels - canvas.n_visible

    end = canvas.start_channel + canvas.n_visible
    assert canvas._ch_names[end - 1] == canvas._ch_names[-1]


def test_resize_updates_visible_rows_and_scrollbar(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    canvas = win._canvas

    _resize_canvas(canvas, qapp, 1200, 240)
    small = canvas.n_visible

    _resize_canvas(canvas, qapp, 1200, 900)
    large = canvas.n_visible

    assert large > small
    assert win._v_scrollbar.pageStep() == large
    assert win._v_scrollbar.maximum() == canvas.n_channels - large


def test_horizontal_time_navigation(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    canvas = win._canvas

    assert canvas.start_time == pytest.approx(0.0)
    win._slide_right()
    assert canvas.start_time == pytest.approx(1.0)
    win._slide_left()
    assert canvas.start_time == pytest.approx(0.0)
    # Cannot slide before t=0.
    win._slide_left_large()
    assert canvas.start_time == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Static fallback must be honest
# ---------------------------------------------------------------------------
def test_static_fallback_is_labelled_honestly(qapp, png_only):
    win = EEGViewerWindow(target_path=str(png_only), title="Channels Before")

    assert win.viewer_mode == "static"
    assert win._canvas is None
    assert win._view is not None  # pan/zoom image view

    assert "static" in win.windowTitle().lower()
    status = win.viewer_status()
    assert status["mode"] == "static"
    assert status["reason"]  # a non-empty explanation
    assert status["n_channels"] == 0

    # The status bar must state that channel scrolling is unavailable.
    msg = win._status.currentMessage().lower()
    assert "static image" in msg
    assert "scroll" in msg


def test_interactive_window_reports_no_fallback_reason(qapp, fif32):
    win = _interactive_window(qapp, fif32)
    assert win.viewer_status()["reason"] == ""
