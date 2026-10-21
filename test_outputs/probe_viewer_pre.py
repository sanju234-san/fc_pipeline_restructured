"""Offline probe of the CURRENT (pre-fix) EEG viewer launch behaviour.

Run with:
    .\.venv\Scripts\python.exe test_outputs\probe_viewer_pre.py

It builds a synthetic 32-channel .fif, then constructs EEGViewerWindow in
two scenarios and reports which implementation is used:
  A) real EEG data path supplied (--data equivalent)
  B) only the PNG plot supplied, with no matching run_context_*.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import mne

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fc_pipeline.viz.eeg_viewer import EEGViewerWindow  # noqa: E402


def build_synthetic_fif(path: Path, n_ch: int = 32, sfreq: float = 250.0) -> Path:
    base = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
            "FC1", "FC2", "CP1", "CP2", "FT9", "FT10", "TP9", "TP10",
            "F1", "F2", "C1", "C2"]
    ch_names = base[:n_ch]
    n_samples = int(10.0 * sfreq)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(ch_names), n_samples)) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=[0.0], duration=[10.0], description=["rest"]))
    raw.save(str(path), overwrite=True, verbose=False)
    return path


def describe(tag: str, win: EEGViewerWindow) -> None:
    canvas = win._canvas
    view = win._view
    print(f"\n=== {tag} ===")
    print(f"  _canvas (EEGSignalCanvas): {'PRESENT' if canvas is not None else 'None'}")
    print(f"  _view   (_ZoomableView)  : {'PRESENT (static image fallback)' if view is not None else 'None'}")
    if canvas is not None:
        print(f"  n_channels={canvas.n_channels}  n_visible={canvas.n_visible}  sfreq={canvas._sfreq}")
        print(f"  first visible names: {canvas._ch_names[:12]}")
    eeg = win._eeg_signals
    print(f"  loaded signal object type: {'dict(data=' + str(type(eeg['data']).__name__) + ', n_chan=' + str(eeg['data'].shape[0]) + ')' if eeg else 'None'}")
    print(f"  window title: {win.windowTitle()}")


def main() -> None:
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    out = Path(__file__).resolve().parent
    fif = build_synthetic_fif(out / "probe_raw32.fif", n_ch=32)
    png = out / "channels_before_probe.png"
    # 1x1 PNG placeholder (no run_context json next to it)
    import base64
    png.write_bytes(base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    ))

    # Scenario A: real data supplied
    win_a = EEGViewerWindow(target_path=str(fif), title="Raw Recording", explicit_data_path=str(fif))
    describe("A) real EEG data supplied", win_a)

    # Scenario B: only the PNG supplied (what Chainlit passes when the run has
    # no recorded raw/preprocessed path); no run_context json exists here.
    win_b = EEGViewerWindow(target_path=str(png), title="Channels Before", explicit_data_path=None)
    describe("B) PNG only, no run_context", win_b)

    print("\n[probe] done")


if __name__ == "__main__":
    main()
