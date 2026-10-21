"""End-to-end verification of the launcher -> subprocess -> status-file contract.

This exercises the *real* code path used by the Chainlit "Open Interactive
Viewer" button: ``viewer_launcher.launch_eeg_viewer_sync`` spawns the PyQt5
viewer as a detached subprocess, the viewer writes a JSON status file, and the
launcher reads the authoritative mode back.

Two scenarios are checked:
  A) real EEG data supplied      -> mode == "interactive", n_channels == 32
  B) PNG only (no EEG data)      -> mode == "static" with a non-empty reason

The spawned viewer processes are terminated at the end so nothing is left
running.  Qt runs offscreen; no display is needed.

Run:
    .\.venv\Scripts\python.exe test_outputs\verify_viewer_launch.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import mne

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fc_pipeline.viz.viewer_launcher import launch_eeg_viewer_sync  # noqa: E402

OUT = Path(__file__).resolve().parent


def _build_fif(path: Path, n_ch: int = 32, sfreq: float = 250.0) -> Path:
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
             "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
             "FC1", "FC2", "CP1", "CP2", "FT9", "FT10", "TP9", "TP10",
             "F1", "F2", "C1", "C2"][:n_ch]
    n_times = int(30.0 * sfreq)
    rng = np.random.default_rng(1)
    data = rng.standard_normal((len(names), n_times)) * 1e-6
    info = mne.create_info(ch_names=names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.save(str(path), overwrite=True, verbose=False)
    return path


def _report(tag: str, result) -> None:
    print(f"\n=== {tag} ===")
    print(f"  ok           : {result.ok}")
    print(f"  mode         : {result.mode}")
    print(f"  n_channels   : {result.n_channels}")
    print(f"  data_type    : {result.data_type}")
    print(f"  data_path    : {result.data_path}")
    print(f"  reason       : {result.reason}")
    print(f"  log_path     : {result.log_path}")
    print(f"  status_path  : {result.status_path}")
    print(f"  pid          : {result.pid}")
    if result.status_path and Path(result.status_path).exists():
        print(f"  status file  : {json.dumps(json.loads(Path(result.status_path).read_text()))}")


def _kill(pid) -> None:
    if not pid:
        return
    try:
        import subprocess
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] could not kill pid {pid}: {exc}")


def main() -> None:
    fif = _build_fif(OUT / "verify32_raw.fif", n_ch=32)
    png = OUT / "channels_before_verify.png"
    import base64
    png.write_bytes(base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        b"YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    ))

    res_a = launch_eeg_viewer_sync(plot_path=png, title="Channels Before", data_path=fif)
    _report("A) real EEG data supplied", res_a)
    _kill(res_a.pid)

    res_b = launch_eeg_viewer_sync(plot_path=png, title="Channels Before", data_path=None)
    _report("B) PNG only, no EEG data resolvable", res_b)
    _kill(res_b.pid)

    print("\n[verify] asserting expected modes ...")
    assert res_a.mode == "interactive", f"expected interactive, got {res_a.mode!r}"
    assert res_a.n_channels == 32, f"expected 32 channels, got {res_a.n_channels}"
    assert res_a.data_type == "raw"
    assert res_b.mode == "static", f"expected static, got {res_b.mode!r}"
    assert res_b.reason, "static fallback must carry a reason"
    print("[verify] OK")


if __name__ == "__main__":
    main()
