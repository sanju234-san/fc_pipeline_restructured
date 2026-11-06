"""Subprocess-safe launcher for the PyQt5 EEG interactive viewer.

Chainlit's event loop is asyncio-based; Qt's event loop is native.
Running both in the same process is fragile. This module spawns the viewer
as a *detached subprocess* so neither loop interferes with the other.

Unlike the previous version, the viewer's stdout/stderr are **no longer
discarded**: they are written to a log file, and the viewer publishes a JSON
status file that this launcher reads back.  The caller therefore learns the
*actual* mode that started — ``interactive`` (real channel canvas) or
``static`` (degraded image view) — plus the resolved channel count and, on
failure, the underlying reason.

Usage from chainlit_app.py::

    from fc_pipeline.viz.viewer_launcher import launch_eeg_viewer
    result = await launch_eeg_viewer(plot_path="/path/to/channels_before.png",
                                     title="channels_before")
    if result.mode == "interactive":
        ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fc_pipeline.viz.eeg_probe import probe_eeg_dataset, resolve_eeg_dataset

logger = logging.getLogger(__name__)

# Absolute path to the eeg_viewer module (used as __main__ entry)
_VIEWER_MODULE = "fc_pipeline.viz.eeg_viewer"

# Default directory (relative to the project) for viewer diagnostics/status
_DEFAULT_LOG_DIR = Path("logs") / "eeg_viewer"

# How long (seconds) to wait for the viewer to report its real mode.
_STATUS_WAIT_SECONDS = 6.0
_STATUS_POLL_INTERVAL = 0.1


@dataclass
class ViewerLaunchResult:
    """Outcome of a viewer launch attempt."""

    ok: bool
    mode: str  # "interactive" | "static" | "failed"
    n_channels: int = 0
    data_type: Optional[str] = None
    data_path: Optional[str] = None
    log_path: Optional[str] = None
    status_path: Optional[str] = None
    pid: Optional[int] = None
    reason: str = ""

    @property
    def interactive(self) -> bool:
        return self.mode == "interactive"


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in text]
    return "".join(keep)[:80] or "viewer"


async def launch_eeg_viewer(
    plot_path: str | Path,
    title: Optional[str] = None,
    data_path: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
    wait_for_status: float = _STATUS_WAIT_SECONDS,
) -> ViewerLaunchResult:
    """Launch the interactive EEG viewer in a detached subprocess.

    Parameters
    ----------
    plot_path : str | Path
        Absolute path to the PNG plot file or EEG data file.
    title : str, optional
        Window title string shown in the viewer toolbar.
        Defaults to the filename stem.
    data_path : str | Path, optional
        Optional path to the underlying raw or preprocessed EEG dataset
        (.edf/.fif) to enable full-channel signal scrolling.
    log_dir : str | Path, optional
        Directory for the per-launch stdout/stderr log and status JSON.
        Defaults to ``logs/eeg_viewer/``.
    wait_for_status : float
        Max seconds to wait for the viewer's status file before falling back
        to the preflight prediction.

    Returns
    -------
    ViewerLaunchResult
        ``mode`` is authoritative where possible: it is read back from the
        viewer's own status file (``interactive`` / ``static``), and only
        falls back to the preflight probe prediction on timeout.
    """

    # Headless (e.g. Colab): no display -> skip PyQt5, caller falls back to browser viewer
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return ViewerLaunchResult(ok=False, mode="failed", reason="no display (headless)")
    plot_path = Path(plot_path)
    if not plot_path.exists():
        logger.warning("launch_eeg_viewer: path does not exist: %s", plot_path)
        return ViewerLaunchResult(ok=False, mode="failed", reason=f"plot path not found: {plot_path}")

    display_title = title or plot_path.stem

    # Check if PyQt5 is installed in current interpreter
    try:
        import PyQt5  # noqa: F401
    except ImportError:
        logger.debug(
            "EEG Viewer desktop pop-out skipped — PyQt5 not installed."
        )
        return ViewerLaunchResult(
            ok=False, mode="failed",
            reason="PyQt5 is not installed (pip install PyQt5>=5.15)",
        )

    # ── Preflight: resolve the EEG file the viewer will use and probe it ──
    resolved_data = resolve_eeg_dataset(plot_path, data_path)
    probe = probe_eeg_dataset(resolved_data) if resolved_data else None
    predicted_mode = "interactive" if probe else "static"
    predicted_reason = ""
    if probe is None:
        if resolved_data is None:
            predicted_reason = (
                "no EEG data file resolved for this plot (no raw/preprocessed path "
                "supplied and no run_context_*.json found next to the plot)"
            )
        else:
            predicted_reason = f"{resolved_data} could not be read as MNE Raw/Epochs"

    # ── Per-launch diagnostics directory ─────────────────────────────────
    status_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
    status_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(plot_path.stem)
    log_path = status_dir / f"{stem}.log"
    status_path = status_dir / f"{stem}.status.json"
    try:
        status_path.unlink()  # never read a stale status from a previous launch
    except FileNotFoundError:
        pass
    except OSError:
        pass

    cmd = [
        sys.executable,
        "-m",
        _VIEWER_MODULE,
        str(plot_path),
        display_title,
        "--log-file", str(log_path),
        "--status-file", str(status_path),
    ]
    if resolved_data is not None:
        cmd.extend(["--data", str(resolved_data)])

    # Ensure src directory is on PYTHONPATH for child process
    src_dir = Path(__file__).resolve().parent.parent.parent
    env = os.environ.copy()
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(src_dir)

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    def _spawn() -> Optional[int]:
        with open(log_path, "ab") as logf:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        logger.info(
            "EEG Viewer launched for %s (pid=%s, log=%s)", plot_path.name, proc.pid, log_path
        )
        return proc.pid

    try:
        pid = await asyncio.to_thread(_spawn)
    except FileNotFoundError:
        logger.error("Could not locate Python interpreter: %s", sys.executable)
        return ViewerLaunchResult(
            ok=False, mode="failed", reason=f"python interpreter not found: {sys.executable}",
            log_path=str(log_path),
        )
    except Exception as exc:
        logger.exception("Unexpected error launching EEG Viewer: %s", exc)
        return ViewerLaunchResult(
            ok=False, mode="failed", reason=f"{type(exc).__name__}: {exc}",
            log_path=str(log_path),
        )

    # ── Read back the viewer's real mode (authoritative) ─────────────────
    def _read_status() -> Optional[dict]:
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict) and data.get("mode") == "loading":
            return None
        return data if isinstance(data, dict) else None

    deadline = time.monotonic() + max(0.0, wait_for_status)
    real_status = None
    while time.monotonic() < deadline:
        real_status = await asyncio.to_thread(_read_status)
        if real_status:
            break
        await asyncio.sleep(_STATUS_POLL_INTERVAL)

    if real_status:
        mode = str(real_status.get("mode", predicted_mode))
        return ViewerLaunchResult(
            ok=True,
            mode=mode,
            n_channels=int(real_status.get("n_channels") or 0),
            data_type=real_status.get("data_type"),
            data_path=real_status.get("data_path") or (str(resolved_data) if resolved_data else None),
            log_path=str(log_path),
            status_path=str(status_path),
            pid=pid,
            reason=str(real_status.get("reason") or ""),
        )

    # Timed out: report the preflight prediction and point at the log.
    logger.warning(
        "EEG Viewer status not published within %.1fs; falling back to preflight "
        "prediction (%s). See %s", wait_for_status, predicted_mode, log_path,
    )
    return ViewerLaunchResult(
        ok=True,
        mode=predicted_mode,
        n_channels=int((probe or {}).get("n_channels") or 0),
        data_type=(probe or {}).get("data_type"),
        data_path=str(resolved_data) if resolved_data else None,
        log_path=str(log_path),
        status_path=str(status_path),
        pid=pid,
        reason=predicted_reason or "viewer mode not confirmed in time (status file pending)",
    )


def launch_eeg_viewer_sync(
    plot_path: str | Path,
    title: Optional[str] = None,
    data_path: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
) -> ViewerLaunchResult:
    """Synchronous wrapper for non-async call sites (e.g. CLI scripts)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            launch_eeg_viewer(plot_path, title, data_path, log_dir)
        )
    finally:
        loop.close()
