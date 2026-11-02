"""Subprocess-safe launcher for the PyQt5 EEG interactive viewer.

Chainlit's event loop is asyncio-based; Qt's event loop is native.
Running both in the same process is fragile. This module spawns the viewer
as a *detached subprocess* so neither loop interferes with the other.

Usage from chainlit_app.py::

    from fc_pipeline.viz.viewer_launcher import launch_eeg_viewer
    await launch_eeg_viewer(plot_path="/path/to/channels_before.png",
                             title="channels_before")
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Absolute path to the eeg_viewer module (used as __main__ entry)
_VIEWER_MODULE = "fc_pipeline.viz.eeg_viewer"


async def launch_eeg_viewer(
    plot_path: str | Path,
    title: Optional[str] = None,
    data_path: Optional[str | Path] = None,
) -> bool:
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
        (.edf or .fif) to enable full-channel signal scrolling.

    Returns
    -------
    bool
        ``True`` if the subprocess was started without error,
        ``False`` if PyQt5 is missing or the path is invalid.
    """
    plot_path = Path(plot_path)
    if not plot_path.exists():
        logger.warning("launch_eeg_viewer: path does not exist: %s", plot_path)
        return False

    display_title = title or plot_path.stem

    cmd = [
        sys.executable,
        "-m",
        _VIEWER_MODULE,
        str(plot_path),
        display_title,
    ]
    if data_path and Path(str(data_path)).exists():
        cmd.extend(["--data", str(data_path)])

    # Check if PyQt5 is installed in current interpreter
    try:
        import PyQt5
    except ImportError:
        logger.warning(
            "EEG Viewer cannot start — PyQt5 not installed. "
            "Run: pip install PyQt5>=5.15"
        )
        return False

    # Ensure src directory is on PYTHONPATH for child process
    src_dir = Path(__file__).resolve().parent.parent.parent
    env = os.environ.copy()
    if "PYTHONPATH" in env and env["PYTHONPATH"]:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(src_dir)

    try:
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

        def _spawn() -> bool:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs,
            )
            logger.info("EEG Viewer launched for %s (pid=%s)", plot_path.name, proc.pid)
            return True

        return await asyncio.to_thread(_spawn)

    except FileNotFoundError:
        logger.error("Could not locate Python interpreter: %s", sys.executable)
        return False
    except Exception as exc:
        logger.exception("Unexpected error launching EEG Viewer: %s", exc)
        return False


def launch_eeg_viewer_sync(
    plot_path: str | Path,
    title: Optional[str] = None,
    data_path: Optional[str | Path] = None,
) -> bool:
    """Synchronous wrapper for non-async call sites (e.g. CLI scripts).

    Parameters
    ----------
    plot_path : str | Path
        Absolute path to the PNG file or EEG data file.
    title : str, optional
        Window title string.
    data_path : str | Path, optional
        Path to underlying EEG data file.

    Returns
    -------
    bool
        ``True`` if launched successfully.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(launch_eeg_viewer(plot_path, title, data_path))
    finally:
        loop.close()
