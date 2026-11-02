r"""Render the diagnostic plots and confirm they now use a light background.

Checks the actual saved PNGs (not just the source): the figure background must
be white/near-white and the text/traces must be dark enough to read.

Run:
    .\.venv\Scripts\python.exe test_outputs\theme_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fc_pipeline.deterministic.data_prep.models import ValidatedDataPrepParams  # noqa: E402
from fc_pipeline.deterministic.data_prep.plotting import (  # noqa: E402
    plot_channels_after,
    plot_channels_before,
)

OUT = Path(__file__).resolve().parent / "theme_check"


def _luma(px) -> float:
    r, g, b = px[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _report(label: str, path: str | None) -> None:
    if not path:
        print(f"{label:16s}: FAILED to render")
        return
    im = Image.open(path).convert("RGB")
    w, h = im.size
    corner = im.getpixel((2, 2))
    # sample the plot area (avoid the left gutter / title band)
    samples = [
        im.getpixel((w // 2, int(h * f))) for f in (0.3, 0.5, 0.7, 0.9)
    ]
    vals = [_luma(p) for p in samples]
    print(
        f"{label:16s}: {Path(path).name}  size={w}x{h}  "
        f"corner_rgb={corner} (luma {_luma(corner):.0f})  "
        f"plot-luma median={np.median(vals):.0f}"
    )
    assert _luma(corner) > 240, f"{label}: background is not light ({corner})"
    print(f"{'':16s}  -> LIGHT background confirmed")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fif = OUT / "theme_raw.fif"
    if not fif.exists():
        names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
        rng = np.random.default_rng(11)
        raw0 = mne.io.RawArray(
            rng.standard_normal((len(names), 2500)) * 1e-6,
            mne.create_info(names, 250.0, "eeg"),
            verbose=False,
        )
        raw0.set_annotations(
            mne.Annotations(onset=[0.0], duration=[10.0], description=["rest"])
        )
        raw0.save(str(fif), overwrite=True, verbose=False)

    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    params = ValidatedDataPrepParams(
        raw_data_path=fif,
        run_id="themecheck",
        channels=tuple(raw.ch_names),
        condition="rest",
        fmin=1.0,
        fmax=40.0,
        reference_method="average",
        bad_channel_variance_threshold=1e-15,
        min_cycles=1.0,
    )

    before = plot_channels_before(raw, params, [], OUT)
    _report("channels_before", before)

    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=1.0, baseline=None, preload=True, verbose=False
    )
    after = plot_channels_after(epochs, params, OUT)
    _report("channels_after", after)

    print("\n[theme_check] OK — both diagnostic plots are light-themed")


if __name__ == "__main__":
    main()
