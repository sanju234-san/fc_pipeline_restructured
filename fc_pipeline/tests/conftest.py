"""Pytest fixtures: synthetic MNE Raw/Epochs datasets and sample GraphState."""

import pytest
import numpy as np
import mne


@pytest.fixture(scope="session")
def synthetic_eeg_path(tmp_path_factory):
    """Create a small 10-20 synthetic EEG .fif file once per test session.

    Fixture Characteristics:
      - Channels: F3, F4, C3, C4, P3, P4, O1, O2 (8 standard 10-20)
      - Sampling Rate: 250.0 Hz (Nyquist: 125.0 Hz)
      - Duration: 10.0 seconds (2500 samples)
      - Annotations: 'rest' (0–5 s), 'task' (5–10 s)
    """
    tmp_dir = tmp_path_factory.mktemp("eeg_fixtures")
    fif_path = tmp_dir / "test_raw.fif"

    ch_names = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    sfreq = 250.0
    n_samples = int(10.0 * sfreq)

    np.random.seed(42)
    data = np.random.randn(len(ch_names), n_samples) * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    annotations = mne.Annotations(
        onset=[0.0, 5.0],
        duration=[5.0, 5.0],
        description=["rest", "task"],
    )
    raw.set_annotations(annotations)
    raw.save(str(fif_path), overwrite=True, verbose=False)
    return fif_path
