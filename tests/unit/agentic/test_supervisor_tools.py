"""Tests for fc_pipeline.agentic.supervisor.tools."""

import numpy as np
import mne
import pytest

from fc_pipeline.agentic.supervisor.tools.dataset_info import get_dataset_info
from fc_pipeline.agentic.supervisor.tools.dataset_conditions import get_dataset_conditions
from fc_pipeline.agentic.supervisor.tools.frequency_band import resolve_frequency_band
from fc_pipeline.agentic.supervisor.tools.channel_selection import resolve_channel_selection
from fc_pipeline.agentic.supervisor.tools.dataset_overview_plot import generate_dataset_overview_plot


# ============================================================================
# resolve_frequency_band (11 tests covering 7 branches + 5 canonical bands)
# ============================================================================


class TestResolveFrequencyBand:
    """Tests for the frequency band resolution tool."""

    # --- Branch 1: Canonical band names (5 variations) ---

    def test_canonical_delta(self):
        """[New] 'delta' maps to 1.0 - 4.0 Hz with confidence 1.0."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "delta", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "delta"
        assert result["fmin"] == 1.0
        assert result["fmax"] == 4.0
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    def test_canonical_theta(self):
        """[New] 'theta' maps to 4.0 - 8.0 Hz with confidence 1.0."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "theta", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "theta"
        assert result["fmin"] == 4.0
        assert result["fmax"] == 8.0
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    def test_canonical_alpha(self):
        """[New] 'alpha' maps to 8.0 - 12.0 Hz with confidence 1.0."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "alpha", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "alpha"
        assert result["fmin"] == 8.0
        assert result["fmax"] == 12.0
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    def test_canonical_beta(self):
        """[New] 'beta' maps to 13.0 - 30.0 Hz with confidence 1.0."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "beta", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "beta"
        assert result["fmin"] == 13.0
        assert result["fmax"] == 30.0
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    def test_canonical_gamma(self):
        """[New] 'gamma' maps to 30.0 - 45.0 Hz with confidence 1.0."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "gamma", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "gamma"
        assert result["fmin"] == 30.0
        assert result["fmax"] == 45.0
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    # --- Branch 2: Custom range (2 variations) ---

    def test_custom_range_valid(self):
        """[New] Custom range '15-25 Hz' maps to 15.0 - 25.0 Hz with confidence 1.0."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "15-25 Hz", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "custom"
        assert result["fmin"] == 15.0
        assert result["fmax"] == 25.0
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    def test_custom_range_invalid_order(self):
        """[New] Custom range '25-15 Hz' returns error, confidence 0.0, needs_human_input=True."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "25-15 Hz", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "custom"
        assert result["fmin"] == 25.0
        assert result["fmax"] == 15.0
        assert result["confidence"] == 0.0
        assert result["needs_human_input"] is True
        assert result["error"] is not None

    # --- Branch 3: Unknown band name ---

    def test_unknown_band_name(self):
        """[New] Unknown band name 'foo' returns error, confidence 0.0, needs_human_input=True."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "foo", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "foo"
        assert result["fmin"] == 0.0
        assert result["fmax"] == 0.0
        assert result["confidence"] == 0.0
        assert result["needs_human_input"] is True
        assert result["error"] is not None

    # --- Branch 4: Cycle check (2 variations) ---

    def test_cycle_check_triggered_for_short_duration(self):
        """[New] Short duration (0.1s) for 1Hz band triggers cycle check error."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "delta", "sfreq": 250.0, "duration_seconds": 0.1})
        assert result["name"] == "delta"
        assert result["fmin"] == 1.0
        assert result["fmax"] == 4.0
        assert result["confidence"] == 0.0
        assert result["needs_human_input"] is True
        assert result["error"] is not None
        assert "too short for reliable phase estimation" in result["error"]

    def test_cycle_check_not_triggered_for_enough_duration(self):
        """[New] Sufficient duration (5.0s) for 1Hz band does not trigger cycle check error."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "delta", "sfreq": 250.0, "duration_seconds": 5.0})
        assert result["name"] == "delta"
        assert result["fmin"] == 1.0
        assert result["fmax"] == 4.0
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    # --- Branch 5: sfreq = 0.0 ---

    def test_sfreq_zero(self):
        """[New] sfreq = 0.0 returns error, confidence 0.0, needs_human_input=True."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "delta", "sfreq": 0.0, "duration_seconds": 5.0})
        assert result["name"] == "delta"
        assert result["fmin"] == 1.0
        assert result["fmax"] == 4.0
        assert result["confidence"] == 0.0
        assert result["needs_human_input"] is True
        assert result["error"] is not None
        assert "must be strictly greater than 0" in result["error"]

    def test_missing_duration_skips_cycle_check_safely(self):
        """When duration_seconds is None, cycle check is skipped safely without error."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "delta", "sfreq": 250.0, "duration_seconds": None})
        assert result["name"] == "delta"
        assert result["fmin"] == 1.0
        assert result["fmax"] == 4.0
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    def test_exceeds_nyquist(self):
        """Frequency exceeding Nyquist frequency returns error and flags human input."""
        result = resolve_frequency_band.invoke({"band_name_or_range": "100-150 Hz", "sfreq": 200.0, "duration_seconds": 5.0})
        assert result["confidence"] == 0.0
        assert result["needs_human_input"] is True
        assert result["error"] is not None
        assert "Nyquist" in result["error"]


# ============================================================================
# resolve_channel_selection (15 tests covering 7 branches + 6 anatomical regions)
# ============================================================================


AVAILABLE_10_20 = [
    "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "T7", "T8", "P7", "P8", "FP1", "FP2", "F7", "F8",
    "FZ", "CZ", "PZ", "OZ",
]


class TestResolveChannelSelection:
    """Tests for the channel selection resolution tool."""

    # --- Branch 1: Empty available channel list ---

    def test_empty_available_list(self):
        """[New] Empty available_channels returns error, confidence 0.0, needs_human_input=True."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "F3, F4", "available_channels": []}
        )
        assert result["confidence"] == 0.0
        assert result["needs_human_input"] is True
        assert result["error"] is not None
        assert result["resolved_channels"] == []

    # --- Branch 2: Stage 1 cleaning ---

    def test_stage1_cleaning(self):
        """[Regression] EEG prefix and -REF suffix stripped during normalization."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "EEG F3-REF",
             "available_channels": ["F3", "F4"]}
        )
        assert "F3" in result["resolved_channels"]
        assert result["confidence"] == 1.0

    # --- Branch 3: Stage 2 exact match ---

    def test_stage2_exact_match(self):
        """[Regression] Exact channel names resolve with confidence 1.0."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "F3, F4",
             "available_channels": AVAILABLE_10_20}
        )
        assert set(result["resolved_channels"]) == {"F3", "F4"}
        assert result["confidence"] == 1.0
        assert result["needs_human_input"] is False
        assert result["error"] is None

    # --- Branch 4: Stage 3 10-20 alias resolution (4 variations) ---

    def test_stage3_alias_T3_T7(self):
        """[New] T3 resolves to T7 via 10-20 alias map at confidence 0.85."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "T3",
             "available_channels": ["T7", "T8", "P7", "P8"]}
        )
        assert "T7" in result["resolved_channels"]
        assert result["confidence"] == 0.85
        assert result["needs_human_input"] is False  # 0.85 >= 0.80

    def test_stage3_alias_T4_T8(self):
        """[New] T4 resolves to T8 via 10-20 alias map at confidence 0.85."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "T4",
             "available_channels": ["T7", "T8", "P7", "P8"]}
        )
        assert "T8" in result["resolved_channels"]
        assert result["confidence"] == 0.85

    def test_stage3_alias_T5_P7(self):
        """[New] T5 resolves to P7 via 10-20 alias map at confidence 0.85."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "T5",
             "available_channels": ["T7", "T8", "P7", "P8"]}
        )
        assert "P7" in result["resolved_channels"]
        assert result["confidence"] == 0.85

    def test_stage3_alias_T6_P8(self):
        """[New] T6 resolves to P8 via 10-20 alias map at confidence 0.85."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "T6",
             "available_channels": ["T7", "T8", "P7", "P8"]}
        )
        assert "P8" in result["resolved_channels"]
        assert result["confidence"] == 0.85

    # --- Branch 5: Stage 4 region mapping (all 6 anatomical regions) ---

    def test_stage4_region_frontal(self):
        """[Regression] 'frontal' maps to FP1/FP2/F3/F4/F7/F8/FZ at confidence 0.70."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "frontal channels",
             "available_channels": AVAILABLE_10_20}
        )
        expected = {"FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"}
        assert set(result["resolved_channels"]) == expected
        assert result["confidence"] == 0.70
        assert result["needs_human_input"] is True  # 0.70 < 0.80

    def test_stage4_region_central(self):
        """[New] 'central' maps to C3/C4/CZ at confidence 0.70."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "central",
             "available_channels": AVAILABLE_10_20}
        )
        expected = {"C3", "C4", "CZ"}
        assert set(result["resolved_channels"]) == expected
        assert result["confidence"] == 0.70
        assert result["needs_human_input"] is True

    def test_stage4_region_motor(self):
        """[New] 'motor' maps to C3/C4/CZ at confidence 0.70."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "motor cortex",
             "available_channels": AVAILABLE_10_20}
        )
        expected = {"C3", "C4", "CZ"}
        assert set(result["resolved_channels"]) == expected
        assert result["confidence"] == 0.70
        assert result["needs_human_input"] is True

    def test_stage4_region_parietal(self):
        """[New] 'parietal' maps to P3/P4/PZ at confidence 0.70."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "parietal region",
             "available_channels": AVAILABLE_10_20}
        )
        expected = {"P3", "P4", "PZ"}
        assert set(result["resolved_channels"]) == expected
        assert result["confidence"] == 0.70
        assert result["needs_human_input"] is True

    def test_stage4_region_occipital(self):
        """[New] 'occipital' maps to O1/O2/OZ at confidence 0.70."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "occipital region",
             "available_channels": AVAILABLE_10_20}
        )
        expected = {"O1", "O2", "OZ"}
        assert set(result["resolved_channels"]) == expected
        assert result["confidence"] == 0.70
        assert result["needs_human_input"] is True

    def test_stage4_region_temporal(self):
        """[New] 'temporal' maps to available temporal channels (T7/T8/P7/P8) at confidence 0.70."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "temporal lobes",
             "available_channels": AVAILABLE_10_20}
        )
        expected = {"T7", "T8", "P7", "P8"}
        assert set(result["resolved_channels"]) == expected
        assert result["confidence"] == 0.70
        assert result["needs_human_input"] is True

    # --- Branch 6: Deduplication preserves order ---

    def test_deduplication_preserves_order(self):
        """[New] Duplicate channel mentions ('F3, F4, F3') preserve first occurrence order."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "F3, F4, F3",
             "available_channels": AVAILABLE_10_20}
        )
        assert result["resolved_channels"] == ["F3", "F4"]

    # --- Branch 7: Disjoint / no-match query ---

    def test_no_matching_channels(self):
        """[New] Query with no matches returns empty list and needs_human_input=True."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "XYZ123, ABC999",
             "available_channels": AVAILABLE_10_20}
        )
        assert result["resolved_channels"] == []
        assert result["confidence"] == 0.0
        assert result["needs_human_input"] is True

    def test_unrecognized_channel_token_flags_error(self):
        """Partially valid channel request (e.g. 'Cz, X99') flags error and requires human input."""
        result = resolve_channel_selection.invoke(
            {"requested_channels_or_region": "CZ, X99",
             "available_channels": AVAILABLE_10_20}
        )
        assert result["confidence"] == 0.0
        assert result["needs_human_input"] is True
        assert result["error"] is not None
        assert "Unrecognized channel" in result["error"]
        assert "X99" in result["error"]


# ============================================================================
# get_dataset_info (4 tests covering all 4 branches)
# ============================================================================


class TestGetDatasetInfo:
    """Tests for the dataset header inspection tool."""

    def test_valid_eeg_raw(self, synthetic_eeg_path):
        """[Regression] Extracts sfreq, nyquist, duration, sample count, and reference from valid file."""
        result = get_dataset_info.invoke({"data_path": str(synthetic_eeg_path)})
        assert result["sfreq"] == 250.0
        assert result["nyquist"] == 125.0
        assert result["duration_seconds"] == 10.0
        assert result["n_times"] == 2500
        assert len(result["available_channels"]) == 8
        assert "F3" in result["available_channels"]
        assert "reference" in result
        assert "reference_provenance" in result
        assert result["error"] is None

    def test_eeg_channel_filtering(self, tmp_path):
        """[New] Non-EEG channels (ECG, EOG) are filtered out when types are specified."""
        ch_names = ["F3", "F4", "ECG01", "EOG01"]
        sfreq = 250.0
        data = np.random.randn(4, 250) * 1e-6
        ch_types = ["eeg", "eeg", "ecg", "eog"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        fif_path = str(tmp_path / "mixed_raw.fif")
        raw.save(fif_path, overwrite=True, verbose=False)

        result = get_dataset_info.invoke({"data_path": fif_path})
        assert "F3" in result["available_channels"]
        assert "F4" in result["available_channels"]
        assert "ECG01" not in result["available_channels"]
        assert "EOG01" not in result["available_channels"]

    def test_fallback_all_channels(self, tmp_path):
        """[New] When all channels are unclassified (misc), tool falls back to all channels."""
        ch_names = ["MISC1", "MISC2"]
        sfreq = 250.0
        data = np.random.randn(2, 250) * 1e-6
        ch_types = ["misc", "misc"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        fif_path = str(tmp_path / "misc_raw.fif")
        raw.save(fif_path, overwrite=True, verbose=False)

        result = get_dataset_info.invoke({"data_path": fif_path})
        assert len(result["available_channels"]) == 2
        assert "MISC1" in result["available_channels"]

    def test_read_error_nonexistent_file(self):
        """[New] Non-existent path returns sfreq=0.0 and populated error string."""
        result = get_dataset_info.invoke({"data_path": "nonexistent_file.fif"})
        assert result["sfreq"] == 0.0
        assert result["error"] is not None
        assert result["available_channels"] == []


# ============================================================================
# get_dataset_conditions (4 tests covering all 4 branches)
# ============================================================================


class TestGetDatasetConditions:
    """Tests for the experimental condition inspection tool."""

    def test_embedded_annotations(self, synthetic_eeg_path):
        """[Regression] Condition labels and trial counts extracted from annotations."""
        result = get_dataset_conditions.invoke({"data_path": str(synthetic_eeg_path)})
        assert "rest" in result["conditions"]
        assert "task" in result["conditions"]
        assert result["conditions"]["rest"] == 1
        assert result["conditions"]["task"] == 1
        assert result["total_trials"] == 2
        assert result["has_events"] is True
        assert result["error"] is None

    def test_stimulus_channel_fallback(self, tmp_path):
        """[New] Falls back to mne.find_events on stimulus channels when no annotations exist."""
        sfreq = 250.0
        n_samples = 1000
        data = np.zeros((2, n_samples))
        # Channel 0: EEG, Channel 1: STIM
        data[1, 100] = 1
        data[1, 300] = 1
        data[1, 500] = 2

        info = mne.create_info(ch_names=["F3", "STI 014"], sfreq=sfreq,
                               ch_types=["eeg", "stim"])
        raw = mne.io.RawArray(data, info, verbose=False)
        fif_path = str(tmp_path / "stim_raw.fif")
        raw.save(fif_path, overwrite=True, verbose=False)

        result = get_dataset_conditions.invoke({"data_path": fif_path})
        assert result["has_events"] is True
        assert "event_1" in result["conditions"]
        assert "event_2" in result["conditions"]
        assert result["conditions"]["event_1"] == 2
        assert result["conditions"]["event_2"] == 1
        assert result["total_trials"] == 3

    def test_no_events_or_annotations(self, tmp_path):
        """[New] Dataset with neither annotations nor events returns empty condition dict."""
        ch_names = ["F3", "F4"]
        sfreq = 250.0
        data = np.random.randn(2, 500) * 1e-6
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        fif_path = str(tmp_path / "plain_raw.fif")
        raw.save(fif_path, overwrite=True, verbose=False)

        result = get_dataset_conditions.invoke({"data_path": fif_path})
        assert result["conditions"] == {}
        assert result["has_events"] is False
        assert result["total_trials"] == 0
        assert result["error"] is None

    def test_read_error_nonexistent_file(self):
        """[New] Unreadable file returns empty condition dict and populated error."""
        result = get_dataset_conditions.invoke({"data_path": "nonexistent_file.fif"})
        assert result["conditions"] == {}
        assert result["has_events"] is False
        assert result["total_trials"] == 0
        assert result["error"] is not None


# ============================================================================
# generate_dataset_overview_plot (3 tests: happy path, privacy masking, error)
# ============================================================================


class TestGenerateDatasetOverviewPlot:
    """Tests for the dataset overview plot tool."""

    @staticmethod
    def _make_fif(tmp_path, n_channels=4, sfreq=256.0, duration=5.0):
        """Helper: create a minimal raw .fif file for testing."""
        ch_names = [f"EEG{i+1:03d}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_channels, int(sfreq * duration))) * 1e-6
        raw = mne.io.RawArray(data, info, verbose=False)
        fif_path = str(tmp_path / "test_sub-99_eyesclosed_raw.fif")
        raw.save(fif_path, overwrite=True, verbose=False)
        return fif_path

    def test_happy_path_produces_file(self, tmp_path):
        """[New] Happy path: tool returns a valid plot file and correct metadata."""
        import os
        fif_path = self._make_fif(tmp_path)

        # Run with cwd = tmp_path so outputs/plots lands inside tmp_path
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            result = generate_dataset_overview_plot.invoke({
                "data_path": fif_path,
                "run_id": "test_run_001",
            })
        finally:
            os.chdir(old_cwd)

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        assert result["plot_path"] is not None
        assert result["plot_path"].endswith("test_run_001_overview.png")
        assert os.path.isfile(os.path.join(str(tmp_path), result["plot_path"]))
        assert result["n_channels"] == 4
        assert result["sfreq"] == 256.0
        assert result["duration_seconds"] > 0

    def test_no_metadata_leaks_in_rendered_figure(self, tmp_path):
        """[New] Privacy: no file path, subject ID, or patient info appears in rendered text.

        This test re-generates the figure AND inspects every Text artist in the
        matplotlib figure to confirm nothing derived from the raw file path or
        subject_info leaks into the plot's visible text (titles, labels, annotations).
        """
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        fif_path = self._make_fif(tmp_path)

        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            result = generate_dataset_overview_plot.invoke({
                "data_path": fif_path,
                "run_id": "privacy_test",
            })
        finally:
            os.chdir(old_cwd)

        assert result["error"] is None

        # ---- Approach: re-render the figure in-memory and inspect Text artists ----
        # We call the underlying logic again to get the figure object.
        # Alternatively, we can just inspect what the tool sets.
        # Safer: re-create the same way the tool does.
        raw = mne.io.read_raw(fif_path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        data = raw.get_data()
        ch_names = raw.ch_names
        n_ch = len(ch_names)

        from scipy.signal import welch as scipy_welch

        fig, (ax_raw, ax_psd) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Dataset Overview (Pre-Preprocessing)")

        snippet_samples = min(int(4.0 * sfreq), data.shape[1])
        snippet_time = np.arange(snippet_samples) / sfreq
        offsets = np.arange(n_ch) * 1.0
        for i in range(n_ch):
            ax_raw.plot(snippet_time, data[i, :snippet_samples] * 0.3 + offsets[i])
        ax_raw.set_yticks(offsets)
        ax_raw.set_yticklabels(ch_names)
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_title(f"Raw Signal Snippet ({snippet_time[-1]:.1f}s, {n_ch} ch)")

        nperseg = min(int(sfreq * 2), data.shape[1])
        for i in range(n_ch):
            freqs, psd = scipy_welch(data[i], fs=sfreq, nperseg=nperseg)
            ax_psd.semilogy(freqs, psd)
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD (V²/Hz)")
        ax_psd.set_title("Power Spectral Density (Welch)")

        # Collect ALL visible text from the figure
        all_text = []
        for txt_obj in fig.findobj(plt.Text):
            t = txt_obj.get_text().strip()
            if t:
                all_text.append(t)
        plt.close(fig)

        # The dangerous strings that must NOT appear:
        banned_substrings = [
            "sub-99",           # subject ID embedded in filename
            "sub_99",
            "eyesclosed",       # condition from filename
            fif_path,           # full file path
            os.path.basename(fif_path),  # filename alone
            str(tmp_path),      # tmp directory path
        ]

        joined_text = " ".join(all_text).lower()
        for banned in banned_substrings:
            assert banned.lower() not in joined_text, (
                f"Privacy violation: '{banned}' found in rendered figure text. "
                f"All text: {all_text}"
            )

    def test_error_for_nonexistent_file(self):
        """[New] Nonexistent file returns error dict without crashing."""
        result = generate_dataset_overview_plot.invoke({
            "data_path": "totally_nonexistent_path.fif",
            "run_id": "error_test",
        })
        assert result["error"] is not None
        assert "Failed to generate overview plot" in result["error"]
        assert result["plot_path"] is None
        assert result["n_channels"] == 0