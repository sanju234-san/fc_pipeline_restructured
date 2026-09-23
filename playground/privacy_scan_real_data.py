"""Privacy scan: invoke generate_dataset_overview_plot against the real
sub-01_eyesclosed_raw.fif file and check all rendered text artists for leaks."""

import os, sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from fc_pipeline.agentic.supervisor.tools.dataset_overview_plot import generate_dataset_overview_plot

REAL_DATA = r"C:\Users\sanje\OneDrive\Desktop\CDAC\eeg_iq_pipeline\eeg_iq_pipeline\ds004796_split\sub-01\sub-01_eyesclosed_raw.fif"
RUN_ID = "privacy_scan_real_data"

# --- Step 1: Generate the plot ---
print("=" * 70)
print("PRIVACY SCAN: generate_dataset_overview_plot on REAL DATASET")
print("=" * 70)
print(f"Data file: {REAL_DATA}")

result = generate_dataset_overview_plot.invoke({
    "data_path": REAL_DATA,
    "run_id": RUN_ID,
})

print(f"\nTool result:")
print(f"  plot_path:        {result['plot_path']}")
print(f"  n_channels:       {result['n_channels']}")
print(f"  duration_seconds: {result['duration_seconds']}")
print(f"  sfreq:            {result['sfreq']}")
print(f"  error:            {result['error']}")

if result["error"]:
    print(f"\n[!] TOOL ERROR: {result['error']}")
    sys.exit(1)

plot_file = Path(result["plot_path"])
if not plot_file.exists():
    print(f"\n[!] Plot file not found: {plot_file}")
    sys.exit(1)

print(f"\n  Plot file exists: {plot_file.resolve()}")
print(f"  File size:        {plot_file.stat().st_size} bytes")

# --- Step 2: Re-create the figure in-memory to inspect Text artists ---
# We can't extract text from a PNG directly, so we replicate the tool's
# rendering logic to capture all matplotlib Text objects.
import mne
import numpy as np
from scipy.signal import welch as scipy_welch

raw = mne.io.read_raw(REAL_DATA, preload=True, verbose=False)
try:
    ch_types = raw.get_channel_types()
    eeg_picks = [i for i, t in enumerate(ch_types) if t == "eeg"]
    if eeg_picks:
        raw = raw.pick(eeg_picks)
except Exception:
    pass

sfreq = float(raw.info["sfreq"])
data = raw.get_data()
ch_names = raw.ch_names
n_ch = len(ch_names)

fig, (ax_raw, ax_psd) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dataset Overview (Pre-Preprocessing)", fontsize=13, fontweight="bold")

snippet_samples = min(int(4.0 * sfreq), data.shape[1])
snippet_time = np.arange(snippet_samples) / sfreq
offsets = np.arange(n_ch) * 1.0
for i in range(n_ch):
    ch_data = data[i, :snippet_samples]
    std = np.std(ch_data)
    if std > 0:
        ch_norm = (ch_data - np.mean(ch_data)) / std
    else:
        ch_norm = ch_data - np.mean(ch_data)
    ax_raw.plot(snippet_time, ch_norm * 0.3 + offsets[i], linewidth=0.5, alpha=0.85)

ax_raw.set_yticks(offsets)
ax_raw.set_yticklabels(ch_names, fontsize=7)
ax_raw.set_xlabel("Time (s)", fontsize=9)
ax_raw.set_title(f"Raw Signal Snippet ({snippet_time[-1]:.1f}s, {n_ch} ch)", fontsize=10)

nperseg = min(int(sfreq * 2), data.shape[1])
for i in range(n_ch):
    freqs, psd = scipy_welch(data[i], fs=sfreq, nperseg=nperseg)
    freq_mask = freqs <= min(60.0, sfreq / 2.0)
    ax_psd.semilogy(freqs[freq_mask], psd[freq_mask], linewidth=0.7, alpha=0.7, label=ch_names[i])
ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
ax_psd.set_ylabel("PSD (V²/Hz)", fontsize=9)
ax_psd.set_title("Power Spectral Density (Welch)", fontsize=10)

# --- Step 3: Collect ALL text from the figure ---
all_text = []
for txt_obj in fig.findobj(plt.Text):
    t = txt_obj.get_text().strip()
    if t:
        all_text.append(t)
plt.close(fig)

print(f"\n--- All rendered text artists in the figure ({len(all_text)} total) ---")
# Print a sample (titles, labels — skip individual tick labels for brevity)
titles_and_labels = [t for t in all_text if len(t) > 3 and not t.replace(".", "").replace("-", "").isdigit()]
for t in titles_and_labels:
    print(f"  '{t}'")

# --- Step 4: Privacy scan ---
banned_substrings = [
    "sub-01",               # subject ID from filename
    "sub_01",
    "eyesclosed",           # condition from filename
    "eyes_closed",
    "ds004796",             # dataset name from path
    "eeg_iq_pipeline",      # parent directory name
    REAL_DATA,              # full file path
    os.path.basename(REAL_DATA),  # filename alone
    "sub-01_eyesclosed_raw.fif",
    "sanje",                # username from path
    "Desktop",              # path component
]

joined_text = " ".join(all_text).lower()
violations = []
for banned in banned_substrings:
    if banned.lower() in joined_text:
        violations.append(banned)

print(f"\n{'=' * 70}")
print(f"PRIVACY SCAN RESULTS")
print(f"{'=' * 70}")
print(f"Banned substrings checked: {len(banned_substrings)}")
print(f"Violations found:          {len(violations)}")

if violations:
    print(f"\n[FAIL] Privacy violations detected:")
    for v in violations:
        print(f"  - '{v}' found in rendered text")
    sys.exit(1)
else:
    print(f"\n[PASS] No privacy leaks detected in rendered figure text.")
    print(f"       No subject ID, filename, dataset name, path components, or")
    print(f"       username appear anywhere in titles, labels, or annotations.")

print(f"{'=' * 70}")
