import sys
sys.path.insert(0, 'src')

from fc_pipeline.research.grpo_reward_fn import metric_classification_reward, composite_reward
from fc_pipeline.config.thresholds import TAU_PHASE, TAU_ZEROLAG

print(f"Thresholds: TAU_PHASE={TAU_PHASE}, TAU_ZEROLAG={TAU_ZEROLAG}")

weak_resp = '{"classification": "weak", "metrics": ["PLI"]}'
r = metric_classification_reward("Metric: PLI\nConnectivity value: 0.05", weak_resp)
assert r == 1.0, f"Expected 1.0, got {r}"
print(f"[OK] weak classification reward = {r}")

zl_resp = '{"classification": "zero_lag", "metrics": ["wPLI"]}'
r2 = metric_classification_reward("Metric: wPLI\nConnectivity value: 0.45", zl_resp)
assert r2 == 1.0, f"Expected 1.0, got {r2}"
print(f"[OK] zero_lag classification reward = {r2}")

pc_resp = '{"classification": "phase_coupled", "metrics": ["PLI"]}'
r3 = metric_classification_reward("Metric: PLI\nConnectivity value: 0.25", pc_resp)
assert r3 == 1.0, f"Expected 1.0, got {r3}"
print(f"[OK] phase_coupled classification reward = {r3}")

from fc_pipeline.research.grpo_dataset import build_grpo_dataset
ds = build_grpo_dataset(n_samples=20, seed=7, use_hf_datasets=False)
assert len(ds) == 20 and "prompt" in ds[0] and "chosen" in ds[0]
print(f"[OK] dataset generator: {len(ds)} records")
print("Sample prompt snippet:", ds[0]["prompt"][:80].replace("\n", " "))
print("Sample chosen:", ds[0]["chosen"])

from fc_pipeline.viz import __doc__ as vdoc
assert vdoc and "PyQt5" in vdoc
print("[OK] viz package import (no Qt required at init)")

from pathlib import Path
from fc_pipeline.viz.browser_viewer import build_plotly_figure, build_viewer_html
test_img = Path("outputs/heatmap_coh_chainlit_20261021_093300_001.png")
if test_img.exists():
    fig = build_plotly_figure(test_img, "Test Heatmap")
    assert fig is not None and len(fig.layout.images) == 1
    print("[OK] build_plotly_figure generated interactive figure with layout image")

    html = build_viewer_html(test_img, "Test Heatmap")
    assert "<!DOCTYPE html>" in html and "zoomIn" in html
    print("[OK] build_viewer_html generated self-contained viewer HTML")

print("\nAll checks passed.")

