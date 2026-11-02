"""fc_pipeline.viz — interactive visualization utilities.

This package provides optional, on-demand interactive viewers for EEG plots.
It is intentionally isolated from the deterministic pipeline:

- ``eeg_viewer``: PyQt5 QMainWindow with zoom/pan/slide for EEG PNG plots.
- ``viewer_launcher``: Async subprocess launcher (Chainlit-safe).

Nothing in this package is imported by the Supervisor, graph, or deterministic
nodes. It is only used from ``chainlit_app.py`` when the user clicks
"Open Interactive Viewer".
"""
