"""Standalone launcher for MLflow UI tracking server."""

import os
import subprocess
import sys
from pathlib import Path

# Ensure MLflow allows file-based tracking store on MLflow >= 3.0
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

ROOT_DIR = Path(__file__).resolve().parent.parent

cmd = [
    sys.executable,
    "-m",
    "mlflow",
    "ui",
    "--backend-store-uri",
    "mlruns",
    "--host",
    "127.0.0.1",
    "--port",
    "5000",
    "--workers",
    "1",
]

if __name__ == "__main__":
    print(f"Starting MLflow UI server from: {ROOT_DIR}")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT_DIR))
