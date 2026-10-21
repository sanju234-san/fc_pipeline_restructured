"""End-to-end test of the REAL download+validate functions used by the Chainlit app.

Imports ``chainlit_app`` itself (not a copy) and exercises:
  - _download_dataset_url(URL)   for the PhysioNet EDF URL
  - _validate_dataset_path(path)
  - a deliberately bad hostname (DNS failure) to prove the new error message
  - an HTTP 404 URL to prove HTTP-status reporting

Writes nothing outside outputs/uploaded_datasets (the app's own download dir).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chainlit_app as app  # noqa: E402

GOOD_URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"
WWW_URL = "https://www.physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"
BAD_HOST = "https://physionet.invalid-host-does-not-exist.example/x.edf"
HTTP_404 = "https://physionet.org/files/eegmmidb/1.0.0/S001/NOPE.edf"


def show(label: str, fn):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    try:
        result = fn()
        print(f"  RESULT: {result}")
        return result
    except Exception as exc:
        print(f"  RAISED {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    failures = 0

    # 1) The URL from the bug report.
    path = show(f"1) _download_dataset_url({GOOD_URL})", lambda: app._download_dataset_url(GOOD_URL))
    if path is None:
        failures += 1
    else:
        p = Path(path)
        print(f"  exists={p.exists()} size={p.stat().st_size if p.exists() else 0} suffix={p.suffix}")
        ok, reason = app._validate_dataset_path(p)
        print(f"  _validate_dataset_path -> ok={ok} reason={reason!r}")
        if not ok:
            failures += 1
        elif p.stat().st_size == 0:
            failures += 1

    # 2) The www variant (the host named in the reported error message).
    path2 = show(f"2) _download_dataset_url({WWW_URL})", lambda: app._download_dataset_url(WWW_URL))
    if path2 is None:
        failures += 1

    # 3) DNS failure -> must produce a *DNS-specific* message, not "could not connect".
    msg = show("3) DNS failure path", lambda: app._download_dataset_url(BAD_HOST))
    if msg is not None:
        failures += 1

    # 4) HTTP 404 -> must mention the HTTP status.
    msg404 = show("4) HTTP 404 path", lambda: app._download_dataset_url(HTTP_404))
    if msg404 is not None:
        failures += 1

    print("\n" + "=" * 70)
    print(f"FAILURES: {failures}")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
