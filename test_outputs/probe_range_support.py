"""Probe: does PhysioNet support HTTP Range (resumable) downloads?

Part of diagnosing the mid-stream IncompleteRead failure. Reads a few
fixed-size ranges over separate connections and reports status codes.
"""
from __future__ import annotations

import requests

URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"


def main() -> None:
    print("HEAD:")
    h = requests.head(URL, timeout=30, allow_redirects=True)
    print(f"  {h.status_code} {h.reason}")
    print(f"  Accept-Ranges : {h.headers.get('Accept-Ranges')}")
    print(f"  Content-Length: {h.headers.get('Content-Length')}")
    print(f"  ETag          : {h.headers.get('ETag')}")

    total = int(h.headers.get("Content-Length") or 0)

    print("\nRange requests (separate connections):")
    got = 0
    start = 0
    for i in range(6):
        headers = {"Range": f"bytes={start}-{min(start + 199999, total - 1)}"}
        with requests.get(URL, headers=headers, timeout=(15, 60), stream=True) as r:
            data = b"".join(r.iter_content(chunk_size=65536))
        print(
            f"  #{i}: req={headers['Range']:>26s}  status={r.status_code}  "
            f"got={len(data):>7d}B  Content-Range={r.headers.get('Content-Range')}"
        )
        got += len(data)
        start += len(data)
        if start >= total:
            break

    print(f"\n  bytes accumulated: {got} / {total}  -> {'COMPLETE' if got == total else 'INCOMPLETE'}")


if __name__ == "__main__":
    main()
