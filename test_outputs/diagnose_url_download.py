"""Diagnose the PhysioNet dataset URL download failure (part of the viewer/GRPO fix task).

Read-only network diagnostics: DNS resolution, TCP/HTTPS connectivity, HTTP status,
redirect chain, headers, and a partial byte-range download probe.
"""
from __future__ import annotations

import socket
import sys
import traceback

URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"
HOST = "physionet.org"
WWW_HOST = "www.physionet.org"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_dns() -> None:
    section("1) DNS resolution")
    for host in (HOST, WWW_HOST):
        try:
            infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
            ips = sorted({i[4][0] for i in infos})
            print(f"  {host:28s} -> OK  {ips}")
        except Exception as exc:
            print(f"  {host:28s} -> FAIL {type(exc).__name__}: {exc}")


def check_tcp() -> None:
    section("2) TCP connect (port 443)")
    for host in (HOST, WWW_HOST):
        try:
            with socket.create_connection((host, 443), timeout=10):
                print(f"  {host:28s} -> TCP OK")
        except Exception as exc:
            print(f"  {host:28s} -> TCP FAIL {type(exc).__name__}: {exc}")


def check_requests() -> None:
    section("3) HTTPS GET via requests (stream, no full download)")
    try:
        import requests
    except ImportError:
        print("  requests NOT installed")
        return

    # Show whether a proxy env var is in play.
    import os
    proxies = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
    print(f"  proxy env vars: {proxies or '(none)'}")

    try:
        with requests.get(URL, stream=True, timeout=30, allow_redirects=True) as resp:
            print(f"  status        : {resp.status_code} {resp.reason}")
            print(f"  final url     : {resp.url}")
            print(f"  redirects     : {[r.status_code for r in resp.history]}")
            print(f"  content-type  : {resp.headers.get('Content-Type')}")
            print(f"  content-length: {resp.headers.get('Content-Length')}")
            print(f"  server        : {resp.headers.get('Server')}")
            first = next(resp.iter_content(chunk_size=4096), b"")
            print(f"  first bytes   : {first[:16]!r} (len={len(first)})")
            print(f"  edf magic?    : {first[:8] == b'0       '}")
    except Exception as exc:
        print(f"  requests FAILED: {type(exc).__name__}: {exc}")
        if exc.__cause__ is not None:
            print(f"  cause         : {type(exc.__cause__).__name__}: {exc.__cause__}")
        traceback.print_exc(limit=3)


def check_head() -> None:
    section("4) HEAD request")
    try:
        import requests
        resp = requests.head(URL, timeout=30, allow_redirects=True)
        print(f"  status: {resp.status_code} {resp.reason} -> {resp.url}")
    except Exception as exc:
        print(f"  HEAD FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Target: {URL}")
    check_dns()
    check_tcp()
    check_requests()
    check_head()
