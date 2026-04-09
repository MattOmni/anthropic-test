#!/usr/bin/env python3
"""
Download a file from a URL to a local destination path.

Supported sources
-----------------
* Google Drive  – any share link (uses gdown)
* Dropbox       – rewrites ?dl=0 to ?dl=1
* Direct URL    – any plain HTTPS/HTTP link (streamed with requests)

Usage
-----
    python scripts/download_file.py <url> <dest_path>
"""

from __future__ import annotations

import os
import sys
import urllib.parse


def is_google_drive(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def is_dropbox(url: str) -> bool:
    return "dropbox.com" in url


def normalise_dropbox(url: str) -> str:
    """Force Dropbox direct-download mode."""
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    qs["dl"] = ["1"]
    new_query = urllib.parse.urlencode({k: v[0] for k, v in qs.items()})
    return parsed._replace(query=new_query).geturl()


def download_drive(url: str, dest: str) -> None:
    try:
        import gdown  # type: ignore[import]
    except ImportError:
        sys.exit("gdown is not installed.  Run: pip install gdown")

    print(f"Google Drive download → {dest}")
    result = gdown.download(url, dest, quiet=False, fuzzy=True)
    if result is None:
        sys.exit("gdown failed to download the file.")


def download_direct(url: str, dest: str) -> None:
    try:
        import requests  # type: ignore[import]
    except ImportError:
        sys.exit("requests is not installed.  Run: pip install requests")

    print(f"Direct download → {dest}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}%  {downloaded:,}/{total:,} bytes", end="", flush=True)
    if total:
        print()


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

    if is_google_drive(url):
        download_drive(url, dest)
    elif is_dropbox(url):
        download_direct(normalise_dropbox(url), dest)
    else:
        download_direct(url, dest)

    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        sys.exit(f"Download produced no output at {dest}")

    size_mb = os.path.getsize(dest) / 1_048_576
    print(f"Saved: {dest} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python scripts/download_file.py <url> <dest_path>")
    download(sys.argv[1], sys.argv[2])
