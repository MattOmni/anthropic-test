#!/usr/bin/env python3
"""
Download a LibriVox audiobook chapter from archive.org for VAD testing.
Saves the file to audio/sample.mp3.
"""

import os
import sys
import requests

AUDIO_DIR = "audio"
OUTPUT_FILE = os.path.join(AUDIO_DIR, "sample.mp3")
ARCHIVE_BASE = "https://archive.org"

# Well-known LibriVox identifiers and their chapter 1 filenames.
# Used as candidates before falling back to live search.
KNOWN_CANDIDATES = [
    (
        "alice_in_wonderland_librivox",
        "alice_in_wonderland_ch_01_carroll.mp3",
        "Alice's Adventures in Wonderland – Ch.1",
    ),
    (
        "alices_adventures_in_wonderland_librivox",
        "alicesadventuresinwonderland_01_carroll.mp3",
        "Alice's Adventures in Wonderland (alt) – Ch.1",
    ),
    (
        "pride_and_prejudice_librivox",
        "prideandprejudice_01_austen.mp3",
        "Pride and Prejudice – Ch.1",
    ),
]


def download_file(url: str, dest: str) -> None:
    """Stream-download *url* to *dest*, printing a simple progress indicator."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}%  {downloaded:,} / {total:,} bytes", end="", flush=True)
    print()


def try_known_candidates() -> bool:
    """Try each hard-coded candidate URL and return True on first success."""
    for identifier, filename, label in KNOWN_CANDIDATES:
        url = f"{ARCHIVE_BASE}/download/{identifier}/{filename}"
        print(f"Trying: {label}")
        print(f"  {url}")
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True)
            if resp.status_code == 200:
                print(f"  Found! Downloading…")
                download_file(url, OUTPUT_FILE)
                print(f"  Saved to {OUTPUT_FILE}")
                return True
            else:
                print(f"  HTTP {resp.status_code} – skipping")
        except requests.RequestException as exc:
            print(f"  Error: {exc} – skipping")
    return False


def search_archive_org() -> bool:
    """
    Query archive.org's search API for a LibriVox fiction audiobook, then
    inspect its file list and download the first MP3 chapter found.
    Returns True on success.
    """
    search_url = f"{ARCHIVE_BASE}/advancedsearch.php"
    params = {
        "q": 'subject:"librivox" AND mediatype:audio AND subject:"fiction"',
        "fl[]": ["identifier", "title"],
        "rows": 20,
        "output": "json",
        "sort[]": "downloads desc",
    }
    print("Searching archive.org for a LibriVox recording…")
    try:
        resp = requests.get(search_url, params=params, timeout=30)
        resp.raise_for_status()
        docs = resp.json()["response"]["docs"]
    except Exception as exc:
        print(f"  Search failed: {exc}")
        return False

    for doc in docs:
        identifier = doc.get("identifier", "")
        title = doc.get("title", identifier)
        print(f"  Checking: {title} ({identifier})")

        try:
            meta_resp = requests.get(f"{ARCHIVE_BASE}/metadata/{identifier}", timeout=20)
            meta_resp.raise_for_status()
            files = meta_resp.json().get("files", [])
        except Exception as exc:
            print(f"    Metadata error: {exc}")
            continue

        mp3_files = [f for f in files if f.get("name", "").endswith(".mp3")]
        if not mp3_files:
            print("    No MP3 files found, skipping")
            continue

        # Prefer a chapter-1 file; fall back to the first MP3.
        chapter_file = next(
            (f for f in mp3_files if "_01_" in f["name"] or "_1_" in f["name"]),
            mp3_files[0],
        )

        url = f"{ARCHIVE_BASE}/download/{identifier}/{chapter_file['name']}"
        print(f"    Downloading: {chapter_file['name']}")
        try:
            download_file(url, OUTPUT_FILE)
            print(f"  Saved to {OUTPUT_FILE}")
            return True
        except Exception as exc:
            print(f"    Download error: {exc}")
            continue

    return False


def main() -> None:
    os.makedirs(AUDIO_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / 1_048_576
        print(f"Sample already exists: {OUTPUT_FILE} ({size_mb:.1f} MB)")
        print("Delete it and re-run to download a fresh copy.")
        return

    success = try_known_candidates() or search_archive_org()

    if not success:
        print("\nERROR: Could not download a LibriVox sample automatically.")
        print("Place any audio file at audio/sample.mp3 and re-run compare_vad.py.")
        sys.exit(1)

    size_mb = os.path.getsize(OUTPUT_FILE) / 1_048_576
    print(f"\nDownload complete: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print("Run:  python compare_vad.py")


if __name__ == "__main__":
    main()
