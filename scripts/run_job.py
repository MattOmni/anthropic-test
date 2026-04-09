#!/usr/bin/env python3
"""
Orchestrator called by the GitHub Actions workflow.

Reads jobs/latest.json, downloads the file, dispatches to the correct
analysis script, then calls generate_report.py to produce the HTML page.

jobs/latest.json schema
-----------------------
{
  "job_id":       "20260409-143022",
  "task":         "vad",
  "file_url":     "https://...",
  "max_duration": 120,         # optional, seconds
  "submitted_at": "2026-04-09T14:30:22Z"
}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
JOBS_FILE   = REPO_ROOT / "jobs" / "latest.json"
RESULTS_DIR = REPO_ROOT / "docs" / "results"

SUPPORTED_TASKS = {"vad", "breath"}


def load_job() -> dict:
    if not JOBS_FILE.exists():
        sys.exit(f"Job file not found: {JOBS_FILE}")
    with open(JOBS_FILE) as fh:
        job = json.load(fh)
    required = {"job_id", "task", "file_url"}
    missing = required - job.keys()
    if missing:
        sys.exit(f"Job manifest missing fields: {missing}")
    if job["task"] not in SUPPORTED_TASKS:
        sys.exit(f"Unknown task '{job['task']}'. Supported: {SUPPORTED_TASKS}")
    return job


def run(cmd: list[str], **kwargs) -> None:
    """Run a subprocess and raise on failure."""
    print("$", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def run_vad(job: dict, audio_path: str, results_dir: Path) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "compare_vad.py"),
        audio_path,
        "--results-dir", str(results_dir),
    ]
    if job.get("max_duration"):
        cmd += ["--max-duration", str(job["max_duration"])]
    run(cmd)


def run_breath(job: dict, audio_path: str, results_dir: Path) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "compare_breath.py"),
        audio_path,
        "--results-dir", str(results_dir),
    ]
    if job.get("max_duration"):
        cmd += ["--max-duration", str(job["max_duration"])]
    run(cmd)


def write_status(job: dict, status: str, extra: dict | None = None) -> None:
    """Write docs/status.json so the GitHub Pages index shows live progress."""
    from datetime import datetime, timezone
    payload = {
        "job_id":       job["job_id"],
        "task":         job["task"],
        "file_url":     job.get("file_url", ""),
        "submitted_at": job.get("submitted_at", ""),
        "status":       status,
        "updated_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        payload.update(extra)
    status_path = REPO_ROOT / "docs" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"status.json → {status}")


def main() -> None:
    job = load_job()
    job_id      = job["job_id"]
    task        = job["task"]
    file_url    = job["file_url"]

    print(f"\n{'='*60}")
    print(f"  Job: {job_id}  |  Task: {task}")
    print(f"  File: {file_url}")
    print(f"{'='*60}\n")

    # Mark job as running immediately so the Pages index shows a live spinner
    from datetime import datetime, timezone
    write_status(job, "running",
                 {"started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})

    # ── Determine output directory ────────────────────────────────────────
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download input file ───────────────────────────────────────────────
    suffix = Path(file_url.split("?")[0]).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_path = tmp.name

    try:
        run([sys.executable, str(REPO_ROOT / "scripts" / "download_file.py"),
             file_url, audio_path])

        # ── Dispatch to task runner ───────────────────────────────────────
        if task == "vad":
            run_vad(job, audio_path, out_dir)
        elif task == "breath":
            run_breath(job, audio_path, out_dir)

    except SystemExit:
        write_status(job, "failed")
        raise
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    # ── Write metadata ────────────────────────────────────────────────────
    meta = {
        "job_id":       job_id,
        "task":         task,
        "file_url":     file_url,
        "submitted_at": job.get("submitted_at", ""),
        "status":       "success",
    }
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    # ── Generate HTML report ──────────────────────────────────────────────
    run([sys.executable, str(REPO_ROOT / "scripts" / "generate_report.py"),
         str(out_dir)])

    # ── Rebuild the index page ────────────────────────────────────────────
    run([sys.executable, str(REPO_ROOT / "scripts" / "generate_report.py"),
         "--index", str(RESULTS_DIR)])

    # Mark complete — JS on the index page will auto-reload when it sees this
    write_status(job, "complete")

    # Export job_id for the git commit message in the workflow
    print(f"\n::set-output name=job_id::{job_id}")
    # Also set as env var for the commit step
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a") as fh:
            fh.write(f"JOB_ID={job_id}\n")

    print(f"\nDone. Results at: docs/results/{job_id}/")


if __name__ == "__main__":
    main()
