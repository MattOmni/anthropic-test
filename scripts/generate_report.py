#!/usr/bin/env python3
"""
Generate HTML reports from VAD analysis results.

Two modes
---------
1. Single result page (default):
       python scripts/generate_report.py <results_dir>
   Reads  <results_dir>/stats.json, comparison.png, meta.json
   Writes <results_dir>/index.html

2. Index / listing page:
       python scripts/generate_report.py --index <docs/results_dir>
   Scans all subdirectories for meta.json
   Writes <docs/results_dir>/../index.html  (i.e. docs/index.html)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_img(path: str) -> str:
    """Return a data-URI string for the PNG at *path*."""
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    return f"data:image/png;base64,{data}"


def _fmt_pct(value) -> str:
    return f"{float(value):.1f}%"


def _read_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Single result page
# ---------------------------------------------------------------------------

_RESULT_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f5f5f5;
    color: #212121;
    padding: 16px;
    max-width: 960px;
    margin: 0 auto;
}
h1 { font-size: 1.3rem; margin-bottom: 4px; }
.subtitle { color: #757575; font-size: 0.85rem; margin-bottom: 20px; word-break: break-all; }
.card {
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.12);
    padding: 16px;
    margin-bottom: 16px;
}
.card h2 { font-size: 1rem; margin-bottom: 12px; color: #1565c0; }
img.chart { width: 100%; border-radius: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { background: #e3f2fd; padding: 8px 10px; text-align: left; }
td { padding: 7px 10px; border-bottom: 1px solid #eeeeee; }
tr:last-child td { border-bottom: none; }
.speech  { color: #2e7d32; font-weight: 600; }
.silence { color: #c62828; }
.agree-hi { color: #2e7d32; }
.agree-lo { color: #c62828; }
.back { display: inline-block; margin-bottom: 16px; color: #1565c0;
        text-decoration: none; font-size: 0.9rem; }
.back:before { content: "← "; }
footer { color: #9e9e9e; font-size: 0.75rem; margin-top: 24px; text-align: center; }
"""


def build_result_page(results_dir: Path) -> str:
    stats_path = results_dir / "stats.json"
    chart_path = results_dir / "comparison.png"
    meta_path  = results_dir / "meta.json"

    stats = _read_json(str(stats_path)) if stats_path.exists() else {}
    meta  = _read_json(str(meta_path))  if meta_path.exists()  else {}

    job_id   = meta.get("job_id",   results_dir.name)
    task     = meta.get("task",     "analysis")
    file_url = meta.get("file_url", "")
    sub_at   = meta.get("submitted_at", "")

    duration = stats.get("duration_s", 0)
    models   = stats.get("models",     {})
    agreement= stats.get("agreement",  {})

    # Chart as inline base64 (so the page is fully self-contained)
    chart_tag = ""
    if chart_path.exists():
        chart_tag = f'<img class="chart" src="{_b64_img(str(chart_path))}" alt="VAD Comparison Chart">'

    # Stats table rows
    model_rows = ""
    for name, m in models.items():
        sp = float(m.get("speech_pct",  0))
        si = float(m.get("silence_pct", 0))
        fr = m.get("frame_ms", 0)
        model_rows += (
            f"<tr><td>{name}</td>"
            f'<td class="speech">{_fmt_pct(sp)}</td>'
            f'<td class="silence">{_fmt_pct(si)}</td>'
            f"<td>{fr} ms</td></tr>\n"
        )

    # Agreement matrix
    agree_rows = ""
    names = list(models.keys())
    if len(names) > 1:
        header_cells = "".join(f"<th>{n}</th>" for n in names)
        agree_rows += f"<tr><th></th>{header_cells}</tr>\n"
        for ni in names:
            cells = ""
            for nj in names:
                val = agreement.get(ni, {}).get(nj, 100.0)
                css = "agree-hi" if float(val) >= 80 else "agree-lo"
                cells += f'<td class="{css}">{_fmt_pct(val)}</td>'
            agree_rows += f"<tr><td>{ni}</td>{cells}</tr>\n"

    unan_speech  = stats.get("unanimous_speech_pct",  "—")
    unan_silence = stats.get("unanimous_silence_pct", "—")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VAD Results – {job_id}</title>
<style>{_RESULT_CSS}</style>
</head>
<body>
<a class="back" href="../">All results</a>
<h1>VAD Comparison</h1>
<p class="subtitle">
  Job <strong>{job_id}</strong> &nbsp;·&nbsp; {task.upper()}
  &nbsp;·&nbsp; {round(duration, 1)} s
  {f"&nbsp;·&nbsp; submitted {sub_at[:19].replace('T',' ')}" if sub_at else ""}
</p>

{"<div class='card'>" + chart_tag + "</div>" if chart_tag else ""}

<div class="card">
  <h2>Speech / Silence per Model</h2>
  <table>
    <tr><th>Model</th><th>Speech</th><th>Silence</th><th>Frame</th></tr>
    {model_rows or "<tr><td colspan='4'>No data</td></tr>"}
  </table>
</div>

{"<div class='card'><h2>Pairwise Agreement</h2><table>" + agree_rows + "</table>" +
 f"<p style='margin-top:10px;font-size:.85rem'>All agree – speech: <strong>{_fmt_pct(unan_speech)}</strong> &nbsp; silence: <strong>{_fmt_pct(unan_silence)}</strong></p>" +
 "</div>" if names else ""}

<footer>
  Source: <a href="{file_url}" style="color:#9e9e9e">{file_url[:80]}{"…" if len(file_url)>80 else ""}</a><br>
  Generated by Claude Code · MattOmni/anthropic-test
</footer>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Index / listing page
# ---------------------------------------------------------------------------

_INDEX_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f5f5f5; color: #212121;
    padding: 16px; max-width: 760px; margin: 0 auto;
}
h1 { font-size: 1.4rem; margin-bottom: 6px; }
.tagline { color: #757575; font-size: 0.85rem; margin-bottom: 20px; }
.job {
    background: #fff; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.12);
    padding: 14px 16px; margin-bottom: 12px;
    text-decoration: none; color: inherit; display: block;
}
.job:hover { box-shadow: 0 2px 8px rgba(0,0,0,.18); }
.job-id   { font-weight: 600; font-size: 0.95rem; color: #1565c0; }
.job-meta { font-size: 0.8rem; color: #757575; margin-top: 3px; word-break: break-all; }
.empty { color: #9e9e9e; font-style: italic; padding: 24px 0; }
footer { color: #9e9e9e; font-size: 0.75rem; margin-top: 24px; text-align: center; }
"""


def build_index_page(results_root: Path) -> str:
    jobs = []
    for d in sorted(results_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = _read_json(str(meta_path))
        except Exception:
            continue
        jobs.append(meta)

    job_cards = ""
    for m in jobs:
        jid      = m.get("job_id",       "—")
        task     = m.get("task",          "—")
        file_url = m.get("file_url",      "")
        sub_at   = m.get("submitted_at",  "")
        status   = m.get("status",        "")
        label = f"{task.upper()} · {jid}"
        ts    = sub_at[:19].replace("T", " ") if sub_at else ""
        short_url = (file_url[:80] + "…") if len(file_url) > 80 else file_url
        job_cards += (
            f'<a class="job" href="results/{jid}/">'
            f'<div class="job-id">{label}</div>'
            f'<div class="job-meta">{ts} &nbsp; {short_url}</div>'
            f"</a>\n"
        )

    if not job_cards:
        job_cards = '<p class="empty">No results yet. Submit a job via Claude Code.</p>'

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analysis Results</title>
<style>{_INDEX_CSS}</style>
</head>
<body>
<h1>Analysis Results</h1>
<p class="tagline">MattOmni/anthropic-test · Updated {now}</p>
{job_cards}
<footer>Powered by Claude Code + GitHub Actions</footer>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target_dir", help="Results dir (single) or results root (--index)")
    p.add_argument("--index", action="store_true",
                   help="Build the index page for a results root directory")
    args = p.parse_args()

    target = Path(args.target_dir).resolve()

    if args.index:
        # Build docs/index.html from all results under docs/results/
        html = build_index_page(target)
        out  = target.parent / "index.html"
        out.write_text(html, encoding="utf-8")
        print(f"Index page written → {out}")
    else:
        html = build_result_page(target)
        out  = target / "index.html"
        out.write_text(html, encoding="utf-8")
        print(f"Result page written → {out}")


if __name__ == "__main__":
    main()
