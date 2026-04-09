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


def _source_label(file_url: str) -> str:
    """Return a short human-readable source name — no URL, no link."""
    import urllib.parse
    lower = file_url.lower()
    if "drive.google.com" in lower or "docs.google.com" in lower:
        return "Google Drive audio file"
    if "dropbox.com" in lower:
        path = urllib.parse.urlparse(file_url).path
        name = path.rstrip("/").split("/")[-1]
        return name or "Dropbox file"
    # Direct URL — extract the filename from the path
    path = urllib.parse.urlparse(file_url.split("?")[0]).path
    name = path.rstrip("/").split("/")[-1]
    return name or "Audio file"


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

    # Task-specific labels (vad vs breath)
    is_breath = stats.get("task") == "breath"
    col1_label = "Breath"    if is_breath else "Speech"
    col2_label = "Non-Breath" if is_breath else "Silence"
    col1_key   = "breath_pct"    if is_breath else "speech_pct"
    col2_key   = "nonbreath_pct" if is_breath else "silence_pct"
    col1_css   = "speech"  # reuse green for first column
    col2_css   = "silence"

    # Stats table rows
    model_rows = ""
    for name, m in models.items():
        sp = float(m.get(col1_key, 0))
        si = float(m.get(col2_key, 0))
        fr = m.get("frame_ms", 0)
        model_rows += (
            f"<tr><td>{name}</td>"
            f'<td class="{col1_css}">{_fmt_pct(sp)}</td>'
            f'<td class="{col2_css}">{_fmt_pct(si)}</td>'
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

    unan_speech  = stats.get("unanimous_breath_pct",    stats.get("unanimous_speech_pct",  "—"))
    unan_silence = stats.get("unanimous_nonbreath_pct", stats.get("unanimous_silence_pct", "—"))

    page_title   = "Breathing Detector" if is_breath else "VAD"
    table_header = (
        f"<tr><th>Detector</th><th>{col1_label}</th><th>{col2_label}</th><th>Frame</th></tr>"
        if is_breath else
        f"<tr><th>Model</th><th>{col1_label}</th><th>{col2_label}</th><th>Frame</th></tr>"
    )
    unan_label1 = "breath"    if is_breath else "speech"
    unan_label2 = "non-breath" if is_breath else "silence"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title} Results – {job_id}</title>
<style>{_RESULT_CSS}</style>
</head>
<body>
<a class="back" href="../">All results</a>
<h1>{page_title} Comparison</h1>
<p class="subtitle">
  Job <strong>{job_id}</strong> &nbsp;·&nbsp; {task.upper()}
  &nbsp;·&nbsp; {round(duration, 1)} s
  {f"&nbsp;·&nbsp; submitted {sub_at[:19].replace('T',' ')}" if sub_at else ""}
</p>

{"<div class='card'>" + chart_tag + "</div>" if chart_tag else ""}

<div class="card">
  <h2>{col1_label} / {col2_label} per {'Detector' if is_breath else 'Model'}</h2>
  <table>
    {table_header}
    {model_rows or "<tr><td colspan='4'>No data</td></tr>"}
  </table>
</div>

{"<div class='card'><h2>Pairwise Agreement</h2><table>" + agree_rows + "</table>" +
 f"<p style='margin-top:10px;font-size:.85rem'>All agree – {unan_label1}: <strong>{_fmt_pct(unan_speech)}</strong> &nbsp; {unan_label2}: <strong>{_fmt_pct(unan_silence)}</strong></p>" +
 "</div>" if names else ""}

<footer>
  Source: {_source_label(file_url)}<br>
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
.empty    { color: #9e9e9e; font-style: italic; padding: 24px 0; }
footer    { color: #9e9e9e; font-size: 0.75rem; margin-top: 24px; text-align: center; }
/* ── live status ── */
#live-banner { display: none; border-radius: 8px; padding: 14px 16px;
               margin-bottom: 16px; }
#live-banner.queued  { background: #fff8e1; border-left: 4px solid #f9a825; }
#live-banner.running { background: #e3f2fd; border-left: 4px solid #1565c0; }
#live-banner.failed  { background: #ffebee; border-left: 4px solid #c62828; }
.banner-row  { display: flex; align-items: center; gap: 10px; }
.banner-title{ font-weight: 600; font-size: 0.95rem; }
.banner-meta { font-size: 0.8rem; color: #555; margin-top: 4px; word-break: break-all; }
.badge { font-size: 0.7rem; font-weight: 700; padding: 2px 7px;
         border-radius: 12px; text-transform: uppercase; white-space: nowrap; }
.badge-queued  { background: #fff3e0; color: #e65100; }
.badge-running { background: #e3f2fd; color: #1565c0; }
.badge-failed  { background: #ffebee; color: #c62828; }
@keyframes spin { to { transform: rotate(360deg); } }
.spinner { display: inline-block; width: 16px; height: 16px; flex-shrink: 0;
           border: 2px solid #90caf9; border-top-color: #1565c0;
           border-radius: 50%; animation: spin 0.9s linear infinite; }
"""

_INDEX_JS = """
(function () {
  var POLL_MS = 10000;
  var banner  = document.getElementById('live-banner');
  var titleEl = document.getElementById('live-title');
  var metaEl  = document.getElementById('live-meta');
  var badgeEl = document.getElementById('live-badge');
  var spinEl  = document.getElementById('live-spinner');

  function poll() {
    fetch('status.json?t=' + Date.now())
      .then(function(r) { return r.json(); })
      .then(function(s) {
        var st = s.status || 'unknown';
        if (st === 'complete') {
          // Results are ready — reload to get updated job list
          location.reload();
          return;
        }
        // Show banner for queued / running / failed
        var task    = (s.task    || 'analysis').toUpperCase();
        var job_id  = s.job_id  || '';
        var url     = s.file_url || '';
        var started = s.started_at ? s.started_at.replace('T',' ').slice(0,19) + ' UTC' : '';
        titleEl.textContent = task + ' · ' + job_id;
        metaEl.textContent  = (started ? started + '  ' : '') + url.slice(0, 80) + (url.length > 80 ? '…' : '');
        badgeEl.textContent = st;
        badgeEl.className   = 'badge badge-' + st;
        spinEl.style.display = (st === 'running') ? 'inline-block' : 'none';
        banner.className    = st;
        banner.style.display = 'block';
        if (st !== 'failed') setTimeout(poll, POLL_MS);
      })
      .catch(function() { setTimeout(poll, POLL_MS * 1.5); });
  }

  poll();
})();
"""


def build_index_page(results_root: Path) -> str:
    jobs = []
    if results_root.exists():
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
        label    = f"{task.upper()} · {jid}"
        ts       = sub_at[:19].replace("T", " ") if sub_at else ""
        short_url = (file_url[:80] + "…") if len(file_url) > 80 else file_url
        job_cards += (
            f'<a class="job" href="results/{jid}/">'
            f'<div class="job-id">{label}</div>'
            f'<div class="job-meta">{ts} &nbsp; {short_url}</div>'
            f"</a>\n"
        )

    if not job_cards:
        job_cards = '<p class="empty">No completed results yet.</p>'

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
<!-- live status banner (shown/hidden by JS) -->
<div id="live-banner">
  <div class="banner-row">
    <span class="spinner" id="live-spinner"></span>
    <span class="banner-title" id="live-title"></span>
    <span class="badge" id="live-badge"></span>
  </div>
  <div class="banner-meta" id="live-meta"></div>
</div>
{job_cards}
<footer>Powered by Claude Code + GitHub Actions</footer>
<script>{_INDEX_JS}</script>
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
