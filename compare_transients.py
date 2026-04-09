#!/usr/bin/env python3
"""
Compare three pop/click/plosive detectors on an audio file.

Usage
-----
    python compare_transients.py [audio_file] [--output transient_comparison.png]
                                 [--results-dir DIR] [--max-duration SECONDS]

The script
  1. Loads the audio to 16 kHz mono (for waveform display).
  2. Runs each available transient detector.
  3. Saves a multi-panel comparison plot.
  4. Prints a statistics summary.

Requirements
------------
    pip install -r requirements.txt
    # ffmpeg must be on PATH for MP3 / M4A input
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SAMPLE_RATE = 16_000
THRESHOLD   = 0.5


# ---------------------------------------------------------------------------
# Audio loading (for waveform panel)
# ---------------------------------------------------------------------------

def load_audio_waveform(path: str, max_duration: float | None = None) -> np.ndarray:
    try:
        import librosa
    except ImportError:
        sys.exit("librosa is not installed.  Run: pip install librosa")

    print(f"Loading: {path}")
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
    audio = audio.astype(np.float32)
    secs = len(audio) / SAMPLE_RATE
    print(f"  {secs:.1f} s  |  {len(audio):,} samples @ {SAMPLE_RATE} Hz")
    return audio


# ---------------------------------------------------------------------------
# Detector loader
# ---------------------------------------------------------------------------

def build_detectors() -> list:
    from transient_models import (
        KurtosisClickDetector,
        SpectralFluxDetector,
        SubBassBurstDetector,
    )

    candidates = [
        ("Kurtosis Click Detector", KurtosisClickDetector, {}),
        ("Spectral Flux Detector",  SpectralFluxDetector,  {}),
        ("Sub-Bass Burst Detector", SubBassBurstDetector,  {}),
    ]

    detectors = []
    for label, cls, kwargs in candidates:
        try:
            d = cls(**kwargs)
            detectors.append(d)
            print(f"  Loaded {d.name}")
        except Exception as exc:
            print(f"  WARNING – could not load {label}: {exc}")

    return detectors


def run_all(
    audio_path: str,
    detectors: list,
    max_duration: float | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for det in detectors:
        print(f"\nRunning {det.name} …")
        try:
            times, probs = det.process(audio_path, sample_rate=SAMPLE_RATE)

            if max_duration is not None and len(times) > 0:
                mask  = times <= max_duration
                times = times[mask]
                probs = probs[mask]

            transient_secs = float(np.sum(probs >= THRESHOLD)) * (
                (times[1] - times[0]) if len(times) > 1 else 0.0
            )
            total_secs = times[-1] if len(times) > 0 else 0.0
            if total_secs > 0:
                print(
                    f"  Transient: {transient_secs:.1f} s / {total_secs:.1f} s "
                    f"({transient_secs / total_secs:.1%})"
                )
            results[det.name] = (times, probs)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[det.name] = (np.array([]), np.array([]))

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    valid = {k: v for k, v in results.items() if len(v[0]) > 0}
    if not valid:
        print("No successful results.")
        return

    all_times = [v[0] for v in valid.values()]
    t_lo = max(t[0]  for t in all_times)
    t_hi = min(t[-1] for t in all_times)
    if t_lo >= t_hi:
        print("Time ranges don't overlap – skipping agreement matrix.")
        return

    n_points = max(2, int((t_hi - t_lo) * 1000))
    t_grid = np.linspace(t_lo, t_hi, n_points)

    names = list(valid.keys())
    decisions: Dict[str, np.ndarray] = {}
    for name, (times, probs) in valid.items():
        interp = np.interp(t_grid, times, probs)
        decisions[name] = interp >= THRESHOLD

    col_w = max(len(n) for n in names) + 2
    num_w = 10

    print("\n" + "═" * 60)
    print("  POP / CLICK / PLOSIVE DETECTOR SUMMARY")
    print("═" * 60)

    print(f"\n  {'Detector':<{col_w}}  {'Transient':>{num_w}}  {'Clean':>{num_w}}")
    print("  " + "-" * (col_w + num_w * 2 + 4))
    for name in names:
        t_pct = float(np.mean(decisions[name])) * 100
        print(f"  {name:<{col_w}}  {t_pct:>{num_w}.1f}%  {100 - t_pct:>{num_w}.1f}%")

    print(f"\n  Pairwise agreement (% of frames both detectors agree on)")
    header = f"  {'':>{col_w}}"
    for n in names:
        header += f"  {n[:num_w]:>{num_w}}"
    print(header)
    print("  " + "-" * (col_w + num_w * len(names) + 4))
    for name_i in names:
        row = f"  {name_i:<{col_w}}"
        for name_j in names:
            agree = float(np.mean(decisions[name_i] == decisions[name_j])) * 100
            row += f"  {agree:>{num_w}.1f}%"
        print(row)

    if len(names) > 1:
        all_arr = np.stack(list(decisions.values()), axis=0)
        unan_t  = float(np.mean(all_arr.all(axis=0)))  * 100
        unan_c  = float(np.mean(~all_arr.any(axis=0))) * 100
        print(f"\n  All agree – transient: {unan_t:.1f}%")
        print(f"  All agree – clean:     {unan_c:.1f}%")

    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

COLORS = {
    "transient": "#e65100",   # deep orange for transients
    "clean":     "#e0e0e0",   # light grey for clean sections
    "waveform":  "#1565c0",
    "line":      "#212121",
    "thresh":    "#9e9e9e",
}


def _add_transient_bands(ax, times: np.ndarray, probs: np.ndarray) -> None:
    ax.fill_between(
        times, probs, THRESHOLD,
        where=(probs >= THRESHOLD),
        interpolate=True,
        color=COLORS["transient"], alpha=0.60, label="Transient",
    )
    ax.fill_between(
        times, probs, THRESHOLD,
        where=(probs < THRESHOLD),
        interpolate=True,
        color=COLORS["clean"], alpha=0.30, label="Clean",
    )
    ax.plot(times, probs, color=COLORS["line"], linewidth=0.7, alpha=0.85)
    ax.axhline(THRESHOLD, color=COLORS["thresh"], linestyle="--", linewidth=0.8)


def plot_comparison(
    audio: np.ndarray,
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
) -> None:
    n_rows = 1 + len(results)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(18, 2.8 * n_rows),
        sharex=True,
        gridspec_kw={"hspace": 0.45},
    )

    duration = len(audio) / SAMPLE_RATE
    t_audio  = np.linspace(0, duration, len(audio))

    # ── Waveform ──────────────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(t_audio, audio, color=COLORS["waveform"], linewidth=0.4, alpha=0.8)
    ax0.set_ylabel("Amplitude", fontsize=9)
    ax0.set_title("Audio Waveform", fontsize=10, fontweight="bold")
    ax0.set_xlim(0, duration)
    ax0.set_ylim(-1.05, 1.05)
    ax0.grid(axis="x", linewidth=0.3, alpha=0.5)

    # ── Detector panels ───────────────────────────────────────────────────
    for idx, (det_name, (times, probs)) in enumerate(results.items()):
        ax = axes[idx + 1]
        ax.set_title(det_name, fontsize=10, fontweight="bold")
        ax.set_ylabel("P(transient)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)

        if len(times) == 0:
            ax.text(
                0.5, 0.5, "Detector failed to run",
                transform=ax.transAxes,
                ha="center", va="center",
                color="red", fontsize=12,
            )
            continue

        _add_transient_bands(ax, times, probs)

        t_pct    = float(np.mean(probs >= THRESHOLD)) * 100
        frame_ms = (times[1] - times[0]) * 1000 if len(times) > 1 else 0.0
        ax.text(
            0.005, 0.87,
            f"Transient {t_pct:.1f}%  |  frame {frame_ms:.0f} ms",
            transform=ax.transAxes,
            fontsize=8, color="#bf360c",
        )

        legend_patches = [
            mpatches.Patch(color=COLORS["transient"], alpha=0.7, label="Transient"),
            mpatches.Patch(color=COLORS["clean"],     alpha=0.55, label="Clean"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.7)

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    fig.suptitle(
        "Pop / Click / Plosive Detector Comparison  –  Kurtosis · Spectral Flux · Sub-Bass Burst",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Stats JSON
# ---------------------------------------------------------------------------

def save_stats_json(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    audio_duration: float,
    output_path: str,
) -> None:
    valid = {k: v for k, v in results.items() if len(v[0]) > 0}
    stats: dict = {
        "task": "pops",
        "duration_s": round(audio_duration, 2),
        "models": {},
    }

    if not valid:
        with open(output_path, "w") as fh:
            json.dump(stats, fh, indent=2)
        return

    all_times = [v[0] for v in valid.values()]
    t_lo = max(t[0]  for t in all_times)
    t_hi = min(t[-1] for t in all_times)
    n_pts = max(2, int((t_hi - t_lo) * 1000))
    t_grid = np.linspace(t_lo, t_hi, n_pts)

    decisions: Dict[str, np.ndarray] = {}
    for name, (times, probs) in valid.items():
        interp = np.interp(t_grid, times, probs)
        decisions[name] = interp >= THRESHOLD
        t_pct = float(np.mean(interp >= THRESHOLD)) * 100
        stats["models"][name] = {
            "transient_pct": round(t_pct, 1),
            "clean_pct":     round(100 - t_pct, 1),
            "frame_ms":      round((valid[name][0][1] - valid[name][0][0]) * 1000, 1)
                             if len(valid[name][0]) > 1 else 0,
        }

    agreement: dict = {}
    names = list(decisions.keys())
    for ni in names:
        agreement[ni] = {}
        for nj in names:
            agreement[ni][nj] = round(
                float(np.mean(decisions[ni] == decisions[nj])) * 100, 1
            )
    stats["agreement"] = agreement

    if len(names) > 1:
        all_arr = np.stack(list(decisions.values()), axis=0)
        stats["unanimous_transient_pct"] = round(float(np.mean(all_arr.all(axis=0)))  * 100, 1)
        stats["unanimous_clean_pct"]     = round(float(np.mean(~all_arr.any(axis=0))) * 100, 1)

    with open(output_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Stats saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Kurtosis, Spectral Flux, and Sub-Bass pop/click detectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("audio_file", nargs="?", default="audio/sample.mp3")
    p.add_argument("--output",   "-o", default="transient_comparison.png")
    p.add_argument(
        "--results-dir", default=None, metavar="DIR",
        help="Write comparison.png + stats.json here (overrides --output)",
    )
    p.add_argument(
        "--max-duration", "-d", type=float, default=None, metavar="SECONDS",
        help="Truncate audio to this many seconds",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.audio_file):
        sys.exit(f"Audio file not found: {args.audio_file}")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        png_path  = os.path.join(args.results_dir, "comparison.png")
        json_path = os.path.join(args.results_dir, "stats.json")
    else:
        png_path  = args.output
        json_path = None

    audio = load_audio_waveform(args.audio_file, max_duration=args.max_duration)

    print("\nLoading transient detectors …")
    detectors = build_detectors()

    if not detectors:
        sys.exit("No detectors could be loaded.")

    results = run_all(args.audio_file, detectors, max_duration=args.max_duration)

    print_stats(results)
    plot_comparison(audio, results, png_path)

    if json_path:
        save_stats_json(results, len(audio) / SAMPLE_RATE, json_path)


if __name__ == "__main__":
    main()
