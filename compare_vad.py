#!/usr/bin/env python3
"""
Compare Silero VAD, WebRTC VAD, and TEN VAD on an audio file.

Usage
-----
    python compare_vad.py [audio_file] [--output comparison.png]
                          [--max-duration SECONDS] [--webrtc-mode 0..3]

The script
  1. Loads and resamples the audio to 16 kHz mono.
  2. Runs each available VAD model frame-by-frame.
  3. Saves a multi-panel comparison plot.
  4. Prints a statistics summary (speech ratio, pairwise agreement).

Requirements
------------
    pip install -r requirements.txt
    # ffmpeg must be on PATH for MP3 / M4A input
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SAMPLE_RATE = 16_000
THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, max_duration: float | None = None) -> np.ndarray:
    """
    Load *path* (WAV, MP3, FLAC, …) as a mono float32 array at 16 kHz.
    Requires librosa and, for MP3 files, ffmpeg on PATH.
    """
    try:
        import librosa  # type: ignore[import]
    except ImportError:
        sys.exit("librosa is not installed.  Run: pip install librosa")

    duration_arg = max_duration  # librosa's `duration` keyword
    print(f"Loading: {path}")
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=duration_arg)
    audio = audio.astype(np.float32)

    secs = len(audio) / SAMPLE_RATE
    print(f"  {secs:.1f} s  |  {len(audio):,} samples @ {SAMPLE_RATE} Hz")
    return audio


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def build_models(webrtc_mode: int) -> list:
    """Instantiate each VAD model, silently skipping any that fail to load."""
    from vad_models import SileroVAD, WebRTCVAD, TenVAD

    candidates = [
        ("Silero VAD", SileroVAD, {}),
        ("WebRTC VAD", WebRTCVAD, {"aggressiveness": webrtc_mode}),
        ("TEN VAD",   TenVAD,    {}),
    ]

    models = []
    for label, cls, kwargs in candidates:
        try:
            m = cls(**kwargs)
            models.append(m)
            print(f"  Loaded {m.name}")
        except Exception as exc:
            print(f"  WARNING – could not load {label}: {exc}")

    return models


def run_all(audio: np.ndarray, models: list) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Run each model and collect (times, probs) pairs."""
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for model in models:
        print(f"\nRunning {model.name} …")
        try:
            times, probs = model.process(audio, SAMPLE_RATE)
            speech_secs = float(np.sum(probs >= THRESHOLD)) * (
                (times[1] - times[0]) if len(times) > 1 else 0.0
            )
            total_secs = len(audio) / SAMPLE_RATE
            print(
                f"  Speech: {speech_secs:.1f} s / {total_secs:.1f} s "
                f"({speech_secs / total_secs:.1%})"
            )
            results[model.name] = (times, probs)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[model.name] = (np.array([]), np.array([]))

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    valid = {k: v for k, v in results.items() if len(v[0]) > 0}
    if not valid:
        print("No successful results – nothing to compare.")
        return

    # Interpolate all models onto a shared 1 ms grid
    all_times = [v[0] for v in valid.values()]
    t_lo = max(t[0]  for t in all_times)
    t_hi = min(t[-1] for t in all_times)
    if t_lo >= t_hi:
        print("Time ranges don't overlap – skipping agreement matrix.")
        return

    n_points = max(2, int((t_hi - t_lo) * 1000))  # ~1 ms resolution
    t_grid = np.linspace(t_lo, t_hi, n_points)

    names = list(valid.keys())
    decisions: Dict[str, np.ndarray] = {}
    for name, (times, probs) in valid.items():
        interp = np.interp(t_grid, times, probs)
        decisions[name] = interp >= THRESHOLD

    col_w = max(len(n) for n in names) + 2
    num_w = 10

    print("\n" + "═" * 60)
    print("  VAD COMPARISON SUMMARY")
    print("═" * 60)

    print(f"\n  {'Model':<{col_w}}  {'Speech':>{num_w}}  {'Silence':>{num_w}}")
    print("  " + "-" * (col_w + num_w * 2 + 4))
    for name in names:
        speech_pct = float(np.mean(decisions[name])) * 100
        print(f"  {name:<{col_w}}  {speech_pct:>{num_w}.1f}%  {100 - speech_pct:>{num_w}.1f}%")

    print(f"\n  Pairwise agreement (% of frames both models agree on)")
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

    # Unanimous speech / silence
    if len(names) > 1:
        all_speech   = np.stack(list(decisions.values()), axis=0).all(axis=0)
        all_silence  = ~np.stack(list(decisions.values()), axis=0).any(axis=0)
        unanimous_s  = float(np.mean(all_speech))  * 100
        unanimous_si = float(np.mean(all_silence)) * 100
        print(f"\n  All models agree – speech:  {unanimous_s:.1f}%")
        print(f"  All models agree – silence: {unanimous_si:.1f}%")

    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

COLORS = {
    "speech":  "#4caf50",
    "silence": "#ef5350",
    "waveform": "#1565c0",
    "line":    "#212121",
    "thresh":  "#9e9e9e",
}


def _add_speech_bands(ax, times: np.ndarray, probs: np.ndarray) -> None:
    """Fill green/red bands and draw the probability trace."""
    ax.fill_between(
        times, probs, THRESHOLD,
        where=(probs >= THRESHOLD),
        interpolate=True,
        color=COLORS["speech"],  alpha=0.55, label="Speech",
    )
    ax.fill_between(
        times, probs, THRESHOLD,
        where=(probs < THRESHOLD),
        interpolate=True,
        color=COLORS["silence"], alpha=0.35, label="Silence",
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

    # ── VAD panels ────────────────────────────────────────────────────────
    for idx, (model_name, (times, probs)) in enumerate(results.items()):
        ax = axes[idx + 1]
        ax.set_title(model_name, fontsize=10, fontweight="bold")
        ax.set_ylabel("P(speech)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)

        if len(times) == 0:
            ax.text(
                0.5, 0.5, "Model failed to run",
                transform=ax.transAxes,
                ha="center", va="center",
                color="red", fontsize=12,
            )
            continue

        _add_speech_bands(ax, times, probs)

        speech_pct = float(np.mean(probs >= THRESHOLD)) * 100
        frame_ms   = (times[1] - times[0]) * 1000 if len(times) > 1 else 0.0
        ax.text(
            0.005, 0.87,
            f"Speech {speech_pct:.1f}%  |  frame {frame_ms:.0f} ms",
            transform=ax.transAxes,
            fontsize=8, color="#1b5e20",
        )

        legend_patches = [
            mpatches.Patch(color=COLORS["speech"],  alpha=0.7, label="Speech"),
            mpatches.Patch(color=COLORS["silence"], alpha=0.55, label="Silence"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.7)

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    fig.suptitle(
        "VAD Model Comparison  –  Silero · WebRTC · TEN",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Silero, WebRTC, and TEN VAD on an audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "audio_file",
        nargs="?",
        default="audio/sample.mp3",
        help="Path to the audio file (default: audio/sample.mp3)",
    )
    p.add_argument(
        "--output", "-o",
        default="comparison.png",
        help="Output plot filename (default: comparison.png)",
    )
    p.add_argument(
        "--max-duration", "-d",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Truncate audio to at most this many seconds before processing",
    )
    p.add_argument(
        "--webrtc-mode",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness: 0 (least) … 3 (most) (default: 3)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.audio_file):
        sys.exit(
            f"Audio file not found: {args.audio_file}\n"
            "Run   python download_sample.py   to fetch a LibriVox chapter."
        )

    audio = load_audio(args.audio_file, max_duration=args.max_duration)

    print("\nLoading VAD models …")
    models = build_models(webrtc_mode=args.webrtc_mode)

    if not models:
        sys.exit("No VAD models could be loaded – install dependencies and retry.")

    results = run_all(audio, models)

    print_stats(results)

    plot_comparison(audio, results, args.output)


if __name__ == "__main__":
    main()
