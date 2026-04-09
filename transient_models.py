"""
Pop / click / plosive detector wrappers — three approaches, one unified interface.

Each exposes:
    times, probs = detector.process(audio_path, sample_rate=16000)

where
  * times  – 1-D float64 array of frame centre timestamps (seconds)
  * probs  – 1-D float64 array of transient probabilities in [0, 1]

Detectors
---------
1. KurtosisClickDetector   – Short-time kurtosis of the waveform.
                             Impulsive events (clicks, pops) produce very
                             high kurtosis.  Works at 10 ms resolution.

2. SpectralFluxDetector    – Librosa onset strength (spectral flux novelty).
                             Responds to any sudden spectral change —
                             plosives, taps, and broadband clicks.

3. SubBassBurstDetector    – Measures energy concentration below 200 Hz
                             combined with a rapid energy onset.  Targets
                             microphone pops and breath thumps.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Kurtosis Click Detector
# ---------------------------------------------------------------------------

class KurtosisClickDetector:
    """
    Short-time kurtosis detector for impulsive noise (clicks, crackles, pops).

    Kurtosis measures the "tailedness" of the amplitude distribution within
    a short frame.  A sinusoid has excess kurtosis ≈ −1.5; Gaussian noise
    ≈ 0; a sharp impulse (click) produces values of 10–100+.

    The raw kurtosis is mapped to [0, 1] via a shifted sigmoid so that
    frames with kurtosis ≥ *center* have probability ≥ 0.5.

    pip install librosa scipy
    """

    FRAME_SEC = 0.010  # 10 ms — long enough for stable kurtosis, short for clicks
    HOP_SEC   = 0.010  # no overlap; 10 ms frames → 100 fps output

    def __init__(
        self,
        center: float = 5.0,   # kurtosis at which prob = 0.5
        scale:  float = 3.0,   # steepness of the sigmoid
    ) -> None:
        self.center = center
        self.scale  = scale

    @property
    def name(self) -> str:
        return "Kurtosis Click Detector"

    def process(self, audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        import librosa
        from scipy.stats import kurtosis as _kurtosis

        wav, _ = librosa.load(audio_path, sr=16000)
        sr         = 16000
        frame_len  = int(sr * self.FRAME_SEC)
        hop_len    = int(sr * self.HOP_SEC)

        times_list: list[float] = []
        probs_list: list[float] = []

        for start in range(0, len(wav) - frame_len, hop_len):
            frame = wav[start : start + frame_len]
            # Fisher definition (excess kurtosis); Normal ≈ 0
            if np.std(frame) < 1e-7:
                kurt = 0.0
            else:
                kurt = float(_kurtosis(frame, fisher=True))

            # Sigmoid centred at self.center
            prob = float(1.0 / (1.0 + np.exp(-(kurt - self.center) / self.scale)))
            times_list.append((start + frame_len / 2) / sr)
            probs_list.append(prob)

        return (
            np.array(times_list, dtype=np.float64),
            np.array(probs_list, dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# 2. Spectral Flux Detector
# ---------------------------------------------------------------------------

class SpectralFluxDetector:
    """
    Transient detector based on librosa's onset strength (spectral flux novelty).

    Computes the frame-to-frame positive change in the mel-spectrogram and
    normalises it to [0, 1] using the 99th-percentile of the file.  Any
    sudden spectral onset — voiced plosive, mouth click, hand clap, pop —
    produces a high score.

    pip install librosa
    """

    HOP_SEC = 0.010

    def __init__(
        self,
        n_mels:  int   = 128,
        fmin:    float = 20.0,
        fmax:    float = 8000.0,
        norm_pct: float = 99.0,  # percentile used to clip/normalise
    ) -> None:
        self.n_mels   = n_mels
        self.fmin     = fmin
        self.fmax     = fmax
        self.norm_pct = norm_pct

    @property
    def name(self) -> str:
        return "Spectral Flux Detector"

    def process(self, audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        import librosa

        wav, _ = librosa.load(audio_path, sr=16000)
        sr         = 16000
        hop_length = int(sr * self.HOP_SEC)

        onset_env = librosa.onset.onset_strength(
            y=wav, sr=sr,
            hop_length=hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            aggregate=np.mean,
        )

        # Normalise: p99 → 1.0, lower values scaled proportionally
        p99 = float(np.percentile(onset_env, self.norm_pct))
        if p99 > 1e-9:
            probs = np.clip(onset_env / p99, 0.0, 1.0)
        else:
            probs = np.zeros_like(onset_env)

        times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
        return times.astype(np.float64), probs.astype(np.float64)


# ---------------------------------------------------------------------------
# 3. Sub-Bass Burst Detector
# ---------------------------------------------------------------------------

class SubBassBurstDetector:
    """
    Microphone pop and low-frequency thump detector.

    Microphone pops occur when air hits the capsule — they are characterised
    by a sudden, high-amplitude burst of sub-bass energy (typically < 200 Hz).
    Speech rarely has dominant energy that low.

    Score = weighted combination of:
      • Low-frequency energy ratio  (energy < lf_hz / total energy)
      • Onset strength              (sudden energy increase)
      • Overall RMS gate            (ignores near-silent frames)

    pip install librosa
    """

    FRAME_SEC = 0.025
    HOP_SEC   = 0.010

    def __init__(
        self,
        lf_hz:    float = 200.0,  # sub-bass ceiling (Hz)
        rms_gate: float = 5e-3,   # minimum RMS to consider a frame active
    ) -> None:
        self.lf_hz    = lf_hz
        self.rms_gate = rms_gate

    @property
    def name(self) -> str:
        return "Sub-Bass Burst Detector"

    def process(self, audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        import librosa

        wav, _ = librosa.load(audio_path, sr=16000)
        sr         = 16000
        n_fft      = int(sr * self.FRAME_SEC)
        hop_length = int(sr * self.HOP_SEC)

        # Short-time Fourier transform
        S = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        lf_mask     = freqs <= self.lf_hz
        total_power = S.sum(axis=0) + 1e-12
        lf_power    = S[lf_mask, :].sum(axis=0)
        lf_ratio    = lf_power / total_power  # 0…1

        # RMS energy gate: suppress near-silent frames
        rms = librosa.feature.rms(y=wav, frame_length=n_fft, hop_length=hop_length)[0]
        energy_gate = np.clip(rms / (self.rms_gate + 1e-12), 0.0, 1.0)

        # Onset strength (detect sudden LF energy bursts)
        onset_env = librosa.onset.onset_strength(
            y=librosa.effects.preemphasis(wav, coef=-0.95),  # boost low freqs
            sr=sr, hop_length=hop_length,
        )
        p99 = float(np.percentile(onset_env, 99))
        onset_norm = np.clip(onset_env / (p99 + 1e-9), 0.0, 1.0)

        # Align lengths (STFT may produce one extra frame)
        n = min(len(lf_ratio), len(energy_gate), len(onset_norm))
        lf_ratio    = lf_ratio[:n]
        energy_gate = energy_gate[:n]
        onset_norm  = onset_norm[:n]

        probs = np.clip(
            lf_ratio * 0.45 + energy_gate * 0.25 + onset_norm * 0.30,
            0.0, 1.0,
        ).astype(np.float64)

        times = (
            np.arange(n) * hop_length / sr + (n_fft / 2) / sr
        ).astype(np.float64)

        return times, probs
