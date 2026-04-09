"""
Breathing detector wrappers — three approaches, one unified interface.

Each exposes:
    times, probs = detector.process(audio_path, sample_rate=16000)

where
  * times  – 1-D float64 array of frame centre timestamps (seconds)
  * probs  – 1-D float64 array of breath probabilities in [0, 1]

Detectors
---------
1. RespiroDetector        – Respiro-en ML model (INTERSPEECH 2024)
                            https://github.com/ydqmkkx/Respiro-en
2. SpectralBreathDetector – Spectral flatness + energy heuristic
3. EnergyZCRDetector      – Amplitude + zero-crossing rate baseline
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Respiro-en  (ML, pretrained)
# ---------------------------------------------------------------------------

class RespiroDetector:
    """
    Frame-level breath detection using the Respiro-en pretrained model.

    Requirements
    ------------
    Clone the repo to models/respiro-en/ before use:
        git clone --depth=1 https://github.com/ydqmkkx/Respiro-en models/respiro-en
    pip install torch torchaudio librosa intervaltree
    """

    FRAME_MS = 10  # hop = sr * 0.01 → 10 ms per output frame

    _SEARCH = [
        REPO_ROOT / "models" / "respiro-en",
        REPO_ROOT / "respiro-en",
    ]

    def __init__(self, threshold: float = 0.064) -> None:
        self.threshold = threshold
        self._model_dir = self._find_model_dir()
        self._model, self._device = self._load_model()

    def _find_model_dir(self) -> Path:
        for d in self._SEARCH:
            if (d / "respiro-en.pt").exists() and (d / "modules.py").exists():
                return d
        raise FileNotFoundError(
            "Respiro-en model not found.\n"
            "Run:  git clone --depth=1 https://github.com/ydqmkkx/Respiro-en models/respiro-en"
        )

    def _load_model(self):
        import torch
        mdir = str(self._model_dir)
        if mdir not in sys.path:
            sys.path.insert(0, mdir)
        from modules import DetectionNet  # type: ignore[import]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = DetectionNet().to(device)
        ckpt   = torch.load(self._model_dir / "respiro-en.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, device

    @property
    def name(self) -> str:
        return "Respiro-en"

    def process(self, audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        import librosa
        mdir = str(self._model_dir)
        if mdir not in sys.path:
            sys.path.insert(0, mdir)
        from modules import feature_extractor  # type: ignore[import]

        wav, _ = librosa.load(audio_path, sr=16000)
        feat, length = feature_extractor(wav)
        feat, length = feat.to(self._device), length.to(self._device)

        with torch.no_grad():
            output = self._model(feat, length)

        probs = output[0].cpu().numpy().astype(np.float64)
        times = np.arange(len(probs)) * self.FRAME_MS / 1000.0
        return times, probs


# ---------------------------------------------------------------------------
# 2. Spectral Breath Detector  (signal processing)
# ---------------------------------------------------------------------------

class SpectralBreathDetector:
    """
    Heuristic detector based on spectral flatness + short-time energy.

    Breathing sounds are broadband (high spectral flatness) and sit in a
    low-to-medium energy band — unlike silence (too quiet) and speech (tonal
    or high-energy).

    pip install librosa scipy
    """

    FRAME_SEC = 0.025
    HOP_SEC   = 0.010  # matches Respiro-en output resolution

    def __init__(
        self,
        energy_lo: float = 5e-4,   # below → silence
        energy_hi: float = 0.12,   # above → likely speech
        flatness_threshold: float = 0.12,
    ) -> None:
        self.energy_lo = energy_lo
        self.energy_hi = energy_hi
        self.flatness_threshold = flatness_threshold

    @property
    def name(self) -> str:
        return "Spectral Breath Detector"

    def process(self, audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        import librosa

        wav, _ = librosa.load(audio_path, sr=16000)
        sr         = 16000
        n_fft      = int(sr * self.FRAME_SEC)
        hop_length = int(sr * self.HOP_SEC)

        rms      = librosa.feature.rms(y=wav, frame_length=n_fft, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=wav, n_fft=n_fft, hop_length=hop_length)[0]
        zcr      = librosa.feature.zero_crossing_rate(
                       wav, frame_length=n_fft, hop_length=hop_length)[0]

        # Energy score: peaks in the middle of [energy_lo, energy_hi]
        in_range   = (rms >= self.energy_lo) & (rms <= self.energy_hi)
        log_mid    = (np.log(self.energy_lo + 1e-9) + np.log(self.energy_hi + 1e-9)) / 2
        log_width  = (np.log(self.energy_hi + 1e-9) - np.log(self.energy_lo + 1e-9)) / 2
        energy_score = np.where(
            in_range,
            1.0 - np.abs(np.log(rms + 1e-9) - log_mid) / (log_width + 1e-9),
            0.0,
        )
        energy_score = np.clip(energy_score, 0.0, 1.0)

        # Flatness score
        flatness_score = np.clip(
            (flatness - self.flatness_threshold) / (1.0 - self.flatness_threshold + 1e-9),
            0.0, 1.0,
        )

        # ZCR: medium values typical of breath (not 0, not very high)
        zcr_score = np.clip(zcr / 0.25, 0, 1) * np.clip(1.0 - zcr / 0.5, 0, 1)

        probs = np.clip(
            energy_score * 0.50 + flatness_score * 0.35 + zcr_score * 0.15,
            0.0, 1.0,
        ).astype(np.float64)

        times = np.arange(len(probs)) * self.HOP_SEC
        return times, probs


# ---------------------------------------------------------------------------
# 3. Energy + ZCR Detector  (simple baseline)
# ---------------------------------------------------------------------------

class EnergyZCRDetector:
    """
    Simplest baseline: a frame is "breath-like" when its RMS energy and
    zero-crossing rate both fall within empirically chosen ranges.

    No ML, no spectral computation — just two scalar features.
    pip install librosa
    """

    FRAME_SEC = 0.025
    HOP_SEC   = 0.010

    def __init__(
        self,
        rms_lo: float = 0.002,
        rms_hi: float = 0.07,
        zcr_lo: float = 0.04,
        zcr_hi: float = 0.22,
    ) -> None:
        self.rms_lo = rms_lo
        self.rms_hi = rms_hi
        self.zcr_lo = zcr_lo
        self.zcr_hi = zcr_hi

    @property
    def name(self) -> str:
        return "Energy + ZCR Detector"

    def process(self, audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        import librosa

        wav, _ = librosa.load(audio_path, sr=16000)
        sr         = 16000
        n_fft      = int(sr * self.FRAME_SEC)
        hop_length = int(sr * self.HOP_SEC)

        rms = librosa.feature.rms(y=wav, frame_length=n_fft, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(
                  wav, frame_length=n_fft, hop_length=hop_length)[0]

        def _soft_membership(vals, lo, hi):
            """Triangular membership function: 1 at midpoint, 0 at edges."""
            mid   = (lo + hi) / 2
            width = (hi - lo) / 2 + 1e-9
            return np.where(
                (vals >= lo) & (vals <= hi),
                1.0 - np.abs(vals - mid) / width,
                0.0,
            )

        rms_score = _soft_membership(rms, self.rms_lo, self.rms_hi)
        zcr_score = _soft_membership(zcr, self.zcr_lo, self.zcr_hi)

        probs = np.clip(rms_score * 0.6 + zcr_score * 0.4, 0.0, 1.0).astype(np.float64)
        times = np.arange(len(probs)) * self.HOP_SEC
        return times, probs
