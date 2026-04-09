"""
VAD model wrappers for Silero VAD, WebRTC VAD, and TEN VAD.

Each wrapper exposes a common interface:

    times, probs = model.process(audio_float32, sample_rate=16000)

where
  * audio_float32  – 1-D numpy float32 array, values in [-1, 1]
  * sample_rate    – must be 16 000 Hz for all models
  * times          – 1-D float array of frame centre timestamps (seconds)
  * probs          – 1-D float array of speech probabilities in [0, 1]

WebRTC VAD only produces binary 0/1 outputs; the other two produce
continuous probabilities.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Silero VAD
# ---------------------------------------------------------------------------

class SileroVAD:
    """
    Wrapper around the silero-vad PyPI package.

    pip install silero-vad
    https://github.com/snakers4/silero-vad
    """

    # Silero's supported chunk sizes at 16 kHz
    CHUNK_SIZE = 512  # 32 ms per frame

    def __init__(self, threshold: float = 0.5) -> None:
        try:
            from silero_vad import load_silero_vad  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "silero-vad is not installed. Run: pip install silero-vad"
            )
        import torch  # type: ignore[import]

        self._torch = torch
        self.model = load_silero_vad()
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Silero VAD"

    def process(self, audio: np.ndarray, sample_rate: int = 16000):
        if sample_rate != 16000:
            raise ValueError("Silero VAD requires 16 kHz audio")

        self.model.reset_states()

        times: list[float] = []
        probs: list[float] = []

        step = self.CHUNK_SIZE
        for start in range(0, len(audio) - step, step):
            chunk = self._torch.FloatTensor(audio[start : start + step])
            prob: float = self.model(chunk, sample_rate).item()
            times.append((start + step / 2) / sample_rate)
            probs.append(prob)

        return np.asarray(times, dtype=np.float64), np.asarray(probs, dtype=np.float64)


# ---------------------------------------------------------------------------
# WebRTC VAD
# ---------------------------------------------------------------------------

class WebRTCVAD:
    """
    Wrapper around webrtcvad (or webrtcvad-wheels).

    pip install webrtcvad-wheels
    https://github.com/wiseman/py-webrtcvad
    """

    # Valid WebRTC frame durations: 10, 20, or 30 ms.
    # 30 ms @ 16 kHz = 480 samples
    FRAME_MS = 30
    CHUNK_SIZE = 480

    def __init__(self, aggressiveness: int = 3) -> None:
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be 0, 1, 2, or 3")
        try:
            import webrtcvad  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "webrtcvad is not installed. Run: pip install webrtcvad-wheels"
            )
        self._vad = webrtcvad.Vad(aggressiveness)
        self.aggressiveness = aggressiveness

    @property
    def name(self) -> str:
        return f"WebRTC VAD (mode {self.aggressiveness})"

    def process(self, audio: np.ndarray, sample_rate: int = 16000):
        if sample_rate != 16000:
            raise ValueError("WebRTC VAD requires 16 kHz audio")

        # Convert float32 → int16 PCM bytes
        audio_i16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

        times: list[float] = []
        probs: list[float] = []

        step = self.CHUNK_SIZE
        for start in range(0, len(audio_i16) - step, step):
            frame_bytes = audio_i16[start : start + step].tobytes()
            is_speech: bool = self._vad.is_speech(frame_bytes, sample_rate)
            times.append((start + step / 2) / sample_rate)
            probs.append(1.0 if is_speech else 0.0)

        return np.asarray(times, dtype=np.float64), np.asarray(probs, dtype=np.float64)


# ---------------------------------------------------------------------------
# TEN VAD
# ---------------------------------------------------------------------------

class TenVAD:
    """
    Wrapper around the ten-vad PyPI package (TEN-framework).

    pip install ten-vad
    https://github.com/TEN-framework/ten-vad

    The model returns a speech probability in [0, 1] for each 16 ms frame.
    """

    CHUNK_SIZE = 256  # 16 ms @ 16 kHz

    def __init__(self, threshold: float = 0.5) -> None:
        try:
            from ten_vad import TenVad as _TenVad  # type: ignore[import]
            self._vad = _TenVad()
        except ImportError:
            raise ImportError(
                "ten-vad is not installed. Run: pip install ten-vad"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise TEN VAD: {exc}") from exc

        self.threshold = threshold
        self._process_fn = self._resolve_process_fn()

    def _resolve_process_fn(self):
        """
        Return a callable that accepts an int16 numpy frame and returns a
        float probability.  Handles several possible API shapes.
        """
        if hasattr(self._vad, "process"):
            raw = self._vad.process
        elif hasattr(self._vad, "detect"):
            raw = self._vad.detect
        else:
            raise AttributeError(
                "ten_vad.TenVad has neither .process() nor .detect(). "
                "Please check the installed version."
            )

        def _call(frame: np.ndarray) -> float:
            result = raw(frame)
            # Possible return shapes:
            #   float / int        → speech probability or binary flag
            #   (bool, float)      → (is_speech, confidence)
            #   (float, ...)       → probability first
            if isinstance(result, (list, tuple)):
                # Prefer the second element if it looks like a probability
                if len(result) >= 2:
                    candidate = float(result[1])
                    if 0.0 <= candidate <= 1.0:
                        return candidate
                return float(result[0])
            return float(result)

        return _call

    @property
    def name(self) -> str:
        return "TEN VAD"

    def process(self, audio: np.ndarray, sample_rate: int = 16000):
        if sample_rate != 16000:
            raise ValueError("TEN VAD requires 16 kHz audio")

        audio_i16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

        times: list[float] = []
        probs: list[float] = []

        step = self.CHUNK_SIZE
        for start in range(0, len(audio_i16) - step, step):
            frame = audio_i16[start : start + step]
            prob = self._process_fn(frame)
            times.append((start + step / 2) / sample_rate)
            probs.append(prob)

        return np.asarray(times, dtype=np.float64), np.asarray(probs, dtype=np.float64)
