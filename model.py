"""Replace this identity baseline with your restoration model.

The function accepts float32 audio shaped [samples, 2] at 44.1 kHz and must
return the same shape with finite values. The model implementation may load
weights from the image or from a read-only mounted directory.
"""

from __future__ import annotations

import numpy as np


def restore(audio: np.ndarray, sample_rate: int, metadata: dict) -> np.ndarray:
    """Return unmodified input audio as an interface smoke test."""
    return audio
