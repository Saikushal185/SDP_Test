"""
Deterministic acoustic feature extraction for Parkinson speech datasets.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_stat_prefix(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
    }


def extract_acoustic_features(audio_array: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Extract a fixed acoustic feature bank from one audio sample.

    Uses `librosa` for spectral and cepstral features and
    `praat-parselmouth` for phonation-style features.
    """
    import librosa
    import parselmouth

    y = np.asarray(audio_array, dtype=np.float32).squeeze()
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)

    max_duration_seconds = 10
    max_samples = sample_rate * max_duration_seconds
    if y.size > max_samples:
        y = y[:max_samples]

    target_length = max(sample_rate // 5, y.size)
    if y.size < target_length:
        y = np.pad(y, (0, target_length - y.size))

    y = np.nan_to_num(y)

    features: Dict[str, float] = {
        "duration_seconds": float(len(y) / float(sample_rate)),
        "signal_energy": float(np.mean(np.square(y))),
        "signal_std": float(np.std(y)),
    }

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    pitch = librosa.yin(
        y,
        fmin=50,
        fmax=min(500, sample_rate // 2 - 1),
        sr=sample_rate,
    )

    features.update(_safe_stat_prefix(zcr, "zcr"))
    features.update(_safe_stat_prefix(rms, "rms"))
    features.update(_safe_stat_prefix(centroid, "spectral_centroid"))
    features.update(_safe_stat_prefix(bandwidth, "spectral_bandwidth"))
    features.update(_safe_stat_prefix(rolloff, "spectral_rolloff"))
    features.update(_safe_stat_prefix(flatness, "spectral_flatness"))
    features.update(_safe_stat_prefix(pitch[np.isfinite(pitch)], "pitch"))

    for mfcc_idx, coeffs in enumerate(mfcc, start=1):
        features.update(_safe_stat_prefix(coeffs, f"mfcc_{mfcc_idx}"))

    try:
        snd = parselmouth.Sound(y, sample_rate)
        point_process = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 500
        )
        features["jitter_local"] = float(
            parselmouth.praat.call(
                point_process,
                "Get jitter (local)",
                0,
                0,
                75,
                500,
                1.3,
            )
        )
        features["shimmer_local"] = float(
            parselmouth.praat.call(
                [snd, point_process],
                "Get shimmer (local)",
                0,
                0,
                75,
                500,
                1.3,
                1.6,
            )
        )
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr_mean"] = float(parselmouth.praat.call(harmonicity, "Get mean", 0, 0))
    except Exception:
        features["jitter_local"] = 0.0
        features["shimmer_local"] = 0.0
        features["hnr_mean"] = 0.0

    return {key: float(np.nan_to_num(value)) for key, value in features.items()}
