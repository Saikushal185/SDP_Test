from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


TOP10_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = TOP10_ROOT.parent
STUDY_ROOT = WORKSPACE_ROOT / "parkinson_feature_study"
LOCAL_PACKAGES = STUDY_ROOT / ".python_packages"

for path in (LOCAL_PACKAGES, STUDY_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


TOP10_FEATURE_NAMES: tuple[str, ...] = (
    "tqwt_entropy_log_dec_35",
    "std_delta_delta_log_energy",
    "std_8th_delta_delta",
    "mean_MFCC_2nd_coef",
    "tqwt_TKEO_mean_dec_16",
    "tqwt_entropy_shannon_dec_35",
    "tqwt_TKEO_std_dec_12",
    "tqwt_maxValue_dec_12",
    "tqwt_entropy_log_dec_11",
    "tqwt_TKEO_mean_dec_12",
)
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac"}
DEFAULT_SAMPLE_RATE = 44_100
TQWT_LEVELS = 36


class AudioFeatureExtractionError(ValueError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def extract_top10_audio_features(contents: bytes, filename: str) -> dict[str, float]:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise AudioFeatureExtractionError(f"Unsupported audio file type. Use one of: {supported}", status_code=415)
    if not contents:
        raise AudioFeatureExtractionError("The uploaded audio file is empty.")

    y, sample_rate = _load_audio(contents, suffix)
    y = _prepare_signal(y, sample_rate)

    mfcc = _mfcc(y, sample_rate)
    log_energy = _frame_log_energy(y)
    delta_delta_log_energy = _delta(log_energy.reshape(1, -1), order=2).reshape(-1)
    delta_delta_mfcc = _delta(mfcc, order=2)

    bands = _tqwt_bands(y, sample_rate, levels=TQWT_LEVELS, q=1.0, redundancy=3.0)
    band_11 = bands[10]
    band_12 = bands[11]
    band_16 = bands[15]
    band_35 = bands[34]
    tkeo_12 = _tkeo(band_12)
    tkeo_16 = _tkeo(band_16)

    features: dict[str, float] = {
        "tqwt_entropy_log_dec_35": _log_entropy(band_35),
        "std_delta_delta_log_energy": float(np.std(delta_delta_log_energy)),
        "std_8th_delta_delta": float(np.std(delta_delta_mfcc[8])),
        "mean_MFCC_2nd_coef": float(np.mean(mfcc[2])),
        "tqwt_TKEO_mean_dec_16": float(np.mean(tkeo_16)),
        "tqwt_entropy_shannon_dec_35": _shannon_entropy(band_35),
        "tqwt_TKEO_std_dec_12": float(np.std(tkeo_12)),
        "tqwt_maxValue_dec_12": float(np.max(band_12)),
        "tqwt_entropy_log_dec_11": _log_entropy(band_11),
        "tqwt_TKEO_mean_dec_12": float(np.mean(tkeo_12)),
    }
    _validate_features(features)
    return {feature_name: features[feature_name] for feature_name in TOP10_FEATURE_NAMES}


def _load_audio(contents: bytes, suffix: str) -> tuple[np.ndarray, int]:
    import librosa

    path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(contents)
            path = temp_file.name
        y, sample_rate = librosa.load(path, sr=DEFAULT_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioFeatureExtractionError(f"Could not read this audio file: {exc}") from exc
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    return np.asarray(y, dtype=float), int(sample_rate)


def _prepare_signal(y: np.ndarray, sample_rate: int) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0 or not np.any(np.abs(y) > 1e-10):
        raise AudioFeatureExtractionError("The uploaded audio does not contain a usable voice signal.")

    max_samples = sample_rate * 10
    min_samples = max(sample_rate // 2, 2048)
    if y.size > max_samples:
        y = y[:max_samples]
    if y.size < min_samples:
        y = np.pad(y, (0, min_samples - y.size))
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y


def _mfcc(y: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    return np.asarray(
        librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512),
        dtype=float,
    )


def _frame_log_energy(y: np.ndarray) -> np.ndarray:
    import librosa

    frame_length = 2048
    hop_length = 512
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    return np.log(energy + 1e-12)


def _delta(values: np.ndarray, order: int) -> np.ndarray:
    import librosa

    frame_count = values.shape[-1]
    if frame_count < 3:
        return np.zeros_like(values, dtype=float)
    width = min(9, frame_count if frame_count % 2 else frame_count - 1)
    if width < 3:
        return np.zeros_like(values, dtype=float)
    return np.asarray(librosa.feature.delta(values, order=order, width=width, mode="nearest"), dtype=float)


def _tqwt_bands(
    y: np.ndarray,
    sample_rate: int,
    levels: int,
    q: float,
    redundancy: float,
) -> list[np.ndarray]:
    beta = 2.0 / (q + 1.0)
    alpha = 1.0 - (beta / redundancy)
    if not 0.0 < alpha < 1.0:
        raise AudioFeatureExtractionError("Invalid TQWT settings.")

    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / sample_rate)
    high = sample_rate / 2.0
    bands: list[np.ndarray] = []
    for _ in range(levels):
        low = max(0.0, high * alpha)
        mask = (freqs >= low) & (freqs <= high)
        band_spectrum = np.zeros_like(spectrum)
        band_spectrum[mask] = spectrum[mask]
        bands.append(np.fft.irfft(band_spectrum, n=y.size).astype(float, copy=False))
        high = low
    return bands


def _tkeo(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.zeros(1, dtype=float)
    return values[1:-1] ** 2 - values[:-2] * values[2:]


def _log_entropy(values: np.ndarray) -> float:
    energy = np.asarray(values, dtype=float) ** 2
    return float(np.sum(np.log(energy + 1e-12)))


def _shannon_entropy(values: np.ndarray) -> float:
    magnitudes = np.abs(np.asarray(values, dtype=float))
    total = float(np.sum(magnitudes))
    if total <= 0.0:
        return 0.0
    probabilities = magnitudes / total
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _validate_features(features: dict[str, Any]) -> None:
    missing = [feature_name for feature_name in TOP10_FEATURE_NAMES if feature_name not in features]
    if missing:
        raise AudioFeatureExtractionError(f"Could not calculate required features: {', '.join(missing)}")
    invalid = [
        feature_name
        for feature_name in TOP10_FEATURE_NAMES
        if not np.isfinite(float(features[feature_name]))
    ]
    if invalid:
        raise AudioFeatureExtractionError(f"Audio produced invalid feature values: {', '.join(invalid)}")
