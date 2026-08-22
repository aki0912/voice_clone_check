from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


EPSILON = 1e-10


@dataclass(frozen=True)
class AudioQuality:
    duration: float
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    snr_db: float
    quality_ok: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mono(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1:
        raise ValueError("音声データは1次元または2次元である必要があります")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        audio,
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float32)


def _frame_rms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(sample_rate * 0.025))
    hop = max(1, int(sample_rate * 0.010))
    if len(audio) < frame:
        return np.array([np.sqrt(np.mean(audio**2) + EPSILON)])
    count = 1 + (len(audio) - frame) // hop
    shape = (count, frame)
    strides = (audio.strides[0] * hop, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    return np.sqrt(np.mean(frames**2, axis=1) + EPSILON)


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_rms = _frame_rms(audio, sample_rate)
    frame_db = 20 * np.log10(frame_rms + EPSILON)
    threshold = max(-45.0, float(np.percentile(frame_db, 15)) + 10.0)
    active = np.flatnonzero(frame_db >= threshold)
    if active.size == 0:
        return audio
    hop = max(1, int(sample_rate * 0.010))
    pad = int(sample_rate * 0.12)
    start = max(0, int(active[0] * hop) - pad)
    end = min(len(audio), int((active[-1] + 3) * hop) + pad)
    return audio[start:end]


def inspect_audio(
    audio: np.ndarray,
    sample_rate: int,
    quality_config: dict[str, float],
) -> AudioQuality:
    if audio.size == 0:
        raise ValueError("音声が空です")
    frame_rms = _frame_rms(audio, sample_rate)
    frame_db = 20 * np.log10(frame_rms + EPSILON)
    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio**2) + EPSILON))
    peak = float(np.max(np.abs(audio)))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    noise_rms = float(np.percentile(frame_rms, 10))
    speech_rms = float(np.percentile(frame_rms, 90))
    snr_db = float(20 * np.log10((speech_rms + EPSILON) / (noise_rms + EPSILON)))
    silence_threshold = max(-45.0, float(np.percentile(frame_db, 90)) - 35.0)
    silence_ratio = float(np.mean(frame_db < silence_threshold))

    warnings: list[str] = []
    if duration < float(quality_config["min_seconds"]):
        warnings.append("音声が短すぎます")
    if duration > float(quality_config["max_seconds"]):
        warnings.append("音声が長すぎます")
    if clipping_ratio > float(quality_config["max_clipping_ratio"]):
        warnings.append("音割れが検出されました")
    if snr_db < float(quality_config["min_snr_db"]):
        warnings.append("背景雑音が多い可能性があります")
    if silence_ratio > float(quality_config["max_silence_ratio"]):
        warnings.append("無音区間が多すぎます")
    if 20 * math.log10(peak + EPSILON) < -24:
        warnings.append("録音レベルが小さすぎます")

    return AudioQuality(
        duration=duration,
        rms_dbfs=20 * math.log10(rms + EPSILON),
        peak_dbfs=20 * math.log10(peak + EPSILON),
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        snr_db=snr_db,
        quality_ok=not warnings,
        warnings=tuple(warnings),
    )


def prepare_recording(
    source_path: str | Path,
    raw_path: str | Path,
    processed_path: str | Path,
    sample_rate: int,
    quality_config: dict[str, float],
) -> tuple[AudioQuality, str]:
    source = Path(source_path)
    raw = Path(raw_path)
    processed = Path(processed_path)
    raw.parent.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, raw)

    audio, source_rate = sf.read(source, always_2d=False)
    mono = _mono(audio)
    converted = _resample(mono, int(source_rate), sample_rate)
    trimmed = trim_silence(converted, sample_rate)
    sf.write(processed, trimmed, sample_rate, subtype="PCM_16")
    quality = inspect_audio(trimmed, sample_rate, quality_config)
    return quality, sha256_file(processed)


def valid_audio_file(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.exists() or candidate.stat().st_size < 100:
        return False
    try:
        info = sf.info(candidate)
    except Exception:
        return False
    return info.frames > 0 and info.samplerate > 0


def suggest_cumulative_boundaries(
    source_path: str | Path,
    targets: list[float],
    *,
    sample_rate: int = 24000,
    search_seconds: float = 1.25,
) -> tuple[list[float], float]:
    """Find quiet phrase boundaries near cumulative duration targets."""
    audio, source_rate = sf.read(source_path, always_2d=False)
    converted = trim_silence(
        _resample(_mono(audio), int(source_rate), sample_rate), sample_rate
    )
    duration = len(converted) / sample_rate
    rms = _frame_rms(converted, sample_rate)
    hop_seconds = 0.010
    boundaries: list[float] = []
    previous = 0.0
    for target in targets[:-1]:
        low = max(previous + 0.5, float(target) - search_seconds)
        high = min(duration - 0.5, float(target) + search_seconds)
        if high <= low:
            boundary = min(max(float(target), previous + 0.5), duration)
        else:
            start = max(0, int(low / hop_seconds))
            end = min(len(rms), int(high / hop_seconds) + 1)
            index = start + int(np.argmin(rms[start:end]))
            boundary = index * hop_seconds
        boundaries.append(round(boundary, 3))
        previous = boundary
    boundaries.append(round(duration, 3))
    return boundaries, duration


def write_cumulative_clips(
    source_path: str | Path,
    boundaries: list[float],
    destinations: list[str | Path],
    *,
    sample_rate: int = 24000,
) -> list[Path]:
    """Write nested clips that share the exact same processed recording start."""
    if len(boundaries) != len(destinations) or not boundaries:
        raise ValueError("境界時刻と出力先の数が一致していません")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("境界時刻は昇順で指定してください")
    audio, source_rate = sf.read(source_path, always_2d=False)
    converted = trim_silence(
        _resample(_mono(audio), int(source_rate), sample_rate), sample_rate
    )
    duration = len(converted) / sample_rate
    if boundaries[0] <= 0 or boundaries[-1] > duration + 0.05:
        raise ValueError("境界時刻が録音範囲外です")
    written: list[Path] = []
    for boundary, destination_value in zip(boundaries, destinations):
        destination = Path(destination_value)
        destination.parent.mkdir(parents=True, exist_ok=True)
        end = min(len(converted), int(round(boundary * sample_rate)))
        sf.write(destination, converted[:end], sample_rate, subtype="PCM_16")
        written.append(destination)
    return written
