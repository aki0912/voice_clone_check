from __future__ import annotations

import gc
import random
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


class ModelBackends:
    """Lazy model loaders keep the UI usable before weights are downloaded."""

    def __init__(
        self,
        tts_model_id: str,
        asr_model_id: str,
        speaker_model_id: str,
        utmos_repo: str,
        sample_rate: int = 24000,
    ):
        self.tts_model_id = tts_model_id
        self.asr_model_id = asr_model_id
        self.speaker_model_id = speaker_model_id
        self.utmos_repo = utmos_repo
        self.sample_rate = sample_rate
        self._tts = None
        self._asr = None
        self._speaker = None
        self._utmos = None

    def _load_tts(self):
        if self._tts is None:
            from mlx_audio.tts.utils import load_model

            self._tts = load_model(self.tts_model_id)
        return self._tts

    def _load_asr(self):
        if self._asr is None:
            from mlx_audio.stt import load

            self._asr = load(self.asr_model_id)
        return self._asr

    def _load_speaker(self):
        if self._speaker is None:
            import onnxruntime as ort

            model_path = self._speaker_model_path()
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 4
            self._speaker = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        return self._speaker

    def _speaker_model_path(self) -> Path:
        filename = Path(self.speaker_model_id).name
        cache_dir = Path.home() / ".cache" / "voice-clone-check" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = cache_dir / filename
        if destination.exists() and destination.stat().st_size > 1024 * 1024:
            return destination
        url = (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            f"speaker-recongition-models/{filename}"
        )
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        return destination

    def _load_utmos(self):
        if self._utmos is None:
            import torch

            self._utmos = torch.hub.load(
                self.utmos_repo,
                "utmos22_strong",
                trust_repo=True,
                verbose=False,
            )
            self._utmos.eval()
        return self._utmos

    def synthesize(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        seed: int,
        output_path: str | Path,
        max_tokens: int = 2048,
    ) -> None:
        import mlx.core as mx

        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
        model = self._load_tts()
        results = list(
            model.generate(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                lang_code="Japanese",
                max_tokens=max_tokens,
                verbose=False,
            )
        )
        if not results:
            raise RuntimeError("Qwen3-TTSが音声を返しませんでした")
        chunks = [_to_numpy(result.audio).reshape(-1) for result in results]
        audio = np.concatenate(chunks)
        output_sample_rate = int(results[0].sample_rate)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio, output_sample_rate, subtype="PCM_16")
        mx.clear_cache()

    def transcribe(self, audio_path: str | Path) -> str:
        result = self._load_asr().generate(str(audio_path), language="Japanese")
        if isinstance(result, str):
            return result.strip()
        return str(getattr(result, "text", result)).strip()

    def speaker_embedding(self, audio_path: str | Path) -> np.ndarray:
        import torch
        import torchaudio.compliance.kaldi as kaldi
        from scipy.signal import resample_poly

        audio, sample_rate = sf.read(str(audio_path), always_2d=False)
        audio_values = np.asarray(audio, dtype=np.float32)
        if audio_values.ndim == 2:
            audio_values = audio_values.mean(axis=1)
        if sample_rate != 16000:
            divisor = np.gcd(sample_rate, 16000)
            audio_values = resample_poly(
                audio_values,
                16000 // divisor,
                sample_rate // divisor,
            ).astype(np.float32)
            sample_rate = 16000
        waveform = torch.from_numpy(audio_values).unsqueeze(0)
        waveform = waveform * (1 << 15)
        features = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=sample_rate,
            window_type="hamming",
            use_energy=False,
        )
        features = features - torch.mean(features, dim=0)
        session = self._load_speaker()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        embedding = session.run(
            [output_name],
            {input_name: features.unsqueeze(0).cpu().numpy()},
        )[0]
        values = _to_numpy(embedding).reshape(-1)
        norm = np.linalg.norm(values)
        if norm == 0:
            raise RuntimeError("話者埋め込みがゼロベクトルです")
        return values / norm

    def utmos(self, audio_path: str | Path) -> float:
        import torch

        audio, sample_rate = sf.read(str(audio_path), always_2d=False)
        audio_values = np.asarray(audio, dtype=np.float32)
        if audio_values.ndim == 2:
            audio_values = audio_values.mean(axis=1)
        waveform = torch.from_numpy(audio_values).unsqueeze(0)
        model = self._load_utmos()
        with torch.inference_mode():
            try:
                score = model(waveform, sample_rate)
            except TypeError:
                score = model(waveform)
        return float(score.detach().cpu().reshape(-1)[0])

    def release(self) -> None:
        self._tts = None
        self._asr = None
        self._speaker = None
        self._utmos = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    if denominator == 0:
        raise ValueError("ゼロベクトルの類似度は計算できません")
    return float(np.dot(left_values, right_values) / denominator)
