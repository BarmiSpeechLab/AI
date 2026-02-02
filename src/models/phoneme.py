from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List
from itertools import groupby

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


@dataclass
class PhonemeModels:
    model: Any
    processor: Any
    device: str
    sampling_rate: int


def load_phoneme_models(model_dir: str) -> PhonemeModels:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()
    return PhonemeModels(
        model=model,
        processor=processor,
        device=device,
        sampling_rate=processor.feature_extractor.sampling_rate,
    )


def _ctc_collapse(ids: List[int], blank_id: int) -> List[int]:
    out, prev = [], None
    for i in ids:
        if i == blank_id:
            prev = None
            continue
        if i != prev:
            out.append(i)
        prev = i
    return out


def extract_arpabet_full(audio_path: str, models: PhonemeModels) -> list[str]:
    audio, sr = librosa.load(audio_path, sr=models.sampling_rate)
    inputs = models.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = inputs.to(models.device)

    with torch.no_grad():
        logits = models.model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    # 특수토큰만 제거하고 |, SIL은 유지(있다면 단어 분리용)
    tokens = models.processor.tokenizer.convert_ids_to_tokens(pred_ids)
    tokens = [t for t in tokens if t not in {"[PAD]", "[UNK]", "<s>", "</s>", "|", "SIL", "SP"}]
    tokens = [k for k, _ in groupby(tokens)]  # 연속 중복 제거
    return tokens


def _decode_ids_to_tokens(ids: List[int], processor: Wav2Vec2Processor) -> List[str]:
    blank_id = processor.tokenizer.pad_token_id
    ids = _ctc_collapse(ids, blank_id)
    tokens = processor.tokenizer.convert_ids_to_tokens(ids)
    # word delimiter/SIL 제거
    return [t for t in tokens if t not in {"[PAD]", "[UNK]", "<s>", "</s>", "|", "SIL", "SP"}]


def extract_arpabet_by_word_segments(
    audio_path: str,
    word_segments: List[dict],
    models: PhonemeModels,
    pad_seconds: float = 0.02,
) -> List[List[str]]:
    audio, sr = librosa.load(audio_path, sr=models.sampling_rate)
    total_len = len(audio)

    results: List[List[str]] = []
    for seg in word_segments:
        start = max(int((seg["start"] - pad_seconds) * sr), 0)
        end = min(int((seg["end"] + pad_seconds) * sr), total_len)
        if end <= start:
            results.append([])
            continue

        chunk = audio[start:end]
        inputs = models.processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = inputs.to(models.device)

        with torch.no_grad():
            logits = models.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

        tokens = _decode_ids_to_tokens(pred_ids, models.processor)
        results.append(tokens)

    return results
