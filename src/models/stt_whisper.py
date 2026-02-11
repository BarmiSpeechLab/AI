import whisperx
import torch
from dataclasses import dataclass
from typing import Any

@dataclass
class WhisperModels:
    model: Any
    align_model: Any
    metadata: Any
    device: str

# 캐시된 모델 정보
_MODEL_CACHE = {
    "key": None,
    "model": None,
    "model_a": None,
    "metadata": None,
    "device": None,
}


def load_whisperx_models(model_name: str = "small.en", vad_method: str = "silero"):
    """
    [기능] WhisperX 전사 모델 + alignment 모델을 로드합니다.
    [리턴]
      model: whisperx transcription model
      model_a: whisperx alignment model
      metadata: alignment metadata
      device: "cuda" or "cpu"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[WhisperX] 사용 디바이스: {device}")
    print("⏳ 모델 로딩 시작...")

    # 1) Transcription model
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        vad_method=vad_method
    )

    # 2) Alignment model
    language_code = "en"
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)

    print("✅ 모델 로딩 완료")
    return model, model_a, metadata, device

# Whisper 모델 캐싱 및 재사용 함수
def get_whisperx_models(model_name: str = "small.en", vad_method: str = "silero"):
    """
    [기능] WhisperX 모델을 '프로세스 내에서 1번만' 로드하고 캐싱해서 재사용합니다.
    - 같은 (model_name, vad_method, device) 조합이면 캐시를 반환합니다.
    - 다르면 기존 캐시를 언로드 후 새로 로드합니다.

    [주의]
    - FastAPI 운영 시, 서버 종료(shutdown) 시점에 unload_whisperx_models() 호출 필요
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model_name, vad_method, device)

    if _MODEL_CACHE["key"] == key and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["model_a"], _MODEL_CACHE["metadata"], _MODEL_CACHE["device"]

    # 기존 캐시가 있다면 정리 후 다시 로드
    unload_whisperx_models()

    model, model_a, metadata, device = load_whisperx_models(model_name=model_name, vad_method=vad_method)

    _MODEL_CACHE["key"] = key
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["model_a"] = model_a
    _MODEL_CACHE["metadata"] = metadata
    _MODEL_CACHE["device"] = device

    return WhisperModels(
        model=model,
        align_model=model_a,
        metadata=metadata,
        device=device
    )

# WhisperX 단어 타이밍 추출 함수
def extract_word_timings(audio_path: str, model, model_a, metadata, device: str, batch_size: int = 16):
    """
    [기능] 오디오 파일을 입력받아 WhisperX로 단어와 시작/끝 시간을 추출합니다.
    [리턴] [{'word': 'Hello', 'start': 0.5, 'end': 0.9}, ...] 형태의 리스트
    """
    print("⏳ 전사(Transcription) 시작...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    print("⏳ 강제 정렬(Alignment) 수행 중...")
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    word_segments = []
    for segment in aligned_result["segments"]:
        for word_info in segment["words"]:
            if "start" in word_info:
                word_segments.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"]
                })

    print(f"✅ WhisperX 완료: 총 {len(word_segments)}개 단어 추출")
    return word_segments

# WhisperX 모델 메모리 정리 함수
def cleanup_whisperx_models(model=None, model_a=None):
    """
    [기능] GPU 메모리 정리
    - API 서빙 단계에서는 보통 매 요청마다 cleanup 안하므로
    - 서버 shutdown 시 unload_whisperx_models()를 호출해 한 번에 정리
    """
    try:
        del model
        del model_a
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# WhisperX 모델 언로드 함수
def unload_whisperx_models():
    """
    [기능] 캐싱된 WhisperX 모델을 언로드합니다.
    - FastAPI shutdown 이벤트에서 호출 권장
    """
    global _MODEL_CACHE

    model = _MODEL_CACHE.get("model")
    model_a = _MODEL_CACHE.get("model_a")

    if model is None and model_a is None:
        _MODEL_CACHE = {"key": None, "model": None, "model_a": None, "metadata": None, "device": None}
        return

    try:
        del model
        del model_a
    except Exception:
        pass

    _MODEL_CACHE = {"key": None, "model": None, "model_a": None, "metadata": None, "device": None}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()