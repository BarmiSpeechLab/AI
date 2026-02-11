import os
import numpy as np
import torch
import librosa
import torchcrepe


def extract_pitch_crepe(
    audio_path: str,
    sr: int = 16000,
    hop_length: int | None = None,
    fmin: int = 50,
    fmax: int = 600,
    crepe_model: str = "tiny",
    batch_size: int = 2048,
    device: str | None = None,
):
    """
    [기능] 오디오 파일 -> CREPE로 raw pitch / periodicity 추출
    [리턴]
      {
        "sr": 16000,
        "hop_length": 160,
        "hop_time": 0.01,
        "fmin": 50,
        "fmax": 600,
        "pitch": np.ndarray shape (T,),
        "periodicity": np.ndarray shape (T,)
      }

    - pitch/periodicity는 프레임 단위(시간축은 hop_time 간격)
    - align_merge.py에서 word start/end와 hop_time으로 인덱싱해서 합치면 됨
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if hop_length is None:
        hop_length = int(sr / 100)  # 10ms -> 100Hz

    # 오디오 로드
    audio, _ = librosa.load(audio_path, sr=sr)

    # torch tensor 변환
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)

    # CREPE 추론
    pitch, periodicity = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length,
        fmin,
        fmax,
        model=crepe_model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )

    pitch = pitch.squeeze().cpu().numpy()
    periodicity = periodicity.squeeze().cpu().numpy()

    hop_time = hop_length / sr

    return {
        "sr": sr,
        "hop_length": hop_length,
        "hop_time": hop_time,
        "fmin": fmin,
        "fmax": fmax,
        "pitch": pitch,
        "periodicity": periodicity,
    }


def cleanup_crepe(device: str | None = None):
    """
    [기능] GPU 메모리 정리(선택)
    - 배치/실험용으로 필요할 때만 호출
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
