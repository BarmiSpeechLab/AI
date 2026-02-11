# src/services/audio_io.py

from __future__ import annotations

import os
import tempfile
import contextlib
from typing import Generator

def bytes_to_audio_file(
    audio_bytes: bytes,
    suffix: str = ".wav",
) -> str:
    """
    [기능] 오디오 bytes를 임시 파일로 저장하고 그 파일 경로를 반환합니다.

    [주의]
    - 반환된 파일은 사용 후 반드시 삭제해야 합니다.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes is empty")

    # 임시 파일 생성
      # fd: OS 레벨 파일 디스크립터(핸들)
      # path: 임시 파일 경로(문자열)
    fd, path = tempfile.mkstemp(suffix=suffix)
    # mkstemp는 OS-level fd를 주므로, open으로 다시 쓰고 fd 닫기
    os.close(fd)

    with open(path, "wb") as f:
        f.write(audio_bytes)

    return path


def safe_remove(path: str) -> None:
    """[기능] 파일이 존재하면 삭제"""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

@contextlib.contextmanager
def temp_audio_file(file_bytes: bytes, suffix: str = ".wav") -> Generator[str, None, None]:
    """
    [기능] with 문과 함께 사용하여 임시 파일을 생성하고,
          사용이 끝나면(블록 탈출 시) 자동으로 삭제합니다.
    """
    path = bytes_to_audio_file(file_bytes, suffix) # 파일 생성
    try:
        yield path # 파일 경로를 빌려줌
    finally:
        safe_remove(path) # 사용 후 삭제
