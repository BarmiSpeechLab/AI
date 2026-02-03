import os
import threading
from collections import deque
from typing import Deque, Dict, Tuple, Optional
from pathlib import Path

import whisper
import requests
from openai import OpenAI
from dotenv import load_dotenv

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]  # src/
PROJECT_DIR = BASE_DIR.parent                  # project root
AUDIO_DIR = Path(os.getenv("CONVERSATION_AUDIO_DIR", BASE_DIR / "uploads"))
OUT_DIR = Path(os.getenv("CONVERSATION_OUT_DIR", BASE_DIR / "conversation_out"))

# --- Config ---
MODEL_NAME = "tiny.en"
SAMPLE_RATE = 16000
MAX_TURNS = 4
SYSTEM_PROMPT = """
            당신은 청각장애/난청 학습자를 돕는 영어 회화 튜터 “설리번 선생님”입니다.
            발음 오류를 명확히 지적하고, 정확한 조음 방법을 구체적으로 안내합니다.
            친절하지만 엄격하게, 틀린 부분은 분명히 알려줍니다.

            규칙:
            - 문장은 짧고 단순하게 말합니다.
            - 모든 답변은 반드시 한국어로만 작성합니다. (영어 문장은 인용으로만 표시)
            - 이해 가능하면 자연스럽게 대화를 이어갑니다.
            - 이해가 어려우면 반드시 다시 말하게 하고, 추정한 문장은 “~인가요?”로 확인합니다.
            - 틀린 소리를 정확히 지적하고, 왜 틀렸는지 1~2문장으로 설명합니다.
            - 발음 팁은 반드시 구체적인 조음 지시로 1~2개 제공합니다.
            - 비판적인 표현은 금지합니다.

            [응답 포맷]
            - 이해 가능한 경우:
                [YOU SAID] "<사용자가 말한 영어 문장>"
                [REPLY] <영어 1문장>
                [NEXT] <영어 질문 1문장>

            - 이해가 어려운 경우:
                [FAIL] 이해하지 못했습니다. 다시 말해 주세요.
                [GUESS] 혹시 "<추정 문장>"인가요?
                [TIP] <발음 교정 1~2문장>
                [ASK] 다시 말해 주세요.

            [예시 1 — 이해 어려운 경우]
            사용자: "I am a duduoon."
            설리번: "[You said] I am a duduoon.
                    이해하지 못했습니다. 혹시 'I am a student'인가요?
                    student로 발음하고 싶으셨다면 'student'의 /st/는 혀끝을 윗잇몸에 가깝게 두고 시작해요.
                    다시 말해 주세요."

            [예시 2-이해 가능한 경우]
            사용자: "How are you?"
            설리번: "[You said] How are you?
                    I'm fine! And you?"
        """

# --- Singleton State ---
_STATE_LOCK = threading.Lock()
_STATE = {
    "model": None,
    "client": None,
    "histories": {},  # task_id -> deque[(user_text, assistant_text)]
}

def _init_state():
    load_dotenv()
    api_key = os.getenv("GMS_KEY")
    if not api_key:
        raise SystemExit("GMS_KEY is missing in .env")

    model = whisper.load_model(MODEL_NAME)
    client = OpenAI(
        api_key=api_key,
        base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1",
    )

    _STATE["model"] = model
    _STATE["client"] = client
    if _STATE["histories"] is None:
        _STATE["histories"] = {}

def get_state():
    if _STATE["model"] is None or _STATE["client"] is None:
        with _STATE_LOCK:
            if _STATE["model"] is None or _STATE["client"] is None:
                _init_state()
    return _STATE


def resolve_audio_path(file_name: str, audio_dir: Optional[Path] = None) -> str:
    p = Path(file_name)

    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())

    base = audio_dir or AUDIO_DIR
    candidate = base / file_name
    return str(candidate.resolve())



def transcribe_audio(model: whisper.Whisper, audio_path: str) -> str:
    result = model.transcribe(
        audio=audio_path,
        language="en",
        temperature=0.0,
        condition_on_previous_text=False,
    )
    return result.get("text", "").strip()


def build_messages(history: deque, user_text: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_text})
    return messages


def speak_tts(text: str, out_path: str) -> None:
    url = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('GMS_KEY')}",
    }
    payload = {
        "model": "gpt-4o-mini-tts",
        "input": text,
        "voice": "nova",
        "response_format": "mp3",
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)


def handle_conversation(task_id: str, file_name: str) -> dict:
    state = get_state()
    model = state["model"]
    client = state["client"]

    audio_path = resolve_audio_path(file_name)
    if not os.path.exists(audio_path):
        return {
            "type": "conversation",
            "taskId": task_id,
            "status": "FAIL",
            "error": f"audio file not found: {audio_path}",
            "analysisResult": None,
        }

    user_text = transcribe_audio(model, audio_path)
    if not user_text:
        return {
            "type": "conversation",
            "taskId": task_id,
            "status": "FAIL",
            "error": "empty transcription",
            "analysisResult": None,
        }

    histories: Dict[str, Deque[Tuple[str, str]]] = state["histories"]
    history = histories.get(task_id)
    if history is None:
        history = deque(maxlen=MAX_TURNS)
        histories[task_id] = history

    messages = build_messages(history, user_text)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.7,
    )
    assistant_text = resp.choices[0].message.content.strip()
    history.append((user_text, assistant_text))

    os.makedirs(OUT_DIR, exist_ok=True)
    tts_file = f"{task_id}.mp3"
    tts_path = OUT_DIR / tts_file
    speak_tts(assistant_text, str(tts_path))

    return {
        "type": "conversation",
        "taskId": task_id,
        "status": "SUCCESS",
        "error": None,
        "analysisResult": {
            "transcript": user_text,
            "assistant": assistant_text,
            "ttsFile": tts_file,
        },
    }


def handle_conversation_request(payload: dict) -> dict:
    task_id = payload.get("taskId") or payload.get("task_id")
    file_name = payload.get("file") or payload.get("filename")
    if not task_id or not file_name:
        return {
            "type": "conversation",
            "taskId": task_id or "",
            "status": "FAIL",
            "error": "taskId and file are required",
            "analysisResult": None,
        }
    return handle_conversation(task_id, file_name)