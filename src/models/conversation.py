import os
import time
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
MODEL_NAME = "small.en"
SAMPLE_RATE = 16000
MAX_TURNS = 4
SYSTEM_PROMPT = """
            당신은 청각장애/난청 학습자를 돕는 영어 회화 튜터 “설리번 선생님”입니다.
            발음 오류를 명확히 지적하고, 정확한 조음 방법을 구체적으로 안내합니다.
            친절하지만 엄격하게, 틀린 부분은 분명히 알려줍니다.

            규칙:
            - 문장은 짧고 단순하게 말합니다.
            - 설명/피드백은 한국어로 작성합니다.
            - 모든 답변은 반드시 한국어로만 작성합니다. (영어 문장은 인용으로만 표시)
            - [nextTurn]에는 theme과 prevTurn 문맥에 맞는 “다음 질문”을 영어 1 문장으로 생성합니다.
            - [THEME], [PREV_TURN] 내용을 반드시 고려해 다음 질문을 생성합니다.
            - 질문과 답변 유형이 맞지 않으면 추측하지 말고 다시 말해 달라고 안내합니다.
            - 이해가 어려우면 반드시 다시 말하게 하고, 추정한 문장은 “~인가요?”로 확인합니다.
            - 추측은 “매우 가까운 경우”에만 합니다. 멀면 추측 금지.
            - 답변 유형이 맞지 않을 때는 올바른 답변 템플릿을 짧게 제시합니다.
            (예: 직업 질문 → “I’m a ___.”, 취미 질문 → “My hobby is ___.”)
            - 틀린 소리를 정확히 지적하고, 왜 틀렸는지 1~2문장으로 설명합니다.
            - 발음 팁은 반드시 구체적인 조음 지시로 1~2개 제공합니다.
            - 비판적인 표현은 금지합니다.
            - 이해 가능하면 자연스럽게 대화를 이어갑니다.

            
            [출력 포맷]:
                [YOU SAID] "사용자가 말한 영어 문장"
                [nextTurn] <다음 질문(영어)>
                [feedback] <한국어 1~2문장: 피드백/재발화 요청/템플릿 제시>

            [응답 포맷]
            - 이해 가능한 경우:
                [YOU SAID] "<사용자가 말한 영어 문장>"
                [nextTurn] <영어 1문장(다음 질문)>
                [feedback] <칭찬 간단하게> 

            - 이해가 어려운 경우:
                [YOU SAID] "<사용자가 말한 영어 문장>"
                [nextTurn] 이해하지 못했습니다. 다시 말해 주세요
                [feedback] 혹시 "<추정 문장>"인가요? <발음 교정 1~2문장>

            [예시 1 — 이해 어려운 경우]
            사용자: "I am a duduoon."
            설리번: "[You said] I am a duduoon.
                    [nextTurn] 이해하지 못했습니다. 다시 말해 주세요.
                    [feedback] 혹시 'I am a student'인가요?
                    student로 발음하고 싶으셨다면 'student'의 /st/는 혀끝을 윗잇몸에 가깝게 두고 시작해요.
                    다시 말해 주세요."

            [예시 2-이해 가능한 경우]
            사용자: "How are you?"
            설리번: "[You said] How are you?
                    [nextTurn] I'm fine! And you?"
                    [feedback] 잘하셨어요! fine의 /f/는 윗니가 아랫입술에 살짝 닿아요. 지금처럼 하시면 됩니다!
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


def build_messages(history: Deque[Tuple[str, str]], user_text: str, prev_turn: str | None = None, theme: str | None = None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if theme or prev_turn:
        ctx = []
        if theme:
            ctx.append(f"[THEME] {theme}")
        if prev_turn:
            ctx.append(f"[PREV_TURN] {prev_turn}")
        messages.append({"role": "system", "content": "\n".join(ctx)})

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


def _extract_reply(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("[nextturn]"):
            return line.split("]", 1)[1].strip()
    return ""


def _extract_feedback(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("[feedback]"):
            return line.split("]", 1)[1].strip()
    return ""


def cleanup_tts(dir_path: Path, max_age_sec: int):
    if max_age_sec <= 0:
        return
    now = time.time()
    for p in dir_path.glob("*.mp3"):
        if now - p.stat().st_mtime > max_age_sec:
            p.unlink(missing_ok=True)


def handle_conversation(task_id: str, file_name: str, prev_turn: str | None = None, theme: str | None = None) -> dict:
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

    messages = build_messages(history, user_text, prev_turn=prev_turn, theme=theme)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.7,
    )
    assistant_text = resp.choices[0].message.content.strip()
    reply = _extract_reply(assistant_text)
    feedback = _extract_feedback(assistant_text) or assistant_text

    # 전사/이해 실패로 보이면 이전 질문을 다시 던짐
    if (not reply) or ("이해하지 못했습니다" in reply) or ("다시 말해" in reply):
        if prev_turn:
            reply = prev_turn

    history.append((user_text, assistant_text))

    os.makedirs(OUT_DIR, exist_ok=True)
    tts_file = f"{task_id}.mp3"
    tts_path = OUT_DIR / tts_file
    speak_tts(assistant_text, str(tts_path))

    # TTS 저장 끝난 뒤
    cleanup_tts(OUT_DIR, max_age_sec=86400)  # 24시간 보관

    return {
        "type": "conversation",
        "taskId": task_id,
        "status": "SUCCESS",
        "error": None,
        "analysisResult": {
            "transScript": user_text,
            "nextTurn": reply,     # [REPLY] 파싱 결과
            "theme": theme,
            "feedback": feedback
        },
    }


def handle_conversation_request(payload: dict) -> dict:
    task_id = payload.get("taskId") or payload.get("task_id")
    file_name = payload.get("filePath") or payload.get("file") or payload.get("filename")

    status = payload.get("status")
    err = payload.get("error")

    ar = payload.get("analysisResult") or {}
    prev_turn = ar.get("prevTurn")
    theme = ar.get("theme")

    if status and status != "SUCCESS":
        return {
            "type": "conversation",
            "taskId": task_id or "",
            "status": "FAIL",
            "error": err or "request status not SUCCESS",
            "analysisResult": None,
        }

    if not task_id or not file_name:
        return {
            "type": "conversation",
            "taskId": task_id or "",
            "status": "FAIL",
            "error": "taskId and filePath are required",
            "analysisResult": None,
        }

    return handle_conversation(task_id, file_name, prev_turn=prev_turn, theme=theme)
