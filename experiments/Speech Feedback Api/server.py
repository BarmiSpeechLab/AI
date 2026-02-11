import os, tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import whisperx
import httpx

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# GPU 로딩
DEVICE = "cuda"  # GPU 없으면 CPU
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# 영어 인식 모델
asr_model = whisperx.load_model(
    "small.en", 
    DEVICE, 
    compute_type=COMPUTE_TYPE,
    vad_method="silero")

GMS_KEY = os.getenv("GMS_KEY")
GMS_CHAT_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/chat/completions"

# LLM 모델
async def generate_feedback_with_llm(payload: dict) -> str:
    """
    payload에는 정답 문장/전사결과/점수지표 등을 담아 LLM에게 피드백 생성 요청
    """
    if not GMS_KEY:
        return "GMS_KEY가 설정되지 않았습니다."

    system_prompt = (
        "너는 영어 발음/낭독 코치다. "
        "사용자가 '정답 문장'을 따라 읽었고, STT 결과와 말하기 지표가 주어진다. "
        "STT는 사용자의 발음을 문맥으로 보정해 정답처럼 보일 수 있으니, "
        "절대 '발음이 정확하다'고 단정하지 말고 지표(missing/extra/pauses/wpm) 기반으로 코칭해라.\n"
        "출력은 한국어로, 아래 형식을 지켜라:\n"
        "1) 총평(1~2문장)\n"
        "2) 체크 포인트(불릿 3~5개)\n"
        "3) 연습 드릴(짧게 2개)\n"
        "가능하면 정답 문장에서 사용자가 놓쳤을 가능성이 큰 단어를 집어서 구체적으로 말해라."
    )

    user_prompt = (
        "아래 데이터를 바탕으로 학습자에게 발음/낭독 피드백을 만들어줘.\n"
        f"{payload}"
    )

    body = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 700
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GMS_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GMS_KEY}",
                "Content-Type": "application/json",
            },
            json=body
        )
        resp.raise_for_status()
        data = resp.json()

    # OpenAI 호환 chat.completions 응답: choices[0].message.content
    return data["choices"][0]["message"]["content"]


@app.post("/grade")
async def grade(
    file: UploadFile = File(...),
    target_text: str = Form(...),  # 정답 문장
):
    # 1) 업로드 파일 저장 (webm 그대로 받음)
    suffix = os.path.splitext(file.filename or "recording.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        audio_path = f.name

    # 2) WhisperX 전사
    audio = whisperx.load_audio(audio_path)
    result = asr_model.transcribe(audio, batch_size=8)

    # 3) 강제정렬(단어 타임스탬프 정밀화)
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    aligned = whisperx.align(
                             result["segments"], align_model, metadata, audio, DEVICE, 
                             return_char_alignments=False
                            )

    # 4) 점수 계산 
    pred_words = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            if w.get("word"):
                pred_words.append(w)

    # 단어 문자열만 뽑기
    pred_tokens = [w["word"].strip().lower() for w in pred_words if w.get("start") is not None]
    tgt_tokens = [t.strip().lower() for t in target_text.split() if t.strip()]

    # 누락/삽입은 “단어 편집거리”로 계산
    missing = [t for t in tgt_tokens if t not in pred_tokens]
    extra   = [p for p in pred_tokens if p not in tgt_tokens]

    # 휴지/속도: 단어 타임스탬프 기반
    pauses = []
    for i in range(1, len(pred_words)):
        gap = pred_words[i]["start"] - pred_words[i-1]["end"]
        if gap >= 0.4:  # 400ms 이상이면 “멈칫”
            pauses.append({"after": pred_words[i-1]["word"], "gap_sec": round(gap, 3)})

    if pred_words:
        speech_start = pred_words[0]["start"]
        speech_end = pred_words[-1]["end"]
        duration = max(0.001, speech_end - speech_start)
        wpm = round(len(pred_tokens) / duration * 60, 1)
    else:
        duration, wpm = 0.0, 0.0

    # 점수화
    score = 100
    score -= 10 * len(missing)
    score -= 5 * len(extra)
    score -= 2 * len(pauses)

    # LLM에 보낼 요약 
    llm_payload = {
        "target_text": target_text,
        "stt_text": aligned.get("text", result.get("text", "")),
        "missing_words": missing,
        "extra_words": extra,
        "pauses": pauses[:10],   # 길어질 수 있으니 상위 10개만
        "wpm": wpm,
        "score_rule_based": score,
    }

    feedback = await generate_feedback_with_llm(llm_payload)

    return {
        "text": aligned.get("text", result.get("text", "")),
        "score": max(0, score),
        "missing": missing,
        "extra": extra,
        "pauses": pauses,
        "wpm": wpm,
        "words": pred_words, 
        "feedback": feedback,
    }


