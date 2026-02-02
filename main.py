import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from src.services.audio_io import temp_audio_file
from src.services.speech_pipeline import analyze_speech_stream
from src.models.stt_whisper import get_whisperx_models
from src.models.phoneme import load_phoneme_models

app = FastAPI(title="Speech Analysis API")

# 전역 변수
loaded_models = None
phoneme_models = None

@app.on_event("startup")
async def startup_event():
    global loaded_models, phoneme_models
    print("⏳ 모델 로딩 중...")
    loaded_models = get_whisperx_models(model_name="small.en", vad_method="silero")
    #phoneme_models = load_phoneme_models("src/fine_tuned_model")
    phoneme_models = load_phoneme_models("wishkim/wav2vec2-l2arctic-phoneme")
    print("✅ 모델 로딩 완료!")


@app.on_event("shutdown")
async def shutdown_event():
    global loaded_models
    loaded_models = None
    print("🛑 서버 종료")


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    taskId: str = Form(...),
    analysisRequest: str = Form(...)
):

    if not file.filename:
        return {"error": "파일 이름이 없습니다"}
    
    # 1. 파일 읽기 (Bytes)
    audio_bytes = await file.read()
    analysis_request = json.loads(analysisRequest)
        
    # 2. 제너레이터 래퍼
    def stream_with_cleanup():
        with temp_audio_file(audio_bytes, suffix=".wav") as audio_path:
            for chunk in analyze_speech_stream(
                audio_path=audio_path,
                loaded_models=loaded_models,
                phoneme_models=phoneme_models,
                analysis_request=analysis_request,
                mode="all"
            ):
                try:
                    payload = json.loads(chunk)
                    t = payload.get("type")

                    # 1) 에러 처리
                    if t == "error":
                        task = payload.get("task")
                        # task 에러는 type 유지 + FAIL
                        if task in ("pron", "inton", "feedback"):
                            mapped_type = "llm" if task == "feedback" else task
                            yield json.dumps({
                                "type": mapped_type,
                                "taskId": taskId,
                                "status": "FAIL",
                                "error": payload.get("message"),
                                "analysisResult": None
                            }, ensure_ascii=False) + "\n"
                        else:
                            # 초기 단계 에러(WhisperX 포함)
                            yield json.dumps({
                                "type": "error",
                                "taskId": taskId,
                                "status": "FAIL",
                                "error": payload.get("message"),
                                "analysisResult": None
                            }, ensure_ascii=False) + "\n"
                        continue
                                    
                    # 2) 정상 결과
                    if t == "feedback":
                        t = "llm"

                    yield json.dumps({
                        "type": t,
                        "taskId": taskId,
                        "status": "SUCCESS",
                        "error": None,
                        "analysisResult": payload.get("data")
                    }, ensure_ascii=False) + "\n"

                except Exception as e:
                    # 파싱/래핑 에러도 FAIL
                    yield json.dumps({
                        "type": "error",
                        "taskId": taskId,
                        "status": "FAIL",
                        "error": str(e),
                        "analysisResult": None
                    }, ensure_ascii=False) + "\n"

    # 3. StreamingResponse 반환
    return StreamingResponse(
        stream_with_cleanup(), 
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)