from fastapi import FastAPI, HTTPException
import uvicorn

from demo import run_intonation_analysis

app = FastAPI()

TEST_FILE = "./wav_data/i_am_a_student_test.wav"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/output")
def output():
    try:
        data = run_intonation_analysis(TEST_FILE)
        return {"file": TEST_FILE, "result": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, reload=False)
