# AI 서버 설정 가이드  

본 가이드는 **[바르미] AI Gateway와 연동되는 외부 AI 서버**를  
개인 노트북 또는 로컬 PC 환경에서 실행하기 위한 설정 절차를 정리한다.

AI 서버는 **모델 로딩 및 추론을 담당**하며,  
서비스 백엔드나 AI Gateway와는 **HTTP API 통신으로만 연결**된다.


## 1. 역할 정리

외부 AI 서버의 책임은 다음과 같다.

- WhisperX, CREPE 등 **AI 분석 모델 로딩**
- 입력 데이터 기반 **추론 수행**
- 분석 결과를 **JSON 형태로 반환**
- FastAPI 기반 API 제공 (`/health`, `/output` 등)


## 2. 프로젝트 기본 구조 예시
```
ai-server/
├── ai_server.py # FastAPI 서버 엔트리포인트
├── analysis/
│ └── intonation.py # WhisperX / CREPE 분석 로직
├── wav_data/
│ └── sample.wav
├── requirements.txt
└── README.md
```

## 4. 가상환경 생성 및 활성화

### conda 사용 시
```bash
conda create -n ai_env python=3.10 -y
conda activate ai_env
```
### venv 사용시 
```bash
python -m venv ai_env
source ai_env/bin/activate   # macOS/Linux
ai_env\Scripts\activate      # Windows
```
## 5. 필수 패키지 설치
### 기본 서버 패키지
```bash
pip install fastapi uvicorn httpx
```
## 6. FastAPI 서버 구현 예시
### ai_server.py

```python
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
```

## 7. 서버 실행
```bash
python ai_server.py
```
- 정상 실행 시 로그:
`Uvicorn running on http://0.0.0.0:8000`

## 8. 로컬 테스트
### Health 체크
```
curl http://localhost:8000/health
```
### Output 테스트
```
curl -X POST http://localhost:8000/output \
  -H "Content-Type: application/json" \
  -d '{"test":"data"}'
```

## 9. ngrok을 이용한 외부 노출
### ngrok 설치
   - https://ngrok.com/download

### 인증 토큰 등록 (1회)
```bash
ngrok config add-authtoken <YOUR_TOKEN>
```
### 터널 실행
```bash
ngrok http 8000
```

출력 예:
```Forwarding  https://xxxx-xxxx.ngrok-free.dev -> http://localhost:8000```

## 10. AI Gateway와의 연동 규칙
AI 서버는 반드시 다음 엔드포인트를 제공해야 한다.

Method	| Path	   |  설명
GET	    | /health  |  서버 상태 확인
POST	| /output  |  AI 분석 결과 반환

- 응답은 반드시 JSON
- HTTP status는 성공 시 200 OK
- 내부 에러는 FastAPI Exception으로 처리