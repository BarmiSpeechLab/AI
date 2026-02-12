## 💡AI 레포지토리입니다.
청각장애인을 위한 시각 기반 영어 발음 교정 서비스, '바르미'의 AI 엔진 레포지토리입니다.

바르미는 단순한 음성 인식을 넘어, 음소(Phoneme) 단위의 정밀 분석과 시각적 피드백을 통해 청각적 제약이 있는 사용자가 직관적으로 영어 발음을 교정할 수 있도록 돕습니다.

### 🛠 Tech Stack
- Core: Python 3.9+ / FastAPI
- Audio Processing: Librosa, PySoundFile
- Models: 
  - **WhisperX**: 정밀한 타임스탬프 기반 음성 인식 (STT)
  - **Wav2Vec2 (Fine-tuned)**: L2-Arctic 데이터셋 기반 음소(Phoneme) 추출
  - **CREPE**: 고해상도 피치(F0) 추정 및 억양 분석
- Inference: PyTorch, HuggingFace Transformers

### 📂 Project Structure
```sh
AI/
├── src/
│   ├── models/           # 핵심 AI 모델 추론 로직 (Whisper, Phoneme, Pitch 등)
│   ├── services/         # 비즈니스 로직 및 파이프라인 (Audio IO, Pipeline 관리)
│   └── uploads/          # 임시 오디오 파일 저장소
├── fine-tuning/          # Wav2Vec2 파인튜닝 실험 및 결과
├── main.py               # FastAPI 엔트리 포인트 (API 서버)
└── requirements.txt      # 의존성 패키지 목록
```

### 🚀 Core Components
1. 정밀 발음 분석 파이프라인 (`/analyze`)

    하나의 API 엔드포인트에서 STT, 음소 분석, 피치 추출, LLM 피드백을 동시에 수행하여 NDJSON 형태로 스트리밍합니다.
    - **음소 단위 발음 평가 (Phoneme-level Evaluation)**
      - **Core**: Fine-tuned `Wav2Vec2-base-960h` (L2-Arctic)
      - **Detail**: Whisper와 같은 일반 ASR의 문맥 보정(Context Correction) 기능을 배제하고, 사용자가 발음한 소리 그대로를 음소 단위로 분해하여 실제 조음 오류를 포착합니다.

    - **억양 및 피치 분석 (Prosody Analysis)**
      - **Core**: `CREPE` (High-resolution Pitch Estimation)
      - **Detail**: 음성의 기본 주파수($F_0$)를 추출하여 원어민의 억양 곡선과 비교합니다. 이를 통해 소리를 듣기 어려운 사용자에게 시각적인 억양 가이드를 제공합니다.

    - **지능형 조음 피드백 (Articulatory Feedback)**
      - **Core**: LLM (Large Language Model)
      - **Detail**: Few-Shot Prompting을 통해 "혀의 위치를 더 높게", "입술을 더 둥글게"와 같이 구체적인 조음 기관의 움직임을 맞춤형으로 가이드합니다.

    - **WhisperX 통합 분석**
      - **Detail**: 정밀한 타임스탬프를 활용하여 문장 내 특정 단어와 음소의 위치를 일치시킵니다.

2. 인터랙티브 대화 서비스 (`/conversation`)
    
    발음 분석 파이프라인을 대화형 인터페이스에 통합하여 실전 교정 학습을 지원합니다.
    - **Scenario-based Learning**: 분석 결과와 AI 응답을 결합하여 사용자의 발음 상태에 최적화된 대화 흐름을 생성합니다.
    - **Low-Latency Streaming**: 오디오 처리와 텍스트 생성을 병렬로 처리하여 자연스러운 대화 경험을 제공합니다.

### 📄 Technical Documents
모델 학습 과정 및 상세 연구 내용은 아래의 **기술 개발 문서(Notion)** 에서 확인하실 수 있습니다.

🏗**️ System Architecture & Implementation**
- [🔗 AI 분석 파이프라인 통합 설계]()

  : `/analyze` API에서 WhisperX, Wav2Vec2, CREPE가 어떤 순서로 데이터를 주고받는지, 그리고 왜 StreamingResponse를 선택했는지에 대한 설계 문서
- [🔗 음소-피치 정렬 로직 (Alignment strategy)]()

  : WhisperX의 타임스탬프와 CREPE의 피치 데이터를 어떻게 매칭시켜서 특정 음소의 억양 곡선을 추출했는지에 대한 기술적 해결 방식
- [🔗 실시간 데이터 통신 규격 (NDJSON)]()

  : 프론트엔드와의 실시간 통신을 위해 설계한 커스텀 데이터 구조(type, status, analysisResult)에 대한 규격서

**🧪 Research & Model Reviews**
- [🔗 Wav2Vec2 음소 인식 파인튜닝 과정](https://faceted-animantarx-5f1.notion.site/fine-tuning?source=copy_link)

  : 파인튜닝 지표(PER) 및 데이터 전처리 과정
- [🔗 억양 분석 모델(CREPE) 기초 연구](https://www.notion.so/1-Intonation-Model-Review-2eefc74069a68043bba5f0e04e8bb209?source=copy_link)

  : CREPE 모델 활용법 및 초기 테스트 기록
- [🔗 IPA 음소 인식 모델 연구](https://www.notion.so/1-2f0fc74069a6806db6e5dbcf09428284?source=copy_link)

  : 음소 인식을 위한 모델 라이브러리 선정 및 사용법 조사
