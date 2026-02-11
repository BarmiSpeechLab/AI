import whisperx
import torch

# 1. 설정 (GPU가 있으면 cuda, 없으면 cpu 자동 선택)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"현재 사용 디바이스: {device}")

# 오디오 파일 경로
audio_file = "../wav_data/i_am_a_student_test.wav" 

# 2. Whisper 모델 로드 (영어 전용 small.en 사용)
# compute_type은 GPU 메모리가 넉넉하면 "float16", 부족하면 "int8"
compute_type = "float16" if device == "cuda" else "int8"

print("1. 텍스트 인식(Transcription) 중...")
model = whisperx.load_model("small.en", 
                            device, 
                            compute_type=compute_type, 
                            vad_method="silero")

# 오디오 로드 및 전사(Transcribe)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=16)

# 3. 정렬(Alignment) 모델 로드 및 수행
print("2. 강제 정렬(Alignment) 수행 중...")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# 정렬 수행
aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# 4. 결과 확인 (우리가 필요한 데이터만 출력)
print("\n" + "="*30)
print("   [중간 결과: 단어별 타이밍]   ")
print("="*30)

# 전체 문장(segments) 안에 있는 단어(words)들을 꺼내봅니다.
for segment in aligned_result["segments"]:
    for word_info in segment["words"]:
        # start/end가 인식된 경우만 출력
        if "start" in word_info:
            print(f"단어: {word_info['word']:15} | 시작: {word_info['start']:.2f}초 ~ 끝: {word_info['end']:.2f}초")
        else:
            print(f"단어: {word_info['word']:15} | (타이밍 인식 불가 - 묵음 등)")