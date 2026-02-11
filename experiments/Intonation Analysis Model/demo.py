import os
import torch
import torchcrepe
import librosa
import numpy as np
import whisperx

def run_intonation_analysis(audio_path):
    """
    [기능] 오디오 파일을 입력받아 '단어별 타이밍'과 '인토네이션(Pitch)'을 통합 분석합니다.
    [리턴] 프론트엔드 그래프 그리기용 JSON 리스트
    """
    
    # 기본 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n 분석 시작 (Device: {device})")
    print(f" 파일 경로: {audio_path}")

    if not os.path.exists(audio_path):
        print(f"파일을 찾을 수 없습니다 -> {audio_path}")
        return []
    
    # 1. 단어 타이밍(Alignment) 구하기
    print("WhisperX: 텍스트 및 타이밍 시작")
    
    # 1-1. 모델 로드 (영어 전용 small.en + VAD silero 사용)
    # 메모리 최적화: GPU면 float16, CPU면 int8
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = whisperx.load_model("small.en", device, compute_type=compute_type, vad_method="silero")
    
    # 1-2. 오디오 로드 및 전사(Transcribe)
    audio_wx = whisperx.load_audio(audio_path)
    result = model.transcribe(audio_wx, batch_size=16)
    
    # 1-3. 강제 정렬(Alignment)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_wx, device, return_char_alignments=False)
    
    # 1-4. 메모리 정리
    del model
    del model_a
    torch.cuda.empty_cache()

    # WhisperX 결과 평탄화
    word_segments = []
    for segment in aligned_result["segments"]:
        for word_info in segment["words"]:
            if "start" in word_info: 
                word_segments.append(word_info)

    print(f"WhisperX 완료: 단어 {len(word_segments)}개 발견.")


    # 2. CREPE로 피치(Pitch) 추출하기
    print("\nCREPE: 높낮이(Pitch) 분석 시작")
    
    # 2-1. 설정
    sr = 16000
    hop_length = int(sr / 100) 
    fmin = 50
    fmax = 600   # 천장 600Hz -> 2000Hz 하니까 인식이 너무 불안정함
    
    # 2-2. Librosa로 오디오 로드
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # 2-3. 볼륨 정규화 (Normalization) -> 오디오 소리 너무 작은 경우를 대비한 처리
    # max_val = np.abs(audio).max()
    # if max_val > 0:
    #     scale_factor = 1.0 / max_val if max_val < 0.9 else 1.0
    #     audio = audio * scale_factor
    #     if scale_factor > 1.0:
    #         print(f"   -> [INFO] 오디오 증폭 적용됨 (x{scale_factor:.2f})")
    
    # 2-4. CREPE 실행
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    
    pitch, periodicity = torchcrepe.predict(
        audio_tensor, sr, hop_length, fmin, fmax, 
        model='tiny', batch_size=2048, device=device, return_periodicity=True
    )
    
    pitch = pitch.squeeze().cpu().numpy()
    periodicity = periodicity.squeeze().cpu().numpy()


    # 3. 데이터 합치기 (Merging)
    print("\n⏳ 데이터 병합 중...")
    final_output = []
    
    for w in word_segments:
        word = w['word']
        start_t = w['start']
        end_t = w['end']
        
        start_idx = int(start_t * 100)
        end_idx = int(end_t * 100)
        
        if start_idx >= len(pitch): continue
        if end_idx > len(pitch): end_idx = len(pitch)
        
        segment_pitch = pitch[start_idx:end_idx]
        segment_prob = periodicity[start_idx:end_idx]
        
        # 신뢰도 0.05 이상
        valid_mask = segment_prob > 0.05
        
        if np.any(valid_mask):
            avg_pitch = np.mean(segment_pitch[valid_mask])
            clean_curve = np.where(valid_mask, segment_pitch, 0).tolist()
        else:
            avg_pitch = 0.0
            clean_curve = [0] * len(segment_pitch)

        final_output.append({
            "word": word,
            "start": start_t,
            "end": end_t,
            "avg_pitch": round(float(avg_pitch), 2),
            "pitch_curve": clean_curve 
        })

    return final_output

# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    # 테스트 파일 경로
    test_file = "./wav_data/i_am_a_student_test.wav"
    
    result_data = run_intonation_analysis(test_file)
    
    if result_data:
        print("\n" + "="*50)
        print(f"총 {len(result_data)}개 구간 데이터 생성됨")
        print("="*50)
        
        for item in result_data:
            print(f"단어: {item['word']:10} | "
                  f"구간: {item['start']:.2f}~{item['end']:.2f}초 | "
                  f"평균 Hz: {item['avg_pitch']} Hz")