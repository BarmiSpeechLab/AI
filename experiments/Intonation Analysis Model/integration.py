import os
import torch
import torchcrepe
import librosa
import numpy as np
import whisperx

def run_intonation_analysis(audio_path):
    """
    [기능] 오디오 -> WhisperX(타이밍) -> CREPE(피치) -> Polyfit(곡선 스무딩)
    [리턴] 곡선 데이터 반환
    """
    
    # 0. 기본 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(audio_path):
        return {"error": "File not found"}


    # 1. WhisperX (단어 타이밍 추출)
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model("small.en", device, compute_type=compute_type, vad_method="silero")
    
    audio_wx = whisperx.load_audio(audio_path)
    result = model.transcribe(audio_wx, batch_size=16)
    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_wx, device, return_char_alignments=False)
    
    del model
    del model_a
    torch.cuda.empty_cache()

    word_segments = []
    for segment in aligned_result["segments"]:
        for word_info in segment["words"]:
            if "start" in word_info: 
                word_segments.append(word_info)


    # 2. CREPE (Raw 피치 추출)
    sr = 16000
    hop_length = int(sr / 100) 
    fmin, fmax = 50, 600
    
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # 볼륨 정규화
    # max_val = np.abs(audio).max()
    # if max_val > 0 and max_val < 0.9:
    #     audio = audio / max_val
    
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    pitch, periodicity = torchcrepe.predict(
        audio_tensor, sr, hop_length, fmin, fmax, 
        model='tiny', batch_size=2048, device=device, return_periodicity=True
    )
    
    pitch = pitch.squeeze().cpu().numpy()
    periodicity = periodicity.squeeze().cpu().numpy()


    # 3. 데이터 병합 및 곡선 스무딩 (Polyfit)
    final_output = []
    
    for w in word_segments:
        word = w['word']
        start_t = w['start']
        end_t = w['end']
        
        # 인덱스 변환
        start_idx = int(start_t * 100)
        end_idx = int(end_t * 100)
        
        if start_idx >= len(pitch): continue
        if end_idx > len(pitch): end_idx = len(pitch)
        
        raw_pitch = pitch[start_idx:end_idx]
        raw_prob = periodicity[start_idx:end_idx]
        
        # 시간축 생성 (Raw)
        raw_time = np.array([start_t + (i * 0.01) for i in range(len(raw_pitch))])
        
        # 유효 데이터 필터링 (신뢰도 0.05 이상)
        mask = raw_prob > 0.05
        valid_time = raw_time[mask]
        valid_pitch = raw_pitch[mask]
        
        # 결과를 담을 변수 초기화
        final_time_list = []
        final_pitch_list = []
        
        # 데이터가 충분하면 -> 다항 회귀(Polyfit)로 부드러운 곡선 생성
        if len(valid_pitch) >= 4:
            t_base = valid_time[0]
            t_norm = valid_time - t_base
            
            # 3차 곡선 피팅
            coeffs = np.polyfit(t_norm, valid_pitch, deg=3)
            poly_func = np.poly1d(coeffs)
            
            # 곡선 데이터 생성 (50등분)
            smooth_time_norm = np.linspace(t_norm.min(), t_norm.max(), 50)
            smooth_pitch = poly_func(smooth_time_norm)
            
            # 리스트로 변환 (반올림 처리)
            final_time_list = np.round(smooth_time_norm + t_base, 3).tolist()
            final_pitch_list = np.round(smooth_pitch, 2).tolist()
            
        # 데이터가 적으면 -> 직선 연결 (원본 데이터 사용)
        elif len(valid_pitch) >= 1:
             final_time_list = np.round(valid_time, 3).tolist()
             final_pitch_list = np.round(valid_pitch, 2).tolist()
        
        # 최종 결과 저장
        final_output.append({
            "word": word,
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "curve_time": final_time_list,
            "curve_pitch": final_pitch_list
        })

    return final_output


# 실행 테스트
if __name__ == "__main__":
    test_file = "./wav_data/i_like_to_dance.wav"
    
    print("⏳ 분석 중...")
    data = run_intonation_analysis(test_file)
    
    if data:
        print("\n 프론트엔드로 보낼 JSON 데이터")
        import json
        # 데이터 구조 확인
        print(json.dumps(data[0], indent=2))