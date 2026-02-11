# pip install torchcrepe librosa numpy scipy

import torch
import torchcrepe
import librosa
import numpy as np
import os

def extract_pitch_for_words(audio_path, word_segments):
    """
    오디오 파일과 WhisperX의 단어 타이밍 정보를 받아서,
    각 단어 구간의 피치(Hz) 정보를 추가해 반환하는 함수
    """
    
    # 1. 설정 및 오디오 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CREPE] 사용 디바이스: {device}")
    
    # CREPE는 16kHz 처리를 권장합니다.
    sr = 16000
    audio, _ = librosa.load(audio_path, sr=sr)

    # 2. 전체 오디오에 대해 CREPE 실행 (한 번에 다 뽑는 게 효율적)
    print("[CREPE] 전체 오디오 피치 분석 중...")
    
    # 10ms(0.01초) 간격으로 점을 찍습니다. (그래프 해상도)
    hop_length = int(sr / 100) 
    
    # 오디오를 텐서로 변환
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    
    # CREPE 예측 (fmin, fmax는 사람 목소리 일반적 범위 50~2000Hz)
    # decoder=torchcrepe.decode.weighted_argmax 가 해상도가 더 좋습니다.
    pitch, periodicity = torchcrepe.predict(
        audio_tensor, 
        sr, 
        hop_length, 
        fmin=50, 
        fmax=600, 
        model='tiny', # 빠르고 가벼움 (정밀도 원하면 'full')
        batch_size=2048,
        device=device,
        return_periodicity=True
    )
    
    # GPU 텐서를 numpy 배열로 변환 (1차원 배열로 펼침)
    pitch = pitch.squeeze().cpu().numpy()
    periodicity = periodicity.squeeze().cpu().numpy()

    # 3. 단어별로 피치 잘라내기 (Slicing)
    results = []
    
    for word_info in word_segments:
        word = word_info['word']
        start_t = word_info['start']
        end_t = word_info['end']
        
        # 초(second) 단위를 배열 인덱스로 변환
        start_idx = int(start_t * 100)  # 0.01초 단위이므로 100을 곱함
        end_idx = int(end_t * 100)
        
        # 인덱스가 범위를 벗어나지 않게 안전장치
        if start_idx >= len(pitch): continue
        if end_idx > len(pitch): end_idx = len(pitch)
        
        # 해당 구간의 피치와 신뢰도(periodicity) 자르기
        segment_pitch = pitch[start_idx:end_idx]
        segment_prob = periodicity[start_idx:end_idx]
        
        # [핵심 로직] 신뢰도가 0.4 이상인(목소리가 확실한) 구간만 평균 계산
        # 무성음(s, t, k, p 등)이나 묵음은 피치가 튀므로 제외해야 함
        valid_mask = segment_prob > 0.05
        
        if np.any(valid_mask):
            avg_pitch = np.mean(segment_pitch[valid_mask])
            # 그래프용 데이터: 신뢰도 낮은 구간은 0이나 NaN으로 처리해서 그래프 끊어주기
            # 여기서는 0으로 처리 (프론트에서 0은 안 그리게 처리 추천)
            clean_curve = np.where(valid_mask, segment_pitch, 0).tolist()
        else:
            avg_pitch = 0.0
            clean_curve = [0] * len(segment_pitch)

        results.append({
            "word": word,
            "start": start_t,
            "end": end_t,
            "avg_pitch": round(float(avg_pitch), 2),  # 소수점 2자리 반올림
            "pitch_curve": clean_curve # 이 배열을 차트에 그리면 됩니다
        })
        
    return results

# ==========================================
# [테스트 실행 파트]
# ==========================================
if __name__ == "__main__":
    # 1. 테스트할 오디오 파일 경로
    test_audio = "../wav_data/i_am_a_student_test.wav"
    
    # 2. WhisperX에서 얻은 결과값 (테스트용 더미 데이터)
    whisper_output_mock = [
        {'word': 'I', 'start': 1.38, 'end': 2.63},
        {'word': 'am', 'start': 2.67, 'end': 2.81},
        {'word': 'a', 'start': 3.56, 'end': 3.75},
        {'word': 'dududon.', 'start': 3.99, 'end': 5.20}
    ]
    
    if os.path.exists(test_audio):
        print(">>> 피치 분석 시작...")
        final_data = extract_pitch_for_words(test_audio, whisper_output_mock)
        
        print("\n" + "="*40)
        print("   [최종 결과: 단어 + 타이밍 + 높낮이]   ")
        print("="*40)
        for item in final_data:
            print(f"단어: {item['word']:10} | "
                  f"구간: {item['start']}~{item['end']}s | "
                  f"평균 Hz: {item['avg_pitch']} Hz")
            # print(f"  -> 그래프 데이터(일부): {item['pitch_curve'][:5]}...") 
    else:
        print("❌ 오디오 파일을 찾을 수 없습니다.")