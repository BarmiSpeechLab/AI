import numpy as np

def merge_words_with_pitch_curve(
    word_segments: list[dict],
    pitch_result: dict,
    periodicity_threshold: float = 0.05,
    polyfit_deg: int = 3,
    smooth_points: int = 50,
):
    """
    [기능] WhisperX 단어 타이밍(word_segments) + CREPE pitch_result를 병합하고,
          단어별 pitch curve를 polyfit으로 스무딩하여 반환합니다.

    [입력]
      - word_segments: [{'word': str, 'start': float, 'end': float}, ...]
      - pitch_result: pitch_crepe.extract_pitch_crepe()의 리턴 dict
        {
          "hop_time": 0.01,
          "pitch": np.ndarray,
          "periodicity": np.ndarray,
          ...
        }

    [리턴]
      [
        {
          "word": "...",
          "start": 0.12,
          "end": 0.34,
          "curve_time": [...],
          "curve_pitch": [...]
        },
        ...
      ]
    """
    pitch = pitch_result["pitch"]
    periodicity = pitch_result["periodicity"]
    hop_time = float(pitch_result["hop_time"])

    final_output = []

    for w in word_segments:
        word = w["word"]
        start_t = w["start"]
        end_t = w["end"]

        # 인덱스 변환 
        start_idx = int(start_t / hop_time)
        end_idx = int(end_t / hop_time)

        if start_idx >= len(pitch):
            continue
        if end_idx > len(pitch):
            end_idx = len(pitch)

        raw_pitch = pitch[start_idx:end_idx]
        raw_prob = periodicity[start_idx:end_idx]

        # 시간축 생성 -> 0.01s 간격
        raw_time = np.array([start_t + (i * hop_time) for i in range(len(raw_pitch))])

        # 유효 데이터 필터링 (신뢰도 threshold 이상)
        mask = raw_prob > periodicity_threshold
        valid_time = raw_time[mask]
        valid_pitch = raw_pitch[mask]

        # 결과를 담을 변수 초기화
        final_time_list = []
        final_pitch_list = []

        # 데이터가 충분하면 -> 다항 회귀(Polyfit)로 부드러운 곡선 생성
        if len(valid_pitch) >= 4:
            t_base = valid_time[0]
            t_norm = valid_time - t_base

            coeffs = np.polyfit(t_norm, valid_pitch, deg=polyfit_deg)
            poly_func = np.poly1d(coeffs)

            smooth_time_norm = np.linspace(t_norm.min(), t_norm.max(), smooth_points)
            smooth_pitch = poly_func(smooth_time_norm)

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
            "curve_pitch": final_pitch_list,
        })

    return final_output
