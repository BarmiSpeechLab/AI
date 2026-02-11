import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def draw_trend_pitch_graph(result_data):
    """
    [기능] 
    1. 곡선: 전체적인 '추세(Trend)'를 부드러운 곡선으로 표현
    2. 점: '평균'이 아니라 '단어 중간 시간'의 그래프 선 위에 표시
    """

    plt.figure(figsize=(15, 6))
    
    # Y축 범위 고정
    plt.ylim(50, 400) 

    for item in result_data:
        word = item['word']
        start_t = item['start']
        end_t = item['end']
        pitch_values = np.array(item['pitch_curve'])
        
        # 0Hz -> NaN 처리
        pitch_values_nan = np.where(pitch_values == 0, np.nan, pitch_values)
        time_axis = np.array([start_t + (i * 0.01) for i in range(len(pitch_values))])

        # 유효한 데이터만 골라내기
        mask = ~np.isnan(pitch_values_nan)
        valid_time = time_axis[mask]
        valid_pitch = pitch_values_nan[mask]
        n_points = len(valid_pitch)

        # 점을 찍을 시간 좌표 (단어의 정중앙)
        mid_time = start_t + (end_t - start_t) / 2
        
        # 점을 찍을 높이 좌표 (계산 전 초기화)
        mid_pitch = 0 

        # 추세선 그리기 및 중간 점 높이 계산
        if n_points >= 4:
            # 1. 3차 곡선 함수 만들기
            t_norm = valid_time - valid_time[0]
            coeffs = np.polyfit(t_norm, valid_pitch, deg=3)
            poly_func = np.poly1d(coeffs)
            
            # 2. 부드러운 선 그리기
            time_smooth_norm = np.linspace(t_norm.min(), t_norm.max(), 100)
            time_smooth_real = time_smooth_norm + valid_time[0]
            pitch_smooth = poly_func(time_smooth_norm)
            
            plt.plot(time_smooth_real, pitch_smooth, linewidth=3, color='#36A2EB', alpha=0.8)
            
            # 점의 높이를 곡선 함수에서 계산
            # 중간 시간(mid_time)을 정규화해서 함수에 넣으면, 곡선 위의 정확한 Y값이 나옴
            mid_pitch = poly_func(mid_time - valid_time[0])
            
        elif n_points >= 2:
            # 데이터가 적으면 직선 연결
            plt.plot(valid_time, valid_pitch, linewidth=2, color='#36A2EB', alpha=0.8)
            
            # 직선 구간에서의 중간값 찾기 (보간법)
            mid_pitch = np.interp(mid_time, valid_time, valid_pitch)
        
        else:
            # 데이터가 거의 없으면 그냥 평균값 사용
            if n_points > 0:
                mid_pitch = np.mean(valid_pitch)

        # 점 찍기
        if n_points > 0:
            # 계산된 위치(mid_time, mid_pitch)에 점 찍기
            # plt.scatter(mid_time, mid_pitch, s=100, color='#36A2EB', zorder=10, edgecolor='white', linewidth=2)
            
            # 텍스트 표시
            plt.text(mid_time, mid_pitch + 25, word, 
                     fontsize=12, fontweight='bold', ha='center', 
                     bbox=dict(facecolor='white', edgecolor='#36A2EB', boxstyle='round,pad=0.3', alpha=0.9),
                     zorder=10)

    plt.title("사용자 인토네이션 추세 (Pitch Trend)", fontsize=18, pad=20, fontweight='bold')
    plt.xlabel("시간 (Seconds)", fontsize=12)
    plt.ylabel("음의 높이 (Hz)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# 실행 테스트
if __name__ == "__main__":
    from demo import run_intonation_analysis
    
    test_file = "./wav_data/i_am_a_student.wav" 
    
    data = run_intonation_analysis(test_file)
    draw_trend_pitch_graph(data)