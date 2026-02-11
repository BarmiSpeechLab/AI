import os
import torch
import whisperx
import eng_to_ipa as ipa_lib
import re

# WhisperX: 오디오에서 단어와 타이밍 추출
def extract_word_timings(audio_path):
    """
    [기능] 오디오 파일을 입력받아 WhisperX로 단어와 시작/끝 시간을 추출합니다.
    [리턴] [{'word': 'Hello', 'start': 0.5, 'end': 0.9}, ...] 형태의 리스트
    """
    
    # 0. 기본 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[WhisperX] 사용 디바이스: {device}")

    # 1. 모델 로드 (영어 전용 small.en + VAD silero)
    print("⏳ 모델 로딩 및 전사(Transcription) 시작...")
    compute_type = "float16" if device == "cuda" else "int8"
    
    # 모델 로드
    model = whisperx.load_model("small.en", device, compute_type=compute_type, vad_method="silero")
    
    # 오디오 로드 및 전사
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    
    # 2. 강제 정렬 (Alignment)
    print("⏳ 강제 정렬(Alignment) 수행 중...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # 메모리 정리
    del model
    del model_a
    torch.cuda.empty_cache()

    word_segments = []
    for segment in aligned_result["segments"]:
        for word_info in segment["words"]:
            if "start" in word_info:
                word_segments.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"]
                })

    print(f"WhisperX 완료: 총 {len(word_segments)}개 단어 추출")
    return word_segments


# IPA 변환: 영어 단어를 발음기호로 변환
def convert_to_ipa(word: str) -> str:
    """
    [기능] 영어 단어 문자열을 받아서 IPA(발음기호)로 변환합니다.
    [리턴] "/həˈloʊ/" 형태의 문자열 (실패 시 빈 문자열 반환)
    """
    # 1. 입력 방어 코드
    w = (word or "").strip()
    if not w:
        return ""

    # 2. 특수문자 제거 (WhisperX는 "student." 처럼 점을 찍어줌)
    # 문자와 공백을 제외한 모든 것 제거
    clean_word = re.sub(r'[^\w\s]', '', w)

    # 3. 변환 수행
    try:
        # retrieve_all=False: 가장 일반적인 발음 하나만 가져옴
        ipa = ipa_lib.convert(clean_word)
    except Exception as e:
        print(f"⚠️ IPA 변환 에러 ({word}): {e}")
        return ""

    # 4. 변환 결과 검증
    # eng_to_ipa는 변환 실패 시 단어 끝에 '*'를 붙여서 반환합니다.
    if "*" in ipa:
        return ""

    # 여러 발음이 나올 경우 첫 번째 것만 사용
    if " " in ipa:
        ipa = ipa.split()[0]

    return f"/{ipa}/"


# 테스트
if __name__ == "__main__":
    test_audio = "../wav_data/i_am_a_student.wav"
    
    print("1단계: WhisperX 단어 추출 테스트")
    
    whisper_results = extract_word_timings(test_audio)
    
    if not whisper_results:
        print("WhisperX 결과가 없습니다.")
    else:
        # IPA 변환 및 결과 확인
        print("2단계: IPA 변환 및 최종 결과 확인")
        print(f"{'Word (Raw)':<15} | {'Clean IPA':<15} | {'Timing (sec)':<15}")

        for item in whisper_results:
            original_word = item['word']
            
            # IPA 변환 함수 호출
            ipa_result = convert_to_ipa(original_word)
            
            # 결과 출력
            timing_str = f"{item['start']:.2f} ~ {item['end']:.2f}"
            print(f"{original_word:<15} | {ipa_result:<15} | {timing_str:<15}")

        print("\n✅ 테스트 완료!")