from __future__ import annotations
import os
import sys
import json
import warnings
import logging

# --- 시스템 설정 ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("whisperx").setLevel(logging.ERROR)

from typing import Any, Dict, List, Literal, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 모듈 임포트 ---
from src.models.stt_whisper import extract_word_timings, WhisperModels
from src.models.pitch_crepe import extract_pitch_crepe
from src.models.align_merge import merge_words_with_pitch_curve
from src.models.g2p import text_to_phonemes
from src.models.pronunciation import phonemes_to_hangul_ipa
from src.models.llm_feedback import generate_llm_feedback 

Mode = Literal["pron", "inton", "all"]


def _process_pronunciation(words: List[str]) -> List[Dict[str, Any]]:
    """
    단어 리스트를 받아 발음 기호(IPA) 및 한글 발음으로 변환
    """
    results = []

    for idx, w in enumerate(words):
        # 1) 텍스트를 음소(ARPAbet)로 변환 -> upl
        upl = text_to_phonemes(w)

        # 2) 음소를 기반으로 한글 표기 및 IPA 추출
        ukor, _ipa_str, uipa = phonemes_to_hangul_ipa(upl)

        word_data = {
            "word": w.upper().replace(".", ""),
            "phonemes": [{"upl": p, "uipa": i} for p, i in zip(upl, uipa)],
            "ukor": ukor
        }
        results.append(word_data)

    return results


def _process_intonation(
    audio_path: str, 
    word_segments: List[Dict[str, Any]], 
    device: str
) -> List[Dict[str, Any]]:
    """
    인토네이션 분석 실행
    -> 음성 파일에서 피치(Pitch)를 추출하고 단어별 타이밍에 매칭
    """
    pitch_result = extract_pitch_crepe(audio_path, device=device)
    return merge_words_with_pitch_curve(word_segments, pitch_result)


def analyze_speech_stream(
    audio_path: str,
    loaded_models: WhisperModels,   # 이미 로딩된 모델을 받음
    reference_data: Dict[str, str], # 분석 비교를 위한 정답 데이터
    mode: Mode = "all",
) -> Generator[str, None, None]:
    """
    [Main Pipeline] 음성 분석 파이프라인 메인 함수
    1. WhisperX (공통)
    2. Pronunciation (G2P)
    3. Intonation (CREPE)
    4. LLM Feedback (gpt-4.1-nano)
    """
    
    # 1. WhisperX: 공통 전처리 단계
    model = loaded_models.model
    model_a = loaded_models.align_model
    metadata = loaded_models.metadata
    device = loaded_models.device
    
    try:
        word_segments = extract_word_timings(
            audio_path=audio_path,
            model=model,
            model_a=model_a,
            metadata=metadata,
            device=device,
            batch_size=16,
        )
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        return

    words = [w["word"] for w in word_segments]
    pron_result_for_feedback = None  # 결과를 담을 변수 

    # 작업 실행 플래그
    do_pron = mode in ("pron", "all")
    do_into = mode in ("inton", "all")

    # 2. 비동기 병렬 분석 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_map = {}

        if do_pron:
            # 발음 분석 작업 제출
            f_pron = executor.submit(_process_pronunciation, words)
            future_map[f_pron] = "pron"
        
        if do_into:
            # 인토네이션 분석 작업 제출
            f_into = executor.submit(_process_intonation, audio_path, word_segments, device)
            future_map[f_into] = "inton"

        # 작업이 완료되는 순서대로 클라이언트에 전송
        for future in as_completed(future_map):
            task_type = future_map[future]
            try:
                result_data = future.result()

                # 발음 결과는 피드백을 위해 따로 저장
                if task_type == "pron":
                    pron_result_for_feedback = result_data
                
                yield json.dumps({
                    "type": task_type,
                    "data": result_data
                }, ensure_ascii=False) + "\n"
                
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "task": task_type,
                    "message": str(e)
                }, ensure_ascii=False) + "\n"

    # 3. LLM 맞춤형 피드백 생성 (분석 완료 후 마지막 단계)
    if pron_result_for_feedback and mode in ("pron", "all"):
        try:
            feedback_text = generate_llm_feedback(pron_result_for_feedback, reference_data)
            
            yield json.dumps({
                "type": "feedback",
                "data": feedback_text
            }, ensure_ascii=False) + "\n"

        except Exception as e:
            yield json.dumps({
                "type": "error", 
                "task": "feedback", 
                "message": str(e)
            }, ensure_ascii=False) + "\n"    

# 로컬 실행 테스트용
if __name__ == "__main__":
    from src.models.stt_whisper import get_whisperx_models

    # 테스트 설정치
    test_file = "./experiments/wav_data/i_like_to_dance_test.wav"
    test_refs = {"I": "aɪ", "LIKE": "l aɪ k", "TO": "t u", "DANCE": "d æ n s"} # 테스트 정답
    
    if os.path.exists(test_file):
        print("\n--- [Test] 모델 로딩 중... ---")
        
        # 테스트를 위해 여기서 모델을 직접 로드합니다. (Main.py의 lifespan 역할)
        # 실제 서버에서는 이미 로드된 걸 쓰지만, 로컬 테스트에선 직접 준비해야 합니다.
        models = get_whisperx_models(model_name="small.en", vad_method="silero")
        print("\n--- [Test] 모델 로딩 완료! ---")

        print("\n--- [Test] 분석 시작 ---")
        # 분석 스트림 시작
        generator = analyze_speech_stream(
            audio_path=test_file, 
            loaded_models=models,
            reference_data=test_refs, # 정답 데이터
            mode="all"
        )
        
        # 스트리밍 결과 출력
        #for chunk in generator:
        #    print(chunk.strip())
        for chunk in generator:
            res = json.loads(chunk)
            t = res["type"]

            if t == "pron":
                #print(f"\n[발음] {len(res['data'])}개 단어 분석 완료")
                print("\n--- [1. 발음 분석 결과] ---")
                print(json.dumps(res["data"], indent=4, ensure_ascii=False))
            elif t == "inton":
                print(f"[강세] 인토네이션 곡선 추출 완료")
            elif t == "feedback":
                print(f"\n--- [3. AI 맞춤형 피드백] ---")
                print(res["data"])
            elif t == "error":
                print(f"에러: {res['message']}")         
    else:
        print(f"파일을 찾을 수 없습니다: {test_file}")