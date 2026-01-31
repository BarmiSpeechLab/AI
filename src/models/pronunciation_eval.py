from typing import List, Dict, Any
from src.models.g2p import text_to_phonemes
from src.models.pronunciation import phonemes_to_hangul_ipa

def calculate_word_per(ref_ipa: List[str], user_ipa: List[str]) -> float:
    """IPA 리스트를 비교하여 PER 계산"""
    n, m = len(ref_ipa), len(user_ipa)
    if n == 0: return float(m)
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_ipa[i-1] == user_ipa[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return round(float(dp[n][m]) / n, 2) if n > 0 else 0.0

def process_pronunciation_eval(words: List[str], reference_data: Dict[str, str]) -> List[Dict[str, Any]]:
    results = []
    
    # reference_data의 키 리스트 (순서대로 비교하기 위함)
    ref_keys = list(reference_data.keys())

    for idx, w in enumerate(words):
        # 1. 유저 발음 분석 (G2P)
        upl_list = text_to_phonemes(w)
        ukor, _, uipa_list = phonemes_to_hangul_ipa(upl_list)

        # 2. 정답 데이터 매칭 (인덱스 기준)
        # 만약 유저가 단어를 더 많이 말하거나 적게 말할 경우를 대비해 인덱스 체크
        if idx < len(ref_keys):
            target_word = ref_keys[idx] # 실제 나와야 하는 단어 (예: DANCE)
            ref_ipa_raw = reference_data[target_word] # 실제 나와야 하는 IPA (예: d æ n s)
            cipa_list = ref_ipa_raw.split()
            # 정답 IPA를 기반으로 한글과 ARPAbet 역추산 (필요시)
            ckor, _, _ = phonemes_to_hangul_ipa(cipa_list) # 이 함수가 IPA 입력을 지원한다고 가정
        else:
            # 정답 범위를 벗어난 단어를 말한 경우
            target_word = "EXTRA"
            cipa_list = []
            ckor = ""

        # 3. PER 계산
        error_rate = calculate_word_per(cipa_list, uipa_list)
        word_is_correct = (error_rate == 0.0)

        phonemes_detail = []
        max_len = max(len(cipa_list), len(uipa_list))
        for i in range(max_len):
            u_ipa = uipa_list[i] if i < len(uipa_list) else None
            phonemes_detail.append({
                "cpl": None, # IPA 기반 비교 시 ARPAbet(cpl)은 생략 가능하거나 IPA로 대체
                "upl": upl_list[i] if i < len(upl_list) else None,
                "cipa": cipa_list[i] if i < len(cipa_list) else None,
                "uipa": u_ipa,
                "type": "vowel" if u_ipa and u_ipa in "aeiouɑæəɛɪɔʊʌ" else "consonant"
            })

        results.append({
            "target_word": target_word, # 원래 나왔어야 할 단어
            "word": w.upper().replace(".", ""), # 유저가 실제 발음한 단어
            "phonemes": phonemes_detail,
            "kor": {"ckor": ckor, "ukor": ukor},
            "error_rate": error_rate,
            "is_correct": word_is_correct
        })
    return results