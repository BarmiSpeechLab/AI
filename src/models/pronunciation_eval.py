from typing import List, Dict, Any
from src.models.pronunciation import phonemes_to_hangul_ipa

ARPABET_TO_IPA = {
    "AA": "ɑ",  "AE": "æ",  "AH": "ʌ",  "AO": "ɔ",  "AW": "aʊ",
    "AX": "ə",  "AY": "aɪ", "B": "b",   "CH": "tʃ", "D": "d",
    "DH": "ð",  "EH": "ɛ",  "ER": "ɝ",  "EY": "eɪ", "F": "f",
    "G": "g",   "HH": "h",  "IH": "ɪ",  "IY": "i",  "JH": "dʒ",
    "K": "k",   "L": "l",   "M": "m",   "N": "n",   "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p",   "R": "ɹ",   "S": "s",
    "SH": "ʃ",  "T": "t",   "TH": "θ",  "UH": "ʊ",  "UW": "u",
    "V": "v",   "W": "w",   "Y": "j",   "Z": "z",   "ZH": "ʒ",
    "SIL": "",  "SP": ""
}
IPA_TO_ARPABET = {v: k for k, v in ARPABET_TO_IPA.items() if v}

ARPABET_TYPES = {
    # vowels
    "AA":"vowel","AE":"vowel","AH":"vowel","AO":"vowel","AW":"vowel","AX":"vowel","AY":"vowel",
    "EH":"vowel","ER":"vowel","EY":"vowel","IH":"vowel","IY":"vowel","OW":"vowel","OY":"vowel",
    "UH":"vowel","UW":"vowel",

    # stops
    "B":"stop","D":"stop","G":"stop","K":"stop","P":"stop","T":"stop",

    # affricates
    "CH":"affricate","JH":"affricate",

    # fricatives
    "DH":"fricative","F":"fricative","S":"fricative","SH":"fricative",
    "TH":"fricative","V":"fricative","Z":"fricative","ZH":"fricative",

    # aspirate
    "HH":"aspirate",

    # liquids
    "L":"liquid","R":"liquid",

    # nasals
    "M":"nasal","N":"nasal","NG":"nasal",

    # semivowels
    "W":"semivowel","Y":"semivowel",
}

def classify_error_level(error_rate: float) -> int:
    if error_rate <= 0.25:
        return 1
    if error_rate <= 0.50:
        return 2
    if error_rate <= 0.75:
        return 3
    return 4

def arpabet_list_to_ipa_list(arpabet_list: List[str]) -> List[str]:
    out = []
    for p in arpabet_list:
        if p in ("SIL", "SP", "|"):
            continue
        out.append(ARPABET_TO_IPA.get(p, p))
    return out

def ipa_list_to_arpabet_list(ipa_list: List[str]) -> List[str]:
    return [IPA_TO_ARPABET.get(p, p) for p in ipa_list if p]

def split_arpabet_by_reference(user_tokens: list[str], reference_data: dict[str, str]) -> list[list[str]]:
    # 1) '|'가 있으면 그걸로 split
    if "|" in user_tokens:
        chunks, cur = [], []
        for t in user_tokens:
            if t == "|":
                chunks.append(cur); cur = []
            else:
                if t not in {"SIL", "SP"}:
                    cur.append(t)
        chunks.append(cur)
        return chunks

    # 2) 없으면 정답 IPA 길이 기준으로 나눔(간단 fallback)
    ref_words = list(reference_data.keys())
    ref_lens = [len(reference_data[w]["phonemes"]) for w in ref_words]

    out, idx = [], 0
    for i, ln in enumerate(ref_lens):
        if i == len(ref_lens) - 1:
            out.append([t for t in user_tokens[idx:] if t not in {"SIL","SP"}])
        else:
            out.append([t for t in user_tokens[idx:idx+ln] if t not in {"SIL","SP"}])
        idx += ln
    return out


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

def process_pronunciation_eval(
    words: List[str],
    reference_data: Dict[str, str],
    user_arpabet_by_word: List[List[str]],
) -> List[Dict[str, Any]]:
    results = []
    ref_keys = list(reference_data.keys())

    for idx, w in enumerate(words):
        # 1) 사용자 ARPAbet (모델 출력)
        uarp_list = user_arpabet_by_word[idx] if idx < len(user_arpabet_by_word) else []
        ukor, _, _ = phonemes_to_hangul_ipa(uarp_list)
        uipa_list = arpabet_list_to_ipa_list(uarp_list)

        # 2) 정답 IPA
        if idx < len(ref_keys):
            target_word = ref_keys[idx]
            ref_entry = reference_data[target_word]
            ref_phonemes = ref_entry.get("phonemes", [])
            ckor = ref_entry.get("ckor", "")

            cipa_list = [p["cipa"] for p in ref_phonemes if p.get("cipa")]
            cpl_list = [p.get("cpl") for p in ref_phonemes]
        else:
            target_word = "EXTRA"
            cipa_list = []
            cpl_list = []
            ckor = ""

        # 3) PER (IPA 기준)
        error_rate = calculate_word_per(cipa_list, uipa_list)
        error_level = classify_error_level(error_rate)
        word_is_correct = (error_rate == 0.0)

        phonemes_detail = []
        max_len = max(len(cipa_list), len(uarp_list), len(uipa_list))
        for i in range(max_len):
            cpl = cpl_list[i] if i < len(cpl_list) else ""
            cipa = cipa_list[i] if i < len(cipa_list) else ""
            uarp = uarp_list[i] if i < len(uarp_list) else "" # 사용자 발음에 대해 모델이 예측한 ARPAbet 토큰 리스트
            uipa = uipa_list[i] if i < len(uipa_list) else ""
            is_correct = bool(cipa) and (cipa == uipa)

            phonemes_detail.append({
                "cpl": cpl,
                "upl": uarp,
                "cipa": cipa,
                "uipa": uipa,
                "type": ARPABET_TYPES.get(cpl) if cpl else "",
                "is_correct": is_correct
            })

        results.append({
            "target_word": target_word,
            "word": w.upper().replace(".", ""),
            "phonemes": phonemes_detail,
            "kor": {"ckor": ckor, "ukor": ukor},
            "error_rate": error_rate,
            "error_level": error_level,
            "is_correct": word_is_correct
        })

    return results