import torch
import librosa
import numpy as np
from itertools import groupby
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# -------------------------------------------------------------------------
# 1. ARPAbet -> IPA 매핑 테이블 (이미지 기반 작성)
# -------------------------------------------------------------------------
arpabet_to_ipa = {
    "AA": "ɑ",  "AE": "æ",  "AH": "ʌ",  "AO": "ɔ",  "AW": "aʊ",
    "AX": "ə",  "AY": "aɪ", "B": "b",   "CH": "tʃ", "D": "d",
    "DH": "ð",  "EH": "ɛ",  "ER": "ɝ",  "EY": "eɪ", "F": "f",
    "G": "g",   "HH": "h",  "IH": "ɪ",  "IY": "i",  "JH": "dʒ",
    "K": "k",   "L": "l",   "M": "m",   "N": "n",   "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p",   "R": "ɹ",   "S": "s",
    "SH": "ʃ",  "T": "t",   "TH": "θ",  "UH": "ʊ",  "UW": "u",
    "V": "v",   "W": "w",   "Y": "j",   "Z": "z",   "ZH": "ʒ",
    "SIL": "",  "SP": ""   # 침묵은 표기하지 않음
}

# -------------------------------------------------------------------------
# 2. 모델 로드
# -------------------------------------------------------------------------
model_path = "./src/fine_tuned_model/"
print(f"모델 로드 중... ({model_path})")

try:
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
except OSError:
    print("모델 경로를 확인해주세요. 현재 메모리의 모델을 사용합니다.")
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# -------------------------------------------------------------------------
# 3. 추론 및 변환 함수
# -------------------------------------------------------------------------
def predict_my_voice_ipa(wav_file_path):
    print(f"\n 분석 및 변환 중: {wav_file_path}")
    
    # 1. 오디오 로드 (16000Hz)
    try:
        speech, rate = librosa.load(wav_file_path, sr=16000)
    except FileNotFoundError:
        print(" 파일을 찾을 수 없습니다.")
        return

    # 2. 모델 입력
    input_values = processor(
        speech, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(device)

    # 3. 예측
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)

    # 4. 토큰 변환 (ARPAbet)
    tokens = processor.tokenizer.convert_ids_to_tokens(pred_ids[0].cpu().numpy())
    
    # 5. 중복 제거 (Collapse)
    merged_tokens = [k for k, g in groupby(tokens)]
    
    # 6. 정제 (Clean ARPAbet)
    # 특수 토큰 제거
    clean_arpabet_list = [
        t for t in merged_tokens 
        if t not in ["[PAD]", "[UNK]", "|", "<s>", "</s>", "SIL", "SP"]
    ]
    
    # 7. [핵심] ARPAbet -> IPA 변환
    ipa_list = []
    for token in clean_arpabet_list:
        # 매핑 테이블에 있으면 변환, 없으면 그대로 출력(혹시 모를 에러 방지)
        ipa_char = arpabet_to_ipa.get(token, token)
        ipa_list.append(ipa_char)

    # 문자열로 합치기
    arpabet_output = " ".join(clean_arpabet_list)
    ipa_output = " ".join(ipa_list) # IPA는 보통 붙여서 쓰거나 좁은 간격

    # 결과 출력
    print("-" * 50)
    print(f"모델 예측 (ARPAbet): {arpabet_output}")
    print(f"변환 결과 (IPA)    : /{ipa_output}/")
    print("-" * 50)

# -------------------------------------------------------------------------
# 실행
# -------------------------------------------------------------------------
# 파일명을 넣어주세요
predict_my_voice_ipa("./experiments/wav_data/apple.wav")