from g2p_en import G2p
import nltk

"""
[NOTE - Docker/Server]
g2p_en는 NLTK 리소스를 사용합니다. 컨테이너/서버 환경에서는 런타임에 nltk.download()를 실행하지 말고,
Docker build 단계에서 아래 리소스를 미리 다운로드해 주세요.

필요 NLTK 데이터:
- taggers/averaged_perceptron_tagger_eng
- corpora/cmudict

예시 (Dockerfile):
RUN python - <<'PY'
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('cmudict')
PY
"""

# 로컬 개발 전용 
# def _ensure_nltk():
#     try:
#         nltk.data.find('taggers/averaged_perceptron_tagger_eng')
#     except LookupError:
#         nltk.download('averaged_perceptron_tagger_eng')
#     try:
#         nltk.data.find('corpora/cmudict')
#     except LookupError:
#         nltk.download('cmudict')

# _ensure_nltk()

_g2p = G2p()

def _check_nltk_installed():
    # 도커 빌드 단계에서 미리 받아두는 전제하에 실행
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('corpora/cmudict')

_check_nltk_installed()

def text_to_phonemes(text: str) -> list[str]:
    raw = _g2p(text)
    clean = []
    for p in raw:
        if p == " ":
            continue
        p_clean = "".join([c for c in p if not c.isdigit()])
        clean.append(p_clean)
    return clean