# IPA_Tokenize.py

def tokenize_ipa(text: str):
    """
    IPA 문자열을 토큰(음소) 리스트로 분리
    - 미국식(eng_to_ipa) 및 일반적인 IPA 표기를 모두 커버
    - 긴 발음 기호부터 우선 매칭하도록 정렬 로직 포함
    """
    tokens = []
    i = 0

    # 다중 문자 IPA 기호들 (길이 순서대로 정렬하여 매칭 오류 방지)
    raw_multi_chars = [
        'tʃ', 'dʒ', 'dʃ', 'ʧ', 'ʤ',       
        'iː', 'uː', 'ɔː', 'ɜː', 'ɑː',     
        'ɚ', 'ɝ', 'ər',                   
        'əʊ', 'oʊ',                       
        'aɪ', 'aʊ', 'ɔɪ', 'eɪ',           
        'ju', 'jʊ'                        
    ]
    # 긴 것부터 매칭되도록 내림차순 정렬
    multi_chars = sorted(raw_multi_chars, key=len, reverse=True)

    while i < len(text):
        matched = False
        for mc in multi_chars:
            if text[i:].startswith(mc):
                if mc == 'ʧ': tokens.append('tʃ')
                elif mc == 'ʤ': tokens.append('dʒ')
                else: tokens.append(mc)
                
                i += len(mc)
                matched = True
                break

        if not matched:
            tokens.append(text[i])
            i += 1

    return tokens


def ipa_to_hangul(ipa: str) -> str:
    """
    [메인 함수] IPA 발음 기호를 한글로 변환
    예: /tɛst/ -> 테스트
    """
    if not ipa:
        return ""

    # 전처리: 슬래시, 강세 기호, 점 제거
    text = ipa.strip('/').replace('ˈ', '').replace('ˌ', '').replace('.', '')

    tokens = tokenize_ipa(text)
    result = []
    i = 0

    while i < len(tokens):
        # 음절 단위 변환
        hangul, consumed, carry = convert_syllable(tokens, i)
        
        if hangul:
            result.append(hangul)
            i += consumed
            
            # 자음군(C+l/r+V) 처리 시, l/r을 다음 음절 초성으로 재사용
            if carry:
                tokens.insert(i, carry) 
        else:
            i += 1

    return ''.join(result)


def convert_syllable(tokens, start_idx):
    """
    토큰 리스트에서 한 음절(Onset-Nucleus-Coda)을 찾아 한글로 변환
    """
    if start_idx >= len(tokens):
        return None, 0, None

    # 모음/자음 집합 정의
    vowels = {'i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'ɒ', 'o', 'u', 'ʊ', 'ʌ', 'ə', 'ɜ',
              'iː', 'uː', 'ɔː', 'ɜː', 'ɑː', 'ɚ', 'ɝ', 'ər', 'əʊ', 'oʊ', 'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'ju', 'jʊ'}

    consonants = {'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 'ʃ', 'ʒ',
                  'θ', 'ð', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j', 'tʃ', 'dʒ', 'dʃ'}

    # (A) 특수 처리: 자음군 C + (l/r) + V (예: please -> 플 + 리즈)
    if start_idx + 2 < len(tokens):
        c1, c2, v = tokens[start_idx], tokens[start_idx + 1], tokens[start_idx + 2]
        if (c1 in consonants) and (c2 in {'l', 'r'}) and (v in vowels):
            # 첫 음절: C1 + ㅡ + ㄹ (플, 클, 블 등)
            hangul = make_syllable(_cho_from_consonant(c1), '으', 'ㄹ')
            # carry: c2(l/r)를 반환하여 다음 음절 초성으로 사용
            return hangul, 2, c2

    # (B) 특수 처리: tʃ + r + V -> '추' (예: tree -> 추리)
    if tokens[start_idx] == 'tʃ' and start_idx + 2 < len(tokens):
        if tokens[start_idx + 1] == 'r' and tokens[start_idx + 2] in vowels:
            return '추', 1, None

    idx = start_idx
    onset = []
    nucleus = None
    coda = []

    # 1. 초성(Onset) 수집
    while idx < len(tokens) and tokens[idx] in consonants:
        onset.append(tokens[idx])
        idx += 1

    # 2. 중성(Nucleus) 수집
    if idx < len(tokens) and tokens[idx] in vowels:
        nucleus = tokens[idx]
        idx += 1
    else:
        # 모음이 없으면 자음만 있는 경우 (트, 스, 츠 등)
        if onset:
            hangul = convert_final_consonant(onset[-1])
            return hangul, len(onset), None
        return None, 0, None

    # 3. 종성(Coda) 후보 수집
    if idx < len(tokens) and tokens[idx] in consonants:
        # 다음 토큰이 모음이 아니어야 종성으로 씀 (모음이면 다음 글자 초성으로 넘어감)
        if idx + 1 >= len(tokens) or tokens[idx + 1] not in vowels:
            # 예외: tʃ+r 구조는 종성으로 잡지 않음
            if not (tokens[idx] == 'tʃ' and idx + 2 < len(tokens) and tokens[idx + 1] == 'r'):
                coda.append(tokens[idx])
                idx += 1

    # 3-1. 비모음 위치 r 소거 (car -> 카(ㅇ))
    if coda and coda[0] == 'r':
        coda = []

    # 3-2. 단어 끝 파열음 분절 (stamp -> 스탬 '프')
    final_split = {'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'tʃ', 'dʒ', 'dʃ'}
    if idx >= len(tokens) and coda and coda[0] in final_split:
        base = build_hangul(onset, nucleus, [])
        tail = convert_final_consonant(coda[0])
        return base + tail, idx - start_idx, None

    hangul = build_hangul(onset, nucleus, coda)
    return hangul, idx - start_idx, None


# 한글 조합을 위한 헬퍼 함수들 ------------------------------------------------
def _cho_from_consonant(c: str) -> str:
    """자음 IPA를 한글 초성으로 매핑"""
    return {
        'p': 'ㅍ', 'b': 'ㅂ', 't': 'ㅌ', 'd': 'ㄷ',
        'k': 'ㅋ', 'g': 'ㄱ', 'f': 'ㅍ', 'v': 'ㅂ',
        's': 'ㅅ', 'z': 'ㅈ', 'ʃ': 'ㅅ', 'ʒ': 'ㅈ',
        'θ': 'ㅅ', 'ð': 'ㄷ', 'h': 'ㅎ',
        'm': 'ㅁ', 'n': 'ㄴ', 'ŋ': 'ㅇ',
        'l': 'ㄹ', 'r': 'ㄹ', 'w': 'ㅇ', 'j': 'ㅇ',
        'tʃ': 'ㅊ', 'dʒ': 'ㅈ', 'dʃ': 'ㅈ'
    }.get(c, 'ㅇ')

def convert_final_consonant(consonant):
    """단독 자음 처리 (예: s -> 스)"""
    map_final = {
        'p': '프', 'b': '브', 't': '트', 'd': '드',
        'k': '크', 'g': '그', 'f': '프', 'v': '브',
        's': '스', 'z': '즈', 'ʃ': '시', 'ʒ': '지',
        'θ': '스', 'ð': '드', 'h': '흐',
        'm': '음', 'n': '은', 'ŋ': '응',
        'l': '을', 'r': '르',
        'tʃ': '치', 'dʒ': '지', 'dʃ': '지'
    }
    return map_final.get(consonant, '으')

def build_hangul(onset, nucleus, coda):
    """초성/중성/종성 리스트를 받아 실제 한글 글자로 조합"""
    cho_map = {
        'p': 'ㅍ', 'b': 'ㅂ', 't': 'ㅌ', 'd': 'ㄷ',
        'k': 'ㅋ', 'g': 'ㄱ', 'f': 'ㅍ', 'v': 'ㅂ',
        's': 'ㅅ', 'z': 'ㅈ', 'ʃ': 'ㅅ', 'ʒ': 'ㅈ',
        'θ': 'ㅅ', 'ð': 'ㄷ', 'h': 'ㅎ',
        'm': 'ㅁ', 'n': 'ㄴ', 'ŋ': 'ㅇ',
        'l': 'ㄹ', 'r': 'ㄹ', 'w': 'ㅇ', 'j': 'ㅇ',
        'tʃ': 'ㅊ', 'dʒ': 'ㅈ', 'dʃ': 'ㅈ'
    }

    jung_map = {
        'iː': '이', 'i': '이', 'ɪ': '이',
        'uː': '우', 'u': '우', 'ʊ': '우',
        'ɔː': '오', 'ɒ': '오', 'ɔ': '오',
        'ɜː': '어', 'ɚ': '어', 'ɝ': '어', 'ər': '어', 'ə': '어', 'ʌ': '어',
        'ɑː': '아', 'ɑ': '아',
        'e': '에', 'ɛ': '에', 'æ': '애',
        'əʊ': '오', 'oʊ': '오', 'aɪ': '아이', 'aʊ': '아우',
        'ɔɪ': '오이', 'eɪ': '에이',
        'ju': '유', 'jʊ': '유'
    }

    jong_map = {
        'p': 'ㅂ', 'b': 'ㅂ', 't': 'ㅅ', 'd': 'ㄷ',
        'k': 'ㄱ', 'g': 'ㄱ', 's': 'ㅅ', 'z': 'ㅈ',
        'ʃ': 'ㅅ', 'ʒ': 'ㅈ', 'θ': 'ㅅ', 'ð': 'ㄷ',
        'm': 'ㅁ', 'n': 'ㄴ', 'ŋ': 'ㅇ',
        'l': 'ㄹ', 'r': 'ㄹ', 'f': 'ㅂ', 'v': 'ㅂ',
        'tʃ': 'ㅊ', 'dʒ': 'ㅈ', 'dʃ': 'ㅈ'
    }

    # 특수 케이스: w + 모음 (워, 와, 위 등)
    if onset and onset[0] == 'w':
        w_combos = {
            'ɔː': '워', 'ɜː': '워', 'ə': '워', 'ər': '워', 'ɚ': '워', 'ɝ': '워',
            'ɑː': '와', 'iː': '위', 'i': '위', 'ɪ': '위',
            'uː': '우', 'u': '우', 'ʊ': '우',
        }
        if nucleus in w_combos:
            jung = w_combos[nucleus]
            jong = jong_map.get(coda[0], '') if coda else ''
            return make_syllable('ㅇ', jung, jong)

    # 특수 케이스: 자음 + j + u (퓨, 뷰, 큐 등)
    if len(onset) >= 2 and onset[-1] == 'j':
        prev_cons = onset[0] if onset else None
        if nucleus in ['uː', 'u', 'ju', 'jʊ'] and prev_cons:
            yu_combos = {
                'p': '퓨', 'b': '뷰', 'm': '뮤',
                't': '튜', 'd': '듀', 'n': '뉴',
                'k': '큐', 'l': '류', 's': '슈', 'g': '규', 'z': '쥬', 'h': '휴'
            }
            if prev_cons in yu_combos:
                jong = jong_map.get(coda[0], '') if coda else ''
                if jong:
                    return yu_combos[prev_cons] + make_syllable('ㅇ', '으', jong)
                return yu_combos[prev_cons]

    # 일반 케이스
    cho = cho_map.get(onset[0], 'ㅇ') if onset else 'ㅇ'
    jung = jung_map.get(nucleus, '어')
    jong = jong_map.get(coda[0], '') if coda else ''

    return make_syllable(cho, jung, jong)

def make_syllable(cho, jung, jong=''):
    """초성/중성/종성을 합쳐 한글 유니코드 생성"""
    chosung = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
               'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    jungsung = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    jongsung = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    jung_to_jamo = {
        '아': 'ㅏ', '애': 'ㅐ', '야': 'ㅑ', '얘': 'ㅒ',
        '어': 'ㅓ', '에': 'ㅔ', '여': 'ㅕ', '예': 'ㅖ',
        '오': 'ㅗ', '와': 'ㅘ', '왜': 'ㅙ', '외': 'ㅚ', '요': 'ㅛ',
        '우': 'ㅜ', '워': 'ㅝ', '웨': 'ㅞ', '위': 'ㅟ', '유': 'ㅠ',
        '으': 'ㅡ', '의': 'ㅢ', '이': 'ㅣ'
    }

    jung_jamo = jung_to_jamo.get(jung, jung)

    if cho not in chosung or jung_jamo not in jungsung:
        return jung

    cho_idx = chosung.index(cho)
    jung_idx = jungsung.index(jung_jamo)
    jong_idx = jongsung.index(jong) if jong in jongsung else 0

    code = 0xAC00 + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx
    return chr(code)