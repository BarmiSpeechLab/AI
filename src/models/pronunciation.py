IPA_MAP = {
    # [모음]
    'AA': {'h': 'ㅏ', 'ipa': 'ɑ'},
    'AE': {'h': 'ㅐ', 'ipa': 'æ'},
    'AH': {'h': 'ㅓ', 'ipa': 'ʌ'},
    'AO': {'h': 'ㅗ', 'ipa': 'ɔ'},
    'EH': {'h': 'ㅔ', 'ipa': 'ɛ'},
    'ER': {'h': 'ㅓ', 'ipa': 'ɝ'}, 
    'IH': {'h': 'ㅣ', 'ipa': 'ɪ'},
    'IY': {'h': 'ㅣ', 'ipa': 'i'},
    'UH': {'h': 'ㅜ', 'ipa': 'ʊ'},
    'UW': {'h': 'ㅜ', 'ipa': 'u'},
    
    # [자음]
    'B':  {'h': 'ㅂ', 'ipa': 'b'},
    'CH': {'h': 'ㅊ', 'ipa': 'tʃ'},
    'D':  {'h': 'ㄷ', 'ipa': 'd'},
    'DH': {'h': 'ㄷ', 'ipa': 'ð'},
    'F':  {'h': 'ㅍ', 'ipa': 'f'},
    'G':  {'h': 'ㄱ', 'ipa': 'g'},
    'HH': {'h': 'ㅎ', 'ipa': 'h'},
    'JH': {'h': 'ㅈ', 'ipa': 'dʒ'},
    'K':  {'h': 'ㅋ', 'ipa': 'k'},
    'L':  {'h': 'ㄹ', 'ipa': 'l'},
    'M':  {'h': 'ㅁ', 'ipa': 'm'},
    'N':  {'h': 'ㄴ', 'ipa': 'n'},
    'NG': {'h': 'ㅇ', 'ipa': 'ŋ'},
    'P':  {'h': 'ㅍ', 'ipa': 'p'},
    'R':  {'h': 'ㄹ', 'ipa': 'r'},
    'S':  {'h': 'ㅅ', 'ipa': 's'},
    'SH': {'h': 'ㅅ', 'ipa': 'ʃ'},
    'T':  {'h': 'ㅌ', 'ipa': 't'},
    'TH': {'h': 'ㅅ', 'ipa': 'θ'},
    'V':  {'h': 'ㅂ', 'ipa': 'v'},
    'W':  {'h': 'w',  'ipa': 'w'}, 
    'Y':  {'h': 'y',  'ipa': 'j'}, 
    'Z':  {'h': 'ㅈ', 'ipa': 'z'},
    'ZH': {'h': 'ㅈ', 'ipa': 'ʒ'},
}

def expand_diphthongs(phonemes):
    """
    [기능] 이중모음(AY/EY/OY/AW/OW)을 단모음 조합으로 확장합니다.
    [입력]  phonemes: ['K', 'OW', 'M', ...] 같은 ARPAbet 토큰 리스트
    [출력]  확장된 토큰 리스트 (예: 'OW' -> ['AO','UW'])
    [용도]  한글 조립 시 이중모음을 더 자연스럽게 처리하기 위한 전처리 단계
    """
    new_list = []
    for p in phonemes:
        if p == 'AY': new_list.extend(['AA', 'IY'])
        elif p == 'EY': new_list.extend(['EH', 'IY'])
        elif p == 'OY': new_list.extend(['AO', 'IY'])
        elif p == 'AW': new_list.extend(['AA', 'UW'])
        elif p == 'OW': new_list.extend(['AO', 'UW'])
        else: new_list.append(p)
    return new_list

def assemble_hangul(phonemes):
    """
    [기능] ARPAbet 음소(phonemes)를 규칙 기반으로 한글 음절로 조립합니다.
    [입력]  phonemes: 확장된(또는 원본) ARPAbet 토큰 리스트
    [출력]  한글 문자열 (예: ['HH','EH','L','OW'] -> '헬로' 비슷한 결과)

    [포함 규칙]
    - L-Doubling: Hello -> '헬로' 처럼 앞 음절에 ㄹ 받침 추가
    - C + Y + V: P-Y-UW 같은 패턴을 '퓨'처럼 처리(is_y 플래그 사용)
    - W + 모음 / Y + 모음: '와/워', '야/여' 등의 특수 처리
    - 자음 단독(뒤에 모음이 없을 때): 'EU(ㅡ)'를 보조 모음으로 붙여 음절 생성
    """
    result = []
    VOWELS = {'AA','AE','AH','AO','EH','ER','IH','IY','UH','UW','EU'} # EU는 가상 모음
    
    i = 0
    while i < len(phonemes):
        curr = phonemes[i]
        
        # --- 1. 자음인 경우 ---
        if curr not in VOWELS:
            
            # ① C + Y + V 패턴 (ex. Computer P-Y-UW -> 퓨)
            if i + 2 < len(phonemes) and phonemes[i+1] == 'Y' and phonemes[i+2] in VOWELS:
                cho = curr
                jung = phonemes[i+2]
                i += 3 # 3개 소모
                
                # 종성 확인
                jong = ''
                if i < len(phonemes) and phonemes[i] not in VOWELS:
                    is_next_vowel = (i + 1 < len(phonemes) and phonemes[i+1] in VOWELS)
                    if not is_next_vowel:
                        jong = phonemes[i]
                        i += 1
                
                # make_char에 is_y=True 플래그 전달
                result.append(make_char(cho, jung, jong, is_y=True))
                continue

            # ② C + V 패턴 (일반적인 경우)
            if i + 1 < len(phonemes) and phonemes[i+1] in VOWELS:
                cho = curr
                jung = phonemes[i+1]
                i += 2
                
                # [L-Doubling] Hello -> 헬로
                # 현재 초성이 L이고, 앞 글자에 받침이 없으면 앞 글자에 ㄹ 받침 추가
                if cho == 'L' and result:
                    prev_char = result[-1]
                    # 한글인지 확인 후 받침 없는지 확인
                    if '가' <= prev_char <= '힣':
                        code = ord(prev_char) - 0xAC00
                        # 종성 인덱스가 0이면 받침 없음
                        if (code % 28) == 0:
                            # ㄹ(8) 받침 추가
                            new_code = code + 8
                            result[-1] = chr(0xAC00 + new_code)

                # 종성 확인
                jong = ''
                if i < len(phonemes) and phonemes[i] not in VOWELS:
                    is_next_vowel = (i + 1 < len(phonemes) and phonemes[i+1] in VOWELS)
                    if not is_next_vowel:
                        jong = phonemes[i]
                        i += 1
                
                result.append(make_char(cho, jung, jong))
            
            # ③ 자음 단독 (Strike의 S, T, K) -> '으' 추가
            else:
                result.append(make_char(curr, 'EU', '')) 
                i += 1
        
        # --- 2. 모음으로 시작 (ㅇ + 모음) ---
        else:
            cho = 'NG' # ㅇ
            jung = curr
            i += 1
            
            jong = ''
            if i < len(phonemes) and phonemes[i] not in VOWELS:
                is_next_vowel = (i + 1 < len(phonemes) and phonemes[i+1] in VOWELS)
                if not is_next_vowel:
                    jong = phonemes[i]
                    i += 1
            
            result.append(make_char(cho, jung, jong))
            
    return "".join(result)

def make_char(c, v, final, is_y=False):
    """
    [기능] (초성 자음, 중성 모음, 종성 자음) 토큰을 한글 '한 글자(음절)'로 합성합니다.
    [입력]
      - c: 초성 ARPAbet 자음 (예: 'K', 'L', 'CH' ...)
      - v: 중성 ARPAbet 모음 (예: 'AA', 'EH', 'UW' ... / 'EU'는 보조 모음)
      - final: 종성 ARPAbet 자음(옵션)
      - is_y: C+Y+V 패턴(퓨/큐 등)을 처리하기 위한 플래그
    [출력] 완성된 한글 음절 1글자 (합성 실패 시 중성 문자 등으로 fallback)

    [특수 처리]
    - is_y=True: 'ㅑ/ㅕ/ㅠ' 계열로 변환
    - c == 'W': '와/워/위' 같은 합성
    - c == 'Y': '야/여/유' 같은 합성
    - 종성 매핑은 일부 자음만 지원(jong_map_table)
    """
    
    # 1. 초성 매핑
    cho_map = {
        'B':'ㅂ', 'CH':'ㅊ', 'D':'ㄷ', 'DH':'ㄷ', 'F':'ㅍ', 'G':'ㄱ', 'HH':'ㅎ', 
        'JH':'ㅈ', 'K':'ㅋ', 'L':'ㄹ', 'M':'ㅁ', 'N':'ㄴ', 'NG':'ㅇ', 'P':'ㅍ', 
        'R':'ㄹ', 'S':'ㅅ', 'SH':'ㅅ', 'T':'ㅌ', 'TH':'ㅅ', 'V':'ㅂ', 'Z':'ㅈ', 'ZH':'ㅈ',
        'W':'ㅇ', 'Y':'ㅇ'
    }
    
    # 2. 중성 매핑
    jung_map = {
        'AA':'ㅏ', 'AE':'ㅐ', 'AH':'ㅓ', 'AO':'ㅗ', 'EH':'ㅔ', 'ER':'ㅓ', 
        'IH':'ㅣ', 'IY':'ㅣ', 'UH':'ㅜ', 'UW':'ㅜ', 'EU':'ㅡ'
    }
    
    cho_char = cho_map.get(c, 'ㅇ')
    jung_char = jung_map.get(v, 'ㅏ')
    
    # [특수] C + Y + V (퓨, 뷰, 큐 ...)
    if is_y:
        y_combos = {
            'AA':'ㅑ', 'AE':'ㅒ', 'AH':'ㅕ', 'AO':'ㅛ', 'EH':'ㅖ', 
            'IH':'ㅣ', 'IY':'ㅣ', 'OW':'ㅛ', 'UH':'ㅠ', 'UW':'ㅠ', 'ER':'ㅕ'
        }
        if v in y_combos:
            jung_char = y_combos[v]
        # 예외: S/SH + Y + U -> 슈
        # J/CH + Y + U -> 쥬/츄
        
    # [특수] W + 모음 (와, 워)
    if c == 'W':
        w_combos = {'AA':'와', 'AE':'왜', 'AH':'워', 'AO':'워', 'EH':'웨', 'IH':'위', 'IY':'위', 'ER':'워'}
        if v in w_combos: jung_char = w_combos[v]
            
    # [특수] Y + 모음 (야, 여) - 초성이 Y인 상황일 때 
    if c == 'Y' and not is_y:
        y_combos = {'AA':'야', 'AE':'얘', 'AH':'여', 'AO':'요', 'EH':'예', 'IH':'이', 'IY':'이', 'OW':'요', 'UH':'유', 'UW':'유'}
        if v in y_combos: jung_char = y_combos[v]

    # 3. 종성 매핑
    jong_char = ''
    jong_map_table = {
        'B':'ㅂ', 'D':'ㄷ', 'G':'ㄱ', 'K':'ㄱ', 'L':'ㄹ', 'M':'ㅁ', 'N':'ㄴ', 'NG':'ㅇ', 'P':'ㅂ', 'T':'ㅅ'
    }
    if final in jong_map_table:
        jong_char = jong_map_table[final]
    
    chosung = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    jungsung = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    jongsung = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    try:
        c_i = chosung.index(cho_char)
        j_i = jungsung.index(jung_char)
        j2_i = jongsung.index(jong_char) if jong_char else 0
        return chr(0xAC00 + (c_i * 588) + (j_i * 28) + j2_i)
    except:
        return jung_char

def phonemes_to_hangul(phonemes: list[str]) -> str:
    """
    [기능] ARPAbet 음소 리스트(phonemes)를 한글 발음 표기로 변환합니다.
    [입력]  phonemes: ['HH','EH','L','OW', ...] (숫자 제거된 토큰 권장)
    [출력]  한글 문자열 (예: '헬로')
    [처리 흐름]
      1) expand_diphthongs: 이중모음 확장
      2) assemble_hangul: 규칙 기반 한글 조립
    """    
    expanded = expand_diphthongs(phonemes)
    return assemble_hangul(expanded)

def phonemes_to_ipa(phonemes: list[str]):
    """
    [기능] ARPAbet 음소 리스트(phonemes)를 IPA 표기로 변환합니다.
    [입력]  phonemes: ['HH','EH','L','OW', ...] (숫자 제거된 토큰 권장)
    [출력]
      - ipa_str: '/hɛloʊ/' 형태의 문자열
      - ipa_list: ['h','ɛ','l','oʊ'] 처럼 토큰 리스트(디버깅/프론트 표시용)

    [주의]
    - IPA_MAP에 없는 토큰은 그대로 출력(보수적 fallback)
    - 이중모음(AY/EY/OY/AW/OW)은 IPA에서 통상 표기로 별도 처리
    """    
    ipa_list = []
    for p in phonemes:
        if p in IPA_MAP:
            ipa_list.append(IPA_MAP[p]["ipa"])
        elif p == 'AY': ipa_list.append('aɪ')
        elif p == 'EY': ipa_list.append('eɪ')
        elif p == 'OY': ipa_list.append('ɔɪ')
        elif p == 'AW': ipa_list.append('aʊ')
        elif p == 'OW': ipa_list.append('oʊ')
        else:
            ipa_list.append(p)

    ipa_str = f"/{''.join(ipa_list)}/"
    return ipa_str, ipa_list


def phonemes_to_hangul_ipa(phonemes: list[str]):
    """
    [기능] 동일한 phonemes 입력으로부터 한글 발음 표기와 IPA 표기를 동시에 생성합니다.
    [입력]  phonemes: ARPAbet 음소 리스트(숫자 제거된 토큰 권장)
    [출력]
      - hangul: 한글 발음 표기 문자열
      - ipa_str: '/.../' 형태의 IPA 문자열
      - ipa_list: IPA 토큰 리스트

    [용도]
    - speech_pipeline에서 g2p 결과(phonemes)를 한 번만 만든 뒤,
      한글/IPA 두 표현을 함께 반환하고 싶을 때 사용
    """    
    hangul = phonemes_to_hangul(phonemes)
    ipa_str, ipa_list = phonemes_to_ipa(phonemes)
    return hangul, ipa_str, ipa_list