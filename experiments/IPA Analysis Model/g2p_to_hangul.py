from g2p_en import G2p
import nltk

# NLTK 데이터 확인 및 다운로드
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict')

g2p = G2p()

# 1. ARPABET 매핑 테이블
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

def convert_text_to_hangul_ipa(text):
    """
    [메인] 텍스트 -> (한글, IPA)
    """
    raw_phonemes = g2p(text)
    
    # 1. 음소 정리 (숫자 제거)
    clean_phonemes = []
    for p in raw_phonemes:
        if p == ' ': continue
        p_clean = ''.join([c for c in p if not c.isdigit()])
        clean_phonemes.append(p_clean)

    # 2. 이중모음 분리 (AY -> AA + IY)
    expanded_phonemes = expand_diphthongs(clean_phonemes)
    
    # 3. 한글 조립
    hangul = assemble_hangul(expanded_phonemes)
    
    # 4. IPA 표기 생성
    ipa_list = []
    for p in clean_phonemes:
        if p in IPA_MAP: ipa_list.append(IPA_MAP[p]['ipa'])
        elif p == 'AY': ipa_list.append('aɪ')
        elif p == 'EY': ipa_list.append('eɪ')
        elif p == 'OY': ipa_list.append('ɔɪ')
        elif p == 'AW': ipa_list.append('aʊ')
        elif p == 'OW': ipa_list.append('oʊ')
        else: ipa_list.append(p)
        
    return hangul, f"/{''.join(ipa_list)}/", ipa_list

def expand_diphthongs(phonemes):
    """이중모음을 단모음으로 분리"""
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
    한글 조립 로직
    - L-Doubling (헬로)
    - C+Y+V (퓨, 큐)
    - W+ER (워)
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
    초성, 중성, 종성 -> 한글 변환
    is_y: C+Y+V 패턴일 때 True (예: 퓨, 큐)
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
    
    # 한글 조합
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

if __name__ == "__main__":
    test_words = ["I", "like", "to", "dance", "Hello", "World", "Strike", "Computer"]
    
    print(f"{'Word':<10} | {'IPA':<20} | {'Hangul'}")
    print("-" * 50)
    
    for word in test_words:
        h, i = convert_text_to_hangul_ipa(word)
        print(f"{word:<10} | {i:<20} | {h}")