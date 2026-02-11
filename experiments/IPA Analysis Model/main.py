import IPA           
import g2p_to_hangul

audio_path = "../wav_data/i_like_to_dance.wav"

# 1. 오디오에서 단어 추출
words = IPA.extract_word_timings(audio_path)

print(f"{'Word':<15} | {'IPA':<15} | {'Hangul':<10} | {'Time (sec)':<15}")
print("-" * 65)

# 2. 각 단어별 처리 (G2P 적용)
for item in words:
    english_word = item['word']
    
    # hangul: 한글 발음
    # ipa_full: 합쳐진 문자열 ("/laɪk/")
    # ipa_tokens: 쪼개진 리스트 (['l', 'aɪ', 'k'])
    hangul, ipa_full, ipa_tokens = g2p_to_hangul.convert_text_to_hangul_ipa(english_word)
    timing_str = f"{item['start']:.2f} ~ {item['end']:.2f}"
    tokens_str = str(ipa_tokens)
    print(f"{english_word:<10} | {ipa_full:<15} | {tokens_str:<20} | {hangul}")
