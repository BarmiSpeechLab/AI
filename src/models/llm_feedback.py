import os
import json
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
gms_key = os.getenv("GMS_KEY")

# GMS API 환경에 맞춘 클라이언트 설정
client = OpenAI(
    api_key=gms_key,
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1" 
)

def generate_llm_feedback(pron_data: List[Dict[str, Any]], reference_data: Dict[str, str]) -> str:
    """
    [기능] 분석 결과와 정답 IPA를 비교하여 LLM 피드백 생성
    - pron_data: 이전 단계에서 만든 [{"word": "LIKE", "phonemes": [...], "ukor": "라익"}]
    - reference_data: {"LIKE": "l aɪ k", "DANCE": "d æ n s"} 형태
    """
    
    # 1. 비교용 데이터 정제
    comparison_summary = []
    for item in pron_data:
        word = item['word']
        user_ipa = " ".join([p['uipa'] for p in item['phonemes']])
        target_ipa = reference_data.get(word, "N/A")
        
        comparison_summary.append({
            "word": word,
            "user_ipa": user_ipa,
            "target_ipa": target_ipa
        })

    # 2. 프롬프트 엔지니어링
    prompt = f"""
        Role: 청각장애인을 위한 전문 영어 발음 코치
        Task: 사용자의 발음(User IPA)을 정답(Target IPA)과 비교하여, 소리를 듣지 않고도 근육의 움직임, 시각적 단서, 촉각적 감각만으로 발음을 교정할 수 있는 1:1 가이드를 작성하라.

        Data: {json.dumps(comparison_summary, ensure_ascii=False)}

        Constraints:
        1. [철저한 비청각적 피드백]: '높은 소리', '부드러운 소리', '울림' 같은 청각적 표현은 절대 금지한다.
        2. [신체 조작 가이드]: 혀의 위치(입천장 어느 부위인지), 턱의 벌어짐 정도(손가락 몇 개 들어가는지), 입술 모양을 구체적으로 설명하라.
        3. [촉각 피드백]: 목의 진동(성대 떨림), 입 앞의 공기 흐름(바람의 세기) 등 손으로 느껴지는 감각을 반드시 포함하라.
        4. [선택적 피드백]: 발음이 정답과 일치하는 단어는 생략하고, 교정이 필요한 단어만 출력하라.
        5. [출력 형식]: 각 단어별로 아래 형식을 지켜 마크다운 리스트로 작성하라.

        Output Format:
        - 단어명 (사용자 발음 -> 정답 발음)
        - 틀린 점: (어느 근육/부위가 잘못되었는지 간략히 설명)
        - 교정 방법: (혀 위치, 입 모양, 촉각적 느낌 순으로 설명/150자 제한)

        Example:
        - THINK ([sɪŋk] -> [θɪŋk])
        - 틀린 점: 혀끝이 입안에 머물러 [s] 소리가 났습니다.
        - 교정 방법: 혀끝을 윗니와 아랫니 사이에 살짝 내미세요. 그 상태에서 공기를 밖으로 천천히 밀어내며 손등에 미지근한 바람이 느껴지게 하세요. 성대는 떨리지 않아야 합니다.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  
            messages=[
                {"role": "system", "content": "너는 언어학적 지식이 풍부한 청각장애인 전담 영어 발음 교정 코치야. 모든 답변은 한국어로 해줘."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.3 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"피드백 생성 중 오류가 발생했습니다: {str(e)}"