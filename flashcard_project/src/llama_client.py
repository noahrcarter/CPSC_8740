#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:14:16 2026

@author: gonie
"""

import requests
import json
import re
from typing import List, Dict

# ===============================
# 설정 (prompt2_task2.py 기준)
# ===============================
API_URL = "http://localhost:8000/generate"  # 작동했던 포트 8000 사용

def get_flashcards_from_llama(text_content: str) -> List[Dict[str, str]]:
    """
    Llama 서버에 텍스트를 보내고 플래시카드(Q&A) JSON을 받아옵니다.
    """
    
    # 1. 프롬프트 구성 (시스템 메시지 + 사용자 데이터)
    system_msg = {
        "role": "system",
        "content": (
            "ROLE: You are an educational assistant designed to help students study.\n"
            "TASK: Create 5 high-quality flashcards based on the provided TEXT.\n"
            "FORMAT: Output ONLY a JSON list. Each item must have keys 'Q' (Question) and 'A' (Answer).\n"
            "CONSTRAINT: Do not output any conversational text. Start directly with '[' and end with ']'."
            "EXAMPLE: [{'Q': 'What is SQL?', 'A': 'SQL is a standard language for storing...'}]"
        )
    }

    user_msg = {
        "role": "user",
        "content": f"TEXT:\n\"\"\"\n{text_content}\n\"\"\"\n\nGenerate 5 flashcards in JSON format."
    }

    messages = [system_msg, user_msg]

    # 2. API 호출 (prompt2_task2.py의 call_llama_api 로직 그대로 사용)
    try:
        print("Sending request to Llama server...")
        response = requests.post(
            API_URL,
            json={
                "messages": messages,
                "max_new_tokens": 1024,  # 플래시카드 5개라 길이를 좀 넉넉하게 잡음
                "temperature": 0.1,      # 포맷 유지를 위해 낮음
                "top_p": 1.0
            },
            timeout=None
        )
        response.raise_for_status()
        
        # 3. 응답 파싱
        raw_response = response.json()["response"]
        
        # JSON 문자열 정리 (가끔 모델이 ```json ... ``` 을 붙일 때 제거용)
        cleaned_json = _clean_json_string(raw_response)
        
        # 리스트로 변환
        flashcards = json.loads(cleaned_json)
        return flashcards

    except requests.exceptions.ConnectionError:
        print("\n[Connection Error] 서버에 연결할 수 없습니다.")
        print("SSH 터널링이 켜져 있는지, 포트가 8000인지 확인해주세요.")
        return []
    except json.JSONDecodeError:
        print("\n[JSON Error] 모델이 JSON 형식이 아닌 텍스트를 반환했습니다.")
        print(f"Raw Output: {raw_response}")
        return []
    except Exception as e:
        print(f"\n[Error] 알 수 없는 오류: {e}")
        return []

def _clean_json_string(text):
    """
    모델 응답에서 순수 JSON 부분만 추출
    """
    text = text.strip()
    # ```json 등의 마크다운 제거
    if "```" in text:
        text = re.sub(r"```(json)?", "", text)
        text = text.replace("```", "")
    
    # 대괄호 [ ] 로 시작하고 끝나는지 확인해서 자름 (안전장치)
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        text = text[start : end+1]
        
    return text.strip()