#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:12:38 2026

@author: gonie
"""

import os
import pandas as pd
from src.pdf_parser import extract_text_from_pdf
from src.llama_client import get_flashcards_from_llama

def main():
    # 1. 파일 설정 (data 폴더 안에 있는 파일명으로 수정하세요)
    target_file = "lecture_note.pdf"
    pdf_path = os.path.join("data", target_file)

    print(f"=== Flashcard Generator Started ===")
    
    # 2. PDF 읽기
    print(f"[1/3] Reading PDF: {target_file}")
    text_data = extract_text_from_pdf(pdf_path)
    
    if not text_data:
        print("종료: 텍스트를 읽어오지 못했습니다.")
        return

    # 3. Llama 서버 호출
    print(f"[2/3] Generating Flashcards via Llama (Port 8000)...")
    cards = get_flashcards_from_llama(text_data)

    if not cards:
        print("종료: 플래시카드를 생성하지 못했습니다.")
        return

    # 4. 결과 출력 및 저장
    print(f"[3/3] Done! Generated {len(cards)} cards.\n")
    
    print("-" * 40)
    for i, card in enumerate(cards):
        print(f"Q{i+1}: {card.get('Q')}")
        print(f"A{i+1}: {card.get('A')}")
        print("-" * 40)

    # (선택) CSV 저장
    df = pd.DataFrame(cards)
    df.to_csv("my_flashcards.csv", index=False, encoding="utf-8-sig")
    print("\n결과가 'my_flashcards.csv'로 저장되었습니다.")

if __name__ == "__main__":
    main()