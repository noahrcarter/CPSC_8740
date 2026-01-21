import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str, max_chars: int = 10000) -> str:
    """
    PDF 파일에서 텍스트를 추출하는 함수
    """
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(pdf_path):
            return f"Error: 파일을 찾을 수 없습니다 -> {pdf_path}"

        reader = PdfReader(pdf_path)
        full_text = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        
        combined_text = "\n".join(full_text)
        
        # 텍스트가 너무 길면 자르기 (토큰 제한 방지)
        if len(combined_text) > max_chars:
            print(f"[알림] 텍스트가 너무 길어서 {max_chars}자로 줄입니다.")
            return combined_text[:max_chars]
            
        return combined_text

    except Exception as e:
        return f"Error reading PDF: {e}"

# ==========================================
# 아래 부분이 추가된 테스트 코드입니다 (단독 실행 시 작동)
# ==========================================
if __name__ == "__main__":
    print("--- PDF 파서 테스트 모드 ---")
    
    # 테스트할 파일 경로를 입력하세요 (예: ../data/lecture_note.pdf)
    # 현재 src 폴더 안에 있다면, 상위 폴더의 data를 가리켜야 합니다.
    test_pdf_path = "../data/lecture_note.pdf" 
    
    print(f"테스트 파일 경로: {test_pdf_path}")
    result = extract_text_from_pdf(test_pdf_path)
    
    print("\n--- 추출 결과 (앞부분 300자) ---")
    print(result[:300])
    print("------------------------------")