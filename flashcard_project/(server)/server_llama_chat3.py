
# server_llama_chat2.py
# Runs on compute node (GPU allocated) ㅇ이전 버전이 오류나서 .. 고침

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# =========================
# Model config
# =========================
MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",        # critical: let accelerate handle devices
    torch_dtype="auto",
)

model.eval()

# =========================
# FastAPI
# =========================
app = FastAPI()

class GenerateRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0


@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Expects:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "max_new_tokens": 200,
        "temperature": 0.0
    }
    """

    # 1. Apply LLaMA chat template
    inputs = tokenizer.apply_chat_template(
        req.messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # IMPORTANT:
    # Do NOT manually move tensors across devices.
    # accelerate will handle it.
    # 모델이 있는 디바이스로 입력 이동
    device = next(model.parameters()).device 
    inputs = inputs.to(device) # 텐서를 GPU로 복사(gradient랑 무관)
    attention_mask = torch.ones_like(inputs, device=device)

    # 2. Generation
    # torch.no_grad() : 역전파용 gradient 기록 안함 (inference시 메모리절약)
    with torch.no_grad():
        if req.temperature > 0:
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # 3. Decode only newly generated tokens
    generated_tokens = outputs[0][inputs.shape[-1]:]
    response_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return {
        "response": response_text.strip()
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
