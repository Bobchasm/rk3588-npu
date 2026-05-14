"""
Qwen1.5B 本地推理（WSL CPU）
用法：python chat.py
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "../models/qwen1.5b/Qwen1.5B"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("加载模型（bfloat16，约3GB）...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"加载完成，耗时 {time.time()-t0:.1f}s\n")


def chat(prompt: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    print(f"输入 {input_len} tokens，生成中...")

    t1 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t1
    new_tokens = outputs.shape[1] - input_len
    print(f"生成 {new_tokens} tokens，{elapsed:.1f}s，{new_tokens/elapsed:.2f} tok/s\n")

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


print("=== Qwen1.5B（输入 exit 退出）===\n")
while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user_input or user_input.lower() == "exit":
        break
    print(f"Qwen: {chat(user_input)}")
