#!/usr/bin/env python3
"""生成测试用的 token id 序列，传到板子上直接用"""

import sys
sys.path.insert(0, '/home/deep/rk3588-npu')

from transformers import AutoTokenizer

model_path = '/home/deep/rk3588-npu/models/qwen1.5b/Qwen1.5B'
t = AutoTokenizer.from_pretrained(model_path)

questions = [
    '1+1等于几',
    '你好',
    '你是谁',
    'Python中如何反转一个列表',
    '天空为什么是蓝色的',
]

for q in questions:
    m = [{'role': 'user', 'content': q}]
    s = t.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ids = t.encode(s)
    print(f'=== {q} ===')
    print(' '.join(map(str, ids)))
    print()
