#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
import json

try:
    from transformers import AutoTokenizer
except ImportError:
    print('transformers not installed', file=sys.stderr)
    sys.exit(1)


def usage():
    print('Usage: tokenizer.py encode <model_dir> <text>')
    print('       tokenizer.py encode_chat <model_dir> <json_messages>')
    print('       tokenizer.py decode <model_dir> <ids>')
    sys.exit(1)


def load_tokenizer(model_dir):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return AutoTokenizer.from_pretrained(model_dir)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        usage()

    mode = sys.argv[1]
    model_dir = sys.argv[2]
    payload = ' '.join(sys.argv[3:])

    tokenizer = load_tokenizer(model_dir)
    if mode == 'encode':
        ids = tokenizer.encode(payload)
        print(' '.join(str(x) for x in ids))
    elif mode == 'encode_chat':
        messages = json.loads(payload)
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        ids = encoded["input_ids"]
        print(' '.join(str(x) for x in ids))
    elif mode == 'decode':
        ids = [int(x) for x in payload.strip().split() if x.strip()]
        print(tokenizer.decode(ids))
    else:
        usage()
