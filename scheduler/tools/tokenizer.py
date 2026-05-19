#!/usr/bin/env python3
import json
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    print('transformers not installed', file=sys.stderr)
    sys.exit(1)


def usage():
    print('Usage: tokenizer.py encode <model_dir> <text>')
    print('       tokenizer.py decode <model_dir> <ids>')
    sys.exit(1)


def load_tokenizer(model_dir):
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
    elif mode == 'decode':
        ids = [int(x) for x in payload.strip().split() if x.strip()]
        print(tokenizer.decode(ids))
    else:
        usage()
