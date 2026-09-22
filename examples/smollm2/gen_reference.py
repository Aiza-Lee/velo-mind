#!/usr/bin/env python3
"""生成 SmolLM2-135M 的 HuggingFace prefill 参考 logits。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="/home/aiza/workspace/assets/ai-models/SmolLM2-135M",
        help="Path to the SmolLM2-135M model directory.",
    )
    parser.add_argument(
        "--out",
        default="examples/smollm2/testdata/reference_prompt_5tokens.safetensors",
        help="Output safetensors file path.",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, my name is",
        help="Prompt text to encode and prefill.",
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=5,
        help="Number of tokens to prefill (truncates the encoded prompt).",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"error: model dir not found: {model_dir}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading tokenizer from {model_dir} ...")
    tok = AutoTokenizer.from_pretrained(model_dir)

    print(f"loading model (FP32 reference) ...")

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    encoded = tok(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"][0]
    seq = min(args.seq, input_ids.shape[0])
    input_ids = input_ids[:seq]
    print(f"prompt: {args.prompt!r}")
    print(f"encoded (first {seq} tokens): {input_ids.tolist()}")
    print(f"decoded:  {tok.decode(input_ids)!r}")

    with torch.no_grad():
        out = model(input_ids.unsqueeze(0))
    logits = out.logits[0]
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"argmax last token: {logits[-1].argmax().item()}")

    save_file(
        {
            "input_tokens":     input_ids.to(torch.int32).numpy(),
            "reference_logits": logits.to(torch.float32).numpy(),
        },
        str(out_path),
    )
    print(f"wrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
