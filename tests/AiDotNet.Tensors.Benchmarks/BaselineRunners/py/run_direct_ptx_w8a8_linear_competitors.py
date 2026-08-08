#!/usr/bin/env python3
"""Eligibility probe for PyTorch's public/resident CUDA INT8 matmul at M=1."""

import json
import sys

import torch


def main():
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2
    x = torch.randint(-16, 17, (1, 1024), device="cuda", dtype=torch.int8)
    weights = torch.randint(-16, 17, (4096, 1024), device="cuda", dtype=torch.int8)
    try:
        torch._int_mm(x, weights.t().contiguous())
    except RuntimeError as error:
        print(json.dumps({
            "status": "ineligible",
            "method": "CUDA _int_mm",
            "reason": str(error).replace("\n", " "),
            "torch": torch.__version__,
        }, separators=(",", ":")))
        return 0
    print(json.dumps({
        "status": "eligible",
        "method": "CUDA _int_mm",
        "reason": "accepted exact M=1 shape",
        "torch": torch.__version__,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
