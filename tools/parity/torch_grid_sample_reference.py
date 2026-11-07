#!/usr/bin/env python3
"""PyTorch grid_sample reference for the mode-aware GridSample parity tests.

Reads a JSON case (input, grid and upstream-gradient values with their shapes), runs
torch.nn.functional.grid_sample forward and backward on loss = sum(output * grad_output) for every
mode, padding_mode and align_corners combination, and writes each combination's output, input gradient
and grid gradient as JSON, keyed "<mode>/<padding>/<align_corners 0|1>".

Usage: torch_grid_sample_reference.py <case.json> <result.json>
"""
import json
import sys

import torch
import torch.nn.functional as F


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    case_path, result_path = sys.argv[1], sys.argv[2]
    with open(case_path, "r", encoding="utf-8") as f:
        c = json.load(f)

    def tensor(values, shape):
        return torch.tensor(values, dtype=torch.float64).reshape(shape)

    inp = tensor(c["input"], c["input_shape"])
    grid = tensor(c["grid"], c["grid_shape"])
    grad_out = tensor(c["grad_output"], c["grad_output_shape"])
    result = {}
    for mode in ("bilinear", "nearest", "bicubic"):
        for padding in ("zeros", "border", "reflection"):
            for align in (False, True):
                x = inp.clone().requires_grad_(True)
                g = grid.clone().requires_grad_(True)
                y = F.grid_sample(x, g, mode=mode, padding_mode=padding, align_corners=align)
                (y * grad_out).sum().backward()
                result[f"{mode}/{padding}/{int(align)}"] = {
                    "output": y.detach().flatten().tolist(),
                    "grad_input": x.grad.flatten().tolist(),
                    "grad_grid": g.grad.flatten().tolist(),
                }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
