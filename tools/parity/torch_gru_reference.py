#!/usr/bin/env python3
"""PyTorch nn.GRU reference for GruSequenceForward parity (#1568).

Reads a JSON case (shapes, weights in PyTorch's layout, an output projection), runs one batch-first nn.GRU layer
forward and backward on loss = mean(y * projection), and writes y and the gradients of the input, h0, weight_ih,
weight_hh, bias_ih and bias_hh as JSON.

Usage: torch_gru_reference.py <case.json> <result.json> [--dtype float64|float32] [--device cpu|cuda]
"""
import json
import sys

import torch


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    case_path, result_path = sys.argv[1], sys.argv[2]
    dtype = torch.float64
    device = "cpu"
    args = sys.argv[3:]
    for i, a in enumerate(args):
        if a == "--dtype":
            dtype = {"float64": torch.float64, "float32": torch.float32}[args[i + 1]]
        if a == "--device":
            device = args[i + 1]
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but torch.cuda.is_available() is False", file=sys.stderr)
        return 3

    with open(case_path, "r", encoding="utf-8") as f:
        c = json.load(f)
    b, t, n_in, h = c["batch"], c["seq"], c["input"], c["hidden"]

    def tensor(values, shape):
        return torch.tensor(values, dtype=dtype, device=device).reshape(shape)

    gru = torch.nn.GRU(n_in, h, num_layers=1, batch_first=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        gru.weight_ih_l0.copy_(tensor(c["w_ih"], (3 * h, n_in)))
        gru.weight_hh_l0.copy_(tensor(c["w_hh"], (3 * h, h)))
        gru.bias_ih_l0.copy_(tensor(c["b_ih"], (3 * h,)))
        gru.bias_hh_l0.copy_(tensor(c["b_hh"], (3 * h,)))
    x = tensor(c["x"], (b, t, n_in)).requires_grad_(True)
    h0 = tensor(c["h0"], (1, b, h)).requires_grad_(True)
    projection = tensor(c["projection"], (b, t, h))

    y, _ = gru(x, h0)
    loss = (y * projection).mean()
    loss.backward()

    def flat(v):
        return v.detach().to("cpu", torch.float64).reshape(-1).tolist()

    result = {
        "torch": torch.__version__,
        "device": device,
        "y": flat(y),
        "x": flat(x.grad),
        "h0": flat(h0.grad),
        "w_ih": flat(gru.weight_ih_l0.grad),
        "w_hh": flat(gru.weight_hh_l0.grad),
        "b_ih": flat(gru.bias_ih_l0.grad),
        "b_hh": flat(gru.bias_hh_l0.grad),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
