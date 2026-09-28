"""Head-to-head PyTorch side: build a network from a neutral spec, train it, time every phase.

Usage:
    python tools/parity/run_torch_network.py --spec parity/networks/mlp.json --workdir <dir>

Writes into <workdir>:
    weights.bin   every parameterised layer's weight then bias, float32 little-endian, in layer
                  order: the exact bytes the Tensors side loads, so both start identical. A linear
                  weight is [in, out]; a conv weight is [out, in, kH, kW], PyTorch's own layout.
    data.bin      input batch [batch, *input] then target [batch, lastOut], float32
    torch.json    timings (median and spread per phase), loss curve, environment

Spec layers (a layer without "type" is linear):
    {"type": "linear", "out": N, "activation": "relu"|"none"}
    {"type": "conv2d", "out": C, "kernel": K, "stride": S, "padding": P, "activation": "relu"|"none"}
    {"type": "maxpool2d", "size": K}          stride equals size
    {"type": "flatten"}
The input is "inputDim": N (a vector per sample) or "inputShape": [C, H, W].

The weights are generated here from the spec's seed with NumPy, not with torch's initialisers,
so the Tensors side reads bytes rather than reproducing an initialisation scheme.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


def _stats(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(3 * len(ordered)) // 4]
    return {"medianMs": statistics.median(ordered), "iqrMs": q3 - q1, "minMs": ordered[0], "samples": len(ordered)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError as missing:
        print(f"DEFERRED: {missing.name} is not installed.", file=sys.stderr)
        return 3

    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"DEFERRED: torch {torch.__version__} has no CUDA device.", file=sys.stderr)
        return 3

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if spec["loss"] != "mse" or spec["optimizer"]["name"] != "sgd" or spec["dtype"] != "float32":
        print(f"unsupported spec for this runner: {spec['loss']}/{spec['optimizer']['name']}/{spec['dtype']}", file=sys.stderr)
        return 2

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec["seed"])

    input_shape = list(spec["inputShape"]) if "inputShape" in spec else [spec["inputDim"]]
    shape = list(input_shape)
    modules = []
    with open(work / "weights.bin", "wb") as f:

        def params(fan_in: int, w_shape: tuple, b_len: int):
            bound = 1.0 / np.sqrt(fan_in)
            w = rng.uniform(-bound, bound, size=w_shape).astype("<f4")
            b = rng.uniform(-bound, bound, size=(b_len,)).astype("<f4")
            f.write(w.tobytes(order="C"))
            f.write(b.tobytes(order="C"))
            return w, b

        for layer in spec["layers"]:
            kind = layer.get("type", "linear")
            if kind == "linear":
                if len(shape) != 1:
                    sys.exit(f"linear layer needs a flat input, got {shape}; add a flatten layer")
                w, b = params(shape[0], (shape[0], layer["out"]), layer["out"])
                linear = torch.nn.Linear(shape[0], layer["out"])
                with torch.no_grad():
                    linear.weight.copy_(torch.from_numpy(w.T.copy()))
                    linear.bias.copy_(torch.from_numpy(b))
                modules.append(linear)
                shape = [layer["out"]]
            elif kind == "conv2d":
                c, h, wd = shape
                k, s, pad = layer["kernel"], layer.get("stride", 1), layer.get("padding", 0)
                w, b = params(c * k * k, (layer["out"], c, k, k), layer["out"])
                conv = torch.nn.Conv2d(c, layer["out"], k, stride=s, padding=pad)
                with torch.no_grad():
                    conv.weight.copy_(torch.from_numpy(w))
                    conv.bias.copy_(torch.from_numpy(b))
                modules.append(conv)
                shape = [layer["out"], (h + 2 * pad - k) // s + 1, (wd + 2 * pad - k) // s + 1]
            elif kind == "maxpool2d":
                k = layer["size"]
                modules.append(torch.nn.MaxPool2d(k))
                shape = [shape[0], shape[1] // k, shape[2] // k]
            elif kind == "flatten":
                modules.append(torch.nn.Flatten())
                shape = [int(np.prod(shape))]
            else:
                sys.exit(f"unsupported layer type {kind!r}")
            if layer.get("activation") == "relu":
                modules.append(torch.nn.ReLU())
    if len(shape) != 1:
        sys.exit(f"the network must end flat, ends at {shape}")
    x_np = rng.standard_normal([spec["batch"]] + input_shape).astype("<f4")
    y_np = rng.standard_normal((spec["batch"], shape[0])).astype("<f4")
    with open(work / "data.bin", "wb") as f:
        f.write(x_np.tobytes(order="C"))
        f.write(y_np.tobytes(order="C"))

    device = torch.device(args.device)
    model = torch.nn.Sequential(*modules).to(device)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=spec["optimizer"]["lr"])
    loss_fn = torch.nn.MSELoss()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    losses: list[float] = []
    phases = {"forward": [], "backward": [], "optimizer": [], "step": []}
    total = spec["warmupSteps"] + spec["measuredSteps"]
    for step in range(total):
        sync()
        t0 = time.perf_counter()
        loss = loss_fn(model(x), y)
        sync()
        t1 = time.perf_counter()
        loss.backward()
        sync()
        t2 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync()
        t3 = time.perf_counter()
        if step < spec["lossAgreementSteps"]:
            losses.append(float(loss.item()))
        if step >= spec["warmupSteps"]:
            phases["forward"].append((t1 - t0) * 1e3)
            phases["backward"].append((t2 - t1) * 1e3)
            phases["optimizer"].append((t3 - t2) * 1e3)
            phases["step"].append((t3 - t0) * 1e3)

    result = {
        "framework": "torch",
        "frameworkVersion": torch.__version__,
        "device": args.device,
        "threads": torch.get_num_threads(),
        "machine": {"os": platform.system(), "arch": platform.machine(), "cpuCount": os.cpu_count(), "python": platform.python_version()},
        "network": spec["name"],
        "phases": {name: _stats(samples) for name, samples in phases.items()},
        "losses": losses,
    }
    (work / "torch.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"stepMedianMs": result["phases"]["step"]["medianMs"], "firstLoss": losses[0] if losses else None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
