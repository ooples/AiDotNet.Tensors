"""Head-to-head PyTorch side: build a network from a neutral spec, train it, time every phase.

Usage:
    python tools/parity/run_torch_network.py --spec parity/networks/mlp.json --workdir <dir>

Writes into <workdir>:
    weights.bin   every layer's weight ([in, out], row-major) then bias, float32 little-endian,
                  in layer order: the exact bytes the Tensors side loads, so both start identical
    data.bin      input batch [batch, inputDim] then target [batch, lastOut], float32
    torch.json    timings (median and spread per phase), loss curve, environment

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

    dims = [spec["inputDim"]] + [layer["out"] for layer in spec["layers"]]
    weights = []
    with open(work / "weights.bin", "wb") as f:
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            bound = 1.0 / np.sqrt(fan_in)
            w = rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype("<f4")
            b = rng.uniform(-bound, bound, size=(fan_out,)).astype("<f4")
            f.write(w.tobytes(order="C"))
            f.write(b.tobytes(order="C"))
            weights.append((w, b))
    x_np = rng.standard_normal((spec["batch"], spec["inputDim"])).astype("<f4")
    y_np = rng.standard_normal((spec["batch"], dims[-1])).astype("<f4")
    with open(work / "data.bin", "wb") as f:
        f.write(x_np.tobytes(order="C"))
        f.write(y_np.tobytes(order="C"))

    device = torch.device(args.device)
    layers = []
    for (w, b), layer in zip(weights, spec["layers"]):
        linear = torch.nn.Linear(w.shape[0], w.shape[1])
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(w.T.copy()))
            linear.bias.copy_(torch.from_numpy(b))
        layers.append(linear)
        if layer["activation"] == "relu":
            layers.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*layers).to(device)
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
