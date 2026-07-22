#!/usr/bin/env python3
"""Resident PyTorch CUDA competitors for issue #846's exact RG-LRU cell."""

import argparse
import json
import os
import statistics
import subprocess
import sys

import torch


BATCH = 1
SEQUENCE = 128
DIMENSION = 256
WARMUPS = 30
SAMPLES = 101
LAUNCHES_PER_SAMPLE = 10


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def require_no_foreign_compute(label):
    result = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
        check=True, capture_output=True, text=True, timeout=5)
    conflicts = []
    for line in result.stdout.splitlines():
        cells = line.split()
        if not cells or cells[0].startswith("#") or len(cells) < 4:
            continue
        try:
            pid = int(cells[1])
            sm = int(cells[3]) if cells[3] != "-" else 0
        except ValueError:
            continue
        if pid != os.getpid() and "C" in cells[2].upper() and sm > 5:
            conflicts.append(f"pid={pid} type={cells[2]} sm={sm}%")
    if conflicts:
        raise RuntimeError(f"[{label}] foreign GPU compute: " + "; ".join(conflicts))


def rglru(value, recurrence_gate, input_gate, decay):
    base = torch.sigmoid(-decay)
    state = torch.zeros((BATCH, DIMENSION), device=value.device, dtype=value.dtype)
    output = []
    for timestep in range(SEQUENCE):
        a = recurrence_gate[:, timestep, :] * base
        scale = torch.sqrt(torch.clamp(1.0 - a * a, min=0.0))
        state = a * state + scale * (input_gate[:, timestep, :] * value[:, timestep, :])
        output.append(state)
    return torch.stack(output, dim=1)


def measure(operation):
    for _ in range(WARMUPS):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    timings = []
    result_bytes = BATCH * SEQUENCE * DIMENSION * 4
    for _ in range(SAMPLES):
        start.record()
        result = None
        for _launch in range(LAUNCHES_PER_SAMPLE):
            result = operation()
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop) * 1000.0 / LAUNCHES_PER_SAMPLE)
        del result
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    return {
        "mean_us": statistics.fmean(timings),
        "median_us": percentile(timings, 0.50),
        "p95_us": percentile(timings, 0.95),
        "p99_us": percentile(timings, 0.99),
        "peak_device_bytes": peak,
        "temporary_device_bytes": max(0, peak - result_bytes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json-lines", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA-enabled PyTorch is required.", file=sys.stderr)
        return 2
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    torch.use_deterministic_algorithms(True)

    for run in range(1, args.runs + 1):
        require_no_foreign_compute(f"rglru-pytorch-run-{run}")
        torch.manual_seed(846_000 + run)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(846_000 + run)
        value = (torch.rand((BATCH, SEQUENCE, DIMENSION), device="cuda", generator=generator) - 0.5) * 0.25
        recurrence = 0.5 + (torch.rand_like(value) - 0.5) * 0.25
        input_gate = 0.5 + (torch.rand_like(value) - 0.5) * 0.25
        decay = (torch.rand((DIMENSION,), device="cuda", generator=generator) - 0.5) * 0.5

        def eager():
            return rglru(value, recurrence, input_gate, decay)

        reference = rglru(
            value.double(), recurrence.double(), input_gate.double(), decay.double())
        candidates = [("PyTorch CUDA eager", eager)]
        compiled = torch.compile(eager, fullgraph=True, mode="max-autotune")
        compiled_probe = compiled()
        torch.cuda.synchronize()
        compiled_error = (compiled_probe - reference).abs().max().item()
        del compiled_probe
        candidates.append(("PyTorch compile max-autotune", compiled))

        for method, operation in candidates:
            candidate = operation()
            error = (candidate - reference).abs().max().item()
            del candidate
            timing = measure(operation)
            record = {
                "status": "ok",
                "run": run,
                "shape": "B1S128D256",
                "method": method,
                "max_error": max(error, compiled_error if operation is compiled else 0.0),
                "gpu": torch.cuda.get_device_name(),
                "compute_capability": ".".join(str(part) for part in torch.cuda.get_device_capability()),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "python": sys.version.split()[0],
                "warmups": WARMUPS,
                "samples": SAMPLES,
                "launches_per_device_sample": LAUNCHES_PER_SAMPLE,
                **timing,
            }
            if args.json_lines:
                print(json.dumps(record, separators=(",", ":")), flush=True)
            else:
                print(record)
        del reference, value, recurrence, input_gate, decay, compiled
        torch.cuda.empty_cache()
        require_no_foreign_compute(f"rglru-pytorch-run-{run}-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
