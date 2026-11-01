# Copyright (c) CrowdTrainer LLC. All rights reserved.
"""Joins CpuOpBreakdownBench and cpu_op_breakdown_torch.py output into one ratio table.

usage: join_op_breakdown.py <log> [<log> ...]
For each operation and shape: Tensors' best over thread counts against PyTorch's best over thread counts and layouts
(NCHW or channels_last), the way the parity harness compares whole steps. Ratio > 1 means PyTorch is faster.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"OPBREAKDOWN (tensors|torch) \| (\S+) \| (.+?) \| t(\d+) \| ([0-9.]+)")


def main() -> None:
    best = defaultdict(lambda: {"tensors": None, "torch": None})
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = LINE.search(line)
                if not m:
                    continue
                side, op, shape, threads, ms = m.group(1), m.group(2), m.group(3), int(m.group(4)), float(m.group(5))
                op = op.removesuffix("+cl")
                key = (op, shape.strip())
                cur = best[key][side]
                if cur is None or ms < cur[0]:
                    best[key][side] = (ms, threads)
    total = {"tensors": 0.0, "torch": 0.0}
    unpaired = []
    print(f"{'op':<13} {'shape':<38} {'tensors ms':>11} {'torch ms':>10} {'ratio':>7}")
    for (op, shape), sides in sorted(best.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        t, p = sides["tensors"], sides["torch"]
        if t is None or p is None:
            # An operation or shape name that drifted between the two benches, or one side's run that failed.
            unpaired.append(f"{op} {shape} ({'tensors' if t is not None else 'torch'} only)")
            continue
        total["tensors"] += t[0]
        total["torch"] += p[0]
        flag = "  <-- slower" if t[0] > p[0] * 1.05 else ""
        print(f"{op:<13} {shape:<38} {t[0]:>8.3f}@{t[1]} {p[0]:>7.3f}@{p[1]} {t[0] / p[0]:>7.2f}{flag}")
    if total["torch"]:
        print(f"sum: tensors {total['tensors']:.2f} ms, torch {total['torch']:.2f} ms, ratio {total['tensors'] / total['torch']:.2f}")
    else:
        print("no paired rows: one side's log is empty or the two benches name their rows differently")
    for row in unpaired:
        print(f"unpaired: {row}")


if __name__ == "__main__":
    main()
