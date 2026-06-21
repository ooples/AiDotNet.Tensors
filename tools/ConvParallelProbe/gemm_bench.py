#!/usr/bin/env python3
"""
#653 GEMM saturation A/B bench (CI gate for AIDOTNET_GEMM_FORWARD_PACKBOTH (#653)).

Runs the ConvParallelProbe `--gemm` single-matmul probe twice per shape - with
the forward-path PackBoth blocking optimization (#653) OFF (=0) then ON (=1) - and compares the
best-case (min_ms) timings.

Two shape classes:
  * TUNED  - shapes the hand-tuned SgemmTiled heuristic was optimized for
             (Square 1152, batched, LM-head, BERT). The lever MUST NOT regress
             these beyond --tol; a regression FAILS the job. This is the gate
             that lets us flip the default safely.
  * SMALLM - transformer FFN/QKV shapes (small M) the lever targets. Reported as
             speedup (informational); a real win needs a many-core runner, since
             on <= ~8 cores the shapes are not under-tiled and the lever no-ops.

Run each (shape, flag) --repeat times and keep the min across repeats so a single
scheduler hiccup on a shared runner doesn't dominate. OFF/ON are interleaved
per shape to control for slow drift in machine load.

Usage:
  python gemm_bench.py --probe <ConvParallelProbe.dll> [--maxdop N] [--reps N]
                       [--repeat N] [--tol 0.10] [--summary <path>]
Exit code 1 if any TUNED shape regresses past the tolerance.
"""
import argparse
import os
import re
import subprocess
import sys

# (label, M, K, N). Square 4608 (97 GFMA) is intentionally omitted - too slow for CI.
TUNED = [
    ("square-1152", 1152, 1152, 1152),
    ("square-768", 768, 768, 768),
    ("bert-768-768", 512, 768, 768),
    ("bert-768-3072", 512, 768, 3072),
    ("lm-head", 64, 768, 50257),
    ("batched-slice", 256, 512, 512),
]
SMALLM = [
    ("ffn-up", 512, 1024, 4096),
    ("ffn-down", 512, 4096, 1024),
    ("qkv-proj", 512, 1024, 1024),
]

MIN_RE = re.compile(r"min_ms=([0-9.]+)")


def fmt_ms(v):
    """Format a min_ms value, or 'n/a' when the probe failed (None)."""
    return f"{v:.3f}" if v is not None else "n/a"


def fmt_ratio(v):
    """Format a ratio/speedup, or 'n/a' for NaN (one side failed). NaN != NaN."""
    return f"{v:.3f}" if v == v else "n/a"


def run_probe_once(probe, m, k, n, maxdop, reps, flag):
    """One probe invocation for one flag value; returns min_ms or None on failure."""
    env = dict(os.environ)
    env["AIDOTNET_DISABLE_GPU"] = "1"          # no OpenCL init stealing cores
    # The #653 forward-GEMM optimizations flip together: wide-nc/mc PackBoth blocking +
    # single-persistent-region K-loop. flag=1 enables both (the eventual default); flag=0 is
    # the pre-#653 baseline.
    env["AIDOTNET_GEMM_FORWARD_PACKBOTH"] = str(flag)
    env["AIDOTNET_GEMM_SINGLE_REGION"] = str(flag)
    cmd = ["dotnet", probe, "--gemm", "--m", str(m), "--k", str(k), "--n", str(n),
           "--reps", str(reps), "--maxdop", str(maxdop)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900, check=False)
    # Surface a non-zero exit (build/runtime failure) explicitly — otherwise it degrades into an
    # ambiguous "no GEMM line" error and hides the real cause.
    if out.returncode != 0:
        sys.stderr.write(f"probe failed rc={out.returncode} for {m}x{k}x{n} flag={flag}\n"
                         f"{out.stdout[-500:]}\n{out.stderr[-500:]}\n")
        return None
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("GEMM ")), None)
    if line is None:
        sys.stderr.write(f"no GEMM line for {m}x{k}x{n} flag={flag}\n{out.stdout[-500:]}\n{out.stderr[-500:]}\n")
        return None
    match = MIN_RE.search(line)
    if not match:
        sys.stderr.write(f"no min_ms in: {line}\n")
        return None
    return float(match.group(1))


def bench(probe, shapes, maxdop, reps, repeat):
    rows = []
    for label, m, k, n in shapes:
        off = None
        on = None
        # Interleave OFF/ON per repeat (not all-OFF-then-all-ON): slow machine-load drift on a
        # shared CI runner then biases both flags equally instead of skewing the ON/OFF ratio.
        for _ in range(repeat):
            o0 = run_probe_once(probe, m, k, n, maxdop, reps, 0)
            o1 = run_probe_once(probe, m, k, n, maxdop, reps, 1)
            if o0 is not None:
                off = o0 if off is None else min(off, o0)
            if o1 is not None:
                on = o1 if on is None else min(on, o1)
        ratio = (on / off) if (off and on) else float("nan")
        rows.append((label, m, k, n, off, on, ratio))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="path to ConvParallelProbe.dll")
    ap.add_argument("--maxdop", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.10, help="max allowed tuned-shape regression (fraction)")
    ap.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = ap.parse_args()

    tuned = bench(args.probe, TUNED, args.maxdop, args.reps, args.repeat)
    smallm = bench(args.probe, SMALLM, args.maxdop, args.reps, args.repeat)

    lines = [
        f"## #653 GEMM small-M tiling A/B  (maxdop={args.maxdop}, reps={args.reps}, repeat={args.repeat}, tol={args.tol:.0%})",
        "",
        "min_ms = best case across repeats. ratio = ON / OFF (lower is better; <1 = faster with the lever).",
        "",
        "### TUNED shapes - must not regress (gate)",
        "| shape | MxKxN | OFF ms | ON ms | ratio | verdict |",
        "|---|---|---|---|---|---|",
    ]
    failed = []
    for label, m, k, n, off, on, ratio in tuned:
        ok = (off is not None and on is not None and ratio <= 1.0 + args.tol)
        if not ok:
            failed.append(label)
        verdict = "ok" if ok else "**REGRESSION**"
        lines.append(f"| {label} | {m}x{k}x{n} | {fmt_ms(off)} | {fmt_ms(on)} | {fmt_ratio(ratio)} | {verdict} |")

    lines += [
        "",
        "### SMALL-M target shapes - expected speedup (informational; needs a many-core runner)",
        "| shape | MxKxN | OFF ms | ON ms | ratio | speedup |",
        "|---|---|---|---|---|---|",
    ]
    for label, m, k, n, off, on, ratio in smallm:
        speedup = (off / on) if (off and on) else float("nan")
        speedup_str = f"{speedup:.2f}x" if speedup == speedup else "n/a"
        lines.append(f"| {label} | {m}x{k}x{n} | {fmt_ms(off)} | {fmt_ms(on)} | {fmt_ratio(ratio)} | {speedup_str} |")

    lines += ["", ("### Result: PASS - no tuned-shape regression" if not failed
                   else f"### Result: FAIL - regressions: {', '.join(failed)}")]

    report = "\n".join(lines)
    print(report)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
