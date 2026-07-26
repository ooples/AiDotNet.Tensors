"""
Repeatable head-to-head: our generated kernels against PyTorch/cuDNN.

Runs both lanes, joins them by kernel name, and prints ratios beside a roofline
analysis. The roofline is the point: a ratio on its own says who is faster, while
achieved bandwidth and FLOP rate say WHY, and therefore what to do about it.

Usage (from the repo root):
    python tools/bakeoff/run_bakeoff.py

Environment:
    BAKEOFF_PYTHON   interpreter with torch+cu13 (default: the aidotnet cache venv)
    BAKEOFF_DOTNET   path to the benchmarks dll
"""

import os
import re
import subprocess
import sys

CARD_PEAK_BANDWIDTH = 760e9      # RTX 3080 spec, bytes/s
CARD_PEAK_FP32 = 29.8e12         # non-tensor-core FP32, FLOP/s

HOME = os.path.expanduser("~")
DEFAULT_PY = os.path.join(HOME, ".cache", "aidotnet-direct-ptx-py312", "Scripts", "python.exe")
TORCH_LIB = os.path.join(HOME, ".cache", "aidotnet-direct-ptx-py312",
                         "Lib", "site-packages", "torch", "lib")
DEFAULT_DLL = os.path.join("tests", "AiDotNet.Tensors.Benchmarks", "bin", "Release",
                           "net10.0", "AiDotNet.Tensors.Benchmarks.dll")

# name -> (bytes moved, flops). Used only for the roofline columns.
WORK = {
    "depthwise_conv2d_3x3":            (2 * 32 * 64 * 56 * 56 * 4, 32 * 64 * 56 * 56 * 9 * 2),
    "depthwise_conv2d_3x3_bias_relu":  (2 * 32 * 64 * 56 * 56 * 4, 32 * 64 * 56 * 56 * 9 * 2),
    "depthwise_conv2d_3x3_bwd_data":   (2 * 32 * 64 * 56 * 56 * 4, 32 * 64 * 56 * 56 * 9 * 2),
    "conv2d_1x1_bias_relu":            (2 * 16 * 64 * 28 * 28 * 4, 16 * 64 * 28 * 28 * 64 * 2),
    "conv2d_1x1_deep_epilogue":        (2 * 16 * 64 * 28 * 28 * 4, 16 * 64 * 28 * 28 * 64 * 2),
    "conv2d_1x1_bwd_data":             (2 * 16 * 64 * 28 * 28 * 4, 16 * 64 * 28 * 28 * 64 * 2),
    "conv2d_3x3_bias_relu":            ((8 * 32 * 28 * 28 + 8 * 64 * 28 * 28) * 4,
                                        8 * 64 * 28 * 28 * 32 * 9 * 2),
    "conv2d_3x3_bwd_data":             ((8 * 64 * 28 * 28 + 8 * 32 * 28 * 28) * 4,
                                        8 * 64 * 28 * 28 * 32 * 9 * 2),
    "maxpool2d_2x2":                   ((32 * 64 * 112 * 112 + 32 * 64 * 56 * 56) * 4, 0),
    "conv_transpose2d_3x3_stride2":    ((16 * 64 * 28 * 28 + 16 * 64 * 56 * 56) * 4,
                                        16 * 64 * 56 * 56 * 9 * 2),

    # Weight gradients. Bytes are the two large operands read (activations and the
    # incoming gradient); the OUTPUT is negligible -- 576 floats for the depthwise case --
    # which is precisely the shape that left the device idle and made split-K necessary.
    "depthwise_conv2d_3x3_bwd_weights": (2 * 32 * 64 * 56 * 56 * 4,
                                         32 * 64 * 56 * 56 * 9 * 2),
    "conv2d_1x1_bwd_weights":           ((16 * 64 * 28 * 28 + 16 * 64 * 28 * 28) * 4,
                                         16 * 64 * 28 * 28 * 64 * 2),
    "conv2d_3x3_bwd_weights":           ((8 * 32 * 28 * 28 + 8 * 64 * 28 * 28) * 4,
                                         8 * 64 * 28 * 28 * 32 * 9 * 2),
}


def run_ours(dll):
    """Parses --kernel-bench output into {kernel: (us, spread_pct)}."""
    out = subprocess.run(["dotnet", dll, "--kernel-bench"],
                         capture_output=True, text=True).stdout
    got = {}
    for line in out.splitlines():
        m = re.match(r"^(\S+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m:
            got[m.group(1)] = (float(m.group(3)), float(m.group(5)))
    return got


def run_torch(python, script):
    """Parses the competitor lane into {kernel: {'eager': us, 'graph': (us, spread)}}."""
    env = dict(os.environ)
    env["PATH"] = TORCH_LIB + os.pathsep + env.get("PATH", "")
    out = subprocess.run([python, script], capture_output=True, text=True, env=env).stdout

    got = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 5 or parts[0] not in ("RESULT", "GRAPH"):
            if parts and parts[0] in ("GRAPHWRONG", "GRAPHFAIL"):
                print("  competitor lane problem: " + line)
            continue
        entry = got.setdefault(parts[1], {})
        if parts[0] == "RESULT":
            entry["eager"] = float(parts[2])
        else:
            entry["graph"] = (float(parts[2]), float(parts[4]))
    return got


def main():
    python = os.environ.get("BAKEOFF_PYTHON", DEFAULT_PY)
    dll = os.environ.get("BAKEOFF_DOTNET", DEFAULT_DLL)
    script = os.path.join("tests", "AiDotNet.Tensors.Benchmarks", "BaselineRunners",
                          "py", "run_codegen_bakeoff.py")

    print("measuring our generated kernels ...")
    ours = run_ours(dll)
    print("measuring PyTorch/cuDNN (eager and CUDA-graph) ...")
    theirs = run_torch(python, script)
    print()

    header = ("kernel", "ours us", "spread", "cuDNN us", "spread", "ratio",
              "our GB/s", "%bw", "our TF/s", "bound")
    print("%-32s%9s%8s%10s%8s%8s%10s%6s%10s  %s" % header)

    wins = losses = 0
    for name in sorted(ours):
        if name not in theirs or "graph" not in theirs[name]:
            print("%-32s  no comparable competitor measurement" % name)
            continue

        our_us, our_spread = ours[name]
        their_us, their_spread = theirs[name]["graph"]
        ratio = their_us / our_us

        nbytes, nflops = WORK.get(name, (0, 0))
        gbs = nbytes / (our_us * 1e-6) if nbytes else 0.0
        tfs = nflops / (our_us * 1e-6) if nflops else 0.0
        pct_bw = gbs / CARD_PEAK_BANDWIDTH * 100

        # A kernel at most of the bandwidth roofline is memory bound and cannot be
        # made much faster; one far from BOTH rooflines is losing to poor data reuse.
        bound = "MEMORY (at roofline)" if pct_bw > 55 else "reuse-limited"
        if ratio >= 1.10:
            wins += 1
        elif ratio <= 0.91:
            losses += 1

        print("%-32s%9.1f%7.1f%%%10.1f%7.1f%%%7.2fx%10.0f%5.0f%%%10.2f  %s"
              % (name, our_us, our_spread, their_us, their_spread, ratio,
                 gbs / 1e9, pct_bw, tfs / 1e12, bound))

    # Write the ratios where the release gate can find them. A kernel without a
    # current-protocol competitor ratio is not releasable: every number before this
    # existed was ours-vs-ours, which cannot tell you whether a kernel is good.
    out = os.path.join("artifacts", "competitor-ratios.tsv")
    os.makedirs("artifacts", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# competitor: PyTorch/cuDNN, CUDA-graph lane, allow_tf32=False, locked clocks\n")
        fh.write("kernel\tours_us\tcompetitor_us\tratio\tprotocol\n")
        for name in sorted(ours):
            if name not in theirs or "graph" not in theirs[name]:
                continue
            our_us, _ = ours[name]
            their_us, _ = theirs[name]["graph"]
            fh.write("%s\t%.3f\t%.3f\t%.4f\tp4\n" % (name, our_us, their_us, their_us / our_us))
    print()
    print("  ratios written to " + out)
    print("  wins at >=1.10x: %d    losses at <=0.91x: %d" % (wins, losses))
    print("  Competitor is the CUDA-GRAPH lane -- the strongest form. Eager PyTorch")
    print("  allocates an output tensor per call and pays full launch overhead, which")
    print("  our fixed-buffer launch does not; graph replay removes both.")
    print("  Cross-process, so this cannot be paired the way an in-process A/B is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
