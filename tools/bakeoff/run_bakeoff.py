"""
Repeatable head-to-head: our generated kernels against PyTorch/cuDNN.

Runs both lanes, joins them by kernel name, and prints ratios beside a roofline
analysis. The roofline is the point: a ratio on its own says who is faster, while
achieved bandwidth and FLOP rate say WHY, and therefore what to do about it.

Usage (from the repo root):
    dotnet AiDotNet.Tensors.Benchmarks.dll --kernel-competitor

The .NET authority supplies both --protocol and --dispatch. Direct Python diagnostics
must pass the exact values printed by that authority; invented fingerprints are not
release evidence.

Environment:
    BAKEOFF_PYTHON   interpreter with torch+cu13 (default: the aidotnet cache venv)
    BAKEOFF_DOTNET   path to the benchmarks dll
"""

import argparse
import os
import re
import subprocess
import sys

LANE_VERSION = "cudnn-graph-fp32-v5"
COMPETITOR_PLAN_STRATEGIES = (
    "default", "default", "default", "default",
    "exhaustive", "exhaustive", "heuristic",
)

HOME = os.path.expanduser("~")
DEFAULT_PY = os.path.join(HOME, ".cache", "aidotnet-direct-ptx-py312", "Scripts", "python.exe")
TORCH_LIB = os.path.join(HOME, ".cache", "aidotnet-direct-ptx-py312",
                         "Lib", "site-packages", "torch", "lib")
DEFAULT_DLL = os.path.join("tests", "AiDotNet.Tensors.Benchmarks", "bin", "Release",
                           "net10.0", "AiDotNet.Tensors.Benchmarks.dll")


def conv_transpose_contract():
    """Returns N,C,IH,IW,OH,OW,output_padding from the .NET catalog authority."""
    raw = os.environ.get("BAKEOFF_CONV_TRANSPOSE_CONTRACT", "")
    try:
        values = tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise RuntimeError("invalid BAKEOFF_CONV_TRANSPOSE_CONTRACT") from exc
    if len(values) != 7:
        raise RuntimeError("BAKEOFF_CONV_TRANSPOSE_CONTRACT must contain 7 integers")
    n, c, ih, iw, oh, ow, output_padding = values
    if min(n, c, ih, iw, oh, ow) <= 0 or output_padding not in (0, 1):
        raise RuntimeError("invalid transposed-convolution catalog contract")
    return values


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


def work_for_run():
    """Adds contract-dependent roofline work without parsing it during import."""
    transpose = conv_transpose_contract()
    work = dict(WORK)
    work["conv_transpose2d_3x3_stride2"] = (
        (transpose[0] * transpose[1] *
         (transpose[2] * transpose[3] + transpose[4] * transpose[5])) * 4,
        transpose[0] * transpose[1] * transpose[2] * transpose[3] * 9 * 2,
    )
    return work


def run_ours(dll, selector):
    """Parses --kernel-bench output into {kernel: (us, spread_pct)}."""
    completed = None
    command = ["dotnet", dll, "--kernel-bench"]
    if selector != "all":
        command.append(selector)
    for attempt in range(1, 4):
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            break
        diagnostic = completed.stderr
        contaminated = ("Foreign GPU workload detected" in diagnostic or
                        "GPU is not benchmark-ready" in diagnostic)
        if not contaminated or attempt == 3:
            raise RuntimeError("generated lane command %r failed (%d): %s" %
                               (command, completed.returncode, diagnostic.strip()))
        print("  generated lane attempt %d contaminated; retrying clean attempt" % attempt,
              flush=True)
    assert completed is not None
    out = completed.stdout
    got = {}
    for line in out.splitlines():
        m = re.match(r"^(\S+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m:
            got[m.group(1)] = (float(m.group(3)), float(m.group(5)))
    if not got:
        raise RuntimeError("generated lane succeeded but produced no parseable kernel rows")
    return got


def run_torch_once(python, script, strategy, selector):
    """Parses the competitor lane into {kernel: {'eager': us, 'graph': (us, spread)}}."""
    env = dict(os.environ)
    env["PATH"] = TORCH_LIB + os.pathsep + env.get("PATH", "")
    env["BAKEOFF_CUDNN_SEARCH"] = strategy
    env["BAKEOFF_SELECTOR"] = selector
    command = [python, script]
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError("competitor lane command %r failed (%d): %s" %
                           (command, completed.returncode, completed.stderr.strip()))
    out = completed.stdout

    got = {}
    device = "unknown"
    for line in out.splitlines():
        parts = line.split("\t")
        if parts and parts[0] == "DEVICE":
            device = " | ".join(parts[1:])
            continue
        if len(parts) < 5 or parts[0] not in ("RESULT", "GRAPH"):
            if parts and parts[0] in ("GRAPHWRONG", "GRAPHFAIL"):
                print("  competitor lane problem: " + line)
            continue
        entry = got.setdefault(parts[1], {})
        if parts[0] == "RESULT":
            entry["eager"] = float(parts[2])
        else:
            entry["graph"] = (float(parts[2]), float(parts[4]))
    return got, device


def run_torch(python, script, max_spread_pct, selector):
    """Selects the fastest stable cuDNN plan seen across fresh-process searches."""
    attempts = []
    expected_device = None
    for attempt, strategy in enumerate(COMPETITOR_PLAN_STRATEGIES, 1):
        print("  cuDNN plan-search attempt %d/%d (%s)" %
              (attempt, len(COMPETITOR_PLAN_STRATEGIES), strategy), flush=True)
        got, device = run_torch_once(python, script, strategy, selector)
        if expected_device is None:
            expected_device = device
        elif device != expected_device:
            raise RuntimeError("competitor device changed between plan-search attempts")
        attempts.append((got, strategy))

    selected = {}
    for name in set().union(*(attempt.keys() for attempt, _ in attempts)):
        rows = [(attempt[name], strategy) for attempt, strategy in attempts
                if name in attempt]
        entry = {}
        eager = [row["eager"] for row, _ in rows if "eager" in row]
        if eager:
            entry["eager"] = min(eager)
        graphs = [(row["graph"][0], row["graph"][1], strategy)
                  for row, strategy in rows if "graph" in row]
        if graphs:
            stable_graphs = [value for value in graphs if value[1] <= max_spread_pct]
            eligible = stable_graphs or graphs
            fastest = min(eligible, key=lambda value: value[0])
            times = [value[0] for value in eligible]
            plan_spread = (max(times) / min(times) - 1.0) * 100.0
            entry["graph"] = (fastest[0], fastest[1], plan_spread, fastest[2])
        selected[name] = entry
    return selected, expected_device or "unknown"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Versioned generated-kernel vs PyTorch/cuDNN release lane")
    parser.add_argument("--protocol", required=True,
                        help="measurement protocol tag supplied by the .NET authority, e.g. p9")
    parser.add_argument("--dispatch", required=True,
                        help="exact generated-dispatch fingerprint supplied by the .NET authority")
    parser.add_argument("--output", default=os.path.join("artifacts", "competitor-ratios.tsv"))
    parser.add_argument("--max-spread-pct", type=float, default=5.0)
    parser.add_argument("--selector", default="all",
                        help="one catalog kernel, or all")
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=760.0,
                        help="roofline display only; default is RTX 3080")
    parser.add_argument("--peak-fp32-tflops", type=float, default=29.8,
                        help="roofline display only; default is RTX 3080")
    return parser.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"p[1-9][0-9]*", args.protocol):
        raise RuntimeError("--protocol must be an explicit version tag such as p9")
    if not re.fullmatch(r"sha256-[0-9a-f]{64}", args.dispatch):
        raise RuntimeError("--dispatch must be an exact sha256 dispatch fingerprint")
    if args.max_spread_pct <= 0:
        raise RuntimeError("--max-spread-pct must be positive")
    work = work_for_run()
    if args.selector != "all" and args.selector not in work:
        raise RuntimeError("--selector does not name a competitor kernel: " + args.selector)

    # A failed refresh must not leave an older current-tag artifact available to the
    # release reader. The requested output belongs to this run; invalidate it before
    # measuring and recreate it only after at least one stable comparison exists.
    if os.path.exists(args.output):
        os.remove(args.output)

    python = os.environ.get("BAKEOFF_PYTHON", DEFAULT_PY)
    dll = os.environ.get("BAKEOFF_DOTNET", DEFAULT_DLL)
    script = os.path.join("tests", "AiDotNet.Tensors.Benchmarks", "BaselineRunners",
                          "py", "run_codegen_bakeoff.py")

    print("measuring our generated kernels ...")
    ours = run_ours(dll, args.selector)
    print("measuring PyTorch/cuDNN (eager and CUDA-graph) ...")
    theirs, competitor_device = run_torch(
        python, script, args.max_spread_pct, args.selector)
    print()

    header = ("kernel", "ours us", "spread", "cuDNN us", "spread", "ratio",
              "our GB/s", "%bw", "our TF/s", "bound")
    print("%-32s%9s%8s%10s%8s%8s%10s%6s%10s  %s" % header)

    wins = losses = refused = comparable = 0
    accepted = []
    for name in sorted(ours):
        if name not in theirs or "graph" not in theirs[name]:
            print("%-32s  no comparable competitor measurement" % name)
            continue

        our_us, our_spread = ours[name]
        their_us, their_spread, their_plan_spread, their_plan_strategy = \
            theirs[name]["graph"]
        stable = our_spread <= args.max_spread_pct and their_spread <= args.max_spread_pct
        ratio = their_us / our_us if stable else None

        nbytes, nflops = work.get(name, (0, 0))
        gbs = nbytes / (our_us * 1e-6) if nbytes else 0.0
        tfs = nflops / (our_us * 1e-6) if nflops else 0.0
        pct_bw = gbs / (args.peak_bandwidth_gbs * 1e9) * 100
        pct_fp32 = tfs / (args.peak_fp32_tflops * 1e12) * 100

        # A kernel at most of the bandwidth roofline is memory bound and cannot be
        # made much faster; one far from BOTH rooflines is losing to poor data reuse.
        bound = ("MEMORY (at roofline)" if pct_bw > 55 else
                 "COMPUTE (at roofline)" if pct_fp32 > 55 else "reuse-limited")
        if not stable:
            refused += 1
            print("%-32s%9.1f%7.1f%%%10.1f%7.1f%%       -%10.0f%5.0f%%%10.2f  UNSTABLE -- refused"
                  % (name, our_us, our_spread, their_us, their_spread,
                     gbs / 1e9, pct_bw, tfs / 1e12))
            continue

        comparable += 1
        accepted.append((name, our_us, their_us, ratio, our_spread, their_spread,
                         their_plan_spread, their_plan_strategy))
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
    if comparable == 0:
        raise RuntimeError("no stable generated/cuDNN comparisons; evidence not written")

    out = args.output
    output_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(output_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# competitor: PyTorch/cuDNN, CUDA-graph lane, allow_tf32=False\n")
        fh.write("# device: %s\n" % competitor_device)
        fh.write("# stability: each side spread <= %.3f%%\n" % args.max_spread_pct)
        fh.write("kernel\tours_us\tcompetitor_us\tratio\tours_spread_pct\t"
                 "competitor_spread_pct\tcompetitor_plan_spread_pct\t"
                 "competitor_plan_strategy\t"
                 "lane\tdispatch\tprotocol\n")
        for (name, our_us, their_us, ratio, our_spread, their_spread,
             their_plan_spread, their_plan_strategy) in accepted:
            fh.write("%s\t%.3f\t%.3f\t%.4f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\n" %
                     (name, our_us, their_us, ratio, our_spread, their_spread,
                      their_plan_spread, their_plan_strategy, LANE_VERSION,
                      args.dispatch, args.protocol))
    print()
    print("  ratios written to " + out)
    print("  wins at >=1.10x: %d    losses at <=0.91x: %d" % (wins, losses))
    print("  refused as unstable: %d (spread gate %.1f%%)" %
          (refused, args.max_spread_pct))
    print("  Competitor is the CUDA-GRAPH lane -- the strongest form. Eager PyTorch")
    print("  allocates an output tensor per call and pays full launch overhead, which")
    print("  our fixed-buffer launch does not; graph replay removes both.")
    print("  Cross-process, so this cannot be paired the way an in-process A/B is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
