# Competitor lane for the codegen bake-off.
#
# Measures PyTorch/cuDNN on exactly the shapes the generated-kernel catalog uses, with
# the SAME timing protocol the C# side uses, so the two columns are comparable:
#
#   * a timed region is LAUNCHES_PER_SAMPLE launches followed by one synchronize, so
#     per-call sync latency is amortised instead of dominating (the C# harness measured
#     a 21.5 us launch floor and a 5.5 P95/median ratio without this);
#   * SAMPLES samples per run, median reported;
#   * up to STABILITY_ATTEMPTS runs, retaining the latest RUNS until they agree;
#     the median and full spread of that accepted window are printed.
#
# Fairness note that the numbers cannot show on their own: our kernels are FUSED
# (conv + bias + ReLU in one kernel). PyTorch runs conv+bias through cuDNN and then a
# separate ReLU kernel, because Inductor cannot fuse through a cuDNN call. So the
# comparison is deliberately "our one kernel" against "the composition PyTorch
# actually executes", and the unfused conv-only time is printed alongside so the
# fusion component of any win is visible rather than hidden.

import os
import sys
import time

import torch

WARMUP = 20
SAMPLES = 51
LAUNCHES_PER_SAMPLE = 50
RUNS = 3
STABILITY_ATTEMPTS = 15
STABLE_SPREAD_PCT = 5.0


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


def time_op(fn):
    """Median us per launch, using batched timed regions. Returns (median, p95)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        for _ in range(LAUNCHES_PER_SAMPLE):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        samples.append(elapsed / LAUNCHES_PER_SAMPLE * 1e6)
    samples.sort()
    mid = samples[len(samples) // 2]
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
    return mid, p95


def best_of_runs(fn):
    """Median of the latest three agreeing runs, with contaminated runs aged out."""
    medians = []
    tails = []
    for _ in range(STABILITY_ATTEMPTS):
        mid, p95 = time_op(fn)
        medians.append(mid)
        tails.append(p95 / mid if mid > 0 else float("nan"))
        if len(medians) > RUNS:
            medians.pop(0)
            tails.pop(0)
        if len(medians) == RUNS:
            spread = ((max(medians) / min(medians) - 1.0) * 100.0
                      if min(medians) > 0 else float("nan"))
            if spread <= STABLE_SPREAD_PCT:
                break
    ordered = sorted(medians)
    spread = ((ordered[-1] / ordered[0] - 1.0) * 100.0
              if ordered[0] > 0 else float("nan"))
    return ordered[len(ordered) // 2], max(tails), spread


def emit(name, fn, note=""):
    median, tail, spread = best_of_runs(fn)
    print(
        "RESULT\t%s\t%.3f\t%.3f\t%.3f\t%s" % (name, median, tail, spread, note),
        flush=True,
    )


def emit_graphed(name, fn, note=""):
    """
    Same op, but captured into a CUDA graph and replayed.

    This is the STRONGEST form of the competitor and the fair one. In the eager lane
    PyTorch allocates a fresh output tensor on every call and pays full launch overhead,
    while our kernel writes into a preallocated buffer -- an advantage to us that has
    nothing to do with kernel quality. Graph replay reuses the captured output buffer
    and collapses launch overhead, so what remains is the kernel work itself.
    """
    try:
        # Capture on a side stream, as the CUDA graph API requires.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        torch.cuda.synchronize()
    except Exception as exc:                      # noqa: BLE001 - report, do not hide
        print("GRAPHFAIL\t%s\t%s" % (name, str(exc).split("\n")[0]), flush=True)
        return

    # PROVE THE REPLAY DOES THE WORK. Graph replay showed dense convolutions 5-6x
    # faster than eager, which is far more than launch overhead can explain, so the
    # obvious suspicion is that the replay is not computing anything. Zero the captured
    # output, replay, and require it to reproduce the eager result. A no-op replay, or
    # one reading stale buffers, fails here instead of being published as a speedup.
    reference = fn().clone()
    captured.zero_()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    deviation = (captured - reference).abs().max().item()
    scale = max(1.0, reference.abs().max().item())
    if deviation / scale > 1e-5:
        print("GRAPHWRONG\t%s\tmax abs dev %.3e -- replay does not reproduce eager"
              % (name, deviation), flush=True)
        return

    median, tail, spread = best_of_runs(graph.replay)
    print(
        "GRAPH\t%s\t%.3f\t%.3f\t%.3f\t%s" % (name, median, tail, spread, note),
        flush=True,
    )


def emit_both(name, fn, note=""):
    emit(name, fn, note)
    emit_graphed(name, fn, note)


def main():
    if not torch.cuda.is_available():
        print("ERROR\tno CUDA device", flush=True)
        return 1

    search = os.environ.get("BAKEOFF_CUDNN_SEARCH", "")
    if search == "default":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 10
    elif search == "exhaustive":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 0
    elif search == "heuristic":
        torch.backends.cudnn.benchmark = False
    else:
        raise RuntimeError("BAKEOFF_CUDNN_SEARCH must be default, exhaustive, or heuristic")

    # TRUE FP32, NOT TF32. PyTorch defaults allow_tf32=True, which routes dense
    # convolution to tensor cores at 10-bit mantissa -- a DIFFERENT operation from the
    # exact fp32 our kernels compute and verify to 0.000E+000 against an fp64 oracle.
    # Profiling caught it: the default path runs
    # cutlass_tensorop_s1688fprop_optimized_tf32_... on tensor cores.
    #
    # Measured on the dense 3x3 shape under CUDA graphs, TF32 is not even faster here
    # (27.55 us vs 24.80 us true FP32) because it pays NCHW->NHWC layout transforms,
    # while giving up 3.1e-04 relative accuracy. So disabling it costs the competitor
    # nothing and makes the comparison honest in both directions.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    torch.manual_seed(0)
    selector = os.environ.get("BAKEOFF_SELECTOR", "all")

    def selected(name):
        return selector == "all" or selector == name

    print("DEVICE\t%s\ttorch %s\tcudnn %s"
          % (torch.cuda.get_device_name(0), torch.__version__, torch.backends.cudnn.version()),
          flush=True)

    # ---- depthwise 3x3, N32/C64/56x56 (groups == channels is how PyTorch spells depthwise)
    x = torch.randn(32, 64, 56, 56, device=dev)
    dw = torch.randn(64, 1, 3, 3, device=dev)
    dwb = torch.randn(64, device=dev)

    if selected("depthwise_conv2d_3x3"):
        emit_both("depthwise_conv2d_3x3",
             lambda: torch.nn.functional.conv2d(x, dw, None, 1, 1, 1, 64))
    if selected("depthwise_conv2d_3x3_bias_relu"):
        emit_both("depthwise_conv2d_3x3_bias_relu",
             lambda: torch.relu(torch.nn.functional.conv2d(x, dw, dwb, 1, 1, 1, 64)),
             "conv+bias then a separate relu kernel")
    if selector == "all":
        emit("depthwise_conv2d_3x3_conv_only",
             lambda: torch.nn.functional.conv2d(x, dw, dwb, 1, 1, 1, 64),
             "unfused reference: conv+bias, no relu")

    # ---- dense 1x1, N16/C64->K64/28x28
    x1 = torch.randn(16, 64, 28, 28, device=dev)
    w1 = torch.randn(64, 64, 1, 1, device=dev)
    b1 = torch.randn(64, device=dev)
    if selected("conv2d_1x1_bias_relu"):
        emit_both("conv2d_1x1_bias_relu",
             lambda: torch.relu(torch.nn.functional.conv2d(x1, w1, b1)),
             "conv+bias then a separate relu kernel")
    if selector == "all":
        emit("conv2d_1x1_conv_only",
             lambda: torch.nn.functional.conv2d(x1, w1, b1),
             "unfused reference: conv+bias, no relu")

    # ---- dense 3x3, N8/C32->K64/28x28
    x3 = torch.randn(8, 32, 28, 28, device=dev)
    w3 = torch.randn(64, 32, 3, 3, device=dev)
    b3 = torch.randn(64, device=dev)
    if selected("conv2d_3x3_bias_relu"):
        emit_both("conv2d_3x3_bias_relu",
             lambda: torch.relu(torch.nn.functional.conv2d(x3, w3, b3, 1, 1)),
             "conv+bias then a separate relu kernel")
    if selector == "all":
        emit("conv2d_3x3_conv_only",
             lambda: torch.nn.functional.conv2d(x3, w3, b3, 1, 1),
             "unfused reference: conv+bias, no relu")

    # ---- DEEP EPILOGUE, N16/C64->K64/28x28. The structural exploit: PyTorch cannot
    # fuse through a cuDNN call, so each elementwise stage costs it a kernel launch and
    # a full round trip of the tensor, while costing us one instruction in a loop we are
    # already running. Measured marginal cost on this shape: bias +2.84 us, relu +8.14,
    # scale +6.42 -- 17.40 us of epilogue against a 23.75 us convolution.
    s1 = torch.randn(64, 1, 1, device=dev)
    if selected("conv2d_1x1_deep_epilogue"):
        emit_both("conv2d_1x1_deep_epilogue",
             lambda: torch.relu(torch.nn.functional.conv2d(x1, w1, b1) * s1),
             "conv+bias then scale then relu: three kernels PyTorch cannot fuse")

    # ---- POINTWISE (#890 phase 4a), 8192x1024 unless noted. Each is PyTorch's own
    # fastest spelling: a fused native op where one exists (gelu, silu, softmax,
    # layer_norm, rms_norm, batch_norm, group_norm), the plain composition otherwise.
    F = torch.nn.functional
    xr = torch.randn(8192, 1024, device=dev)
    yr = torch.randn(8192, 1024, device=dev)
    br = torch.randn(1024, device=dev)
    if selected("bias_gelu"):
        emit_both("bias_gelu", lambda: F.gelu(xr + br, approximate="tanh"),
                  "add then gelu: two kernels")
    if selected("bias_swish"):
        emit_both("bias_swish", lambda: F.silu(xr + br), "add then silu: two kernels")
    if selected("residual_add_relu"):
        emit_both("residual_add_relu", lambda: torch.relu(xr + yr), "add then relu: two kernels")
    if selected("multiply"):
        emit_both("multiply", lambda: xr * yr)
    if selected("sigmoid"):
        emit_both("sigmoid", lambda: torch.sigmoid(xr))
    if selected("tanh"):
        emit_both("tanh", lambda: torch.tanh(xr))

    xbn = torch.randn(32, 64, 56, 56, device=dev)
    gbn = torch.randn(64, device=dev)
    bbn = torch.randn(64, device=dev)
    rmean = torch.zeros(64, device=dev)
    rvar = torch.ones(64, device=dev)
    if selected("batchnorm_inference_relu"):
        emit_both("batchnorm_inference_relu",
                  lambda: torch.relu_(F.batch_norm(xbn, rmean, rvar, gbn, bbn, False)),
                  "cuDNN inference batch_norm then in-place relu")

    # ---- NORMALISATION programs. Ours run as 2-3 passes (statistics, then the affine);
    # PyTorch's are single fused kernels, so these measure the price of the passes.
    gr = torch.randn(1024, device=dev)
    if selected("softmax_rows"):
        emit_both("softmax_rows", lambda: torch.softmax(xr, dim=1))
    if selected("layernorm"):
        emit_both("layernorm", lambda: F.layer_norm(xr, (1024,), gr, br, 1e-5))
    x64 = torch.randn(8192, 64, device=dev)
    g64 = torch.randn(64, device=dev)
    b64 = torch.randn(64, device=dev)
    if selected("layernorm_d64"):
        emit_both("layernorm_d64", lambda: F.layer_norm(x64, (64,), g64, b64, 1e-5))
    if selected("rmsnorm"):
        emit_both("rmsnorm", lambda: F.rms_norm(xr, (1024,), gr, 1e-6))
    if selected("batchnorm2d"):
        emit_both("batchnorm2d",
                  lambda: F.batch_norm(xbn, rmean.clone(), rvar.clone(), gbn, bbn, True, 0.1, 1e-5),
                  "training-mode cuDNN batch_norm")
    # ---- BACKWARD, against PyTorch's own fused backward kernels (not autograd, which
    # would re-run the forward). Each returns dx; the op computes every gradient regardless.
    dyr = torch.randn(8192, 1024, device=dev)
    mean_r = xr.mean(dim=1, keepdim=True)
    rstd_r = torch.rsqrt(xr.var(dim=1, unbiased=False, keepdim=True) + 1e-5)
    if selected("layernorm_backward"):
        emit_both("layernorm_backward",
                  lambda: torch.ops.aten.native_layer_norm_backward(
                      dyr, xr, [1024], mean_r, rstd_r, gr, br, [True, True, True])[0],
                  "native_layer_norm_backward: dx, dgamma, dbeta")
    _, rms_rstd = torch.ops.aten._fused_rms_norm(xr, [1024], gr, 1e-6)
    if selected("rmsnorm_backward"):
        emit_both("rmsnorm_backward",
                  lambda: torch.ops.aten._fused_rms_norm_backward(
                      dyr, xr, [1024], rms_rstd, gr, [True, True])[0],
                  "_fused_rms_norm_backward: dx, dgamma")
    ysm = torch.softmax(xr, dim=1)
    if selected("softmax_backward"):
        emit_both("softmax_backward",
                  lambda: torch._softmax_backward_data(dyr, ysm, 1, torch.float32))

    xgn = torch.randn(32, 64, 32, 32, device=dev)
    if selected("groupnorm"):
        emit_both("groupnorm", lambda: F.group_norm(xgn, 8, gbn, bbn, 1e-5))
    _, gn_mean, gn_rstd = torch.ops.aten.native_group_norm(xgn, gbn, bbn, 32, 64, 1024, 8, 1e-5)
    dygn = torch.randn_like(xgn)
    if selected("groupnorm_backward"):
        emit_both("groupnorm_backward",
                  lambda: torch.ops.aten.native_group_norm_backward(
                      dygn, xgn, gn_mean, gn_rstd, gbn, 32, 64, 1024, 8, [True, True, True])[0],
                  "native_group_norm_backward: dx, dgamma, dbeta")
    _, bn_mean, bn_invstd = torch.ops.aten.native_batch_norm(
        xbn, gbn, bbn, rmean.clone(), rvar.clone(), True, 0.1, 1e-5)
    dybn = torch.randn_like(xbn)
    if selected("batchnorm2d_backward"):
        emit_both("batchnorm2d_backward",
                  lambda: torch.ops.aten.native_batch_norm_backward(
                      dybn, xbn, gbn, rmean, rvar, bn_mean, bn_invstd, True, 1e-5, [True, True, True])[0],
                  "native_batch_norm_backward (cuDNN): dx, dgamma, dbeta")

    # ---- 2x2 max pool, N32/C64/112x112
    xp = torch.randn(32, 64, 112, 112, device=dev)
    if selected("maxpool2d_2x2"):
        emit_both("maxpool2d_2x2", lambda: torch.nn.functional.max_pool2d(xp, 2, 2))

    # ---- transposed depthwise 3x3 stride 2; shape comes from the .NET catalog
    if selected("conv_transpose2d_3x3_stride2"):
        tn, tc, tih, tiw, toh, tow, output_padding = conv_transpose_contract()
        xt = torch.randn(tn, tc, tih, tiw, device=dev)
        wt = torch.randn(tc, 1, 3, 3, device=dev)
        probe = torch.nn.functional.conv_transpose2d(
            xt, wt, None, stride=2, padding=1,
            output_padding=output_padding, groups=tc)
        if tuple(probe.shape) != (tn, tc, toh, tow):
            raise RuntimeError("cuDNN transposed-convolution result disagrees with catalog contract")
        del probe
        emit_both("conv_transpose2d_3x3_stride2",
             lambda: torch.nn.functional.conv_transpose2d(
                 xt, wt, None, stride=2, padding=1,
                 output_padding=output_padding, groups=tc))

    # ---- gradient with respect to the data, matching the derived adjoint kernels
    from torch.nn.grad import conv2d_input

    gdw = torch.randn(32, 64, 56, 56, device=dev)
    if selected("depthwise_conv2d_3x3_bwd_data"):
        emit_both("depthwise_conv2d_3x3_bwd_data",
             lambda: conv2d_input(list(x.shape), dw, gdw, 1, 1, 1, 64))

    g1 = torch.randn(16, 64, 28, 28, device=dev)
    if selected("conv2d_1x1_bwd_data"):
        emit_both("conv2d_1x1_bwd_data",
             lambda: conv2d_input(list(x1.shape), w1, g1))

    g3 = torch.randn(8, 64, 28, 28, device=dev)
    if selected("conv2d_3x3_bwd_data"):
        emit_both("conv2d_3x3_bwd_data",
             lambda: conv2d_input(list(x3.shape), w3, g3, 1, 1))

    # ---- gradient with respect to the WEIGHTS.
    #
    # These three had no competitor row at all, so the split-K wins on them (17.1x,
    # 35.1x and 2.0x) were measured only against our own prior lowering. A 17x over
    # ourselves says nothing about where we stand, and the release gate correctly
    # refused to call them releasable while the competitor column read MISSING.
    #
    # Every tensor below is one the forward and data-gradient rows already use, so the
    # comparison is on exactly the shapes the catalog benches -- the same discipline the
    # rest of this file follows.
    from torch.nn.grad import conv2d_weight

    if selected("depthwise_conv2d_3x3_bwd_weights"):
        emit_both("depthwise_conv2d_3x3_bwd_weights",
             lambda: conv2d_weight(x, list(dw.shape), gdw, 1, 1, 1, 64))

    if selected("conv2d_1x1_bwd_weights"):
        emit_both("conv2d_1x1_bwd_weights",
             lambda: conv2d_weight(x1, list(w1.shape), g1))

    if selected("conv2d_3x3_bwd_weights"):
        emit_both("conv2d_3x3_bwd_weights",
             lambda: conv2d_weight(x3, list(w3.shape), g3, 1, 1))

    return 0


if __name__ == "__main__":
    sys.exit(main())
