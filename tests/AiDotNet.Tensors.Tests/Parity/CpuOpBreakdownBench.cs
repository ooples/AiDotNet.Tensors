// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The kernel switches it sets are .NET 5+ (the nightly runs it on net10.0).
#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// Per-operation CPU timings for every convolution and linear layer of the MLP, CNN and ResNet-18 parity networks
/// (forward, input gradient, weight gradient), at 1, 2 and 4 threads. <c>tools/parity/cpu_op_breakdown_torch.py</c> times
/// the same operations on PyTorch with the same output format, so the two tables line up row for row and show which
/// operation loses on a given CPU. Opt-in (<c>AIDOTNET_OP_BREAKDOWN=1</c>): the nightly parity job runs it on
/// workflow_dispatch, where the hosted runner's CPU (AVX2 or AVX-512, AMD or Intel) differs from run to run.
/// </summary>
[Collection("ConvPerfSerial")]
public class CpuOpBreakdownBench
{
    private readonly ITestOutputHelper _output;

    public CpuOpBreakdownBench(ITestOutputHelper output) => _output = output;

    // (network, batch, inChannels, outChannels, height/width, kernel, stride, padding)
    private static readonly (string Net, int B, int Cin, int Cout, int Hw, int K, int S, int P)[] Convs =
    {
        ("cnn", 64, 1, 32, 28, 3, 1, 1),
        ("cnn", 64, 32, 64, 14, 3, 1, 1),
        ("resnet18", 8, 3, 64, 32, 3, 1, 1),
        ("resnet18", 8, 64, 64, 32, 3, 1, 1),
        ("resnet18", 8, 64, 128, 32, 3, 2, 1),
        ("resnet18", 8, 64, 128, 32, 1, 2, 0),
        ("resnet18", 8, 128, 128, 16, 3, 1, 1),
        ("resnet18", 8, 128, 256, 16, 3, 2, 1),
        ("resnet18", 8, 128, 256, 16, 1, 2, 0),
        ("resnet18", 8, 256, 256, 8, 3, 1, 1),
        ("resnet18", 8, 256, 512, 8, 3, 2, 1),
        ("resnet18", 8, 256, 512, 8, 1, 2, 0),
        ("resnet18", 8, 512, 512, 4, 3, 1, 1),
        // The stride-1 equivalents of the 1x1 stride-2 downsample convolutions (on the subsampled input).
        ("probe", 8, 64, 128, 16, 1, 1, 0),
        ("probe", 8, 128, 256, 8, 1, 1, 0),
        ("probe", 8, 256, 512, 4, 1, 1, 0),
    };

    // MobileNetV2's 3x3 depthwise convolutions at 224 (network, batch, channels, height/width, stride), padding 1.
    private static readonly (string Net, int B, int C, int Hw, int S)[] Depthwise =
    {
        ("mobilenetv2", 16, 32, 112, 1),
        ("mobilenetv2", 16, 96, 112, 2),
        ("mobilenetv2", 16, 144, 56, 1),
        ("mobilenetv2", 16, 144, 56, 2),
        ("mobilenetv2", 16, 192, 28, 1),
        ("mobilenetv2", 16, 192, 28, 2),
        ("mobilenetv2", 16, 384, 14, 1),
        ("mobilenetv2", 16, 576, 14, 1),
        ("mobilenetv2", 16, 576, 14, 2),
        ("mobilenetv2", 16, 960, 7, 1),
    };

    // (network, batch, in, out)
    private static readonly (string Net, int B, int In, int Out)[] Linears =
    {
        ("mlp", 128, 784, 512),
        ("mlp", 128, 512, 256),
        ("mlp", 128, 256, 10),
        ("cnn", 64, 3136, 128),
        ("cnn", 64, 128, 10),
        ("resnet18", 8, 512, 10),
    };

    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void PrintCpuOpBreakdown()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_OP_BREAKDOWN") == "1",
            "Opt-in: set AIDOTNET_OP_BREAKDOWN=1.");
        var engine = new CpuEngine();
        var previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        int previousThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        int previousPassBytes = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardPassKernelBytes;
        // Kernel-tuning experiments: the direct forward's input-channel pass budget, in bytes of tile kernel slice.
        if (int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_OP_FWD_PASS_BYTES"), out int passBytes) && passBytes > 0)
            AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardPassKernelBytes = passBytes;
        int previousRunPassBytes = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardRunPassKernelBytes;
        int previousRunPositions = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardRunPositions;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_OP_RUN_PASS_BYTES"), out int runPassBytes) && runPassBytes > 0)
            AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardRunPassKernelBytes = runPassBytes;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_OP_RUN_POSITIONS"), out int runPositions) && runPositions > 0)
            AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardRunPositions = runPositions;
        int previousMinPhase = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.MinStridedPhasePositions;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_OP_MIN_PHASE"), out int minPhase) && minPhase > 0)
            AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.MinStridedPhasePositions = minPhase;
        bool previous2D = AiDotNet.Tensors.Engines.Simd.SimdGemm.UseDirectParallel2D;
        if (Environment.GetEnvironmentVariable("AIDOTNET_OP_NO_2D") == "1")
            AiDotNet.Tensors.Engines.Simd.SimdGemm.UseDirectParallel2D = false;
        int previousDwMaxWidth = AiDotNet.Tensors.Engines.Simd.DepthwiseConv3x3.s_channelBlockMaxWidth;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_OP_DW_CB_MAXW"), out int dwMaxWidth))
            AiDotNet.Tensors.Engines.Simd.DepthwiseConv3x3.s_channelBlockMaxWidth = dwMaxWidth;
        // The stride-1 channel-blocked route has its own width limit, so the 28x28 stride-1 rows can compare both routes.
        int previousDwMaxWidthStride1 = AiDotNet.Tensors.Engines.Simd.DepthwiseConv3x3.s_channelBlockMaxWidthStride1;
        if (int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_OP_DW_CB_MAXW_S1"), out int dwMaxWidthStride1))
            AiDotNet.Tensors.Engines.Simd.DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = dwMaxWidthStride1;
        var rng = new Random(7);
        try
        {
            foreach (int threads in ThreadCounts())
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                foreach (var c in Convs)
                {
                    var x = Random(rng, c.B, c.Cin, c.Hw, c.Hw);
                    var w = Random(rng, c.Cout, c.Cin, c.K, c.K);
                    int o = (c.Hw + 2 * c.P - c.K) / c.S + 1;
                    var dy = Random(rng, c.B, c.Cout, o, o);
                    int[] stride = { c.S, c.S }, pad = { c.P, c.P }, dil = { 1, 1 };
                    string shape = $"{c.Net} b{c.B} {c.Cin}->{c.Cout} {c.Hw}x{c.Hw} k{c.K} s{c.S}";
                    // The Into forms with preallocated outputs, as FusedConv2D and its tape backward call them.
                    var y = new Tensor<float>(new[] { c.B, c.Cout, o, o });
                    var dx = new Tensor<float>(x.Shape.ToArray());
                    var dw = new Tensor<float>(w.Shape.ToArray());
                    Row("conv.fwd", shape, threads, () => engine.Conv2DInto(y, x, w, stride, pad, dil));
                    Row("conv.dgrad", shape, threads, () => engine.Conv2DBackwardInputInto(dx, dy, w, x.Shape.ToArray(), stride, pad, dil, accumulate: false));
                    Row("conv.wgrad", shape, threads, () => engine.Conv2DBackwardKernelInto(dw, dy, x, w.Shape.ToArray(), stride, pad, dil, accumulate: false));
                }
                foreach (var d in Depthwise)
                {
                    var x = Random(rng, d.B, d.C, d.Hw, d.Hw);
                    var w = Random(rng, d.C, 1, 3, 3);
                    int o = (d.Hw + 2 - 3) / d.S + 1;
                    var dy = Random(rng, d.B, d.C, o, o);
                    int[] stride = { d.S, d.S }, pad = { 1, 1 };
                    string shape = $"{d.Net} b{d.B} dw{d.C} {d.Hw}x{d.Hw} k3 s{d.S}";
                    var dwY = new Tensor<float>(new[] { d.B, d.C, o, o });
                    Row("dwconv.fwd", shape, threads, () => engine.DepthwiseConv2DInto(dwY, x, w, stride, pad));
                    // A fast wrong kernel must not read as a win: the timed output against a direct double-precision sum.
                    if (Selected("dwconv.fwd", shape))
                        AssertDepthwiseForward(x.ToArray(), w.ToArray(), dwY.ToArray(), d.B, d.C, d.Hw, d.S, o, shape, threads);
                    // The gradients into preallocated outputs, as the forward (the engine entry points rent a fresh
                    // multi-megabyte result per call, which this bench's loop never returns, so they timed the GC).
                    // Labelled [kernel]: these time the SIMD kernel alone, without the engine's result rental.
                    var dwDx = new float[x.Length];
                    var dwDw = new float[w.Length];
                    float[] dyA = dy.ToArray(), wA = w.ToArray(), xA = x.ToArray();
                    Row("dwconv.dgrad[kernel]", shape, threads, () => AiDotNet.Tensors.Engines.Simd.DepthwiseConv3x3.BackwardInput(dyA, wA, dwDx, d.B, d.C, d.Hw, d.Hw, d.S));
                    Row("dwconv.wgrad[kernel]", shape, threads, () => AiDotNet.Tensors.Engines.Simd.DepthwiseConv3x3.BackwardKernel(dyA, xA, dwDw, d.B, d.C, d.Hw, d.Hw, d.S));
                }
                foreach (var l in Linears)
                {
                    var x = Random(rng, l.B, l.In);
                    var w = Random(rng, l.In, l.Out);
                    var dy = Random(rng, l.B, l.Out);
                    var wT = engine.TensorTranspose(w);
                    var xT = engine.TensorTranspose(x);
                    string shape = $"{l.Net} b{l.B} {l.In}->{l.Out}";
                    Row("linear.fwd", shape, threads, () => engine.TensorMatMul(x, w));
                    Row("linear.dgrad", shape, threads, () => engine.TensorMatMul(dy, wT));
                    Row("linear.wgrad", shape, threads, () => engine.TensorMatMul(xT, dy));
                    // The kernels a dense layer's backward actually runs (dX = dY·Wᵀ, dW = Xᵀ·dY, both transposes
                    // read in place), into preallocated outputs.
                    float[] xL = x.ToArray(), wL = w.ToArray(), dyL = dy.ToArray();
                    var dxL = new float[l.B * l.In];
                    var dwL = new float[l.In * l.Out];
                    Row("linear.dgrad[kernel]", shape, threads, () => AiDotNet.Tensors.Engines.Autodiff.BackwardFunctions<float>.BackwardGemmInputGradient(dyL, wL, dxL, l.B, l.In, l.Out));
                    Row("linear.wgrad[kernel]", shape, threads, () => AiDotNet.Tensors.Engines.Autodiff.BackwardFunctions<float>.BackwardGemmWeightGradient(xL, dyL, dwL, l.B, l.In, l.Out));
                }
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previousThreads;
            AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardPassKernelBytes = previousPassBytes;
            AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardRunPassKernelBytes = previousRunPassBytes;
            AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.s_forwardRunPositions = previousRunPositions;
            AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.MinStridedPhasePositions = previousMinPhase;
            AiDotNet.Tensors.Engines.Simd.SimdGemm.UseDirectParallel2D = previous2D;
            AiDotNet.Tensors.Engines.Simd.DepthwiseConv3x3.s_channelBlockMaxWidth = previousDwMaxWidth;
            AiDotNet.Tensors.Engines.Simd.DepthwiseConv3x3.s_channelBlockMaxWidthStride1 = previousDwMaxWidthStride1;
            AiDotNetEngine.Current = previousEngine;
        }
    }

    // y[b, c, i, j] = sum over the in-bounds taps of x[b, c, i s + u - 1, j s + v - 1] w[c, 0, u, v] (3x3, padding 1).
    private static void AssertDepthwiseForward(float[] x, float[] w, float[] y, int batch, int channels, int hw, int s,
        int o, string shape, int threads)
    {
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int i = 0; i < o; i++)
                    for (int j = 0; j < o; j++)
                    {
                        double sum = 0, magnitude = 0;
                        for (int u = 0; u < 3; u++)
                            for (int v = 0; v < 3; v++)
                            {
                                int r = i * s + u - 1, q = j * s + v - 1;
                                if (r < 0 || r >= hw || q < 0 || q >= hw) continue;
                                double term = (double)x[((b * channels + c) * hw + r) * hw + q] * w[c * 9 + u * 3 + v];
                                sum += term;
                                magnitude += Math.Abs(term);
                            }
                        float actual = y[((b * channels + c) * o + i) * o + j];
                        Assert.True(Math.Abs(actual - sum) <= 1e-5 * (1 + magnitude),
                            $"dwconv.fwd {shape} t{threads} [{b},{c},{i},{j}]: expected {sum}, got {actual}");
                    }
    }

    // One line per operation, in the format the torch script prints: OPBREAKDOWN <op> | <shape> | t<threads> | <median ms>.
    private void Row(string op, string shape, int threads, Func<Tensor<float>> run) => Row(op, shape, threads, () => { run(); });

    // Profiling filters: AIDOTNET_OP_FILTER (a substring of "<op> | <shape>"), AIDOTNET_OP_THREADS (comma-separated)
    // and AIDOTNET_OP_SAMPLES (timed runs per row, 31 by default), so a profiler can sample one operation alone.
    private static readonly string? Filter = Environment.GetEnvironmentVariable("AIDOTNET_OP_FILTER");
    private static readonly int Samples = int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_OP_SAMPLES"), out int n) && n > 0 ? n : 31;

    private static int[] ThreadCounts()
        => Environment.GetEnvironmentVariable("AIDOTNET_OP_THREADS") is { Length: > 0 } list
            ? list.Split(',').Select(int.Parse).ToArray()
            : new[] { 1, 2, 4 };

    // Whether AIDOTNET_OP_FILTER selects the row; a [kernel] label matches the same filter as PyTorch's row of that name.
    private static bool Selected(string op, string shape)
        => Filter is not { Length: > 0 }
           || $"{op.Replace("[kernel]", "")} | {shape}".IndexOf(Filter, StringComparison.Ordinal) >= 0;

    private void Row(string op, string shape, int threads, Action run)
    {
        if (!Selected(op, shape)) return;
        // Warm up for at least 5 calls and 250 ms: tiered compilation promotes a method after 30 calls plus a 100 ms
        // delay, so five calls timed the small operations' kernels unoptimized (a 28x28 depthwise forward read 1.5 ms
        // against 0.18 fully compiled). A training run reaches the optimized code within its first steps.
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 5 || sw.ElapsedMilliseconds < 250; i++) run();
        var samples = new List<double>(Samples);
        for (int i = 0; i < Samples; i++)
        {
            sw.Restart();
            run();
            samples.Add(sw.Elapsed.TotalMilliseconds);
        }
        samples.Sort();
        _output.WriteLine($"OPBREAKDOWN tensors | {op} | {shape} | t{threads} | {samples[samples.Count / 2]:F4}");
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        var data = new float[shape.Aggregate(1, (a, d) => a * d)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }
}
#endif
