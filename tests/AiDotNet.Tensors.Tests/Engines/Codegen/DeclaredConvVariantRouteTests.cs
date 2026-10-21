// Copyright (c) AiDotNet. All rights reserved.
// Depthwise and transposed convolutions on declared programs (#890 phase 4b): each of the six operators as a grouped
// or adjoint reading of the dense convolution's spec, under every schedule its slot registers, and each public entry
// point end to end through its tuned slot (whose reference is the hand-written kernel). The bar is a direct fp64
// convolution written out here, independent of the spec, so the operand roles of the transposed passes -- a forward
// that is a data gradient, a weight gradient whose operands trade places -- are checked as well as the arithmetic.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class DeclaredConvVariantRouteTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA driver was detected, but the CUDA backend failed to initialize.");
        }
        return backend;
    }

    // Depthwise: C channels, multiplier M (output channel c*M + m). Transposed: Ci -> Co channels, output padding Op.
    private readonly record struct Case(int N, int Ci, int Co, int H, int W, int Kernel, int Stride, int Pad,
        int OutputPad, int Multiplier)
    {
        public bool Depthwise => Co == 0;
        public int OutC => Depthwise ? Ci * Multiplier : Co;
        public int OutH => Depthwise ? (H + 2 * Pad - Kernel) / Stride + 1 : (H - 1) * Stride - 2 * Pad + Kernel + OutputPad;
        public int OutW => Depthwise ? (W + 2 * Pad - Kernel) / Stride + 1 : (W - 1) * Stride - 2 * Pad + Kernel + OutputPad;
        public int InputCount => N * Ci * H * W;
        public int OutputCount => N * OutC * OutH * OutW;
        public int WeightCount => Ci * (Depthwise ? Multiplier : Co) * Kernel * Kernel;
    }

    private static Case Depthwise(int which) => which switch
    {
        0 => new Case(2, 6, 0, 9, 8, 3, 2, 1, 0, 1),     // stride 2, ragged
        1 => new Case(2, 4, 0, 10, 10, 3, 1, 1, 0, 2),   // channel multiplier 2
        _ => new Case(1, 8, 0, 16, 16, 3, 1, 1, 0, 1),   // a weight-gradient reduction a split divides
    };

    private static Case Transposed(int which) => which switch
    {
        0 => new Case(2, 6, 4, 5, 4, 3, 2, 1, 1, 0),     // stride 2, output padding 1
        _ => new Case(1, 8, 8, 8, 8, 3, 1, 1, 0, 0),     // stride 1; the weight reduction a split divides
    };

    private static float[] Fill(int count, int salt)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((((i * 37 + salt * 101) % 97) - 48) / 64.0);
        return values;
    }

    // Visits every (input element, output element, weight element) product of the operator's forward pass.
    private static void ForEachTap(Case g, Action<int, int, int> visit)
    {
        if (g.Depthwise)
        {
            for (int n = 0; n < g.N; n++)
            for (int c = 0; c < g.Ci; c++)
            for (int m = 0; m < g.Multiplier; m++)
            for (int oh = 0; oh < g.OutH; oh++)
            for (int ow = 0; ow < g.OutW; ow++)
            for (int kh = 0; kh < g.Kernel; kh++)
            for (int kw = 0; kw < g.Kernel; kw++)
            {
                int ih = oh * g.Stride - g.Pad + kh, iw = ow * g.Stride - g.Pad + kw;
                if (ih < 0 || ih >= g.H || iw < 0 || iw >= g.W) continue;
                int oc = c * g.Multiplier + m;
                visit(((n * g.Ci + c) * g.H + ih) * g.W + iw, ((n * g.OutC + oc) * g.OutH + oh) * g.OutW + ow,
                    (oc * g.Kernel + kh) * g.Kernel + kw);
            }
            return;
        }
        // y[n, co, ih*s - p + kh, iw*s - p + kw] += x[n, ci, ih, iw] * w[ci, co, kh, kw]
        for (int n = 0; n < g.N; n++)
        for (int ci = 0; ci < g.Ci; ci++)
        for (int ih = 0; ih < g.H; ih++)
        for (int iw = 0; iw < g.W; iw++)
        for (int co = 0; co < g.Co; co++)
        for (int kh = 0; kh < g.Kernel; kh++)
        for (int kw = 0; kw < g.Kernel; kw++)
        {
            int oh = ih * g.Stride - g.Pad + kh, ow = iw * g.Stride - g.Pad + kw;
            if (oh < 0 || oh >= g.OutH || ow < 0 || ow >= g.OutW) continue;
            visit(((n * g.Ci + ci) * g.H + ih) * g.W + iw, ((n * g.Co + co) * g.OutH + oh) * g.OutW + ow,
                ((ci * g.Co + co) * g.Kernel + kh) * g.Kernel + kw);
        }
    }

    private static bool IsForward(DeclaredConvVariant v) =>
        v is DeclaredConvVariant.DepthwiseForward or DeclaredConvVariant.TransposeForward;

    private static bool IsWeights(DeclaredConvVariant v) =>
        v is DeclaredConvVariant.DepthwiseBackwardWeights or DeclaredConvVariant.TransposeBackwardWeights;

    // The operator's operands in its entry point's order (A, B) and the fp64 result it must produce.
    private static (float[] A, float[] B, double[] Expected) Problem(DeclaredConvVariant v, Case g)
    {
        float[] x = Fill(g.InputCount, 1), w = Fill(g.WeightCount, 2), dy = Fill(g.OutputCount, 4);
        if (IsForward(v))
        {
            var y = new double[g.OutputCount];
            ForEachTap(g, (xi, yi, wi) => y[yi] += (double)x[xi] * w[wi]);
            return (x, w, y);
        }
        if (IsWeights(v))
        {
            var dw = new double[g.WeightCount];
            ForEachTap(g, (xi, yi, wi) => dw[wi] += (double)x[xi] * dy[yi]);
            // Depthwise takes (gradOutput, input); transposed takes (input, gradOutput).
            return v == DeclaredConvVariant.DepthwiseBackwardWeights ? (dy, x, dw) : (x, dy, dw);
        }
        var dx = new double[g.InputCount];
        ForEachTap(g, (xi, yi, wi) => dx[xi] += (double)dy[yi] * w[wi]);
        return (dy, w, dx);
    }

    private static CudaConvVariantArgs Args(DeclaredConvVariant v, Case g, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o) =>
        new(v, a, b, o, g.N, g.Ci, g.H, g.W, g.OutC, g.OutH, g.OutW, g.Kernel, g.Kernel, g.Stride, g.Stride, g.Pad,
            g.Pad, g.OutputPad, g.OutputPad, g.Depthwise ? g.Multiplier : 1);

    private static void AssertClose(double[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(error <= 1e-5, $"{label}: element {i} expected {expected[i]}, got {actual[i]}");
        }
    }

    public static IEnumerable<object[]> ScheduleCases()
    {
        foreach (DeclaredConvVariant v in Enum.GetValues(typeof(DeclaredConvVariant)))
        {
            bool depthwise = v <= DeclaredConvVariant.DepthwiseBackwardWeights;
            int shapes = depthwise ? 3 : 2;
            for (int shape = 0; shape < shapes; shape++)
                for (int schedule = 0; schedule < CudaBackend.DeclaredConvVariantSchedules(v).Count; schedule++)
                    yield return new object[] { (int)v, shape, schedule };
        }
    }

    /// <summary>
    /// Every schedule the operator's slot registers, on every shape it admits, against the direct convolution. The
    /// output starts filled, so an element the program fails to write cannot pass as a zero.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(ScheduleCases))]
    public void EveryAdmittedScheduleMatchesADirectConvolution(int variantIndex, int shape, int schedule)
    {
        var variant = (DeclaredConvVariant)variantIndex;
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        var g = variant <= DeclaredConvVariant.DepthwiseBackwardWeights ? Depthwise(shape) : Transposed(shape);
        var sc = CudaBackend.DeclaredConvVariantSchedules(variant)[schedule];
        var (a, b, expected) = Problem(variant, g);
        using var ba = backend.AllocateBuffer(a);
        using var bb = backend.AllocateBuffer(b);
        using var bo = backend.AllocateBuffer(Fill(expected.Length, 5));
        var args = Args(variant, g, ba, bb, bo);
        Skip.IfNot(backend.DeclaredConvVariantAdmits(variant, args.Shape(), sc),
            $"{variant} {sc.Suffix} does not lower shape {shape} (its slot never offers it there).");
        long before = backend.DeclaredProgramLaunches;
        Assert.True(backend.TryDeclaredConvVariant(args, sc), $"{variant} {sc.Suffix} admitted shape {shape} but declined");
        backend.Synchronize();
        Assert.True(backend.DeclaredProgramLaunches > before);
        AssertClose(expected, backend.DownloadBuffer(bo), $"{variant} {sc.Suffix} shape {shape}");
    }

    /// <summary>
    /// The specs themselves (no GPU): each operator's derived spec, interpreted in fp64, is the direct convolution. A
    /// failure here is the spec's (the grouped catalog spec or its adjoint); one only on the device is the emitter's.
    /// </summary>
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(2, 0)]
    [InlineData(2, 2)]
    [InlineData(3, 0)]
    [InlineData(4, 0)]
    [InlineData(5, 0)]
    public void TheDerivedSpecIsTheDirectConvolution(int variantIndex, int shape)
    {
        var variant = (DeclaredConvVariant)variantIndex;
        var g = variant <= DeclaredConvVariant.DepthwiseBackwardWeights ? Depthwise(shape) : Transposed(shape);
        var (a, b, expected) = Problem(variant, g);
        var shapeClass = new CudaConvVariantArgs(variant, NoBuffer, NoBuffer, NoBuffer, g.N, g.Ci, g.H, g.W, g.OutC,
            g.OutH, g.OutW, g.Kernel, g.Kernel, g.Stride, g.Stride, g.Pad, g.Pad, g.OutputPad, g.OutputPad,
            g.Depthwise ? g.Multiplier : 1).Shape();
        var e = CudaBackend.ConvVariantEquivalent(variant, shapeClass) ?? throw new InvalidOperationException("no equivalent");
        var spec = CudaBackend.DeclaredConvSpec(e.Pass, e.Batch, e.InChannels, e.OutChannels, e.Height, e.Width, e.Kernel,
            e.Kernel, e.Stride, e.Stride, e.Pad, e.Pad, 1, 1, e.OutHeight, e.OutWidth, biasRelu: false, e.Groups)
            ?? throw new InvalidOperationException("no spec");
        // The declared program binds the entry point's (A, B) as the spec's first and second inputs.
        var inputs = new List<double[]>();
        foreach (var operand in new[] { a, b }) inputs.Add(Array.ConvertAll(operand, v => (double)v));
        double[] actual = spec.Interpret(inputs);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * Math.Max(1.0, Math.Abs(expected[i])),
                $"{variant} shape {shape}: element {i} expected {expected[i]}, interpreted {actual[i]}");
    }

    // Shape-class arguments carry no buffers; this one is never dereferenced.
    private static readonly IGpuBuffer NoBuffer = new UnusedBuffer();

    private sealed class UnusedBuffer : IGpuBuffer
    {
        public int Size => 0;
        public long SizeInBytes => 0;
        public IntPtr Handle => IntPtr.Zero;
        public void Dispose() { }
    }

    /// <summary>
    /// Convolutions as resident-program stages: a dense convolution, a depthwise one on its output, a transposed one on
    /// that, and the depthwise weight gradient, run as ONE cooperative launch, bit-identical to the same stages
    /// launched separately (and the first stage against the direct convolution).
    /// </summary>
    [SkippableFact]
    public void ConvolutionStagesRunAsOneResidentLaunchBitIdenticalToSeparateLaunches()
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.SupportsResidentProgram, "A declared resident program needs a cooperative launch.");
        const int N = 2, C = 4, K = 6, H = 8, W = 8, Co = 3;
        var dense = new DeclaredConvKey(DeclaredConvPass.Forward, N, C, K, H, W, 3, 1, 1, 1, false,
            new DeclaredConvSchedule(64, 64, 0));
        var depthwise = new Case(N, K, 0, H, W, 3, 1, 1, 0, 1);
        var transposed = new Case(N, K, Co, H, W, 3, 2, 1, 1, 0);
        float[] x = Fill(N * C * H * W, 11), w1 = Fill(K * C * 9, 12), wd = Fill(K * 9, 13), wt = Fill(K * Co * 9, 14);
        using var barrier = backend.AllocateResidentBarrier();
        var results = new float[2][][];
        for (int mode = 0; mode < 2; mode++)
        {
            using var bx = backend.AllocateBuffer(x);
            using var bw1 = backend.AllocateBuffer(w1);
            using var bwd = backend.AllocateBuffer(wd);
            using var bwt = backend.AllocateBuffer(wt);
            using var h = backend.AllocateBuffer(N * K * H * W);
            using var d = backend.AllocateBuffer(N * K * H * W);
            using var t = backend.AllocateBuffer(transposed.OutputCount);
            using var dwd = backend.AllocateBuffer(K * 9);
            var stages = new[]
            {
                backend.ConvStage(dense, bx, bw1, h),
                backend.ConvVariantStage(Args(DeclaredConvVariant.DepthwiseForward, depthwise, h, bwd, d),
                    new DeclaredConvSchedule(0, 0, 0)),
                backend.ConvVariantStage(Args(DeclaredConvVariant.TransposeForward, transposed, d, bwt, t),
                    new DeclaredConvSchedule(64, 64, 0)),
                backend.ConvVariantStage(Args(DeclaredConvVariant.DepthwiseBackwardWeights, depthwise, d, h, dwd),
                    new DeclaredConvSchedule(0, 0, 0)),
            };
            long before = backend.DeclaredProgramLaunches;
            if (mode == 0) backend.RunDeclaredStagesSeparately(stages);
            else Assert.True(backend.RunDeclaredResidentProgram(stages, barrier) >= 1, backend.LastResidentDecline);
            backend.Synchronize();
            Assert.Equal(mode == 0 ? stages.Length : 1, backend.DeclaredProgramLaunches - before);
            results[mode] = new[] { backend.DownloadBuffer(h), backend.DownloadBuffer(d), backend.DownloadBuffer(t),
                backend.DownloadBuffer(dwd) };
        }

        var expected = new double[N * K * H * W];
        for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
        for (int oh = 0; oh < H; oh++)
        for (int ow = 0; ow < W; ow++)
        for (int c = 0; c < C; c++)
        for (int kh = 0; kh < 3; kh++)
        for (int kw = 0; kw < 3; kw++)
        {
            int ih = oh - 1 + kh, iw = ow - 1 + kw;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
            expected[((n * K + k) * H + oh) * W + ow] +=
                (double)x[((n * C + c) * H + ih) * W + iw] * w1[((k * C + c) * 3 + kh) * 3 + kw];
        }
        AssertClose(expected, results[0][0], "dense stage");
        for (int o = 0; o < results[0].Length; o++)
            for (int i = 0; i < results[0][o].Length; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(results[0][o][i]) == TestHelpers.MathCompat.SingleToInt32Bits(results[1][o][i]),
                    $"buffer {o}[{i}]: separate {results[0][o][i]} resident {results[1][o][i]}");
    }

    /// <summary>Every operator has at least one declared schedule that lowers each of its test shapes.</summary>
    [SkippableFact]
    public void EveryOperatorHasADeclaredScheduleForEveryShape()
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        foreach (DeclaredConvVariant v in Enum.GetValues(typeof(DeclaredConvVariant)))
        {
            bool depthwise = v <= DeclaredConvVariant.DepthwiseBackwardWeights;
            for (int shape = 0; shape < (depthwise ? 3 : 2); shape++)
            {
                var g = depthwise ? Depthwise(shape) : Transposed(shape);
                using var dummy = backend.AllocateBuffer(1);
                var shapeClass = Args(v, g, dummy, dummy, dummy).Shape();
                bool any = false;
                foreach (var sc in CudaBackend.DeclaredConvVariantSchedules(v))
                    any |= backend.DeclaredConvVariantAdmits(v, shapeClass, sc);
                Assert.True(any, $"{v} has no declared schedule for shape {shape}");
            }
        }
    }

    /// <summary>
    /// The public entry points through their tuned slots: whichever candidate the slot serves (the hand-written
    /// reference or a declared schedule), the result is the direct convolution's. This is also what checks the
    /// references' wiring: the slot gates every declared candidate against them.
    /// </summary>
    [SkippableTheory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 0)]
    [InlineData(4, 0)]
    [InlineData(5, 1)]
    public void TheEntryPointMatchesADirectConvolution(int variantIndex, int shape)
    {
        var variant = (DeclaredConvVariant)variantIndex;
        using var backend = OpenBackendOrSkip();
        var g = variant <= DeclaredConvVariant.DepthwiseBackwardWeights ? Depthwise(shape) : Transposed(shape);
        var (a, b, expected) = Problem(variant, g);
        using var ba = backend.AllocateBuffer(a);
        using var bb = backend.AllocateBuffer(b);
        using var bo = backend.AllocateBuffer(Fill(expected.Length, 5));
        // Several calls: the slot measures its candidates over the first ones, then serves its decision.
        for (int call = 0; call < 6; call++)
        {
            switch (variant)
            {
                case DeclaredConvVariant.DepthwiseForward:
                    backend.DepthwiseConv2D(ba, bb, bo, g.N, g.Ci, g.H, g.W, g.OutH, g.OutW, g.Kernel, g.Kernel,
                        g.Stride, g.Stride, g.Pad, g.Pad);
                    break;
                case DeclaredConvVariant.DepthwiseBackwardData:
                    backend.DepthwiseConv2DBackwardInput(ba, bb, bo, g.N, g.Ci, g.H, g.W, g.Multiplier, g.OutH, g.OutW,
                        g.Kernel, g.Kernel, g.Stride, g.Stride, g.Pad, g.Pad);
                    break;
                case DeclaredConvVariant.DepthwiseBackwardWeights:
                    backend.DepthwiseConv2DBackwardKernel(ba, bb, bo, g.N, g.Ci, g.H, g.W, g.Multiplier, g.OutH, g.OutW,
                        g.Kernel, g.Kernel, g.Stride, g.Stride, g.Pad, g.Pad);
                    break;
                case DeclaredConvVariant.TransposeForward:
                    backend.ConvTranspose2D(ba, bb, bo, g.N, g.Ci, g.H, g.W, g.Co, g.OutH, g.OutW, g.Kernel, g.Kernel,
                        g.Stride, g.Stride, g.Pad, g.Pad, g.OutputPad, g.OutputPad);
                    break;
                case DeclaredConvVariant.TransposeBackwardData:
                    backend.ConvTranspose2DBackwardInput(ba, bb, bo, g.N, g.Ci, g.H, g.W, g.Co, g.OutH, g.OutW,
                        g.Kernel, g.Kernel, g.Stride, g.Stride, g.Pad, g.Pad, g.OutputPad, g.OutputPad);
                    break;
                default:
                    backend.ConvTranspose2DBackwardKernel(ba, bb, bo, g.N, g.Ci, g.H, g.W, g.Co, g.OutH, g.OutW,
                        g.Kernel, g.Kernel, g.Stride, g.Stride, g.Pad, g.Pad, g.OutputPad, g.OutputPad);
                    break;
            }
            backend.Synchronize();
            AssertClose(expected, backend.DownloadBuffer(bo), $"{variant} entry point, call {call}");
        }
    }
}
