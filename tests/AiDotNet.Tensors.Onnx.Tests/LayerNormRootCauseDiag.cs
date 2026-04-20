using System.Diagnostics;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Drill-down: the LayerNorm SIMD rewrite only moved us from 653 → 634 µs
/// on BERT [1,256,768] (3%), nowhere near the 5-10× improvement we expected.
/// ORT is at ~70-85 µs on this shape. This harness breaks the 634 µs call
/// into tiers so we can see whether compute, allocation, or dispatch
/// overhead is dominating.
///
/// <para>Tiers timed:</para>
/// <list type="number">
///   <item><b>Raw kernel, serial, pre-allocated buffers</b> — just the three
///   SIMD passes on float[]. Pure compute cost.</item>
///   <item><b>Raw kernel, Parallel.For, pre-allocated buffers</b> — adds
///   the parallel dispatch cost.</item>
///   <item><b>CpuEngine.LayerNorm (virtual)</b> — goes through the full
///   public method: shape validation, new T[] allocations for output/mean/
///   variance, TensorAllocator.Rent, DifferentiableOps.RecordIfActive,
///   AutoTracer.RecordOp.</item>
/// </list>
///
/// <para>The delta between [1] and [3] is the allocation + plumbing cost.
/// If [1] is &lt;100 µs and [3] is 634 µs, the problem is 5× overhead from
/// allocations and wrapping, NOT compute. That means the fix is zero-
/// allocation write-through (via TensorAllocator.RentUninitialized) not
/// more SIMD tuning.</para>
///
/// <para>Gated behind <c>AIDOTNET_RUN_PERF_HARNESS=1</c>.</para>
/// </summary>
public class LayerNormRootCauseDiag
{
    private readonly ITestOutputHelper _output;
    public LayerNormRootCauseDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 10;
    private const int Iters = 100;

    [SkippableFact]
    public void LocaliseLayerNormBottleneck()
    {
        Skip.IfNot(
            System.Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        // BERT hot shape: [1, 256, 768]
        const int B = 1, S = 256, H = 768;
        int batch = B * S, feat = H;

        var input = Rand(0xAA01, batch * feat);
        var gamma = Rand(0xAA02, feat);
        var beta  = Rand(0xAA03, feat);

        _output.WriteLine($"=== BERT LayerNorm [1,256,768] (batch={batch}, feat={feat}) ===");

        double t1 = TimeRawKernelSerial(input, gamma, beta, batch, feat);
        _output.WriteLine($"  [1] Scalar kernel, serial:                      {t1:F1} µs/call");

        double t2 = TimeRawKernelParallel(input, gamma, beta, batch, feat);
        _output.WriteLine($"  [2] Scalar kernel, Parallel.For:                {t2:F1} µs/call");

#if NET5_0_OR_GREATER
        double t1s = TimeSimdKernelSerial(input, gamma, beta, batch, feat);
        _output.WriteLine($"  [3] SIMD kernel, serial:                        {t1s:F1} µs/call");

        double t2s = TimeSimdKernelParallel(input, gamma, beta, batch, feat);
        _output.WriteLine($"  [4] SIMD kernel, Parallel.For:                  {t2s:F1} µs/call");

        double t1f = TimeFusedOnePassSerial(input, gamma, beta, batch, feat);
        _output.WriteLine($"  [5] SIMD one-pass fused E[X²]-E[X]², serial:    {t1f:F1} µs/call");

        double t2f = TimeFusedOnePassParallel(input, gamma, beta, batch, feat);
        _output.WriteLine($"  [6] SIMD one-pass fused, Parallel.For:          {t2f:F1} µs/call");
#endif

        double t3 = TimeCpuEngineLayerNorm(input, gamma, beta, batch, feat);
        _output.WriteLine($"  [7] CpuEngine.LayerNorm (full public API):      {t3:F1} µs/call");

        double t8 = TimeCpuEngineLayerNormInto(input, gamma, beta, batch, feat);
        _output.WriteLine($"  [8] CpuEngine.LayerNormFloatInto (write-thru):  {t8:F1} µs/call");

#if NET5_0_OR_GREATER
        _output.WriteLine("");
        _output.WriteLine($"  Parallel.For overhead (scalar {t1:F0} → {t2:F0}): saves {t1 - t2:F0} µs");
        _output.WriteLine($"  SIMD gain at serial (scalar {t1:F0} → SIMD {t1s:F0}): saves {t1 - t1s:F0} µs");
        _output.WriteLine($"  One-pass gain at parallel (3-pass {t2s:F0} → 1-pass {t2f:F0}): saves {t2s - t2f:F0} µs");
        _output.WriteLine($"  Engine overhead (kernel {t2s:F0} → engine {t3:F0}): {t3 - t2s:+F0;-F0} µs");
#endif
    }

    // ─── timed paths ────────────────────────────────────────────────────────

    private static double TimeRawKernelSerial(
        float[] input, float[] gamma, float[] beta, int batch, int fs)
    {
        var output = new float[batch * fs];
        var mean = new float[batch];
        var variance = new float[batch];
        const float eps = 1e-5f;

        for (int i = 0; i < Warmup; i++)
            for (int b = 0; b < batch; b++)
                RunRowScalarSafe(input, gamma, beta, output, mean, variance, b, fs, eps);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            for (int b = 0; b < batch; b++)
                RunRowScalarSafe(input, gamma, beta, output, mean, variance, b, fs, eps);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static double TimeRawKernelParallel(
        float[] input, float[] gamma, float[] beta, int batch, int fs)
    {
        var output = new float[batch * fs];
        var mean = new float[batch];
        var variance = new float[batch];
        const float eps = 1e-5f;

        for (int i = 0; i < Warmup; i++)
            System.Threading.Tasks.Parallel.For(0, batch, b =>
                RunRowScalarSafe(input, gamma, beta, output, mean, variance, b, fs, eps));

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            System.Threading.Tasks.Parallel.For(0, batch, b =>
                RunRowScalarSafe(input, gamma, beta, output, mean, variance, b, fs, eps));
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

#if NET5_0_OR_GREATER
    private static double TimeSimdKernelSerial(
        float[] input, float[] gamma, float[] beta, int batch, int fs)
    {
        var output = new float[batch * fs];
        var mean = new float[batch];
        var variance = new float[batch];
        const float eps = 1e-5f;

        for (int i = 0; i < Warmup; i++)
            for (int b = 0; b < batch; b++)
                RunRowSimd3Pass(input, gamma, beta, output, mean, variance, b, fs, eps);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            for (int b = 0; b < batch; b++)
                RunRowSimd3Pass(input, gamma, beta, output, mean, variance, b, fs, eps);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static double TimeSimdKernelParallel(
        float[] input, float[] gamma, float[] beta, int batch, int fs)
    {
        var output = new float[batch * fs];
        var mean = new float[batch];
        var variance = new float[batch];
        const float eps = 1e-5f;

        for (int i = 0; i < Warmup; i++)
            System.Threading.Tasks.Parallel.For(0, batch, b =>
                RunRowSimd3Pass(input, gamma, beta, output, mean, variance, b, fs, eps));

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            System.Threading.Tasks.Parallel.For(0, batch, b =>
                RunRowSimd3Pass(input, gamma, beta, output, mean, variance, b, fs, eps));
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static double TimeFusedOnePassSerial(
        float[] input, float[] gamma, float[] beta, int batch, int fs)
    {
        var output = new float[batch * fs];
        var mean = new float[batch];
        var variance = new float[batch];
        const float eps = 1e-5f;

        for (int i = 0; i < Warmup; i++)
            for (int b = 0; b < batch; b++)
                RunRowSimdOnePass(input, gamma, beta, output, mean, variance, b, fs, eps);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            for (int b = 0; b < batch; b++)
                RunRowSimdOnePass(input, gamma, beta, output, mean, variance, b, fs, eps);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static double TimeFusedOnePassParallel(
        float[] input, float[] gamma, float[] beta, int batch, int fs)
    {
        var output = new float[batch * fs];
        var mean = new float[batch];
        var variance = new float[batch];
        const float eps = 1e-5f;

        for (int i = 0; i < Warmup; i++)
            System.Threading.Tasks.Parallel.For(0, batch, b =>
                RunRowSimdOnePass(input, gamma, beta, output, mean, variance, b, fs, eps));

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            System.Threading.Tasks.Parallel.For(0, batch, b =>
                RunRowSimdOnePass(input, gamma, beta, output, mean, variance, b, fs, eps));
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RunRowSimd3Pass(
        float[] input, float[] gamma, float[] beta,
        float[] output, float[] mean, float[] variance,
        int b, int fs, float eps)
    {
        int off = b * fs;

        // Pass 1: SIMD sum
        var vSum = Vector256<float>.Zero;
        int f = 0;
        for (; f + 8 <= fs; f += 8)
        {
            var v = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref input[off + f]));
            vSum = Avx.Add(vSum, v);
        }
        float sum = SimdKernels.HorizontalSum(vSum);
        for (; f < fs; f++) sum += input[off + f];
        float m = sum / fs;
        mean[b] = m;

        // Pass 2: SIMD sum-of-squares with FMA
        var vSumSq = Vector256<float>.Zero;
        var vM = Vector256.Create(m);
        f = 0;
        for (; f + 8 <= fs; f += 8)
        {
            var v = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref input[off + f]));
            var d = Avx.Subtract(v, vM);
            vSumSq = Fma.MultiplyAdd(d, d, vSumSq);
        }
        float sumSq = SimdKernels.HorizontalSum(vSumSq);
        for (; f < fs; f++)
        {
            float d = input[off + f] - m;
            sumSq += d * d;
        }
        float v2 = sumSq / fs;
        variance[b] = v2;

        // Pass 3: SIMD transform
        float invStd = 1f / System.MathF.Sqrt(v2 + eps);
        var vInvStd = Vector256.Create(invStd);
        var vMNeg = Vector256.Create(m);
        f = 0;
        for (; f + 8 <= fs; f += 8)
        {
            var v = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref input[off + f]));
            var vG = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref gamma[f]));
            var vB = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref beta[f]));
            var d = Avx.Subtract(v, vMNeg);
            var scaled = Avx.Multiply(d, vInvStd);
            var final = Fma.MultiplyAdd(scaled, vG, vB);
            Unsafe.WriteUnaligned(
                ref Unsafe.As<float, byte>(ref output[off + f]), final);
        }
        for (; f < fs; f++)
            output[off + f] = (input[off + f] - m) * invStd * gamma[f] + beta[f];
    }

    // One-pass fused: compute sum AND sum-of-squares in a single pass using
    // Var[X] = E[X²] - (E[X])². Cuts input loads from 3× to 2× (one pass for
    // mean/variance, one pass for the transform). For fs=768, input row is
    // 3 KB — fits in L1 — but one-pass still wins on prefetch latency and
    // fewer loop headers.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RunRowSimdOnePass(
        float[] input, float[] gamma, float[] beta,
        float[] output, float[] mean, float[] variance,
        int b, int fs, float eps)
    {
        int off = b * fs;

        // Pass 1: fused sum + sum-of-squares
        var vSum = Vector256<float>.Zero;
        var vSumSq = Vector256<float>.Zero;
        int f = 0;
        for (; f + 8 <= fs; f += 8)
        {
            var v = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref input[off + f]));
            vSum = Avx.Add(vSum, v);
            vSumSq = Fma.MultiplyAdd(v, v, vSumSq);
        }
        float sum = SimdKernels.HorizontalSum(vSum);
        float sumSq = SimdKernels.HorizontalSum(vSumSq);
        for (; f < fs; f++)
        {
            float x = input[off + f];
            sum += x;
            sumSq += x * x;
        }
        float m = sum / fs;
        float v2 = sumSq / fs - m * m;
        if (v2 < 0f) v2 = 0f;  // numerical safety: can go slightly negative
        mean[b] = m;
        variance[b] = v2;

        // Pass 2: SIMD transform
        float invStd = 1f / System.MathF.Sqrt(v2 + eps);
        var vInvStd = Vector256.Create(invStd);
        var vMNeg = Vector256.Create(m);
        f = 0;
        for (; f + 8 <= fs; f += 8)
        {
            var v = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref input[off + f]));
            var vG = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref gamma[f]));
            var vB = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref beta[f]));
            var d = Avx.Subtract(v, vMNeg);
            var scaled = Avx.Multiply(d, vInvStd);
            var final = Fma.MultiplyAdd(scaled, vG, vB);
            Unsafe.WriteUnaligned(
                ref Unsafe.As<float, byte>(ref output[off + f]), final);
        }
        for (; f < fs; f++)
            output[off + f] = (input[off + f] - m) * invStd * gamma[f] + beta[f];
    }
#endif

    private static double TimeCpuEngineLayerNorm(
        float[] input, float[] gamma, float[] beta, int batch, int fs)
    {
        var engine = new CpuEngine();
        // Build tensors matching BERT shape [1, 256, 768]
        var xT = new Tensor<float>(new[] { 1, batch, fs });
        var gT = new Tensor<float>(new[] { fs });
        var bT = new Tensor<float>(new[] { fs });
        input.AsSpan().CopyTo(xT.AsWritableSpan());
        gamma.AsSpan().CopyTo(gT.AsWritableSpan());
        beta .AsSpan().CopyTo(bT.AsWritableSpan());

        for (int i = 0; i < Warmup; i++)
            _ = engine.LayerNorm(xT, gT, bT, 1e-5, out _, out _);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            _ = engine.LayerNorm(xT, gT, bT, 1e-5, out _, out _);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    // Write-through variant — simulates the ONNX plan-step path where the
    // output buffer is pre-allocated by the plan and we skip the allocation
    // + CopyTo overhead of the public API.
    private static double TimeCpuEngineLayerNormInto(
        float[] input, float[] gamma, float[] beta, int batch, int fs)
    {
        var engine = new CpuEngine();
        var xT = new Tensor<float>(new[] { 1, batch, fs });
        var gT = new Tensor<float>(new[] { fs });
        var bT = new Tensor<float>(new[] { fs });
        var outT = new Tensor<float>(new[] { 1, batch, fs });  // pre-allocated output
        input.AsSpan().CopyTo(xT.AsWritableSpan());
        gamma.AsSpan().CopyTo(gT.AsWritableSpan());
        beta .AsSpan().CopyTo(bT.AsWritableSpan());

        for (int i = 0; i < Warmup; i++)
            engine.LayerNormFloatInto(xT, gT, bT, 1e-5, outT);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            engine.LayerNormFloatInto(xT, gT, bT, 1e-5, outT);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    // Simple per-row scalar layer norm — mirrors CpuEngine's scalar fallback
    // so the [1] row represents the best-case compute we could hope for with
    // no allocation or plumbing.
    private static void RunRowScalarSafe(
        float[] input, float[] gamma, float[] beta,
        float[] output, float[] mean, float[] variance,
        int b, int fs, float eps)
    {
        int off = b * fs;
        float sum = 0f;
        for (int f = 0; f < fs; f++) sum += input[off + f];
        float m = sum / fs;
        mean[b] = m;
        float sumSq = 0f;
        for (int f = 0; f < fs; f++)
        {
            float d = input[off + f] - m;
            sumSq += d * d;
        }
        float v2 = sumSq / fs;
        variance[b] = v2;
        float invStd = 1f / System.MathF.Sqrt(v2 + eps);
        for (int f = 0; f < fs; f++)
            output[off + f] = (input[off + f] - m) * invStd * gamma[f] + beta[f];
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new System.Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
