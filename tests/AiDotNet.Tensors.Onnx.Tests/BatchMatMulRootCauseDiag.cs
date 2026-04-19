using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Phase 2D root-cause harness for the remaining BERT attention
/// bottleneck after the stage4 Conv fix (0ae6811): 3D×3D batched
/// matmul <c>[12,256,64]×[12,64,256]</c> (attn scores) runs at 11.3×
/// ORT; <c>[12,256,256]×[12,256,64]</c> (attn×V) at 8.8×. Both
/// ~100 M FLOPs per call.
///
/// <para>Tiers timed:</para>
/// <list type="number">
///   <item>Direct engine.TensorMatMul (3D×3D dispatch) — ours today.</item>
///   <item>Explicit per-slice SimdGemm.Sgemm in Parallel.For (external
///   parallel, lets SGEMM also parallelise internally — the nested-
///   parallel case).</item>
///   <item>Explicit per-slice SimdGemm.SgemmSequential in Parallel.For
///   (external parallel, SGEMM stays sequential — the correct pattern
///   per SgemmSequential's docstring).</item>
///   <item>Serial per-slice SGEMM (no outer parallel) — baseline.</item>
/// </list>
///
/// <para>The ratio [1]/[3] measures how much the current code pays for
/// nested parallelism. If [3] is much faster than [1], the fix is to
/// swap <c>MatrixMultiplyHelper.TryGemm</c> for
/// <c>SimdGemm.SgemmSequential</c> inside TensorMatMulFullBatched's
/// parallel loop.</para>
/// </summary>
public class BatchMatMulRootCauseDiag
{
    private readonly ITestOutputHelper _output;
    public BatchMatMulRootCauseDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 3;
    private const int Iters  = 20;

    [SkippableFact]
    public void LocaliseBertAttentionBottleneck()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        _output.WriteLine($"Avx512Sgemm.CanUse = {Avx512Sgemm.CanUse}");
        _output.WriteLine($"CPU cores = {Environment.ProcessorCount}");
        _output.WriteLine("");

        // BERT-base attention shapes (seq=256, head_dim=64, heads=12).
        (int batch, int m, int k, int n, string label)[] cases =
        {
            (12, 256, 64, 256,   "attn scores: [12,256,64]×[12,64,256] → [12,256,256]"),
            (12, 256, 256, 64,   "attn × V:    [12,256,256]×[12,256,64] → [12,256,64]"),
        };

        foreach (var cs in cases)
        {
            double flops = 2.0 * cs.batch * cs.m * cs.k * cs.n;
            _output.WriteLine($"=== {cs.label}  ({flops/1e6:F1} M FLOPs) ===");

            double t1 = TimeDirectEngineBatchMatMul(cs.batch, cs.m, cs.k, cs.n);
            _output.WriteLine($"  [1] Direct engine.TensorMatMul:                 {t1 * 1000:F1} µs  ({flops / t1 / 1e9:F1} GFLOP/s)");

            double t2 = TimeParallelForWithParallelSgemm(cs.batch, cs.m, cs.k, cs.n);
            _output.WriteLine($"  [2] Parallel.For + SimdGemm.Sgemm (nested):     {t2 * 1000:F1} µs  ({flops / t2 / 1e9:F1} GFLOP/s)");

            double t3 = TimeParallelForWithSequentialSgemm(cs.batch, cs.m, cs.k, cs.n);
            _output.WriteLine($"  [3] Parallel.For + SgemmSequential (flat):      {t3 * 1000:F1} µs  ({flops / t3 / 1e9:F1} GFLOP/s)");

            double t4 = TimeSerialPerSliceSgemm(cs.batch, cs.m, cs.k, cs.n);
            _output.WriteLine($"  [4] Serial per-slice SGEMM:                     {t4 * 1000:F1} µs  ({flops / t4 / 1e9:F1} GFLOP/s)");

            _output.WriteLine($"  Ratio [1]/[3] = {t1 / t3:F1}×  (current vs correct parallelism)");
            _output.WriteLine("");
        }
    }

    // ─── timed paths ────────────────────────────────────────────────────────

    private static double TimeDirectEngineBatchMatMul(int batch, int m, int k, int n)
    {
        var engine = new CpuEngine();
        var a = new Tensor<float>(new[] { batch, m, k });
        var b = new Tensor<float>(new[] { batch, k, n });
        Rand(0xBB01, batch * m * k).AsSpan().CopyTo(a.AsWritableSpan());
        Rand(0xBB02, batch * k * n).AsSpan().CopyTo(b.AsWritableSpan());

        for (int i = 0; i < Warmup; i++) _ = engine.TensorMatMul(a, b);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) _ = engine.TensorMatMul(a, b);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeParallelForWithParallelSgemm(int batch, int m, int k, int n)
    {
        var a = Rand(0xBB03, batch * m * k);
        var b = Rand(0xBB04, batch * k * n);
        var c = new float[batch * m * n];
        int sliceA = m * k, sliceB = k * n, sliceC = m * n;

        for (int i = 0; i < Warmup; i++)
            System.Threading.Tasks.Parallel.For(0, batch, bi =>
                SimdGemm.Sgemm(a.AsSpan(bi * sliceA, sliceA),
                               b.AsSpan(bi * sliceB, sliceB),
                               c.AsSpan(bi * sliceC, sliceC), m, k, n));

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            System.Threading.Tasks.Parallel.For(0, batch, bi =>
                SimdGemm.Sgemm(a.AsSpan(bi * sliceA, sliceA),
                               b.AsSpan(bi * sliceB, sliceB),
                               c.AsSpan(bi * sliceC, sliceC), m, k, n));
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeParallelForWithSequentialSgemm(int batch, int m, int k, int n)
    {
        var a = Rand(0xBB05, batch * m * k);
        var b = Rand(0xBB06, batch * k * n);
        var c = new float[batch * m * n];
        int sliceA = m * k, sliceB = k * n, sliceC = m * n;

        for (int i = 0; i < Warmup; i++)
            System.Threading.Tasks.Parallel.For(0, batch, bi =>
                SimdGemm.SgemmSequential(a.AsSpan(bi * sliceA, sliceA),
                                         b.AsSpan(bi * sliceB, sliceB),
                                         c.AsSpan(bi * sliceC, sliceC), m, k, n));

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            System.Threading.Tasks.Parallel.For(0, batch, bi =>
                SimdGemm.SgemmSequential(a.AsSpan(bi * sliceA, sliceA),
                                         b.AsSpan(bi * sliceB, sliceB),
                                         c.AsSpan(bi * sliceC, sliceC), m, k, n));
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeSerialPerSliceSgemm(int batch, int m, int k, int n)
    {
        var a = Rand(0xBB07, batch * m * k);
        var b = Rand(0xBB08, batch * k * n);
        var c = new float[batch * m * n];
        int sliceA = m * k, sliceB = k * n, sliceC = m * n;

        for (int i = 0; i < Warmup; i++)
            for (int bi = 0; bi < batch; bi++)
                SimdGemm.Sgemm(a.AsSpan(bi * sliceA, sliceA),
                               b.AsSpan(bi * sliceB, sliceB),
                               c.AsSpan(bi * sliceC, sliceC), m, k, n);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            for (int bi = 0; bi < batch; bi++)
                SimdGemm.Sgemm(a.AsSpan(bi * sliceA, sliceA),
                               b.AsSpan(bi * sliceB, sliceB),
                               c.AsSpan(bi * sliceC, sliceC), m, k, n);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
