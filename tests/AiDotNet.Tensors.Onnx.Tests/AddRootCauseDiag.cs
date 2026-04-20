using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// BERT Add [1,256,768] + [1,256,768] shows at 259 µs / 4.69× ORT — that's
/// 9 GB/s for a pure memory-bound bulk add, far below what a single AMD
/// core should hit (~15-25 GB/s) and way below parallel bandwidth
/// (50+ GB/s). Prime suspect: the Add method's parallel gate
/// <c>length / 500_000</c> returns 0 for a 196608-element tensor, so the
/// add falls onto a single core. ORT presumably parallelises aggressively.
///
/// Tiers measured:
/// <list type="number">
///   <item>Raw <c>SimdKernels.VectorAddUnsafe</c> on pre-pinned pointers,
///   serial. Best-case single-core memory-bound throughput.</item>
///   <item>Same kernel, Parallel.For chunk-split. Shows whether parallel
///   memory bandwidth helps this shape.</item>
///   <item>Full <c>CpuEngine.TensorAdd</c>. Includes Pin()/allocation/
///   tracer plumbing.</item>
/// </list>
///
/// <para>Gated behind <c>AIDOTNET_RUN_PERF_HARNESS=1</c>.</para>
/// </summary>
public class AddRootCauseDiag
{
    private readonly ITestOutputHelper _output;
    public AddRootCauseDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 10;
    private const int Iters = 200;

    [SkippableFact]
    public void LocaliseAddBottleneck()
    {
        Skip.IfNot(
            System.Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        // BERT hot shape: [1, 256, 768]
        int length = 1 * 256 * 768;

        var a = Rand(0xCC01, length);
        var b = Rand(0xCC02, length);

        _output.WriteLine($"=== BERT Add [1,256,768] (length={length}, {(long)length * 4 / 1024} KB per tensor) ===");

        double t1 = TimeRawSerial(a, b, length);
        _output.WriteLine($"  [1] SimdKernels.VectorAdd, serial:                          {t1:F1} µs/call  ({3L * length * 4 / t1 / 1e3:F1} GB/s)");

        double t2 = TimeRawParallel(a, b, length, 2);
        _output.WriteLine($"  [2] VectorAdd, 2-chunk parallel:                            {t2:F1} µs/call  ({3L * length * 4 / t2 / 1e3:F1} GB/s)");

        double t3 = TimeRawParallel(a, b, length, 4);
        _output.WriteLine($"  [3] VectorAdd, 4-chunk parallel:                            {t3:F1} µs/call  ({3L * length * 4 / t3 / 1e3:F1} GB/s)");

        double t4 = TimeRawParallel(a, b, length, 8);
        _output.WriteLine($"  [4] VectorAdd, 8-chunk parallel:                            {t4:F1} µs/call  ({3L * length * 4 / t4 / 1e3:F1} GB/s)");

        double t5 = TimeEngineAdd(a, b, length);
        _output.WriteLine($"  [5] CpuEngine.TensorAdd (full public API):                  {t5:F1} µs/call  ({3L * length * 4 / t5 / 1e3:F1} GB/s)");

        _output.WriteLine("");
        _output.WriteLine($"  Best parallel speedup: {t1:F0} → {System.Math.Min(System.Math.Min(t2, t3), t4):F0} µs ({t1 / System.Math.Min(System.Math.Min(t2, t3), t4):F2}×)");
        _output.WriteLine($"  Engine overhead (best kernel → engine): {t5 - System.Math.Min(System.Math.Min(t2, t3), t4):+F0;-F0} µs");
    }

    private static double TimeRawSerial(float[] a, float[] b, int length)
    {
        var r = new float[length];
        for (int i = 0; i < Warmup; i++) SimdKernels.VectorAdd(a, b, r);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) SimdKernels.VectorAdd(a, b, r);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static double TimeRawParallel(float[] a, float[] b, int length, int chunks)
    {
        var r = new float[length];
        int chunkSize = (length + chunks - 1) / chunks;
        chunkSize = (chunkSize + 31) & ~31;

        void Run()
        {
            System.Threading.Tasks.Parallel.For(0, chunks, ch =>
            {
                int start = ch * chunkSize;
                int count = System.Math.Min(chunkSize, length - start);
                if (count > 0)
                    SimdKernels.VectorAdd(
                        new System.ReadOnlySpan<float>(a, start, count),
                        new System.ReadOnlySpan<float>(b, start, count),
                        new System.Span<float>(r, start, count));
            });
        }

        for (int i = 0; i < Warmup; i++) Run();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) Run();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static double TimeEngineAdd(float[] a, float[] b, int length)
    {
        var engine = new CpuEngine();
        var shape = new[] { 1, 256, 768 };
        var aT = new Tensor<float>(shape);
        var bT = new Tensor<float>(shape);
        a.AsSpan().CopyTo(aT.AsWritableSpan());
        b.AsSpan().CopyTo(bT.AsWritableSpan());

        for (int i = 0; i < Warmup; i++) _ = engine.TensorAdd(aT, bT);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) _ = engine.TensorAdd(aT, bT);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new System.Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
