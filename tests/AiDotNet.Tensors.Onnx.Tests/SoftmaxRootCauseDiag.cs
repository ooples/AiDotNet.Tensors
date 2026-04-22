using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Drills the BERT Softmax bottleneck. StableOpPerfHarness measured Softmax
/// at ~2200 µs on [1, 12, 256, 256] axis=-1 on an AMD Zen 2 (no VML, no
/// fast-gather, no AVX-512). Theoretical floor:
///   - Memory bandwidth: 3 passes × 786 KB / 50 GB/s aggregate = ~190 µs
///   - Exp throughput: 786K × 15 cycles / (16 cores × 4 GHz) = ~184 µs
///   - Sum: ~370 µs ideal
/// We are 6× above that. This harness localises where the time goes:
///
/// <list type="number">
///   <item>Raw single-row kernel × 3072 rows (single thread)</item>
///   <item>Raw kernel via PersistentParallelExecutor (matches engine path)</item>
///   <item>Direct CpuEngine.Softmax (full public API: GraphMode +
///   AutoTracer + AutoTensorCache.RentOrAllocate + Pin())</item>
///   <item>SoftmaxInto write-through (skips alloc + tracer)</item>
/// </list>
///
/// <para>Gated behind <c>AIDOTNET_RUN_PERF_HARNESS=1</c>.</para>
/// </summary>
public class SoftmaxRootCauseDiag
{
    private readonly ITestOutputHelper _output;
    public SoftmaxRootCauseDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 30;
    private const int Iters = 200;

    [SkippableFact]
    public void LocaliseSoftmaxBottleneck()
    {
        Skip.IfNot(
            System.Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        // BERT shape: [1, 12, 256, 256] axis=-1
        const int outerSize = 1 * 12 * 256;  // 3072 rows
        const int axisSize = 256;

        var input = Rand(0xF01, outerSize * axisSize);

        _output.WriteLine($"=== BERT Softmax [1,12,256,256] axis=-1 ({outerSize} rows × {axisSize} cols) ===");

        double tDirect = TimeFullEngineSoftmax(input, outerSize, axisSize);
        _output.WriteLine($"  [1] CpuEngine.Softmax (full public API):       {tDirect:F1} µs/call");

        double tInto = TimeEngineSoftmaxInto(input, outerSize, axisSize);
        _output.WriteLine($"  [2] CpuEngine.SoftmaxInto (write-through):     {tInto:F1} µs/call");

        double tSerial = TimeSerialKernel(input, outerSize, axisSize);
        _output.WriteLine($"  [3] SimdKernels.Softmax serial (1 core):       {tSerial:F1} µs/call");
        _output.WriteLine($"     Engine parallel scaling vs serial: {tSerial / tInto:F2}× on {System.Environment.ProcessorCount} cores");
        _output.WriteLine($"     Engine overhead vs SoftmaxInto: {tDirect - tInto:F0} µs");
    }

    private static double TimeFullEngineSoftmax(float[] input, int outer, int axis)
    {
        var engine = new CpuEngine();
        var t = new Tensor<float>(new[] { 1, 12, 256, 256 });
        input.AsSpan().CopyTo(t.AsWritableSpan());
        for (int i = 0; i < Warmup; i++) _ = engine.Softmax(t, -1);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) _ = engine.Softmax(t, -1);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static double TimeEngineSoftmaxInto(float[] input, int outer, int axis)
    {
        var engine = new CpuEngine();
        var inT = new Tensor<float>(new[] { 1, 12, 256, 256 });
        var outT = new Tensor<float>(new[] { 1, 12, 256, 256 });
        input.AsSpan().CopyTo(inT.AsWritableSpan());
        for (int i = 0; i < Warmup; i++) engine.SoftmaxInto(outT, inT, -1);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) engine.SoftmaxInto(outT, inT, -1);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    // Single-thread reference: SimdKernels.Softmax processes rows
    // sequentially on the calling thread. The engine path adds
    // PersistentParallelExecutor + per-row chunking; comparing this to
    // SoftmaxInto isolates the parallel scaling efficiency.
    private static double TimeSerialKernel(float[] input, int outer, int axis)
    {
        var output = new float[input.Length];
        for (int i = 0; i < Warmup; i++)
            AiDotNet.Tensors.Engines.Simd.SimdKernels.Softmax(input, output, outer, axis);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            AiDotNet.Tensors.Engines.Simd.SimdKernels.Softmax(input, output, outer, axis);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new System.Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 6.0 - 3.0);
        return a;
    }
}
