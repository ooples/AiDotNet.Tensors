using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Guards the small-batch serial-gate fix in the float MaxPool2DInto kernels
/// (2×2 / 3×3 NoPad + Generic). The gate now requires <c>bc*hw ≥ grain</c> before
/// taking the parallel (closure + delegate) path, so small-batch pools (e.g. bs1:
/// 16ch × 14² = 12 544 &lt; grain) take the fast direct-serial inline path instead of
/// paying ~25 µs of dispatch overhead to run serial anyway. The gate change only
/// alters serial-vs-parallel routing, so output must stay BIT-IDENTICAL to the
/// allocating <c>MaxPool2D</c> reference across every batch on both sides of the gate.
/// </summary>
public class MaxPool2DIntoSerialGateTests
{
    private readonly ITestOutputHelper _output;
    public MaxPool2DIntoSerialGateTests(ITestOutputHelper output) => _output = output;

    public static TheoryData<int, int, int, int, int, int> Cases => new()
    {
        // batch, channels, H, W, poolSize, padding  (stride = poolSize)
        { 1,  16, 28, 28, 2, 0 },    // bs1 below-grain → must hit fast serial inline (the fix)
        { 1,  32, 14, 14, 2, 0 },    // bs1 pool2
        { 4,  16, 28, 28, 2, 0 },    // bc=64 ≥ MaxDOP but below grain → fast serial
        { 128,16, 28, 28, 2, 0 },    // above grain → parallel path
        { 1,  16, 28, 28, 3, 0 },    // 3×3 NoPad bs1
        { 64, 16, 28, 28, 3, 0 },    // 3×3 NoPad above grain
        { 1,  16, 28, 28, 3, 1 },    // 3×3 padded (border path)
        { 1,  16, 28, 28, 4, 1 },    // generic (4×4, padded)
        { 128, 8, 16, 16, 2, 0 },    // parallel 2×2
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void MaxPool2DInto_MatchesAllocatingReference_AcrossGateBoundary(
        int batch, int channels, int h, int w, int poolSize, int padding)
    {
        var e = new CpuEngine();
        int stride = poolSize;
        var input = Rand(new[] { batch, channels, h, w }, 1234);

        // Reference: the allocating MaxPool2D (independent code path).
        var reference = e.MaxPool2D(input, poolSize, stride, padding);
        var into = new Tensor<float>(reference.Shape.ToArray());
        e.MaxPool2DInto(into, input, poolSize, stride, padding);

        var rSpan = reference.AsSpan();
        var iSpan = into.AsSpan();
        Assert.Equal(rSpan.Length, iSpan.Length);
        for (int idx = 0; idx < rSpan.Length; idx++)
            Assert.Equal(rSpan[idx], iSpan[idx]);   // bit-exact: same kernel math, only routing differs
    }

    /// <summary>Env-gated latency check: the bs1 pool should no longer pay the parallel
    /// dispatch overhead. (AIDOTNET_RUN_JIT_PERF=1)</summary>
    [Fact]
    public void MaxPool2DInto_Bs1_LatencyProbe()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        var e = new CpuEngine();
        // The two bs1 pools in the parity CNN stem (conv1→pool1, conv2→pool2).
        Probe(e, "pool1 bs1 16ch×28²→14²", new[] { 1, 16, 28, 28 }, new[] { 1, 16, 14, 14 });
        Probe(e, "pool2 bs1 32ch×14²→7²", new[] { 1, 32, 14, 14 }, new[] { 1, 32, 7, 7 });
    }

    private void Probe(CpuEngine e, string label, int[] inShape, int[] outShape)
    {
        var input = Rand(inShape, 7);
        var into = new Tensor<float>(outShape);
        for (int i = 0; i < 50; i++) e.MaxPool2DInto(into, input, 2, 2);
        double best = double.MaxValue;
        for (int r = 0; r < 400; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            e.MaxPool2DInto(into, input, 2, 2);
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        _output.WriteLine($"MaxPool2DInto {label}: {best*1000:F2} us (best-of-400)");
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed); var t = new Tensor<float>(shape); var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
