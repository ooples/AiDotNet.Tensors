using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression tests for the per-head attention GEMM oversubscription bug: the SDPA float
/// path parallelizes over B·H heads with <see cref="CpuParallelSettings.ParallelForOrSerial"/>
/// (which enters a parallel region per worker), but each head's QK^T / P·V GEMM went through
/// the BlasManaged strategies, which resolved their worker count to <c>ProcessorCount</c>
/// regardless of the region. For a large per-head GEMM that triggers multithreading, that is
/// <c>heads × ProcessorCount</c> threads on <c>ProcessorCount</c> cores — catastrophic
/// ThreadPool oversubscription that thrashes (e.g. SDPA float [1,8,1024,64] burned thousands
/// of CPU-seconds without completing).
///
/// The fix routes every BlasManaged thread-count decision through
/// <see cref="CpuParallelSettings.ResolveWorkerThreads"/>, which collapses an unset
/// <c>NumThreads</c> to 1 when already inside a parallel region — the wiring the
/// <see cref="CpuParallelSettings.IsInParallelRegion"/> doc always described.
/// </summary>
public class SdpaNestedParallelismTests
{
    private readonly ITestOutputHelper _output;
    private readonly CpuEngine _engine;

    public SdpaNestedParallelismTests(ITestOutputHelper output)
    {
        _output = output;
        _engine = new CpuEngine();
    }

    // ---- Core fix: ResolveWorkerThreads honors explicit NumThreads AND the region ----

    [Fact]
    public void ResolveWorkerThreads_TopLevel_UnsetUsesAllProcessors()
    {
        Assert.False(CpuParallelSettings.IsInParallelRegion);
        Assert.Equal(Environment.ProcessorCount, CpuParallelSettings.ResolveWorkerThreads(0));
    }

    [Fact]
    public void ResolveWorkerThreads_ExplicitCount_AlwaysRespected()
    {
        Assert.Equal(4, CpuParallelSettings.ResolveWorkerThreads(4));
        // Even inside a region an explicit positive count is honored (caller knows best).
        using (CpuParallelSettings.EnterParallelRegion())
            Assert.Equal(4, CpuParallelSettings.ResolveWorkerThreads(4));
    }

    [Fact]
    public void ResolveWorkerThreads_NegativeMeansSingleThread()
    {
        Assert.Equal(1, CpuParallelSettings.ResolveWorkerThreads(-1));
    }

    [Fact]
    public void ResolveWorkerThreads_NestedRegion_UnsetCollapsesToSingleThread()
    {
        // THE regression: an unset NumThreads inside a parallel region must resolve to 1,
        // not ProcessorCount — otherwise nested GEMMs oversubscribe.
        using (CpuParallelSettings.EnterParallelRegion())
            Assert.Equal(1, CpuParallelSettings.ResolveWorkerThreads(0));
        // Region scope restored on dispose.
        Assert.False(CpuParallelSettings.IsInParallelRegion);
        Assert.Equal(Environment.ProcessorCount, CpuParallelSettings.ResolveWorkerThreads(0));
    }

    // ---- SDPA path correctness (the fix must not change results) ----

    [Theory]
    [InlineData(2, 4, 48, 32)]   // multi-batch, multi-head, moderate seq
    [InlineData(1, 8, 64, 64)]   // single batch, more heads
    public void Sdpa_Float_MatchesNaiveReference(int b, int h, int seq, int d)
    {
        var rng = new Random(20260527);
        var q = MakeRandom4D(rng, b, h, seq, d);
        var k = MakeRandom4D(rng, b, h, seq, d);
        var v = MakeRandom4D(rng, b, h, seq, d);

        var got = _engine.ScaledDotProductAttention<float>(q, k, v, null, null, out _);
        var expected = NaiveSdpa(q, k, v, b, h, seq, d);

        var gs = got.AsSpan();
        for (int i = 0; i < gs.Length; i++)
            Assert.True(Math.Abs(gs[i] - expected[i]) <= 1e-3f * (1 + Math.Abs(expected[i])),
                $"index {i}: got {gs[i]:E5} vs expected {expected[i]:E5}");
    }

    /// <summary>
    /// The oversubscription repro: large per-head GEMM ([1024,1024,64]) across 8 heads. Gated
    /// behind AIDOTNET_RUN_JIT_PERF because it is heavy; pre-fix it thrashed for minutes, post-fix
    /// it completes in well under a second of compute. Reports best-of-3 wall time.
    /// </summary>
    [Trait("Category", "Performance")]
    [Fact]
    public void Sdpa_Float_LargeHeads_DoesNotOversubscribe()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        const int b = 1, h = 8, seq = 1024, d = 64;
        var rng = new Random(7);
        var q = MakeRandom4D(rng, b, h, seq, d);
        var k = MakeRandom4D(rng, b, h, seq, d);
        var v = MakeRandom4D(rng, b, h, seq, d);
        _ = _engine.ScaledDotProductAttention<float>(q, k, v, null, null, out _); // warm
        double best = double.MaxValue;
        for (int r = 0; r < 3; r++)
        {
            var sw = Stopwatch.StartNew();
            _ = _engine.ScaledDotProductAttention<float>(q, k, v, null, null, out _);
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        _output.WriteLine($"SDPA float [1,8,1024,64] best-of-3: {best:F1} ms");
    }

    // ---- helpers ----

    private static Tensor<float> MakeRandom4D(Random rng, int a, int b, int c, int d)
    {
        var t = new Tensor<float>(new[] { a, b, c, d });
        var sp = t.AsWritableSpan();
        for (int i = 0; i < sp.Length; i++) sp[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static float[] NaiveSdpa(Tensor<float> q, Tensor<float> k, Tensor<float> v,
        int b, int h, int seq, int d)
    {
        var qs = q.AsSpan(); var ks = k.AsSpan(); var vs = v.AsSpan();
        var outp = new float[b * h * seq * d];
        float scale = (float)(1.0 / Math.Sqrt(d));
        var scores = new double[seq];
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h; hi++)
            {
                int baseOff = ((bi * h) + hi) * seq * d;
                for (int i = 0; i < seq; i++)
                {
                    double maxv = double.NegativeInfinity;
                    for (int j = 0; j < seq; j++)
                    {
                        double s = 0;
                        for (int e = 0; e < d; e++)
                            s += (double)qs[baseOff + i * d + e] * ks[baseOff + j * d + e];
                        s *= scale;
                        scores[j] = s;
                        if (s > maxv) maxv = s;
                    }
                    double sum = 0;
                    for (int j = 0; j < seq; j++) { scores[j] = Math.Exp(scores[j] - maxv); sum += scores[j]; }
                    double inv = sum != 0 ? 1.0 / sum : 0.0;
                    for (int e = 0; e < d; e++)
                    {
                        double acc = 0;
                        for (int j = 0; j < seq; j++) acc += scores[j] * inv * vs[baseOff + j * d + e];
                        outp[baseOff + i * d + e] = (float)acc;
                    }
                }
            }
        return outp;
    }
}
