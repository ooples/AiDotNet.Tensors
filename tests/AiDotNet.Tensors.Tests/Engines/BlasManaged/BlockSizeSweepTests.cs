using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-Q (#407) — measurement-based (Mc, Nc, Kc) autotune.
///
/// The critical contract is correctness: a candidate blocking injected via
/// <see cref="AutotuneDispatcher.BlockOverride"/> must still compute a correct
/// GEMM (blocking changes only the loop tiling / memory traffic, never the
/// math), and <see cref="BlockSizeSweep.Measure{T}"/> must return a blocking
/// clamped + aligned to the shape and leave no override leaked on the thread.
/// </summary>
public class BlockSizeSweepTests
{
    // Above TinyShapeWorkThreshold (100k) so the regular dispatch path runs and
    // the BlockOverride actually flows through AutotuneDispatcher.Decide.
    private const int M = 96, N = 160, K = 128;

    [Theory]
    // A spread of blockings: tiny, BLIS-default, over-shape (forces clamp), odd.
    [InlineData(64, 256, 128)]
    [InlineData(128, 512, 256)]
    [InlineData(32, 64, 64)]
    [InlineData(192, 1024, 512)]   // all exceed the shape → clamp to (M, N, K)
    [InlineData(40, 96, 48)]       // non-mr/nr-aligned values
    public void BlockOverride_ProducesCorrectGemm_AcrossBlockings(int mc, int nc, int kc)
    {
        var (a, b) = MakeInputs(M, K, N, seed: 7);
        var reference = RefGemm(a, b, M, N, K);

        var c = new float[M * N];
        AutotuneDispatcher.BlockOverride = (mc, nc, kc);
        try
        {
            BlasManagedLib.Gemm<float>(
                a, K, false,
                b, N, false,
                c, N,
                M, N, K);
        }
        finally
        {
            AutotuneDispatcher.BlockOverride = null;
        }

        AssertClose(reference, c, atol: 1e-2f);
    }

    [Fact]
    public void Measure_ReturnsValidClampedBlocking_AndClearsOverride()
    {
        // FFN-ish shape that passes ShouldMeasure (n > 512) and has real block choice.
        const int m = 256, n = 768, k = 256, mr = 8, nr = 8;

        var result = BlockSizeSweep.Measure<float>(
            m, n, k, transA: false, transB: false, mr, nr, procs: Environment.ProcessorCount, isDeterministic: false);

        Assert.InRange(result.Mc, 1, m);
        Assert.InRange(result.Nc, 1, n);
        Assert.InRange(result.Kc, 1, k);
        Assert.Equal(0, result.Mc % mr); // mr-aligned
        Assert.Equal(0, result.Nc % nr); // nr-aligned
        Assert.True(result.MeasuredMs > 0, "winner should carry a positive measured time");

        // The probe must not leak a block override onto the calling thread.
        Assert.Null(AutotuneDispatcher.BlockOverride);
    }

    [Fact]
    public void Measure_Winner_ProducesCorrectGemm()
    {
        // The blocking the sweep selects must itself produce a correct GEMM.
        var winner = BlockSizeSweep.Measure<float>(
            M, N, K, transA: false, transB: false, mr: 8, nr: 8,
            procs: Environment.ProcessorCount, isDeterministic: false);

        var (a, b) = MakeInputs(M, K, N, seed: 11);
        var reference = RefGemm(a, b, M, N, K);

        var c = new float[M * N];
        AutotuneDispatcher.BlockOverride = (winner.Mc, winner.Nc, winner.Kc);
        try
        {
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K);
        }
        finally { AutotuneDispatcher.BlockOverride = null; }

        AssertClose(reference, c, atol: 1e-2f);
    }

    [Fact]
    public void Measure_Double_ReturnsValidBlocking()
    {
        const int m = 192, n = 640, k = 192, mr = 4, nr = 8;
        var result = BlockSizeSweep.Measure<double>(
            m, n, k, transA: false, transB: false, mr, nr, procs: Environment.ProcessorCount, isDeterministic: false);

        Assert.InRange(result.Mc, 1, m);
        Assert.InRange(result.Nc, 1, n);
        Assert.InRange(result.Kc, 1, k);
        Assert.Null(AutotuneDispatcher.BlockOverride);
    }

    [Fact]
    public void Decide_HonorsBlockOverride_AndReturnsClampedValues()
    {
        // With an override set, Decide must return it (clamped) without touching
        // the cache — this is the recursion guard the sweep relies on.
        AutotuneDispatcher.BlockOverride = (1000, 2000, 3000); // all exceed shape
        try
        {
            var d = AutotuneDispatcher.Decide<float>(
                m: 128, n: 192, k: 96,
                transA: false, transB: false,
                mr: 8, nr: 8,
                procs: 4,
                isDeterministic: false,
                hasEpilogue: false,
                packingMode: PackingMode.Auto);

            Assert.InRange(d.Mc, 1, 128);
            Assert.InRange(d.Nc, 1, 192);
            Assert.InRange(d.Kc, 1, 96);
        }
        finally { AutotuneDispatcher.BlockOverride = null; }
    }

    [Fact]
    public void Decide_ForcedMeasurement_EndToEnd_ProducesCorrectGemm()
    {
        // Drive the full cache-miss → BlockSizeSweep.Measure → Store → use path
        // (via the internal force-measure hook, so the test doesn't depend on the
        // process-start env var) and confirm the resulting GEMM is correct and no
        // override leaks. Shape passes ShouldMeasure (n > 512).
        const int m = 256, n = 768, k = 256;
        var (a, b) = MakeInputs(m, k, n, seed: 5);
        var reference = RefGemm(a, b, m, n, k);

        var c = new float[m * n];
        AutotuneDispatcher.ForceMeasureOnMiss = true;
        try
        {
            BlasManagedLib.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k);
        }
        finally { AutotuneDispatcher.ForceMeasureOnMiss = false; }

        Assert.Null(AutotuneDispatcher.BlockOverride);
        AssertClose(reference, c, atol: 5e-2f); // k=256 → looser float tolerance
    }

    [Theory]
    [InlineData(1000, 2000, 3000, 96, 160, 128, 8, 8)] // over-shape → clamp to dims (aligned)
    [InlineData(40, 96, 48, 96, 160, 128, 8, 8)]       // align down to mr/nr multiples
    [InlineData(3, 3, 3, 96, 160, 128, 8, 8)]          // below mr/nr → clamp up to mr/nr
    public void ClampBlocking_ClampsToShapeAndAlignment(
        int mc, int nc, int kc, int m, int n, int k, int mr, int nr)
    {
        var (cmc, cnc, ckc) = AutotuneDispatcher.ClampBlocking(mc, nc, kc, m, n, k, mr, nr);

        Assert.InRange(cmc, 1, m);
        Assert.InRange(cnc, 1, n);
        Assert.InRange(ckc, 1, k);
        // mr/nr alignment when the clamped value is large enough to align.
        if (cmc >= mr) Assert.Equal(0, cmc % mr);
        if (cnc >= nr) Assert.Equal(0, cnc % nr);
    }

    // ----------------- Helpers -----------------

    private static (float[] a, float[] b) MakeInputs(int m, int k, int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        return (a, b);
    }

    // Naive row-major reference: A[m,k] · B[k,n] = C[m,n], no transpose.
    private static float[] RefGemm(float[] a, float[] b, int m, int n, int k)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int kk = 0; kk < k; kk++)
                    s += (double)a[i * k + kk] * b[kk * n + j];
                c[i * n + j] = (float)s;
            }
        return c;
    }

    private static void AssertClose(float[] expected, float[] actual, float atol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff < atol,
                $"Mismatch at {i}: ref={expected[i]:G6}, got={actual[i]:G6}, diff={diff:G3} (atol={atol:G3}).");
        }
    }
}
