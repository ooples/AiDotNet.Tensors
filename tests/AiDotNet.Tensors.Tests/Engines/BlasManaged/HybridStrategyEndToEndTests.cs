using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #375 hybrid end-to-end: the table-routed (and, in later phases, learned-cache-routed)
/// strategy must always produce numerically correct output. transB shapes bypass the
/// Sub-S machine-code fast path, so they exercise the hybrid strategy selection.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class HybridStrategyEndToEndTests
{
    [Theory]
    [InlineData(96, 128, 64)]    // small low-K, transB → bypasses Sub-S, reaches strategy
    [InlineData(128, 128, 128)]  // true cube
    public void Table_Routed_Gemm_Matches_Reference_FP64(int M, int N, int K)
    {
        var rng = new Random(3);
        var a = new double[M * K];
        var b = new double[N * K]; // [N,K] for transB=true
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        var reference = new double[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double s = 0;
                for (int kk = 0; kk < K; kk++) s += a[i * K + kk] * b[j * K + kk];
                reference[i * N + j] = s;
            }

        var c = new double[M * N];
        BlasManagedLib.Gemm<double>(a, K, false, b, K, true, c, N, M, N, K);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - c[i]) < 1e-9,
                $"mismatch at {i}: ref {reference[i]} got {c[i]}");
    }

    [Fact]
    public void LearnedCache_Overrides_Table()
    {
        // Store a learned entry differing from the table's default, then assert
        // SelectStrategy returns the learned strategy (precedence learned > table).
        const int M = 200, N = 200, K = 64;  // table → Streaming on this box; learned → PackBoth
        var shape = BlasManagedAutotune.EncodeShape<float>(M, N, K, false, false, 0, 0, false,
            AiDotNet.Tensors.Helpers.BlasProvider.IsDeterministicMode);
        BlasManagedAutotune.StoreStrategy(shape, PackingMode.ForcePackBoth, ParallelismAxis.M,
            64, 64, 64, 8, BlasKernelVersion.Current);
        var opts = default(BlasOptions<float>);
        Assert.Equal(PackingMode.ForcePackBoth, Dispatcher.SelectStrategy<float>(M, N, K, in opts));
    }

    [Fact]
    public void Repeated_SmallShape_Eventually_GetsLearnedEntry()
    {
        // Two transB calls of the same small shape → 2nd sighting enqueues a background
        // measurement; after a short wait the learned cache should hold a strategy.
        const int M = 72, N = 72, K = 48;
        var a = new double[M * K];
        var b = new double[N * K]; // [N,K] for transB
        var c = new double[M * N];
        // Background autotuner is disabled assembly-wide for tests; enable locally. This
        // test class is in the DisableParallelization serial collection, so the worker
        // never overlaps other tests during this window.
        BackgroundAutotuner.Enabled = true;
        try
        {
            for (int i = 0; i < 2; i++)
                BlasManagedLib.Gemm<double>(a, K, false, b, K, true, c, N, M, N, K);
            System.Threading.Thread.Sleep(2000); // let the below-normal background worker run
            var shape = BlasManagedAutotune.EncodeShape<double>(M, N, K, false, true, 0, 0, false,
                AiDotNet.Tensors.Helpers.BlasProvider.IsDeterministicMode);
            Assert.NotNull(BlasManagedAutotune.TryLookupStrategy(shape));
        }
        finally { BackgroundAutotuner.Enabled = false; }
    }
}
