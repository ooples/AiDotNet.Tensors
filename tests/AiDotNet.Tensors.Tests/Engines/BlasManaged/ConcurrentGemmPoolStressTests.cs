using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Reproduces the load-dependent host-process crash the full suite hit after im2col packing was
/// migrated to the persistent pool's Execute&lt;TLocal&gt;. Drives the REAL unsafe machine-kernel GEMM
/// (PackingMode.Auto → im2col packing on the pool) from many concurrent threads, so the singleton pool
/// sees concurrent callers each nesting a thread-local packing dispatch inside a ParallelForRegion tile
/// dispatch — the exact shape that faulted. Verifies every result against a single-threaded reference:
/// buffer aliasing/use-after-free shows up either as a wrong result here or as an access-violation crash
/// inside the packing/microkernel.
/// </summary>
public class ConcurrentGemmPoolStressTests
{
    private static void ReferenceGemm(double[] a, double[] b, double[] c, int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
                c[i * n + j] = sum;
            }
    }

    [Fact]
    public void ConcurrentMachineKernelGemm_NoCorruptionNoCrash()
    {
        // ViT-class shape: M/6 * ... produces MANY mc-blocks, so im2col packing actually parallelizes
        // through Execute<TLocal> (rented per-participant panels) rather than running inline.
        const int M = 192, N = 768, K = 384;
        var rng = new Random(20260710);
        var a = new double[M * K];
        var b = new double[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        var cRef = new double[M * N];
        ReferenceGemm(a, b, cRef, M, N, K);
        double refScale = cRef.Max(Math.Abs);
        double tol = 1e-9 * Math.Max(1.0, refScale) * K;

        int threads = Math.Max(8, Environment.ProcessorCount);
        const int itersPerThread = 60;
        var errors = new ConcurrentQueue<string>();

        Parallel.For(0, threads, t =>
        {
            for (int it = 0; it < itersPerThread && errors.IsEmpty; it++)
            {
                var c = new double[M * N];
                BlasManagedLib.Gemm<double>(a, K, false, b, N, false, c, N, M, N, K,
                    new BlasOptions<double> { PackingMode = PackingMode.Auto });

                double maxErr = 0;
                int worst = -1;
                for (int i = 0; i < c.Length; i++)
                {
                    double e = Math.Abs(c[i] - cRef[i]);
                    if (e > maxErr) { maxErr = e; worst = i; }
                }
                if (maxErr > tol)
                    errors.Enqueue($"t{t} it{it}: maxErr={maxErr:G6} @ {worst} (tol={tol:G6})");
            }
        });

        Assert.True(errors.IsEmpty,
            "Concurrent machine-kernel GEMM corrupted results (buffer aliasing in pooled im2col packing):\n  "
            + string.Join("\n  ", errors.Take(8)));
    }
}
