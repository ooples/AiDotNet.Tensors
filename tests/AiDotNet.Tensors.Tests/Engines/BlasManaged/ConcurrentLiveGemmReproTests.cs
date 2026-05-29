using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

// TEMP: confirm whether concurrent LIVE (no-handle, parallel pack-B) GEMMs with
// distinct data corrupt each other. The full-suite repro showed ParallelPackBTest
// (serial-vs-parallel Gemm) drifting 27.2 under the concurrent suite, while a
// pre-packed-B stress stayed clean — implicating the live parallel pack-B path.
public class ConcurrentLiveGemmReproTests
{
    [Fact]
    public void ConcurrentLiveGemms_DistinctData_StayCorrect()
    {
        int M = 512, N = 1024, K = 768;
        int threads = Math.Max(8, Environment.ProcessorCount * 2);
        var drifts = new ConcurrentBag<string>();

        Parallel.For(0, threads, tid =>
        {
            var rng = new Random(100 + tid);
            var a = new float[M * K];
            var b = new float[K * N];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            // Per-thread serial reference (NumThreads=-1 forces single-thread GEMM).
            var cRef = new float[M * N];
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cRef, N, M, N, K,
                new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { NumThreads = -1 });

            for (int it = 0; it < 40; it++)
            {
                var c = new float[M * N];
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K); // live, parallel
                double maxDelta = 0;
                for (int i = 0; i < cRef.Length; i++)
                    maxDelta = Math.Max(maxDelta, Math.Abs(cRef[i] - c[i]));
                if (maxDelta > 1e-2) { drifts.Add($"tid {tid} it {it}: drift {maxDelta:G6}"); break; }
            }
        });

        Assert.True(drifts.IsEmpty,
            $"{drifts.Count} concurrent live GEMMs drifted from their own serial reference. " +
            $"First: {(drifts.IsEmpty ? "" : System.Linq.Enumerable.First(drifts))}");
    }
}
