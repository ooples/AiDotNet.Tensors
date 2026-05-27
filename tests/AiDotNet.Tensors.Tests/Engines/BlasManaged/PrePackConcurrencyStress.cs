using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #446 investigation: reproduce the intermittent pre-pack output corruption seen
/// on CI (drift 59.7) by hammering PrePackB + Gemm(PackedB) from many threads while
/// concurrently mutating the global CpuParallelSettings (as the ParallelFor tests
/// do). If a production race exists in the shared allocator / packed-buffer path,
/// this should surface it locally; if it stays bit-exact under heavy stress, the CI
/// flake is environment-specific cross-test contamination (→ test isolation).
/// </summary>
[Trait("Category", "Performance")] // heavy (19k ops); a concurrency regression guard
public class PrePackConcurrencyStress
{
    private readonly ITestOutputHelper _output;
    public PrePackConcurrencyStress(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void PrePack_UnderConcurrentGemms_StaysBitExact()
    {
        int threads = Math.Max(4, Environment.ProcessorCount);
        const int itersPerThread = 200;
        var (m, n, k) = (8, 1024, 1024); // the exact CI-failing shape
        int failures = 0;
        double worstDrift = 0;
        var sync = new object();

        // A background agitator that flips the global parallel settings the way the
        // ParallelFor / DeterministicReductions tests do — to mimic cross-test churn.
        using var stop = new CancellationTokenSource();
        var agitator = Task.Run(() =>
        {
            var rng = new Random(999);
            while (!stop.IsCancellationRequested)
            {
                CpuParallelSettings.DeterministicReductions = rng.Next(2) == 0;
                CpuParallelSettings.MaxDegreeOfParallelism = 1 + rng.Next(Environment.ProcessorCount);
                Thread.Yield();
            }
        });

        Parallel.For(0, threads, t =>
        {
            var rng = new Random(1234 + t);
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
            var cLive = new float[m * n];
            var cPre = new float[m * n];

            for (int it = 0; it < itersPerThread; it++)
            {
                BlasManagedLib.Gemm<float>(a, k, false, b, n, false, cLive, n, m, n, k);
                var handle = BlasManagedLib.PrePackB<float>(b, n, false, k, n);
                try
                {
                    var opts = new BlasOptions<float> { PackedB = handle };
                    BlasManagedLib.Gemm<float>(a, k, false, b, n, false, cPre, n, m, n, k, opts);
                }
                finally { handle.Dispose(); }

                double drift = 0;
                for (int i = 0; i < cLive.Length; i++)
                    drift = Math.Max(drift, Math.Abs((double)cLive[i] - cPre[i]));
                if (drift > 1e-3)
                    lock (sync) { failures++; worstDrift = Math.Max(worstDrift, drift); }
            }
        });

        stop.Cancel();
        agitator.Wait();
        CpuParallelSettings.DeterministicReductions = false;
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;

        _output.WriteLine($"threads={threads} iters/thread={itersPerThread} failures={failures} worstDrift={worstDrift:G6}");
        Assert.True(failures == 0,
            $"pre-pack corrupted under concurrency: {failures} mismatches, worst drift {worstDrift:G6}");
    }
}
