using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Times the fused LSTM forward+backward (#587 kernel) at the AiDotNet benchmark's LSTM
/// shape (batch 64, in 32, hidden 64, seq 32) to track the effect of routing the big
/// forward/backward GEMMs to the parallel BlasManaged.Gemm.
/// </summary>
public class LstmFusedBackwardPerfBench
{
    private readonly ITestOutputHelper _o;
    public LstmFusedBackwardPerfBench(ITestOutputHelper o) => _o = o;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return t;
    }

    [Fact(Skip = "Benchmark — run manually")]
    public void Bench_FusedLstmForwardBackward()
    {
        var eng = new CpuEngine();
        int batch = 64, seq = 32, inF = 32, hidden = 64, G = 4 * hidden;
        var input = Rand(new[] { batch, seq, inF }, 1);
        var wIh = Rand(new[] { G, inF }, 2);
        var wHh = Rand(new[] { G, hidden }, 3);
        var bIh = Rand(new[] { G }, 4);
        var bHh = Rand(new[] { G }, 5);

        void Step()
        {
            using var tape = new GradientTape<float>();
            var outp = eng.LstmSequenceForward(input, null, null, wIh, wHh, bIh, bHh, returnSequences: true);
            var loss = eng.ReduceSum(eng.TensorMultiply(outp, outp), null);
            var grads = tape.ComputeGradients(loss, new[] { input, wIh, wHh, bIh, bHh });
            if (grads.Count == 0) throw new Exception("no grads");
        }

        for (int i = 0; i < 50; i++) Step(); // warm

        // GC.GetAllocatedBytesForCurrentThread is netcoreapp3.0+, not on net471.
        // Fall back to GC.GetTotalMemory on net471 — same shape (relative delta
        // around the timed loop), coarser absolute number, but the perf-bench
        // readout is only a diagnostic for the dev box anyway.
        const int allocIters = 100;
#if NET6_0_OR_GREATER
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < allocIters; i++) Step();
        double kbPerStep = (GC.GetAllocatedBytesForCurrentThread() - a0) / 1024.0 / allocIters;
#else
        long a0 = GC.GetTotalMemory(forceFullCollection: false);
        for (int i = 0; i < allocIters; i++) Step();
        double kbPerStep = (GC.GetTotalMemory(forceFullCollection: false) - a0) / 1024.0 / allocIters;
#endif
        _o.WriteLine($"alloc/step = {kbPerStep:F1} KB");

        const int iters = 500;
        double best = double.MaxValue, sum = 0;
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            Step();
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            best = Math.Min(best, ms);
            sum += ms;
        }
        _o.WriteLine($"ENGINE={eng.GetType().Name} fused LSTM fwd+bwd [b{batch},s{seq},in{inF},h{hidden}]: min {best:F4} ms | avg {sum / iters:F4} ms");
    }
}
