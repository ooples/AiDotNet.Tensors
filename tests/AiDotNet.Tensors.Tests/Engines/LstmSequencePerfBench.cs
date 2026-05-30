// Copyright (c) AiDotNet. All rights reserved.
// #477 perf measurement for LstmSequenceForward at the AIsEval workload
// ([128,32,32]->64, last-step output). Reports median / p95 wall-clock so the
// fused-recurrent-kernel work can be measured against the torch ~2.78ms target
// without needing TorchSharp in the harness. Category=Performance => excluded
// from the normal/CI run.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

[Trait("Category", "Performance")]
public class LstmSequencePerfBench
{
    private readonly ITestOutputHelper _out;
    public LstmSequencePerfBench(ITestOutputHelper output) => _out = output;

    [Fact]
    public void LstmSequenceForward_AisEvalWorkload_Timing()
    {
        const int batch = 128, seq = 32, inF = 32, hidden = 64;
        const int warmup = 50, measured = 200;

        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom(batch, seq, inF);
        var wIh = Tensor<float>.CreateRandom(4 * hidden, inF);
        var wHh = Tensor<float>.CreateRandom(4 * hidden, hidden);

        for (int i = 0; i < warmup; i++)
            engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null);

        var times = new double[measured];
        for (int i = 0; i < measured; i++)
        {
            var sw = Stopwatch.StartNew();
            engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        double median = times[measured / 2];
        double p95 = times[(int)(measured * 0.95)];
        double min = times[0];
        _out.WriteLine($"LSTM [128,32,32]->64 last-step, {measured} runs:");
        _out.WriteLine($"  min={min:F3} ms  median={median:F3} ms  p95={p95:F3} ms   (torch target ~2.78 ms)");

        Assert.True(median > 0);
    }

    [Fact]
    public void LstmSequenceForward_AllocationProfile()
    {
        const int batch = 128, seq = 32, inF = 32, hidden = 64;
        const int warmup = 50, calls = 500;

        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom(batch, seq, inF);
        var wIh = Tensor<float>.CreateRandom(4 * hidden, inF);
        var wHh = Tensor<float>.CreateRandom(4 * hidden, hidden);

        for (int i = 0; i < warmup; i++)
            engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null);

        long before = AllocatedBytes();
        int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
        for (int i = 0; i < calls; i++)
            engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null);
        long after = AllocatedBytes();

        double perCall = (after - before) / (double)calls;
        _out.WriteLine($"LSTM [128,32,32]->64 allocation over {calls} calls:");
        _out.WriteLine($"  bytes/call = {perCall:F0}");
        _out.WriteLine($"  gen0 collections = {GC.CollectionCount(0) - g0}, gen1 = {GC.CollectionCount(1) - g1}, gen2 = {GC.CollectionCount(2) - g2}");

        Assert.True(perCall >= 0);
    }

    // GC.GetAllocatedBytesForCurrentThread() is .NET Core 3.0+ only; on net471 the
    // build broke (CS0117). Fall back to a coarse whole-heap proxy there so this
    // diagnostic bench compiles and runs on both target frameworks.
    private static long AllocatedBytes()
    {
#if NET5_0_OR_GREATER
        return GC.GetAllocatedBytesForCurrentThread();
#else
        return GC.GetTotalMemory(forceFullCollection: false);
#endif
    }
}
