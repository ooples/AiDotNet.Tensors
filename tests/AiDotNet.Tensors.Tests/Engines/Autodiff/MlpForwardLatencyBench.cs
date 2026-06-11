using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// MLP inference (fused MlpForward) latency at the report's batch sizes, to locate the
/// small-batch loss vs PyTorch (report 2: PyTorch wins all 4 MLP batches; batch-1
/// 0.172 ms vs our 0.262 ms). 784->512->128->10 ReLU, matching the benchmark.
/// </summary>
public class MlpForwardLatencyBench
{
    private readonly ITestOutputHelper _o;
    public MlpForwardLatencyBench(ITestOutputHelper o) => _o = o;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    [Fact(Skip = "Benchmark — run manually")]
    public void Probe_MlpForwardLatency()
    {
        var eng = AiDotNetEngine.Current;
        int[] sizes = { 784, 512, 128, 10 };
        var weights = new Tensor<float>[sizes.Length - 1];
        var biases = new Tensor<float>?[sizes.Length - 1];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = Rand(new[] { sizes[i], sizes[i + 1] }, i + 1);
            biases[i] = Rand(new[] { sizes[i + 1] }, 100 + i);
        }

        _o.WriteLine($"ENGINE={eng.GetType().Name}  MLP 784->512->128->10 ReLU");
        // PyTorch report-2 avg latency (ms) at batch 1/8/32/128 for reference.
        var ptMs = new System.Collections.Generic.Dictionary<int, double> { { 1, 0.172 }, { 8, 0.270 }, { 32, 0.427 }, { 128, 1.448 } };
        foreach (int B in new[] { 1, 8, 32, 128 })
        {
            var x = Rand(new[] { B, 784 }, 7 + B);
            for (int i = 0; i < 50; i++) eng.MlpForward(x, weights, biases, FusedActivationType.ReLU, FusedActivationType.None);
            const int iters = 2000;
            double best = double.MaxValue, sum = 0;
            var sw = new System.Diagnostics.Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                eng.MlpForward(x, weights, biases, FusedActivationType.ReLU, FusedActivationType.None);
                sw.Stop();
                double ms = sw.Elapsed.TotalMilliseconds;
                best = Math.Min(best, ms); sum += ms;
            }
            double avg = sum / iters;
            double pt = ptMs[B];
            string verdict = avg < pt ? "WIN" : $"LOSE {avg / pt:F2}x";
            _o.WriteLine($"  B={B,3}: min {best,7:F4} ms | avg {avg,7:F4} ms | PyTorch {pt,6:F3} ms => {verdict}  (throughput {B / (avg / 1000.0),10:F0} samp/s)");
        }
    }
}
