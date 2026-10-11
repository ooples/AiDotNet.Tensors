using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

// #1141 head-to-head: one attention training step on the GPU engine (forward, sum(out * gy), backward to Q, K, V,
// device synchronize) for the weightless SDPA ([B, H, S, D]) and MultiHeadAttentionCore ([B, S, H * D]). Median and
// minimum of `reps` steps after warmup, in ms. Pair with torch_attn.py for the PyTorch side.
// Usage: AttnH2H <batch> <heads> <headDim> <reps> <seq,seq,...>
internal static class Program
{
    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        int n = 1; foreach (int d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    // A device-resident copy (the output of a GPU op), as attention inputs are in a real training step; a host tensor
    // would be uploaded again on every call.
    private static Tensor<float> Resident(IEngine e, Tensor<float> host) => e.TensorMultiplyScalar(host, 1f);

    private static (double Median, double Min) Time(Action step, int reps)
    {
        for (int i = 0; i < 5; i++) step();
        var ts = new double[reps];
        var sw = new Stopwatch();
        for (int i = 0; i < reps; i++) { sw.Restart(); step(); ts[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(ts);
        return (ts[reps / 2], ts[0]);
    }

    private static int Main(string[] args)
    {
        int b = int.Parse(args[0]), h = int.Parse(args[1]), d = int.Parse(args[2]), reps = int.Parse(args[3]);
        var seqs = Array.ConvertAll(args[4].Split(','), int.Parse);
        var gpu = new DirectGpuTensorEngine();
        IEngine e = gpu;
        var backend = gpu.GetBackend() ?? throw new InvalidOperationException("no GPU backend");
        var rng = new Random(0);
        Console.WriteLine($"backend={backend.BackendName} B={b} H={h} D={d} reps={reps}");
        Console.WriteLine("api   S      causal  median_ms  min_ms");
        foreach (int s in seqs)
            foreach (bool causal in new[] { false, true })
            {
                var q = Resident(e, Random(rng, b, h, s, d)); var k = Resident(e, Random(rng, b, h, s, d));
                var v = Resident(e, Random(rng, b, h, s, d)); var gy = Resident(e, Random(rng, b, h, s, d));
                var (sdpaMed, sdpaMin) = Time(() =>
                {
                    using var tape = new GradientTape<float>();
                    var o = e.ScaledDotProductAttention(q, k, v, null, null, causal);
                    var loss = e.ReduceSum(e.TensorMultiply(o, gy), null);
                    tape.ComputeGradients(loss, new[] { q, k, v });
                    backend.Synchronize();
                }, reps);
                Console.WriteLine($"sdpa  {s,-6} {causal,-7} {sdpaMed,9:F3} {sdpaMin,7:F3}");

                var qm = Resident(e, Random(rng, b, s, h * d)); var km = Resident(e, Random(rng, b, s, h * d));
                var vm = Resident(e, Random(rng, b, s, h * d)); var gm = Resident(e, Random(rng, b, s, h * d));
                var (mhaMed, mhaMin) = Time(() =>
                {
                    using var tape = new GradientTape<float>();
                    var o = e.MultiHeadAttentionCore(qm, km, vm, h, null, causal);
                    var loss = e.ReduceSum(e.TensorMultiply(o, gm), null);
                    tape.ComputeGradients(loss, new[] { qm, km, vm });
                    backend.Synchronize();
                }, reps);
                Console.WriteLine($"mha   {s,-6} {causal,-7} {mhaMed,9:F3} {mhaMin,7:F3}");
            }
        return 0;
    }
}
