using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

// Kernel-library allocation+latency sweep. For each (kernel, size): warm, then measure
//   - bytes allocated PER CALL via GC.GetTotalAllocatedBytes (DETERMINISTIC — the A/B metric), and
//   - MIN wall-time per call (min-of-N).
// The deterministic baseline we re-measure after each fix to verify the work actually improves.
// AIDOTNET_FORCE_SERIAL=1 + OPENBLAS_NUM_THREADS=1 isolates a single thread for PerfView leaf views.
internal static class Program
{
    static readonly CpuEngine E = new();
    static readonly Random R = new(7);

    static Tensor<float> Rand(params int[] shape)
    {
        long n = 1; foreach (var s in shape) n *= s;
        var d = new float[n];
        for (long i = 0; i < n; i++) d[i] = (float)(R.NextDouble() * 2 - 1);
        return new Tensor<float>(d, shape);
    }

    // Measure one kernel call: bytes/call (deterministic) + min µs/call (min-of-reps).
    static void Bench(string name, string size, Func<object> call, int warm = 5, int reps = 20)
    {
        for (int i = 0; i < warm; i++) { var _ = call(); }
        long a0 = GC.GetTotalAllocatedBytes(false);
        double minUs = double.MaxValue;
        for (int i = 0; i < reps; i++)
        {
            var sw = Stopwatch.GetTimestamp();
            var _ = call();
            double us = (Stopwatch.GetTimestamp() - sw) * 1e6 / Stopwatch.Frequency;
            if (us < minUs) minUs = us;
        }
        long a1 = GC.GetTotalAllocatedBytes(false);
        double mbPerCall = (a1 - a0) / 1048576.0 / reps;
        Console.WriteLine($"{name,-22} {size,-18} {mbPerCall,10:F3} MB/call  {minUs,12:F1} us/call");
    }

    static int Main(string[] args)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_FORCE_SERIAL") == "1")
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.Current = E;
        Console.WriteLine($"=== KernelSweep (serial={AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism}) ===");
        Console.WriteLine($"{"kernel",-22} {"size",-18} {"alloc",10}            time");

        // --- GEMM (TensorMatMul) ---
        foreach (var (m, k, n, tag) in new[] {
            (64,64,64,"64^3"), (256,256,256,"256^3"), (256,1152,1152,"dit-proj"),
            (256,1152,4608,"dit-fc1"), (512,512,512,"512^3"), (1024,1024,1024,"1024^3"), (2048,2048,2048,"2048^3") })
        {
            var a = Rand(m, k); var b = Rand(k, n);
            Bench("TensorMatMul", tag, () => E.TensorMatMul(a, b));
        }

        // --- FusedLinear (input[M,K] @ w[K,N] + bias) ---
        foreach (var (m, k, n, tag) in new[] {
            (256,1152,1152,"dit-qkvo"), (256,1152,4608,"dit-fc1"), (256,4608,1152,"dit-fc2"), (1024,1024,1024,"1024^3") })
        {
            var input = Rand(m, k); var w = Rand(k, n); var bias = Rand(n);
            Bench("FusedLinear", tag, () => E.FusedLinear(input, w, bias, FusedActivationType.None));
        }

        // --- ScaledDotProductAttention [B,H,S,D] ---
        foreach (var (bb, h, s, d, tag) in new[] {
            (1,16,128,72,"seq128"), (1,16,256,72,"seq256"), (1,16,512,72,"seq512"), (1,16,1024,72,"seq1024") })
        {
            var q = Rand(bb, h, s, d); var k = Rand(bb, h, s, d); var v = Rand(bb, h, s, d);
            Bench("SDPA", tag, () => E.ScaledDotProductAttention(q, k, v, null, 1.0 / Math.Sqrt(d), out _));
        }

        // --- Broadcast add / multiply (AdaLN-style: [M,N] (+|*) [1,N]) ---
        foreach (var (m, n, tag) in new[] { (256,1152,"256x1152"), (256,4608,"256x4608"), (1024,1024,"1024^2") })
        {
            var a = Rand(m, n); var b = Rand(1, n);
            Bench("TensorBroadcastAdd", tag, () => E.TensorBroadcastAdd(a, b));
            Bench("TensorBroadcastMultiply", tag, () => E.TensorBroadcastMultiply(a, b));
        }

        // --- TensorScaledDotProductAttention (no-weights inference SDPA) ---
        foreach (var (bb, h, s, d, tag) in new[] {
            (1,16,128,72,"seq128"), (1,16,256,72,"seq256"), (1,16,512,72,"seq512"), (1,16,1024,72,"seq1024") })
        {
            var q = Rand(bb, h, s, d); var k = Rand(bb, h, s, d); var v = Rand(bb, h, s, d);
            Bench("TensorSDPA", tag, () => E.TensorScaledDotProductAttention(q, k, v));
        }

        Console.WriteLine("=== done ===");
        return 0;
    }
}
