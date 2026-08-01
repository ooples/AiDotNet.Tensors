using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

// Allocation audit of the kernel library. For each (kernel, size): warm, then measure bytes
// allocated PER CALL via GC.GetTotalAllocatedBytes (deterministic — immune to box noise), and
// the alloc/output RATIO. ratio≈1 ⇒ the op only allocates its result (good); ratio≫1 ⇒ hidden
// intermediate allocation that can be eliminated (the SDPA pattern — the win candidates).
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
    static Tensor<float> RandPos(params int[] shape) // strictly positive (for ops that need it)
    {
        long n = 1; foreach (var s in shape) n *= s;
        var d = new float[n];
        for (long i = 0; i < n; i++) d[i] = (float)(R.NextDouble() * 0.9 + 0.1);
        return new Tensor<float>(d, shape);
    }

    static void Bench(string name, string size, Func<Tensor<float>> call, int warm = 5, int reps = 20)
    {
        Tensor<float>? last = null;
        try
        {
            for (int i = 0; i < warm; i++) last = call();
            long a0 = GC.GetTotalAllocatedBytes(false);
            double minUs = double.MaxValue;
            for (int i = 0; i < reps; i++)
            {
                var sw = Stopwatch.GetTimestamp();
                last = call();
                double us = (Stopwatch.GetTimestamp() - sw) * 1e6 / Stopwatch.Frequency;
                if (us < minUs) minUs = us;
            }
            long a1 = GC.GetTotalAllocatedBytes(false);
            double mb = (a1 - a0) / 1048576.0 / reps;
            double outMB = (last?.Length ?? 0) * 4.0 / 1048576.0;
            double ratio = outMB > 1e-9 ? mb / outMB : 0;
            string flag = ratio >= 2.0 ? "  <<< " + ratio.ToString("F1") + "x" : "";
            Console.WriteLine($"{name,-26} {size,-16} {mb,9:F3} MB/call  out={outMB,8:F3}  ratio={ratio,6:F2}x{flag}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"{name,-26} {size,-16} ERROR: {ex.GetType().Name}: {ex.Message.Split('\n')[0]}");
        }
    }

    static int Main(string[] args)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_FORCE_SERIAL") == "1")
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.Current = E;
        Console.WriteLine("=== KernelSweep allocation audit (ratio >= 2x flagged as a win candidate) ===");
        Console.WriteLine($"{"kernel",-26} {"size",-16} {"alloc",9}           output     ratio");

        // ---- GEMM / FusedLinear (baseline: should be ~1x = output only) ----
        foreach (var (m, k, n, t) in new[] { (256, 1152, 1152, "dit-proj"), (256, 1152, 4608, "dit-fc1"), (1024, 1024, 1024, "1024^3") })
            { var a = Rand(m, k); var b = Rand(k, n); Bench("TensorMatMul", t, () => E.TensorMatMul(a, b)); }
        foreach (var (m, k, n, t) in new[] { (256, 1152, 4608, "dit-fc1"), (256, 4608, 1152, "dit-fc2") })
            { var a = Rand(m, k); var w = Rand(k, n); var bs = Rand(n); Bench("FusedLinear", t, () => E.FusedLinear(a, w, bs, FusedActivationType.None)); }

        // ---- Normalizations (out mean/var/rms = potential removable intermediates) ----
        foreach (var (rows, dd, t) in new[] { (256, 1152, "256x1152"), (1024, 1024, "1024x1024") })
        {
            var x = Rand(rows, dd); var g = Rand(dd); var b = Rand(dd);
            Bench("LayerNorm", t, () => E.LayerNorm(x, g, b, 1e-5, out _, out _));
            Bench("RMSNorm", t, () => E.RMSNorm(x, g, 1e-5, out _));
        }
        foreach (var (nn, c, h, w, t) in new[] { (2, 32, 32, 32, "2x32x32x32"), (1, 64, 56, 56, "1x64x56x56") })
        {
            var x = Rand(nn, c, h, w); var g = Rand(c); var b = Rand(c);
            Bench("GroupNorm", t, () => E.GroupNorm(x, 8, g, b, 1e-5, out _, out _));
            Bench("BatchNormInference", t, () => E.BatchNormInference(x, g, b, Rand(c), RandPos(c), 1e-5));
        }

        // ---- Softmax / activations (should be ~1x) ----
        foreach (var (rows, cols, t) in new[] { (256, 1152, "256x1152"), (4096, 256, "4096x256") })
        {
            var x = Rand(rows, cols);
            Bench("Softmax", t, () => E.Softmax(x, -1));
            Bench("GELU", t, () => E.GELU(x));
            Bench("Swish", t, () => E.Swish(x));
            Bench("Mish", t, () => E.Mish(x));
        }

        // ---- Conv / pool (Conv2D im2col is a classic hidden-allocation case) ----
        foreach (var (nn, c, h, w, oc, t) in new[] { (1, 64, 32, 32, 64, "1x64x32x32"), (1, 128, 28, 28, 128, "1x128x28x28") })
        {
            var x = Rand(nn, c, h, w); var ker = Rand(oc, c, 3, 3);
            Bench("Conv2D 3x3", t, () => E.Conv2D(x, ker, 1, 1, 1));
            Bench("MaxPool2D 2x2", t, () => E.MaxPool2D(x, 2, 2, 0));
            Bench("AvgPool2D 2x2", t, () => E.AvgPool2D(x, 2, 2, 0));
        }

        // ---- Layout ops (transpose/permute/concat — each allocates a copy) ----
        { var x = Rand(1152, 1152); Bench("TensorTranspose", "1152^2", () => E.TensorTranspose(x)); }
        { var x = Rand(2, 16, 256, 72); Bench("TensorPermute 0231", "2x16x256x72", () => E.TensorPermute(x, new[] { 0, 2, 1, 3 })); }
        { var a = Rand(256, 576); var b = Rand(256, 576); Bench("TensorConcatenate", "2x256x576", () => E.TensorConcatenate(new[] { a, b }, 1)); }

        Console.WriteLine("=== done ===");
        return 0;
    }
}
