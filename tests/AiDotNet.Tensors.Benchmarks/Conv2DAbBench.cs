using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Focused A/B benchmark for the Conv3x3Stride1 variants (per-channel,
/// 4-oc-blocked, 2-oc-blocked). Same process, same warmup, same data —
/// answers "which variant is fastest at the BDN benchmark shape?" without
/// the BDN-subprocess overhead that obscured the prior comparison.
///
/// Approach:
///  1. Build deterministic input/kernel tensors once
///  2. Reflectively poke the static <c>SimdConvHelper.ActiveConv3x3Variant</c>
///     field to switch variants without env vars
///  3. Warmup each variant for fixed iters, then time fixed iters
///  4. Report mean + min + stddev side-by-side
///
/// Run with: <c>dotnet run -c Release -- --ab-conv2d</c>
/// </summary>
internal static class Conv2DAbBench
{
    public static void RunConv2DDouble()
    {
        Console.WriteLine("=== Conv2D<double> micro-benchmark ===");
        var engine = new CpuEngine();
        var shapes = new (int b, int ic, int h, int w, int oc)[]
        {
            (1, 3, 32, 32, 16),     // BDN Conv2D_Double benchmark shape
            (1, 16, 64, 64, 32),    // larger ResNet
            (4, 3, 32, 32, 16),     // batched
        };
        foreach (var (b, ic, h, w, oc) in shapes)
        {
            var rng = new Random(42);
            var inputData = new double[b * ic * h * w];
            var kernelData = new double[oc * ic * 9];
            for (int i = 0; i < inputData.Length; i++)  inputData[i]  = rng.NextDouble() * 2 - 1;
            for (int i = 0; i < kernelData.Length; i++) kernelData[i] = rng.NextDouble() * 2 - 1;
            var input = new Tensor<double>(inputData, new[] { b, ic, h, w });
            var kernel = new Tensor<double>(kernelData, new[] { oc, ic, 3, 3 });

            for (int i = 0; i < 10; i++) { var rWarm = engine.Conv2D(input, kernel, 1, 1, 1); _ = rWarm; }
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var rMeas = engine.Conv2D(input, kernel, 1, 1, 1);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
            }
            double mean = samples.Average();
            double min = samples.Min();
            Console.WriteLine($"  Shape [{b},{ic},{h},{w}]→[{oc},3,3]:  mean={mean,8:F1} µs   min={min,8:F1} µs");
        }
        Console.WriteLine();
    }

    public static void RunMatMul()
    {
        Console.WriteLine("=== MatMul (TensorMatMul, float) micro-benchmark ===");
        var engine = new CpuEngine();
        var shapes = new (int M, int K, int N)[]
        {
            (256, 256, 256),    // 16.8M FMAs — current SgemmDirect target
            (512, 512, 512),    // 134M FMAs — SgemmTiled with Mc=192
            (128, 128, 128),    // 2.1M FMAs — small, probably below parallel threshold
            (768, 768, 768),    // 453M FMAs — SgemmTiled
            (256, 64, 256),     // 4.2M FMAs — attention shape
        };
        foreach (var (M, K, N) in shapes)
        {
            var rng = new Random(42);
            var aData = new float[M * K];
            var bData = new float[K * N];
            for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);
            var a = new Tensor<float>(aData, new[] { M, K });
            var b = new Tensor<float>(bData, new[] { K, N });

            for (int i = 0; i < 10; i++) { var rWarm = engine.TensorMatMul(a, b); _ = rWarm; }
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var rMeas = engine.TensorMatMul(a, b);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
            }
            double mean = samples.Average();
            double min = samples.Min();
            long workFmas = (long)M * K * N;
            double gflops = workFmas * 2.0 / (min * 1000.0);
            Console.WriteLine($"  Shape [{M},{K}]×[{K},{N}] ({workFmas / 1_000_000.0:F1}M FMAs):  " +
                              $"mean={mean,8:F1} µs   min={min,8:F1} µs   ({gflops,5:F1} GFLOPS)");
        }
        Console.WriteLine();
    }

    public static void RunLayerNorm()
    {
        Console.WriteLine("=== LayerNorm micro-benchmark ===");
        var engine = new CpuEngine();
        var shapes = new (int batch, int features)[]
        {
            (32768, 64),    // BDN benchmark
            (1, 768),       // BERT [1, 768]
            (1024, 64),     // small batch
            (4096, 128),    // medium
            (1, 1024),      // larger feature
        };
        foreach (var (batch, fs) in shapes)
        {
            var rng = new Random(42);
            var inp = new float[batch * fs];
            var gam = new float[fs];
            var bet = new float[fs];
            for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < gam.Length; i++) gam[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < bet.Length; i++) bet[i] = (float)(rng.NextDouble() * 2 - 1);
            var input = new Tensor<float>(inp, new[] { batch, fs });
            var gamma = new Tensor<float>(gam, new[] { fs });
            var beta  = new Tensor<float>(bet, new[] { fs });

            for (int i = 0; i < 10; i++) { var rWarm = engine.LayerNorm(input, gamma, beta, 1e-5, out _, out _); _ = rWarm; }
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var rMeas = engine.LayerNorm(input, gamma, beta, 1e-5, out _, out _);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
            }
            double mean = samples.Average();
            double min = samples.Min();
            Console.WriteLine($"  Shape [{batch},{fs}]:  mean={mean,8:F1} µs   min={min,8:F1} µs");
        }
        Console.WriteLine();
    }

    public static void RunAttentionQkt()
    {
        Console.WriteLine("=== AttentionQKT (Q · Kᵀ) micro-benchmark ===");
        var engine = new CpuEngine();

        // Standard attention shapes: [seqLen, headDim] × [seqLen, headDim]ᵀ → [seqLen, seqLen]
        var shapes = new (int seqLen, int headDim)[]
        {
            (512, 64),    // BDN benchmark shape
            (256, 64),    // small attention
            (1024, 64),   // larger seq
            (512, 128),   // larger head_dim
        };

        foreach (var (seqLen, headDim) in shapes)
        {
            var rng = new Random(42);
            var qData = new float[seqLen * headDim];
            var kData = new float[seqLen * headDim];
            for (int i = 0; i < qData.Length; i++) qData[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < kData.Length; i++) kData[i] = (float)(rng.NextDouble() * 2 - 1);
            var q = new Tensor<float>(qData, new[] { seqLen, headDim });
            var k = new Tensor<float>(kData, new[] { seqLen, headDim });

            for (int i = 0; i < 10; i++) { var _ = engine.TensorMatMulTransposed(q, k); }
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var _ = engine.TensorMatMulTransposed(q, k);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
            }
            double mean = samples.Average();
            double min = samples.Min();
            long workFmas = (long)seqLen * seqLen * headDim;
            double gflops = workFmas * 2.0 / (min * 1000.0); // 2 ops per FMA
            Console.WriteLine($"  Shape Q[{seqLen},{headDim}] K[{seqLen},{headDim}]^T " +
                              $"({workFmas / 1_000_000.0:F1}M FMAs):  " +
                              $"mean={mean,8:F1} µs   min={min,8:F1} µs   ({gflops,5:F1} GFLOPS)");
        }
        Console.WriteLine();
    }

    public static void RunSoftmaxDouble()
    {
        Console.WriteLine("=== Softmax<double> micro-benchmark ===");
        var engine = new CpuEngine();
        // BDN benchmark shape: [512, 1024]
        var shapes = new (int rows, int cols)[]
        {
            (512, 1024),  // BDN benchmark
            (32, 1024),   // small batch
            (4, 32768),   // single very-long row
            (1024, 256),  // many rows
        };
        foreach (var (rows, cols) in shapes)
        {
            var rng = new Random(42);
            var data = new double[rows * cols];
            for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 4 - 2;
            var input = new Tensor<double>(data, new[] { rows, cols });

            // Warmup
            for (int i = 0; i < 10; i++) { var _ = engine.Softmax(input, axis: -1); }

            // Measure
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var _ = engine.Softmax(input, axis: -1);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
            }
            double mean = samples.Average();
            double min = samples.Min();
            Console.WriteLine($"  Shape [{rows},{cols}]:  mean={mean,8:F1} µs   min={min,8:F1} µs");
        }
        Console.WriteLine();
    }

    public static void Run()
    {
        Console.WriteLine("=== Conv3x3Stride1 A/B variant benchmark ===");
        Console.WriteLine($"OS: {System.Runtime.InteropServices.RuntimeInformation.OSDescription}");
        Console.WriteLine($"Cores: {Environment.ProcessorCount}");
        Console.WriteLine();

        // Reflective access to internal SimdConvHelper.ActiveConv3x3Variant
        var helperType = typeof(CpuEngine).Assembly
            .GetType("AiDotNet.Tensors.Helpers.SimdConvHelper", throwOnError: true);
        var variantField = helperType!.GetField("ActiveConv3x3Variant",
            BindingFlags.Static | BindingFlags.NonPublic);
        var variantEnum = helperType.GetNestedType("Conv3x3Variant",
            BindingFlags.NonPublic);
        if (variantField is null || variantEnum is null)
            throw new InvalidOperationException("Could not locate SimdConvHelper.ActiveConv3x3Variant via reflection.");

        // Shapes: cover the BDN benchmark + a few representative ResNet/transformer shapes
        var shapes = new (int b, int ic, int h, int w, int oc)[]
        {
            (1, 16, 64, 64, 32),    // BDN benchmark
            (1, 32, 32, 32, 64),    // medium ResNet
            (1, 64, 32, 32, 64),    // wider ResNet
            (1, 64, 16, 16, 128),   // deeper ResNet
            (4, 16, 64, 64, 32),    // batched
        };

        foreach (var (b, ic, h, w, oc) in shapes)
        {
            Console.WriteLine($"--- shape: input=[{b},{ic},{h},{w}] kernel=[{oc},{ic},3,3] ---");
            var (input, kernel) = MakeTensors(b, ic, h, w, oc, seed: 42);
            var engine = new CpuEngine();

            var results = new (string name, double meanUs, double minUs, double stddevUs)[3];
            string[] variants = { "PerChannel", "Block2", "Block4" };
            for (int v = 0; v < variants.Length; v++)
            {
                object enumValue = Enum.Parse(variantEnum, variants[v]);
                variantField.SetValue(null, enumValue);
                results[v] = TimeIt(engine, input, kernel, label: variants[v]);
            }

            Console.WriteLine($"  {"Variant",-12} {"Mean (µs)",10} {"Min (µs)",10} {"StdDev (µs)",12}");
            foreach (var r in results)
                Console.WriteLine($"  {r.name,-12} {r.meanUs,10:F1} {r.minUs,10:F1} {r.stddevUs,12:F1}");
            var fastest = results.OrderBy(r => r.minUs).First();
            Console.WriteLine($"  -> fastest by min: {fastest.name} @ {fastest.minUs:F1} µs");
            Console.WriteLine();
        }

        // Restore default
        variantField.SetValue(null, Enum.Parse(variantEnum, "Auto"));
    }

    private static (Tensor<float> input, Tensor<float> kernel) MakeTensors(
        int b, int ic, int h, int w, int oc, int seed)
    {
        var rng = new Random(seed);
        var inputData = new float[b * ic * h * w];
        var kernelData = new float[oc * ic * 9];
        for (int i = 0; i < inputData.Length; i++)  inputData[i]  = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernelData.Length; i++) kernelData[i] = (float)(rng.NextDouble() * 2 - 1);
        return (new Tensor<float>(inputData, new[] { b, ic, h, w }),
                new Tensor<float>(kernelData, new[] { oc, ic, 3, 3 }));
    }

    private static (string name, double meanUs, double minUs, double stddevUs)
        TimeIt(CpuEngine engine, Tensor<float> input, Tensor<float> kernel, string label)
    {
        const int warmupIters = 20;
        const int measureIters = 100;

        // Warmup — JIT, populate caches, etc.
        for (int i = 0; i < warmupIters; i++)
        {
            var r = engine.Conv2D(input, kernel, 1, 1, 1);
            // Drop reference so AutoTensorCache can recycle. No explicit return
            // since we want the same alloc behavior the public API has.
            _ = r;
        }

        // Measure
        var samples = new double[measureIters];
        var sw = new Stopwatch();
        for (int i = 0; i < measureIters; i++)
        {
            sw.Restart();
            var r = engine.Conv2D(input, kernel, 1, 1, 1);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMicroseconds;
            _ = r;
        }

        double mean = samples.Average();
        double min = samples.Min();
        double sumSq = 0;
        foreach (var s in samples) sumSq += (s - mean) * (s - mean);
        double stddev = Math.Sqrt(sumSq / samples.Length);
        return (label, mean, min, stddev);
    }
}
