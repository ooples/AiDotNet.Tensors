using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
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
    public static void RunBinaryOps()
    {
        Console.WriteLine("=== Binary tensor ops (Add/Subtract/Multiply/Divide) micro-bench ===");
        var engine = new CpuEngine();
        var sizes = new[] { 100_000, 1_000_000 };
        foreach (var n in sizes)
        {
            var rng = new Random(42);
            var aData = new float[n]; var bData = new float[n];
            for (int i = 0; i < n; i++) { aData[i] = (float)(rng.NextDouble() * 2 - 1); bData[i] = (float)(rng.NextDouble() * 2 - 1) + 0.5f; }
            var a = new Tensor<float>(aData, new[] { n });
            var b = new Tensor<float>(bData, new[] { n });

            var ops = new (string name, Func<Tensor<float>> op)[]
            {
                ("TensorAdd     ", () => engine.TensorAdd(a, b)),
                ("TensorSubtract", () => engine.TensorSubtract(a, b)),
                ("TensorMultiply", () => engine.TensorMultiply(a, b)),
                ("TensorDivide  ", () => engine.TensorDivide(a, b)),
            };
            Console.WriteLine($"--- size = {n:N0} ---");
            foreach (var (name, op) in ops)
            {
                // Return result tensors so the binary-op micro-bench
                // measures the kernel cost, not allocator pressure.
                // Same fix-pattern applied to RunMatMul/RunLayerNorm/etc.
                for (int i = 0; i < 10; i++)
                {
                    var rWarm = op();
                    TensorPool.Return(rWarm);
                }
                const int iters = 50;
                var samples = new double[iters];
                var sw = new Stopwatch();
                for (int i = 0; i < iters; i++)
                {
                    sw.Restart();
                    var rMeas = op();
                    sw.Stop();
                    samples[i] = sw.Elapsed.TotalMicroseconds;
                    TensorPool.Return(rMeas);
                }
                double mean = samples.Average();
                double min = samples.Min();
                Console.WriteLine($"  {name}:  mean={mean,8:F1} µs   min={min,8:F1} µs");
            }
            Console.WriteLine();
        }
    }

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

            // Return Conv2D output to TensorPool — same allocator-warm
            // rationale as RunMatMul/RunLayerNorm above.
            for (int i = 0; i < 10; i++)
            {
                var rWarm = engine.Conv2D(input, kernel, 1, 1, 1);
                TensorPool.Return(rWarm);
            }
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var rMeas = engine.Conv2D(input, kernel, 1, 1, 1);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
                TensorPool.Return(rMeas);
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

            // Return result tensors to TensorPool inside warmup + measure
            // loops so the micro-bench measures kernel time, not allocator
            // pressure. Without this the per-iter `engine.TensorMatMul`
            // allocates a fresh output tensor every call and timings get
            // dominated by GC churn — disagreeing with the BDN-side
            // benchmarks that run with the pool warm.
            for (int i = 0; i < 10; i++)
            {
                var rWarm = engine.TensorMatMul(a, b);
                TensorPool.Return(rWarm);
            }
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var rMeas = engine.TensorMatMul(a, b);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
                TensorPool.Return(rMeas);
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

            // Return LayerNorm output + per-row mean/var to TensorPool
            // — same rationale as RunMatMul above. LayerNorm allocates
            // three result tensors; without pooling the inner loop
            // becomes allocation-bound rather than kernel-bound.
            for (int i = 0; i < 10; i++)
            {
                var rWarm = engine.LayerNorm(input, gamma, beta, 1e-5, out var meanWarm, out var varWarm);
                TensorPool.Return(rWarm);
                TensorPool.Return(meanWarm);
                TensorPool.Return(varWarm);
            }
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var rMeas = engine.LayerNorm(input, gamma, beta, 1e-5, out var meanOut, out var varOut);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
                TensorPool.Return(rMeas);
                TensorPool.Return(meanOut);
                TensorPool.Return(varOut);
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

            // Return attention output to TensorPool — same allocator-warm
            // rationale as RunMatMul.
            for (int i = 0; i < 10; i++)
            {
                var rWarm = engine.TensorMatMulTransposed(q, k);
                TensorPool.Return(rWarm);
            }
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var rMeas = engine.TensorMatMulTransposed(q, k);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
                TensorPool.Return(rMeas);
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

            // Warmup — return softmax output so the pool stays warm.
            for (int i = 0; i < 10; i++)
            {
                var rWarm = engine.Softmax(input, axis: -1);
                TensorPool.Return(rWarm);
            }

            // Measure
            const int iters = 50;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                var rMeas = engine.Softmax(input, axis: -1);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
                TensorPool.Return(rMeas);
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
        // Pool-return each result so the ArrayPool stays warm across iterations
        // (matches the BDN benchmarks' apples-to-apples allocator pattern).
        for (int i = 0; i < warmupIters; i++)
        {
            var r = engine.Conv2D(input, kernel, 1, 1, 1);
            AiDotNet.Tensors.Helpers.TensorPool.Return(r);
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
            AiDotNet.Tensors.Helpers.TensorPool.Return(r);
        }

        double mean = samples.Average();
        double min = samples.Min();
        double sumSq = 0;
        foreach (var s in samples) sumSq += (s - mean) * (s - mean);
        double stddev = Math.Sqrt(sumSq / samples.Length);
        return (label, mean, min, stddev);
    }

    /// <summary>
    /// Sub-G readiness diagnostic: identify why BlasManaged is 16× slower than
    /// OpenBLAS on 64×64×64 FP64 (the worst remaining loss in the 2026-05-26
    /// baseline). Tries every PackingMode + thread count combination and reports
    /// the min wall-clock so we can see whether the bottleneck is:
    ///   (a) strategy selection — wrong PackingMode for this shape,
    ///   (b) parallelism — serial dispatch when parallel would win,
    ///   (c) microkernel — even the best-case path is slow.
    /// Run with: <c>dotnet run -c Release -- --ab-blas-small-square-fp64</c>
    /// </summary>
    public static void RunBlasSmallSquareFp64()
    {
        Console.WriteLine("=== BlasManaged 64×64×64 FP64 strategy A/B (Sub-G worst-loss target) ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}, AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported}, AVX-512F={System.Runtime.Intrinsics.X86.Avx512F.IsSupported}");
        Console.WriteLine();

        // Also probe nearby shapes so we don't regress them when adjusting the
        // dispatcher threshold:
        //   - 96×128×64: BERT_Attn_score (k=64 — same K, larger m×n)
        //   - 96×64×128: BERT_Attn_ctx   (k=128 — PackBoth gate boundary)
        //   - 128×128×128: square cube above the proposed cutoff
        var shapesToProbe = new (int M, int N, int K, string note, bool fp32)[]
        {
            // FP64 cubes (verify prior fix sticks)
            (64,   64,   64,  "64³ FP64 — was worst-loss (16.1× behind OpenBLAS)",       false),
            (96,   128,  64,  "BERT_Attn_score FP64 — sanity (PackAOnly still wins)",     false),
            (128,  128,  128, "128³ FP64 — above Streaming cutoff",                       false),

            // Remaining Sub-G worst-loss shapes from refreshed baseline:
            (3136, 64,   64,  "ResNet50_layer1 FP32 — thin-N 9.1× regression",            true),
            (3136, 32,   32,  "MobileNetV2_pw FP32 — thin-N 9.7× regression",             true),
            (512,  512,  64,  "Instrumented 512x512x64 FP64 — thin-K 6.8× regression",    false),
            (1024, 3072, 768, "BERT_FFN_up FP32 — compute-bound 10.7× regression (big)",  true),

            // FP32 sanity probes
            (64,   64,   64,  "64³ FP32 — verify FP32 fix sticks",                        true),
        };

        foreach (var (M_, N_, K_, note, fp32) in shapesToProbe)
        {
            if (fp32) ProbeStrategiesFp32(M_, N_, K_, note);
            else ProbeStrategies(M_, N_, K_, note);
            Console.WriteLine();
        }
        return;
    }

    private static void ProbeStrategies(int M, int N, int K, string note)
    {
        Console.WriteLine($"=== {M}×{N}×{K} FP64  ({note}) ===");
        var rng = new Random(42);
        var a = new double[M * K];
        var b = new double[K * N];
        var c = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double fmas = 2.0 * M * N * K;

        var modes = new (string label, AiDotNet.Tensors.Engines.BlasManaged.PackingMode mode)[]
        {
            ("Auto (default)",   AiDotNet.Tensors.Engines.BlasManaged.PackingMode.Auto),
            ("ForceStreaming",   AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForceStreaming),
            ("ForcePackAOnly",   AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForcePackAOnly),
            ("ForcePackBoth",    AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForcePackBoth),
        };
        var threadCounts = new[] { -1, 1, 2, 4, 8, 16 };  // -1 = explicit single-thread

        Console.WriteLine($"{"Strategy",-22}  {"Threads",8}  {"min μs",8}  {"GFLOPS",8}");
        Console.WriteLine(new string('-', 60));
        foreach (var (label, mode) in modes)
        {
            foreach (var nt in threadCounts)
            {
                // Skip combos that don't make sense.
                if (mode == AiDotNet.Tensors.Engines.BlasManaged.PackingMode.Auto && nt > 0)
                    continue;  // Auto: just measure once with default thread count.

                var opts = new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<double>
                {
                    PackingMode = mode,
                    NumThreads = nt,
                };

                // Warmup.
                for (int i = 0; i < 5; i++)
                    AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
                        a, K, false, b, N, false, c, N, M, N, K, opts);

                // GC settle.
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
                GC.WaitForPendingFinalizers();
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);

                const int iters = 200;
                var samples = new double[iters];
                var sw = new Stopwatch();
                for (int i = 0; i < iters; i++)
                {
                    sw.Restart();
                    AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
                        a, K, false, b, N, false, c, N, M, N, K, opts);
                    sw.Stop();
                    samples[i] = sw.Elapsed.TotalMicroseconds;
                }
                double min = samples.Min();
                double gflops = fmas / (min * 1e-6) / 1e9;
                string ntLabel = nt < 0 ? "default" : nt.ToString();
                Console.WriteLine($"{label,-22}  {ntLabel,8}  {min,8:F2}  {gflops,8:F2}");

                if (mode == AiDotNet.Tensors.Engines.BlasManaged.PackingMode.Auto) break;
            }
        }
        Console.WriteLine();
    }

    private static void ProbeStrategiesFp32(int M, int N, int K, string note)
    {
        Console.WriteLine($"=== {M}×{N}×{K} FP32  ({note}) ===");
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        double fmas = 2.0 * M * N * K;

        var modes = new (string label, AiDotNet.Tensors.Engines.BlasManaged.PackingMode mode)[]
        {
            ("Auto (default)",   AiDotNet.Tensors.Engines.BlasManaged.PackingMode.Auto),
            ("ForceStreaming",   AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForceStreaming),
            ("ForcePackAOnly",   AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForcePackAOnly),
            ("ForcePackBoth",    AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForcePackBoth),
        };

        Console.WriteLine($"{"Strategy",-22}  {"min μs",8}  {"GFLOPS",8}");
        Console.WriteLine(new string('-', 45));
        foreach (var (label, mode) in modes)
        {
            var opts = new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float>
            {
                PackingMode = mode,
            };
            for (int i = 0; i < 5; i++)
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
                    a, K, false, b, N, false, c, N, M, N, K, opts);
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);

            const int iters = 200;
            var samples = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
                    a, K, false, b, N, false, c, N, M, N, K, opts);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMicroseconds;
            }
            double min = samples.Min();
            double gflops = fmas / (min * 1e-6) / 1e9;
            Console.WriteLine($"{label,-22}  {min,8:F2}  {gflops,8:F2}");
        }
    }
}
