using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// A single benchmark measurement capturing operation performance and correctness.
/// </summary>
public readonly record struct BenchmarkResult(
    string Phase,
    string Operation,
    int Size,
    double GFlops,
    double MaxError,
    double AvgError,
    double TimeMs,
    DateTime Timestamp);

/// <summary>
/// Captures GPU benchmark baselines to CSV for A/B comparison across optimization phases.
/// Each phase records (phase, op, size, gflops, maxError, avgError, timeMs, timestamp).
/// </summary>
public static class GpuBaselineResults
{
    private const string DefaultCsvPath = "gpu_baseline_results.csv";

    /// <summary>
    /// Runs the full baseline capture suite and writes results to CSV.
    /// </summary>
    public static void CaptureBaseline(string phase = "phase0", string? csvPath = null)
    {
        csvPath ??= DefaultCsvPath;
        var results = new List<BenchmarkResult>();

        Console.WriteLine("===========================================");
        Console.WriteLine($"GPU BASELINE CAPTURE — {phase}");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        DirectGpuEngine? engine = null;
        try
        {
            engine = new DirectGpuEngine();
            if (!engine.IsAvailable)
            {
                Console.WriteLine("[SKIP] DirectGpu not available.");
                return;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            return;
        }

        Console.WriteLine($"Backend: {engine.BackendName}");
        Console.WriteLine($"Device:  {engine.DeviceName}");
        Console.WriteLine($"CUs:     {engine.ComputeUnits}");
        Console.WriteLine($"VRAM:    {engine.GlobalMemoryGB:F1} GB");
        Console.WriteLine();

        var backend = engine.Backend;
        if (backend == null)
        {
            Console.WriteLine("[ERROR] Could not access backend for GPU-resident benchmarks.");
            engine.Dispose();
            return;
        }

        // === GEMM benchmarks ===
        Console.WriteLine("--- GEMM ---");
        int[] gemmSizes = [256, 512, 1024, 2048, 4096];
        foreach (var size in gemmSizes)
        {
            var r = BenchmarkGemm(backend, size, size, size, phase);
            results.Add(r);
            Console.WriteLine($"  GEMM {size}x{size}: {r.GFlops:F1} GFLOPS, {r.TimeMs:F2} ms, maxErr={r.MaxError:E2}");
        }

        // === Activation benchmarks ===
        Console.WriteLine("--- Activations ---");
        int[] actSizes = [1024, 16384, 262144, 1048576];
        string[] actOps = ["relu", "sigmoid", "tanh", "gelu", "softmax"];
        foreach (var actOp in actOps)
        {
            foreach (var size in actSizes)
            {
                var r = BenchmarkActivation(backend, actOp, size, phase);
                results.Add(r);
                Console.WriteLine($"  {actOp} N={size}: {r.GFlops:F1} GFLOPS, {r.TimeMs:F4} ms, maxErr={r.MaxError:E2}");
            }
        }

        // === Normalization benchmarks ===
        Console.WriteLine("--- Normalization ---");
        string[] normOps = ["batchnorm", "layernorm", "rmsnorm"];
        foreach (var normOp in normOps)
        {
            var r = BenchmarkNormalization(backend, normOp, phase);
            results.Add(r);
            Console.WriteLine($"  {normOp}: {r.GFlops:F1} GFLOPS, {r.TimeMs:F4} ms, maxErr={r.MaxError:E2}");
        }

        // === Attention benchmarks ===
        Console.WriteLine("--- Attention ---");
        int[] seqLens = [128, 256, 512];
        foreach (var seqLen in seqLens)
        {
            var r = BenchmarkAttention(backend, seqLen, phase);
            results.Add(r);
            Console.WriteLine($"  FlashAttn seq={seqLen}: {r.GFlops:F1} GFLOPS, {r.TimeMs:F2} ms, maxErr={r.MaxError:E2}");
        }

        // === Conv2D benchmarks ===
        Console.WriteLine("--- Conv2D ---");
        var r2 = BenchmarkConv2D(backend, phase);
        results.Add(r2);
        Console.WriteLine($"  Conv2D N=4,C=64,H=56,K=3: {r2.GFlops:F1} GFLOPS, {r2.TimeMs:F2} ms, maxErr={r2.MaxError:E2}");

        // Write CSV
        WriteCsv(results, csvPath, appendIfExists: true);
        Console.WriteLine();
        Console.WriteLine($"Results written to: {Path.GetFullPath(csvPath)}");
        Console.WriteLine($"Total measurements: {results.Count}");

        engine.Dispose();
    }

    private static BenchmarkResult BenchmarkGemm(IDirectGpuBackend backend, int M, int N, int K, string phase)
    {
        var rand = new Random(42);
        var A = CreateRandomArray(M * K, rand);
        var B = CreateRandomArray(K * N, rand);

        using var bufA = backend.AllocateBuffer(A);
        using var bufB = backend.AllocateBuffer(B);
        using var bufC = backend.AllocateBuffer(M * N);

        // Warmup
        for (int i = 0; i < 3; i++)
            backend.Gemm(bufA, bufB, bufC, M, N, K, 1.0f, 0.0f);
        backend.Synchronize();

        // Benchmark
        int runs = 10;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.Gemm(bufA, bufB, bufC, M, N, K, 1.0f, 0.0f);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double flops = 2.0 * M * N * K;
        double gflops = flops / (avgMs * 1e6);

        // Correctness: compute CPU reference for small sizes
        double maxError = 0, avgError = 0;
        if (M * N <= 1048576)
        {
            var gpuResult = new float[M * N];
            backend.DownloadBuffer(bufC, gpuResult);
            (maxError, avgError) = ComputeGemmError(A, B, gpuResult, M, N, K);
        }

        return new BenchmarkResult(phase, $"gemm_{M}x{N}x{K}", M, gflops, maxError, avgError, avgMs, DateTime.UtcNow);
    }

    private static BenchmarkResult BenchmarkActivation(IDirectGpuBackend backend, string op, int size, string phase)
    {
        var rand = new Random(42);
        var input = CreateRandomArray(size, rand);

        using var bufIn = backend.AllocateBuffer(input);
        using var bufOut = backend.AllocateBuffer(size);

        // Warmup
        for (int i = 0; i < 5; i++)
            DispatchActivation(backend, op, bufIn, bufOut, size);
        backend.Synchronize();

        // Benchmark
        int runs = 50;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            DispatchActivation(backend, op, bufIn, bufOut, size);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double gflops = size / (avgMs * 1e6);

        // Correctness
        var gpuResult = new float[size];
        backend.DownloadBuffer(bufOut, gpuResult);
        var cpuResult = ComputeActivationCpu(input, op);
        var (maxError, avgError) = CompareArrays(cpuResult, gpuResult);

        return new BenchmarkResult(phase, $"{op}_N{size}", size, gflops, maxError, avgError, avgMs, DateTime.UtcNow);
    }

    private static BenchmarkResult BenchmarkNormalization(IDirectGpuBackend backend, string op, string phase)
    {
        int batch = 32, channels = 512, spatial = 49;
        int totalSize = batch * channels * spatial;
        var rand = new Random(42);

        var input = CreateRandomArray(totalSize, rand);
        var gamma = CreateOnesArray(channels);
        var beta = new float[channels];

        using var bufIn = backend.AllocateBuffer(input);
        using var bufOut = backend.AllocateBuffer(totalSize);
        using var bufGamma = backend.AllocateBuffer(gamma);
        using var bufBeta = backend.AllocateBuffer(beta);

        if (op == "batchnorm")
        {
            var runMean = new float[channels];
            var runVar = CreateOnesArray(channels);
            using var bufRunMean = backend.AllocateBuffer(runMean);
            using var bufRunVar = backend.AllocateBuffer(runVar);
            using var bufSaveMean = backend.AllocateBuffer(channels);
            using var bufSaveVar = backend.AllocateBuffer(channels);

            for (int i = 0; i < 3; i++)
                backend.BatchNorm(bufIn, bufOut, bufGamma, bufBeta, bufRunMean, bufRunVar, bufSaveMean, bufSaveVar,
                    batch, channels, spatial, 1e-5f, 0.1f, false);
            backend.Synchronize();

            int runs = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
                backend.BatchNorm(bufIn, bufOut, bufGamma, bufBeta, bufRunMean, bufRunVar, bufSaveMean, bufSaveVar,
                    batch, channels, spatial, 1e-5f, 0.1f, false);
            backend.Synchronize();
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / runs;
            double gflops = totalSize * 5.0 / (avgMs * 1e6);
            return new BenchmarkResult(phase, $"batchnorm_B{batch}_C{channels}_S{spatial}", totalSize, gflops, 0, 0, avgMs, DateTime.UtcNow);
        }

        if (op == "layernorm")
        {
            int normalizedSize = channels * spatial;
            var lnGamma = CreateOnesArray(normalizedSize);
            var lnBeta = new float[normalizedSize];
            using var bufLnGamma = backend.AllocateBuffer(lnGamma);
            using var bufLnBeta = backend.AllocateBuffer(lnBeta);
            using var bufLnMean = backend.AllocateBuffer(batch);
            using var bufLnVar = backend.AllocateBuffer(batch);

            for (int i = 0; i < 3; i++)
                backend.LayerNorm(bufIn, bufOut, bufLnGamma, bufLnBeta, bufLnMean, bufLnVar,
                    batch, normalizedSize, 1e-5f);
            backend.Synchronize();

            int runs = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
                backend.LayerNorm(bufIn, bufOut, bufLnGamma, bufLnBeta, bufLnMean, bufLnVar,
                    batch, normalizedSize, 1e-5f);
            backend.Synchronize();
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / runs;
            double gflops = totalSize * 5.0 / (avgMs * 1e6);
            return new BenchmarkResult(phase, $"layernorm_B{batch}_N{normalizedSize}", totalSize, gflops, 0, 0, avgMs, DateTime.UtcNow);
        }

        // rmsnorm
        {
            int normalizedSize = channels * spatial;
            var rmsGamma = CreateOnesArray(normalizedSize);
            using var bufRmsGamma = backend.AllocateBuffer(rmsGamma);
            using var bufRms = backend.AllocateBuffer(batch);

            for (int i = 0; i < 3; i++)
                backend.RmsNorm(bufIn, bufOut, bufRmsGamma, bufRms, batch, normalizedSize, 1e-5f);
            backend.Synchronize();

            int runs = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
                backend.RmsNorm(bufIn, bufOut, bufRmsGamma, bufRms, batch, normalizedSize, 1e-5f);
            backend.Synchronize();
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / runs;
            double gflops = totalSize * 4.0 / (avgMs * 1e6);
            return new BenchmarkResult(phase, $"rmsnorm_B{batch}_N{normalizedSize}", totalSize, gflops, 0, 0, avgMs, DateTime.UtcNow);
        }
    }

    private static BenchmarkResult BenchmarkAttention(IDirectGpuBackend backend, int seqLen, string phase)
    {
        int batch = 2, numHeads = 8, headDim = 64;
        int qkvSize = batch * numHeads * seqLen * headDim;

        var rand = new Random(42);
        var query = CreateRandomArray(qkvSize, rand, scale: 0.1f);
        var key = CreateRandomArray(qkvSize, rand, scale: 0.1f);
        var value = CreateRandomArray(qkvSize, rand, scale: 0.1f);

        using var bufQ = backend.AllocateBuffer(query);
        using var bufK = backend.AllocateBuffer(key);
        using var bufV = backend.AllocateBuffer(value);
        using var bufOut = backend.AllocateBuffer(qkvSize);

        float scale = 1.0f / MathF.Sqrt(headDim);

        for (int i = 0; i < 3; i++)
            backend.FlashAttention(bufQ, bufK, bufV, bufOut, null, batch, numHeads, seqLen, headDim, scale, false);
        backend.Synchronize();

        int runs = 10;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.FlashAttention(bufQ, bufK, bufV, bufOut, null, batch, numHeads, seqLen, headDim, scale, false);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double flops = 4.0 * batch * numHeads * seqLen * seqLen * headDim;
        double gflops = flops / (avgMs * 1e6);

        return new BenchmarkResult(phase, $"flashattn_seq{seqLen}_h{numHeads}_d{headDim}", seqLen, gflops, 0, 0, avgMs, DateTime.UtcNow);
    }

    private static BenchmarkResult BenchmarkConv2D(IDirectGpuBackend backend, string phase)
    {
        int N = 4, C = 64, H = 56, W = 56;
        int outC = 64, kH = 3, kW = 3;
        int outH = H - kH + 1;
        int outW = W - kW + 1;

        int inputSize = N * C * H * W;
        int kernelSize = outC * C * kH * kW;
        int outputSize = N * outC * outH * outW;

        var rand = new Random(42);
        var input = CreateRandomArray(inputSize, rand);
        var kernel = CreateRandomArray(kernelSize, rand, scale: 0.1f);

        using var bufIn = backend.AllocateBuffer(input);
        using var bufKernel = backend.AllocateBuffer(kernel);
        using var bufOut = backend.AllocateBuffer(outputSize);

        for (int i = 0; i < 3; i++)
            backend.Conv2D(bufIn, bufKernel, bufOut, N, C, H, W, outC, outH, outW, kH, kW, 1, 1, 0, 0, 1, 1);
        backend.Synchronize();

        int runs = 10;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.Conv2D(bufIn, bufKernel, bufOut, N, C, H, W, outC, outH, outW, kH, kW, 1, 1, 0, 0, 1, 1);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double flops = 2.0 * N * outC * outH * outW * C * kH * kW;
        double gflops = flops / (avgMs * 1e6);

        return new BenchmarkResult(phase, $"conv2d_N{N}_C{C}_H{H}_K{kH}", inputSize, gflops, 0, 0, avgMs, DateTime.UtcNow);
    }

    #region Helpers

    private static float[] CreateRandomArray(int size, Random rand, float scale = 1.0f)
    {
        var arr = new float[size];
        for (int i = 0; i < size; i++)
            arr[i] = (float)(rand.NextDouble() - 0.5) * 2 * scale;
        return arr;
    }

    private static float[] CreateOnesArray(int size)
    {
        var arr = new float[size];
        Array.Fill(arr, 1.0f);
        return arr;
    }

    private static void DispatchActivation(IDirectGpuBackend backend, string op, IGpuBuffer bufIn, IGpuBuffer bufOut, int size)
    {
        switch (op)
        {
            case "relu": backend.Relu(bufIn, bufOut, size); break;
            case "sigmoid": backend.Sigmoid(bufIn, bufOut, size); break;
            case "tanh": backend.Tanh(bufIn, bufOut, size); break;
            case "gelu": backend.Gelu(bufIn, bufOut, size); break;
            case "softmax": backend.Softmax(bufIn, bufOut, 1, size); break;
        }
    }

    private static float[] ComputeActivationCpu(float[] input, string op)
    {
        var result = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = op switch
            {
                "relu" => MathF.Max(0, input[i]),
                "sigmoid" => 1.0f / (1.0f + MathF.Exp(-input[i])),
                "tanh" => MathF.Tanh(input[i]),
                "gelu" => 0.5f * input[i] * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (input[i] + 0.044715f * input[i] * input[i] * input[i]))),
                _ => input[i]
            };
        }

        if (op == "softmax")
        {
            float max = float.MinValue;
            for (int i = 0; i < input.Length; i++)
                if (input[i] > max) max = input[i];

            float sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = MathF.Exp(input[i] - max);
                sum += result[i];
            }

            for (int i = 0; i < input.Length; i++)
                result[i] /= sum;
        }

        return result;
    }

    private static (double maxError, double avgError) ComputeGemmError(float[] A, float[] B, float[] gpuC, int M, int N, int K)
    {
        double maxError = 0, sumError = 0;
        int count = 0;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double cpuVal = 0;
                for (int k = 0; k < K; k++)
                    cpuVal += (double)A[i * K + k] * B[k * N + j];

                double error = Math.Abs(cpuVal - gpuC[i * N + j]);
                if (error > maxError) maxError = error;
                sumError += error;
                count++;
            }
        }
        return (maxError, count > 0 ? sumError / count : 0);
    }

    private static (double maxError, double avgError) CompareArrays(float[] expected, float[] actual)
    {
        double maxError = 0, sumError = 0;
        int count = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(actual[i]) || float.IsInfinity(actual[i])) continue;
            double error = Math.Abs(expected[i] - actual[i]);
            if (error > maxError) maxError = error;
            sumError += error;
            count++;
        }
        return (maxError, count > 0 ? sumError / count : 0);
    }

    private static void WriteCsv(List<BenchmarkResult> results, string csvPath, bool appendIfExists)
    {
        bool writeHeader = !appendIfExists || !File.Exists(csvPath);

        using var writer = new StreamWriter(csvPath, append: appendIfExists);
        if (writeHeader)
            writer.WriteLine("phase,operation,size,gflops,max_error,avg_error,time_ms,timestamp");

        foreach (var r in results)
        {
            writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0},{1},{2},{3:F4},{4:E6},{5:E6},{6:F6},{7:O}",
                r.Phase, r.Operation, r.Size, r.GFlops, r.MaxError, r.AvgError, r.TimeMs, r.Timestamp));
        }
    }

    /// <summary>
    /// Compares two CSV baseline files and prints regression/improvement report.
    /// </summary>
    public static void CompareBaselines(string beforeCsv, string afterCsv)
    {
        if (!File.Exists(beforeCsv) || !File.Exists(afterCsv))
        {
            Console.WriteLine($"[ERROR] Missing CSV file: {(File.Exists(beforeCsv) ? afterCsv : beforeCsv)}");
            return;
        }

        var before = ParseCsv(beforeCsv);
        var after = ParseCsv(afterCsv);

        Console.WriteLine("===========================================");
        Console.WriteLine("A/B COMPARISON REPORT");
        Console.WriteLine("===========================================");
        Console.WriteLine();
        Console.WriteLine($"{"Operation",-40} {"Before GFLOPS",14} {"After GFLOPS",13} {"Speedup",10} {"Status",12}");
        Console.WriteLine(new string('-', 91));

        bool hasRegression = false;
        foreach (var (op, beforeResult) in before)
        {
            if (!after.TryGetValue(op, out var afterResult)) continue;

            double speedup = afterResult.GFlops / Math.Max(beforeResult.GFlops, 0.001);
            string status = speedup >= 1.0 ? "OK" : "REGRESSION";
            if (speedup < 1.0) hasRegression = true;

            Console.WriteLine($"{op,-40} {beforeResult.GFlops,14:F1} {afterResult.GFlops,13:F1} {speedup,9:F2}x {status,12}");
        }

        Console.WriteLine();
        if (hasRegression)
            Console.WriteLine("[FAIL] Regressions detected — investigate before merging.");
        else
            Console.WriteLine("[PASS] No regressions detected.");
    }

    private static Dictionary<string, BenchmarkResult> ParseCsv(string csvPath)
    {
        var results = new Dictionary<string, BenchmarkResult>();
        var lines = File.ReadAllLines(csvPath);
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            if (parts.Length < 7) continue;

            var r = new BenchmarkResult(
                parts[0],
                parts[1],
                int.TryParse(parts[2], out var size) ? size : 0,
                double.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out var gf) ? gf : 0,
                double.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out var me) ? me : 0,
                double.TryParse(parts[5], NumberStyles.Float, CultureInfo.InvariantCulture, out var ae) ? ae : 0,
                double.TryParse(parts[6], NumberStyles.Float, CultureInfo.InvariantCulture, out var tm) ? tm : 0,
                DateTime.UtcNow);

            results[r.Operation] = r;
        }
        return results;
    }

    #endregion
}
