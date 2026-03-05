using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks GPU attention kernels: FlashAttention, ScaledDotProductAttention.
/// Tests at seq=128, 256, 512, 1024 with batch=2, heads=8, headDim=64.
/// </summary>
public static class GpuAttentionBenchmark
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("GPU ATTENTION BENCHMARK");
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

            Console.WriteLine($"Backend: {engine.BackendName}");
            Console.WriteLine($"Device:  {engine.DeviceName}");
            Console.WriteLine();

            var backend = engine.Backend;
            if (backend == null)
            {
                Console.WriteLine("[ERROR] Could not access backend.");
                return;
            }

            int[] seqLens = [128, 256, 512, 1024];
            int batch = 2, numHeads = 8, headDim = 64;

            Console.WriteLine($"Config: batch={batch}, heads={numHeads}, headDim={headDim}");
            Console.WriteLine();

            // FlashAttention benchmarks
            Console.WriteLine("--- FlashAttention ---");
            Console.WriteLine($"{"SeqLen",-10} {"Time(ms)",10} {"GFLOPS",10} {"TFLOPS",8} {"Bandwidth",10}");
            Console.WriteLine(new string('-', 50));

            foreach (var seqLen in seqLens)
            {
                BenchmarkFlashAttention(backend, batch, numHeads, seqLen, headDim);
            }

            Console.WriteLine();

            // ScaledDotProduct benchmarks
            Console.WriteLine("--- ScaledDotProductAttention ---");
            Console.WriteLine($"{"SeqLen",-10} {"Time(ms)",10} {"GFLOPS",10} {"TFLOPS",8} {"Bandwidth",10}");
            Console.WriteLine(new string('-', 50));

            foreach (var seqLen in seqLens)
            {
                BenchmarkScaledDotProduct(backend, batch, numHeads, seqLen, headDim);
            }

            Console.WriteLine();

            // Causal vs non-causal comparison
            Console.WriteLine("--- Causal vs Non-Causal FlashAttention (seq=512) ---");
            Console.WriteLine($"{"Mode",-15} {"Time(ms)",10} {"GFLOPS",10} {"TFLOPS",8} {"Bandwidth",10}");
            Console.WriteLine(new string('-', 58));
            BenchmarkFlashAttention(backend, batch, numHeads, 512, headDim, isCausal: false, label: "Non-causal");
            BenchmarkFlashAttention(backend, batch, numHeads, 512, headDim, isCausal: true, label: "Causal");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
        }
        finally
        {
            engine?.Dispose();
        }
    }

    private static void BenchmarkFlashAttention(IDirectGpuBackend backend, int batch, int numHeads, int seqLen, int headDim,
        bool isCausal = false, string? label = null)
    {
        int qkvSize = batch * numHeads * seqLen * headDim;
        var rand = new Random(42);

        var query = CreateRandom(qkvSize, rand, 0.1f);
        var key = CreateRandom(qkvSize, rand, 0.1f);
        var value = CreateRandom(qkvSize, rand, 0.1f);

        using var bufQ = backend.AllocateBuffer(query);
        using var bufK = backend.AllocateBuffer(key);
        using var bufV = backend.AllocateBuffer(value);
        using var bufOut = backend.AllocateBuffer(qkvSize);

        float scale = 1.0f / MathF.Sqrt(headDim);

        // Warmup
        for (int i = 0; i < 3; i++)
            backend.FlashAttention(bufQ, bufK, bufV, bufOut, null, batch, numHeads, seqLen, headDim, scale, isCausal);
        backend.Synchronize();

        int runs = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.FlashAttention(bufQ, bufK, bufV, bufOut, null, batch, numHeads, seqLen, headDim, scale, isCausal);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        // FLOPs: 2 * B * H * S^2 * D (QK^T) + 2 * B * H * S^2 * D (Attn*V)
        double flops = 4.0 * batch * numHeads * seqLen * seqLen * headDim;
        double gflops = flops / (avgMs * 1e6);
        double tflops = gflops / 1000;
        // Memory bandwidth: Q + K + V + O read/write
        double memBytes = 4.0 * qkvSize * 4 * 2; // 4 tensors, float32, r+w
        double bandwidth = memBytes / (avgMs * 1e6);

        string displayLabel = label ?? seqLen.ToString();
        Console.WriteLine($"{displayLabel,-15} {avgMs,10:F3} {gflops,10:F1} {tflops,8:F3} {bandwidth,10:F1}");
    }

    private static void BenchmarkScaledDotProduct(IDirectGpuBackend backend, int batch, int numHeads, int seqLen, int headDim)
    {
        int qkvSize = batch * numHeads * seqLen * headDim;
        int attnWeightsSize = batch * numHeads * seqLen * seqLen;
        var rand = new Random(42);

        var query = CreateRandom(qkvSize, rand, 0.1f);
        var key = CreateRandom(qkvSize, rand, 0.1f);
        var value = CreateRandom(qkvSize, rand, 0.1f);

        using var bufQ = backend.AllocateBuffer(query);
        using var bufK = backend.AllocateBuffer(key);
        using var bufV = backend.AllocateBuffer(value);
        using var bufOut = backend.AllocateBuffer(qkvSize);
        using var bufAttnWeights = backend.AllocateBuffer(attnWeightsSize);

        float scale = 1.0f / MathF.Sqrt(headDim);

        // Warmup
        for (int i = 0; i < 3; i++)
            backend.ScaledDotProductAttention(bufQ, bufK, bufV, bufOut, bufAttnWeights, null, batch, numHeads, seqLen, headDim, scale, false);
        backend.Synchronize();

        int runs = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.ScaledDotProductAttention(bufQ, bufK, bufV, bufOut, bufAttnWeights, null, batch, numHeads, seqLen, headDim, scale, false);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double flops = 4.0 * batch * numHeads * seqLen * seqLen * headDim;
        double gflops = flops / (avgMs * 1e6);
        double tflops = gflops / 1000;
        double memBytes = 4.0 * qkvSize * 4 * 2;
        double bandwidth = memBytes / (avgMs * 1e6);

        Console.WriteLine($"{seqLen,-10} {avgMs,10:F3} {gflops,10:F1} {tflops,8:F3} {bandwidth,10:F1}");
    }

    private static float[] CreateRandom(int size, Random rand, float scale)
    {
        var arr = new float[size];
        for (int i = 0; i < size; i++)
            arr[i] = (float)(rand.NextDouble() - 0.5) * 2 * scale;
        return arr;
    }
}
