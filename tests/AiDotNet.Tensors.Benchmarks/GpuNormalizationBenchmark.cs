using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks GPU normalization kernels: BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RmsNorm.
/// Default config: batch=32, channels=512, spatial=49 (7x7 feature map).
/// </summary>
public static class GpuNormalizationBenchmark
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("GPU NORMALIZATION BENCHMARK");
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

            Console.WriteLine($"{"Operation",-20} {"Config",-30} {"Time(ms)",10} {"GFLOPS",10} {"Elements",12}");
            Console.WriteLine(new string('-', 84));

            var configs = new (int batch, int channels, int spatial)[]
            {
                (32, 512, 49),   // Standard ResNet-like
                (16, 256, 196),  // Larger spatial
                (64, 128, 49),   // Larger batch
            };

            foreach (var (batch, channels, spatial) in configs)
            {
                BenchmarkBatchNorm(backend, batch, channels, spatial);
                BenchmarkLayerNorm(backend, batch, channels, spatial);
                BenchmarkGroupNorm(backend, batch, channels, spatial, numGroups: 32);
                BenchmarkInstanceNorm(backend, batch, channels, spatial);
                BenchmarkRmsNorm(backend, batch, channels, spatial);
                Console.WriteLine();
            }
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

    private static void BenchmarkBatchNorm(IDirectGpuBackend backend, int batch, int channels, int spatial)
    {
        int totalSize = batch * channels * spatial;
        var rand = new Random(42);

        var input = CreateRandom(totalSize, rand);
        var gamma = CreateOnes(channels);
        var beta = new float[channels];
        var runMean = new float[channels];
        var runVar = CreateOnes(channels);

        using var bufIn = backend.AllocateBuffer(input);
        using var bufOut = backend.AllocateBuffer(totalSize);
        using var bufGamma = backend.AllocateBuffer(gamma);
        using var bufBeta = backend.AllocateBuffer(beta);
        using var bufRunMean = backend.AllocateBuffer(runMean);
        using var bufRunVar = backend.AllocateBuffer(runVar);
        using var bufSaveMean = backend.AllocateBuffer(channels);
        using var bufSaveVar = backend.AllocateBuffer(channels);

        // Warmup
        for (int i = 0; i < 5; i++)
            backend.BatchNorm(bufIn, bufOut, bufGamma, bufBeta, bufRunMean, bufRunVar, bufSaveMean, bufSaveVar,
                batch, channels, spatial, 1e-5f, 0.1f, false);
        backend.Synchronize();

        int runs = 50;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.BatchNorm(bufIn, bufOut, bufGamma, bufBeta, bufRunMean, bufRunVar, bufSaveMean, bufSaveVar,
                batch, channels, spatial, 1e-5f, 0.1f, false);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double gflops = totalSize * 5.0 / (avgMs * 1e6);
        Console.WriteLine($"{"BatchNorm",-20} {"B=" + batch + ",C=" + channels + ",S=" + spatial,-30} {avgMs,10:F4} {gflops,10:F2} {totalSize,12}");
    }

    private static void BenchmarkLayerNorm(IDirectGpuBackend backend, int batch, int channels, int spatial)
    {
        int normalizedSize = channels * spatial;
        int totalSize = batch * normalizedSize;
        var rand = new Random(42);

        var input = CreateRandom(totalSize, rand);
        var gamma = CreateOnes(normalizedSize);
        var beta = new float[normalizedSize];

        using var bufIn = backend.AllocateBuffer(input);
        using var bufOut = backend.AllocateBuffer(totalSize);
        using var bufGamma = backend.AllocateBuffer(gamma);
        using var bufBeta = backend.AllocateBuffer(beta);
        using var bufMean = backend.AllocateBuffer(batch);
        using var bufVar = backend.AllocateBuffer(batch);

        for (int i = 0; i < 5; i++)
            backend.LayerNorm(bufIn, bufOut, bufGamma, bufBeta, bufMean, bufVar, batch, normalizedSize, 1e-5f);
        backend.Synchronize();

        int runs = 50;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.LayerNorm(bufIn, bufOut, bufGamma, bufBeta, bufMean, bufVar, batch, normalizedSize, 1e-5f);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double gflops = totalSize * 5.0 / (avgMs * 1e6);
        Console.WriteLine($"{"LayerNorm",-20} {"B=" + batch + ",N=" + normalizedSize,-30} {avgMs,10:F4} {gflops,10:F2} {totalSize,12}");
    }

    private static void BenchmarkGroupNorm(IDirectGpuBackend backend, int batch, int channels, int spatial, int numGroups)
    {
        int totalSize = batch * channels * spatial;
        var rand = new Random(42);

        var input = CreateRandom(totalSize, rand);
        var gamma = CreateOnes(channels);
        var beta = new float[channels];
        int statSize = batch * numGroups;

        using var bufIn = backend.AllocateBuffer(input);
        using var bufOut = backend.AllocateBuffer(totalSize);
        using var bufGamma = backend.AllocateBuffer(gamma);
        using var bufBeta = backend.AllocateBuffer(beta);
        using var bufMean = backend.AllocateBuffer(statSize);
        using var bufVar = backend.AllocateBuffer(statSize);

        for (int i = 0; i < 5; i++)
            backend.GroupNorm(bufIn, bufOut, bufGamma, bufBeta, bufMean, bufVar, batch, numGroups, channels, spatial, 1e-5f);
        backend.Synchronize();

        int runs = 50;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.GroupNorm(bufIn, bufOut, bufGamma, bufBeta, bufMean, bufVar, batch, numGroups, channels, spatial, 1e-5f);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double gflops = totalSize * 5.0 / (avgMs * 1e6);
        Console.WriteLine($"{"GroupNorm(G=" + numGroups + ")",-20} {"B=" + batch + ",C=" + channels + ",S=" + spatial,-30} {avgMs,10:F4} {gflops,10:F2} {totalSize,12}");
    }

    private static void BenchmarkInstanceNorm(IDirectGpuBackend backend, int batch, int channels, int spatial)
    {
        int totalSize = batch * channels * spatial;
        int statSize = batch * channels;
        var rand = new Random(42);

        var input = CreateRandom(totalSize, rand);
        var gamma = CreateOnes(channels);
        var beta = new float[channels];

        using var bufIn = backend.AllocateBuffer(input);
        using var bufOut = backend.AllocateBuffer(totalSize);
        using var bufGamma = backend.AllocateBuffer(gamma);
        using var bufBeta = backend.AllocateBuffer(beta);
        using var bufMean = backend.AllocateBuffer(statSize);
        using var bufVar = backend.AllocateBuffer(statSize);

        for (int i = 0; i < 5; i++)
            backend.InstanceNorm(bufIn, bufOut, bufGamma, bufBeta, bufMean, bufVar, batch, channels, spatial, 1e-5f);
        backend.Synchronize();

        int runs = 50;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.InstanceNorm(bufIn, bufOut, bufGamma, bufBeta, bufMean, bufVar, batch, channels, spatial, 1e-5f);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double gflops = totalSize * 5.0 / (avgMs * 1e6);
        Console.WriteLine($"{"InstanceNorm",-20} {"B=" + batch + ",C=" + channels + ",S=" + spatial,-30} {avgMs,10:F4} {gflops,10:F2} {totalSize,12}");
    }

    private static void BenchmarkRmsNorm(IDirectGpuBackend backend, int batch, int channels, int spatial)
    {
        int normalizedSize = channels * spatial;
        int totalSize = batch * normalizedSize;
        var rand = new Random(42);

        var input = CreateRandom(totalSize, rand);
        var gamma = CreateOnes(normalizedSize);

        using var bufIn = backend.AllocateBuffer(input);
        using var bufOut = backend.AllocateBuffer(totalSize);
        using var bufGamma = backend.AllocateBuffer(gamma);
        using var bufRms = backend.AllocateBuffer(batch);

        for (int i = 0; i < 5; i++)
            backend.RmsNorm(bufIn, bufOut, bufGamma, bufRms, batch, normalizedSize, 1e-5f);
        backend.Synchronize();

        int runs = 50;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            backend.RmsNorm(bufIn, bufOut, bufGamma, bufRms, batch, normalizedSize, 1e-5f);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double gflops = totalSize * 4.0 / (avgMs * 1e6);
        Console.WriteLine($"{"RmsNorm",-20} {"B=" + batch + ",N=" + normalizedSize,-30} {avgMs,10:F4} {gflops,10:F2} {totalSize,12}");
    }

    private static float[] CreateRandom(int size, Random rand)
    {
        var arr = new float[size];
        for (int i = 0; i < size; i++)
            arr[i] = (float)(rand.NextDouble() - 0.5) * 2;
        return arr;
    }

    private static float[] CreateOnes(int size)
    {
        var arr = new float[size];
        Array.Fill(arr, 1.0f);
        return arr;
    }
}
