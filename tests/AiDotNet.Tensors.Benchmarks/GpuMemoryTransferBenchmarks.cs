// Copyright (c) AiDotNet. All rights reserved.
// Benchmarks specifically for GPU memory transfer overhead analysis.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks to analyze GPU memory transfer overhead.
/// Goal: Identify if data transfer is the bottleneck.
/// </summary>
/// <remarks>
/// <para><b>Memory Transfer Bottlenecks:</b></para>
/// <list type="bullet">
/// <item>PCIe bandwidth limitations (typically 16-32 GB/s)</item>
/// <item>Staging buffer allocations</item>
/// <item>Device memory allocations</item>
/// <item>Synchronization overhead</item>
/// </list>
/// </remarks>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class GpuMemoryTransferBenchmarks
{
    private DirectGpuEngine? _directGpuEngine;
    private float[] _inputA = null!;
    private float[] _inputB = null!;
    private bool _gpuAvailable;

    [Params(1024, 16384, 262144, 1048576, 4194304)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _inputA = new float[N];
        _inputB = new float[N];

        for (int i = 0; i < N; i++)
        {
            _inputA[i] = (float)(random.NextDouble() * 2 - 1);
            _inputB[i] = (float)(random.NextDouble() * 2 - 1);
        }

        try
        {
            _directGpuEngine = new DirectGpuEngine();
            _gpuAvailable = _directGpuEngine.IsAvailable;
        }
        catch
        {
            _gpuAvailable = false;
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _directGpuEngine?.Dispose();
    }

    [Benchmark(Description = "MatMul (includes transfer)")]
    public float[]? MatMulWithTransfer()
    {
        if (!_gpuAvailable) return null;
        int dim = (int)Math.Sqrt(N);
        if (dim * dim != N) dim = 1024;
        return _directGpuEngine!.MatMul(_inputA, _inputB, dim, dim, dim);
    }
}

/// <summary>
/// Detailed memory transfer diagnostics with manual timing.
/// </summary>
public static class GpuMemoryDiagnostics
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("GPU MEMORY TRANSFER DIAGNOSTICS");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        // Try DirectGpu engine
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

        Console.WriteLine($"[OK] DirectGpu Engine: {engine.BackendName}");
        Console.WriteLine($"     Device: {engine.DeviceName}");
        Console.WriteLine($"     Vendor: {engine.DeviceVendor}");
        Console.WriteLine($"     Compute Units: {engine.ComputeUnits}");
        Console.WriteLine($"     Global Memory: {engine.GlobalMemoryGB:F2} GB");
        Console.WriteLine();

        var random = RandomHelper.CreateSeededRandom(42);

        // Test various data sizes
        Console.WriteLine("===========================================");
        Console.WriteLine("TRANSFER TIME VS DATA SIZE");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        long[] sizes = { 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216 };

        Console.WriteLine($"{"Size",-12} {"Elements",-12} {"MB",-10} {"MatMul (ms)",12} {"GB/s",10}");
        Console.WriteLine(new string('-', 58));

        foreach (var size in sizes)
        {
            var inputA = new float[size];
            var inputB = new float[size];

            for (int i = 0; i < (int)size; i++)
            {
                inputA[i] = (float)(random.NextDouble() * 2 - 1);
                inputB[i] = (float)(random.NextDouble() * 2 - 1);
            }

            // For matrix multiplication, use square dimensions
            int dim = (int)Math.Sqrt(size);

            try
            {
                // Warmup
                engine.MatMul(inputA, inputB, dim, dim, dim);

                // Benchmark
                int runs = size < 100000 ? 50 : 10;
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < runs; i++)
                {
                    engine.MatMul(inputA, inputB, dim, dim, dim);
                }
                sw.Stop();

                double avgMs = sw.Elapsed.TotalMilliseconds / runs;
                double sizeBytes = size * 4; // float = 4 bytes
                double sizeMb = sizeBytes / (1024 * 1024);
                // Transfer: 2 inputs + 1 output = 3 * size * 4 bytes
                double transferBytes = 3 * sizeBytes;
                double bandwidthGbps = transferBytes / (avgMs * 1e6);

                Console.WriteLine($"{size,-12} {dim}x{dim,-7} {sizeMb,-10:F2} {avgMs,12:F4} {bandwidthGbps,10:F2}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{size,-12} {dim}x{dim,-7} ERROR: {ex.Message.Substring(0, Math.Min(40, ex.Message.Length))}");
            }
        }

        Console.WriteLine();

        // Analyze overhead components
        Console.WriteLine("===========================================");
        Console.WriteLine("OVERHEAD COMPONENT ANALYSIS");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        // Smallest possible operation to measure base overhead
        var smallA = new float[16];
        var smallB = new float[16];

        // Warmup
        for (int i = 0; i < 50; i++)
        {
            engine.MatMul(smallA, smallB, 4, 4, 4);
        }

        // Measure minimum overhead
        int overheadRuns = 500;
        var sw2 = Stopwatch.StartNew();
        for (int i = 0; i < overheadRuns; i++)
        {
            engine.MatMul(smallA, smallB, 4, 4, 4);
        }
        sw2.Stop();
        double minOverheadMs = sw2.Elapsed.TotalMilliseconds / overheadRuns;

        Console.WriteLine($"Minimum Operation Overhead: {minOverheadMs:F4} ms");
        Console.WriteLine($"Maximum Operations/Second:  {1000.0 / minOverheadMs:N0} ops/s");
        Console.WriteLine();

        // Calculate theoretical limits
        Console.WriteLine("THEORETICAL LIMITS");
        Console.WriteLine("------------------");
        Console.WriteLine("PCIe 3.0 x16: ~16 GB/s");
        Console.WriteLine("PCIe 4.0 x16: ~32 GB/s");
        Console.WriteLine("PCIe 5.0 x16: ~64 GB/s");
        Console.WriteLine();

        // Estimate what the bottleneck is
        Console.WriteLine("BOTTLENECK ANALYSIS");
        Console.WriteLine("-------------------");

        // Large transfer to measure bandwidth
        var largeA = new float[4194304]; // 16 MB
        var largeB = new float[4194304];
        for (int i = 0; i < largeA.Length; i++)
        {
            largeA[i] = (float)(random.NextDouble() * 2 - 1);
            largeB[i] = (float)(random.NextDouble() * 2 - 1);
        }

        // Warmup
        engine.MatMul(largeA, largeB, 2048, 2048, 2048);

        sw2.Restart();
        for (int i = 0; i < 5; i++)
        {
            engine.MatMul(largeA, largeB, 2048, 2048, 2048);
        }
        sw2.Stop();
        double largeOpMs = sw2.Elapsed.TotalMilliseconds / 5;

        // Estimate compute time vs transfer time
        double transferTimeEstimate = minOverheadMs; // Base overhead
        double computeTimeEstimate = largeOpMs - transferTimeEstimate;

        // 2048^3 * 2 FLOPs for GEMM
        double flops = 2.0 * 2048 * 2048 * 2048;
        double gflops = flops / (computeTimeEstimate * 1e6);

        Console.WriteLine($"Large MatMul (2048x2048): {largeOpMs:F2} ms");
        Console.WriteLine($"Estimated Transfer Time:  {transferTimeEstimate:F2} ms");
        Console.WriteLine($"Estimated Compute Time:   {computeTimeEstimate:F2} ms");
        Console.WriteLine($"Effective GFLOPS:         {gflops:F1}");
        Console.WriteLine();

        double transferRatio = transferTimeEstimate / largeOpMs * 100;
        if (transferRatio > 50)
        {
            Console.WriteLine($"[BOTTLENECK] Transfer-bound: {transferRatio:F0}% of time in transfers");
            Console.WriteLine("             Consider: persistent buffers, batch operations, pipelining");
        }
        else if (transferRatio > 20)
        {
            Console.WriteLine($"[INFO] Mixed workload: {transferRatio:F0}% transfer, {100 - transferRatio:F0}% compute");
        }
        else
        {
            Console.WriteLine($"[OK] Compute-bound: {100 - transferRatio:F0}% of time in GPU compute");
        }

        engine.Dispose();
    }
}

/// <summary>
/// Benchmarks to compare different GPU backends.
/// </summary>
public static class GpuBackendComparisonBenchmark
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("GPU BACKEND COMPARISON");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        var backends = new List<(string Name, Func<float[], float[], int, float[]?> Op)>();

        // Try OpenCL
        try
        {
            if (OpenClNativeBindings.IsAvailable)
            {
                var openClEngine = new DirectGpuEngine();
                if (openClEngine.IsAvailable && openClEngine.BackendName.Contains("OpenCL"))
                {
                    backends.Add(("OpenCL", (a, b, dim) =>
                    {
                        return openClEngine.MatMul(a, b, dim, dim, dim);
                    }));
                    Console.WriteLine($"[OK] OpenCL: {openClEngine.DeviceName}");
                }
            }
        }
        catch
        {
            Console.WriteLine("[SKIP] OpenCL not available");
        }

        // Try Vulkan
        try
        {
            var vulkan = VulkanBackend.Instance;
            if (vulkan.Initialize())
            {
                Console.WriteLine($"[OK] Vulkan: {vulkan.DeviceName}");
                // Vulkan backend doesn't have MatMul yet, just elementwise ops
                Console.WriteLine("     (Note: Vulkan only supports element-wise operations currently)");
            }
        }
        catch
        {
            Console.WriteLine("[SKIP] Vulkan not available");
        }

        if (backends.Count == 0)
        {
            Console.WriteLine("[ERROR] No GPU backends available for comparison.");
            return;
        }

        Console.WriteLine();

        // Run comparison
        var random = RandomHelper.CreateSeededRandom(42);
        int[] sizes = { 256, 512, 1024, 2048 };

        Console.WriteLine($"{"Backend",-12} {"Size",-10} {"Time (ms)",12} {"GFLOPS",10}");
        Console.WriteLine(new string('-', 46));

        foreach (var (name, op) in backends)
        {
            foreach (var size in sizes)
            {
                var inputA = new float[size * size];
                var inputB = new float[size * size];

                for (int i = 0; i < inputA.Length; i++)
                {
                    inputA[i] = (float)(random.NextDouble() * 2 - 1);
                    inputB[i] = (float)(random.NextDouble() * 2 - 1);
                }

                try
                {
                    // Warmup
                    op(inputA, inputB, size);

                    int runs = 10;
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < runs; i++)
                    {
                        op(inputA, inputB, size);
                    }
                    sw.Stop();

                    double avgMs = sw.Elapsed.TotalMilliseconds / runs;
                    double flops = 2.0 * size * size * size;
                    double gflops = flops / (avgMs * 1e6);

                    Console.WriteLine($"{name,-12} {size}x{size,-5} {avgMs,12:F4} {gflops,10:F1}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"{name,-12} {size}x{size,-5} ERROR: {ex.Message.Substring(0, Math.Min(30, ex.Message.Length))}");
                }
            }
        }
    }
}
