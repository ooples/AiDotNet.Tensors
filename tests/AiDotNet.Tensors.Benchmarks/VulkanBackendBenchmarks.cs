// Copyright (c) AiDotNet. All rights reserved.
// Comprehensive benchmarks for Vulkan GPU backend to identify performance bottlenecks.

using System;
using System.Diagnostics;
using System.Numerics.Tensors;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Comprehensive benchmarks for Vulkan GPU backend.
/// Goal: Identify performance bottlenecks in GPU operations.
/// </summary>
/// <remarks>
/// <para><b>Key Metrics:</b></para>
/// <list type="bullet">
/// <item>GPU throughput (elements/sec)</item>
/// <item>Memory transfer overhead</item>
/// <item>Kernel launch latency</item>
/// <item>GPU vs CPU breakeven point</item>
/// </list>
/// </remarks>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class VulkanBackendBenchmarks
{
    private VulkanBackend? _vulkanBackend;
    private float[] _inputA = null!;
    private float[] _inputB = null!;
    private float[] _output = null!;
    private float[] _cpuOutput = null!;
    private bool _vulkanAvailable;

    [Params(256, 1024, 4096, 16384, 65536, 262144, 1048576)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _inputA = new float[N];
        _inputB = new float[N];
        _output = new float[N];
        _cpuOutput = new float[N];

        for (int i = 0; i < N; i++)
        {
            _inputA[i] = (float)(random.NextDouble() * 2 - 1);
            _inputB[i] = (float)(random.NextDouble() * 2 - 1) + 0.1f; // Avoid division by zero
        }

        // Try to initialize Vulkan backend
        try
        {
            _vulkanBackend = VulkanBackend.Instance;
            _vulkanAvailable = _vulkanBackend.Initialize();
        }
        catch
        {
            _vulkanAvailable = false;
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // VulkanBackend is a singleton, don't dispose
    }

    #region Vector Add Benchmarks

    [Benchmark(Description = "VectorAdd - Vulkan GPU")]
    [BenchmarkCategory("VectorAdd")]
    public void VectorAddVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Add(_inputA, _inputB, _output);
    }

    [Benchmark(Description = "VectorAdd - CPU TensorPrimitives", Baseline = true)]
    [BenchmarkCategory("VectorAdd")]
    public void VectorAddCpu()
    {
        TensorPrimitives.Add<float>(_inputA, _inputB, _cpuOutput);
    }

    [Benchmark(Description = "VectorAdd - CPU Scalar Loop")]
    [BenchmarkCategory("VectorAdd")]
    public void VectorAddScalarCpu()
    {
        for (int i = 0; i < N; i++)
        {
            _cpuOutput[i] = _inputA[i] + _inputB[i];
        }
    }

    #endregion

    #region Vector Subtract Benchmarks

    [Benchmark(Description = "VectorSubtract - Vulkan GPU")]
    [BenchmarkCategory("VectorSubtract")]
    public void VectorSubtractVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Subtract(_inputA, _inputB, _output);
    }

    [Benchmark(Description = "VectorSubtract - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorSubtract")]
    public void VectorSubtractCpu()
    {
        TensorPrimitives.Subtract<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Vector Multiply Benchmarks

    [Benchmark(Description = "VectorMultiply - Vulkan GPU")]
    [BenchmarkCategory("VectorMultiply")]
    public void VectorMultiplyVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Multiply(_inputA, _inputB, _output);
    }

    [Benchmark(Description = "VectorMultiply - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorMultiply")]
    public void VectorMultiplyCpu()
    {
        TensorPrimitives.Multiply<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Vector Divide Benchmarks

    [Benchmark(Description = "VectorDivide - Vulkan GPU")]
    [BenchmarkCategory("VectorDivide")]
    public void VectorDivideVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Divide(_inputA, _inputB, _output);
    }

    [Benchmark(Description = "VectorDivide - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorDivide")]
    public void VectorDivideCpu()
    {
        TensorPrimitives.Divide<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Scalar Multiply Benchmarks

    [Benchmark(Description = "ScalarMultiply - Vulkan GPU")]
    [BenchmarkCategory("ScalarMultiply")]
    public void ScalarMultiplyVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.ScalarMultiply(_inputA, 2.5f, _output);
    }

    [Benchmark(Description = "ScalarMultiply - CPU TensorPrimitives")]
    [BenchmarkCategory("ScalarMultiply")]
    public void ScalarMultiplyCpu()
    {
        TensorPrimitives.Multiply<float>(_inputA, 2.5f, _cpuOutput);
    }

    #endregion

    #region Activation Function Benchmarks

    [Benchmark(Description = "ReLU - Vulkan GPU")]
    [BenchmarkCategory("ReLU")]
    public void ReLUVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.ReLU(_inputA, _output);
    }

    [Benchmark(Description = "ReLU - CPU TensorPrimitives")]
    [BenchmarkCategory("ReLU")]
    public void ReLUCpu()
    {
        for (int i = 0; i < N; i++)
        {
            _cpuOutput[i] = Math.Max(0, _inputA[i]);
        }
    }

    [Benchmark(Description = "Sigmoid - Vulkan GPU")]
    [BenchmarkCategory("Sigmoid")]
    public void SigmoidVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Sigmoid(_inputA, _output);
    }

    [Benchmark(Description = "Sigmoid - CPU Scalar")]
    [BenchmarkCategory("Sigmoid")]
    public void SigmoidCpu()
    {
        for (int i = 0; i < N; i++)
        {
            _cpuOutput[i] = 1.0f / (1.0f + MathF.Exp(-_inputA[i]));
        }
    }

    [Benchmark(Description = "Tanh - Vulkan GPU")]
    [BenchmarkCategory("Tanh")]
    public void TanhVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Tanh(_inputA, _output);
    }

    [Benchmark(Description = "Tanh - CPU TensorPrimitives")]
    [BenchmarkCategory("Tanh")]
    public void TanhCpu()
    {
        TensorPrimitives.Tanh<float>(_inputA, _cpuOutput);
    }

    #endregion
}

/// <summary>
/// Benchmarks to measure GPU overhead and breakeven points.
/// Goal: Find the minimum data size where GPU beats CPU.
/// </summary>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class GpuBreakevenBenchmarks
{
    private VulkanBackend? _vulkanBackend;
    private float[][] _smallInputsA = null!;
    private float[][] _smallInputsB = null!;
    private float[][] _smallOutputs = null!;
    private bool _vulkanAvailable;

    private static readonly int[] SmallSizes = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _smallInputsA = new float[SmallSizes.Length][];
        _smallInputsB = new float[SmallSizes.Length][];
        _smallOutputs = new float[SmallSizes.Length][];

        for (int i = 0; i < SmallSizes.Length; i++)
        {
            int size = SmallSizes[i];
            _smallInputsA[i] = new float[size];
            _smallInputsB[i] = new float[size];
            _smallOutputs[i] = new float[size];

            for (int j = 0; j < size; j++)
            {
                _smallInputsA[i][j] = (float)(random.NextDouble() * 2 - 1);
                _smallInputsB[i][j] = (float)(random.NextDouble() * 2 - 1);
            }
        }

        try
        {
            _vulkanBackend = VulkanBackend.Instance;
            _vulkanAvailable = _vulkanBackend.Initialize();
        }
        catch
        {
            _vulkanAvailable = false;
        }
    }

    [Benchmark(Description = "Small Vector Add - Breakeven Analysis")]
    public void SmallVectorAddBreakevenAnalysis()
    {
        if (!_vulkanAvailable) return;

        for (int i = 0; i < SmallSizes.Length; i++)
        {
            _vulkanBackend!.Add(_smallInputsA[i], _smallInputsB[i], _smallOutputs[i]);
        }
    }
}

/// <summary>
/// Detailed Vulkan diagnostics benchmark with manual timing.
/// </summary>
public static class VulkanDiagnosticsBenchmark
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("VULKAN BACKEND DIAGNOSTICS & BOTTLENECK ANALYSIS");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        // Check availability
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize())
            {
                Console.WriteLine("[ERROR] Vulkan backend failed to initialize.");
                Console.WriteLine("        Possible causes:");
                Console.WriteLine("        - No Vulkan driver installed");
                Console.WriteLine("        - GPU does not support Vulkan compute");
                Console.WriteLine("        - Missing vulkan-1.dll / libvulkan.so");
                return;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to create Vulkan backend: {ex.Message}");
            return;
        }

        Console.WriteLine("[OK] Vulkan Backend Initialized");
        Console.WriteLine($"     Device: {backend.DeviceName}");
        Console.WriteLine($"     Vendor: {backend.VendorName}");
        Console.WriteLine($"     Max Workgroup Size: {backend.MaxWorkgroupSize}");
        Console.WriteLine($"     Max Shared Memory: {backend.MaxSharedMemorySize / 1024.0:F1} KB");
        Console.WriteLine();

        // Warmup
        Console.WriteLine("[WARMUP] Running warmup passes...");
        var warmupA = new float[1024];
        var warmupB = new float[1024];
        var warmupOut = new float[1024];
        for (int i = 0; i < 10; i++)
        {
            backend.Add(warmupA, warmupB, warmupOut);
        }
        Console.WriteLine("[WARMUP] Complete.");
        Console.WriteLine();

        // Benchmark different sizes
        RunSizeBenchmarks(backend);

        // Benchmark different operations
        RunOperationBenchmarks(backend);

        // Benchmark overhead
        RunOverheadAnalysis(backend);
    }

    private static void RunSizeBenchmarks(VulkanBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("SIZE SCALABILITY ANALYSIS");
        Console.WriteLine("Goal: Find optimal data sizes for GPU operations");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        int[] sizes = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 };
        var random = RandomHelper.CreateSeededRandom(42);

        Console.WriteLine($"{"Size",-12} {"GPU (ms)",10} {"CPU (ms)",10} {"Speedup",10} {"Elements/ms",14} {"Status",-12}");
        Console.WriteLine(new string('-', 70));

        foreach (var size in sizes)
        {
            var inputA = new float[size];
            var inputB = new float[size];
            var outputGpu = new float[size];
            var outputCpu = new float[size];

            for (int i = 0; i < size; i++)
            {
                inputA[i] = (float)(random.NextDouble() * 2 - 1);
                inputB[i] = (float)(random.NextDouble() * 2 - 1);
            }

            // Warmup
            backend.Add(inputA, inputB, outputGpu);
            TensorPrimitives.Add<float>(inputA, inputB, outputCpu);

            // Benchmark GPU
            int runs = size < 10000 ? 100 : 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
            {
                backend.Add(inputA, inputB, outputGpu);
            }
            sw.Stop();
            double gpuMs = sw.Elapsed.TotalMilliseconds / runs;

            // Benchmark CPU
            sw.Restart();
            for (int i = 0; i < runs; i++)
            {
                TensorPrimitives.Add<float>(inputA, inputB, outputCpu);
            }
            sw.Stop();
            double cpuMs = sw.Elapsed.TotalMilliseconds / runs;

            double speedup = cpuMs / gpuMs;
            double elementsPerMs = size / gpuMs;
            string status = speedup > 1.0 ? "GPU FASTER" : "CPU FASTER";

            Console.WriteLine($"{size,-12} {gpuMs,10:F4} {cpuMs,10:F4} {speedup,10:F2}x {elementsPerMs,14:N0} {status,-12}");
        }

        Console.WriteLine();
    }

    private static void RunOperationBenchmarks(VulkanBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("OPERATION TYPE ANALYSIS");
        Console.WriteLine("Goal: Identify slow operations that need optimization");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        int size = 262144; // 256K elements - should favor GPU
        var random = RandomHelper.CreateSeededRandom(42);

        var inputA = new float[size];
        var inputB = new float[size];
        var output = new float[size];

        for (int i = 0; i < size; i++)
        {
            inputA[i] = (float)(random.NextDouble() * 2 - 1);
            inputB[i] = (float)(random.NextDouble() * 2 - 1) + 0.1f;
        }

        var operations = new (string Name, Action Op, double ExpectedFlops)[]
        {
            ("Add", () => backend.Add(inputA, inputB, output), size),
            ("Subtract", () => backend.Subtract(inputA, inputB, output), size),
            ("Multiply", () => backend.Multiply(inputA, inputB, output), size),
            ("Divide", () => backend.Divide(inputA, inputB, output), size),
            ("ScalarMul", () => backend.ScalarMultiply(inputA, 2.5f, output), size),
            ("ReLU", () => backend.ReLU(inputA, output), size),
            ("Sigmoid", () => backend.Sigmoid(inputA, output), 4 * size), // exp + add + div
            ("Tanh", () => backend.Tanh(inputA, output), 2 * size), // tanh is complex
        };

        Console.WriteLine($"{"Operation",-12} {"Time (ms)",12} {"Elements/ms",14} {"GFLOPS",10} {"Status",-12}");
        Console.WriteLine(new string('-', 62));

        int runs = 50;
        foreach (var (name, op, flops) in operations)
        {
            // Warmup
            for (int i = 0; i < 5; i++) op();

            // Benchmark
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
            {
                op();
            }
            sw.Stop();
            double avgMs = sw.Elapsed.TotalMilliseconds / runs;
            double elementsPerMs = size / avgMs;
            double gflops = flops / (avgMs * 1e6);

            string status = gflops > 1.0 ? "OK" : "SLOW";
            Console.WriteLine($"{name,-12} {avgMs,12:F4} {elementsPerMs,14:N0} {gflops,10:F2} {status,-12}");
        }

        Console.WriteLine();
    }

    private static void RunOverheadAnalysis(VulkanBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("OVERHEAD ANALYSIS");
        Console.WriteLine("Goal: Measure fixed costs vs data-dependent costs");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        // Measure the minimum overhead by running on tiny data
        var tinyA = new float[1];
        var tinyB = new float[1];
        var tinyOut = new float[1];

        // Warmup
        for (int i = 0; i < 100; i++)
        {
            backend.Add(tinyA, tinyB, tinyOut);
        }

        // Measure overhead
        int runs = 1000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
        {
            backend.Add(tinyA, tinyB, tinyOut);
        }
        sw.Stop();
        double overheadMs = sw.Elapsed.TotalMilliseconds / runs;

        Console.WriteLine($"Minimum Operation Overhead: {overheadMs:F4} ms");
        Console.WriteLine($"Maximum Operations/Second:  {1000.0 / overheadMs:N0} ops/s");
        Console.WriteLine();

        // Calculate breakeven point
        // GPU time = overhead + data_time
        // CPU time = data * time_per_element
        // Breakeven: overhead + data * gpu_data_time = data * cpu_time_per_element

        var largeA = new float[1000000];
        var largeB = new float[1000000];
        var largeOut = new float[1000000];
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < largeA.Length; i++)
        {
            largeA[i] = (float)(random.NextDouble() * 2 - 1);
            largeB[i] = (float)(random.NextDouble() * 2 - 1);
        }

        // Warmup
        backend.Add(largeA, largeB, largeOut);
        TensorPrimitives.Add<float>(largeA, largeB, largeOut);

        // Benchmark large data
        sw.Restart();
        for (int i = 0; i < 20; i++)
        {
            backend.Add(largeA, largeB, largeOut);
        }
        sw.Stop();
        double gpuLargeMs = sw.Elapsed.TotalMilliseconds / 20;

        sw.Restart();
        for (int i = 0; i < 20; i++)
        {
            TensorPrimitives.Add<float>(largeA, largeB, largeOut);
        }
        sw.Stop();
        double cpuLargeMs = sw.Elapsed.TotalMilliseconds / 20;

        double gpuDataTimePerElement = (gpuLargeMs - overheadMs) / 1000000.0;
        double cpuTimePerElement = cpuLargeMs / 1000000.0;

        Console.WriteLine($"GPU Data Processing Rate: {1.0 / gpuDataTimePerElement / 1e6:F2} million elements/ms");
        Console.WriteLine($"CPU Data Processing Rate: {1.0 / cpuTimePerElement / 1e6:F2} million elements/ms");
        Console.WriteLine();

        // Calculate breakeven
        if (cpuTimePerElement > gpuDataTimePerElement)
        {
            double breakevenElements = overheadMs / (cpuTimePerElement - gpuDataTimePerElement);
            Console.WriteLine($"Estimated GPU Breakeven Point: ~{breakevenElements:N0} elements");
            Console.WriteLine($"                             = ~{breakevenElements * 4 / 1024:F1} KB of float data");
        }
        else
        {
            Console.WriteLine("[WARNING] CPU appears faster per-element than GPU data transfer.");
            Console.WriteLine("          GPU may only benefit for very compute-intensive operations.");
        }

        Console.WriteLine();

        // Memory bandwidth analysis
        Console.WriteLine("MEMORY BANDWIDTH ANALYSIS");
        Console.WriteLine("-------------------------");

        // For Add: we read 2 arrays and write 1, so 3 * size * 4 bytes
        double bytesTransferred = 3.0 * 1000000 * 4;
        double bandwidthGBps = bytesTransferred / (gpuLargeMs * 1e6);
        Console.WriteLine($"Effective Memory Bandwidth: {bandwidthGBps:F2} GB/s");
        Console.WriteLine($"(Based on Add operation with 1M elements)");
        Console.WriteLine();
    }
}
