// Copyright (c) AiDotNet. All rights reserved.
// Comprehensive benchmarks for Metal GPU backend to identify performance bottlenecks.
// Metal is only available on macOS/iOS with Apple Silicon.

using System;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
#endif

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Comprehensive benchmarks for Metal GPU backend on Apple Silicon.
/// Goal: Identify performance bottlenecks in Metal GPU operations.
/// </summary>
/// <remarks>
/// <para><b>Key Metrics:</b></para>
/// <list type="bullet">
/// <item>Unified memory advantage (zero-copy operations)</item>
/// <item>Metal Performance Shaders (MPS) utilization</item>
/// <item>Kernel compilation and dispatch latency</item>
/// <item>GPU vs CPU breakeven point on Apple Silicon</item>
/// </list>
/// </remarks>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class MetalBackendBenchmarks
{
#if NET7_0_OR_GREATER
    private MetalBackend? _metalBackend;
    private IGpuBuffer? _bufferA;
    private IGpuBuffer? _bufferB;
    private IGpuBuffer? _bufferC;
#endif
    private float[] _inputA = null!;
    private float[] _inputB = null!;
    private float[] _cpuOutput = null!;
    private bool _metalAvailable;

    [Params(256, 1024, 4096, 16384, 65536, 262144, 1048576)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _inputA = new float[N];
        _inputB = new float[N];
        _cpuOutput = new float[N];

        for (int i = 0; i < N; i++)
        {
            _inputA[i] = (float)(random.NextDouble() * 2 - 1);
            _inputB[i] = (float)(random.NextDouble() * 2 - 1) + 0.1f;
        }

#if NET7_0_OR_GREATER
        // Metal is only available on macOS
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            _metalAvailable = false;
            return;
        }

        try
        {
            _metalBackend = new MetalBackend();
            _metalAvailable = _metalBackend.IsAvailable;

            if (_metalAvailable)
            {
                _bufferA = _metalBackend.AllocateBuffer(_inputA);
                _bufferB = _metalBackend.AllocateBuffer(_inputB);
                _bufferC = _metalBackend.AllocateBuffer(N);
            }
        }
        catch
        {
            _metalAvailable = false;
        }
#else
        _metalAvailable = false;
#endif
    }

    [GlobalCleanup]
    public void Cleanup()
    {
#if NET7_0_OR_GREATER
        _bufferA?.Dispose();
        _bufferB?.Dispose();
        _bufferC?.Dispose();
        _metalBackend?.Dispose();
#endif
    }

    #region Vector Add Benchmarks

    [Benchmark(Description = "VectorAdd - Metal GPU")]
    [BenchmarkCategory("VectorAdd")]
    public void VectorAddMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Add(_bufferA!, _bufferB!, _bufferC!, N);
#endif
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

    [Benchmark(Description = "VectorSubtract - Metal GPU")]
    [BenchmarkCategory("VectorSubtract")]
    public void VectorSubtractMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Subtract(_bufferA!, _bufferB!, _bufferC!, N);
#endif
    }

    [Benchmark(Description = "VectorSubtract - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorSubtract")]
    public void VectorSubtractCpu()
    {
        TensorPrimitives.Subtract<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Vector Multiply Benchmarks

    [Benchmark(Description = "VectorMultiply - Metal GPU")]
    [BenchmarkCategory("VectorMultiply")]
    public void VectorMultiplyMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Multiply(_bufferA!, _bufferB!, _bufferC!, N);
#endif
    }

    [Benchmark(Description = "VectorMultiply - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorMultiply")]
    public void VectorMultiplyCpu()
    {
        TensorPrimitives.Multiply<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Vector Divide Benchmarks

    [Benchmark(Description = "VectorDivide - Metal GPU")]
    [BenchmarkCategory("VectorDivide")]
    public void VectorDivideMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Divide(_bufferA!, _bufferB!, _bufferC!, N);
#endif
    }

    [Benchmark(Description = "VectorDivide - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorDivide")]
    public void VectorDivideCpu()
    {
        TensorPrimitives.Divide<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Scalar Multiply Benchmarks

    [Benchmark(Description = "ScalarMultiply - Metal GPU")]
    [BenchmarkCategory("ScalarMultiply")]
    public void ScalarMultiplyMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Scale(_bufferA!, _bufferC!, 2.5f, N);
#endif
    }

    [Benchmark(Description = "ScalarMultiply - CPU TensorPrimitives")]
    [BenchmarkCategory("ScalarMultiply")]
    public void ScalarMultiplyCpu()
    {
        TensorPrimitives.Multiply<float>(_inputA, 2.5f, _cpuOutput);
    }

    #endregion

    #region Activation Function Benchmarks

    [Benchmark(Description = "ReLU - Metal GPU")]
    [BenchmarkCategory("ReLU")]
    public void ReLUMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Relu(_bufferA!, _bufferC!, N);
#endif
    }

    [Benchmark(Description = "ReLU - CPU Scalar")]
    [BenchmarkCategory("ReLU")]
    public void ReLUCpu()
    {
        for (int i = 0; i < N; i++)
        {
            _cpuOutput[i] = Math.Max(0, _inputA[i]);
        }
    }

    [Benchmark(Description = "Sigmoid - Metal GPU")]
    [BenchmarkCategory("Sigmoid")]
    public void SigmoidMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Sigmoid(_bufferA!, _bufferC!, N);
#endif
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

    [Benchmark(Description = "Tanh - Metal GPU")]
    [BenchmarkCategory("Tanh")]
    public void TanhMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Tanh(_bufferA!, _bufferC!, N);
#endif
    }

    [Benchmark(Description = "Tanh - CPU TensorPrimitives")]
    [BenchmarkCategory("Tanh")]
    public void TanhCpu()
    {
        TensorPrimitives.Tanh<float>(_inputA, _cpuOutput);
    }

    [Benchmark(Description = "GELU - Metal GPU")]
    [BenchmarkCategory("GELU")]
    public void GELUMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Gelu(_bufferA!, _bufferC!, N);
#endif
    }

    [Benchmark(Description = "Swish - Metal GPU")]
    [BenchmarkCategory("Swish")]
    public void SwishMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Swish(_bufferA!, _bufferC!, N);
#endif
    }

    #endregion
}

/// <summary>
/// Matrix operation benchmarks for Metal backend.
/// </summary>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class MetalGemmBenchmarks
{
#if NET7_0_OR_GREATER
    private MetalBackend? _metalBackend;
    private IGpuBuffer? _matrixA;
    private IGpuBuffer? _matrixB;
    private IGpuBuffer? _matrixC;
#endif
    private float[] _cpuMatrixA = null!;
    private float[] _cpuMatrixB = null!;
    private float[] _cpuMatrixC = null!;
    private bool _metalAvailable;

    [Params(32, 64, 128, 256, 512, 1024)]
    public int MatrixSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        int size = MatrixSize * MatrixSize;

        _cpuMatrixA = new float[size];
        _cpuMatrixB = new float[size];
        _cpuMatrixC = new float[size];

        for (int i = 0; i < size; i++)
        {
            _cpuMatrixA[i] = (float)(random.NextDouble() * 2 - 1);
            _cpuMatrixB[i] = (float)(random.NextDouble() * 2 - 1);
        }

#if NET7_0_OR_GREATER
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            _metalAvailable = false;
            return;
        }

        try
        {
            _metalBackend = new MetalBackend();
            _metalAvailable = _metalBackend.IsAvailable;

            if (_metalAvailable)
            {
                _matrixA = _metalBackend.AllocateBuffer(_cpuMatrixA);
                _matrixB = _metalBackend.AllocateBuffer(_cpuMatrixB);
                _matrixC = _metalBackend.AllocateBuffer(size);
            }
        }
        catch
        {
            _metalAvailable = false;
        }
#else
        _metalAvailable = false;
#endif
    }

    [GlobalCleanup]
    public void Cleanup()
    {
#if NET7_0_OR_GREATER
        _matrixA?.Dispose();
        _matrixB?.Dispose();
        _matrixC?.Dispose();
        _metalBackend?.Dispose();
#endif
    }

    [Benchmark(Description = "GEMM - Metal GPU")]
    [BenchmarkCategory("GEMM")]
    public void GemmMetal()
    {
#if NET7_0_OR_GREATER
        if (!_metalAvailable) return;
        _metalBackend!.Gemm(_matrixA!, _matrixB!, _matrixC!, MatrixSize, MatrixSize, MatrixSize);
#endif
    }

    [Benchmark(Description = "GEMM - CPU Naive", Baseline = true)]
    [BenchmarkCategory("GEMM")]
    public void GemmCpuNaive()
    {
        int M = MatrixSize, N = MatrixSize, K = MatrixSize;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += _cpuMatrixA[i * K + k] * _cpuMatrixB[k * N + j];
                }
                _cpuMatrixC[i * N + j] = sum;
            }
        }
    }
}

/// <summary>
/// Detailed Metal diagnostics benchmark with manual timing.
/// Tests unified memory advantage on Apple Silicon.
/// </summary>
public static class MetalDiagnosticsBenchmark
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("METAL BACKEND DIAGNOSTICS & BOTTLENECK ANALYSIS");
        Console.WriteLine("Apple Silicon Unified Memory Architecture");
        Console.WriteLine("===========================================");
        Console.WriteLine();

#if NET7_0_OR_GREATER
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            Console.WriteLine("[ERROR] Metal is only available on macOS/iOS.");
            Console.WriteLine("        Running on: " + RuntimeInformation.OSDescription);
            return;
        }

        MetalBackend? backend = null;
        try
        {
            backend = new MetalBackend();
            if (!backend.IsAvailable)
            {
                Console.WriteLine("[ERROR] Metal backend failed to initialize.");
                Console.WriteLine("        Possible causes:");
                Console.WriteLine("        - No Metal-compatible GPU found");
                Console.WriteLine("        - Metal framework not available");
                return;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to create Metal backend: {ex.Message}");
            return;
        }

        Console.WriteLine("[OK] Metal Backend Initialized");
        Console.WriteLine($"     Device: {backend.DeviceName}");
        Console.WriteLine($"     Vendor: {backend.DeviceVendor}");
        Console.WriteLine($"     Compute Units: {backend.ComputeUnits}");
        Console.WriteLine($"     Global Memory: {backend.GlobalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB");
        Console.WriteLine($"     Local Memory: {backend.LocalMemoryBytes / 1024.0:F1} KB");
        Console.WriteLine();

        // Warmup
        Console.WriteLine("[WARMUP] Running warmup passes...");
        using (var warmupA = backend.AllocateBuffer(new float[1024]))
        using (var warmupB = backend.AllocateBuffer(new float[1024]))
        using (var warmupC = backend.AllocateBuffer(1024))
        {
            for (int i = 0; i < 10; i++)
            {
                backend.Add(warmupA, warmupB, warmupC, 1024);
            }
        }
        Console.WriteLine("[WARMUP] Complete.");
        Console.WriteLine();

        RunSizeBenchmarks(backend);
        RunOperationBenchmarks(backend);
        RunUnifiedMemoryTest(backend);

        backend.Dispose();
#else
        Console.WriteLine("[ERROR] Metal benchmarks require .NET 7.0 or greater.");
#endif
    }

#if NET7_0_OR_GREATER
    private static void RunSizeBenchmarks(MetalBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("SIZE SCALABILITY ANALYSIS");
        Console.WriteLine("Goal: Find optimal data sizes for Metal operations");
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
            var outputCpu = new float[size];

            for (int i = 0; i < size; i++)
            {
                inputA[i] = (float)(random.NextDouble() * 2 - 1);
                inputB[i] = (float)(random.NextDouble() * 2 - 1);
            }

            using var bufferA = backend.AllocateBuffer(inputA);
            using var bufferB = backend.AllocateBuffer(inputB);
            using var bufferC = backend.AllocateBuffer(size);

            // Warmup
            backend.Add(bufferA, bufferB, bufferC, size);
            TensorPrimitives.Add<float>(inputA, inputB, outputCpu);

            // Benchmark GPU
            int runs = size < 10000 ? 100 : 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
            {
                backend.Add(bufferA, bufferB, bufferC, size);
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

    private static void RunOperationBenchmarks(MetalBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("OPERATION TYPE ANALYSIS");
        Console.WriteLine("Goal: Identify slow operations that need optimization");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        int size = 262144; // 256K elements
        var random = RandomHelper.CreateSeededRandom(42);

        var inputData = new float[size];
        var inputData2 = new float[size];

        for (int i = 0; i < size; i++)
        {
            inputData[i] = (float)(random.NextDouble() * 2 - 1);
            inputData2[i] = (float)(random.NextDouble() * 2 - 1) + 0.1f;
        }

        using var bufferA = backend.AllocateBuffer(inputData);
        using var bufferB = backend.AllocateBuffer(inputData2);
        using var bufferC = backend.AllocateBuffer(size);

        var operations = new (string Name, Action Op, double ExpectedFlops)[]
        {
            ("Add", () => backend.Add(bufferA, bufferB, bufferC, size), size),
            ("Subtract", () => backend.Subtract(bufferA, bufferB, bufferC, size), size),
            ("Multiply", () => backend.Multiply(bufferA, bufferB, bufferC, size), size),
            ("Divide", () => backend.Divide(bufferA, bufferB, bufferC, size), size),
            ("Scale", () => backend.Scale(bufferA, bufferC, 2.5f, size), size),
            ("ReLU", () => backend.Relu(bufferA, bufferC, size), size),
            ("Sigmoid", () => backend.Sigmoid(bufferA, bufferC, size), 4 * size),
            ("Tanh", () => backend.Tanh(bufferA, bufferC, size), 2 * size),
            ("GELU", () => backend.Gelu(bufferA, bufferC, size), 10 * size),
            ("Swish", () => backend.Swish(bufferA, bufferC, size), 5 * size),
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

    private static void RunUnifiedMemoryTest(MetalBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("UNIFIED MEMORY ARCHITECTURE TEST");
        Console.WriteLine("Goal: Measure zero-copy advantage on Apple Silicon");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        // On Apple Silicon, unified memory means CPU and GPU share memory
        // This should eliminate PCIe transfer overhead

        int[] sizes = { 1024, 10240, 102400, 1024000 };

        Console.WriteLine($"{"Size (KB)",-12} {"Alloc (ms)",12} {"Upload (ms)",12} {"Download (ms)",14}");
        Console.WriteLine(new string('-', 52));

        foreach (var size in sizes)
        {
            var data = new float[size];
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)random.NextDouble();
            }

            // Measure allocation time
            var sw = Stopwatch.StartNew();
            using var buffer = backend.AllocateBuffer(size);
            sw.Stop();
            double allocMs = sw.Elapsed.TotalMilliseconds;

            // Measure upload time
            sw.Restart();
            backend.UploadToBuffer(buffer, data);
            sw.Stop();
            double uploadMs = sw.Elapsed.TotalMilliseconds;

            // Measure download time
            var result = new float[size];
            sw.Restart();
            backend.DownloadBuffer(buffer, result);
            sw.Stop();
            double downloadMs = sw.Elapsed.TotalMilliseconds;

            double sizeKb = size * sizeof(float) / 1024.0;
            Console.WriteLine($"{sizeKb,-12:F1} {allocMs,12:F4} {uploadMs,12:F4} {downloadMs,14:F4}");
        }

        Console.WriteLine();
        Console.WriteLine("Note: On Apple Silicon, upload/download should be very fast due to unified memory.");
        Console.WriteLine("      If these times are high, there may be synchronization overhead.");
        Console.WriteLine();
    }
#endif
}
