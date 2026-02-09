// Copyright (c) AiDotNet. All rights reserved.
// Comprehensive benchmarks for WebGPU backend to identify performance bottlenecks.
// WebGPU is designed for browser-based GPU compute in Blazor WebAssembly.

#if NET7_0_OR_GREATER
using System;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using AiDotNet.Tensors.Helpers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Comprehensive benchmarks for WebGPU backend.
/// Goal: Identify performance bottlenecks in browser GPU operations.
/// </summary>
/// <remarks>
/// <para><b>Key Metrics:</b></para>
/// <list type="bullet">
/// <item>Async operation overhead (compared to sync backends)</item>
/// <item>Browser GPU throughput vs native</item>
/// <item>Shader compilation latency</item>
/// <item>GPU vs CPU breakeven point in browser context</item>
/// </list>
/// <para><b>Note:</b></para>
/// <para>
/// WebGPU benchmarks use async operations. In Blazor WebAssembly, these
/// operations run on the browser's GPU via the WebGPU API.
/// </para>
/// </remarks>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class WebGpuBackendBenchmarks
{
    private WebGpuBackend? _webGpuBackend;
    private IGpuBuffer? _bufferA;
    private IGpuBuffer? _bufferB;
    private IGpuBuffer? _bufferC;
    private float[] _inputA = null!;
    private float[] _inputB = null!;
    private float[] _cpuOutput = null!;
    private bool _webGpuAvailable;

    [Params(256, 1024, 4096, 16384, 65536, 262144)]
    public int N { get; set; }

    [GlobalSetup]
    public async Task SetupAsync()
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

        try
        {
            _webGpuBackend = new WebGpuBackend();
            _webGpuAvailable = await _webGpuBackend.InitializeAsync();

            if (_webGpuAvailable)
            {
                _bufferA = _webGpuBackend.AllocateBuffer(_inputA);
                _bufferB = _webGpuBackend.AllocateBuffer(_inputB);
                _bufferC = _webGpuBackend.AllocateBuffer(N);
            }
        }
        catch
        {
            _webGpuAvailable = false;
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _bufferA?.Dispose();
        _bufferB?.Dispose();
        _bufferC?.Dispose();
        _webGpuBackend?.Dispose();
    }

    #region Vector Add Benchmarks

    [Benchmark(Description = "VectorAdd - WebGPU")]
    [BenchmarkCategory("VectorAdd")]
    public async Task VectorAddWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.AddAsync(_bufferA!, _bufferB!, _bufferC!, N);
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

    [Benchmark(Description = "VectorSubtract - WebGPU")]
    [BenchmarkCategory("VectorSubtract")]
    public async Task VectorSubtractWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.SubAsync(_bufferA!, _bufferB!, _bufferC!, N);
    }

    [Benchmark(Description = "VectorSubtract - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorSubtract")]
    public void VectorSubtractCpu()
    {
        TensorPrimitives.Subtract<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Vector Multiply Benchmarks

    [Benchmark(Description = "VectorMultiply - WebGPU")]
    [BenchmarkCategory("VectorMultiply")]
    public async Task VectorMultiplyWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.MulAsync(_bufferA!, _bufferB!, _bufferC!, N);
    }

    [Benchmark(Description = "VectorMultiply - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorMultiply")]
    public void VectorMultiplyCpu()
    {
        TensorPrimitives.Multiply<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Vector Divide Benchmarks

    [Benchmark(Description = "VectorDivide - WebGPU")]
    [BenchmarkCategory("VectorDivide")]
    public async Task VectorDivideWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.DivAsync(_bufferA!, _bufferB!, _bufferC!, N);
    }

    [Benchmark(Description = "VectorDivide - CPU TensorPrimitives")]
    [BenchmarkCategory("VectorDivide")]
    public void VectorDivideCpu()
    {
        TensorPrimitives.Divide<float>(_inputA, _inputB, _cpuOutput);
    }

    #endregion

    #region Scalar Multiply Benchmarks

    [Benchmark(Description = "ScalarMultiply - WebGPU")]
    [BenchmarkCategory("ScalarMultiply")]
    public async Task ScalarMultiplyWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.ScaleAsync(_bufferA!, _bufferC!, 2.5f, N);
    }

    [Benchmark(Description = "ScalarMultiply - CPU TensorPrimitives")]
    [BenchmarkCategory("ScalarMultiply")]
    public void ScalarMultiplyCpu()
    {
        TensorPrimitives.Multiply<float>(_inputA, 2.5f, _cpuOutput);
    }

    #endregion

    #region Activation Function Benchmarks

    [Benchmark(Description = "ReLU - WebGPU")]
    [BenchmarkCategory("ReLU")]
    public async Task ReLUWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.ReLUAsync(_bufferA!, _bufferC!, N);
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

    [Benchmark(Description = "Sigmoid - WebGPU")]
    [BenchmarkCategory("Sigmoid")]
    public async Task SigmoidWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.SigmoidAsync(_bufferA!, _bufferC!, N);
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

    [Benchmark(Description = "Tanh - WebGPU")]
    [BenchmarkCategory("Tanh")]
    public async Task TanhWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.TanhAsync(_bufferA!, _bufferC!, N);
    }

    [Benchmark(Description = "Tanh - CPU TensorPrimitives")]
    [BenchmarkCategory("Tanh")]
    public void TanhCpu()
    {
        TensorPrimitives.Tanh<float>(_inputA, _cpuOutput);
    }

    [Benchmark(Description = "GELU - WebGPU")]
    [BenchmarkCategory("GELU")]
    public async Task GELUWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.GeLUAsync(_bufferA!, _bufferC!, N);
    }

    [Benchmark(Description = "Swish - WebGPU")]
    [BenchmarkCategory("Swish")]
    public async Task SwishWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.SwishAsync(_bufferA!, _bufferC!, N);
    }

    #endregion

    #region Unary Operations

    [Benchmark(Description = "Sqrt - WebGPU")]
    [BenchmarkCategory("UnaryMath")]
    public async Task SqrtWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.SqrtAsync(_bufferA!, _bufferC!, N);
    }

    [Benchmark(Description = "Exp - WebGPU")]
    [BenchmarkCategory("UnaryMath")]
    public async Task ExpWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.ExpAsync(_bufferA!, _bufferC!, N);
    }

    [Benchmark(Description = "Log - WebGPU")]
    [BenchmarkCategory("UnaryMath")]
    public async Task LogWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.LogAsync(_bufferA!, _bufferC!, N);
    }

    [Benchmark(Description = "Abs - WebGPU")]
    [BenchmarkCategory("UnaryMath")]
    public async Task AbsWebGpu()
    {
        if (!_webGpuAvailable) return;
        await _webGpuBackend!.AbsAsync(_bufferA!, _bufferC!, N);
    }

    #endregion
}

/// <summary>
/// Reduction operation benchmarks for WebGPU backend.
/// </summary>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class WebGpuReductionBenchmarks
{
    private WebGpuBackend? _webGpuBackend;
    private IGpuBuffer? _buffer;
    private float[] _inputData = null!;
    private bool _webGpuAvailable;

    [Params(1024, 10240, 102400, 1024000)]
    public int N { get; set; }

    [GlobalSetup]
    public async Task SetupAsync()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _inputData = new float[N];
        for (int i = 0; i < N; i++)
        {
            _inputData[i] = (float)(random.NextDouble() * 2 - 1);
        }

        try
        {
            _webGpuBackend = new WebGpuBackend();
            _webGpuAvailable = await _webGpuBackend.InitializeAsync();

            if (_webGpuAvailable)
            {
                _buffer = _webGpuBackend.AllocateBuffer(_inputData);
            }
        }
        catch
        {
            _webGpuAvailable = false;
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _buffer?.Dispose();
        _webGpuBackend?.Dispose();
    }

    [Benchmark(Description = "Sum - WebGPU")]
    [BenchmarkCategory("Reduction")]
    public async Task<float> SumWebGpu()
    {
        if (!_webGpuAvailable) return 0;
        return await _webGpuBackend!.SumAsync(_buffer!, N);
    }

    [Benchmark(Description = "Sum - CPU TensorPrimitives", Baseline = true)]
    [BenchmarkCategory("Reduction")]
    public float SumCpu()
    {
        return TensorPrimitives.Sum<float>(_inputData);
    }

    [Benchmark(Description = "Max - WebGPU")]
    [BenchmarkCategory("Reduction")]
    public async Task<float> MaxWebGpu()
    {
        if (!_webGpuAvailable) return 0;
        return await _webGpuBackend!.MaxAsync(_buffer!, N);
    }

    [Benchmark(Description = "Max - CPU TensorPrimitives")]
    [BenchmarkCategory("Reduction")]
    public float MaxCpu()
    {
        return TensorPrimitives.Max<float>(_inputData);
    }

    [Benchmark(Description = "Min - WebGPU")]
    [BenchmarkCategory("Reduction")]
    public async Task<float> MinWebGpu()
    {
        if (!_webGpuAvailable) return 0;
        return await _webGpuBackend!.MinAsync(_buffer!, N);
    }

    [Benchmark(Description = "Min - CPU TensorPrimitives")]
    [BenchmarkCategory("Reduction")]
    public float MinCpu()
    {
        return TensorPrimitives.Min<float>(_inputData);
    }
}

/// <summary>
/// Detailed WebGPU diagnostics benchmark with manual timing.
/// Measures async overhead and browser GPU performance.
/// </summary>
public static class WebGpuDiagnosticsBenchmark
{
    public static async Task RunAsync()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("WEBGPU BACKEND DIAGNOSTICS & BOTTLENECK ANALYSIS");
        Console.WriteLine("Browser GPU Compute via WebGPU API");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        WebGpuBackend? backend = null;
        try
        {
            backend = new WebGpuBackend();
            bool initialized = await backend.InitializeAsync();

            if (!initialized)
            {
                Console.WriteLine("[ERROR] WebGPU backend failed to initialize.");
                Console.WriteLine("        Possible causes:");
                Console.WriteLine("        - Browser does not support WebGPU");
                Console.WriteLine("        - WebGPU not enabled in browser flags");
                Console.WriteLine("        - Running outside Blazor WebAssembly context");
                return;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to create WebGPU backend: {ex.Message}");
            return;
        }

        Console.WriteLine("[OK] WebGPU Backend Initialized");
        Console.WriteLine($"     Backend Type: {backend.BackendType}");
        Console.WriteLine($"     Device: {backend.DeviceName}");
        Console.WriteLine($"     Max Buffer Size: {backend.MaxBufferSize / (1024.0 * 1024):F1} MB");
        Console.WriteLine($"     Max Workgroup Size: {backend.MaxWorkgroupSize}");
        Console.WriteLine();

        // Warmup
        Console.WriteLine("[WARMUP] Running warmup passes...");
        using (var warmupA = backend.AllocateBuffer(new float[1024]))
        using (var warmupB = backend.AllocateBuffer(new float[1024]))
        using (var warmupC = backend.AllocateBuffer(1024))
        {
            for (int i = 0; i < 10; i++)
            {
                await backend.AddAsync(warmupA, warmupB, warmupC, 1024);
            }
        }
        Console.WriteLine("[WARMUP] Complete.");
        Console.WriteLine();

        await RunSizeBenchmarksAsync(backend);
        await RunOperationBenchmarksAsync(backend);
        await RunAsyncOverheadAnalysisAsync(backend);

        backend.Dispose();
    }

    private static async Task RunSizeBenchmarksAsync(WebGpuBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("SIZE SCALABILITY ANALYSIS");
        Console.WriteLine("Goal: Find optimal data sizes for WebGPU operations");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        int[] sizes = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144 };
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
            await backend.AddAsync(bufferA, bufferB, bufferC, size);
            TensorPrimitives.Add<float>(inputA, inputB, outputCpu);

            // Benchmark GPU
            int runs = size < 10000 ? 50 : 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
            {
                await backend.AddAsync(bufferA, bufferB, bufferC, size);
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

    private static async Task RunOperationBenchmarksAsync(WebGpuBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("OPERATION TYPE ANALYSIS");
        Console.WriteLine("Goal: Identify slow operations that need optimization");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        int size = 65536; // 64K elements - smaller for browser context
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

        var operations = new (string Name, Func<Task> Op, double ExpectedFlops)[]
        {
            ("Add", async () => await backend.AddAsync(bufferA, bufferB, bufferC, size), size),
            ("Subtract", async () => await backend.SubAsync(bufferA, bufferB, bufferC, size), size),
            ("Multiply", async () => await backend.MulAsync(bufferA, bufferB, bufferC, size), size),
            ("Divide", async () => await backend.DivAsync(bufferA, bufferB, bufferC, size), size),
            ("Scale", async () => await backend.ScaleAsync(bufferA, bufferC, 2.5f, size), size),
            ("ReLU", async () => await backend.ReLUAsync(bufferA, bufferC, size), size),
            ("Sigmoid", async () => await backend.SigmoidAsync(bufferA, bufferC, size), 4 * size),
            ("Tanh", async () => await backend.TanhAsync(bufferA, bufferC, size), 2 * size),
            ("GELU", async () => await backend.GeLUAsync(bufferA, bufferC, size), 10 * size),
            ("Sqrt", async () => await backend.SqrtAsync(bufferA, bufferC, size), size),
            ("Exp", async () => await backend.ExpAsync(bufferA, bufferC, size), size),
        };

        Console.WriteLine($"{"Operation",-12} {"Time (ms)",12} {"Elements/ms",14} {"GFLOPS",10} {"Status",-12}");
        Console.WriteLine(new string('-', 62));

        int runs = 30;
        foreach (var (name, op, flops) in operations)
        {
            // Warmup
            for (int i = 0; i < 3; i++) await op();

            // Benchmark
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
            {
                await op();
            }
            sw.Stop();
            double avgMs = sw.Elapsed.TotalMilliseconds / runs;
            double elementsPerMs = size / avgMs;
            double gflops = flops / (avgMs * 1e6);

            string status = gflops > 0.1 ? "OK" : "SLOW";
            Console.WriteLine($"{name,-12} {avgMs,12:F4} {elementsPerMs,14:N0} {gflops,10:F2} {status,-12}");
        }

        Console.WriteLine();
    }

    private static async Task RunAsyncOverheadAnalysisAsync(WebGpuBackend backend)
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("ASYNC OVERHEAD ANALYSIS");
        Console.WriteLine("Goal: Measure cost of async/await in WebGPU operations");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        // Measure minimum overhead with tiny operations
        using var tinyA = backend.AllocateBuffer(new float[1]);
        using var tinyB = backend.AllocateBuffer(new float[1]);
        using var tinyC = backend.AllocateBuffer(1);

        // Warmup
        for (int i = 0; i < 50; i++)
        {
            await backend.AddAsync(tinyA, tinyB, tinyC, 1);
        }

        // Measure overhead
        int runs = 500;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
        {
            await backend.AddAsync(tinyA, tinyB, tinyC, 1);
        }
        sw.Stop();
        double overheadMs = sw.Elapsed.TotalMilliseconds / runs;

        Console.WriteLine($"Minimum Async Operation Overhead: {overheadMs:F4} ms");
        Console.WriteLine($"Maximum Operations/Second:        {1000.0 / overheadMs:N0} ops/s");
        Console.WriteLine();

        // Compare batched vs individual operations
        int batchSize = 10000;
        using var batchA = backend.AllocateBuffer(new float[batchSize]);
        using var batchB = backend.AllocateBuffer(new float[batchSize]);
        using var batchC = backend.AllocateBuffer(batchSize);

        // Individual element operations (simulated)
        sw.Restart();
        await backend.AddAsync(batchA, batchB, batchC, batchSize);
        sw.Stop();
        double batchedMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine($"Single batched operation ({batchSize} elements): {batchedMs:F4} ms");
        Console.WriteLine($"Estimated time for {batchSize} individual ops:    {overheadMs * batchSize:F1} ms");
        Console.WriteLine($"Batching advantage:                              {(overheadMs * batchSize) / batchedMs:N0}x");
        Console.WriteLine();

        Console.WriteLine("Note: WebGPU operations have high per-call overhead due to async boundaries.");
        Console.WriteLine("      Always batch operations when possible for optimal performance.");
        Console.WriteLine();
    }
}
#endif
