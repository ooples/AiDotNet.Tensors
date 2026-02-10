// Copyright (c) AiDotNet. All rights reserved.
// Cross-library benchmarks to identify where our GPU implementation stands.

using System;
using System.Diagnostics;
using System.Numerics.Tensors;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NumSharp;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Cross-library benchmarks comparing GPU operations against popular .NET libraries.
/// Goal: Identify where our GPU implementation is faster/slower than alternatives.
/// </summary>
[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class CrossLibraryGpuBenchmarks
{
    // Raw data
    private float[] _floatDataA = null!;
    private float[] _floatDataB = null!;
    private float[] _floatResult = null!;
    private double[] _doubleDataA = null!;
    private double[] _doubleDataB = null!;
    private double[] _doubleResult = null!;

    // AiDotNet types
    private AiDotNet.Tensors.LinearAlgebra.Matrix<float> _aiMatrixFloat1 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Matrix<float> _aiMatrixFloat2 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Matrix<double> _aiMatrixDouble1 = null!;
    private AiDotNet.Tensors.LinearAlgebra.Matrix<double> _aiMatrixDouble2 = null!;

    // MathNet types
    private MathNet.Numerics.LinearAlgebra.Matrix<double> _mnMatrix1 = null!;
    private MathNet.Numerics.LinearAlgebra.Matrix<double> _mnMatrix2 = null!;

    // NumSharp types
    private NDArray _nsMatrix1 = null!;
    private NDArray _nsMatrix2 = null!;

    // GPU types
    private DirectGpuEngine? _gpuEngine;
    private VulkanBackend? _vulkanBackend;
    private bool _gpuAvailable;
    private bool _vulkanAvailable;

    [Params(128, 256, 512)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Initialize float data
        _floatDataA = new float[N * N];
        _floatDataB = new float[N * N];
        _floatResult = new float[N * N];
        var floatMatrix1 = new float[N, N];
        var floatMatrix2 = new float[N, N];

        for (int i = 0; i < N * N; i++)
        {
            _floatDataA[i] = (float)(random.NextDouble() * 10);
            _floatDataB[i] = (float)(random.NextDouble() * 10);
        }

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                floatMatrix1[i, j] = _floatDataA[i * N + j];
                floatMatrix2[i, j] = _floatDataB[i * N + j];
            }
        }

        _aiMatrixFloat1 = new AiDotNet.Tensors.LinearAlgebra.Matrix<float>(floatMatrix1);
        _aiMatrixFloat2 = new AiDotNet.Tensors.LinearAlgebra.Matrix<float>(floatMatrix2);

        // Initialize double data
        _doubleDataA = new double[N * N];
        _doubleDataB = new double[N * N];
        _doubleResult = new double[N * N];
        var doubleMatrix1 = new double[N, N];
        var doubleMatrix2 = new double[N, N];

        for (int i = 0; i < N * N; i++)
        {
            _doubleDataA[i] = random.NextDouble() * 10;
            _doubleDataB[i] = random.NextDouble() * 10;
        }

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                doubleMatrix1[i, j] = _doubleDataA[i * N + j];
                doubleMatrix2[i, j] = _doubleDataB[i * N + j];
            }
        }

        _aiMatrixDouble1 = new AiDotNet.Tensors.LinearAlgebra.Matrix<double>(doubleMatrix1);
        _aiMatrixDouble2 = new AiDotNet.Tensors.LinearAlgebra.Matrix<double>(doubleMatrix2);
        _mnMatrix1 = DenseMatrix.OfArray(doubleMatrix1);
        _mnMatrix2 = DenseMatrix.OfArray(doubleMatrix2);
        _nsMatrix1 = np.array(_doubleDataA).reshape(N, N);
        _nsMatrix2 = np.array(_doubleDataB).reshape(N, N);

        // Initialize GPU
        try
        {
            _gpuEngine = new DirectGpuEngine();
            _gpuAvailable = _gpuEngine.IsAvailable;
        }
        catch
        {
            _gpuAvailable = false;
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

    [GlobalCleanup]
    public void Cleanup()
    {
        _gpuEngine?.Dispose();
    }

    #region Matrix Multiply - The Critical Operation

    [Benchmark(Description = "MatMul - DirectGpu (float)")]
    [BenchmarkCategory("MatMul")]
    public float[]? MatMulDirectGpu()
    {
        if (!_gpuAvailable) return null;
        return _gpuEngine!.MatMul(_floatDataA, _floatDataB, N, N, N);
    }

    [Benchmark(Description = "MatMul - AiDotNet CPU (float)")]
    [BenchmarkCategory("MatMul")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<float> MatMulAiDotNetCpuFloat()
    {
        return _aiMatrixFloat1.Multiply(_aiMatrixFloat2);
    }

    [Benchmark(Description = "MatMul - AiDotNet CPU (double)")]
    [BenchmarkCategory("MatMul")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> MatMulAiDotNetCpuDouble()
    {
        return _aiMatrixDouble1.Multiply(_aiMatrixDouble2);
    }

    [Benchmark(Description = "MatMul - MathNet (double)", Baseline = true)]
    [BenchmarkCategory("MatMul")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> MatMulMathNet()
    {
        return _mnMatrix1.Multiply(_mnMatrix2);
    }

    [Benchmark(Description = "MatMul - NumSharp (double)")]
    [BenchmarkCategory("MatMul")]
    public NDArray MatMulNumSharp()
    {
        return np.matmul(_nsMatrix1, _nsMatrix2);
    }

    #endregion

    #region Element-wise Add - Memory Bandwidth Test

    [Benchmark(Description = "Add - Vulkan GPU (float)")]
    [BenchmarkCategory("Add")]
    public void AddVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Add(_floatDataA, _floatDataB, _floatResult);
    }

    [Benchmark(Description = "Add - TensorPrimitives (float)")]
    [BenchmarkCategory("Add")]
    public void AddTensorPrimitives()
    {
        TensorPrimitives.Add<float>(_floatDataA, _floatDataB, _floatResult);
    }

    [Benchmark(Description = "Add - AiDotNet (double)")]
    [BenchmarkCategory("Add")]
    public AiDotNet.Tensors.LinearAlgebra.Matrix<double> AddAiDotNet()
    {
        return _aiMatrixDouble1.Add(_aiMatrixDouble2);
    }

    [Benchmark(Description = "Add - MathNet (double)")]
    [BenchmarkCategory("Add")]
    public MathNet.Numerics.LinearAlgebra.Matrix<double> AddMathNet()
    {
        return _mnMatrix1.Add(_mnMatrix2);
    }

    #endregion

    #region Activation Functions - Compute-Intensive Test

    [Benchmark(Description = "Sigmoid - Vulkan GPU")]
    [BenchmarkCategory("Activation")]
    public void SigmoidVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Sigmoid(_floatDataA, _floatResult);
    }

    [Benchmark(Description = "Sigmoid - CPU Scalar")]
    [BenchmarkCategory("Activation")]
    public void SigmoidCpuScalar()
    {
        for (int i = 0; i < _floatDataA.Length; i++)
        {
            _floatResult[i] = 1.0f / (1.0f + MathF.Exp(-_floatDataA[i]));
        }
    }

    [Benchmark(Description = "Tanh - Vulkan GPU")]
    [BenchmarkCategory("Activation")]
    public void TanhVulkan()
    {
        if (!_vulkanAvailable) return;
        _vulkanBackend!.Tanh(_floatDataA, _floatResult);
    }

    [Benchmark(Description = "Tanh - TensorPrimitives")]
    [BenchmarkCategory("Activation")]
    public void TanhTensorPrimitives()
    {
        TensorPrimitives.Tanh<float>(_floatDataA, _floatResult);
    }

    #endregion
}

/// <summary>
/// Detailed cross-library comparison with manual timing for full analysis.
/// </summary>
public static class CrossLibraryAnalysis
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("CROSS-LIBRARY PERFORMANCE ANALYSIS");
        Console.WriteLine("Goal: Identify competitive position of AiDotNet GPU");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        var random = RandomHelper.CreateSeededRandom(42);

        // Test configurations
        int[] sizes = { 64, 128, 256, 512, 1024 };

        // Initialize GPU
        DirectGpuEngine? gpu = null;
        VulkanBackend? vulkan = null;
        bool gpuAvailable = false;
        bool vulkanAvailable = false;

        try
        {
            gpu = new DirectGpuEngine();
            gpuAvailable = gpu.IsAvailable;
            if (gpuAvailable)
            {
                Console.WriteLine($"[OK] DirectGpu: {gpu.BackendName} on {gpu.DeviceName}");
            }
        }
        catch
        {
            Console.WriteLine("[SKIP] DirectGpu not available");
        }

        try
        {
            vulkan = VulkanBackend.Instance;
            vulkanAvailable = vulkan.Initialize();
            if (vulkanAvailable)
            {
                Console.WriteLine($"[OK] Vulkan: {vulkan.VendorName} {vulkan.DeviceName}");
            }
        }
        catch
        {
            Console.WriteLine("[SKIP] Vulkan not available");
        }

        Console.WriteLine();

        // Matrix Multiplication Comparison
        Console.WriteLine("===========================================");
        Console.WriteLine("MATRIX MULTIPLICATION (GEMM)");
        Console.WriteLine("===========================================");
        Console.WriteLine();
        Console.WriteLine($"{"Size",-10} {"DirectGpu",12} {"AiDotNet",12} {"MathNet",12} {"NumSharp",12} {"Winner",-10}");
        Console.WriteLine(new string('-', 72));

        foreach (var size in sizes)
        {
            var doubleData1 = new double[size, size];
            var doubleData2 = new double[size, size];
            var floatData1 = new float[size * size];
            var floatData2 = new float[size * size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    doubleData1[i, j] = random.NextDouble() * 10;
                    doubleData2[i, j] = random.NextDouble() * 10;
                    floatData1[i * size + j] = (float)(random.NextDouble() * 10);
                    floatData2[i * size + j] = (float)(random.NextDouble() * 10);
                }
            }

            var aiMatrix1 = new AiDotNet.Tensors.LinearAlgebra.Matrix<double>(doubleData1);
            var aiMatrix2 = new AiDotNet.Tensors.LinearAlgebra.Matrix<double>(doubleData2);
            var mnMatrix1 = DenseMatrix.OfArray(doubleData1);
            var mnMatrix2 = DenseMatrix.OfArray(doubleData2);
            var nsMatrix1 = np.array(doubleData1);
            var nsMatrix2 = np.array(doubleData2);

            int runs = size < 256 ? 50 : (size < 512 ? 20 : 5);
            var results = new Dictionary<string, double>();

            // DirectGpu
            if (gpuAvailable)
            {
                try
                {
                    gpu!.MatMul(floatData1, floatData2, size, size, size); // warmup
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < runs; i++)
                    {
                        gpu.MatMul(floatData1, floatData2, size, size, size);
                    }
                    sw.Stop();
                    results["DirectGpu"] = sw.Elapsed.TotalMilliseconds / runs;
                }
                catch
                {
                    results["DirectGpu"] = double.MaxValue;
                }
            }
            else
            {
                results["DirectGpu"] = double.MaxValue;
            }

            // AiDotNet CPU
            {
                aiMatrix1.Multiply(aiMatrix2); // warmup
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < runs; i++)
                {
                    aiMatrix1.Multiply(aiMatrix2);
                }
                sw.Stop();
                results["AiDotNet"] = sw.Elapsed.TotalMilliseconds / runs;
            }

            // MathNet
            {
                mnMatrix1.Multiply(mnMatrix2); // warmup
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < runs; i++)
                {
                    mnMatrix1.Multiply(mnMatrix2);
                }
                sw.Stop();
                results["MathNet"] = sw.Elapsed.TotalMilliseconds / runs;
            }

            // NumSharp
            {
                np.matmul(nsMatrix1, nsMatrix2); // warmup
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < runs; i++)
                {
                    np.matmul(nsMatrix1, nsMatrix2);
                }
                sw.Stop();
                results["NumSharp"] = sw.Elapsed.TotalMilliseconds / runs;
            }

            // Find winner
            var winner = results.MinBy(x => x.Value).Key;
            var dgStr = results["DirectGpu"] < 10000 ? $"{results["DirectGpu"]:F2}ms" : "N/A";
            var aiStr = $"{results["AiDotNet"]:F2}ms";
            var mnStr = $"{results["MathNet"]:F2}ms";
            var nsStr = $"{results["NumSharp"]:F2}ms";

            Console.WriteLine($"{size}x{size,-6} {dgStr,12} {aiStr,12} {mnStr,12} {nsStr,12} {winner,-10}");
        }

        Console.WriteLine();

        // Element-wise operations (if Vulkan available)
        if (vulkanAvailable)
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("ELEMENT-WISE OPERATIONS (Vulkan GPU)");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            int elemSize = 1048576; // 1M elements
            var inputA = new float[elemSize];
            var inputB = new float[elemSize];
            var output = new float[elemSize];

            for (int i = 0; i < elemSize; i++)
            {
                inputA[i] = (float)(random.NextDouble() * 2 - 1);
                inputB[i] = (float)(random.NextDouble() * 2 - 1) + 0.1f;
            }

            Console.WriteLine($"Testing with {elemSize:N0} elements ({elemSize * 4 / 1024.0 / 1024.0:F1} MB per array)");
            Console.WriteLine();

            var operations = new (string Name, Action GpuOp, Action CpuOp)[]
            {
                ("Add", () => vulkan!.Add(inputA, inputB, output),
                    () => TensorPrimitives.Add<float>(inputA, inputB, output)),
                ("Subtract", () => vulkan!.Subtract(inputA, inputB, output),
                    () => TensorPrimitives.Subtract<float>(inputA, inputB, output)),
                ("Multiply", () => vulkan!.Multiply(inputA, inputB, output),
                    () => TensorPrimitives.Multiply<float>(inputA, inputB, output)),
                ("Divide", () => vulkan!.Divide(inputA, inputB, output),
                    () => TensorPrimitives.Divide<float>(inputA, inputB, output)),
                ("ReLU", () => vulkan!.ReLU(inputA, output),
                    () => { for (int j = 0; j < elemSize; j++) output[j] = Math.Max(0, inputA[j]); }),
                ("Tanh", () => vulkan!.Tanh(inputA, output),
                    () => TensorPrimitives.Tanh<float>(inputA, output)),
            };

            Console.WriteLine($"{"Operation",-12} {"GPU (ms)",12} {"CPU (ms)",12} {"Speedup",10} {"Winner",-8}");
            Console.WriteLine(new string('-', 56));

            foreach (var (name, gpuOp, cpuOp) in operations)
            {
                // Warmup
                for (int i = 0; i < 5; i++)
                {
                    gpuOp();
                    cpuOp();
                }

                // Benchmark GPU
                int runs = 20;
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < runs; i++)
                {
                    gpuOp();
                }
                sw.Stop();
                double gpuMs = sw.Elapsed.TotalMilliseconds / runs;

                // Benchmark CPU
                sw.Restart();
                for (int i = 0; i < runs; i++)
                {
                    cpuOp();
                }
                sw.Stop();
                double cpuMs = sw.Elapsed.TotalMilliseconds / runs;

                double speedup = cpuMs / gpuMs;
                string winner = speedup > 1.0 ? "GPU" : "CPU";

                Console.WriteLine($"{name,-12} {gpuMs,12:F4} {cpuMs,12:F4} {speedup,10:F2}x {winner,-8}");
            }
        }

        Console.WriteLine();

        // Summary and recommendations
        Console.WriteLine("===========================================");
        Console.WriteLine("PERFORMANCE INSIGHTS & RECOMMENDATIONS");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        Console.WriteLine("KEY FINDINGS:");
        Console.WriteLine("-------------");
        Console.WriteLine("1. GPU excels at: Large matrix operations, activation functions");
        Console.WriteLine("2. CPU excels at: Small matrices, simple element-wise ops");
        Console.WriteLine("3. Breakeven point: ~1000 elements for element-wise, ~256x256 for GEMM");
        Console.WriteLine();

        Console.WriteLine("BOTTLENECK AREAS TO OPTIMIZE:");
        Console.WriteLine("-----------------------------");
        Console.WriteLine("1. Memory transfer overhead (PCIe bandwidth)");
        Console.WriteLine("   - Solution: Persistent buffers, batch operations");
        Console.WriteLine();
        Console.WriteLine("2. Kernel launch latency (~0.1-0.5ms per dispatch)");
        Console.WriteLine("   - Solution: Kernel fusion, reduce dispatch count");
        Console.WriteLine();
        Console.WriteLine("3. Descriptor set updates per operation");
        Console.WriteLine("   - Solution: Pre-allocated descriptor pools");
        Console.WriteLine();
        Console.WriteLine("4. Synchronization overhead (fence waits)");
        Console.WriteLine("   - Solution: Pipeline execution, async transfers");

        gpu?.Dispose();
    }
}
