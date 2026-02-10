using BenchmarkDotNet.Running;
using AiDotNet.Tensors.Benchmarks;

namespace AiDotNet.Tensors.Benchmarks;

class Program
{
    static void Main(string[] args)
    {
        // Run quick performance test first for immediate feedback
        if (args.Length == 0 || args[0] == "--quick")
        {
            QuickPerformanceTest.Run();
            return;
        }

        // Run full BenchmarkDotNet suite if requested
        if (args[0] == "--full")
        {
            var summary = BenchmarkRunner.Run<TrigonometricOperatorBenchmarks>();
            return;
        }

        // Run linear algebra benchmarks
        if (args[0] == "--linalg")
        {
            BenchmarkRunner.Run<LinearAlgebraBenchmarks>();
            BenchmarkRunner.Run<SmallMatrixBenchmarks>();
            return;
        }

        if (args[0] == "--cpu-matmul")
        {
            CpuMatMulDiagnostics.Run();
            return;
        }

#if !NET462
        // Run cuBLAS vs DirectGpu GEMM benchmark
        if (args[0] == "--cublas")
        {
            CuBlasGemmBenchmark.Run();
            return;
        }

        // Run OpenCL GEMM benchmark (AMD/Intel GPUs)
        if (args[0] == "--opencl")
        {
            OpenClGemmBenchmark.Run();
            return;
        }

        // Run CLBlast vs AiDotNet OpenCL comparison benchmark
        if (args[0] == "--clblast")
        {
            ClBlastBenchmark.Run();
            return;
        }

        // Run DirectGpu comprehensive benchmark (all 10 optimizations)
        if (args[0] == "--directgpu")
        {
            DirectGpuGemmBenchmark.RunComprehensive();
            return;
        }

        // Run Vulkan backend diagnostics and bottleneck analysis
        if (args[0] == "--vulkan")
        {
            VulkanDiagnosticsBenchmark.Run();
            return;
        }

        // Run GPU memory transfer diagnostics
        if (args[0] == "--memory")
        {
            GpuMemoryDiagnostics.Run();
            return;
        }

        // Run cross-library performance analysis
        if (args[0] == "--crosslib")
        {
            CrossLibraryAnalysis.Run();
            return;
        }

        // Run GPU backend comparison
        if (args[0] == "--backends")
        {
            GpuBackendComparisonBenchmark.Run();
            return;
        }

        // Run full BenchmarkDotNet GPU benchmarks
        if (args[0] == "--vulkan-bench")
        {
            BenchmarkRunner.Run<VulkanBackendBenchmarks>();
            return;
        }

        // Run cross-library GPU BenchmarkDotNet suite
        if (args[0] == "--crosslib-bench")
        {
            BenchmarkRunner.Run<CrossLibraryGpuBenchmarks>();
            return;
        }

        // Run Metal backend diagnostics (macOS only)
        if (args[0] == "--metal")
        {
            MetalDiagnosticsBenchmark.Run();
            return;
        }

        // Run formal Metal backend benchmarks
        if (args[0] == "--metal-bench")
        {
            BenchmarkRunner.Run<MetalBackendBenchmarks>();
            BenchmarkRunner.Run<MetalGemmBenchmarks>();
            return;
        }

#if NET7_0_OR_GREATER
        // Run WebGPU backend diagnostics (browser context)
        if (args[0] == "--webgpu")
        {
            WebGpuDiagnosticsBenchmark.RunAsync().GetAwaiter().GetResult();
            return;
        }

        // Run formal WebGPU backend benchmarks
        if (args[0] == "--webgpu-bench")
        {
            BenchmarkRunner.Run<WebGpuBackendBenchmarks>();
            BenchmarkRunner.Run<WebGpuReductionBenchmarks>();
            return;
        }
#endif

        // Run all GPU backend comparison (Vulkan + Metal + WebGPU)
        if (args[0] == "--all-gpu")
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("ALL GPU BACKENDS COMPREHENSIVE BENCHMARK");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            Console.WriteLine("Running Vulkan diagnostics...");
            VulkanDiagnosticsBenchmark.Run();

            Console.WriteLine("Running Metal diagnostics...");
            MetalDiagnosticsBenchmark.Run();

#if NET7_0_OR_GREATER
            Console.WriteLine("Running WebGPU diagnostics...");
            WebGpuDiagnosticsBenchmark.RunAsync().GetAwaiter().GetResult();
#endif
            return;
        }
#endif

        Console.WriteLine("Usage:");
        Console.WriteLine("  --quick    : Run quick performance validation (default)");
        Console.WriteLine("  --full     : Run full BenchmarkDotNet suite (trigonometric)");
        Console.WriteLine("  --linalg   : Run linear algebra benchmarks vs MathNet.Numerics");
        Console.WriteLine("  --cpu-matmul: Run CPU matrix multiply diagnostics");
#if !NET462
        Console.WriteLine("  --cublas   : Run cuBLAS vs DirectGpu GEMM benchmark");
        Console.WriteLine("  --opencl   : Run OpenCL GEMM benchmark (AMD/Intel GPUs)");
        Console.WriteLine("  --clblast  : Run CLBlast vs AiDotNet OpenCL comparison (AMD/Intel)");
        Console.WriteLine("  --directgpu: Run DirectGpu comprehensive benchmark (all 10 optimizations)");
        Console.WriteLine();
        Console.WriteLine("GPU Bottleneck Analysis:");
        Console.WriteLine("  --vulkan   : Run Vulkan backend diagnostics and bottleneck analysis");
        Console.WriteLine("  --metal    : Run Metal backend diagnostics (macOS/Apple Silicon)");
        Console.WriteLine("  --webgpu   : Run WebGPU backend diagnostics (browser GPU compute)");
        Console.WriteLine("  --memory   : Run GPU memory transfer diagnostics");
        Console.WriteLine("  --crosslib : Run cross-library performance analysis (MathNet, NumSharp)");
        Console.WriteLine("  --backends : Run GPU backend comparison (OpenCL vs Vulkan)");
        Console.WriteLine("  --all-gpu  : Run ALL GPU backends comparison (Vulkan + Metal + WebGPU)");
        Console.WriteLine();
        Console.WriteLine("BenchmarkDotNet (formal benchmarks):");
        Console.WriteLine("  --vulkan-bench  : Run formal Vulkan backend benchmarks");
        Console.WriteLine("  --metal-bench   : Run formal Metal backend benchmarks (macOS)");
        Console.WriteLine("  --webgpu-bench  : Run formal WebGPU backend benchmarks (.NET 7+)");
        Console.WriteLine("  --crosslib-bench: Run formal cross-library GPU benchmarks");
#endif
    }
}
