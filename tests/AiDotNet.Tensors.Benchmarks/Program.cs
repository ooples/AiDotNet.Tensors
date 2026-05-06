using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using AiDotNet.Tensors.Benchmarks;

namespace AiDotNet.Tensors.Benchmarks;

class Program
{
    // TensorFlow.Binding NuGet is not built in Release mode, so we must disable the validator
    private static readonly IConfig BenchConfig = DefaultConfig.Instance
        .WithOptions(ConfigOptions.DisableOptimizationsValidator)
        .WithBuildTimeout(TimeSpan.FromMinutes(5));

    static void Main(string[] args)
    {
        // Run quick performance test first for immediate feedback
        if (args.Length == 0 || args[0] == "--quick")
        {
            QuickPerformanceTest.Run();
            return;
        }

        // #209 close-parity: A/B test the Conv3x3Stride1 variants
        // (per-channel, 2-oc-blocked, 4-oc-blocked) on shared shapes
        // in the same process. Faster turnaround than full BDN.
        if (args[0] == "--ab-conv2d")
        {
            Conv2DAbBench.Run();
            return;
        }

        // Softmax<double> micro-benchmark — same-process measurement
        // for the new SoftmaxRowDoubleUnsafe SIMD kernel.
        if (args[0] == "--ab-softmax-double")
        {
            Conv2DAbBench.RunSoftmaxDouble();
            return;
        }

        if (args[0] == "--ab-attention-qkt")
        {
            Conv2DAbBench.RunAttentionQkt();
            return;
        }

        if (args[0] == "--ab-layernorm")
        {
            Conv2DAbBench.RunLayerNorm();
            return;
        }

        if (args[0] == "--ab-matmul")
        {
            Conv2DAbBench.RunMatMul();
            return;
        }

        if (args[0] == "--ab-conv2d-double")
        {
            Conv2DAbBench.RunConv2DDouble();
            return;
        }

        if (args[0] == "--ab-binary-ops")
        {
            Conv2DAbBench.RunBinaryOps();
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
            BenchmarkRunner.Run<LinearAlgebraBenchmarks>(BenchConfig);
            BenchmarkRunner.Run<SmallMatrixBenchmarks>(BenchConfig);
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

        // GPU activation benchmarks (ReLU, Sigmoid, Tanh, GELU, Softmax)
        if (args[0] == "--activation")
        {
            GpuActivationBenchmark.Run();
            return;
        }

        // GPU normalization benchmarks (BN, LN, GN, IN, RmsNorm)
        if (args[0] == "--norm")
        {
            GpuNormalizationBenchmark.Run();
            return;
        }

        // GPU attention benchmarks (FlashAttention, ScaledDotProduct)
        if (args[0] == "--attn")
        {
            GpuAttentionBenchmark.Run();
            return;
        }

        // GPU convolution benchmarks (Conv2D, Depthwise)
        if (args[0] == "--conv")
        {
            GpuConvolutionBenchmark.Run();
            return;
        }

        // Capture full baseline to CSV for A/B testing across phases
        if (args[0] == "--baseline")
        {
            string phase = args.Length > 1 ? args[1] : "phase0";
            string? csvPath = args.Length > 2 ? args[2] : null;
            GpuBaselineResults.CaptureBaseline(phase, csvPath);
            return;
        }

        // Compare two baseline CSVs for regressions
        if (args[0] == "--compare")
        {
            if (args.Length < 3)
            {
                Console.WriteLine("Usage: --compare <before.csv> <after.csv>");
                return;
            }
            GpuBaselineResults.CompareBaselines(args[1], args[2]);
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

        // Run competitive benchmarks vs TorchSharp (GPU)
        if (args[0] == "--vs-torchsharp-gpu")
        {
            BenchmarkRunner.Run<TorchSharpComparisonBenchmarks>(BenchConfig);
            return;
        }

        // Run competitive benchmarks vs TorchSharp (CPU)
        if (args[0] == "--vs-torchsharp-cpu")
        {
            BenchmarkRunner.Run<TorchSharpCpuComparisonBenchmarks>(BenchConfig);
            return;
        }

        // Issue #296 acceptance harness: real-async ChainAsync vs PyTorch
        if (args[0] == "--296-chain")
        {
            BenchmarkRunner.Run<CompiledPlanChainingBenchmarks>(BenchConfig);
            return;
        }

        if (args[0] == "--296-throughput")
        {
            BenchmarkRunner.Run<StreamThroughputBenchmark>(BenchConfig);
            return;
        }

        // Run competitive benchmarks vs TensorFlow (GPU)
        if (args[0] == "--vs-tensorflow-gpu")
        {
            BenchmarkRunner.Run<TensorFlowComparisonBenchmarks>(BenchConfig);
            return;
        }

        // Run competitive benchmarks vs TensorFlow (CPU)
        if (args[0] == "--vs-tensorflow-cpu")
        {
            BenchmarkRunner.Run<TensorFlowCpuComparisonBenchmarks>(BenchConfig);
            return;
        }

        // Run competitive benchmarks vs ML.NET (CPU)
        if (args[0] == "--vs-mlnet-cpu")
        {
            BenchmarkRunner.Run<MlNetCpuComparisonBenchmarks>(BenchConfig);
            return;
        }

        // Run TensorWorkspace zero-allocation benchmarks
        if (args[0] == "--workspace")
        {
            BenchmarkRunner.Run<TensorWorkspaceBenchmarks>(BenchConfig);
            return;
        }

        // Run layer-level benchmarks (ResBlock, Attention)
        if (args[0] == "--layers")
        {
            BenchmarkRunner.Run<LayerLevelBenchmarks>(BenchConfig);
            return;
        }

        // Run UNet-scale model benchmarks
        if (args[0] == "--unet")
        {
            BenchmarkRunner.Run<UNetForwardBenchmarks>(BenchConfig);
            return;
        }

        // Run autograd + compiled plan vs PyTorch comparison
        if (args[0] == "--vs-autograd")
        {
            BenchmarkRunner.Run<AutogradComparisonBenchmarks>(BenchConfig);
            return;
        }

        // Run TensorCodec comprehensive vs PyTorch benchmarks (full suite ~40min)
        if (args[0] == "--vs-tensorcodec")
        {
            BenchmarkRunner.Run<TensorCodecVsPyTorchBenchmarks>(BenchConfig);
            return;
        }

        // Run deterministic vs MKL matmul A/B benchmarks (Issue #131 Step 2, ~5-15min)
        if (args[0] == "--vs-deterministic-matmul")
        {
            BenchmarkRunner.Run<DeterministicMatMulBenchmarks>(BenchConfig);
            return;
        }

        // DiT-XL A/B matmul benchmarks: our blocked C# GEMM vs MKL.NET at the exact
        // shapes a DiT-XL forward pass exercises. Used as the acceptance gate for
        // the "finish MKL replacement" feature branch (~10-20min).
        if (args[0] == "--dit-xl-matmul")
        {
            BenchmarkRunner.Run<DitXLMatMulBenchmarks>(BenchConfig);
            return;
        }

        // DiT-XL SDPA benchmark: measures ScaledDotProductAttention at the exact
        // 4D shape DiT-XL exercises, to verify the Issue #162 BLAS-backed fast
        // path eliminates the scalar virtual-dispatch wall clock. (~1-2min).
        if (args[0] == "--dit-xl-sdpa")
        {
            BenchmarkRunner.Run<DitXLSdpaBenchmarks>(BenchConfig);
            return;
        }

        // Transformer-FFN matmul suite (Issue #245): covers small-M
        // tall-skinny shapes from ChronosBolt/MOMENT/TimesFM/VisionTS that
        // DiT-XL square benchmarks miss. Acceptance surface for Issues
        // #242/#243/#244 fixes (~8-15min).
        if (args[0] == "--transformer-ffn")
        {
            BenchmarkRunner.Run<TransformerFFNBenchmarks>(BenchConfig);
            return;
        }

        // Run TensorCodec gaps only — focused on operations still losing to PyTorch (~15min)
        if (args[0] == "--vs-tensorcodec-gaps")
        {
            var gapsConfig = ManualConfig.Create(BenchConfig)
                .AddFilter(new BenchmarkDotNet.Filters.SimpleFilter(
                    b => b.Descriptor.WorkloadMethodDisplayInfo.Contains("GELU")
                      || b.Descriptor.WorkloadMethodDisplayInfo.Contains("Log")
                      || b.Descriptor.WorkloadMethodDisplayInfo.Contains("Abs")
                      || b.Descriptor.WorkloadMethodDisplayInfo.Contains("Pow")
                      || b.Descriptor.WorkloadMethodDisplayInfo.Contains("GroupNorm")
                      || b.Descriptor.WorkloadMethodDisplayInfo.Contains("LayerNorm")
                      || b.Descriptor.WorkloadMethodDisplayInfo.Contains("Mean")
                      || b.Descriptor.WorkloadMethodDisplayInfo.Contains("LogSoftmax")
                      || b.Descriptor.WorkloadMethodDisplayInfo.Contains("MLP_MSE")));
            BenchmarkRunner.Run<TensorCodecVsPyTorchBenchmarks>(gapsConfig);
            return;
        }

        // Run all competitive benchmarks (TorchSharp, ML.NET, TensorFlow CPU)
        if (args[0] == "--vs-all")
        {
            BenchmarkRunner.Run<TorchSharpCpuComparisonBenchmarks>(BenchConfig);
            BenchmarkRunner.Run<MlNetCpuComparisonBenchmarks>(BenchConfig);
            BenchmarkRunner.Run<TensorFlowCpuComparisonBenchmarks>(BenchConfig);
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
        Console.WriteLine("  --activation: Run GPU activation benchmarks (ReLU, Sigmoid, Tanh, GELU, Softmax)");
        Console.WriteLine("  --norm      : Run GPU normalization benchmarks (BN, LN, GN, IN, RmsNorm)");
        Console.WriteLine("  --attn      : Run GPU attention benchmarks (FlashAttention, SDPA)");
        Console.WriteLine("  --conv      : Run GPU convolution benchmarks (Conv2D, Depthwise)");
        Console.WriteLine("  --baseline  : Capture full baseline to CSV (--baseline [phase] [csv])");
        Console.WriteLine("  --compare   : Compare baselines (--compare before.csv after.csv)");
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
        Console.WriteLine();
        Console.WriteLine("Competitive Benchmarks vs Other Libraries:");
        Console.WriteLine("  --vs-torchsharp-gpu : AiDotNet GPU vs TorchSharp CUDA");
        Console.WriteLine("  --vs-autograd       : AiDotNet autograd + compiled plan vs PyTorch");
        Console.WriteLine("  --vs-tensorcodec    : TensorCodec comprehensive vs PyTorch (full suite ~40min)");
        Console.WriteLine("  --vs-tensorcodec-gaps: TensorCodec gaps only (ops losing to PyTorch ~15min)");
        Console.WriteLine("  --vs-torchsharp-cpu : AiDotNet CPU vs TorchSharp CPU");
        Console.WriteLine("  --vs-tensorflow-gpu : AiDotNet GPU vs TensorFlow.NET GPU");
        Console.WriteLine("  --vs-tensorflow-cpu : AiDotNet CPU vs TensorFlow.NET CPU");
        Console.WriteLine("  --vs-mlnet-cpu      : AiDotNet CPU vs ML.NET");
        Console.WriteLine("  --vs-all            : Run all CPU competitive benchmarks");
        Console.WriteLine("  --workspace         : Run TensorWorkspace zero-allocation benchmarks");
        Console.WriteLine("  --vs-deterministic-matmul: Deterministic vs non-deterministic SimdGemm on HRE + square shapes (post-MKL-removal both paths are SimdGemm; pair with iter-17 MKL baseline for vs-MKL comparison)");
        Console.WriteLine("  --dit-xl-matmul     : SimdGemm at DiT-XL shapes; compare against docs/mkl-replacement/baseline/baseline-iter17.md for vs-MKL numbers");
        Console.WriteLine("  --dit-xl-sdpa       : ScaledDotProductAttention at DiT-XL shape [4,16,256,72] (Issue #162 SDPA fix)");
        Console.WriteLine("  --transformer-ffn   : Small-M transformer FFN matmul (Sgemm+Dgemm+batched) — Issue #245 coverage");
#endif
    }
}
