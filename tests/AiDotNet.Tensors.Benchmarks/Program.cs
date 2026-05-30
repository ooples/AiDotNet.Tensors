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

        // Issue #436: same-machine head-to-head of the fused inference
        // primitives (MLP / MHA / LSTM) vs TorchSharp at the AIsEval shapes.
        // Win = AiDotNet p95 < PyTorch median.
        if (args[0] == "--ab-aiseval-h2h")
        {
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.AisEvalHeadToHeadBench.Run();
            return;
        }

        if (args[0] == "--ab-aiseval-diag")
        {
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.AisEvalHeadToHeadBench.Diag();
            return;
        }

        if (args[0] == "--ab-aiseval-floor")
        {
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.AisEvalHeadToHeadBench.RawGemmFloor();
            return;
        }

        if (args[0] == "--ab-aiseval-dopsweep")
        {
            AiDotNet.Tensors.Benchmarks.PyTorchComparison.AisEvalHeadToHeadBench.DopSweep();
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

        // #415 Phase F: same-process FP64 Conv2D backward probe at the 9
        // ResNet50/VGG16 shapes from the issue. Quick-feedback sibling of
        // the BDN-driven Conv2DBackwardDoubleBenchmarks.cs.
        if (args[0] == "--ab-conv2d-backward-double")
        {
            Conv2DAbBench.RunConv2DBackwardDouble();
            return;
        }

        if (args[0] == "--ab-binary-ops")
        {
            Conv2DAbBench.RunBinaryOps();
            return;
        }

        // Sub-G worst-loss diagnostic: identify which strategy/thread count
        // gives best BlasManaged perf for the 64×64×64 FP64 shape.
        if (args[0] == "--ab-blas-small-square-fp64")
        {
            Conv2DAbBench.RunBlasSmallSquareFp64();
            return;
        }

        // Sub-G #375 Layer D — differentiator benchmarks (structural wins PyTorch
        // can't match). Each runs standalone; the cold-start child workload is
        // dispatched via --cold-start-aidotnet.
        if (args[0] == "--cold-start-aidotnet")
        {
            Environment.Exit(PyTorchComparison.ColdStartBench.RunAiDotNetWorkload());
        }
        if (args[0] == "--cold-start")
        {
            PyTorchComparison.ColdStartBench.Run();
            return;
        }
        if (args[0] == "--determinism-bench")
        {
            PyTorchComparison.DeterminismBench.Run();
            return;
        }
        if (args[0] == "--per-call-threads")
        {
            PyTorchComparison.PerCallThreadsBench.Run();
            return;
        }
        if (args[0] == "--frozen-weight-inference")
        {
            PyTorchComparison.FrozenWeightInferenceBench.Run();
            return;
        }
        if (args[0] == "--pytorch-headtohead")
        {
            string outPath = args.Length > 1 ? args[1] : "artifacts/perf/pytorch-comparison.md";
            PyTorchComparison.HeadToHeadCatalogBench.Run(outPath);
            return;
        }
        if (args[0] == "--investigate-gap")
        {
            PyTorchComparison.GapInvestigationBench.Run();
            return;
        }
        if (args[0] == "--cache-probe")
        {
            PyTorchComparison.GapInvestigationBench.CacheProbe();
            return;
        }
        if (args[0] == "--hybrid-lever-check")
        {
            PyTorchComparison.GapInvestigationBench.LeverCheck();
            return;
        }
        if (args[0] == "--prewarm-autotune")
        {
            PyTorchComparison.GapInvestigationBench.PrewarmAutotune();
            return;
        }
        if (args[0] == "--selectstrategy-hotpath")
        {
            PyTorchComparison.GapInvestigationBench.SelectStrategyHotPath();
            return;
        }
        if (args[0] == "--hybrid-win")
        {
            PyTorchComparison.GapInvestigationBench.HybridWin();
            return;
        }

        // Issue #403 Phase A.3: per-substep allocation profile + shape catalog
        // for one pass through the DCGAN-step probe substeps. Cheap to run
        // (one call per substep) so reviewers can compare alloc shape against
        // the issue's hypothesis without spinning up a full BDN run.
        if (args[0] == "--dcgan-probe")
        {
            var profile = DCGANStepProbe.RunAllocationProfile();
            Console.WriteLine(profile.Format());

            Console.WriteLine();
            var timing = DCGANStepProbe.RunWallClockProfile();
            Console.WriteLine(timing.Format());

            Console.WriteLine();
            Console.WriteLine(DCGANStepProbe.RunBareGemmProbe());

            Console.WriteLine();
            using (var shapes = new AiDotNet.Tensors.Helpers.ShapeInstrumenter())
            {
                var probe = new DCGANStepProbe();
                probe.Setup();
                probe.Conv2DForward_Fp64();
                probe.Conv2DBackwardInput_Fp64();
                probe.Conv2DBackwardKernel_Fp64();
                probe.BatchNormForward_Fp64();
                probe.BatchNormBackward_Fp64();
                shapes.PrintCatalog();
            }
            return;
        }

#if NET8_0_OR_GREATER
        // Issue #294 acceptance criterion #6: PyTorch CPU parity
        // benchmark — matmul, FlashAttention, LayerNorm, BCE
        // forward+backward. Drives Python subprocess via
        // PythonBaselineRunner; if torch isn't installed, prints
        // AiDotNet timings only as a regression baseline.
        if (args[0] == "--pytorch-parity")
        {
            Issue294PyTorchParityBenchmark.Run();
            return;
        }
#endif

        // PR #319 grain-size migration regression check (wall-clock harness).
        if (args[0] == "--pr319-grain")
        {
            PR319GrainSizeHarness.Run();
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

        // Sub-S (#409) Phase S.1 — microkernel-only GFLOPS harness.
        if (args[0] == "--microkernel-gflops")
        {
            MicrokernelGflopsBench.Run();
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

        if (args[0] == "--296-diffusion")
        {
            BenchmarkRunner.Run<DiffusionPipelineBenchmark>(BenchConfig);
            return;
        }

        // Issue #308 acceptance harness: cross-position HRR sequence primitives.
        if (args[0] == "--308-vsa")
        {
            BenchmarkRunner.Run<VsaIssue308Benchmarks>(BenchConfig);
            return;
        }

#if NET8_0_OR_GREATER
        if (args[0] == "--305-init")
        {
            BenchmarkRunner.Run<Issue305FirstForwardInitBenchmarks>(BenchConfig);
            return;
        }

        if (args[0] == "--305-init-gpu")
        {
            Issue305GpuInitBenchmark.Run();
            return;
        }
#endif

        if (args[0] == "--304-gemv")
        {
            Issue304GemvBenchmark.Run();
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

        // FP64 companion to --dit-xl-matmul: includes DiT-XL shapes at double
        // precision PLUS the cluster #6 cluster (VGG FC layers, ResNet head /
        // bottleneck shapes, ACEStep MHA + MLP). Stage 0 baseline of the
        // FP64 perf-closure effort. (~15-25min).
        if (args[0] == "--dit-xl-matmul-double")
        {
            BenchmarkRunner.Run<DitXLMatMulDoubleBenchmarks>(BenchConfig);
            return;
        }

        // FP64 Conv2D backward (dW + dX) across the {kernel × stride × padding}
        // matrix that cluster #6 (VGG16 / ResNet50 / DenseNet) exercises at
        // ImageNet input scale. Stage 0 baseline for Stage 1 / 5 / 7 conv work.
        // (~15-30min).
        if (args[0] == "--conv2d-backward-double")
        {
            BenchmarkRunner.Run<Conv2DBackwardDoubleBenchmarks>(BenchConfig);
            return;
        }

        // FP64 primitive ops (BatchNorm, LayerNorm, Softmax, activations,
        // pools). Stage 0 baseline for Stage 4 SIMD ports of currently-scalar
        // FP64 primitives. (~10-20min).
        if (args[0] == "--fp64-primitives")
        {
            BenchmarkRunner.Run<Fp64PrimitiveBenchmarks>(BenchConfig);
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

        // Issue #327 Transformer TrainBatched baseline harness
        if (args[0] == "--327-transformer")
        {
            Issue327TransformerTrainBatchedBenchmark.Run();
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
        Console.WriteLine("  --microkernel-gflops: Microkernel-only GFLOPS vs register-FMA peak (Sub-S #409)");
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
        Console.WriteLine("  --dcgan-probe : Run DCGAN step allocation / shape probe (#403 Phase A)");
        Console.WriteLine("  --ab-aiseval-h2h      : AIsEval head-to-head (AiDotNet fused vs TorchSharp)");
        Console.WriteLine("  --ab-aiseval-diag     : AIsEval diagnostics (pooled vs baseline alloc/latency)");
        Console.WriteLine("  --ab-aiseval-floor    : AIsEval raw-GEMM floor + OpenBLAS/MKL floor check");
        Console.WriteLine("  --ab-aiseval-dopsweep : AIsEval MaxDOP sweep for MHA/LSTM");
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
        Console.WriteLine("  --dit-xl-matmul-double : FP64 GEMM at DiT-XL + cluster-#6 shapes; Stage 0 baseline (docs/mkl-replacement/PLAN.md)");
        Console.WriteLine("  --conv2d-backward-double : FP64 Conv2DBackward dW+dX across cluster-#6 shapes; Stage 0 baseline");
        Console.WriteLine("  --fp64-primitives   : FP64 BN/LN/Softmax/activations/pools; Stage 0 baseline for Stage 4");
        Console.WriteLine("  --dit-xl-sdpa       : ScaledDotProductAttention at DiT-XL shape [4,16,256,72] (Issue #162 SDPA fix)");
        Console.WriteLine("  --transformer-ffn   : Small-M transformer FFN matmul (Sgemm+Dgemm+batched) — Issue #245 coverage");
        Console.WriteLine();
        Console.WriteLine("A/B microkernel + dispatch benchmarks:");
        Console.WriteLine("  --ab-matmul            : A/B GEMM (FP32) across catalog shapes");
        Console.WriteLine("  --ab-conv2d            : A/B Conv2D (FP32)");
        Console.WriteLine("  --ab-conv2d-double     : A/B Conv2D (FP64)");
        Console.WriteLine("  --ab-blas-small-square-fp64 : A/B small-square FP64 GEMM microkernel");
        Console.WriteLine("  --ab-attention-qkt     : A/B attention Q·Kᵀ");
        Console.WriteLine("  --ab-softmax-double    : A/B softmax (FP64)");
        Console.WriteLine("  --ab-layernorm         : A/B LayerNorm");
        Console.WriteLine("  --ab-binary-ops        : A/B elementwise binary ops");
        Console.WriteLine();
        Console.WriteLine("PyTorch-comparison diagnostics:");
        Console.WriteLine("  --per-call-threads       : Per-call NumThreads sweep vs PyTorch");
        Console.WriteLine("  --frozen-weight-inference: Frozen-weight (pre-packed) inference vs PyTorch");
        Console.WriteLine("  --pytorch-headtohead     : Head-to-head catalog (--pytorch-headtohead [out.md])");
        Console.WriteLine("  --investigate-gap        : Single/multi-thread kernel-gap investigation vs Torch");
        Console.WriteLine("  --cold-start             : Cold-start latency (AiDotNet vs PyTorch subprocess)");
        Console.WriteLine("  --cold-start-aidotnet    : Cold-start latency (AiDotNet only)");
        Console.WriteLine("  --determinism-bench      : Deterministic vs fast-mode reduction cost");
        Console.WriteLine("  --layers                 : End-to-end layer (Linear/Conv/Norm) microbench");
        Console.WriteLine("  --unet                   : U-Net step microbench");
        Console.WriteLine("  --pr319-grain            : PR #319 grain-size migration regression harness");
#if NET8_0_OR_GREATER
        Console.WriteLine("  --pytorch-parity         : PyTorch CPU parity (matmul/FlashAttn/LayerNorm/BCE fwd+bwd, .NET 8+)");
#endif
        Console.WriteLine();
        Console.WriteLine("Issue #296 acceptance benchmarks (real async ICompiledPlan):");
        Console.WriteLine("  --296-chain         : Single-batch latency vs PyTorch (BS=1/32/128, two-stage Linear→ReLU→Linear)");
        Console.WriteLine("  --296-throughput    : Multi-batch pipelined throughput vs PyTorch (NumBatches=8/32)");
        Console.WriteLine("  --296-diffusion     : 50-step denoising loop vs PyTorch nn.Sequential");
        Console.WriteLine("  --304-gemv          : Issue #304 [N,D]x[D,1] GEMV compiled-cache benchmark vs TorchSharp CPU");
        Console.WriteLine("  --308-vsa           : Cross-position HRR ShiftSlots/HrrBindShifted benchmark");
#if NET8_0_OR_GREATER
        Console.WriteLine("  --305-init          : First-forward weight-init peak benchmark vs old temp+copy and TorchSharp");
        Console.WriteLine("  --305-init-gpu      : GPU random initialization benchmark for Issue #305");
        Console.WriteLine("  --327-transformer   : Issue #327 Transformer TrainBatched d=128/L=4/B=32 baseline harness");
#endif
#endif
    }
}
