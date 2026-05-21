#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// FP64 Conv2D backward benchmarks — dW + dX across the {kernel × stride × padding}
/// matrix that cluster #6 (VGG16/ResNet50/DenseNet) exercises at ImageNet input
/// scale. Stage 0 of the FP64 perf-closure effort: gives us per-shape baselines
/// to A/B Stages 1/2/5/7 against.
///
/// Shape catalog covers:
///   - 3×3 stride=1 padding=1 (every VGG conv + ResNet50 residual block 3×3)
///   - 3×3 stride=2 padding=1 (ResNet50 stage-transition convs)
///   - 1×1 stride=1 padding=0 (ResNet50 bottleneck projections — 32 per forward)
///   - 1×1 stride=2 padding=0 (ResNet50 stage downsamples)
///   - 5×5 stride=1 padding=2 (various)
///   - 7×7 stride=2 padding=3 (ResNet50 conv1 stem)
///
/// At canonical ImageNet feature-map spatial dims: 224, 112, 56, 28, 14, 7.
///
/// Run:
///   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks \
///     -- --conv2d-backward-double
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 2, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class Conv2DBackwardDoubleBenchmarks
{
    [ParamsAllValues]
    public bool DeterministicMode { get; set; }

    [ParamsAllValues]
    public bool UseParallelGemm { get; set; }

    private CpuEngine _engine = null!;

    // ═══ 3×3 stride=1 padding=1 at every VGG/ResNet feature-map size ═══
    // Each tuple: (inC, outC, H, W) for input [1, inC, H, W], kernel [outC, inC, 3, 3]
    // VGG block 1: (3→64)/64→64 at 224×224
    // VGG block 2: 64→128/128→128 at 112×112
    // VGG block 3: 128→256/256→256 at 56×56
    // VGG block 4: 256→512/512→512 at 28×28
    // VGG block 5: 512→512 at 14×14
    private Tensor<double> _grad_64x224 = null!;       // VGG block 1 (post-conv)
    private Tensor<double> _input_64x224 = null!;
    private Tensor<double> _kernel_3x3_64x64 = null!;

    private Tensor<double> _grad_128x112 = null!;      // VGG block 2
    private Tensor<double> _input_128x112 = null!;
    private Tensor<double> _kernel_3x3_128x128 = null!;

    private Tensor<double> _grad_256x56 = null!;       // VGG block 3 / ResNet stage 2
    private Tensor<double> _input_256x56 = null!;
    private Tensor<double> _kernel_3x3_256x256 = null!;

    private Tensor<double> _grad_512x28 = null!;       // VGG block 4 / ResNet stage 3
    private Tensor<double> _input_512x28 = null!;
    private Tensor<double> _kernel_3x3_512x512 = null!;

    private Tensor<double> _grad_512x14 = null!;       // VGG block 5
    private Tensor<double> _input_512x14 = null!;

    // ═══ 1×1 stride=1 (ResNet50 bottleneck inner projections) ═══
    // Bottleneck pattern: 1×1 reduce → 3×3 → 1×1 expand
    // Stage 2: 256→64 (reduce) / 64→256 (expand) at 56×56
    // Stage 3: 512→128 / 128→512 at 28×28
    // Stage 4: 1024→256 / 256→1024 at 14×14
    // Stage 5: 2048→512 / 512→2048 at 7×7
    private Tensor<double> _grad_64x56_1x1 = null!;    // bottleneck reduce 256→64
    private Tensor<double> _input_256x56_1x1 = null!;
    private Tensor<double> _kernel_1x1_64x256 = null!;

    private Tensor<double> _grad_2048x7_1x1 = null!;   // stage-5 expand
    private Tensor<double> _input_512x7_1x1 = null!;
    private Tensor<double> _kernel_1x1_2048x512 = null!;

    // ═══ 7×7 stride=2 padding=3 (ResNet50 stem conv1) ═══
    private Tensor<double> _grad_64x112_7x7 = null!;   // post-stem
    private Tensor<double> _input_3x224 = null!;
    private Tensor<double> _kernel_7x7_64x3 = null!;

    // ═══ 3×3 stride=2 padding=1 (ResNet50 stage transitions) ═══
    private Tensor<double> _grad_128x28_3x3s2 = null!;
    private Tensor<double> _input_128x56_3x3s2 = null!;
    private Tensor<double> _kernel_3x3_128x128_s2 = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        // 3×3 s=1 p=1
        _grad_64x224  = Tensor<double>.CreateRandom([1, 64, 224, 224]);
        _input_64x224 = Tensor<double>.CreateRandom([1, 64, 224, 224]);
        _kernel_3x3_64x64 = Tensor<double>.CreateRandom([64, 64, 3, 3]);

        _grad_128x112 = Tensor<double>.CreateRandom([1, 128, 112, 112]);
        _input_128x112 = Tensor<double>.CreateRandom([1, 128, 112, 112]);
        _kernel_3x3_128x128 = Tensor<double>.CreateRandom([128, 128, 3, 3]);

        _grad_256x56  = Tensor<double>.CreateRandom([1, 256, 56, 56]);
        _input_256x56 = Tensor<double>.CreateRandom([1, 256, 56, 56]);
        _kernel_3x3_256x256 = Tensor<double>.CreateRandom([256, 256, 3, 3]);

        _grad_512x28  = Tensor<double>.CreateRandom([1, 512, 28, 28]);
        _input_512x28 = Tensor<double>.CreateRandom([1, 512, 28, 28]);
        _kernel_3x3_512x512 = Tensor<double>.CreateRandom([512, 512, 3, 3]);

        _grad_512x14  = Tensor<double>.CreateRandom([1, 512, 14, 14]);
        _input_512x14 = Tensor<double>.CreateRandom([1, 512, 14, 14]);

        // 1×1 s=1
        _grad_64x56_1x1 = Tensor<double>.CreateRandom([1, 64, 56, 56]);
        _input_256x56_1x1 = Tensor<double>.CreateRandom([1, 256, 56, 56]);
        _kernel_1x1_64x256 = Tensor<double>.CreateRandom([64, 256, 1, 1]);

        _grad_2048x7_1x1 = Tensor<double>.CreateRandom([1, 2048, 7, 7]);
        _input_512x7_1x1 = Tensor<double>.CreateRandom([1, 512, 7, 7]);
        _kernel_1x1_2048x512 = Tensor<double>.CreateRandom([2048, 512, 1, 1]);

        // 7×7 s=2 p=3
        _grad_64x112_7x7 = Tensor<double>.CreateRandom([1, 64, 112, 112]);
        _input_3x224     = Tensor<double>.CreateRandom([1, 3, 224, 224]);
        _kernel_7x7_64x3 = Tensor<double>.CreateRandom([64, 3, 7, 7]);

        // 3×3 s=2 p=1
        _grad_128x28_3x3s2  = Tensor<double>.CreateRandom([1, 128, 28, 28]);
        _input_128x56_3x3s2 = Tensor<double>.CreateRandom([1, 128, 56, 56]);
        _kernel_3x3_128x128_s2 = Tensor<double>.CreateRandom([128, 128, 3, 3]);
    }

    [IterationSetup]
    public void IterationSetup()
    {
        AiDotNetEngine.SetDeterministicMode(DeterministicMode);
        SimdGemm.UseParallelGemm = UseParallelGemm;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        AiDotNetEngine.SetDeterministicMode(false);
        SimdGemm.UseParallelGemm = true;
    }

    // ───────────── 3×3 stride=1 padding=1 dW (kernel gradient) ─────────────

    [Benchmark(Description = "dW 3x3 s1 p1 [1,64,224,224]→[1,64,224,224]")]
    public Tensor<double> dW_3x3_s1p1_b64_h224()
        => _engine.Conv2DBackwardKernel(_grad_64x224, _input_64x224,
            new[] { 64, 64, 3, 3 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    [Benchmark(Description = "dW 3x3 s1 p1 [1,128,112,112]→[1,128,112,112]")]
    public Tensor<double> dW_3x3_s1p1_b128_h112()
        => _engine.Conv2DBackwardKernel(_grad_128x112, _input_128x112,
            new[] { 128, 128, 3, 3 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    [Benchmark(Description = "dW 3x3 s1 p1 [1,256,56,56]→[1,256,56,56]")]
    public Tensor<double> dW_3x3_s1p1_b256_h56()
        => _engine.Conv2DBackwardKernel(_grad_256x56, _input_256x56,
            new[] { 256, 256, 3, 3 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    [Benchmark(Description = "dW 3x3 s1 p1 [1,512,28,28]→[1,512,28,28]")]
    public Tensor<double> dW_3x3_s1p1_b512_h28()
        => _engine.Conv2DBackwardKernel(_grad_512x28, _input_512x28,
            new[] { 512, 512, 3, 3 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    [Benchmark(Description = "dW 3x3 s1 p1 [1,512,14,14]→[1,512,14,14]")]
    public Tensor<double> dW_3x3_s1p1_b512_h14()
        => _engine.Conv2DBackwardKernel(_grad_512x14, _input_512x14,
            new[] { 512, 512, 3, 3 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    // ───────────── 3×3 stride=1 padding=1 dX (input gradient) ─────────────

    [Benchmark(Description = "dX 3x3 s1 p1 [1,64,224,224]")]
    public Tensor<double> dX_3x3_s1p1_b64_h224()
        => _engine.Conv2DBackwardInput(_grad_64x224, _kernel_3x3_64x64,
            new[] { 1, 64, 224, 224 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    [Benchmark(Description = "dX 3x3 s1 p1 [1,128,112,112]")]
    public Tensor<double> dX_3x3_s1p1_b128_h112()
        => _engine.Conv2DBackwardInput(_grad_128x112, _kernel_3x3_128x128,
            new[] { 1, 128, 112, 112 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    [Benchmark(Description = "dX 3x3 s1 p1 [1,256,56,56]")]
    public Tensor<double> dX_3x3_s1p1_b256_h56()
        => _engine.Conv2DBackwardInput(_grad_256x56, _kernel_3x3_256x256,
            new[] { 1, 256, 56, 56 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    [Benchmark(Description = "dX 3x3 s1 p1 [1,512,28,28]")]
    public Tensor<double> dX_3x3_s1p1_b512_h28()
        => _engine.Conv2DBackwardInput(_grad_512x28, _kernel_3x3_512x512,
            new[] { 1, 512, 28, 28 }, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

    // ───────────── 1×1 stride=1 padding=0 dW + dX ─────────────

    [Benchmark(Description = "dW 1x1 s1 [1,64,56,56]/[1,256,56,56]")]
    public Tensor<double> dW_1x1_s1_64_256()
        => _engine.Conv2DBackwardKernel(_grad_64x56_1x1, _input_256x56_1x1,
            new[] { 64, 256, 1, 1 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

    [Benchmark(Description = "dW 1x1 s1 [1,2048,7,7]/[1,512,7,7]")]
    public Tensor<double> dW_1x1_s1_2048_512()
        => _engine.Conv2DBackwardKernel(_grad_2048x7_1x1, _input_512x7_1x1,
            new[] { 2048, 512, 1, 1 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

    [Benchmark(Description = "dX 1x1 s1 [1,64,56,56]/k[64,256,1,1]")]
    public Tensor<double> dX_1x1_s1_64_256()
        => _engine.Conv2DBackwardInput(_grad_64x56_1x1, _kernel_1x1_64x256,
            new[] { 1, 256, 56, 56 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

    [Benchmark(Description = "dX 1x1 s1 [1,2048,7,7]/k[2048,512,1,1]")]
    public Tensor<double> dX_1x1_s1_2048_512()
        => _engine.Conv2DBackwardInput(_grad_2048x7_1x1, _kernel_1x1_2048x512,
            new[] { 1, 512, 7, 7 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

    // ───────────── 7×7 stride=2 padding=3 (ResNet stem) ─────────────

    [Benchmark(Description = "dW 7x7 s2 p3 ResNet stem [1,3,224,224]→[1,64,112,112]")]
    public Tensor<double> dW_7x7_s2p3_stem()
        => _engine.Conv2DBackwardKernel(_grad_64x112_7x7, _input_3x224,
            new[] { 64, 3, 7, 7 }, new[] { 2, 2 }, new[] { 3, 3 }, new[] { 1, 1 });

    [Benchmark(Description = "dX 7x7 s2 p3 ResNet stem")]
    public Tensor<double> dX_7x7_s2p3_stem()
        => _engine.Conv2DBackwardInput(_grad_64x112_7x7, _kernel_7x7_64x3,
            new[] { 1, 3, 224, 224 }, new[] { 2, 2 }, new[] { 3, 3 }, new[] { 1, 1 });

    // ───────────── 3×3 stride=2 padding=1 (ResNet transitions) ─────────────

    [Benchmark(Description = "dW 3x3 s2 p1 [1,128,28,28]/[1,128,56,56]")]
    public Tensor<double> dW_3x3_s2p1()
        => _engine.Conv2DBackwardKernel(_grad_128x28_3x3s2, _input_128x56_3x3s2,
            new[] { 128, 128, 3, 3 }, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });

    [Benchmark(Description = "dX 3x3 s2 p1 [1,128,28,28]/k[128,128,3,3]")]
    public Tensor<double> dX_3x3_s2p1()
        => _engine.Conv2DBackwardInput(_grad_128x28_3x3s2, _kernel_3x3_128x128_s2,
            new[] { 1, 128, 56, 56 }, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });
}
#endif
