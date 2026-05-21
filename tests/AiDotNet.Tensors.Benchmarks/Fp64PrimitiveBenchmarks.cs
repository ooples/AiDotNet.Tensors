#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// FP64 primitive ops: BatchNorm, LayerNorm, Softmax, activations, pools.
/// Stage 0 of the FP64 perf-closure effort: gives us a baseline for the
/// ops that are CURRENTLY SCALAR in FP64 (per the audit in
/// `docs/mkl-replacement/PLAN.md` Stage 4) so Stage 4's Vector256&lt;double&gt;
/// SIMD ports can be A/B'd.
///
/// Shapes target cluster-#6 hot paths:
///   - BN at VGG block dims (64×224, 128×112, 256×56, 512×28, 512×14)
///     and ResNet50 BN (50 BN per step, at the same dims as the convs)
///   - LN at ACEStep MHA hidden (128 tokens × 512 hidden)
///   - Softmax at ACEStep MHA attention (128 × 128 attention scores)
///   - Activations at all VGG / ResNet feature-map sizes
///   - MaxPool / AvgPool at VGG pool dims
///
/// Run:
///   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks \
///     -- --fp64-primitives
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 2, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class Fp64PrimitiveBenchmarks
{
    private CpuEngine _engine = null!;

    // BatchNorm inputs at cluster-#6 feature-map sizes
    private Tensor<double> _bn_64x224 = null!;
    private Tensor<double> _bn_128x112 = null!;
    private Tensor<double> _bn_256x56 = null!;
    private Tensor<double> _bn_512x28 = null!;
    private Tensor<double> _bn_512x14 = null!;
    private Tensor<double> _bn_gamma_64 = null!;
    private Tensor<double> _bn_beta_64 = null!;
    private Tensor<double> _bn_gamma_128 = null!;
    private Tensor<double> _bn_beta_128 = null!;
    private Tensor<double> _bn_gamma_256 = null!;
    private Tensor<double> _bn_beta_256 = null!;
    private Tensor<double> _bn_gamma_512 = null!;
    private Tensor<double> _bn_beta_512 = null!;

    // LayerNorm inputs at ACEStep / transformer dims
    private Tensor<double> _ln_128x512 = null!;
    private Tensor<double> _ln_gamma_512 = null!;
    private Tensor<double> _ln_beta_512 = null!;

    // Softmax input (last-dim reduction): ACEStep attention scores
    private Tensor<double> _softmax_128x128 = null!;
    // Larger softmax (LM head logits, classification): 1000 classes
    private Tensor<double> _softmax_1x1000 = null!;

    // Activation inputs (1D and 4D) at VGG/ResNet shapes
    private Tensor<double> _act_64x224 = null!;
    private Tensor<double> _act_128x112 = null!;
    private Tensor<double> _act_256x56 = null!;
    private Tensor<double> _act_512x28 = null!;
    private Tensor<double> _act_4096 = null!;       // VGG FC2 output
    private Tensor<double> _act_25088 = null!;      // VGG FC1 input

    // Pool inputs at VGG pool dims (before each pool transition)
    private Tensor<double> _pool_64x224 = null!;    // → 64x112
    private Tensor<double> _pool_128x112 = null!;   // → 128x56
    private Tensor<double> _pool_256x56 = null!;    // → 256x28
    private Tensor<double> _pool_512x28 = null!;    // → 512x14
    private Tensor<double> _pool_512x14 = null!;    // → 512x7

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _bn_64x224  = Tensor<double>.CreateRandom([1, 64, 224, 224]);
        _bn_128x112 = Tensor<double>.CreateRandom([1, 128, 112, 112]);
        _bn_256x56  = Tensor<double>.CreateRandom([1, 256, 56, 56]);
        _bn_512x28  = Tensor<double>.CreateRandom([1, 512, 28, 28]);
        _bn_512x14  = Tensor<double>.CreateRandom([1, 512, 14, 14]);
        _bn_gamma_64  = Tensor<double>.CreateRandom([64]);
        _bn_beta_64   = Tensor<double>.CreateRandom([64]);
        _bn_gamma_128 = Tensor<double>.CreateRandom([128]);
        _bn_beta_128  = Tensor<double>.CreateRandom([128]);
        _bn_gamma_256 = Tensor<double>.CreateRandom([256]);
        _bn_beta_256  = Tensor<double>.CreateRandom([256]);
        _bn_gamma_512 = Tensor<double>.CreateRandom([512]);
        _bn_beta_512  = Tensor<double>.CreateRandom([512]);

        _ln_128x512   = Tensor<double>.CreateRandom([128, 512]);
        _ln_gamma_512 = Tensor<double>.CreateRandom([512]);
        _ln_beta_512  = Tensor<double>.CreateRandom([512]);

        _softmax_128x128 = Tensor<double>.CreateRandom([128, 128]);
        _softmax_1x1000  = Tensor<double>.CreateRandom([1, 1000]);

        _act_64x224  = Tensor<double>.CreateRandom([1, 64, 224, 224]);
        _act_128x112 = Tensor<double>.CreateRandom([1, 128, 112, 112]);
        _act_256x56  = Tensor<double>.CreateRandom([1, 256, 56, 56]);
        _act_512x28  = Tensor<double>.CreateRandom([1, 512, 28, 28]);
        _act_4096    = Tensor<double>.CreateRandom([1, 4096]);
        _act_25088   = Tensor<double>.CreateRandom([1, 25088]);

        _pool_64x224  = Tensor<double>.CreateRandom([1, 64, 224, 224]);
        _pool_128x112 = Tensor<double>.CreateRandom([1, 128, 112, 112]);
        _pool_256x56  = Tensor<double>.CreateRandom([1, 256, 56, 56]);
        _pool_512x28  = Tensor<double>.CreateRandom([1, 512, 28, 28]);
        _pool_512x14  = Tensor<double>.CreateRandom([1, 512, 14, 14]);
    }

    // ───────────── BatchNorm forward ─────────────

    [Benchmark(Description = "BN fwd [1,64,224,224]")]
    public Tensor<double> BN_64x224()
        => _engine.BatchNorm(_bn_64x224, _bn_gamma_64, _bn_beta_64, 1e-5, out _, out _);

    [Benchmark(Description = "BN fwd [1,128,112,112]")]
    public Tensor<double> BN_128x112()
        => _engine.BatchNorm(_bn_128x112, _bn_gamma_128, _bn_beta_128, 1e-5, out _, out _);

    [Benchmark(Description = "BN fwd [1,256,56,56]")]
    public Tensor<double> BN_256x56()
        => _engine.BatchNorm(_bn_256x56, _bn_gamma_256, _bn_beta_256, 1e-5, out _, out _);

    [Benchmark(Description = "BN fwd [1,512,28,28]")]
    public Tensor<double> BN_512x28()
        => _engine.BatchNorm(_bn_512x28, _bn_gamma_512, _bn_beta_512, 1e-5, out _, out _);

    [Benchmark(Description = "BN fwd [1,512,14,14]")]
    public Tensor<double> BN_512x14()
        => _engine.BatchNorm(_bn_512x14, _bn_gamma_512, _bn_beta_512, 1e-5, out _, out _);

    // ───────────── LayerNorm forward ─────────────

    [Benchmark(Description = "LN fwd [128,512] (ACEStep MHA hidden)")]
    public Tensor<double> LN_128x512()
        => _engine.LayerNorm(_ln_128x512, _ln_gamma_512, _ln_beta_512, 1e-5, out _, out _);

    // ───────────── Softmax forward ─────────────

    [Benchmark(Description = "Softmax [128,128] last-dim (ACEStep attn scores)")]
    public Tensor<double> Softmax_128x128() => _engine.Softmax(_softmax_128x128);

    [Benchmark(Description = "Softmax [1,1000] last-dim (LM head)")]
    public Tensor<double> Softmax_1x1000() => _engine.Softmax(_softmax_1x1000);

    // ───────────── Activations ─────────────

    [Benchmark(Description = "ReLU [1,64,224,224]")]
    public Tensor<double> ReLU_64x224() => _engine.ReLU(_act_64x224);

    [Benchmark(Description = "ReLU [1,128,112,112]")]
    public Tensor<double> ReLU_128x112() => _engine.ReLU(_act_128x112);

    [Benchmark(Description = "ReLU [1,512,28,28]")]
    public Tensor<double> ReLU_512x28() => _engine.ReLU(_act_512x28);

    [Benchmark(Description = "ReLU [1,4096] (VGG FC mid)")]
    public Tensor<double> ReLU_FC4096() => _engine.ReLU(_act_4096);

    [Benchmark(Description = "GELU [1,4096] (transformer-style FFN)")]
    public Tensor<double> GELU_FC4096() => _engine.GELU(_act_4096);

    // ───────────── MaxPool ─────────────

    [Benchmark(Description = "MaxPool 2x2 s=2 [1,64,224,224]→[1,64,112,112]")]
    public Tensor<double> MaxPool_64x224()
        => _engine.MaxPool2D(_pool_64x224, 2, 2);

    [Benchmark(Description = "MaxPool 2x2 s=2 [1,128,112,112]→[1,128,56,56]")]
    public Tensor<double> MaxPool_128x112()
        => _engine.MaxPool2D(_pool_128x112, 2, 2);

    [Benchmark(Description = "MaxPool 2x2 s=2 [1,256,56,56]→[1,256,28,28]")]
    public Tensor<double> MaxPool_256x56()
        => _engine.MaxPool2D(_pool_256x56, 2, 2);

    [Benchmark(Description = "MaxPool 2x2 s=2 [1,512,28,28]→[1,512,14,14]")]
    public Tensor<double> MaxPool_512x28()
        => _engine.MaxPool2D(_pool_512x28, 2, 2);

    [Benchmark(Description = "MaxPool 2x2 s=2 [1,512,14,14]→[1,512,7,7]")]
    public Tensor<double> MaxPool_512x14()
        => _engine.MaxPool2D(_pool_512x14, 2, 2);

    // ───────────── AveragePool ─────────────

    private Tensor<double> _avgpool_2048x7 = null!;

    [Benchmark(Description = "AvgPool 7x7 global [1,2048,7,7]→[1,2048,1,1] (ResNet head)")]
    public Tensor<double> AvgPool_ResNetHead()
    {
        _avgpool_2048x7 ??= Tensor<double>.CreateRandom([1, 2048, 7, 7]);
        return _engine.AvgPool2D(_avgpool_2048x7, 7, 1);
    }
}
#endif
