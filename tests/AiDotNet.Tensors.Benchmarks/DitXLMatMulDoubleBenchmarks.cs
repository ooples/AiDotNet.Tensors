#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// FP64 companion to <see cref="DitXLMatMulBenchmarks"/>. Stage 0 of the
/// `docs/mkl-replacement/PLAN.md` Stage-0/1 FP64 closure effort: gives us a
/// baseline to A/B every subsequent SimdGemmDouble port against (2% revert
/// rule per `docs/mkl-replacement/baseline/iter31-35-results.md`).
///
/// Adds the FP64 shape coverage missing today:
///   - VGG16 FC backward shapes (FC1 [25088,4096], FC2 [4096,4096], FC3 [4096,1000])
///   - ResNet50 head shape [batch,2048]×[2048,1000]
///   - ACEStep per-head MHA shapes ([seq=128, 64] × [64, seq=128], [128,128]×[128,64])
///   - ACEStep MLP FFN shapes [128,512]×[512,2048] and [128,2048]×[2048,512]
///   - The exact DiT-XL shapes mirrored for FP64 (so we can A/B vs FP32 iter34 numbers)
///
/// Run:
///   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks \
///     -- --dit-xl-matmul-double
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 2, warmupCount: 5, iterationCount: 15)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class DitXLMatMulDoubleBenchmarks
{
    [ParamsAllValues]
    public bool DeterministicMode { get; set; }

    [ParamsAllValues]
    public bool UseParallelGemm { get; set; }

    private CpuEngine _engine = null!;

    // ═══ DiT-XL FP64 shapes (mirror of float bench) ═══
    private Tensor<double> _act_1024x1152 = null!;
    private Tensor<double> _weight_1152x1152 = null!;
    private Tensor<double> _weight_1152x3456 = null!;
    private Tensor<double> _weight_1152x4608 = null!;
    private Tensor<double> _act_1024x4608 = null!;
    private Tensor<double> _weight_4608x1152 = null!;
    private Tensor<double> _square_1152 = null!;
    private Tensor<double> _square_4608 = null!;
    private Tensor<double> _q_256x72 = null!;
    private Tensor<double> _k_72x256 = null!;
    private Tensor<double> _attn_256x256 = null!;
    private Tensor<double> _v_256x72 = null!;
    private Tensor<double> _batched_1_256_1152 = null!;
    private Tensor<double> _batched_4_256_1152 = null!;

    // ═══ Cluster #6 FP64 shapes: VGG16 FC backward ═══
    // VGG16's three Dense layers: 25088→4096, 4096→4096, 4096→1000.
    // Per forward+backward, each FC contributes M·K·N FMA per step.
    // FC1 forward [1,25088]×[25088,4096] = 102.7 M FMA — most of step.
    private Tensor<double> _vgg_fc1_act = null!;        // [1, 25088]
    private Tensor<double> _vgg_fc1_w = null!;          // [25088, 4096]
    private Tensor<double> _vgg_fc2_act = null!;        // [1, 4096]
    private Tensor<double> _vgg_fc2_w = null!;          // [4096, 4096]
    private Tensor<double> _vgg_fc3_w = null!;          // [4096, 1000]

    // ═══ ResNet50 FP64 shapes: head FC + bottleneck 1×1 conv lowered to GEMM ═══
    // Final classifier 2048→1000; bottleneck 1×1 convs lower to [H·W, C_in] × [C_in, C_out].
    private Tensor<double> _resnet_head_act = null!;    // [1, 2048]
    private Tensor<double> _resnet_head_w = null!;      // [2048, 1000]
    private Tensor<double> _resnet_1x1_49x2048 = null!; // bottleneck at H·W=7×7=49
    private Tensor<double> _resnet_1x1_w_2048x512 = null!;
    private Tensor<double> _resnet_1x1_196x1024 = null!;// bottleneck at H·W=14×14=196
    private Tensor<double> _resnet_1x1_w_1024x256 = null!;

    // ═══ ACEStep FP64 shapes: MHA per-head + MLP FFN ═══
    // 8 heads × dim 64; seq_len 128 (latent from diffusion step).
    private Tensor<double> _ace_perhead_q = null!;      // [128, 64]
    private Tensor<double> _ace_perhead_k = null!;      // [64, 128]
    private Tensor<double> _ace_perhead_av_a = null!;   // [128, 128]
    private Tensor<double> _ace_perhead_av_v = null!;   // [128, 64]
    private Tensor<double> _ace_mlp_act = null!;        // [128, 512]
    private Tensor<double> _ace_mlp_up_w = null!;       // [512, 2048]
    private Tensor<double> _ace_mlp_hidden = null!;     // [128, 2048]
    private Tensor<double> _ace_mlp_down_w = null!;     // [2048, 512]

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _act_1024x1152    = Tensor<double>.CreateRandom([1024, 1152]);
        _weight_1152x1152 = Tensor<double>.CreateRandom([1152, 1152]);
        _weight_1152x3456 = Tensor<double>.CreateRandom([1152, 3456]);
        _weight_1152x4608 = Tensor<double>.CreateRandom([1152, 4608]);
        _act_1024x4608    = Tensor<double>.CreateRandom([1024, 4608]);
        _weight_4608x1152 = Tensor<double>.CreateRandom([4608, 1152]);

        _square_1152 = Tensor<double>.CreateRandom([1152, 1152]);
        _square_4608 = Tensor<double>.CreateRandom([4608, 4608]);

        _q_256x72     = Tensor<double>.CreateRandom([256, 72]);
        _k_72x256     = Tensor<double>.CreateRandom([72, 256]);
        _attn_256x256 = Tensor<double>.CreateRandom([256, 256]);
        _v_256x72     = Tensor<double>.CreateRandom([256, 72]);

        _batched_1_256_1152 = Tensor<double>.CreateRandom([1, 256, 1152]);
        _batched_4_256_1152 = Tensor<double>.CreateRandom([4, 256, 1152]);

        _vgg_fc1_act = Tensor<double>.CreateRandom([1, 25088]);
        _vgg_fc1_w   = Tensor<double>.CreateRandom([25088, 4096]);
        _vgg_fc2_act = Tensor<double>.CreateRandom([1, 4096]);
        _vgg_fc2_w   = Tensor<double>.CreateRandom([4096, 4096]);
        _vgg_fc3_w   = Tensor<double>.CreateRandom([4096, 1000]);

        _resnet_head_act    = Tensor<double>.CreateRandom([1, 2048]);
        _resnet_head_w      = Tensor<double>.CreateRandom([2048, 1000]);
        _resnet_1x1_49x2048 = Tensor<double>.CreateRandom([49, 2048]);
        _resnet_1x1_w_2048x512 = Tensor<double>.CreateRandom([2048, 512]);
        _resnet_1x1_196x1024 = Tensor<double>.CreateRandom([196, 1024]);
        _resnet_1x1_w_1024x256 = Tensor<double>.CreateRandom([1024, 256]);

        _ace_perhead_q    = Tensor<double>.CreateRandom([128, 64]);
        _ace_perhead_k    = Tensor<double>.CreateRandom([64, 128]);
        _ace_perhead_av_a = Tensor<double>.CreateRandom([128, 128]);
        _ace_perhead_av_v = Tensor<double>.CreateRandom([128, 64]);
        _ace_mlp_act      = Tensor<double>.CreateRandom([128, 512]);
        _ace_mlp_up_w     = Tensor<double>.CreateRandom([512, 2048]);
        _ace_mlp_hidden   = Tensor<double>.CreateRandom([128, 2048]);
        _ace_mlp_down_w   = Tensor<double>.CreateRandom([2048, 512]);
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

    // ───────────── DiT-XL FP64 shapes ─────────────

    [Benchmark(Description = "DiT FP64 attn out [1024,1152]×[1152,1152]")]
    public Tensor<double> DiT_AttnOutputProj_F64()
        => _engine.TensorMatMul(_act_1024x1152, _weight_1152x1152);

    [Benchmark(Description = "DiT FP64 QKV fused [1024,1152]×[1152,3456]")]
    public Tensor<double> DiT_QKVFusedProj_F64()
        => _engine.TensorMatMul(_act_1024x1152, _weight_1152x3456);

    [Benchmark(Description = "DiT FP64 MLP up [1024,1152]×[1152,4608]")]
    public Tensor<double> DiT_MlpUp_F64()
        => _engine.TensorMatMul(_act_1024x1152, _weight_1152x4608);

    [Benchmark(Description = "DiT FP64 MLP down [1024,4608]×[4608,1152]")]
    public Tensor<double> DiT_MlpDown_F64()
        => _engine.TensorMatMul(_act_1024x4608, _weight_4608x1152);

    [Benchmark(Description = "Square FP64 [1152,1152]²")]
    public Tensor<double> Square_1152_F64() => _engine.TensorMatMul(_square_1152, _square_1152);

    [Benchmark(Description = "Square FP64 [4608,4608]²")]
    public Tensor<double> Square_4608_F64() => _engine.TensorMatMul(_square_4608, _square_4608);

    [Benchmark(Description = "Attn FP64 Q·K^T per-head [256,72]×[72,256]")]
    public Tensor<double> Attn_QK_F64() => _engine.TensorMatMul(_q_256x72, _k_72x256);

    [Benchmark(Description = "Attn FP64 A·V per-head [256,256]×[256,72]")]
    public Tensor<double> Attn_AV_F64() => _engine.TensorMatMul(_attn_256x256, _v_256x72);

    [Benchmark(Description = "Batched3D FP64 B=1 [1,256,1152]×[1152,4608]")]
    public Tensor<double> Batched_B1_F64() => _engine.TensorMatMul(_batched_1_256_1152, _weight_1152x4608);

    [Benchmark(Description = "Batched3D FP64 B=4 [4,256,1152]×[1152,4608]")]
    public Tensor<double> Batched_B4_F64() => _engine.TensorMatMul(_batched_4_256_1152, _weight_1152x4608);

    // ───────────── Cluster #6 VGG16 FC shapes ─────────────

    [Benchmark(Description = "VGG FC1 fwd [1,25088]×[25088,4096]")]
    public Tensor<double> VGG_FC1_Fwd() => _engine.TensorMatMul(_vgg_fc1_act, _vgg_fc1_w);

    [Benchmark(Description = "VGG FC2 fwd [1,4096]×[4096,4096]")]
    public Tensor<double> VGG_FC2_Fwd() => _engine.TensorMatMul(_vgg_fc2_act, _vgg_fc2_w);

    [Benchmark(Description = "VGG FC3 fwd [1,4096]×[4096,1000]")]
    public Tensor<double> VGG_FC3_Fwd() => _engine.TensorMatMul(_vgg_fc2_act, _vgg_fc3_w);

    // ───────────── Cluster #6 ResNet50 shapes ─────────────

    [Benchmark(Description = "ResNet head [1,2048]×[2048,1000]")]
    public Tensor<double> ResNet_HeadFC() => _engine.TensorMatMul(_resnet_head_act, _resnet_head_w);

    [Benchmark(Description = "ResNet bottleneck 1×1 at H·W=49 [49,2048]×[2048,512]")]
    public Tensor<double> ResNet_Bottleneck_HW49() => _engine.TensorMatMul(_resnet_1x1_49x2048, _resnet_1x1_w_2048x512);

    [Benchmark(Description = "ResNet bottleneck 1×1 at H·W=196 [196,1024]×[1024,256]")]
    public Tensor<double> ResNet_Bottleneck_HW196() => _engine.TensorMatMul(_resnet_1x1_196x1024, _resnet_1x1_w_1024x256);

    // ───────────── Cluster #6 ACEStep shapes ─────────────

    [Benchmark(Description = "ACEStep MHA per-head Q·K^T [128,64]×[64,128]")]
    public Tensor<double> ACE_QK() => _engine.TensorMatMul(_ace_perhead_q, _ace_perhead_k);

    [Benchmark(Description = "ACEStep MHA per-head A·V [128,128]×[128,64]")]
    public Tensor<double> ACE_AV() => _engine.TensorMatMul(_ace_perhead_av_a, _ace_perhead_av_v);

    [Benchmark(Description = "ACEStep MLP up [128,512]×[512,2048]")]
    public Tensor<double> ACE_MlpUp() => _engine.TensorMatMul(_ace_mlp_act, _ace_mlp_up_w);

    [Benchmark(Description = "ACEStep MLP down [128,2048]×[2048,512]")]
    public Tensor<double> ACE_MlpDown() => _engine.TensorMatMul(_ace_mlp_hidden, _ace_mlp_down_w);
}
#endif
