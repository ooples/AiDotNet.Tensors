#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// A/B benchmark: native-BLAS (default, non-deterministic) matmul vs our blocked
/// C# GEMM for the exact shapes a DiT-XL forward pass exercises. This is the
/// workload that led the downstream HRE/DiT consumer to report CPU shards being
/// cancelled at the 45-minute CI budget.
///
/// Note on "default path": pre-branch this PR, the default path went through
/// MKL.NET's managed SGEMM bindings. After `feat/finish-mkl-replacement` /
/// the MKL.NET package removal, the default path routes through whatever native
/// BLAS the user has installed (OpenBLAS, user-supplied cblas, or an externally
/// installed MKL native DLL) via BlasProvider's P/Invoke loader — no managed
/// MKL.NET binding. Benchmark semantics are unchanged: Det=false exercises the
/// native-BLAS path when available, Det=true forces our blocked SimdGemm path.
///
/// DiT-XL config (from the DiT paper + HuggingFace diffusers default):
///   hidden_size = 1152, num_heads = 16 (head_dim = 72)
///   depth = 28 transformer blocks, mlp_ratio = 4 → mlp_hidden = 4608
///   patch_size = 2, typical latent 32×32 → 16×16 = 256 tokens per sample
///
/// For batch B=4, seq S=256, we flatten [B*S, H] = [1024, 1152] for the
/// per-block projections. This is the "square-ish" regime where our blocked
/// path has been trailing MKL by 1.35–1.64× (per Issue #131 iter 9, 2026-04-11).
///
/// The goal of this benchmark is to quantify the current gap at *DiT-XL*
/// shapes specifically, not the HRE-tiny shapes the existing
/// <see cref="DeterministicMatMulBenchmarks"/> tracks. Beat MKL here and we
/// beat MKL where it actually matters for downstream vision/diffusion work.
///
/// Run:
///   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks \
///     -- --dit-xl-matmul
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 2, warmupCount: 5, iterationCount: 15)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class DitXLMatMulBenchmarks
{
    /// <summary>off = native-BLAS GEMM (if a cblas-compatible library is on the path),
    /// on = blocked C# (our SimdGemm path). Post-MKL.NET-removal: off no longer implies
    /// MKL.NET — the native BLAS could be OpenBLAS, user-supplied cblas, or an externally
    /// installed MKL DLL; whichever the BlasProvider loader finds. BDN expands both values.</summary>
    [ParamsAllValues]
    public bool DeterministicMode { get; set; }

    /// <summary>off = single-threaded SgemmTiled, on = parallel dispatch.</summary>
    [ParamsAllValues]
    public bool UseParallelGemm { get; set; }

    private CpuEngine _engine = null!;

    // ═══ Per-block projection inputs (flattened 2D: [B*S, H] where B=4, S=256) ═══
    // DiT-XL h=1152, mlp_hidden=4608. All shapes here are what a single
    // transformer block of a DiT-XL evaluates, at the real batch a DiT sampling
    // loop would use.
    private Tensor<float> _act_1024x1152 = null!;        // [B*S=1024, H=1152] — block input
    private Tensor<float> _weight_1152x1152 = null!;     // attention-output projection
    private Tensor<float> _weight_1152x3456 = null!;     // QKV fused projection (3×H)
    private Tensor<float> _weight_1152x4608 = null!;     // MLP up projection (4×H)
    private Tensor<float> _act_1024x4608 = null!;        // MLP intermediate activation
    private Tensor<float> _weight_4608x1152 = null!;     // MLP down projection (4×H → H)

    // ═══ Square-at-hidden-size ═══
    // The architectural "square" used in weight-only GEMMs and weight-gradient
    // accumulation during eval — 1152² and 4608² stress the same blocked
    // kernels we're trying to make MKL-competitive.
    private Tensor<float> _square_1152 = null!;          // [1152, 1152]
    private Tensor<float> _square_4608 = null!;          // [4608, 4608]

    // ═══ Per-head attention matmul (small K, many heads) ═══
    // Q·K^T  shape per head: [S=256, head_dim=72] × [72, 256]
    // A·V    shape per head: [256, 256] × [256, 72]
    // These get dispatched per-head-slice — a place MKL's dispatch overhead
    // hurts us AND a place batched-GEMM can help us.
    private Tensor<float> _q_256x72 = null!;
    private Tensor<float> _k_72x256 = null!;
    private Tensor<float> _attn_256x256 = null!;
    private Tensor<float> _v_256x72 = null!;

    // ═══ Full batched 3D: [B, S, H] × [H, O] transformer pattern ═══
    // This is the shape that hits TensorMatMulBatched's per-slice BLAS dispatch.
    private Tensor<float> _batched_1_256_1152 = null!;   // [B=1, S=256, H=1152] — typical inference
    private Tensor<float> _batched_4_256_1152 = null!;   // [B=4, S=256, H=1152] — typical DiT sample

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _act_1024x1152        = Tensor<float>.CreateRandom([1024, 1152]);
        _weight_1152x1152     = Tensor<float>.CreateRandom([1152, 1152]);
        _weight_1152x3456     = Tensor<float>.CreateRandom([1152, 3456]);
        _weight_1152x4608     = Tensor<float>.CreateRandom([1152, 4608]);
        _act_1024x4608        = Tensor<float>.CreateRandom([1024, 4608]);
        _weight_4608x1152     = Tensor<float>.CreateRandom([4608, 1152]);

        _square_1152 = Tensor<float>.CreateRandom([1152, 1152]);
        _square_4608 = Tensor<float>.CreateRandom([4608, 4608]);

        _q_256x72     = Tensor<float>.CreateRandom([256, 72]);
        _k_72x256     = Tensor<float>.CreateRandom([72, 256]);
        _attn_256x256 = Tensor<float>.CreateRandom([256, 256]);
        _v_256x72     = Tensor<float>.CreateRandom([256, 72]);

        _batched_1_256_1152 = Tensor<float>.CreateRandom([1, 256, 1152]);
        _batched_4_256_1152 = Tensor<float>.CreateRandom([4, 256, 1152]);
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

    // ───────────── Per-block 2D projections ─────────────
    // All 4 projections inside one DiT-XL transformer block, flattened to 2D.
    // In a real forward pass these execute 28× (once per block).

    [Benchmark(Description = "DiT block: attn out proj [1024,1152]×[1152,1152]")]
    public Tensor<float> DiT_AttnOutputProj()
        => _engine.TensorMatMul(_act_1024x1152, _weight_1152x1152);

    [Benchmark(Description = "DiT block: QKV fused proj [1024,1152]×[1152,3456]")]
    public Tensor<float> DiT_QKVFusedProj()
        => _engine.TensorMatMul(_act_1024x1152, _weight_1152x3456);

    [Benchmark(Description = "DiT block: MLP up [1024,1152]×[1152,4608]")]
    public Tensor<float> DiT_MlpUp()
        => _engine.TensorMatMul(_act_1024x1152, _weight_1152x4608);

    [Benchmark(Description = "DiT block: MLP down [1024,4608]×[4608,1152]")]
    public Tensor<float> DiT_MlpDown()
        => _engine.TensorMatMul(_act_1024x4608, _weight_4608x1152);

    // ───────────── Square-at-hidden-size ─────────────

    [Benchmark(Description = "Square [1152,1152]²  (DiT hidden)")]
    public Tensor<float> Square_1152()
        => _engine.TensorMatMul(_square_1152, _square_1152);

    [Benchmark(Description = "Square [4608,4608]²  (DiT MLP hidden)")]
    public Tensor<float> Square_4608()
        => _engine.TensorMatMul(_square_4608, _square_4608);

    // ───────────── Per-head attention (small K, tall/wide) ─────────────

    [Benchmark(Description = "Attn Q·K^T per-head [256,72]×[72,256]")]
    public Tensor<float> Attn_QK_PerHead()
        => _engine.TensorMatMul(_q_256x72, _k_72x256);

    [Benchmark(Description = "Attn A·V per-head [256,256]×[256,72]")]
    public Tensor<float> Attn_AV_PerHead()
        => _engine.TensorMatMul(_attn_256x256, _v_256x72);

    // ───────────── 3D batched matmul ─────────────
    // Exercises TensorMatMulBatched — the per-slice BLAS dispatch that was
    // flagged as a candidate for batched-GEMM consolidation.

    [Benchmark(Description = "Batched 3D [1,256,1152]×[1152,4608]")]
    public Tensor<float> Batched3D_B1_MlpUp()
        => _engine.TensorMatMul(_batched_1_256_1152, _weight_1152x4608);

    [Benchmark(Description = "Batched 3D [4,256,1152]×[1152,4608]")]
    public Tensor<float> Batched3D_B4_MlpUp()
        => _engine.TensorMatMul(_batched_4_256_1152, _weight_1152x4608);
}
#endif
