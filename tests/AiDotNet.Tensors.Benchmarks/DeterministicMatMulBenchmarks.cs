#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// A/B benchmark: MKL.NET default matmul vs the deterministic blocked C# fallback
/// exposed by <see cref="AiDotNetEngine.SetDeterministicMode"/>.
///
/// Purpose: measure how competitive the bit-exact blocked path is vs MKL.NET
/// across matmul shapes that matter for real model training — starting with the
/// actual HRE (Harmonic Resonance Engine) hot shapes from Issue #131, and
/// expanding out to common small/medium LLM shapes where the crossover likely
/// sits. If the blocked path is within a reasonable margin at the shapes that
/// matter, we can flip the deterministic default or remove MKL.NET entirely
/// per the project's strategic direction.
///
/// A/B is driven by two bool params:
///  - <see cref="DeterministicMode"/>: off = MKL.NET GEMM, on = blocked C# fallback
///  - <see cref="UseParallelGemm"/>: off = single-threaded SgemmTiled, on = parallel-M
///
/// BDN runs each benchmark under all 4 combinations, letting us isolate the
/// parallelism win from the deterministic-dispatch overhead independently.
///
/// Run with:
///   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks \
///     -- --vs-deterministic-matmul
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 2, warmupCount: 5, iterationCount: 15)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class DeterministicMatMulBenchmarks
{
    /// <summary>
    /// BDN expands this to {false, true}. Each benchmark method runs twice —
    /// once per mode — so the same shape is measured under both dispatch paths.
    /// </summary>
    [ParamsAllValues]
    public bool DeterministicMode { get; set; }

    /// <summary>
    /// BDN expands this to {false, true}. Controls <see cref="SimdGemm.UseParallelGemm"/>,
    /// the toggle that routes SgemmTiled through the parallel-M variant. Only meaningful
    /// when the blocked C# path is actually used — i.e. when DeterministicMode is true OR
    /// when the shape falls below the BLAS work threshold. For default-mode MKL runs this
    /// has no effect (MKL manages its own parallelism).
    /// </summary>
    [ParamsAllValues]
    public bool UseParallelGemm { get; set; }

    private CpuEngine _engine = null!;

    // ═══ HRE baseline (Issue #131 TrainingConfig defaults) ═══
    // batch=4, seq=16, embed=32, vocab=256 → batch*seq = 64
    // Attention-like projection: [batch*seq, embed] × [embed, embed]
    private Tensor<float> _hre_base_attn_a = null!; // [64, 32]
    private Tensor<float> _hre_base_attn_b = null!; // [32, 32]
    // Output head: [batch*seq, embed] × [embed, vocab]
    private Tensor<float> _hre_base_out_a = null!;  // [64, 32]
    private Tensor<float> _hre_base_out_b = null!;  // [32, 256]

    // ═══ HRE scaled-up (Issue #131 ScaledUp config) ═══
    // batch=4, seq=16, embed=64, vocab=256 → batch*seq = 64
    private Tensor<float> _hre_scaled_attn_a = null!; // [64, 64]
    private Tensor<float> _hre_scaled_attn_b = null!; // [64, 64]
    private Tensor<float> _hre_scaled_out_a = null!;  // [64, 64]
    private Tensor<float> _hre_scaled_out_b = null!;  // [64, 256]

    // ═══ Small LLM band: where MKL overhead may still hurt ═══
    private Tensor<float> _small_128 = null!;  // [128, 128]
    private Tensor<float> _small_256 = null!;  // [256, 256]

    // ═══ Mid LLM band: MKL's traditional strong zone ═══
    private Tensor<float> _mid_512 = null!;    // [512, 512]
    private Tensor<float> _mid_1024 = null!;   // [1024, 1024]

    // ═══ Large-K output head: transformer LM head over real vocab ═══
    // batch*seq = 64, embed = 128, vocab ≈ 50257 (GPT-2)
    private Tensor<float> _lm_head_a = null!;  // [64, 128]
    private Tensor<float> _lm_head_b = null!;  // [128, 50257]

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _hre_base_attn_a = Tensor<float>.CreateRandom([64, 32]);
        _hre_base_attn_b = Tensor<float>.CreateRandom([32, 32]);
        _hre_base_out_a  = Tensor<float>.CreateRandom([64, 32]);
        _hre_base_out_b  = Tensor<float>.CreateRandom([32, 256]);

        _hre_scaled_attn_a = Tensor<float>.CreateRandom([64, 64]);
        _hre_scaled_attn_b = Tensor<float>.CreateRandom([64, 64]);
        _hre_scaled_out_a  = Tensor<float>.CreateRandom([64, 64]);
        _hre_scaled_out_b  = Tensor<float>.CreateRandom([64, 256]);

        _small_128 = Tensor<float>.CreateRandom([128, 128]);
        _small_256 = Tensor<float>.CreateRandom([256, 256]);

        _mid_512  = Tensor<float>.CreateRandom([512, 512]);
        _mid_1024 = Tensor<float>.CreateRandom([1024, 1024]);

        _lm_head_a = Tensor<float>.CreateRandom([64, 128]);
        _lm_head_b = Tensor<float>.CreateRandom([128, 50257]);
    }

    [IterationSetup]
    public void IterationSetup()
    {
        // Apply mode per iteration so BDN's parameter expansion routes each
        // benchmark through the correct dispatch path.
        AiDotNetEngine.SetDeterministicMode(DeterministicMode);
        SimdGemm.UseParallelGemm = UseParallelGemm;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // Leave the engine in default mode after the benchmark run.
        AiDotNetEngine.SetDeterministicMode(false);
        SimdGemm.UseParallelGemm = true;
    }

    // ───────────── HRE baseline (Issue #131 defaults) ─────────────

    [Benchmark(Description = "HRE-base attn [64,32]×[32,32]")]
    public Tensor<float> HRE_Baseline_Attention()
        => _engine.TensorMatMul(_hre_base_attn_a, _hre_base_attn_b);

    [Benchmark(Description = "HRE-base out-head [64,32]×[32,256]")]
    public Tensor<float> HRE_Baseline_OutputHead()
        => _engine.TensorMatMul(_hre_base_out_a, _hre_base_out_b);

    // ───────────── HRE scaled-up ─────────────

    [Benchmark(Description = "HRE-scaled attn [64,64]×[64,64]")]
    public Tensor<float> HRE_Scaled_Attention()
        => _engine.TensorMatMul(_hre_scaled_attn_a, _hre_scaled_attn_b);

    [Benchmark(Description = "HRE-scaled out-head [64,64]×[64,256]")]
    public Tensor<float> HRE_Scaled_OutputHead()
        => _engine.TensorMatMul(_hre_scaled_out_a, _hre_scaled_out_b);

    // ───────────── Small LLM band ─────────────

    [Benchmark(Description = "Square [128,128]×[128,128]")]
    public Tensor<float> Square_128()
        => _engine.TensorMatMul(_small_128, _small_128);

    [Benchmark(Description = "Square [256,256]×[256,256]")]
    public Tensor<float> Square_256()
        => _engine.TensorMatMul(_small_256, _small_256);

    // ───────────── Mid LLM band ─────────────

    [Benchmark(Description = "Square [512,512]×[512,512]")]
    public Tensor<float> Square_512()
        => _engine.TensorMatMul(_mid_512, _mid_512);

    [Benchmark(Description = "Square [1024,1024]×[1024,1024]")]
    public Tensor<float> Square_1024()
        => _engine.TensorMatMul(_mid_1024, _mid_1024);

    // ───────────── Large-K LM head (GPT-2 vocab) ─────────────

    [Benchmark(Description = "LM-head [64,128]×[128,50257]")]
    public Tensor<float> LM_Head_GPT2Vocab()
        => _engine.TensorMatMul(_lm_head_a, _lm_head_b);
}
#endif
