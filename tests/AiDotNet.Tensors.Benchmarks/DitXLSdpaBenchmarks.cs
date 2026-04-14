#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmark for <see cref="IEngine.ScaledDotProductAttention{T}"/> at the exact
/// 4D shape DiT-XL exercises. Exists to quantify the Issue #162 SDPA wall-clock
/// problem and verify the BLAS-backed fast path lands the expected speedup.
///
/// DiT-XL attention config:
///   batch       = 4   (typical DiT sampling batch)
///   heads       = 16  (DiT-XL paper default)
///   seq         = 256 (latent 32×32, patch_size=2 → 16×16 tokens)
///   head_dim    = 72  (hidden_size=1152 / 16 heads)
///
/// 28 such calls per forward pass. Previous scalar virtual-dispatch path spent
/// ~93 ms per call × 28 ≈ 2.6 seconds in SDPA alone per DiT-XL forward; BLAS
/// fast path should be ~25 ms per call (measured on this shape), collapsing
/// the SDPA total to ~0.7 seconds per forward.
///
/// Run:
///   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks \
///     -- --dit-xl-sdpa
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class DitXLSdpaBenchmarks
{
    private CpuEngine _engine = null!;

    // DiT-XL per-block attention shape: [B=4, H=16, S=256, d=72]
    // Float (fast path) and double (falls through to scalar legacy path) so we
    // can compare the BLAS-backed fast path to the scalar virtual-dispatch
    // baseline on the same hardware at the same shape.
    private Tensor<float> _q_f = null!;
    private Tensor<float> _k_f = null!;
    private Tensor<float> _v_f = null!;
    private Tensor<double> _q_d = null!;
    private Tensor<double> _k_d = null!;
    private Tensor<double> _v_d = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _q_f = Tensor<float>.CreateRandom([4, 16, 256, 72]);
        _k_f = Tensor<float>.CreateRandom([4, 16, 256, 72]);
        _v_f = Tensor<float>.CreateRandom([4, 16, 256, 72]);

        _q_d = Tensor<double>.CreateRandom([4, 16, 256, 72]);
        _k_d = Tensor<double>.CreateRandom([4, 16, 256, 72]);
        _v_d = Tensor<double>.CreateRandom([4, 16, 256, 72]);
    }

    [Benchmark(Description = "SDPA DiT-XL float — BLAS fast path")]
    public Tensor<float> SDPA_Float_FastPath()
    {
        return _engine.ScaledDotProductAttention(_q_f, _k_f, _v_f, mask: null, scale: null, out _);
    }

    [Benchmark(Description = "SDPA DiT-XL double — scalar legacy path (baseline)")]
    public Tensor<double> SDPA_Double_ScalarPath()
    {
        return _engine.ScaledDotProductAttention(_q_d, _k_d, _v_d, mask: null, scale: null, out _);
    }
}
#endif
