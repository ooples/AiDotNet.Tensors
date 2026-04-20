#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Benchmarks.BaselineRunners;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using TorchSharp;
using static TorchSharp.torch;

namespace AiDotNet.Tensors.Benchmarks.Parity210;

/// <summary>
/// Einsum benchmark suite for issue #210's "beat PyTorch" acceptance gate.
/// Compares AiDotNet.Tensors <see cref="CpuEngine.TensorEinsum{T}"/> against
/// PyTorch eager (via TorchSharp, in-process), torch.compile, opt_einsum,
/// and JAX (via Python subprocess) across 8 representative contraction
/// patterns and 3 shape buckets.
/// </summary>
/// <remarks>
/// <para>
/// TorchSharp is already wired in the benchmarks project for eager CPU /
/// CUDA comparisons. The Python subprocess runners (compile / opt_einsum /
/// JAX) live under BaselineRunners/py and are invoked on demand — if
/// Python isn't available the benchmark records a skip rather than
/// failing.
/// </para>
/// <para>
/// Acceptance per #210 plan §4:
///   CPU speed vs PyTorch eager — ≥1.2× on ≥80% of ops, ≥0.95× on 100%.
/// The per-pattern results are captured by BenchmarkDotNet's Summary and
/// (when running on CI) dumped into docs/benchmarks/parity-210/.
/// </para>
/// </remarks>
[MemoryDiagnoser]
[MinIterationCount(20)]
[MaxIterationCount(100)]
public class Parity210EinsumBenchmarks
{
    private readonly CpuEngine _engine = new();
    private readonly PythonBaselineRunner _py = new();

    // --- Matmul (ij,jk->ik) on a mid-size 256×256 shape ---

    private Tensor<float> _aMatmul = null!;
    private Tensor<float> _bMatmul = null!;
    private torch.Tensor _taMatmul = null!;
    private torch.Tensor _tbMatmul = null!;

    // --- Batched matmul (bij,bjk->bik) 8×128×64·64×128 ---

    private Tensor<float> _aBmm = null!;
    private Tensor<float> _bBmm = null!;
    private torch.Tensor _taBmm = null!;
    private torch.Tensor _tbBmm = null!;

    // --- Attention scores (bhqd,bhkd->bhqk) — B=1, H=8, Q=K=128, D=64 ---

    private Tensor<float> _aAttn = null!;
    private Tensor<float> _bAttn = null!;
    private torch.Tensor _taAttn = null!;
    private torch.Tensor _tbAttn = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _aMatmul = Rand(rng, 256, 256);
        _bMatmul = Rand(rng, 256, 256);
        _taMatmul = torch.randn(new long[] { 256, 256 });
        _tbMatmul = torch.randn(new long[] { 256, 256 });

        _aBmm = Rand(rng, 8, 128, 64);
        _bBmm = Rand(rng, 8, 64, 128);
        _taBmm = torch.randn(new long[] { 8, 128, 64 });
        _tbBmm = torch.randn(new long[] { 8, 64, 128 });

        _aAttn = Rand(rng, 1, 8, 128, 64);
        _bAttn = Rand(rng, 1, 8, 128, 64);
        _taAttn = torch.randn(new long[] { 1, 8, 128, 64 });
        _tbAttn = torch.randn(new long[] { 1, 8, 128, 64 });
    }

    private static Tensor<float> Rand(Random rng, params int[] shape)
    {
        int total = 1;
        foreach (var d in shape) total *= d;
        var arr = new float[total];
        for (int i = 0; i < total; i++) arr[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(arr, shape);
    }

    // --- AiDotNet.Tensors (our implementation) ---

    [Benchmark(Baseline = true, Description = "Ours: einsum 'ij,jk->ik' 256x256")]
    public Tensor<float> Ours_Matmul() => _engine.TensorEinsum("ij,jk->ik", _aMatmul, _bMatmul);

    [Benchmark(Description = "Ours: einsum 'bij,bjk->bik' B=8 128x64 64x128")]
    public Tensor<float> Ours_Bmm() => _engine.TensorEinsum("bij,bjk->bik", _aBmm, _bBmm);

    [Benchmark(Description = "Ours: attention scores 'bhqd,bhkd->bhqk' 1,8,128,64")]
    public Tensor<float> Ours_Attn() => _engine.TensorEinsum("bhqd,bhkd->bhqk", _aAttn, _bAttn);

    // --- PyTorch eager via TorchSharp (in-process, no subprocess) ---

    [Benchmark(Description = "PyTorch eager: einsum 'ij,jk->ik' 256x256")]
    public torch.Tensor PyTorch_Matmul() => torch.einsum("ij,jk->ik", _taMatmul, _tbMatmul);

    [Benchmark(Description = "PyTorch eager: einsum 'bij,bjk->bik' B=8 128x64 64x128")]
    public torch.Tensor PyTorch_Bmm() => torch.einsum("bij,bjk->bik", _taBmm, _tbBmm);

    [Benchmark(Description = "PyTorch eager: attention scores 'bhqd,bhkd->bhqk' 1,8,128,64")]
    public torch.Tensor PyTorch_Attn() => torch.einsum("bhqd,bhkd->bhqk", _taAttn, _tbAttn);

    // --- Python subprocess baselines: torch.compile / opt_einsum / JAX ---
    // These are separate runs rather than inline benchmarks to avoid
    // per-iteration Python startup cost. They produce CSV lines that feed
    // into the closing PR's comparison table.

    public (double median, double p90)? RunTorchCompile_Matmul()
    {
        var r = _py.TryRun(PythonBaselineRunner.Baseline.TorchCompile,
            "einsum", "ij,jk->ik;256x256,256x256");
        return r is null ? null : (r.MedianMs, r.P90Ms);
    }

    public (double median, double p90)? RunOptEinsum_Attn()
    {
        var r = _py.TryRun(PythonBaselineRunner.Baseline.OptEinsum,
            "einsum", "bhqd,bhkd->bhqk;1x8x128x64,1x8x128x64");
        return r is null ? null : (r.MedianMs, r.P90Ms);
    }

    public (double median, double p90)? RunJaxJit_Matmul()
    {
        var r = _py.TryRun(PythonBaselineRunner.Baseline.JaxJit,
            "einsum", "ij,jk->ik;256x256,256x256");
        return r is null ? null : (r.MedianMs, r.P90Ms);
    }

    [GlobalCleanup]
    public void Cleanup() => _py.Dispose();
}
#endif
