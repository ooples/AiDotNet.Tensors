// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Profiling;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// BenchmarkDotNet-driven measurement for #220 acceptance criterion #5:
/// "enabled profiler adds ≤1% to a standard training step."
///
/// <para>BDN handles JIT warmup, GC settling, and per-iteration outlier
/// rejection — none of which a plain xunit microbenchmark can do. We
/// measure two scenarios:</para>
///
/// <list type="bullet">
///   <item><b>Baseline</b>: tight loop of TensorMatMul calls without an
///     active profiler. Every <c>OpScope</c> short-circuits to the
///     singleton no-op disposable (a single volatile read).</item>
///   <item><b>Profiled</b>: same loop with a session active. Each
///     OpScope emits a Complete event.</item>
/// </list>
///
/// <para>Run via <c>dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks
/// -- --filter *ProfilerOverheadBenchmarks*</c>. The output table makes it
/// trivial to enforce the ≤1% target on a real workload — small ops
/// (64×64) show the worst-case relative overhead, while paper-scale ops
/// (1024×1024 transformer matmul) show production-typical overhead.</para>
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 5, iterationCount: 10)]
[MemoryDiagnoser]
public class ProfilerOverheadBenchmarks
{
    private readonly CpuEngine _engine = new();
    private Tensor<float> _a64 = null!, _b64 = null!;
    private Tensor<float> _a512 = null!, _b512 = null!;
    private Tensor<float> _a1024 = null!, _b1024 = null!;

    [GlobalSetup]
    public void Setup()
    {
        _a64 = Tensor<float>.CreateRandom(64, 64);
        _b64 = Tensor<float>.CreateRandom(64, 64);
        _a512 = Tensor<float>.CreateRandom(512, 512);
        _b512 = Tensor<float>.CreateRandom(512, 512);
        _a1024 = Tensor<float>.CreateRandom(1024, 1024);
        _b1024 = Tensor<float>.CreateRandom(1024, 1024);
    }

    // ─── 64×64 matmul (worst-case relative overhead — small op) ───────────

    [Benchmark(Baseline = true), BenchmarkCategory("MatMul64")]
    public Tensor<float> MatMul_64_NoProfiler() => _engine.TensorMatMul(_a64, _b64);

    [Benchmark, BenchmarkCategory("MatMul64")]
    public Tensor<float> MatMul_64_WithProfiler()
    {
        using var prof = Profiler.Profile();
        return _engine.TensorMatMul(_a64, _b64);
    }

    // ─── 512×512 matmul (representative inner training step) ──────────────

    [Benchmark(Baseline = true), BenchmarkCategory("MatMul512")]
    public Tensor<float> MatMul_512_NoProfiler() => _engine.TensorMatMul(_a512, _b512);

    [Benchmark, BenchmarkCategory("MatMul512")]
    public Tensor<float> MatMul_512_WithProfiler()
    {
        using var prof = Profiler.Profile();
        return _engine.TensorMatMul(_a512, _b512);
    }

    // ─── 1024×1024 matmul (paper-scale; here the ≤1% target lives) ───────

    [Benchmark(Baseline = true), BenchmarkCategory("MatMul1024")]
    public Tensor<float> MatMul_1024_NoProfiler() => _engine.TensorMatMul(_a1024, _b1024);

    [Benchmark, BenchmarkCategory("MatMul1024")]
    public Tensor<float> MatMul_1024_WithProfiler()
    {
        using var prof = Profiler.Profile();
        return _engine.TensorMatMul(_a1024, _b1024);
    }

    // ─── Many tiny ops in a row (stresses the per-op hot path) ────────────

    [Benchmark(Baseline = true), BenchmarkCategory("ManyOps")]
    public Tensor<float> ManyOps_NoProfiler()
    {
        Tensor<float>? r = null;
        for (int i = 0; i < 100; i++) r = _engine.TensorMatMul(_a64, _b64);
        return r!;
    }

    [Benchmark, BenchmarkCategory("ManyOps")]
    public Tensor<float> ManyOps_WithProfiler()
    {
        using var prof = Profiler.Profile();
        Tensor<float>? r = null;
        for (int i = 0; i < 100; i++) r = _engine.TensorMatMul(_a64, _b64);
        return r!;
    }
}
