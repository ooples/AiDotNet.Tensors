#if NET8_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
// Issue #245 — transformer-FFN matmul benchmark coverage.
//
// Why this suite exists:
//   The existing DitXLMatMulBenchmarks cover DiT-XL-class square shapes
//   (1152² and similar). Paper-scale transformer models used by the
//   AiDotNet sibling repo (ChronosBolt, TimeMoE, MOMENT, TimesFM,
//   VisionTS, etc.) produce fundamentally different shapes: tall-skinny
//   [M≤64, K in 256-4096] × [K, N in 512-4096]. The previous suite did
//   not catch these, which is exactly where the 2D-parallel-grid fix
//   (Issue #244) and the AVX2 Dgemm kernel (Issue #243) matter most.
//
// What this suite measures:
//   - Sgemm + Dgemm at small-M × medium-large N,K (single-batch FFN).
//   - Attention QKV / output projections at hidden-dim² per block.
//   - Rank-3 × rank-2 batched pattern ([B, T, K] × [K, N]) — hits
//     TensorMatMulBatched which has different tiling from TensorMatMul2D.
//
// Run (~8-15 min at iter=15, launch=2):
//   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks \
//     -- --transformer-ffn

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 2, warmupCount: 5, iterationCount: 15)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class TransformerFFNBenchmarks
{
    private CpuEngine _engine = null!;

    // ── Small-M FFN (float): (batch*seq, hidden) × (hidden, ffn) ─────────
    // ChronosBolt T5-small: hidden=512, ffn=2048, typical B*S=32-64.
    private Tensor<float> _fSmall_32x512 = null!;
    private Tensor<float> _fSmall_64x512 = null!;
    private Tensor<float> _fSmall_8x1024 = null!;
    private Tensor<float> _fSmall_16x256 = null!;
    private Tensor<float> _fSmall_32x768 = null!;

    private Tensor<float> _fW_512x2048 = null!;
    private Tensor<float> _fW_2048x512 = null!;
    private Tensor<float> _fW_1024x4096 = null!;
    private Tensor<float> _fW_4096x1024 = null!;
    private Tensor<float> _fW_256x1024 = null!;
    private Tensor<float> _fW_768x3072 = null!;

    // Attention QKV/O (square at hidden dim) for float.
    private Tensor<float> _fW_512x512 = null!;
    private Tensor<float> _fW_1024x1024 = null!;
    private Tensor<float> _fW_768x768 = null!;

    // ── Same shapes in double ────────────────────────────────────────────
    // The AVX2 Dgemm kernel (Issue #243) is the one that went from
    // 2.8 GFLOPS → ~12 GFLOPS on [64, 2048]×[2048, 512] — doubles are the
    // headline regression surface.
    private Tensor<double> _dSmall_32x512 = null!;
    private Tensor<double> _dSmall_64x512 = null!;
    private Tensor<double> _dSmall_8x1024 = null!;
    private Tensor<double> _dSmall_16x256 = null!;

    private Tensor<double> _dW_512x2048 = null!;
    private Tensor<double> _dW_2048x512 = null!;
    private Tensor<double> _dW_1024x4096 = null!;
    private Tensor<double> _dW_256x1024 = null!;

    private Tensor<double> _dW_512x512 = null!;
    private Tensor<double> _dW_1024x1024 = null!;

    // ── Rank-3 × rank-2 batched: [B, T, K] × [K, N] ──────────────────────
    // Exercises TensorMatMulBatched — the per-slice dispatch that Issue #244
    // parallelism fix propagates through.
    private Tensor<float> _fBatched_1_64_512 = null!;     // B=1, T=64, K=512
    private Tensor<float> _fBatched_2_32_1024 = null!;    // B=2, T=32, K=1024
    private Tensor<float> _fBatched_4_16_768 = null!;     // B=4, T=16, K=768

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _fSmall_32x512  = Tensor<float>.CreateRandom(32, 512);
        _fSmall_64x512  = Tensor<float>.CreateRandom(64, 512);
        _fSmall_8x1024  = Tensor<float>.CreateRandom(8, 1024);
        _fSmall_16x256  = Tensor<float>.CreateRandom(16, 256);
        _fSmall_32x768  = Tensor<float>.CreateRandom(32, 768);

        _fW_512x2048   = Tensor<float>.CreateRandom(512, 2048);
        _fW_2048x512   = Tensor<float>.CreateRandom(2048, 512);
        _fW_1024x4096  = Tensor<float>.CreateRandom(1024, 4096);
        _fW_4096x1024  = Tensor<float>.CreateRandom(4096, 1024);
        _fW_256x1024   = Tensor<float>.CreateRandom(256, 1024);
        _fW_768x3072   = Tensor<float>.CreateRandom(768, 3072);

        _fW_512x512    = Tensor<float>.CreateRandom(512, 512);
        _fW_1024x1024  = Tensor<float>.CreateRandom(1024, 1024);
        _fW_768x768    = Tensor<float>.CreateRandom(768, 768);

        _dSmall_32x512  = Tensor<double>.CreateRandom(32, 512);
        _dSmall_64x512  = Tensor<double>.CreateRandom(64, 512);
        _dSmall_8x1024  = Tensor<double>.CreateRandom(8, 1024);
        _dSmall_16x256  = Tensor<double>.CreateRandom(16, 256);

        _dW_512x2048   = Tensor<double>.CreateRandom(512, 2048);
        _dW_2048x512   = Tensor<double>.CreateRandom(2048, 512);
        _dW_1024x4096  = Tensor<double>.CreateRandom(1024, 4096);
        _dW_256x1024   = Tensor<double>.CreateRandom(256, 1024);

        _dW_512x512    = Tensor<double>.CreateRandom(512, 512);
        _dW_1024x1024 = Tensor<double>.CreateRandom(1024, 1024);

        _fBatched_1_64_512   = Tensor<float>.CreateRandom(1, 64, 512);
        _fBatched_2_32_1024  = Tensor<float>.CreateRandom(2, 32, 1024);
        _fBatched_4_16_768   = Tensor<float>.CreateRandom(4, 16, 768);
    }

    // ── Sgemm small-M FFN ────────────────────────────────────────────────

    [Benchmark(Description = "Sgemm ChronosBolt FFN up   [32, 512] x [512, 2048]")]
    public Tensor<float> Sgemm_ChronosBolt_FFNUp_32()
        => _engine.TensorMatMul(_fSmall_32x512, _fW_512x2048);

    [Benchmark(Description = "Sgemm ChronosBolt FFN up   [64, 512] x [512, 2048]")]
    public Tensor<float> Sgemm_ChronosBolt_FFNUp_64()
        => _engine.TensorMatMul(_fSmall_64x512, _fW_512x2048);

    [Benchmark(Description = "Sgemm MOMENT      FFN up   [8, 1024] x [1024, 4096]")]
    public Tensor<float> Sgemm_MOMENT_FFNUp()
        => _engine.TensorMatMul(_fSmall_8x1024, _fW_1024x4096);

    [Benchmark(Description = "Sgemm TimesFM     FFN up   [16, 256] x [256, 1024]")]
    public Tensor<float> Sgemm_TimesFM_FFNUp()
        => _engine.TensorMatMul(_fSmall_16x256, _fW_256x1024);

    [Benchmark(Description = "Sgemm VisionTS    FFN up   [32, 768] x [768, 3072]")]
    public Tensor<float> Sgemm_VisionTS_FFNUp()
        => _engine.TensorMatMul(_fSmall_32x768, _fW_768x3072);

    // ── Sgemm attention projections (square at hidden) ──────────────────

    [Benchmark(Description = "Sgemm Attn QKV    [32, 512]  x [512, 512]")]
    public Tensor<float> Sgemm_Attn_512()
        => _engine.TensorMatMul(_fSmall_32x512, _fW_512x512);

    [Benchmark(Description = "Sgemm Attn QKV    [8, 1024]  x [1024, 1024]")]
    public Tensor<float> Sgemm_Attn_1024()
        => _engine.TensorMatMul(_fSmall_8x1024, _fW_1024x1024);

    [Benchmark(Description = "Sgemm Attn QKV    [32, 768]  x [768, 768]")]
    public Tensor<float> Sgemm_Attn_768()
        => _engine.TensorMatMul(_fSmall_32x768, _fW_768x768);

    // ── Dgemm small-M FFN (headline: Issue #243 target) ─────────────────

    [Benchmark(Description = "Dgemm ChronosBolt FFN up   [32, 512] x [512, 2048]")]
    public Tensor<double> Dgemm_ChronosBolt_FFNUp_32()
        => _engine.TensorMatMul(_dSmall_32x512, _dW_512x2048);

    [Benchmark(Description = "Dgemm ChronosBolt FFN up   [64, 512] x [512, 2048]")]
    public Tensor<double> Dgemm_ChronosBolt_FFNUp_64()
        => _engine.TensorMatMul(_dSmall_64x512, _dW_512x2048);

    [Benchmark(Description = "Dgemm ChronosBolt FFN down [64, 2048] x [2048, 512]")]
    public Tensor<double> Dgemm_ChronosBolt_FFNDown_64()
    {
        var x = Tensor<double>.CreateRandom(64, 2048);
        return _engine.TensorMatMul(x, _dW_2048x512);
    }

    [Benchmark(Description = "Dgemm MOMENT      FFN up   [8, 1024] x [1024, 4096]")]
    public Tensor<double> Dgemm_MOMENT_FFNUp()
        => _engine.TensorMatMul(_dSmall_8x1024, _dW_1024x4096);

    [Benchmark(Description = "Dgemm TimesFM     FFN up   [16, 256] x [256, 1024]")]
    public Tensor<double> Dgemm_TimesFM_FFNUp()
        => _engine.TensorMatMul(_dSmall_16x256, _dW_256x1024);

    // ── Dgemm attention (square at hidden, small-M outer) ──────────────

    [Benchmark(Description = "Dgemm Attn QKV    [32, 512]  x [512, 512]")]
    public Tensor<double> Dgemm_Attn_512()
        => _engine.TensorMatMul(_dSmall_32x512, _dW_512x512);

    [Benchmark(Description = "Dgemm Attn QKV    [8, 1024]  x [1024, 1024]")]
    public Tensor<double> Dgemm_Attn_1024()
        => _engine.TensorMatMul(_dSmall_8x1024, _dW_1024x1024);

    // ── Rank-3 × rank-2 batched (transformer layer shape) ──────────────
    // These go through TensorMatMulBatched — the per-slice dispatch path
    // that the Issue #244 fix propagates tiling through.

    [Benchmark(Description = "Sgemm Batched     [1, 64, 512]  x [512, 2048]")]
    public Tensor<float> Sgemm_Batched_B1_FFN()
        => _engine.TensorMatMul(_fBatched_1_64_512, _fW_512x2048);

    [Benchmark(Description = "Sgemm Batched     [2, 32, 1024] x [1024, 4096]")]
    public Tensor<float> Sgemm_Batched_B2_FFN()
        => _engine.TensorMatMul(_fBatched_2_32_1024, _fW_1024x4096);

    [Benchmark(Description = "Sgemm Batched     [4, 16, 768]  x [768, 768]")]
    public Tensor<float> Sgemm_Batched_B4_Attn()
        => _engine.TensorMatMul(_fBatched_4_16_768, _fW_768x768);
}
#endif
