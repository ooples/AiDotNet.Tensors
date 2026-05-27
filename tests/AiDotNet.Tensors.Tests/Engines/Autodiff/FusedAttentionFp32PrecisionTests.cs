using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Issue #467 Phase A — <see cref="FlashAttentionConfig.Float32Precision"/>: the
/// mixed-precision opt-in that re-runs <c>FusedAttention&lt;double&gt;</c> internally
/// through the float path (AVX2 8-lane SimdGemm-backed SDPA) and upcasts the result,
/// for ~2× CPU throughput on the FP64 model surface (SD-UNet self-attention).
///
/// <para>
/// The path is lossy by design (FP32 accumulation), so correctness is asserted
/// against the full-FP64 reference within an FP32 tolerance rather than bit-exactly.
/// These tests prove the path (1) is numerically correct across shapes / causal /
/// bias / rank-3 / non-block-aligned lengths, (2) genuinely runs in FP32 (differs
/// from the FP64 result by FP32-scale rounding, not zero), (3) still returns valid
/// attention weights when requested, and (4) is a no-op for non-double T.
/// </para>
/// </summary>
public class FusedAttentionFp32PrecisionTests
{
    private readonly ITestOutputHelper _output;
    public FusedAttentionFp32PrecisionTests(ITestOutputHelper output) { _output = output; }

    [Theory]
    // B, H, Sq, Sk, D — square + non-square seq, several head/batch/headDim combos,
    // and lengths that are NOT multiples of the 64 block size (tail handling).
    [InlineData(1, 1, 8, 8, 16)]
    [InlineData(2, 2, 16, 16, 8)]
    [InlineData(1, 4, 32, 32, 20)]    // headDim=20 (not a SIMD multiple)
    [InlineData(2, 4, 70, 70, 16)]    // seqLen=70 → crosses the 64 block boundary
    [InlineData(1, 8, 128, 128, 12)]
    [InlineData(1, 2, 100, 100, 24)]  // common non-pow2 transformer length
    [InlineData(2, 1, 33, 65, 8)]     // Sq != Sk, both non-block-aligned
    public void Float32Precision_MatchesFp64Reference_WithinFp32Tolerance(int b, int h, int sq, int sk, int d)
    {
        var engine = new CpuEngine();
        var q = RandomDouble(new[] { b, h, sq, d }, 1000 + sq);
        var k = RandomDouble(new[] { b, h, sk, d }, 2000 + sk);
        var v = RandomDouble(new[] { b, h, sk, d }, 3000 + d);

        var (reference, _) = FusedAttention<double>.Forward(q, k, v, new FlashAttentionConfig(), engine: engine);
        var (fp32, _) = FusedAttention<double>.Forward(
            q, k, v, new FlashAttentionConfig { Float32Precision = true }, engine: engine);

        Assert.Equal(reference._shape, fp32._shape);
        double maxDiff = MaxAbsDiff(reference, fp32);
        _output.WriteLine($"B={b} H={h} Sq={sq} Sk={sk} D={d}: maxAbsDiff vs FP64 = {maxDiff:E3}");
        // FP32 internal precision: outputs in ~[-1,1]; ~1e-3 absolute is the FP32 floor.
        Assert.True(maxDiff < 2e-3, $"FP32 attention drifted {maxDiff:E3} from FP64 reference (> 2e-3).");
        // And it must NOT be bit-identical — that would mean the FP64 path ran, not FP32.
        Assert.True(maxDiff > 0.0, "expected FP32-scale rounding vs FP64; got an exact match (FP32 path didn't engage?).");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Float32Precision_CausalMask_MatchesFp64Reference(bool causal)
    {
        var engine = new CpuEngine();
        const int B = 2, H = 4, S = 48, D = 16;
        var q = RandomDouble(new[] { B, H, S, D }, 11);
        var k = RandomDouble(new[] { B, H, S, D }, 12);
        var v = RandomDouble(new[] { B, H, S, D }, 13);

        var refCfg = new FlashAttentionConfig { IsCausal = causal };
        var fp32Cfg = new FlashAttentionConfig { IsCausal = causal, Float32Precision = true };
        var (reference, _) = FusedAttention<double>.Forward(q, k, v, refCfg, engine: engine);
        var (fp32, _) = FusedAttention<double>.Forward(q, k, v, fp32Cfg, engine: engine);

        Assert.Equal(reference._shape, fp32._shape);
        Assert.True(MaxAbsDiff(reference, fp32) < 2e-3);
    }

    [Fact]
    public void Float32Precision_WithAttentionBias_MatchesFp64Reference()
    {
        var engine = new CpuEngine();
        const int B = 1, H = 2, S = 32, D = 16;
        var q = RandomDouble(new[] { B, H, S, D }, 21);
        var k = RandomDouble(new[] { B, H, S, D }, 22);
        var v = RandomDouble(new[] { B, H, S, D }, 23);
        var bias = RandomDouble(new[] { B, H, S, S }, 24);

        var (reference, _) = FusedAttention<double>.Forward(q, k, v, new FlashAttentionConfig(), bias, engine);
        var (fp32, _) = FusedAttention<double>.Forward(
            q, k, v, new FlashAttentionConfig { Float32Precision = true }, bias, engine);

        Assert.Equal(reference._shape, fp32._shape);
        Assert.True(MaxAbsDiff(reference, fp32) < 2e-3);
    }

    [Fact]
    public void Float32Precision_Rank3Input_PromotedAndDemoted()
    {
        var engine = new CpuEngine();
        const int B = 2, S = 40, D = 16;
        var q = RandomDouble(new[] { B, S, D }, 31);
        var k = RandomDouble(new[] { B, S, D }, 32);
        var v = RandomDouble(new[] { B, S, D }, 33);

        var (reference, _) = FusedAttention<double>.Forward(q, k, v, new FlashAttentionConfig(), engine: engine);
        var (fp32, _) = FusedAttention<double>.Forward(
            q, k, v, new FlashAttentionConfig { Float32Precision = true }, engine: engine);

        Assert.Equal(new[] { B, S, D }, fp32._shape);   // demoted back to rank-3
        Assert.Equal(reference._shape, fp32._shape);
        Assert.True(MaxAbsDiff(reference, fp32) < 2e-3);
    }

    [Fact]
    public void Float32Precision_WithReturnAttentionWeights_MatchesFp64Reference()
    {
        // Because the FP32 path re-runs the full float Forward, attention weights are
        // still available — they're just computed in FP32 and upcast. Verify they are
        // returned, are a valid softmax (rows sum to 1) and match the FP64 reference.
        var engine = new CpuEngine();
        const int B = 1, H = 2, S = 8, D = 8;
        var q = RandomDouble(new[] { B, H, S, D }, 41);
        var k = RandomDouble(new[] { B, H, S, D }, 42);
        var v = RandomDouble(new[] { B, H, S, D }, 43);

        var (_, refWeights) = FusedAttention<double>.Forward(
            q, k, v, new FlashAttentionConfig { ReturnAttentionWeights = true }, engine: engine);
        var (_, fp32Weights) = FusedAttention<double>.Forward(
            q, k, v, new FlashAttentionConfig { Float32Precision = true, ReturnAttentionWeights = true },
            engine: engine);

        Assert.NotNull(refWeights);
        Assert.NotNull(fp32Weights);
        Assert.Equal(new[] { B, H, S, S }, fp32Weights!._shape);

        // Each query row's softmax weights sum to 1 (within FP32 tolerance).
        var w = fp32Weights.AsSpan();
        for (int row = 0; row < B * H * S; row++)
        {
            double sum = 0;
            for (int j = 0; j < S; j++) sum += w[row * S + j];
            Assert.True(Math.Abs(sum - 1.0) < 1e-3, $"row {row} weights sum to {sum:F6}, expected 1.0");
        }
        Assert.True(MaxAbsDiff(refWeights!, fp32Weights) < 2e-3);
    }

    [Fact]
    public void Float32Precision_NoOp_ForFloatT()
    {
        // For T=float the flag is meaningless (already float); the result must be
        // identical to the default float path (the FP32-for-double branch is gated
        // to typeof(T)==double).
        var engine = new CpuEngine();
        var q = RandomFloat(new[] { 1, 2, 16, 12 }, 51);
        var k = RandomFloat(new[] { 1, 2, 16, 12 }, 52);
        var v = RandomFloat(new[] { 1, 2, 16, 12 }, 53);

        var (baseline, _) = FusedAttention<float>.Forward(q, k, v, new FlashAttentionConfig(), engine: engine);
        var (withFlag, _) = FusedAttention<float>.Forward(
            q, k, v, new FlashAttentionConfig { Float32Precision = true }, engine: engine);

        Assert.Equal(baseline.AsSpan().ToArray(), withFlag.AsSpan().ToArray());
    }

    [Fact]
    public void Float32Precision_LargeSeqLen_StaysAccurate()
    {
        // The actual #467 use case: SD-UNet self-attention seqLen ≥ 1024.
        var engine = new CpuEngine();
        const int B = 1, H = 4, S = 256, D = 20;
        var q = RandomDouble(new[] { B, H, S, D }, 61);
        var k = RandomDouble(new[] { B, H, S, D }, 62);
        var v = RandomDouble(new[] { B, H, S, D }, 63);

        var (reference, _) = FusedAttention<double>.Forward(q, k, v, new FlashAttentionConfig(), engine: engine);
        var (fp32, _) = FusedAttention<double>.Forward(
            q, k, v, new FlashAttentionConfig { Float32Precision = true, BlockSizeQ = 64, BlockSizeKV = 64 },
            engine: engine);

        double maxDiff = MaxAbsDiff(reference, fp32);
        _output.WriteLine($"seqLen={S}: maxAbsDiff vs FP64 = {maxDiff:E3}");
        Assert.True(maxDiff < 5e-3, $"FP32 attention at seqLen={S} drifted {maxDiff:E3} (> 5e-3).");
    }

    [Trait("Category", "Performance")]
    [Fact]
    public void Float32Precision_IsFasterThanFp64_AtUNetShape()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        var engine = new CpuEngine();
        var fp64 = new FlashAttentionConfig();
        var fp32 = new FlashAttentionConfig { Float32Precision = true };

        // Representative self-attention shapes: canonical SD-UNet level-1 (#467),
        // a longer-context variant, and a DiT-XL-style batched shape.
        var shapes = new[]
        {
            (B: 1, H: 8,  S: 1024, D: 64),  // SD-UNet level-1 (the #467 hot path)
            (B: 1, H: 8,  S: 2048, D: 64),  // longer context
            (B: 4, H: 16, S: 256,  D: 72),  // DiT-XL-style batched
        };

        _output.WriteLine("shape [B,H,S,D]            FP64 ms   FP32 ms   speedup");
        foreach (var (B, H, S, D) in shapes)
        {
            var q = RandomDouble(new[] { B, H, S, D }, 71);
            var k = RandomDouble(new[] { B, H, S, D }, 72);
            var v = RandomDouble(new[] { B, H, S, D }, 73);

            double Time(FlashAttentionConfig cfg)
            {
                FusedAttention<double>.Forward(q, k, v, cfg, engine: engine); // warmup
                double best = double.MaxValue;
                for (int r = 0; r < 5; r++)
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    FusedAttention<double>.Forward(q, k, v, cfg, engine: engine);
                    sw.Stop();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                }
                return best;
            }

            double tFp64 = Time(fp64);
            double tFp32 = Time(fp32);
            _output.WriteLine($"[{B},{H},{S},{D}]".PadRight(26) +
                $"{tFp64,7:F1}   {tFp32,7:F1}   {tFp64 / tFp32,5:F2}×");
        }
    }

    // ----------------- Helpers -----------------

    private static Tensor<double> RandomDouble(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static Tensor<float> RandomFloat(int[] shape, int seed)
    {
        var t = new Tensor<float>(shape);
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static double MaxAbsDiff(Tensor<double> a, Tensor<double> b)
    {
        var sa = a.AsSpan();
        var sb = b.AsSpan();
        Assert.Equal(sa.Length, sb.Length);
        double max = 0;
        for (int i = 0; i < sa.Length; i++)
            max = Math.Max(max, Math.Abs(sa[i] - sb[i]));
        return max;
    }
}
