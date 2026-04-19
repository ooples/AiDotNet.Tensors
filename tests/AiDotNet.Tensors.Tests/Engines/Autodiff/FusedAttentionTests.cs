using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for issue #198 — <see cref="FusedAttention{T}.Forward"/>.
/// Covers the feature gaps the existing
/// <see cref="IEngine.ScaledDotProductAttention{T}"/> doesn't support:
/// rank-3 inputs, additive attention bias, config toggles.
/// </summary>
public class FusedAttentionTests
{
    [Fact]
    public void Forward_Rank4_NoBias_MatchesEngineSdpa()
    {
        // With no bias, FusedAttention should route through engine.SDPA
        // and produce the same bytes (they share the kernel).
        var engine = new CpuEngine();
        const int B = 2, H = 2, S = 4, D = 3;
        var q = RandomTensor(new[] { B, H, S, D }, 1);
        var k = RandomTensor(new[] { B, H, S, D }, 2);
        var v = RandomTensor(new[] { B, H, S, D }, 3);

        var (ours, _) = FusedAttention<float>.Forward(q, k, v, engine: engine);
        var ref_ = engine.ScaledDotProductAttention(q, k, v, mask: null, scale: null, out _);

        Assert.Equal(ref_.AsSpan().ToArray(), ours.AsSpan().ToArray());
    }

    [Fact]
    public void Forward_Rank3_IsPromotedAndDemoted()
    {
        var engine = new CpuEngine();
        const int B = 2, S = 4, D = 3;
        var q = RandomTensor(new[] { B, S, D }, 10);
        var k = RandomTensor(new[] { B, S, D }, 11);
        var v = RandomTensor(new[] { B, S, D }, 12);

        var (out3D, _) = FusedAttention<float>.Forward(q, k, v, engine: engine);
        Assert.Equal(new[] { B, S, D }, out3D._shape);

        // Manual reshape + engine SDPA should match.
        var q4 = engine.Reshape(q, new[] { B, 1, S, D });
        var k4 = engine.Reshape(k, new[] { B, 1, S, D });
        var v4 = engine.Reshape(v, new[] { B, 1, S, D });
        var ref_ = engine.ScaledDotProductAttention(q4, k4, v4, mask: null, scale: null, out _);
        var refReshaped = engine.Reshape(ref_, new[] { B, S, D });

        Assert.Equal(refReshaped.AsSpan().ToArray(), out3D.AsSpan().ToArray());
    }

    [Fact]
    public void Forward_ReturnAttentionWeights_NullByDefault()
    {
        var engine = new CpuEngine();
        var q = RandomTensor(new[] { 1, 1, 4, 3 }, 21);
        var k = RandomTensor(new[] { 1, 1, 4, 3 }, 22);
        var v = RandomTensor(new[] { 1, 1, 4, 3 }, 23);

        var (_, weights) = FusedAttention<float>.Forward(q, k, v, engine: engine);
        Assert.Null(weights);

        var (_, weights2) = FusedAttention<float>.Forward(
            q, k, v, new FlashAttentionConfig { ReturnAttentionWeights = true }, engine: engine);
        Assert.NotNull(weights2);
        Assert.Equal(new[] { 1, 1, 4, 4 }, weights2!._shape);
    }

    [Fact]
    public void Forward_AttentionBias_AltersOutput()
    {
        // Additive bias should change the output vs the unbiased case.
        var engine = new CpuEngine();
        const int B = 1, H = 1, Sq = 3, Sk = 3, D = 2;
        var q = RandomTensor(new[] { B, H, Sq, D }, 100);
        var k = RandomTensor(new[] { B, H, Sk, D }, 101);
        var v = RandomTensor(new[] { B, H, Sk, D }, 102);

        var (noBias, _) = FusedAttention<float>.Forward(q, k, v, engine: engine);

        // Strong positive bias on the first key dominates the softmax.
        var bias = new Tensor<float>(new[] { B, H, Sq, Sk });
        var bs = bias.AsWritableSpan();
        for (int i = 0; i < bs.Length; i++) bs[i] = 0f;
        for (int q_i = 0; q_i < Sq; q_i++)
            bs[q_i * Sk + 0] = 10f;  // heavy weight on k=0
        var (withBias, _) = FusedAttention<float>.Forward(q, k, v, attentionBias: bias, engine: engine);

        Assert.NotEqual(noBias.AsSpan().ToArray(), withBias.AsSpan().ToArray());
        // Output should approach V[0] due to dominant softmax weight.
        var vData = v.AsSpan().ToArray();
        var biased = withBias.AsSpan().ToArray();
        for (int q_i = 0; q_i < Sq; q_i++)
            for (int d = 0; d < D; d++)
                Assert.Equal(vData[d], biased[q_i * D + d], 2);
    }

    [Fact]
    public void Forward_NullInput_Throws()
    {
        var engine = new CpuEngine();
        var q = RandomTensor(new[] { 1, 1, 2, 2 }, 1);
        Assert.Throws<ArgumentNullException>(() =>
            FusedAttention<float>.Forward(null!, q, q, engine: engine));
        Assert.Throws<ArgumentNullException>(() =>
            FusedAttention<float>.Forward(q, null!, q, engine: engine));
        Assert.Throws<ArgumentNullException>(() =>
            FusedAttention<float>.Forward(q, q, null!, engine: engine));
    }

    [Fact]
    public void Forward_RankMismatch_Throws()
    {
        var engine = new CpuEngine();
        var q3 = RandomTensor(new[] { 1, 2, 3 }, 1);
        var k4 = RandomTensor(new[] { 1, 1, 2, 3 }, 2);
        // q rank-3 but k rank-4 → should throw.
        Assert.Throws<ArgumentException>(() =>
            FusedAttention<float>.Forward(q3, k4, k4, engine: engine));
    }

    [Fact]
    public void FlashAttentionConfig_Defaults_AreNull()
    {
        // Per issue — options use nullable + industry-default internal.
        var c = new FlashAttentionConfig();
        Assert.Null(c.Scale);
        Assert.False(c.IsCausal);
        Assert.False(c.ReturnAttentionWeights);
        Assert.Null(c.BlockSize);
    }

    [Fact]
    public void Forward_CustomScale_DifferentFromDefault()
    {
        // Scale affects the softmax distribution; different scale → different output.
        var engine = new CpuEngine();
        var q = RandomTensor(new[] { 1, 1, 3, 4 }, 1);
        var k = RandomTensor(new[] { 1, 1, 3, 4 }, 2);
        var v = RandomTensor(new[] { 1, 1, 3, 4 }, 3);
        var (defaultOut, _) = FusedAttention<float>.Forward(q, k, v, engine: engine);
        var (customOut, _) = FusedAttention<float>.Forward(
            q, k, v, new FlashAttentionConfig { Scale = 0.1 }, engine: engine);
        Assert.NotEqual(defaultOut.AsSpan().ToArray(), customOut.AsSpan().ToArray());
    }

    private static Tensor<float> RandomTensor(int[] shape, int seed)
    {
        var t = new Tensor<float>(shape);
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
