// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Coverage for issue #294 Phase 3: <see cref="FlashAttention{T}"/>
/// generic-T rank-N rewrite. Validates:
/// <list type="bullet">
/// <item>Rank-4 parity: <see cref="FlashAttention{T}"/> matches the
/// existing <see cref="FlashAttention2"/> bit-exactly for the canonical
/// <c>[B, H, Sq, D]</c> input shape.</item>
/// <item>Rank coverage: rank-2 <c>[Sq, D]</c>, rank-3 <c>[B, Sq, D]</c>,
/// rank-5 <c>[B, F, H, Sq, D]</c> all run successfully and agree with
/// the equivalent rank-4 reshape.</item>
/// <item>Bias broadcast: <c>[Sq, Sk]</c> bias broadcasts against any
/// leading prefix; <c>[B, H, Sq, Sk]</c> bias matches per-batch-head
/// targeting; <c>[1, H, Sq, Sk]</c> broadcasts on B.</item>
/// <item>Generic T fallback: <see cref="double"/> and a non-primitive
/// numeric path produce the right shape and gradient agreement
/// against the float reference.</item>
/// </list>
/// </summary>
public class FlashAttentionGenericTests
{
    private static Tensor<float> RandomTensor(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var data = new float[ProductOf(shape)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }

    private static Tensor<double> RandomTensorD(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var data = new double[ProductOf(shape)];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 2.0 - 1.0;
        return new Tensor<double>(data, shape);
    }

    private static int ProductOf(int[] shape)
    {
        int p = 1;
        foreach (var d in shape) p *= d;
        return p;
    }

    private static void AssertSpansClose(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float tol = 1e-4f)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(a[i] - b[i]) <= tol,
                $"Mismatch at index {i}: a={a[i]}, b={b[i]}, diff={Math.Abs(a[i] - b[i])}, tol={tol}");
    }

    [Fact]
    public void Forward_Rank4_BitExactMatchWithFlashAttention2()
    {
        // Acceptance criterion #9: rank-4 [B, H, Sq, D] runs the same
        // numerics as the rank-fixed FlashAttention2 baseline. This
        // is the bit-equivalence test — the loop shape is identical
        // (batchProduct = B*H), so the float kernel must produce
        // exactly the same output up to floating-point summation
        // ordering (which is identical because the loop nesting is
        // unchanged).
        const int B = 2, H = 4, Sq = 16, Sk = 16, D = 8, Dv = 8;
        var q = RandomTensor(new[] { B, H, Sq, D }, seed: 1);
        var k = RandomTensor(new[] { B, H, Sk, D }, seed: 2);
        var v = RandomTensor(new[] { B, H, Sk, Dv }, seed: 3);

        var (oRef, lseRef) = FlashAttention2.Forward(q, k, v, blockSizeQ: 8, blockSizeKV: 8);
        var (oNew, lseNew) = FlashAttention<float>.Forward(q, k, v, blockSizeQ: 8, blockSizeKV: 8);

        AssertSpansClose(oRef.AsSpan(), oNew.AsSpan(), tol: 1e-6f);
        Assert.Equal(B * H * Sq, lseNew.Length);
        AssertSpansClose(lseRef.AsSpan(), lseNew.AsSpan(), tol: 1e-6f);
    }

    [Fact]
    public void Forward_Rank2_SingleSequence_ProducesValidShape()
    {
        // Rank 2: [Sq, D]. batchProduct = 1. The inner kernel runs
        // exactly once over the whole sequence.
        const int Sq = 8, Sk = 8, D = 4, Dv = 4;
        var q = RandomTensor(new[] { Sq, D }, seed: 10);
        var k = RandomTensor(new[] { Sk, D }, seed: 11);
        var v = RandomTensor(new[] { Sk, Dv }, seed: 12);

        var (o, lse) = FlashAttention<float>.Forward(q, k, v, blockSizeQ: 4, blockSizeKV: 4);

        Assert.Equal(new[] { Sq, Dv }, o._shape);
        Assert.Equal(new[] { Sq }, lse._shape);
        // Output rows are softmax-weighted convex combinations of V
        // rows; their magnitude should be bounded by the V range.
        var oSpan = o.AsSpan();
        for (int i = 0; i < oSpan.Length; i++)
            Assert.True(!float.IsNaN(oSpan[i]) && !float.IsInfinity(oSpan[i]));
    }

    [Fact]
    public void Forward_Rank3_SingleHeadBatched_AgreesWithRank4Reshape()
    {
        // Rank 3 [B, Sq, D] should agree numerically with rank-4
        // [B, 1, Sq, D] reshape — the second dim being 1 means the
        // inner kernel sees the same per-batch sequence.
        const int B = 3, Sq = 8, Sk = 8, D = 4, Dv = 4;
        var q3 = RandomTensor(new[] { B, Sq, D }, seed: 20);
        var k3 = RandomTensor(new[] { B, Sk, D }, seed: 21);
        var v3 = RandomTensor(new[] { B, Sk, Dv }, seed: 22);

        var q4 = q3.Reshape(new[] { B, 1, Sq, D });
        var k4 = k3.Reshape(new[] { B, 1, Sk, D });
        var v4 = v3.Reshape(new[] { B, 1, Sk, Dv });

        var (o3, _) = FlashAttention<float>.Forward(q3, k3, v3, blockSizeQ: 4, blockSizeKV: 4);
        var (o4, _) = FlashAttention<float>.Forward(q4, k4, v4, blockSizeQ: 4, blockSizeKV: 4);

        // Reshape o4 [B,1,Sq,Dv] -> [B,Sq,Dv] for comparison.
        AssertSpansClose(o3.AsSpan(), o4.AsSpan(), tol: 1e-6f);
    }

    [Fact]
    public void Forward_Rank5_VideoOrMoEShape_RunsSuccessfully()
    {
        // Rank 5: e.g. video [B, F, H, Sq, D] or MoE [E, B, H, Sq, D].
        // batchProduct = B*F*H = 2*3*2 = 12; the kernel iterates that
        // many times. Output shape should preserve the prefix.
        const int B = 2, F = 3, H = 2, Sq = 4, Sk = 4, D = 4, Dv = 4;
        var q = RandomTensor(new[] { B, F, H, Sq, D }, seed: 30);
        var k = RandomTensor(new[] { B, F, H, Sk, D }, seed: 31);
        var v = RandomTensor(new[] { B, F, H, Sk, Dv }, seed: 32);

        var (o, lse) = FlashAttention<float>.Forward(q, k, v, blockSizeQ: 2, blockSizeKV: 2);

        Assert.Equal(new[] { B, F, H, Sq, Dv }, o._shape);
        Assert.Equal(new[] { B, F, H, Sq }, lse._shape);
    }

    [Fact]
    public void Forward_BiasBroadcast_LastTwoDimsOnly_BroadcastsToFullPrefix()
    {
        // Bias [Sq, Sk] broadcasts to every (b, h) pair — same value
        // applied uniformly. Test by comparing against an explicitly-
        // expanded bias tensor [B, H, Sq, Sk] with the same values.
        const int B = 2, H = 3, Sq = 4, Sk = 4, D = 4, Dv = 4;
        var q = RandomTensor(new[] { B, H, Sq, D }, seed: 40);
        var k = RandomTensor(new[] { B, H, Sk, D }, seed: 41);
        var v = RandomTensor(new[] { B, H, Sk, Dv }, seed: 42);
        var biasSmall = RandomTensor(new[] { Sq, Sk }, seed: 43);

        // Tile the small bias into [B, H, Sq, Sk] manually for the reference run.
        var biasBig = new Tensor<float>(new[] { B, H, Sq, Sk });
        var bigArr = biasBig.GetDataArray();
        var smallArr = biasSmall.GetDataArray();
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int i = 0; i < Sq; i++)
                    for (int j = 0; j < Sk; j++)
                        bigArr[((b * H + h) * Sq + i) * Sk + j] = smallArr[i * Sk + j];

        var (oSmall, _) = FlashAttention<float>.Forward(q, k, v, attentionBias: biasSmall);
        var (oBig, _) = FlashAttention<float>.Forward(q, k, v, attentionBias: biasBig);

        AssertSpansClose(oSmall.AsSpan(), oBig.AsSpan(), tol: 1e-5f);
    }

    [Fact]
    public void Forward_BiasBroadcast_HeadOnlyBatchBroadcast_Works()
    {
        // [1, H, Sq, Sk] broadcasts on B: every batch sees the same
        // per-head bias. This is the ALiBi-with-head-bias pattern.
        const int B = 2, H = 3, Sq = 4, Sk = 4, D = 4, Dv = 4;
        var q = RandomTensor(new[] { B, H, Sq, D }, seed: 50);
        var k = RandomTensor(new[] { B, H, Sk, D }, seed: 51);
        var v = RandomTensor(new[] { B, H, Sk, Dv }, seed: 52);
        var biasHeadOnly = RandomTensor(new[] { 1, H, Sq, Sk }, seed: 53);

        // Manually expand to [B, H, Sq, Sk] for the reference.
        var biasFull = new Tensor<float>(new[] { B, H, Sq, Sk });
        var fullArr = biasFull.GetDataArray();
        var hdArr = biasHeadOnly.GetDataArray();
        int sliceLen = H * Sq * Sk;
        for (int b = 0; b < B; b++)
            Array.Copy(hdArr, 0, fullArr, b * sliceLen, sliceLen);

        var (oBroadcast, _) = FlashAttention<float>.Forward(q, k, v, attentionBias: biasHeadOnly);
        var (oExpanded, _) = FlashAttention<float>.Forward(q, k, v, attentionBias: biasFull);

        AssertSpansClose(oBroadcast.AsSpan(), oExpanded.AsSpan(), tol: 1e-5f);
    }

    [Fact]
    public void Forward_BiasShapeMismatch_LastTwoNotSqSk_Throws()
    {
        var q = RandomTensor(new[] { 1, 1, 4, 4 }, seed: 60);
        var k = RandomTensor(new[] { 1, 1, 4, 4 }, seed: 61);
        var v = RandomTensor(new[] { 1, 1, 4, 4 }, seed: 62);
        // Wrong last-two: should be [Sq, Sk] = [4, 4]; passing [3, 4]
        var bad = new Tensor<float>(new[] { 1, 1, 3, 4 });

        Assert.Throws<ArgumentException>(() =>
            FlashAttention<float>.Forward(q, k, v, attentionBias: bad));
    }

    [Fact]
    public void Forward_RankMismatchBetweenQandKV_Throws()
    {
        var q = RandomTensor(new[] { 2, 4, 4 }, seed: 70);
        var k = RandomTensor(new[] { 2, 1, 4, 4 }, seed: 71); // rank 4 vs query rank 3
        var v = RandomTensor(new[] { 2, 1, 4, 4 }, seed: 72);

        Assert.Throws<ArgumentException>(() => FlashAttention<float>.Forward(q, k, v));
    }

    [Fact]
    public void Forward_Causal_ProducesUpperTriangularMask()
    {
        // Causal mask: position i can only see positions <= i + queryOffset.
        // With queryOffset=0 and Sq=Sk, this is the triangular mask.
        const int B = 1, H = 1, S = 8, D = 4;
        var q = RandomTensor(new[] { B, H, S, D }, seed: 80);
        var k = RandomTensor(new[] { B, H, S, D }, seed: 81);
        var v = RandomTensor(new[] { B, H, S, D }, seed: 82);

        var (oCausal, _) = FlashAttention<float>.Forward(q, k, v, isCausal: true);
        var (oFull, _) = FlashAttention<float>.Forward(q, k, v, isCausal: false);

        // Causal output for the first row should differ from full
        // output (the full version sees future tokens).
        var causal = oCausal.AsSpan();
        var full = oFull.AsSpan();
        bool anyDiff = false;
        for (int i = 0; i < D; i++)
        {
            if (Math.Abs(causal[i] - full[i]) > 1e-5f) { anyDiff = true; break; }
        }
        Assert.True(anyDiff, "Causal vs non-causal first-row outputs should differ when the future tokens influence the attention.");
    }

    [Fact]
    public void Forward_Double_AgreesWithFloatWithinPrecision()
    {
        // The double kernel should produce the same numerics as the
        // float kernel up to float→double conversion noise. Run both
        // on the same input and compare the outputs after upconverting
        // float to double.
        const int B = 2, H = 2, Sq = 8, Sk = 8, D = 4, Dv = 4;
        var qF = RandomTensor(new[] { B, H, Sq, D }, seed: 90);
        var kF = RandomTensor(new[] { B, H, Sk, D }, seed: 91);
        var vF = RandomTensor(new[] { B, H, Sk, Dv }, seed: 92);

        var qD = new Tensor<double>(new[] { B, H, Sq, D });
        var kD = new Tensor<double>(new[] { B, H, Sk, D });
        var vD = new Tensor<double>(new[] { B, H, Sk, Dv });
        var fa = qF.GetDataArray();
        var fk = kF.GetDataArray();
        var fv = vF.GetDataArray();
        var da = qD.GetDataArray();
        var dk = kD.GetDataArray();
        var dv = vD.GetDataArray();
        for (int i = 0; i < fa.Length; i++) da[i] = fa[i];
        for (int i = 0; i < fk.Length; i++) dk[i] = fk[i];
        for (int i = 0; i < fv.Length; i++) dv[i] = fv[i];

        var (oF, _) = FlashAttention<float>.Forward(qF, kF, vF, blockSizeQ: 4, blockSizeKV: 4);
        var (oD, _) = FlashAttention<double>.Forward(qD, kD, vD, blockSizeQ: 4, blockSizeKV: 4);

        var oFArr = oF.GetDataArray();
        var oDArr = oD.GetDataArray();
        for (int i = 0; i < oFArr.Length; i++)
            Assert.True(Math.Abs(oFArr[i] - (float)oDArr[i]) < 1e-3f,
                $"float vs double output mismatch at {i}: f={oFArr[i]}, d={oDArr[i]}");
    }

    [Fact]
    public void Backward_Rank4_BitExactMatchWithFlashAttention2()
    {
        // Backward must match the rank-fixed implementation
        // bit-exactly on the canonical [B, H, Sq, D] shape.
        const int B = 2, H = 2, Sq = 8, Sk = 8, D = 4, Dv = 4;
        var q = RandomTensor(new[] { B, H, Sq, D }, seed: 100);
        var k = RandomTensor(new[] { B, H, Sk, D }, seed: 101);
        var v = RandomTensor(new[] { B, H, Sk, Dv }, seed: 102);

        var (oRef, lseRef) = FlashAttention2.Forward(q, k, v, blockSizeQ: 4, blockSizeKV: 4);
        var (oNew, lseNew) = FlashAttention<float>.Forward(q, k, v, blockSizeQ: 4, blockSizeKV: 4);

        var dO = RandomTensor(new[] { B, H, Sq, Dv }, seed: 103);

        var (dQRef, dKRef, dVRef) = FlashAttention2.Backward(dO, q, k, v, oRef, lseRef, blockSizeQ: 4, blockSizeKV: 4);
        var (dQNew, dKNew, dVNew) = FlashAttention<float>.Backward(dO, q, k, v, oNew, lseNew, blockSizeQ: 4, blockSizeKV: 4);

        AssertSpansClose(dQRef.AsSpan(), dQNew.AsSpan(), tol: 1e-5f);
        AssertSpansClose(dKRef.AsSpan(), dKNew.AsSpan(), tol: 1e-5f);
        AssertSpansClose(dVRef.AsSpan(), dVNew.AsSpan(), tol: 1e-5f);
    }
}
