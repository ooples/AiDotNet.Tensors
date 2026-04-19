using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Audit tests closing the #198 spec gaps the first PR missed:
/// - D. queryOffset / KV-cache causal semantics
/// - E. BlockSizeQ / BlockSizeKV + DropoutRate config surface
/// - G. Backward pass verified against finite-difference gradients
/// - I. ALiBi-style bias vector
/// </summary>
public class FusedAttentionAuditTests
{
    [Fact]
    public void FlashAttentionConfig_DefaultFactory_HasSpecDefaults()
    {
        var c = FlashAttentionConfig.Default;
        Assert.Null(c.Scale);
        Assert.Null(c.ScaleFactor);
        Assert.False(c.IsCausal);
        Assert.False(c.ReturnAttentionWeights);
        Assert.Null(c.BlockSizeQ);
        Assert.Null(c.BlockSizeKV);
        Assert.Null(c.BlockSize);
        Assert.Equal(0, c.QueryOffset);
        Assert.Null(c.DropoutRate);
    }

    [Fact]
    public void FlashAttentionConfig_BlockSize_SetsBothAxes()
    {
        var c = new FlashAttentionConfig { BlockSize = 64 };
        Assert.Equal(64, c.BlockSizeQ);
        Assert.Equal(64, c.BlockSizeKV);
    }

    [Fact]
    public void FlashAttentionConfig_ScaleFactor_IsAliasForScale()
    {
        var c = new FlashAttentionConfig { ScaleFactor = 0.1 };
        Assert.Equal(0.1, c.Scale);
    }

    [Fact]
    public void QueryOffset_KVCacheDecode_MatchesManualSdpaOnFullSequence()
    {
        // Typical autoregressive decode: sq=1, queryOffset=t, sk=t+1.
        // The single decode query should attend to every past key.
        var engine = new CpuEngine();
        const int B = 1, H = 1, Dk = 4, Dv = 4, t = 3;
        var qSlice = RandomTensor(new[] { B, H, 1, Dk }, 301);
        var k = RandomTensor(new[] { B, H, t + 1, Dk }, 302);
        var v = RandomTensor(new[] { B, H, t + 1, Dv }, 303);
        var cfg = new FlashAttentionConfig { IsCausal = true, QueryOffset = t };

        var (out1, _) = FusedAttention<float>.Forward(qSlice, k, v, cfg, engine: engine);
        Assert.Equal(new[] { B, H, 1, Dv }, out1._shape);

        // Reference: non-causal SDPA over the full sequence — at the last
        // query position (index t), the causal mask matches "attend to all"
        // which is exactly what the decoder wants here. We build a matching
        // reference by taking a [B,H,t+1,Dk] query whose row t IS qSlice[0],
        // running causal SDPA, then comparing row t.
        var qFull = new Tensor<float>(new[] { B, H, t + 1, Dk });
        var qFullSpan = qFull.AsWritableSpan();
        var qSliceSpan = qSlice.AsSpan();
        for (int d = 0; d < Dk; d++)
            qFullSpan[t * Dk + d] = qSliceSpan[d];
        var cfgFull = new FlashAttentionConfig { IsCausal = true, QueryOffset = 0 };
        var (refOut, _) = FusedAttention<float>.Forward(qFull, k, v, cfgFull, engine: engine);
        var refRow = refOut.AsSpan().Slice(t * Dv, Dv).ToArray();

        var ourRow = out1.AsSpan().ToArray();
        for (int i = 0; i < Dv; i++)
            Assert.Equal(refRow[i], ourRow[i], 4);
    }

    [Fact]
    public void QueryOffset_OutOfBounds_Throws()
    {
        var engine = new CpuEngine();
        var q = RandomTensor(new[] { 1, 1, 2, 4 }, 1);
        var k = RandomTensor(new[] { 1, 1, 3, 4 }, 2);
        var v = RandomTensor(new[] { 1, 1, 3, 4 }, 3);
        // queryOffset=2 + seqQ=2 = 4 > seqKV=3 → reject
        var cfg = new FlashAttentionConfig { QueryOffset = 2 };
        Assert.Throws<ArgumentException>(() =>
            FusedAttention<float>.Forward(q, k, v, cfg, engine: engine));
    }

    [Fact]
    public void Backward_MatchesFiniteDifference_OnTinyInputs()
    {
        // Finite-difference gradient check — small inputs, float64 math
        // for accuracy. This is the #198-I acceptance test.
        var engine = new CpuEngine();
        const int B = 1, H = 1, Sq = 2, Sk = 3, Dk = 2, Dv = 2;
        var q = RandomTensorD(new[] { B, H, Sq, Dk }, 11);
        var k = RandomTensorD(new[] { B, H, Sk, Dk }, 12);
        var v = RandomTensorD(new[] { B, H, Sk, Dv }, 13);
        var cfg = new FlashAttentionConfig { Scale = 1.0 };

        // Loss = sum(output) → analytical dLoss/dOutput is ones-tensor.
        var (output, _) = FusedAttention<double>.Forward(q, k, v, cfg, engine: engine);
        var onesGrad = new Tensor<double>(output._shape);
        var ogSpan = onesGrad.AsWritableSpan();
        for (int i = 0; i < ogSpan.Length; i++) ogSpan[i] = 1.0;

        var (dQ, dK, dV) = FusedAttention<double>.Backward(onesGrad, q, k, v, cfg, engine: engine);

        // Perturb each element of q/k/v and compare
        // (loss(x+eps) - loss(x-eps)) / (2*eps) to the analytical gradient.
        CheckFiniteDiff("dQ", q, dQ, (perturbed) =>
            FusedAttention<double>.Forward(perturbed, k, v, cfg, engine: engine).Output);
        CheckFiniteDiff("dK", k, dK, (perturbed) =>
            FusedAttention<double>.Forward(q, perturbed, v, cfg, engine: engine).Output);
        CheckFiniteDiff("dV", v, dV, (perturbed) =>
            FusedAttention<double>.Forward(q, k, perturbed, cfg, engine: engine).Output);
    }

    private static void CheckFiniteDiff(
        string name,
        Tensor<double> param,
        Tensor<double> analyticalGrad,
        Func<Tensor<double>, Tensor<double>> forward)
    {
        const double eps = 1e-5;
        var analyticalData = analyticalGrad.AsSpan().ToArray();
        var paramData = param.GetDataArray();

        for (int i = 0; i < paramData.Length; i++)
        {
            double orig = paramData[i];
            paramData[i] = orig + eps;
            double lossPlus = Sum(forward(param));
            paramData[i] = orig - eps;
            double lossMinus = Sum(forward(param));
            paramData[i] = orig;
            double numeric = (lossPlus - lossMinus) / (2 * eps);
            double analytical = analyticalData[i];
            double err = Math.Abs(numeric - analytical);
            Assert.True(err < 1e-4,
                $"{name}[{i}]: numeric {numeric} vs analytical {analytical}, err {err}");
        }
    }

    [Fact]
    public void Backward_WithAttentionBias_RunsAndReturnsShapes()
    {
        // Smoke test that backward survives the bias path + produces
        // correctly-shaped gradients. Finite-diff is expensive enough that
        // the previous test uses no bias; this one exercises the path.
        var engine = new CpuEngine();
        const int B = 1, H = 1, Sq = 2, Sk = 2, Dk = 2, Dv = 2;
        var q = RandomTensor(new[] { B, H, Sq, Dk }, 1);
        var k = RandomTensor(new[] { B, H, Sk, Dk }, 2);
        var v = RandomTensor(new[] { B, H, Sk, Dv }, 3);
        var bias = RandomTensor(new[] { B, H, Sq, Sk }, 4);
        var grad = RandomTensor(new[] { B, H, Sq, Dv }, 5);

        var (dQ, dK, dV) = FusedAttention<float>.Backward(
            grad, q, k, v, new FlashAttentionConfig(), bias, engine);
        Assert.Equal(q._shape, dQ._shape);
        Assert.Equal(k._shape, dK._shape);
        Assert.Equal(v._shape, dV._shape);
    }

    [Fact]
    public void AlibiStyleBias_ApproximatesMonotonicPositionalDecay()
    {
        // ALiBi adds a linear bias that grows negative with distance. With
        // no input signal difference between keys, the attention over the
        // closest key should dominate. Verify output ≈ V at key=last.
        var engine = new CpuEngine();
        const int B = 1, H = 1, Sq = 1, Sk = 4, D = 2;
        var q = OnesTensor(new[] { B, H, Sq, D });
        var k = OnesTensor(new[] { B, H, Sk, D });
        var v = new Tensor<float>(new[] { B, H, Sk, D });
        var vSpan = v.AsWritableSpan();
        // Distinct per-key values so we can tell which key won.
        for (int pos = 0; pos < Sk; pos++)
            for (int d = 0; d < D; d++)
                vSpan[pos * D + d] = pos + 1f;

        // ALiBi bias: distance_from_query × -slope, where the query is at
        // position Sk-1 (causal tail). Nearest key has 0 bias; farthest is
        // -slope * (Sk - 1).
        const float slope = 4f;
        var bias = new Tensor<float>(new[] { B, H, Sq, Sk });
        var bSpan = bias.AsWritableSpan();
        for (int kPos = 0; kPos < Sk; kPos++)
        {
            int distance = (Sk - 1) - kPos;
            bSpan[kPos] = -slope * distance;
        }

        var cfg = new FlashAttentionConfig();
        var (output, _) = FusedAttention<float>.Forward(q, k, v, cfg, bias, engine);
        var o = output.AsSpan().ToArray();
        // Output should approach V[last] = Sk (Sk+0 intercept is 4 here).
        for (int d = 0; d < D; d++)
            Assert.Equal((float)Sk, o[d], 1);
    }

    private static Tensor<float> RandomTensor(int[] shape, int seed)
    {
        var t = new Tensor<float>(shape);
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor<double> RandomTensorD(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static Tensor<float> OnesTensor(int[] shape)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = 1f;
        return t;
    }

    private static double Sum(Tensor<double> t)
    {
        var s = t.AsSpan();
        double acc = 0;
        for (int i = 0; i < s.Length; i++) acc += s[i];
        return acc;
    }
}
