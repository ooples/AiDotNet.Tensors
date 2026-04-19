using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class FlashAttention2Tests
{
    [Fact]
    public void Forward_MatchesReferenceSoftmaxMatmul()
    {
        // FA-2 must match a naive (QK^T / √d) * softmax * V reference
        // within floating-point tolerance.
        const int B = 1, H = 1, Sq = 4, Sk = 4, D = 4;
        var q = Random(new[] { B, H, Sq, D }, 1);
        var k = Random(new[] { B, H, Sk, D }, 2);
        var v = Random(new[] { B, H, Sk, D }, 3);

        var (output, _) = FlashAttention2.Forward(q, k, v, blockSizeQ: 2, blockSizeKV: 2);
        var reference = NaiveAttention(q, k, v);

        var o = output.AsSpan().ToArray();
        var r = reference.AsSpan().ToArray();
        for (int i = 0; i < o.Length; i++)
            Assert.Equal(r[i], o[i], 4);
    }

    [Fact]
    public void Forward_Causal_MatchesReferenceWithMask()
    {
        const int B = 1, H = 1, S = 4, D = 4;
        var q = Random(new[] { B, H, S, D }, 10);
        var k = Random(new[] { B, H, S, D }, 11);
        var v = Random(new[] { B, H, S, D }, 12);

        var (output, _) = FlashAttention2.Forward(q, k, v, blockSizeQ: 2, blockSizeKV: 2, isCausal: true);
        var reference = NaiveAttention(q, k, v, isCausal: true);

        var o = output.AsSpan().ToArray();
        var r = reference.AsSpan().ToArray();
        for (int i = 0; i < o.Length; i++)
            Assert.Equal(r[i], o[i], 4);
    }

    [Fact]
    public void Forward_VariousBlockSizes_IdenticalOutputs()
    {
        // Block size is a memory / scheduling knob, not a numerics
        // parameter — output must be identical across choices.
        const int B = 1, H = 1, S = 8, D = 4;
        var q = Random(new[] { B, H, S, D }, 20);
        var k = Random(new[] { B, H, S, D }, 21);
        var v = Random(new[] { B, H, S, D }, 22);

        var (outSmall, _) = FlashAttention2.Forward(q, k, v, blockSizeQ: 2, blockSizeKV: 2);
        var (outBig, _) = FlashAttention2.Forward(q, k, v, blockSizeQ: 8, blockSizeKV: 8);
        var (outMixed, _) = FlashAttention2.Forward(q, k, v, blockSizeQ: 4, blockSizeKV: 2);

        var s1 = outSmall.AsSpan().ToArray();
        var s2 = outBig.AsSpan().ToArray();
        var s3 = outMixed.AsSpan().ToArray();
        for (int i = 0; i < s1.Length; i++)
        {
            Assert.Equal(s1[i], s2[i], 4);
            Assert.Equal(s1[i], s3[i], 4);
        }
    }

    [Fact]
    public void Backward_GradientsMatchFiniteDifference()
    {
        const int B = 1, H = 1, Sq = 3, Sk = 3, D = 2;
        var q = Random(new[] { B, H, Sq, D }, 100);
        var k = Random(new[] { B, H, Sk, D }, 101);
        var v = Random(new[] { B, H, Sk, D }, 102);

        var (output, lse) = FlashAttention2.Forward(q, k, v, blockSizeQ: 2, blockSizeKV: 2);
        var gradOut = new Tensor<float>(output._shape);
        var gs = gradOut.AsWritableSpan();
        for (int i = 0; i < gs.Length; i++) gs[i] = 1f; // dLoss/dOutput = 1 (loss = sum)

        var (dQ, dK, dV) = FlashAttention2.Backward(gradOut, q, k, v, output, lse, 2, 2);

        const float eps = 1e-3f;
        CheckFiniteDiff(q, dQ, (pert) => LossSum(FlashAttention2.Forward(pert, k, v).Output));
        CheckFiniteDiff(k, dK, (pert) => LossSum(FlashAttention2.Forward(q, pert, v).Output));
        CheckFiniteDiff(v, dV, (pert) => LossSum(FlashAttention2.Forward(q, k, pert).Output));
    }

    [Fact]
    public void Forward_CausalAcrossBlockBoundary()
    {
        // Specifically exercise the "whole K-block after the last visible
        // key" break condition — needs Sq ≥ Bc and an early-break block.
        const int B = 1, H = 1, S = 8, D = 2;
        var q = Random(new[] { B, H, S, D }, 200);
        var k = Random(new[] { B, H, S, D }, 201);
        var v = Random(new[] { B, H, S, D }, 202);

        var (outputTiled, _) = FlashAttention2.Forward(q, k, v, 2, 2, isCausal: true);
        var reference = NaiveAttention(q, k, v, isCausal: true);

        var o = outputTiled.AsSpan().ToArray();
        var r = reference.AsSpan().ToArray();
        for (int i = 0; i < o.Length; i++)
            Assert.Equal(r[i], o[i], 4);
    }

    // ─────── helpers ────────

    private static Tensor<float> Random(int[] shape, int seed)
    {
        var t = new Tensor<float>(shape);
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static float LossSum(Tensor<float> t)
    {
        float acc = 0f;
        var s = t.AsSpan();
        for (int i = 0; i < s.Length; i++) acc += s[i];
        return acc;
    }

    private static void CheckFiniteDiff(Tensor<float> param, Tensor<float> grad, Func<Tensor<float>, float> loss)
    {
        const float eps = 1e-3f;
        var p = param.GetDataArray();
        var g = grad.AsSpan().ToArray();
        for (int i = 0; i < p.Length; i++)
        {
            float orig = p[i];
            p[i] = orig + eps; float fp = loss(param);
            p[i] = orig - eps; float fm = loss(param);
            p[i] = orig;
            float numeric = (fp - fm) / (2 * eps);
            Assert.True(Math.Abs(numeric - g[i]) < 1e-2f,
                $"finite-diff [{i}]: numeric={numeric} analytical={g[i]}");
        }
    }

    private static Tensor<float> NaiveAttention(
        Tensor<float> q, Tensor<float> k, Tensor<float> v, bool isCausal = false)
    {
        int B = q._shape[0], H = q._shape[1], Sq = q._shape[2], D = q._shape[3];
        int Sk = k._shape[2], Dv = v._shape[3];
        float scale = 1f / (float)Math.Sqrt(D);
        var output = new Tensor<float>(new[] { B, H, Sq, Dv });
        var qData = q.GetDataArray();
        var kData = k.GetDataArray();
        var vData = v.GetDataArray();
        var oData = output.GetDataArray();

        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {
            var scores = new float[Sq * Sk];
            for (int i = 0; i < Sq; i++)
            for (int j = 0; j < Sk; j++)
            {
                if (isCausal && j > i) { scores[i * Sk + j] = float.NegativeInfinity; continue; }
                float acc = 0f;
                int qRow = ((b * H + h) * Sq + i) * D;
                int kRow = ((b * H + h) * Sk + j) * D;
                for (int d = 0; d < D; d++) acc += qData[qRow + d] * kData[kRow + d];
                scores[i * Sk + j] = acc * scale;
            }
            // Softmax per row.
            for (int i = 0; i < Sq; i++)
            {
                float m = float.NegativeInfinity;
                for (int j = 0; j < Sk; j++) if (scores[i * Sk + j] > m) m = scores[i * Sk + j];
                float sum = 0f;
                for (int j = 0; j < Sk; j++) { scores[i * Sk + j] = (float)Math.Exp(scores[i * Sk + j] - m); sum += scores[i * Sk + j]; }
                for (int j = 0; j < Sk; j++) scores[i * Sk + j] /= (sum == 0f ? 1f : sum);
            }
            // P @ V.
            for (int i = 0; i < Sq; i++)
            for (int d = 0; d < Dv; d++)
            {
                float acc = 0f;
                for (int j = 0; j < Sk; j++)
                {
                    int vRow = ((b * H + h) * Sk + j) * Dv;
                    acc += scores[i * Sk + j] * vData[vRow + d];
                }
                int oRow = ((b * H + h) * Sq + i) * Dv;
                oData[oRow + d] = acc;
            }
        }
        return output;
    }
}
