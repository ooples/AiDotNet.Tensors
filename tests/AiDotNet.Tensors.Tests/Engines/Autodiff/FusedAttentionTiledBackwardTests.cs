using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// #1662 lever #3 parity gate: <see cref="FusedAttention{T}.Backward"/> must produce
/// dQ / dK / dV that match a straightforward, independent reference across sequence
/// lengths and head counts.
///
/// <para>The reference is a NAIVE nested-loop implementation of the analytic attention
/// backward (dV = P^T dO, dP = dO V^T, dS = P*(dP - rowsum(dP*P)), dQ = scale·dS K,
/// dK = scale·dS^T Q) — it shares no code with the kernel under test, so it validates the
/// kernel's tiling logic, not just self-consistency. Written test-first: it passes against
/// the current stored-P backward (proving the reference is correct) and must keep passing
/// after the backward is rewritten to a tiled O(S)-memory form.</para>
/// </summary>
public class FusedAttentionTiledBackwardTests
{
    [Theory]
    [InlineData(1, 2, 16, 8)]    // tiny (full-matrix path)
    [InlineData(2, 4, 64, 16)]   // medium, multi-batch/head (full-matrix path)
    [InlineData(1, 8, 256, 32)]  // long sequence -> tiled path, 2 even key tiles (128+128)
    [InlineData(1, 4, 200, 16)]  // tiled path, UNEVEN last tile (128+72)
    [InlineData(2, 4, 320, 16)]  // tiled path, multi-batch/head, 3 tiles (128+128+64)
    public void Backward_MatchesNaiveReference_dQdKdV(int B, int H, int S, int Dh)
    {
        var engine = new CpuEngine();
        var q  = RandomTensor(new[] { B, H, S, Dh }, 1);
        var k  = RandomTensor(new[] { B, H, S, Dh }, 2);
        var v  = RandomTensor(new[] { B, H, S, Dh }, 3);
        var dO = RandomTensor(new[] { B, H, S, Dh }, 4);

        var (refQ, refK, refV) = ReferenceBackward(B, H, S, Dh, dO, q, k, v);
        var (gQ, gK, gV) = FusedAttention<float>.Backward(dO, q, k, v, engine: engine);

        AssertClose(refQ, gQ.AsSpan().ToArray(), 1e-3f, "dQ");
        AssertClose(refK, gK.AsSpan().ToArray(), 1e-3f, "dK");
        AssertClose(refV, gV.AsSpan().ToArray(), 1e-3f, "dV");
    }

    // Naive analytic attention backward over [B,H,S,Dh] row-major tensors. No engine ops.
    private static (float[] dQ, float[] dK, float[] dV) ReferenceBackward(
        int B, int H, int S, int Dh, Tensor<float> dOT, Tensor<float> qT, Tensor<float> kT, Tensor<float> vT)
    {
        float[] dO = dOT.AsSpan().ToArray();
        float[] q = qT.AsSpan().ToArray();
        float[] k = kT.AsSpan().ToArray();
        float[] v = vT.AsSpan().ToArray();
        float scale = (float)(1.0 / Math.Sqrt(Dh));

        var dQ = new float[B * H * S * Dh];
        var dK = new float[B * H * S * Dh];
        var dV = new float[B * H * S * Dh];

        int Off(int b, int h) => ((b * H) + h) * S * Dh; // start of [b,h,:,:]

        var P = new float[S * S];   // per-(b,h) softmax weights
        var dP = new float[S * S];
        var dS = new float[S * S];
        var rowsum = new float[S];

        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {
            int o = Off(b, h);

            // scores[i,j] = scale * <q_i, k_j>, then softmax over j -> P[i,j]
            for (int i = 0; i < S; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < S; j++)
                {
                    float s = 0f;
                    for (int d = 0; d < Dh; d++) s += q[o + i * Dh + d] * k[o + j * Dh + d];
                    s *= scale;
                    P[i * S + j] = s;
                    if (s > max) max = s;
                }
                float sum = 0f;
                for (int j = 0; j < S; j++) { float e = (float)Math.Exp(P[i * S + j] - max); P[i * S + j] = e; sum += e; }
                for (int j = 0; j < S; j++) P[i * S + j] /= sum;
            }

            // dP[i,j] = <dO_i, v_j>
            for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
            {
                float s = 0f;
                for (int e = 0; e < Dh; e++) s += dO[o + i * Dh + e] * v[o + j * Dh + e];
                dP[i * S + j] = s;
            }

            // dV[j,e] = sum_i P[i,j] * dO[i,e]
            for (int j = 0; j < S; j++)
            for (int e = 0; e < Dh; e++)
            {
                float s = 0f;
                for (int i = 0; i < S; i++) s += P[i * S + j] * dO[o + i * Dh + e];
                dV[o + j * Dh + e] = s;
            }

            // rowsum_i = sum_j dP[i,j]*P[i,j];  dS[i,j] = P[i,j]*(dP[i,j]-rowsum_i)
            for (int i = 0; i < S; i++)
            {
                float rs = 0f;
                for (int j = 0; j < S; j++) rs += dP[i * S + j] * P[i * S + j];
                rowsum[i] = rs;
                for (int j = 0; j < S; j++) dS[i * S + j] = P[i * S + j] * (dP[i * S + j] - rs);
            }

            // dQ[i,d] = scale * sum_j dS[i,j]*k[j,d]
            for (int i = 0; i < S; i++)
            for (int d = 0; d < Dh; d++)
            {
                float s = 0f;
                for (int j = 0; j < S; j++) s += dS[i * S + j] * k[o + j * Dh + d];
                dQ[o + i * Dh + d] = scale * s;
            }

            // dK[j,d] = scale * sum_i dS[i,j]*q[i,d]
            for (int j = 0; j < S; j++)
            for (int d = 0; d < Dh; d++)
            {
                float s = 0f;
                for (int i = 0; i < S; i++) s += dS[i * S + j] * q[o + i * Dh + d];
                dK[o + j * Dh + d] = scale * s;
            }
        }
        return (dQ, dK, dV);
    }

    private static void AssertClose(float[] expected, float[] actual, float tol, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxAbs = 0f;
        int worst = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = Math.Abs(expected[i] - actual[i]);
            // relative tolerance for larger magnitudes
            float bound = tol * (1f + Math.Abs(expected[i]));
            if (d > bound && d > maxAbs) { maxAbs = d; worst = i; }
        }
        Assert.True(worst < 0,
            $"{name} mismatch at index {worst}: expected {(worst >= 0 ? expected[worst] : 0)}, " +
            $"actual {(worst >= 0 ? actual[worst] : 0)} (absdiff {maxAbs}, tol {tol})");
    }

    private static Tensor<float> RandomTensor(int[] shape, int seed)
    {
        var r = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(r.NextDouble() - 0.5);
        return t;
    }
}
