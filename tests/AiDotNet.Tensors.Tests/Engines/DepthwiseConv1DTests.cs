using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

// DepthwiseConv1D is implemented by reshaping [B,C,L] -> [B,C,1,L] and delegating to the
// (already-tested) DepthwiseConv2D. These tests pin the reshape wiring: the forward and both
// backward passes must match an independent scalar 1D depthwise-conv reference across a spread
// of strides, paddings, and channel-multipliers. Kernel layout is [channels, multiplier, K];
// output channels = channels * multiplier (oc = ic*mult + m), no cross-channel mixing.
public class DepthwiseConv1DTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static Tensor<float> Rnd(int[] shape, int seed)
    {
        int n = 1; foreach (var d in shape) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)System.Math.Sin(i * 0.7 + seed) * 0.5f;
        return new Tensor<float>(a, shape);
    }

    private static int OutLen(int L, int K, int s, int p) => (L + 2 * p - K) / s + 1;

    // out[b,oc,ol] = sum_k input[b,ic, ol*s + k - p] * kernel[ic,m,k]
    private static float[] RefForward(float[] inp, float[] k, int batch, int inC, int mult,
        int L, int K, int s, int p, int oL)
    {
        int outC = inC * mult;
        var outp = new float[batch * outC * oL];
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outC; oc++)
            {
                int ic = oc / mult, m = oc % mult;
                for (int ol = 0; ol < oL; ol++)
                {
                    float sum = 0f;
                    for (int kk = 0; kk < K; kk++)
                    {
                        int il = ol * s + kk - p;
                        if (il < 0 || il >= L) continue;
                        sum += inp[(b * inC + ic) * L + il] * k[(ic * mult + m) * K + kk];
                    }
                    outp[(b * outC + oc) * oL + ol] = sum;
                }
            }
        return outp;
    }

    private static float[] RefInput(float[] go, float[] k, int batch, int inC, int mult,
        int L, int K, int s, int p, int oL)
    {
        int outC = inC * mult;
        var gi = new float[batch * inC * L];
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outC; oc++)
            {
                int ic = oc / mult, m = oc % mult;
                for (int ol = 0; ol < oL; ol++)
                {
                    float g = go[(b * outC + oc) * oL + ol];
                    for (int kk = 0; kk < K; kk++)
                    {
                        int il = ol * s + kk - p;
                        if (il < 0 || il >= L) continue;
                        gi[(b * inC + ic) * L + il] += g * k[(ic * mult + m) * K + kk];
                    }
                }
            }
        return gi;
    }

    private static float[] RefKernel(float[] go, float[] inp, int batch, int inC, int mult,
        int L, int K, int s, int p, int oL)
    {
        int outC = inC * mult;
        var gk = new float[inC * mult * K];
        for (int ic = 0; ic < inC; ic++)
            for (int m = 0; m < mult; m++)
            {
                int oc = ic * mult + m;
                for (int kk = 0; kk < K; kk++)
                {
                    float sum = 0f;
                    for (int b = 0; b < batch; b++)
                        for (int ol = 0; ol < oL; ol++)
                        {
                            int il = ol * s + kk - p;
                            if (il < 0 || il >= L) continue;
                            sum += go[(b * outC + oc) * oL + ol] * inp[(b * inC + ic) * L + il];
                        }
                    gk[(ic * mult + m) * K + kk] = sum;
                }
            }
        return gk;
    }

    [Theory]
    [InlineData(2, 3, 1, 8, 3, 1, 1)] // stride 1, "same" padding, multiplier 1
    [InlineData(1, 4, 1, 16, 5, 2, 2)] // stride 2, multiplier 1
    [InlineData(2, 3, 2, 10, 3, 1, 0)] // multiplier 2, no padding
    [InlineData(1, 2, 1, 12, 7, 3, 1)] // large kernel
    public void Forward_MatchesScalarReference(int batch, int inC, int mult, int L, int K, int s, int p)
    {
        var input = Rnd(new[] { batch, inC, L }, 1);
        var kernel = Rnd(new[] { inC, mult, K }, 2);
        int oL = OutLen(L, K, s, p);

        var actual = _engine.DepthwiseConv1D(input, kernel, s, p);
        var expected = RefForward(input.GetDataArray(), kernel.GetDataArray(), batch, inC, mult, L, K, s, p, oL);

        Assert.Equal(new[] { batch, inC * mult, oL }, actual.Shape.ToArray());
        var act = actual.GetDataArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], act[i], 4);
    }

    [Theory]
    [InlineData(2, 3, 1, 8, 3, 1, 1)]
    [InlineData(1, 4, 1, 16, 5, 2, 2)]
    [InlineData(2, 3, 2, 10, 3, 1, 0)]
    public void BackwardInput_MatchesScalarReference(int batch, int inC, int mult, int L, int K, int s, int p)
    {
        int oL = OutLen(L, K, s, p);
        var gradOut = Rnd(new[] { batch, inC * mult, oL }, 3);
        var kernel = Rnd(new[] { inC, mult, K }, 4);

        var actual = _engine.DepthwiseConv1DBackwardInput(gradOut, kernel, new[] { batch, inC, L }, s, p);
        var expected = RefInput(gradOut.GetDataArray(), kernel.GetDataArray(), batch, inC, mult, L, K, s, p, oL);

        Assert.Equal(new[] { batch, inC, L }, actual.Shape.ToArray());
        var act = actual.GetDataArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], act[i], 4);
    }

    [Theory]
    [InlineData(2, 3, 1, 8, 3, 1, 1)]
    [InlineData(1, 4, 1, 16, 5, 2, 2)]
    [InlineData(2, 3, 2, 10, 3, 1, 0)]
    public void BackwardKernel_MatchesScalarReference(int batch, int inC, int mult, int L, int K, int s, int p)
    {
        int oL = OutLen(L, K, s, p);
        var gradOut = Rnd(new[] { batch, inC * mult, oL }, 5);
        var input = Rnd(new[] { batch, inC, L }, 6);

        var actual = _engine.DepthwiseConv1DBackwardKernel(gradOut, input, new[] { inC, mult, K }, s, p);
        var expected = RefKernel(gradOut.GetDataArray(), input.GetDataArray(), batch, inC, mult, L, K, s, p, oL);

        Assert.Equal(new[] { inC, mult, K }, actual.Shape.ToArray());
        var act = actual.GetDataArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], act[i], 4);
    }
}
