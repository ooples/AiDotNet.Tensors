using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The float32 conv lays the whole batch's im2col columns side by side and runs one GEMM - forward, kernel and input
/// gradients - for strided convs and small output planes. Checked against a direct-loop reference; the gradients both
/// overwriting and accumulating.
/// </summary>
public class ConvBatchedTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    public static TheoryData<int, int, int, int, int, int, int, int, int> Shapes => new()
    {
        // batch, inC, outC, h, w, k, stride, pad, dilation
        { 4, 8, 16, 16, 16, 3, 2, 1, 1 },  // strided 3x3 (ResNet downsample)
        { 4, 8, 16, 16, 16, 1, 2, 0, 1 },  // strided 1x1 projection shortcut
        { 3, 16, 16, 8, 8, 3, 1, 1, 1 },   // stride 1, 8x8 plane: batched input gradient
        { 2, 12, 24, 4, 4, 3, 1, 1, 1 },   // stride 1, 4x4 plane
        { 3, 5, 7, 11, 9, 3, 2, 1, 1 },    // odd channels, non-square
        { 2, 6, 10, 13, 13, 3, 2, 2, 2 },  // dilated, extra padding
        { 2, 4, 6, 9, 7, 5, 3, 1, 1 },     // 5x5 stride 3
        { 2, 136, 8, 4, 4, 3, 1, 1, 1 },   // colH 1224: the forward GEMM runs on BlasManaged
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void MatchesDirectReference(int batch, int inC, int outC, int h, int w, int k, int stride, int pad, int dil)
    {
        int oh = (h + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        int ow = (w + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        var x = Rand(new[] { batch, inC, h, w }, 1);
        var kernel = Rand(new[] { outC, inC, k, k }, 2);
        var dy = Rand(new[] { batch, outC, oh, ow }, 3);

        var refDx = new float[x.Length];
        var refDw = new float[kernel.Length];
        for (int b = 0; b < batch; b++)
            for (int o = 0; o < outC; o++)
                for (int y = 0; y < oh; y++)
                    for (int z = 0; z < ow; z++)
                    {
                        float g = dy[((b * outC + o) * oh + y) * ow + z];
                        for (int c = 0; c < inC; c++)
                            for (int i = 0; i < k; i++)
                                for (int j = 0; j < k; j++)
                                {
                                    int ih = y * stride - pad + i * dil, iw = z * stride - pad + j * dil;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    int xi = ((b * inC + c) * h + ih) * w + iw;
                                    int wi = ((o * inC + c) * k + i) * k + j;
                                    refDx[xi] += g * kernel[wi];
                                    refDw[wi] += g * x[xi];
                                }
                    }

        var refY = new float[dy.Length];
        for (int b = 0; b < batch; b++)
            for (int o = 0; o < outC; o++)
                for (int y = 0; y < oh; y++)
                    for (int z = 0; z < ow; z++)
                    {
                        float sum = 0;
                        for (int c = 0; c < inC; c++)
                            for (int i = 0; i < k; i++)
                                for (int j = 0; j < k; j++)
                                {
                                    int ih = y * stride - pad + i * dil, iw = z * stride - pad + j * dil;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    sum += x[((b * inC + c) * h + ih) * w + iw] * kernel[((o * inC + c) * k + i) * k + j];
                                }
                        refY[((b * outC + o) * oh + y) * ow + z] = sum;
                    }

        var engine = new CpuEngine();
        AssertClose(refY, engine.Conv2D(x, kernel, new[] { stride, stride }, new[] { pad, pad }, new[] { dil, dil }), "forward");
        var st = new[] { stride, stride };
        var pd = new[] { pad, pad };
        var dl = new[] { dil, dil };
        var dx = Rand(x._shape, 4);   // garbage: overwrite must ignore it
        var dw = Rand(kernel._shape, 5);
        engine.Conv2DBackwardInputInto(dx, dy, kernel, x._shape, st, pd, dl, accumulate: false);
        engine.Conv2DBackwardKernelInto(dw, dy, x, kernel._shape, st, pd, dl, accumulate: false);
        AssertClose(refDx, dx, "dX");
        AssertClose(refDw, dw, "dW");

        // Accumulating adds onto what is there.
        var dxAcc = Rand(x._shape, 6);
        var dwAcc = Rand(kernel._shape, 7);
        var dxBase = (float[])dxAcc.GetFlattenedData().Clone();
        var dwBase = (float[])dwAcc.GetFlattenedData().Clone();
        engine.Conv2DBackwardInputInto(dxAcc, dy, kernel, x._shape, st, pd, dl, accumulate: true);
        engine.Conv2DBackwardKernelInto(dwAcc, dy, x, kernel._shape, st, pd, dl, accumulate: true);
        for (int i = 0; i < refDx.Length; i++) refDx[i] += dxBase[i];
        for (int i = 0; i < refDw.Length; i++) refDw[i] += dwBase[i];
        AssertClose(refDx, dxAcc, "dX accumulate");
        AssertClose(refDw, dwAcc, "dW accumulate");
    }

    private static void AssertClose(float[] expected, Tensor<float> actual, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = 1e-4f * Math.Max(1f, Math.Abs(expected[i]));
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol, $"{name}[{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }
}
