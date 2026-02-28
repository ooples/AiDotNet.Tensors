using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class TensorLevelOpsTests
{
    private readonly CpuEngine _engine = new();
    private const float Tolerance = 1e-5f;

    #region TensorSigmoid

    [Fact]
    public void TensorSigmoid_ReturnsCorrectValues()
    {
        var tensor = new Tensor<float>(new[] { 4 });
        tensor[0] = 0f;
        tensor[1] = 1f;
        tensor[2] = -1f;
        tensor[3] = 5f;

        var result = _engine.TensorSigmoid(tensor);

        Assert.Equal(0.5f, result[0], Tolerance);
        Assert.Equal(0.7310586f, result[1], Tolerance);
        Assert.Equal(0.2689414f, result[2], Tolerance);
        Assert.True(result[3] > 0.99f && result[3] < 1f);
    }

    [Fact]
    public void TensorSigmoid_PreservesShape()
    {
        var tensor = new Tensor<float>(new[] { 2, 3 });
        var result = _engine.TensorSigmoid(tensor);

        Assert.Equal(new[] { 2, 3 }, result.Shape);
    }

    #endregion

    #region TensorReLU

    [Fact]
    public void TensorReLU_ReturnsCorrectValues()
    {
        var tensor = new Tensor<float>(new[] { 5 });
        tensor[0] = -3f;
        tensor[1] = -0.5f;
        tensor[2] = 0f;
        tensor[3] = 0.5f;
        tensor[4] = 3f;

        var result = _engine.TensorReLU(tensor);

        Assert.Equal(0f, result[0], Tolerance);
        Assert.Equal(0f, result[1], Tolerance);
        Assert.Equal(0f, result[2], Tolerance);
        Assert.Equal(0.5f, result[3], Tolerance);
        Assert.Equal(3f, result[4], Tolerance);
    }

    #endregion

    #region TensorGELU

    [Fact]
    public void TensorGELU_ReturnsCorrectValues()
    {
        var tensor = new Tensor<float>(new[] { 3 });
        tensor[0] = 0f;
        tensor[1] = 1f;
        tensor[2] = -1f;

        var result = _engine.TensorGELU(tensor);

        // GELU(0) = 0
        Assert.Equal(0f, result[0], Tolerance);
        // GELU(1) ~= 0.8413 (from standard GELU approximation)
        Assert.True(result[1] > 0.8f && result[1] < 0.9f);
        // GELU(-1) ~= -0.1587
        Assert.True(result[2] > -0.2f && result[2] < -0.1f);
    }

    #endregion

    #region TensorSiLU

    [Fact]
    public void TensorSiLU_ReturnsCorrectValues()
    {
        // SiLU(x) = x * sigmoid(x)
        var tensor = new Tensor<float>(new[] { 4 });
        tensor[0] = 0f;
        tensor[1] = 1f;
        tensor[2] = -1f;
        tensor[3] = 2f;

        var result = _engine.TensorSiLU(tensor);

        // SiLU(0) = 0 * 0.5 = 0
        Assert.Equal(0f, result[0], Tolerance);
        // SiLU(1) = 1 * sigmoid(1) ~= 0.7311
        Assert.Equal(0.7310586f, result[1], Tolerance);
        // SiLU(-1) = -1 * sigmoid(-1) ~= -0.2689
        Assert.Equal(-0.2689414f, result[2], Tolerance);
        // SiLU(2) = 2 * sigmoid(2) = 2 * (1/(1+exp(-2))) ~= 2 * 0.8808 ~= 1.7616
        Assert.Equal(2f * (1f / (1f + (float)Math.Exp(-2.0))), result[3], Tolerance);
    }

    #endregion

    #region TensorTanh

    [Fact]
    public void TensorTanh_ReturnsCorrectValues()
    {
        var tensor = new Tensor<float>(new[] { 3 });
        tensor[0] = 0f;
        tensor[1] = 1f;
        tensor[2] = -1f;

        var result = _engine.TensorTanh(tensor);

        Assert.Equal(0f, result[0], Tolerance);
        Assert.Equal((float)Math.Tanh(1.0), result[1], Tolerance);
        Assert.Equal((float)Math.Tanh(-1.0), result[2], Tolerance);
    }

    #endregion

    #region TensorLeakyReLU

    [Fact]
    public void TensorLeakyReLU_ReturnsCorrectValues()
    {
        var tensor = new Tensor<float>(new[] { 4 });
        tensor[0] = -2f;
        tensor[1] = 0f;
        tensor[2] = 1f;
        tensor[3] = 3f;

        var result = _engine.TensorLeakyReLU(tensor, 0.01f);

        Assert.Equal(-0.02f, result[0], Tolerance);
        Assert.Equal(0f, result[1], Tolerance);
        Assert.Equal(1f, result[2], Tolerance);
        Assert.Equal(3f, result[3], Tolerance);
    }

    #endregion

    #region TensorMish

    [Fact]
    public void TensorMish_ReturnsCorrectValues()
    {
        // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        var tensor = new Tensor<float>(new[] { 3 });
        tensor[0] = 0f;
        tensor[1] = 1f;
        tensor[2] = -1f;

        var result = _engine.TensorMish(tensor);

        // Mish(0) = 0 * tanh(ln(2)) = 0
        Assert.Equal(0f, result[0], Tolerance);
        // Mish(1) ~= 0.8651
        Assert.True(result[1] > 0.85f && result[1] < 0.87f);
        // Mish(-1) ~= -0.3034
        Assert.True(result[2] > -0.32f && result[2] < -0.28f);
    }

    #endregion

    #region TensorHardSwish

    [Fact]
    public void TensorHardSwish_ReturnsCorrectValues()
    {
        // HardSwish(x) = x * relu6(x + 3) / 6
        var tensor = new Tensor<float>(new[] { 5 });
        tensor[0] = -4f;  // <= -3: 0
        tensor[1] = -3f;  // = -3: 0
        tensor[2] = 0f;   // between: 0 * 3/6 = 0
        tensor[3] = 3f;   // = 3: 3
        tensor[4] = 4f;   // >= 3: 4

        var result = _engine.TensorHardSwish(tensor);

        Assert.Equal(0f, result[0], Tolerance);
        Assert.Equal(0f, result[1], Tolerance);
        Assert.Equal(0f, result[2], Tolerance);
        Assert.Equal(3f, result[3], Tolerance);
        Assert.Equal(4f, result[4], Tolerance);
    }

    #endregion

    #region TensorLayerNorm

    [Fact]
    public void TensorLayerNorm_NormalizesCorrectly()
    {
        // 2D tensor [2, 3]
        var input = new Tensor<float>(new[] { 2, 3 });
        input[0, 0] = 1f; input[0, 1] = 2f; input[0, 2] = 3f;
        input[1, 0] = 4f; input[1, 1] = 5f; input[1, 2] = 6f;

        var gamma = new Tensor<float>(new[] { 3 });
        gamma[0] = 1f; gamma[1] = 1f; gamma[2] = 1f;

        var beta = new Tensor<float>(new[] { 3 });
        beta[0] = 0f; beta[1] = 0f; beta[2] = 0f;

        var result = _engine.TensorLayerNorm(input, gamma, beta, 1e-5);

        Assert.Equal(new[] { 2, 3 }, result.Shape);
        // After normalization with identity gamma and zero beta,
        // each row should have mean ~0 and std ~1

        for (int row = 0; row < 2; row++)
        {
            // Compute mean of the row
            float sum = 0f;
            for (int col = 0; col < 3; col++)
            {
                sum += result[row, col];
            }

            float mean = sum / 3f;

            // Compute standard deviation of the row
            float varianceSum = 0f;
            for (int col = 0; col < 3; col++)
            {
                float diff = result[row, col] - mean;
                varianceSum += diff * diff;
            }

            float variance = varianceSum / 3f;
            float std = (float)Math.Sqrt(variance);

            Assert.Equal(0f, mean, Tolerance);
            Assert.Equal(1f, std, Tolerance);
        }
    }

    #endregion

    #region ReduceStd

    [Fact]
    public void ReduceStd_ComputesCorrectStandardDeviation()
    {
        var input = new Tensor<float>(new[] { 4 });
        input[0] = 2f;
        input[1] = 4f;
        input[2] = 4f;
        input[3] = 4f;

        var result = _engine.ReduceStd(input, new[] { 0 }, keepDims: false);

        // Mean = 3.5, Variance = ((2-3.5)^2 + (4-3.5)^2 + (4-3.5)^2 + (4-3.5)^2) / 4 = 3/4 = 0.75
        // Std = sqrt(0.75) ~= 0.8660
        Assert.True(result.Length == 1);
        Assert.Equal((float)Math.Sqrt(0.75), result[0], 0.01f);
    }

    [Fact]
    public void ReduceStd_AlongAxis_ComputesCorrectly()
    {
        // [2, 3] tensor
        var input = new Tensor<float>(new[] { 2, 3 });
        input[0, 0] = 1f; input[0, 1] = 2f; input[0, 2] = 3f;
        input[1, 0] = 4f; input[1, 1] = 5f; input[1, 2] = 6f;

        var result = _engine.ReduceStd(input, new[] { 0 }, keepDims: false);

        // Std along axis 0: std of [1,4], [2,5], [3,6]
        // Each pair has variance = ((x-mean)^2 + (y-mean)^2) / 2 = (1.5^2 + 1.5^2) / 2 = 2.25
        // Std = sqrt(2.25) = 1.5
        Assert.Equal(new[] { 3 }, result.Shape);
        Assert.Equal(1.5f, result[0], Tolerance);
        Assert.Equal(1.5f, result[1], Tolerance);
        Assert.Equal(1.5f, result[2], Tolerance);
    }

    #endregion

    #region TensorLerp

    [Fact]
    public void TensorLerp_InterpolatesCorrectly()
    {
        var a = new Tensor<float>(new[] { 4 });
        a[0] = 0f; a[1] = 10f; a[2] = 20f; a[3] = 100f;

        var b = new Tensor<float>(new[] { 4 });
        b[0] = 10f; b[1] = 20f; b[2] = 40f; b[3] = 200f;

        // t = 0.0 -> returns a
        var result0 = _engine.TensorLerp(a, b, 0f);
        Assert.Equal(0f, result0[0], Tolerance);
        Assert.Equal(10f, result0[1], Tolerance);
        Assert.Equal(20f, result0[2], Tolerance);
        Assert.Equal(100f, result0[3], Tolerance);

        // t = 1.0 -> returns b
        var result1 = _engine.TensorLerp(a, b, 1f);
        Assert.Equal(10f, result1[0], Tolerance);
        Assert.Equal(20f, result1[1], Tolerance);
        Assert.Equal(40f, result1[2], Tolerance);
        Assert.Equal(200f, result1[3], Tolerance);

        // t = 0.5 -> midpoint
        var resultHalf = _engine.TensorLerp(a, b, 0.5f);
        Assert.Equal(5f, resultHalf[0], Tolerance);
        Assert.Equal(15f, resultHalf[1], Tolerance);
        Assert.Equal(30f, resultHalf[2], Tolerance);
        Assert.Equal(150f, resultHalf[3], Tolerance);

        // t = 0.25
        var resultQuarter = _engine.TensorLerp(a, b, 0.25f);
        Assert.Equal(2.5f, resultQuarter[0], Tolerance);
        Assert.Equal(12.5f, resultQuarter[1], Tolerance);
        Assert.Equal(25f, resultQuarter[2], Tolerance);
        Assert.Equal(125f, resultQuarter[3], Tolerance);
    }

    [Fact]
    public void TensorLerp_PreservesShape()
    {
        var a = new Tensor<float>(new[] { 2, 3 });
        var b = new Tensor<float>(new[] { 2, 3 });

        var result = _engine.TensorLerp(a, b, 0.5f);

        Assert.Equal(new[] { 2, 3 }, result.Shape);
    }

    #endregion

    #region TensorAddScaled

    [Fact]
    public void TensorAddScaled_ComputesCorrectly()
    {
        // Common diffusion pattern: alpha * signal + sigma * noise
        var signal = new Tensor<float>(new[] { 4 });
        signal[0] = 1f; signal[1] = 2f; signal[2] = 3f; signal[3] = 4f;

        var noise = new Tensor<float>(new[] { 4 });
        noise[0] = 0.1f; noise[1] = 0.2f; noise[2] = 0.3f; noise[3] = 0.4f;

        // result = 0.8 * signal + 0.6 * noise
        var result = _engine.TensorAddScaled(signal, noise, 0.8f, 0.6f);

        Assert.Equal(0.8f * 1f + 0.6f * 0.1f, result[0], Tolerance);
        Assert.Equal(0.8f * 2f + 0.6f * 0.2f, result[1], Tolerance);
        Assert.Equal(0.8f * 3f + 0.6f * 0.3f, result[2], Tolerance);
        Assert.Equal(0.8f * 4f + 0.6f * 0.4f, result[3], Tolerance);
    }

    [Fact]
    public void TensorAddScaled_WithUnitScales_EquivalentToAdd()
    {
        var a = new Tensor<float>(new[] { 3 });
        a[0] = 1f; a[1] = 2f; a[2] = 3f;

        var b = new Tensor<float>(new[] { 3 });
        b[0] = 4f; b[1] = 5f; b[2] = 6f;

        // scaleA=1, scaleB=1 -> equivalent to TensorAdd
        var result = _engine.TensorAddScaled(a, b, 1f, 1f);
        var expected = _engine.TensorAdd(a, b);

        Assert.Equal(expected[0], result[0], Tolerance);
        Assert.Equal(expected[1], result[1], Tolerance);
        Assert.Equal(expected[2], result[2], Tolerance);
    }

    [Fact]
    public void TensorAddScaled_PreservesShape()
    {
        var a = new Tensor<float>(new[] { 2, 3 });
        var b = new Tensor<float>(new[] { 2, 3 });

        var result = _engine.TensorAddScaled(a, b, 1f, 1f);

        Assert.Equal(new[] { 2, 3 }, result.Shape);
    }

    #endregion

    #region TensorMaxPool2D

    [Fact]
    public void TensorMaxPool2D_DelegatesToMaxPool2D()
    {
        // Simple 4D tensor [1, 1, 4, 4]
        var input = new Tensor<float>(new[] { 1, 1, 4, 4 });
        for (int i = 0; i < 16; i++)
        {
            input.AsWritableSpan()[i] = i;
        }

        var result = _engine.TensorMaxPool2D(input, poolSize: 2, stride: 2);

        // With pool size 2, stride 2: [1, 1, 2, 2]
        Assert.Equal(4, result.Shape.Length);
        Assert.Equal(1, result.Shape[0]);
        Assert.Equal(1, result.Shape[1]);
        Assert.Equal(2, result.Shape[2]);
        Assert.Equal(2, result.Shape[3]);

        // Verify actual pooled values:
        // Input 4x4 (row-major): 0,1,2,3 / 4,5,6,7 / 8,9,10,11 / 12,13,14,15
        // Top-left 2x2 [0,1,4,5] -> max = 5
        Assert.Equal(5f, result.AsSpan()[0], Tolerance);
        // Top-right 2x2 [2,3,6,7] -> max = 7
        Assert.Equal(7f, result.AsSpan()[1], Tolerance);
        // Bottom-left 2x2 [8,9,12,13] -> max = 13
        Assert.Equal(13f, result.AsSpan()[2], Tolerance);
        // Bottom-right 2x2 [10,11,14,15] -> max = 15
        Assert.Equal(15f, result.AsSpan()[3], Tolerance);
    }

    #endregion

    #region TensorAvgPool2D

    [Fact]
    public void TensorAvgPool2D_DelegatesToAvgPool2D()
    {
        var input = new Tensor<float>(new[] { 1, 1, 4, 4 });
        for (int i = 0; i < 16; i++)
        {
            input.AsWritableSpan()[i] = i; // Varying values 0..15
        }

        var result = _engine.TensorAvgPool2D(input, poolSize: 2, stride: 2);

        Assert.Equal(4, result.Shape.Length);
        Assert.Equal(1, result.Shape[0]);
        Assert.Equal(1, result.Shape[1]);
        Assert.Equal(2, result.Shape[2]);
        Assert.Equal(2, result.Shape[3]);

        // Input 4x4 (row-major): 0,1,2,3 / 4,5,6,7 / 8,9,10,11 / 12,13,14,15
        // Top-left 2x2 [0,1,4,5] -> avg = (0+1+4+5)/4 = 2.5
        Assert.Equal(2.5f, result.AsSpan()[0], Tolerance);
        // Top-right 2x2 [2,3,6,7] -> avg = (2+3+6+7)/4 = 4.5
        Assert.Equal(4.5f, result.AsSpan()[1], Tolerance);
        // Bottom-left 2x2 [8,9,12,13] -> avg = (8+9+12+13)/4 = 10.5
        Assert.Equal(10.5f, result.AsSpan()[2], Tolerance);
        // Bottom-right 2x2 [10,11,14,15] -> avg = (10+11+14+15)/4 = 12.5
        Assert.Equal(12.5f, result.AsSpan()[3], Tolerance);
    }

    #endregion

    #region TensorConv2D

    [Fact]
    public void TensorConv2D_DelegatesToConv2D()
    {
        // [1, 1, 4, 4] input, [1, 1, 3, 3] kernel
        var input = new Tensor<float>(new[] { 1, 1, 4, 4 });
        for (int i = 0; i < 16; i++)
        {
            input.AsWritableSpan()[i] = 1f;
        }

        var kernel = new Tensor<float>(new[] { 1, 1, 3, 3 });
        for (int i = 0; i < 9; i++)
        {
            kernel.AsWritableSpan()[i] = 1f;
        }

        var result = _engine.TensorConv2D(input, kernel, stride: 1, padding: 0);

        // 4x4 input with 3x3 kernel, stride 1, no padding -> 2x2 output
        Assert.Equal(4, result.Shape.Length);
        Assert.Equal(1, result.Shape[0]);
        Assert.Equal(1, result.Shape[1]);
        Assert.Equal(2, result.Shape[2]);
        Assert.Equal(2, result.Shape[3]);

        // Each element should be sum of 3x3 window of 1s = 9
        for (int i = 0; i < result.Length; i++)
        {
            Assert.Equal(9f, result.AsSpan()[i], Tolerance);
        }
    }

    #endregion

    #region API Consistency

    [Fact]
    public void TensorPrefixAliases_MatchOriginalMethods()
    {
        var input = new Tensor<float>(new[] { 4 });
        input[0] = -1f; input[1] = 0f; input[2] = 0.5f; input[3] = 2f;

        // TensorSigmoid matches Sigmoid
        var sig1 = _engine.Sigmoid(input);
        var sig2 = _engine.TensorSigmoid(input);
        for (int i = 0; i < 4; i++)
            Assert.Equal(sig1[i], sig2[i], Tolerance);

        // TensorReLU matches ReLU
        var relu1 = _engine.ReLU(input);
        var relu2 = _engine.TensorReLU(input);
        for (int i = 0; i < 4; i++)
            Assert.Equal(relu1[i], relu2[i], Tolerance);

        // TensorGELU matches GELU
        var gelu1 = _engine.GELU(input);
        var gelu2 = _engine.TensorGELU(input);
        for (int i = 0; i < 4; i++)
            Assert.Equal(gelu1[i], gelu2[i], Tolerance);

        // TensorTanh matches Tanh
        var tanh1 = _engine.Tanh(input);
        var tanh2 = _engine.TensorTanh(input);
        for (int i = 0; i < 4; i++)
            Assert.Equal(tanh1[i], tanh2[i], Tolerance);

        // TensorMish matches Mish
        var mish1 = _engine.Mish(input);
        var mish2 = _engine.TensorMish(input);
        for (int i = 0; i < 4; i++)
            Assert.Equal(mish1[i], mish2[i], Tolerance);

        // TensorSiLU matches Swish
        var silu1 = _engine.Swish(input);
        var silu2 = _engine.TensorSiLU(input);
        for (int i = 0; i < 4; i++)
            Assert.Equal(silu1[i], silu2[i], Tolerance);

        // TensorLeakyReLU matches LeakyReLU
        var lrelu1 = _engine.LeakyReLU(input, 0.01f);
        var lrelu2 = _engine.TensorLeakyReLU(input, 0.01f);
        for (int i = 0; i < 4; i++)
            Assert.Equal(lrelu1[i], lrelu2[i], Tolerance);

        // TensorHardSwish matches HardSwish
        var hs1 = _engine.HardSwish(input);
        var hs2 = _engine.TensorHardSwish(input);
        for (int i = 0; i < 4; i++)
            Assert.Equal(hs1[i], hs2[i], Tolerance);
    }

    #endregion
}
