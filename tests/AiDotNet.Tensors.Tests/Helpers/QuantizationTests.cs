using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class QuantizationTests
{
    [Fact]
    public void QuantizeInt8_Symmetric_RoundTrips()
    {
        var input = new Tensor<float>(new[] { 8 });
        var span = input.AsWritableSpan();
        span[0] = -1f; span[1] = -0.5f; span[2] = 0f; span[3] = 0.5f;
        span[4] = 1f; span[5] = -0.25f; span[6] = 0.75f; span[7] = -0.75f;

        var qparams = QuantizationParams.FromTensor(input, QuantizationMode.Symmetric);
        var quantized = Quantization.QuantizeInt8(input, qparams);
        var dequantized = Quantization.DequantizeInt8(quantized, qparams);

        double mse = Quantization.ComputeQuantizationError(input, dequantized);
        Assert.True(mse < 0.001, $"INT8 symmetric quantization error {mse} too high");
    }

    [Fact]
    public void QuantizeInt8_Asymmetric_RoundTrips()
    {
        var input = new Tensor<float>(new[] { 8 });
        var span = input.AsWritableSpan();
        span[0] = 0f; span[1] = 0.1f; span[2] = 0.5f; span[3] = 0.9f;
        span[4] = 1f; span[5] = 0.25f; span[6] = 0.75f; span[7] = 0.3f;

        var qparams = QuantizationParams.FromTensor(input, QuantizationMode.Asymmetric);
        var quantized = Quantization.QuantizeInt8(input, qparams);
        var dequantized = Quantization.DequantizeInt8(quantized, qparams);

        double mse = Quantization.ComputeQuantizationError(input, dequantized);
        Assert.True(mse < 0.001, $"INT8 asymmetric quantization error {mse} too high");
    }

    [Fact]
    public void QuantizeFP16_RoundTrips()
    {
        var input = new Tensor<float>(new[] { 4 });
        var span = input.AsWritableSpan();
        span[0] = -1.5f; span[1] = 0f; span[2] = 3.14159f; span[3] = 100f;

        var fp16 = Quantization.QuantizeFP16(input);
        var dequantized = Quantization.DequantizeFP16(fp16);

        double mse = Quantization.ComputeQuantizationError(input, dequantized);
        Assert.True(mse < 0.01, $"FP16 quantization error {mse} too high");
    }

    [Fact]
    public void QuantizeInt8_PreservesShape()
    {
        var input = new Tensor<float>(new[] { 2, 3, 4 });
        var qparams = QuantizationParams.FromTensor(input);
        var quantized = Quantization.QuantizeInt8(input, qparams);

        Assert.Equal(input.Shape, quantized.Shape);
    }

    [Fact]
    public void QuantizeInt8_ClampsToRange()
    {
        var input = new Tensor<float>(new[] { 4 });
        var span = input.AsWritableSpan();
        span[0] = -1000f; span[1] = 1000f; span[2] = 0f; span[3] = 1f;

        var qparams = new QuantizationParams(scale: 1f, zeroPoint: 0, QuantizationMode.Symmetric);
        var quantized = Quantization.QuantizeInt8(input, qparams);

        Assert.Equal(-128, quantized.AsSpan()[0]); // clamped
        Assert.Equal(127, quantized.AsSpan()[1]);   // clamped
        Assert.Equal(0, quantized.AsSpan()[2]);
        Assert.Equal(1, quantized.AsSpan()[3]);
    }

    [Fact]
    public void QuantizationParams_Symmetric_ZeroPointIsZero()
    {
        var input = new Tensor<float>(new[] { 100 });
        var rng = new Random(42);
        for (int i = 0; i < 100; i++)
            input.AsWritableSpan()[i] = (float)(rng.NextDouble() * 2 - 1);

        var qparams = QuantizationParams.FromTensor(input, QuantizationMode.Symmetric);
        Assert.Equal(0, qparams.ZeroPoint);
        Assert.True(qparams.Scale > 0);
    }

    [Fact]
    public void QuantizationParams_AllZeros_HandledGracefully()
    {
        var input = new Tensor<float>(new[] { 4 }); // all zeros
        var qparams = QuantizationParams.FromTensor(input);
        Assert.True(qparams.Scale > 0); // shouldn't be zero
    }

    [Fact]
    public void LargeArray_QuantizationErrorBounded()
    {
        var input = new Tensor<float>(new[] { 10000 });
        var rng = new Random(42);
        for (int i = 0; i < 10000; i++)
            input.AsWritableSpan()[i] = (float)(rng.NextDouble() * 10 - 5);

        var qparams = QuantizationParams.FromTensor(input);
        var quantized = Quantization.QuantizeInt8(input, qparams);
        var dequantized = Quantization.DequantizeInt8(quantized, qparams);

        double mse = Quantization.ComputeQuantizationError(input, dequantized);
        // INT8 with 256 levels over range [-5,5] should have error < (10/256)^2 / 12 ≈ 0.0013
        Assert.True(mse < 0.01, $"Quantization MSE {mse} exceeds INT8 theoretical limit");
    }
}
