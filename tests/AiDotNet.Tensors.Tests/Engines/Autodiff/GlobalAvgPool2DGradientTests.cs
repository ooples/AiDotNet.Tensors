using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// GlobalAvgPool2D recorded nothing on the gradient tape, so a network that pooled globally before its head (ResNet)
/// got no gradient for any parameter upstream of the pool.
/// </summary>
public sealed class GlobalAvgPool2DGradientTests
{
    private const int Batch = 2, Channels = 3, Height = 4, Width = 5;
    private const float Tolerance = 1e-6f;

    [Fact]
    public void GlobalAvgPool2D_UnderATape_PassesTheMeanGradientToEveryInputCell()
    {
        var engine = new CpuEngine();
        var rng = new Random(9);
        var data = new float[Batch * Channels * Height * Width];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        var x = new Tensor<float>(data, new[] { Batch, Channels, Height, Width });
        var weights = new float[Batch * Channels];
        for (int i = 0; i < weights.Length; i++) weights[i] = i + 1;
        var w = new Tensor<float>(weights, new[] { Batch, Channels, 1, 1 });

        Tensor<float> dx;
        using (var tape = new GradientTape<float>())
        {
            var pooled = engine.GlobalAvgPool2D(x);
            var loss = engine.ReduceSum(engine.TensorMultiply(pooled, w), null);
            var grads = tape.ComputeGradients(loss, new[] { x });
            Assert.True(grads.ContainsKey(x), "no gradient reached the pool's input");
            dx = grads[x];
        }

        // d/dx of sum_bc w_bc * mean_hw x_bchw is w_bc / (H*W) at every cell.
        for (int b = 0; b < Batch; b++)
            for (int c = 0; c < Channels; c++)
                for (int h = 0; h < Height; h++)
                    for (int col = 0; col < Width; col++)
                    {
                        float expected = weights[b * Channels + c] / (Height * Width);
                        float actual = dx[b, c, h, col];
                        Assert.True(Math.Abs(actual - expected) <= Tolerance, $"dx[{b},{c},{h},{col}] = {actual}, expected {expected}");
                    }
    }
}
