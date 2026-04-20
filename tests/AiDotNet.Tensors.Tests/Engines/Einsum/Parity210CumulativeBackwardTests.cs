using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210CumulativeBackwardTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float tol = 1e-3f) => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    [Fact]
    public void CumProd_Backward_MatchesFiniteDifference()
    {
        // x = (2, 3, 4); y = (2, 6, 24). L = Σy = 32. dL/dy = (1, 1, 1).
        // dL/dx_1 = Σ_{i≥1} y_i/x_1 = 2/2 + 6/2 + 24/2 = 1 + 3 + 12 = 16.
        // dL/dx_2 = 6/3 + 24/3 = 2 + 8 = 10.
        // dL/dx_3 = 24/4 = 6.
        var x = T(new[] { 2f, 3f, 4f }, 3);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorCumProd(x, 0);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        var gx = grads[x];
        Assert.True(Close(16f, gx[0]));
        Assert.True(Close(10f, gx[1]));
        Assert.True(Close(6f, gx[2]));
    }

    [Fact]
    public void CumMax_Backward_GradientFlowsToArgMax()
    {
        // x = (3, 1, 5, 2); y = (3, 3, 5, 5). Running argmax: (0, 0, 2, 2).
        // dL/dy = (1, 1, 1, 1). dL/dx_0 gets contributions from y_0,y_1: 2.
        // dL/dx_2 gets y_2,y_3: 2. x_1 and x_3: 0.
        var x = T(new[] { 3f, 1f, 5f, 2f }, 4);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorCumMax(x, 0);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        var gx = grads[x];
        Assert.Equal(2f, gx[0]);
        Assert.Equal(0f, gx[1]);
        Assert.Equal(2f, gx[2]);
        Assert.Equal(0f, gx[3]);
    }

    [Fact]
    public void CumMin_Backward_GradientFlowsToArgMin()
    {
        var x = T(new[] { 5f, 2f, 7f, 1f }, 4);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorCumMin(x, 0);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        var gx = grads[x];
        // Running argmin: (0, 1, 1, 3). x_0 gets y_0 only = 1. x_1 gets y_1,y_2 = 2.
        // x_3 gets y_3 = 1. x_2 gets 0.
        Assert.Equal(1f, gx[0]);
        Assert.Equal(2f, gx[1]);
        Assert.Equal(0f, gx[2]);
        Assert.Equal(1f, gx[3]);
    }

    [Fact]
    public void LogCumSumExp_Backward_SumsToOneAcrossOutputSlots()
    {
        // For y_i = log Σ_{j≤i} e^{x_j} and L = Σy, we have
        // Σ_k dL/dx_k = Σ_i Σ_{k≤i} exp(x_k - y_i) = Σ_i 1 = N.
        var x = T(new[] { 1f, 2f, 3f }, 3);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorLogCumSumExp(x, 0);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        var gx = grads[x];
        float total = 0;
        for (int i = 0; i < 3; i++) total += gx[i];
        Assert.True(Close(3f, total), $"expected sum=3, got {total}");
    }
}
