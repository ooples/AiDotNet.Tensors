using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210TakeBackwardTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);

    [Fact]
    public void Take_Backward_ScattersGradToSelectedPositions()
    {
        // x = [10, 20, 30, 40, 50]; indices = [0, 2, 4]; L = Σ take = 10+30+50.
        // dL/dx = 1 at positions 0, 2, 4; 0 elsewhere.
        var x = T(new[] { 10f, 20f, 30f, 40f, 50f }, 5);
        var idx = I(new[] { 0, 2, 4 }, 3);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorTake(x, idx);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        var gx = grads[x];
        Assert.Equal(new[] { 1f, 0f, 1f, 0f, 1f }, gx.AsSpan().ToArray());
    }

    [Fact]
    public void Take_Backward_DuplicateIndices_Accumulate()
    {
        // indices hit position 0 twice → grad at 0 = 2.
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var idx = I(new[] { 0, 0, 1 }, 3);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorTake(x, idx);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        var gx = grads[x];
        Assert.Equal(2f, gx[0]);
        Assert.Equal(1f, gx[1]);
        Assert.Equal(0f, gx[2]);
    }

    [Fact]
    public void Take_Backward_2DInput()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var idx = I(new[] { 0, 5 }, 2);   // flat indices: (0,0) and (1,2)
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorTake(x, idx);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        var gx = grads[x];
        Assert.Equal(1f, gx[0, 0]);
        Assert.Equal(1f, gx[1, 2]);
        Assert.Equal(0f, gx[0, 1]);  // not indexed
    }
}
