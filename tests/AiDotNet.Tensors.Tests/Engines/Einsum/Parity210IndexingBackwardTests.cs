using System.Linq;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210IndexingBackwardTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);
    private static Tensor<Bit> M(bool[] data, params int[] shape) =>
        new Tensor<Bit>(data.Select(b => b ? Bit.True : Bit.False).ToArray(), shape);

    [Fact]
    public void IndexAdd_Backward_InputGradFlowsThroughUntouched()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var idx = I(new[] { 0 }, 1);
        var src = T(new[] { 10f }, 1);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorIndexAdd(x, 0, idx, src);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        // result = input + scatter(source); d(result)/d(input) = 1 everywhere.
        Assert.Equal(new[] { 1f, 1f, 1f }, grads[x].AsSpan().ToArray());
    }

    [Fact]
    public void IndexCopy_Backward_ZerosAtCopiedPositions()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 4);
        var idx = I(new[] { 1 }, 1);
        var src = T(new[] { 99f }, 1);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorIndexCopy(x, 0, idx, src);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        // Position 1 was overwritten → grad 0 there; others pass through.
        Assert.Equal(new[] { 1f, 0f, 1f, 1f }, grads[x].AsSpan().ToArray());
    }

    [Fact]
    public void IndexFill_Backward_ZerosAtFilledPositions()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var idx = I(new[] { 0, 2 }, 2);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorIndexFill(x, 0, idx, -1f);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        Assert.Equal(new[] { 0f, 1f, 0f }, grads[x].AsSpan().ToArray());
    }

    [Fact]
    public void TakeAlongDim_Backward_ScattersGradAlongDim()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        // Pick column 2 for row 0, column 0 for row 1.
        var idx = I(new[] { 2, 0 }, 2, 1);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorTakeAlongDim(x, idx, dim: 1);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        // Gradients land at x[0,2] and x[1,0]; rest zero.
        var gx = grads[x];
        Assert.Equal(0f, gx[0, 0]); Assert.Equal(0f, gx[0, 1]); Assert.Equal(1f, gx[0, 2]);
        Assert.Equal(1f, gx[1, 0]); Assert.Equal(0f, gx[1, 1]); Assert.Equal(0f, gx[1, 2]);
    }

    [Fact]
    public void MaskedScatter_Backward_ZerosAtScatteredPositions()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 4);
        var mask = M(new[] { false, true, false, true }, 4);
        var src = T(new[] { 99f, 100f }, 2);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorMaskedScatter(x, mask, src);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        // Masked positions were overwritten → grad 0; unmasked → grad 1.
        Assert.Equal(new[] { 1f, 0f, 1f, 0f }, grads[x].AsSpan().ToArray());
    }
}
