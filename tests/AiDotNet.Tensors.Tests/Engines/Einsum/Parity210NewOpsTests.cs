using System.Linq;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210NewOpsTests
{
    private static CpuEngine E => new CpuEngine();

    private static Tensor<float> Arr(float[] data, int[] shape) => new Tensor<float>(data, shape);

    // --- Roll ---------------------------------------------------------

    [Fact]
    public void Roll_Shift1_Axis0_WrapsLastRowToFront()
    {
        var x = Arr(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });
        var r = E.TensorRoll(x, new[] { 1 }, new[] { 0 });
        // Rolled by 1 on axis 0 → row 2 moves to front.
        Assert.Equal(new float[] { 5, 6, 1, 2, 3, 4 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Roll_NegativeShift_WrapsForward()
    {
        var x = Arr(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var r = E.TensorRoll(x, new[] { -1 }, new[] { 0 });
        Assert.Equal(new float[] { 2, 3, 4, 1 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Roll_Backward_Undoes()
    {
        var x = Arr(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var r = E.TensorRoll(x, new[] { 2 }, new[] { 0 });
        var loss = E.ReduceSum(r, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        // Sum gradient is 1 at every position of the input.
        Assert.Equal(new float[] { 1, 1, 1, 1 }, grads[x].AsSpan().ToArray());
    }

    // --- Flip ---------------------------------------------------------

    [Fact]
    public void Flip_Axis0_Reverses()
    {
        var x = Arr(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });
        var r = E.TensorFlip(x, new[] { 0 });
        Assert.Equal(new float[] { 5, 6, 3, 4, 1, 2 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Flip_TwoAxes_ReversesBoth()
    {
        var x = Arr(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var r = E.TensorFlip(x, new[] { 0, 1 });
        Assert.Equal(new float[] { 4, 3, 2, 1 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Flip_IsInvolution()
    {
        var x = Arr(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var r = E.TensorFlip(E.TensorFlip(x, new[] { 0, 1 }), new[] { 0, 1 });
        Assert.Equal(x.AsSpan().ToArray(), r.AsSpan().ToArray());
    }

    // --- RepeatInterleave --------------------------------------------

    [Fact]
    public void RepeatInterleave_Scalar_DuplicatesAlongDim()
    {
        var x = Arr(new float[] { 1, 2, 3 }, new[] { 3 });
        var r = E.TensorRepeatInterleave(x, 2, 0);
        Assert.Equal(new[] { 6 }, r.Shape.ToArray());
        Assert.Equal(new float[] { 1, 1, 2, 2, 3, 3 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void RepeatInterleave_2D_AlongAxis1()
    {
        var x = Arr(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var r = E.TensorRepeatInterleave(x, 3, 1);
        Assert.Equal(new[] { 2, 6 }, r.Shape.ToArray());
        Assert.Equal(new float[] { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void RepeatInterleave_Backward_SumsChunks()
    {
        var x = Arr(new float[] { 1, 2, 3 }, new[] { 3 });
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var r = E.TensorRepeatInterleave(x, 2, 0);
        var loss = E.ReduceSum(r, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        // Each source element contributed to 2 output positions.
        Assert.Equal(new float[] { 2, 2, 2 }, grads[x].AsSpan().ToArray());
    }

    // --- Cumulative --------------------------------------------------

    [Fact]
    public void CumProd_1D()
    {
        var x = Arr(new float[] { 2, 3, 4 }, new[] { 3 });
        var r = E.TensorCumProd(x, 0);
        Assert.Equal(new float[] { 2, 6, 24 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void CumMax_1D()
    {
        var x = Arr(new float[] { 3, 1, 5, 2, 4 }, new[] { 5 });
        var r = E.TensorCumMax(x, 0);
        Assert.Equal(new float[] { 3, 3, 5, 5, 5 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void CumMin_1D()
    {
        var x = Arr(new float[] { 3, 1, 5, 2, 4 }, new[] { 5 });
        var r = E.TensorCumMin(x, 0);
        Assert.Equal(new float[] { 3, 1, 1, 1, 1 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void LogCumSumExp_EquivalentToScanLogSumExp()
    {
        var x = Arr(new float[] { 1f, 2f, 3f }, new[] { 3 });
        var r = E.TensorLogCumSumExp(x, 0);
        // Reference: r[i] = log(sum_{j<=i} exp(x[j]))
        var expected = new float[3];
        expected[0] = System.MathF.Log(System.MathF.Exp(1f));
        expected[1] = System.MathF.Log(System.MathF.Exp(1f) + System.MathF.Exp(2f));
        expected[2] = System.MathF.Log(System.MathF.Exp(1f) + System.MathF.Exp(2f) + System.MathF.Exp(3f));
        var actual = r.AsSpan().ToArray();
        for (int i = 0; i < 3; i++)
            Assert.True(System.MathF.Abs(expected[i] - actual[i]) < 1e-5, $"[{i}]: {expected[i]} vs {actual[i]}");
    }

    // --- IsClose / AllClose / IsIn -----------------------------------

    [Fact]
    public void IsClose_ExactEqual_True()
    {
        var a = Arr(new float[] { 1, 2, 3 }, new[] { 3 });
        var b = Arr(new float[] { 1, 2, 3 }, new[] { 3 });
        var r = E.TensorIsClose(a, b, 1e-5f, 1e-8f);
        foreach (var bit in r.AsSpan().ToArray()) Assert.True((bool)bit);
    }

    [Fact]
    public void IsClose_WithinTolerance_True()
    {
        var a = Arr(new float[] { 1.00001f }, new[] { 1 });
        var b = Arr(new float[] { 1.00000f }, new[] { 1 });
        var r = E.TensorIsClose(a, b, 1e-4f, 1e-8f);
        Assert.True((bool)r.AsSpan()[0]);
    }

    [Fact]
    public void IsClose_OutsideTolerance_False()
    {
        var a = Arr(new float[] { 1f }, new[] { 1 });
        var b = Arr(new float[] { 2f }, new[] { 1 });
        var r = E.TensorIsClose(a, b, 1e-5f, 1e-8f);
        Assert.False((bool)r.AsSpan()[0]);
    }

    [Fact]
    public void AllClose_ReducesOverTensor()
    {
        var a = Arr(new float[] { 1, 2, 3 }, new[] { 3 });
        var b = Arr(new float[] { 1, 2.00001f, 3 }, new[] { 3 });
        Assert.True(E.TensorAllClose(a, b, 1e-4f, 1e-8f));
        Assert.False(E.TensorAllClose(a, b, 1e-8f, 1e-10f));
    }

    [Fact]
    public void IsIn_FindsMembers()
    {
        var xs = Arr(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var test = Arr(new float[] { 2, 4, 6 }, new[] { 3 });
        var r = E.TensorIsIn(xs, test);
        Assert.Equal(new[] { false, true, false, true }, r.AsSpan().ToArray().Select(b => (bool)b));
    }

    [Fact]
    public void IsIn_Invert_Negates()
    {
        var xs = Arr(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var test = Arr(new float[] { 2, 4 }, new[] { 2 });
        var r = E.TensorIsIn(xs, test, invert: true);
        Assert.Equal(new[] { true, false, true, false }, r.AsSpan().ToArray().Select(b => (bool)b));
    }

    // --- Clamp family ------------------------------------------------

    [Fact]
    public void ClampMin_RaisesBelowMin()
    {
        var x = Arr(new float[] { -1, 0, 1, 2 }, new[] { 4 });
        var r = E.TensorClampMin(x, 0f);
        Assert.Equal(new float[] { 0, 0, 1, 2 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ClampMax_LowersAboveMax()
    {
        var x = Arr(new float[] { -1, 0, 1, 2 }, new[] { 4 });
        var r = E.TensorClampMax(x, 1f);
        Assert.Equal(new float[] { -1, 0, 1, 1 }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ClampMin_Backward_GradientPassesWhereInRange()
    {
        var x = Arr(new float[] { -1, 0, 1 }, new[] { 3 });
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var r = E.TensorClampMin(x, 0f);
        var loss = E.ReduceSum(r, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        // x=-1 is below min → grad 0; x=0 and x=1 are in-range → grad 1.
        Assert.Equal(new float[] { 0, 1, 1 }, grads[x].AsSpan().ToArray());
    }

    [Fact]
    public void ClampMax_Backward_GradientPassesWhereInRange()
    {
        var x = Arr(new float[] { -1, 0, 1 }, new[] { 3 });
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var r = E.TensorClampMax(x, 0f);
        var loss = E.ReduceSum(r, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        Assert.Equal(new float[] { 1, 1, 0 }, grads[x].AsSpan().ToArray());
    }

    [Fact]
    public void Aminmax_ReturnsSinglePassMinMax()
    {
        var x = Arr(new float[] { 3, -1, 5, 2, -4, 7 }, new[] { 6 });
        var (min, max) = E.TensorAminmax(x);
        Assert.Equal(-4f, min);
        Assert.Equal(7f, max);
    }
}
