using System.Linq;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class MaskedSelectTests
{
    private static Tensor<float> Seq(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        int total = 1; foreach (var d in shape) total *= d;
        var idx = new int[shape.Length];
        for (int i = 0; i < total; i++)
        {
            t[idx] = i + 1;
            int k = shape.Length - 1;
            while (k >= 0)
            {
                idx[k]++;
                if (idx[k] < shape[k]) break;
                idx[k] = 0;
                k--;
            }
        }
        return t;
    }

    private static Tensor<Bit> MaskFromBools(bool[] data, int[] shape)
        => new Tensor<Bit>(data.Select(b => b ? Bit.True : Bit.False).ToArray(), shape);

    [Fact]
    public void Select_AllTrue_ReturnsFlattenedInputOrder()
    {
        var engine = new CpuEngine();
        var x = Seq(2, 3);
        var mask = MaskFromBools(Enumerable.Repeat(true, 6).ToArray(), new[] { 2, 3 });

        var r = engine.TensorMaskedSelect(x, mask);
        Assert.Equal(new[] { 6 }, r.Shape.ToArray());
        for (int i = 0; i < 6; i++) Assert.Equal(i + 1f, r[i]);
    }

    [Fact]
    public void Select_AllFalse_ReturnsEmpty()
    {
        var engine = new CpuEngine();
        var x = Seq(3);
        var mask = MaskFromBools(new[] { false, false, false }, new[] { 3 });
        var r = engine.TensorMaskedSelect(x, mask);
        Assert.Equal(new[] { 0 }, r.Shape.ToArray());
    }

    [Fact]
    public void Select_PartialMask_PicksMatchingElements()
    {
        var engine = new CpuEngine();
        var x = Seq(2, 3);
        // row 0: [T, F, T]; row 1: [F, T, F]. Selected: x[0,0], x[0,2], x[1,1].
        var mask = MaskFromBools(new[] { true, false, true, false, true, false }, new[] { 2, 3 });

        var r = engine.TensorMaskedSelect(x, mask);
        Assert.Equal(new[] { 3 }, r.Shape.ToArray());
        Assert.Equal(x[0, 0], r[0]);
        Assert.Equal(x[0, 2], r[1]);
        Assert.Equal(x[1, 1], r[2]);
    }

    [Fact]
    public void Select_ShapeMismatch_Throws()
    {
        var engine = new CpuEngine();
        var x = Seq(2, 3);
        var mask = MaskFromBools(new[] { true, false, true, false, true, false }, new[] { 3, 2 });
        Assert.Throws<System.ArgumentException>(() => engine.TensorMaskedSelect(x, mask));
    }

    [Fact]
    public void Backward_ScattersGradientToMaskedPositions()
    {
        var engine = new CpuEngine();
        var x = Seq(2, 3);
        // Mask pattern: (0,1) and (1,2) are true.
        var maskArr = new[] { false, true, false, false, false, true };
        var mask = MaskFromBools(maskArr, new[] { 2, 3 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var selected = engine.TensorMaskedSelect(x, mask);
        // loss = sum(selected) — gradient at masked positions is 1, elsewhere 0.
        var loss = engine.ReduceSum(selected, null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        var gx = grads[x];

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
            {
                float expected = maskArr[i * 3 + j] ? 1f : 0f;
                Assert.Equal(expected, gx[i, j]);
            }
    }
}
