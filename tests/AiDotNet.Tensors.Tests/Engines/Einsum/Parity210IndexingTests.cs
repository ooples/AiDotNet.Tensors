using System.Linq;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210IndexingTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);
    private static Tensor<Bit> M(bool[] data, params int[] shape) =>
        new Tensor<Bit>(data.Select(b => b ? Bit.True : Bit.False).ToArray(), shape);

    // --- IndexAdd -----------------------------------------------------

    [Fact]
    public void IndexAdd_1D_AddsAtIndices()
    {
        var x = T(new[] { 10f, 20f, 30f, 40f }, 4);
        var idx = I(new[] { 0, 2 }, 2);
        var src = T(new[] { 1f, 3f }, 2);
        var r = E.TensorIndexAdd(x, 0, idx, src);
        Assert.Equal(new[] { 11f, 20f, 33f, 40f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void IndexAdd_DuplicateIndices_Accumulate()
    {
        var x = T(new[] { 10f, 20f, 30f }, 3);
        var idx = I(new[] { 1, 1, 0 }, 3);
        var src = T(new[] { 5f, 7f, 100f }, 3);
        var r = E.TensorIndexAdd(x, 0, idx, src);
        // position 1 gets +5 +7 = +12; position 0 gets +100.
        Assert.Equal(new[] { 110f, 32f, 30f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void IndexAdd_2D_Axis0()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var idx = I(new[] { 0 }, 1);
        var src = T(new[] { 10f, 20f }, 1, 2);
        var r = E.TensorIndexAdd(x, 0, idx, src);
        // row 0: [1+10, 2+20] = [11, 22]; row 1 unchanged.
        Assert.Equal(new[] { 11f, 22f, 3f, 4f }, r.AsSpan().ToArray());
    }

    // --- IndexFill ----------------------------------------------------

    [Fact]
    public void IndexFill_1D_SetsPositions()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 4);
        var idx = I(new[] { 0, 3 }, 2);
        var r = E.TensorIndexFill(x, 0, idx, -1f);
        Assert.Equal(new[] { -1f, 2f, 3f, -1f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void IndexFill_2D_Axis1_SetsColumns()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var idx = I(new[] { 1 }, 1);
        var r = E.TensorIndexFill(x, 1, idx, 0f);
        // column 1 zeroed in both rows.
        Assert.Equal(new[] { 1f, 0f, 3f, 4f, 0f, 6f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void IndexFill_OutOfRange_Throws()
    {
        var x = T(new[] { 1f, 2f }, 2);
        var idx = I(new[] { 5 }, 1);
        Assert.Throws<System.IndexOutOfRangeException>(
            () => E.TensorIndexFill(x, 0, idx, 0f));
    }

    // --- MaskedScatter -----------------------------------------------

    [Fact]
    public void MaskedScatter_FillsMaskedPositions()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f }, 5);
        var mask = M(new[] { false, true, false, true, false }, 5);
        var src = T(new[] { 100f, 200f }, 2);
        var r = E.TensorMaskedScatter(x, mask, src);
        // positions 1 and 3 filled with src[0] and src[1].
        Assert.Equal(new[] { 1f, 100f, 3f, 200f, 5f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void MaskedScatter_AllFalse_IsIdentity()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var mask = M(new[] { false, false, false }, 3);
        var src = T(new[] { 99f }, 1);
        var r = E.TensorMaskedScatter(x, mask, src);
        Assert.Equal(x.AsSpan().ToArray(), r.AsSpan().ToArray());
    }

    [Fact]
    public void MaskedScatter_SourceTooSmall_Throws()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var mask = M(new[] { true, true, true }, 3);
        var src = T(new[] { 99f }, 1);
        Assert.Throws<System.ArgumentException>(
            () => E.TensorMaskedScatter(x, mask, src));
    }
}
