using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

public class Parity210PackedGatherScatterTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void GatherPacked_Int4_PreservesStorageFormat()
    {
        // 3 rows × 2 packed bytes (= 4 int4 nibbles per row). Values per
        // byte = 2. Row content represented as (lo-nibble, hi-nibble).
        // Row 0: packed=[(1,2),(3,4)]; Row 1: packed=[(5,6),(7,-8)];
        // Row 2: packed=[(-1,-2),(-3,0)].
        var packed = new Tensor<byte>(new byte[]
        {
            PackedInt4.FromInts(1, 2).RawValue, PackedInt4.FromInts(3, 4).RawValue,
            PackedInt4.FromInts(5, 6).RawValue, PackedInt4.FromInts(7, -8).RawValue,
            PackedInt4.FromInts(-1, -2).RawValue, PackedInt4.FromInts(-3, 0).RawValue,
        }, new[] { 3, 2 });

        var idx = new Tensor<int>(new[] { 2, 0 }, new[] { 2 });
        var gathered = E.TensorGatherPacked(packed, idx, axis: 0, valuesPerByte: 2);

        Assert.Equal(new[] { 2, 2 }, gathered.Shape.ToArray());
        // Row 0 of output = source row 2 = PackedInt4(-1,-2), PackedInt4(-3,0).
        var lo0 = new PackedInt4(gathered[0, 0]).LoNibble;
        var hi0 = new PackedInt4(gathered[0, 0]).HiNibble;
        var lo1 = new PackedInt4(gathered[0, 1]).LoNibble;
        var hi1 = new PackedInt4(gathered[0, 1]).HiNibble;
        Assert.Equal(-1, lo0); Assert.Equal(-2, hi0);
        Assert.Equal(-3, lo1); Assert.Equal(0, hi1);
        // Row 1 = source row 0.
        Assert.Equal(1, new PackedInt4(gathered[1, 0]).LoNibble);
        Assert.Equal(4, new PackedInt4(gathered[1, 1]).HiNibble);
    }

    [Fact]
    public void ScatterPacked_Int4_OverwritesAtIndexedRows()
    {
        var packed = new Tensor<byte>(new byte[]
        {
            PackedInt4.FromInts(1, 1).RawValue, PackedInt4.FromInts(1, 1).RawValue,
            PackedInt4.FromInts(2, 2).RawValue, PackedInt4.FromInts(2, 2).RawValue,
            PackedInt4.FromInts(3, 3).RawValue, PackedInt4.FromInts(3, 3).RawValue,
        }, new[] { 3, 2 });
        var source = new Tensor<byte>(new byte[]
        {
            PackedInt4.FromInts(7, 7).RawValue, PackedInt4.FromInts(7, 7).RawValue,
        }, new[] { 1, 2 });
        var idx = new Tensor<int>(new[] { 1 }, new[] { 1 });

        var r = E.TensorScatterPacked(packed, idx, source, axis: 0, valuesPerByte: 2);
        Assert.Equal(1, new PackedInt4(r[0, 0]).LoNibble);
        Assert.Equal(7, new PackedInt4(r[1, 0]).LoNibble);
        Assert.Equal(3, new PackedInt4(r[2, 0]).LoNibble);
    }

    [Fact]
    public void GatherPacked_Int1_BitLevelStorageWidth()
    {
        // Each byte holds 8 bits.
        var packed = new Tensor<byte>(new byte[] { 0xFF, 0x00, 0xAA, 0x55 }, new[] { 4 });
        // Can't gather on the last axis with valuesPerByte > 1 — use a
        // shape that puts packing on the leading dim instead.
        // Actually this test's axis=0 is the packing axis and that's the
        // last dim too, so it's invalid. Let me reshape into [4, 1] to
        // make axis 0 a non-last axis.
        var packed2d = packed.Reshape(new[] { 4, 1 });
        var idx = new Tensor<int>(new[] { 0, 2 }, new[] { 2 });
        var r = E.TensorGatherPacked(packed2d, idx, axis: 0, valuesPerByte: 8);
        Assert.Equal(0xFF, r[0, 0]);
        Assert.Equal(0xAA, r[1, 0]);
    }

    [Fact]
    public void GatherPacked_InvalidValuesPerByte_Throws()
    {
        var packed = new Tensor<byte>(new byte[] { 0 }, new[] { 1, 1 });
        var idx = new Tensor<int>(new[] { 0 }, new[] { 1 });
        Assert.Throws<System.ArgumentOutOfRangeException>(
            () => E.TensorGatherPacked(packed, idx, axis: 0, valuesPerByte: 3));
    }

    [Fact]
    public void GatherPacked_AxisOnPackingBoundary_Throws()
    {
        // Packed on last axis, try to gather on last axis → error.
        var packed = new Tensor<byte>(new byte[] { 0, 0, 0, 0 }, new[] { 2, 2 });
        var idx = new Tensor<int>(new[] { 0 }, new[] { 1 });
        Assert.Throws<System.ArgumentException>(
            () => E.TensorGatherPacked(packed, idx, axis: 1, valuesPerByte: 2));
    }
}
