using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Strided views are materialized run by run (axes coalesced, innermost run block-copied or strided-looped) by
/// Contiguous(), CopyTo(Span) and CopyLogicalTo. Each must equal an independent per-element walk of the view bit
/// for bit, for transposes, permutes, offset slices, size-1 axes, zero (broadcast) strides and negative strides.
/// </summary>
public class StridedCopyRunTests
{
    private static float[] Source(int length)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = i * 0.25f - 3.5f;
        return data;
    }

    private static float[] Reference(float[] source, int offset, int[] shape, int[] strides)
    {
        int total = 1;
        foreach (var e in shape) total *= e;
        var result = new float[total];
        for (int flat = 0; flat < total; flat++)
        {
            int remaining = flat, src = offset;
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                int idx = remaining % shape[d];
                remaining /= shape[d];
                src += idx * strides[d];
            }
            result[flat] = source[src];
        }
        return result;
    }

    private static int Bits(float v) => System.BitConverter.ToInt32(System.BitConverter.GetBytes(v), 0);

    private static void AssertBitEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (Bits(expected[i]) != Bits(actual[i]))
                Assert.Fail($"{what}: element {i} expected {expected[i]:R} got {actual[i]:R}");
    }

    public static TheoryData<int, int, int[], int[]> Views => new()
    {
        // sourceLength, offset, shape, strides
        { 12, 0, new[] { 3, 4 }, new[] { 1, 3 } },                       // 2-D transpose
        { 64, 5, new[] { 4, 7 }, new[] { 8, 1 } },                       // offset row slice (padded rows)
        { 120, 0, new[] { 2, 3, 4, 5 }, new[] { 60, 5, 15, 1 } },        // permute (0,2,1,3): last axis contiguous
        { 120, 0, new[] { 5, 4, 3, 2 }, new[] { 1, 5, 20, 60 } },        // full reverse permute
        { 120, 0, new[] { 2, 1, 3, 1, 20 }, new[] { 60, 999, 20, 999, 1 } }, // size-1 axes with junk strides
        { 60, 0, new[] { 3, 4, 5 }, new[] { 20, 5, 1 } },                // already row-major (all merge)
        { 20, 0, new[] { 3, 4, 5 }, new[] { 0, 5, 1 } },                 // broadcast leading axis
        { 20, 0, new[] { 4, 3 }, new[] { 1, 0 } },                       // broadcast inner axis
        { 30, 29, new[] { 5, 6 }, new[] { -6, -1 } },                    // negative strides (flipped)
        { 10, 0, new[] { 3, 4 }, new[] { 1, 1 } },                       // overlapping sliding window (no merge)
        { 40, 1, new[] { 2, 3, 4 }, new[] { 12, 4, 4 } },                // outer stride equals inner stride
        { 4096, 3, new[] { 8, 2, 16, 4 }, new[] { 128, 64, 1, 16 } },    // attention-like head permute
        { 1, 0, new int[0], new int[0] },                                // rank 0
    };

    [Theory]
    [MemberData(nameof(Views))]
    public void CopyStridedToRowMajor_MatchesElementWalk(int sourceLength, int offset, int[] shape, int[] strides)
    {
        var source = Source(sourceLength);
        int total = 1;
        foreach (var e in shape) total *= e;
        var destination = new float[total];
        TensorBase<float>.CopyStridedToRowMajor(source, offset, shape, strides, destination);
        AssertBitEqual(Reference(source, offset, shape, strides), destination, "strided copy");
    }

    [Fact]
    public void Contiguous_CopyTo_And_CopyLogicalTo_OfPermutedView_MatchIndexer()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 3, 4, 5, 6 });
        for (int i = 0; i < x.Length; i++) x[i] = i * 0.5f - 11f;
        var view = x.Transpose(new[] { 2, 0, 3, 1 });   // strided view, no copy
        Assert.False(view.IsContiguous);

        var expected = new float[view.Length];
        int flat = 0;
        for (int a = 0; a < 5; a++)
            for (int b = 0; b < 3; b++)
                for (int c = 0; c < 6; c++)
                    for (int d = 0; d < 4; d++)
                        expected[flat++] = x[b, d, a, c];

        AssertBitEqual(expected, view.Contiguous().ToArray(), "Contiguous");
        var copied = new float[view.Length];
        view.CopyTo(copied.AsSpan());
        AssertBitEqual(expected, copied, "CopyTo");
        var logical = new float[view.Length];
        view.CopyLogicalTo(logical);
        AssertBitEqual(expected, logical, "CopyLogicalTo");
        GC.KeepAlive(engine);
    }
}