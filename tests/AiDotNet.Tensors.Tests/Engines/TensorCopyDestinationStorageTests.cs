using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// TensorCopy must write into the destination's live storage, whatever that storage looks like.
/// </summary>
/// <remarks>
/// It used to write into <c>destination.GetDataArray()</c>, which returns the backing array only when it is
/// exactly the tensor (offset 0, storage length equal to Length). A pooled tensor whose rented array is longer,
/// or a view at a non-zero offset, got a fresh copy instead, and the whole write was silently discarded. That is
/// how AiDotNet's FTRL and ASGD optimizers, which assign their result to a parameter with TensorCopy, never
/// updated pooled model parameters at all.
/// </remarks>
public class TensorCopyDestinationStorageTests
{
    private static Tensor<float> Filled(int[] shape, float start)
    {
        var t = new Tensor<float>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = start + i;
        return t;
    }

    [Fact]
    public void TensorCopy_IntoARowViewAtANonZeroOffset_WritesTheSharedStorage()
    {
        var engine = new CpuEngine();
        var matrix = Filled(new[] { 3, 4 }, 100f);
        var row = matrix.Slice(1);                       // a view: elements 4..7 of the matrix's storage
        Assert.Equal(4, row.Length);

        var source = Filled(new[] { 4 }, 1f);
        long versionBefore = row.Version;
        engine.TensorCopy(source, row);

        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, row.AsSpan().ToArray());
        Assert.Equal(new[] { 100f, 101f, 102f, 103f, 1f, 2f, 3f, 4f, 108f, 109f, 110f, 111f }, matrix.AsSpan().ToArray());
        Assert.True(row.Version > versionBefore, "a write through TensorCopy must bump the destination's version");
    }
}
