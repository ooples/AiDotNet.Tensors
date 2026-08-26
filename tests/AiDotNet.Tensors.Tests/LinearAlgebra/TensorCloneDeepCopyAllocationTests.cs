// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// <see cref="TensorBase{T}.CloneDeepCopy"/> must copy a tensor's bytes ONCE.
/// </summary>
/// <remarks>
/// The non-fast path used to call <see cref="TensorBase{T}.ToArray"/> and copy out of the result,
/// which allocates a second full-length array purely as a staging buffer. That is not an exotic
/// path: <see cref="Tensor{T}.CloneShared"/> refuses to share any tensor whose backing storage is
/// longer than its logical length, and pooled tensors (TensorAllocator.Rent) are exactly that, so
/// pooled weights fell through to the deep copy and then paid 2x their own size to be cloned.
///
/// Measured on a 41.9M-parameter VAE decoder in AiDotNet: cloning it allocated 2,231MB of
/// System.Double[] against 320MB of actual weights, and that transient spike is what pushed the
/// 339-layer clone sweep into OutOfMemoryException.
/// </remarks>
public class TensorCloneDeepCopyAllocationTests
{
    private const int Rows = 4;
    private const int Cols = 256 * 1024;   // 1M floats total => 4MB, well clear of measurement noise

    private static Tensor<float> MakeLarge()
    {
        var data = new float[Rows * Cols];
        for (int i = 0; i < data.Length; i++) data[i] = i % 1000;
        return new Tensor<float>(data, new[] { Rows, Cols });
    }

#if NET6_0_OR_GREATER
    // GC.GetTotalAllocatedBytes is not available on net471, and this is the one assertion that
    // needs it; the correctness tests below run on every target.
    [Fact]
    public void CloneDeepCopy_OfAnOffsetView_AllocatesTheTensorOnce()
    {
        var view = MakeLarge().Slice(1);
        long bytes = view.Length * sizeof(float);

        // Warm up so JIT and any first-touch pooling are not counted.
        view.CloneDeepCopy();

        // THREAD-LOCAL, not GC.GetTotalAllocatedBytes. That counter is process-wide, so with xunit
        // running classes in parallel this measurement picked up other tests' allocations and read
        // 2.60x for a copy that is genuinely 1x when run alone -- a flaky assertion, and a false
        // accusation against this code path.
        long before = GC.GetAllocatedBytesForCurrentThread();
        var clone = view.CloneDeepCopy();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.NotNull(clone);

        // One buffer, plus small object overhead. The staging array made this ~2x.
        Assert.True(
            allocated < bytes * 3 / 2,
            $"CloneDeepCopy allocated {allocated:N0} bytes for a {bytes:N0}-byte tensor "
                + $"({allocated / (double)bytes:N2}x). Copying once should stay near 1x; ~2x means "
                + "the copy is still staging through a full intermediate array.");
    }
#endif

    [Fact]
    public void CloneDeepCopy_OfAnOffsetView_ReproducesEveryValue()
    {
        var view = MakeLarge().Slice(2);
        var clone = view.CloneDeepCopy();

        Assert.Equal(view.Length, clone.Length);
        Assert.Equal(view.Shape, clone.Shape);

        var expected = view.ToArray();
        var actual = clone.ToArray();
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
            {
                Assert.Fail($"element {i} differs: expected {expected[i]}, got {actual[i]}");
            }
        }
    }

    [Fact]
    public void CloneDeepCopy_OfAnOffsetView_IsIndependentOfTheSource()
    {
        var source = MakeLarge();
        var view = source.Slice(1);
        var clone = (Tensor<float>)view.CloneDeepCopy();

        var originalFirst = clone.GetFlat(0);
        view.SetFlat(0, originalFirst + 500f);

        Assert.Equal(originalFirst, clone.GetFlat(0));
    }

    [Fact]
    public void CloneDeepCopy_OfAStridedView_ReproducesEveryValue()
    {
        // A transpose is the non-contiguous case, which walks FlatIndexToStorageIndex.
        var source = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var strided = source.Transpose(new[] { 1, 0 });

        var clone = strided.CloneDeepCopy();

        Assert.Equal(strided.Shape, clone.Shape);
        Assert.Equal(strided.ToArray(), clone.ToArray());
    }
}
