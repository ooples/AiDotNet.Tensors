using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Tests for <see cref="Tensor{T}.CopyTo(Span{T})"/> and
/// <see cref="Tensor{T}.CopyTo(Memory{T})"/>.
///
/// CopyTo is the correct-by-default replacement for the
/// <c>Contiguous().AsSpan().CopyTo(dst)</c> idiom — it copies in one pass
/// without allocating an intermediate materialized tensor, and it
/// transparently handles both contiguous and strided (post-permute /
/// post-transpose / post-slice) layouts that <c>AsSpan()</c> rejects.
/// </summary>
public class TensorCopyToTests
{
    [Fact]
    public void Contiguous_Copy_MatchesSource()
    {
        var src = new Tensor<double>([2, 3]);
        for (int i = 0; i < src.Length; i++)
            src.AsWritableSpan()[i] = i + 1.5;

        Span<double> dst = stackalloc double[src.Length];
        src.CopyTo(dst);

        for (int i = 0; i < src.Length; i++)
            Assert.Equal(i + 1.5, dst[i]);
    }

    [Fact]
    public void Contiguous_Copy_ToOversizedDestination_LeavesTrailUntouched()
    {
        var src = new Tensor<double>([4]);
        for (int i = 0; i < 4; i++) src.AsWritableSpan()[i] = i + 10.0;

        Span<double> dst = stackalloc double[8];
        for (int i = 0; i < 8; i++) dst[i] = -1.0;

        src.CopyTo(dst);

        for (int i = 0; i < 4; i++) Assert.Equal(i + 10.0, dst[i]);
        for (int i = 4; i < 8; i++) Assert.Equal(-1.0, dst[i]);
    }

    [Fact]
    public void Contiguous_Copy_ToUndersizedDestination_Throws()
    {
        var src = new Tensor<double>([4]);
        for (int i = 0; i < 4; i++) src.AsWritableSpan()[i] = i;

        var dst = new double[3];
        Assert.Throws<ArgumentException>(() => src.CopyTo(dst.AsSpan()));
    }

    /// <summary>
    /// The motivating regression: a post-permute strided view that would
    /// throw "Cannot get a contiguous span" on <see cref="Tensor{T}.AsSpan"/>
    /// must still be copyable via <see cref="Tensor{T}.CopyTo(Span{T})"/>,
    /// and the resulting flat layout must match what would have been
    /// produced by <c>.Contiguous().AsSpan().CopyTo(dst)</c>.
    /// </summary>
    [Fact]
    public void StridedAfterPermute_Copy_MatchesContiguousThenCopy()
    {
        // Build a CHW tensor like the CIFAR data loaders do.
        var chw = new Tensor<double>([3, 2, 2]);
        double v = 0;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 2; y++)
                for (int x = 0; x < 2; x++)
                    chw[c, y, x] = v++;  // 0..11 in CHW order

        // Permute to HWC: axes [1, 2, 0] means new[0]=old[1] (H),
        // new[1]=old[2] (W), new[2]=old[0] (C).
        var hwc = AiDotNetEngine.Current.TensorPermute(chw, [1, 2, 0]);
        Assert.False(hwc.IsContiguous);  // strided view, guards the regression

        var expected = hwc.Contiguous();
        Span<double> viaCopy = stackalloc double[hwc.Length];
        Span<double> viaContiguous = stackalloc double[hwc.Length];

        hwc.CopyTo(viaCopy);
        expected.AsSpan().CopyTo(viaContiguous);

        for (int i = 0; i < hwc.Length; i++)
            Assert.Equal(viaContiguous[i], viaCopy[i]);

        // Spot-check the semantics: HWC[y,x,c] should equal the original
        // CHW[c,y,x]. This catches any axis-order drift in either method.
        int hwcIdx = 0;
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(chw[c, y, x], viaCopy[hwcIdx++]);
    }

    [Fact]
    public void StridedAfterTranspose_Copy_MatchesLogicalOrder()
    {
        // 2x3 matrix stored row-major as [1,2,3,4,5,6].
        var src = new Tensor<double>([2, 3]);
        for (int i = 0; i < src.Length; i++) src.AsWritableSpan()[i] = i + 1.0;

        // Transpose via Permute to force a strided view (Transpose shortcut
        // might return a materialized result for a 2D matrix on some paths).
        var t = AiDotNetEngine.Current.TensorPermute(src, [1, 0]);
        Assert.False(t.IsContiguous);

        Span<double> dst = stackalloc double[t.Length];
        t.CopyTo(dst);

        // After transpose: [0,0]=1 [0,1]=4 [1,0]=2 [1,1]=5 [2,0]=3 [2,1]=6
        Assert.Equal(new double[] { 1, 4, 2, 5, 3, 6 }, dst.ToArray());
    }

    [Fact]
    public void StridedAfterSlice_Copy_MatchesLogicalOrder()
    {
        // [4] source, slice to the middle two elements via
        // TensorSlice(tensor, starts, lengths).
        var src = new Tensor<double>([4]);
        for (int i = 0; i < 4; i++) src.AsWritableSpan()[i] = i + 100.0;

        var slice = AiDotNetEngine.Current.TensorSlice(src, new[] { 1 }, new[] { 2 });
        Assert.Equal(2, slice.Length);

        Span<double> dst = stackalloc double[slice.Length];
        slice.CopyTo(dst);

        Assert.Equal(101.0, dst[0]);
        Assert.Equal(102.0, dst[1]);
    }

    [Fact]
    public void Memory_Overload_DelegatesToSpan()
    {
        var src = new Tensor<double>([3]);
        for (int i = 0; i < 3; i++) src.AsWritableSpan()[i] = i * 7.0;

        Memory<double> dst = new double[3];
        src.CopyTo(dst);

        Assert.Equal(0.0, dst.Span[0]);
        Assert.Equal(7.0, dst.Span[1]);
        Assert.Equal(14.0, dst.Span[2]);
    }

    [Fact]
    public void FloatTensor_CopyTo_WorksSymmetrically()
    {
        // Confirms the generic dispatch isn't accidentally double-specific.
        var src = new Tensor<float>([2, 3]);
        for (int i = 0; i < src.Length; i++)
            src.AsWritableSpan()[i] = i + 0.25f;

        // Exercise strided path on float as well.
        var t = AiDotNetEngine.Current.TensorPermute(src, [1, 0]);
        Assert.False(t.IsContiguous);

        Span<float> dst = stackalloc float[t.Length];
        t.CopyTo(dst);

        // src row-major: [0.25, 1.25, 2.25, 3.25, 4.25, 5.25]
        // transposed:    [0.25, 3.25, 1.25, 4.25, 2.25, 5.25]
        Assert.Equal(new[] { 0.25f, 3.25f, 1.25f, 4.25f, 2.25f, 5.25f }, dst.ToArray());
    }

    /// <summary>
    /// Zero-length tensor: the logical contents span is empty, so CopyTo
    /// must be a no-op (write nothing, not throw). Guards against a
    /// regression where the outer loop's <c>i &lt; Length</c> precondition
    /// gets rewritten in a way that still touches the destination for
    /// <c>Length == 0</c>.
    /// </summary>
    [Fact]
    public void ZeroLength_Copy_IsNoOp()
    {
        var src = new Tensor<double>([0, 3]);  // shape with a zero dim -> Length = 0
        Assert.Equal(0, src.Length);

        Span<double> dst = stackalloc double[4];
        for (int i = 0; i < 4; i++) dst[i] = 42.0;

        src.CopyTo(dst);  // must not throw and must not touch dst

        for (int i = 0; i < 4; i++) Assert.Equal(42.0, dst[i]);
    }

    /// <summary>
    /// Rank-0 tensor holds exactly one element and has an empty shape /
    /// stride array. Verifies the stride-decomposition loop degenerates
    /// gracefully (inner loop <c>d &lt; 0</c> never executes, srcIdx stays
    /// at the storage offset, single element copied).
    /// </summary>
    [Fact]
    public void Rank0Scalar_Copy_WritesSingleElement()
    {
        var src = new Tensor<double>(new int[] { });
        Assert.Equal(0, src.Rank);
        Assert.Equal(1, src.Length);
        src.AsWritableSpan()[0] = 3.14;

        Span<double> dst = stackalloc double[1];
        src.CopyTo(dst);
        Assert.Equal(3.14, dst[0]);
    }

    /// <summary>
    /// Rank-4 coverage (NCHW mini-batch). The permute path exercises the
    /// strided decomposition across four dimensions, which is the target
    /// shape class for the motivating CIFAR / EuroSat loaders.
    /// </summary>
    [Fact]
    public void Rank4AfterPermute_Copy_MatchesContiguousThenCopy()
    {
        // [N=2, C=3, H=2, W=2] — 24 elements, each distinct.
        var nchw = new Tensor<double>([2, 3, 2, 2]);
        for (int i = 0; i < nchw.Length; i++) nchw.AsWritableSpan()[i] = i;

        // NCHW -> NHWC: axes [0, 2, 3, 1].
        var nhwc = AiDotNetEngine.Current.TensorPermute(nchw, [0, 2, 3, 1]);
        Assert.False(nhwc.IsContiguous);

        Span<double> viaCopy = stackalloc double[nhwc.Length];
        Span<double> viaContiguous = stackalloc double[nhwc.Length];

        nhwc.CopyTo(viaCopy);
        nhwc.Contiguous().AsSpan().CopyTo(viaContiguous);

        // Byte-for-byte parity against the materialize-then-copy idiom.
        for (int i = 0; i < nhwc.Length; i++)
            Assert.Equal(viaContiguous[i], viaCopy[i]);

        // Value-level spot check: NHWC[n,y,x,c] == NCHW[n,c,y,x].
        for (int n = 0; n < 2; n++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < 2; y++)
                    for (int x = 0; x < 2; x++)
                        Assert.Equal(nchw[n, c, y, x], nhwc[n, y, x, c]);
    }

    /// <summary>
    /// Sparse tensors use a fundamentally different storage model
    /// (coordinate / CSR / CSC indexing), so a row-major CopyTo isn't
    /// well-defined. The implementation guards with
    /// <c>ThrowIfSparse()</c>; verify the throw actually surfaces.
    /// </summary>
    [Fact]
    public void SparseTensor_CopyTo_Throws()
    {
        // Minimal 2x2 sparse diagonal.
        var sparse = new SparseTensor<double>(
            rows: 2, columns: 2,
            rowIndices: [0, 1],
            columnIndices: [0, 1],
            values: [1.0, 2.0]);

        var dst = new double[sparse.Length];
        // Base class ThrowIfSparse throws NotSupportedException; accept
        // whichever exception type TensorBase<T>.ThrowIfSparse is configured
        // to raise.
        Assert.ThrowsAny<Exception>(() => sparse.CopyTo(dst.AsSpan()));
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Zero-allocation regression: CopyTo must not materialize an
    /// intermediate contiguous tensor (that's the whole point). Uses the
    /// per-thread allocation counter to verify the fast path allocates
    /// nothing measurable. Only runs on net5+ since
    /// <c>GC.GetAllocatedBytesForCurrentThread</c> is unavailable on
    /// net471.
    /// </summary>
    [Fact]
    public void Contiguous_Copy_DoesNotAllocate()
    {
        var src = new Tensor<double>([1024]);
        for (int i = 0; i < src.Length; i++) src.AsWritableSpan()[i] = i;
        var dst = new double[src.Length];

        // Warm-up to JIT and settle the heap.
        src.CopyTo(dst.AsSpan());
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long before = GC.GetAllocatedBytesForCurrentThread();
        src.CopyTo(dst.AsSpan());
        long after = GC.GetAllocatedBytesForCurrentThread();

        // Allow a tiny slack for instrumentation; the intent is "no new
        // Tensor / array allocation on the fast path."
        Assert.True(after - before < 128, $"Unexpected allocation: {after - before} bytes");
    }
#endif
}
