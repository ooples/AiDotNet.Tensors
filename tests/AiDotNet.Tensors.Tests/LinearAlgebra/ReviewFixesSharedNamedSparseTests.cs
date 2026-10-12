using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Review fixes on #1120 for shared memory, named dimensions and the sparse torch ops: shared-memory metadata on
/// storage-sharing views, owner-only files, file removal when the storage is released, size validation before any
/// allocation; names owned by the tensor, AlignTo as a storage-sharing view with the right gradient, AlignTo input
/// validation; zero-preserving coalesce / sparse_mask (torch's order and duplicates), beta == 0 ignoring the input,
/// and products kept for zero-valued stored rows.
/// </summary>
public class ReviewFixesSharedNamedSparseTests
{
    // ---------------------------------------------------------------- shared memory

    [Fact]
    public void SharedMemory_ViewsReportTheMapping_AndWritesReachTheOtherSide()
    {
        var t = new Tensor<float>(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), new[] { 3, 4 }).ShareMemory();
        string path = t.SharedMemoryPath!;
        Assert.NotNull(path);
        var view = t.Reshape(new[] { 4, 3 });
        Assert.True(view.IsShared);
        Assert.Equal(path, view.SharedMemoryPath);
        var other = Tensor<float>.FromSharedMemory(path, new[] { 12 });
        view[1, 1] = 42f;   // element 4
        Assert.Equal(42f, other[4]);
    }

    [Fact]
    public void SharedMemory_LargeTensor_IsWrittenThroughBoundedChunks()
    {
        // 3 MiB: more than one 1 MiB chunk, with a partial last chunk.
        int n = 3 * (1 << 18) + 7;
        var data = Enumerable.Range(0, n).Select(i => (float)(i % 1000)).ToArray();
        var t = new Tensor<float>(data, new[] { n }).ShareMemory();
        var mapped = Tensor<float>.FromSharedMemory(t.SharedMemoryPath!, new[] { n });
        Assert.Equal(data, mapped.ToArray());
    }

    [Fact]
    public void SharedMemory_FileIsOwnerOnly_OnUnix()
    {
#if NET7_0_OR_GREATER
        if (OperatingSystem.IsWindows()) return;   // Windows: the per-user temporary directory's ACL applies
        var t = new Tensor<float>(new float[8], new[] { 8 }).ShareMemory();
        var mode = File.GetUnixFileMode(t.SharedMemoryPath!);
        Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, mode);
#endif
    }

    [Fact]
    public void SharedMemory_FileIsDeleted_WhenTheStorageIsReleased()
    {
        var t = new Tensor<float>(new float[16], new[] { 16 }).ShareMemory();
        string path = t.SharedMemoryPath!;
        Assert.True(File.Exists(path));
        t._storage.Release();   // the last (only) reference
        Assert.False(File.Exists(path));
    }

    [Fact]
    public void FromSharedMemory_RejectsMoreThanIntMaxBytes_BeforeAllocating()
    {
        string path = Path.GetTempFileName();
        try
        {
            // 600M floats = 2.4 GB: rejected from the shape alone, without mapping or allocating anything.
            Assert.Throws<ArgumentOutOfRangeException>(() => Tensor<float>.FromSharedMemory(path, new[] { 600_000_000 }));
            Assert.Throws<ArgumentException>(() => Tensor<float>.FromSharedMemory(path, new[] { -1 }));
        }
        finally { File.Delete(path); }
    }

    // ---------------------------------------------------------------- named dimensions

    [Fact]
    public void Names_AreCopied_AndExposedReadOnly()
    {
        var names = new string?[] { "N", "C" };
        var t = new Tensor<float>(new float[6], new[] { 2, 3 }).Rename(names);
        names[0] = "C";   // would make a duplicate if the tensor kept the caller's array
        Assert.Equal(new[] { "N", "C" }, t.Names.ToArray());
        Assert.False(t.Names is string?[]);
    }

    [Fact]
    public void AlignTo_IsAStorageSharingView_WithInsertedUnitDimensions()
    {
        var x = new Tensor<float>(Enumerable.Range(0, 6).Select(i => (float)i).ToArray(), new[] { 2, 3 }).Rename("N", "C");
        var aligned = x.AlignTo("C", "H", "N");
        Assert.Equal(new[] { 3, 1, 2 }, aligned.Shape.ToArray());
        Assert.Equal(new[] { "C", "H", "N" }, aligned.Names.ToArray());
        Assert.Equal(x[1, 2], aligned[2, 0, 1]);
        aligned[2, 0, 1] = 99f;   // a write through the aligned view reaches the source
        Assert.Equal(99f, x[1, 2]);
    }

    [Fact]
    public void AlignTo_Gradient_UndoesThePermutation()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(Enumerable.Range(0, 6).Select(i => (float)i).ToArray(), new[] { 2, 3 }).Rename("N", "C");
        var w = new Tensor<float>(Enumerable.Range(1, 6).Select(i => (float)i).ToArray(), new[] { 3, 1, 2 });
        using var tape = new GradientTape<float>();
        var aligned = x.AlignTo("C", "H", "N");
        var loss = engine.ReduceSum(engine.TensorMultiply(aligned.Contiguous(), w), null);
        var g = tape.ComputeGradients(loss, new[] { x })[x];
        // d loss / d x[n, c] = w[c, 0, n]
        for (int n = 0; n < 2; n++)
            for (int c = 0; c < 3; c++)
                Assert.Equal(w[c, 0, n], g[n, c]);
    }

    [Fact]
    public void AlignTo_Gradient_UndoesACyclicPermutation()
    {
        // (A, B, C) -> (C, A, B) is a 3-cycle: its inverse is not itself, unlike a 2-D swap.
        var engine = new CpuEngine();
        var x = new Tensor<float>(Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), new[] { 2, 3, 4 }).Rename("A", "B", "C");
        var w = new Tensor<float>(Enumerable.Range(1, 24).Select(i => (float)i).ToArray(), new[] { 4, 2, 3 });
        using var tape = new GradientTape<float>();
        var aligned = x.AlignTo("C", "A", "B");
        var loss = engine.ReduceSum(engine.TensorMultiply(aligned.Contiguous(), w), null);
        var g = tape.ComputeGradients(loss, new[] { x })[x];
        for (int a = 0; a < 2; a++)
            for (int bb = 0; bb < 3; bb++)
                for (int c = 0; c < 4; c++)
                    Assert.Equal(w[c, a, bb], g[a, bb, c]);
    }

    [Fact]
    public void AlignTo_RejectsNullOrEmptyNamesAndTwoEllipses()
    {
        var x = new Tensor<float>(new float[6], new[] { 2, 3 }).Rename("N", "C");
        Assert.Throws<ArgumentException>(() => x.AlignTo("N", ""));
        Assert.Throws<ArgumentException>(() => x.AlignTo(new string[] { "N", null! }));
        Assert.Throws<ArgumentException>(() => x.AlignTo("...", "N", "..."));
    }

    [Fact]
    public void Rename_OfANonContiguousTensor_SharesStorage()
    {
        var x = new Tensor<float>(Enumerable.Range(0, 6).Select(i => (float)i).ToArray(), new[] { 2, 3 });
        var transposed = x.Transpose(new[] { 1, 0 });   // non-contiguous view
        var named = transposed.Rename("A", "B");
        named[2, 1] = 77f;
        Assert.Equal(77f, x[1, 2]);
    }

    // ---------------------------------------------------------------- sparse torch ops

    private static Tensor<double> Arange(int rows, int cols) =>
        new(Enumerable.Range(0, rows * cols).Select(i => (double)i).ToArray(), new[] { rows, cols });

    [Fact]
    public void SparseMask_KeepsTheMaskIndicesAsStored_LikeTorch()
    {
        // torch: src.sparse_mask(mask) with mask indices [[2,0,1,2,2],[2,1,0,0,0]] (uncoalesced, a duplicate at (2,0),
        // a stored zero) returns those indices unchanged, values [8, 1, 3, 6, 6].
        var mask = new SparseTensor<double>(3, 3, new[] { 2, 0, 1, 2, 2 }, new[] { 2, 1, 0, 0, 0 }, new[] { 5.0, 0.0, 3.0, 1.0, -1.0 });
        var r = SparseTensor<double>.SparseMask(Arange(3, 3), mask);
        Assert.Equal(new[] { 2, 0, 1, 2, 2 }, r.RowIndices);
        Assert.Equal(new[] { 2, 1, 0, 0, 0 }, r.ColumnIndices);
        Assert.Equal(new[] { 8.0, 1.0, 3.0, 6.0, 6.0 }, r.Values);
    }

    [Fact]
    public void Indices_KeepStoredZeros_LikeTorchCoalesce()
    {
        // torch: coalesce keeps (0,1) = 0 and (2,0) = 1 + -1 = 0 -> indices [[0,1,2,2],[1,0,0,2]].
        var s = new SparseTensor<double>(3, 3, new[] { 2, 0, 1, 2, 2 }, new[] { 2, 1, 0, 0, 0 }, new[] { 5.0, 0.0, 3.0, 1.0, -1.0 });
        Assert.Equal(new[] { 0, 1, 2, 2, 1, 0, 0, 2 }, s.Indices().ToArray());
    }

    [Fact]
    public void Saddmm_BetaZero_IgnoresTheInput_EvenNaN()
    {
        var input = new SparseTensor<double>(2, 2, new[] { 0, 1 }, new[] { 0, 1 }, new[] { double.NaN, double.PositiveInfinity });
        var mat1 = new SparseTensor<double>(2, 2, new[] { 0 }, new[] { 1 }, new[] { 2.0 });
        var mat2 = Arange(2, 2);
        var r = SparseTensor<double>.Saddmm(input, mat1, mat2, 0.0, 1.0).ToDense().ToArray();
        Assert.Equal(new[] { 4.0, 6.0, 0.0, 0.0 }, r);   // 2 * row 1 of mat2 = [4, 6]; input ignored
    }

    [Fact]
    public void Smm_KeepsARowThatStoresOnlyZeros_AndScalesWithActiveRows()
    {
        // Row 0 stores an explicit zero: torch keeps that row's (zero) products in the pattern.
        var s = new SparseTensor<double>(1_000_000, 2, new[] { 0, 999_999 }, new[] { 0, 1 }, new[] { 0.0, 3.0 });
        var r = SparseTensor<double>.Smm(s, Arange(2, 4));
        Assert.Equal(8, r.NonZeroCount);   // two active rows x 4 columns
        Assert.Equal(new[] { 0, 0, 0, 0, 999_999, 999_999, 999_999, 999_999 }, r.RowIndices);
        Assert.Equal(new[] { 0.0, 0.0, 0.0, 0.0, 12.0, 15.0, 18.0, 21.0 }, r.Values);
    }
}
