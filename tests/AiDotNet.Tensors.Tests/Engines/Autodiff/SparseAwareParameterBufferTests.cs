using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Issue #286 — sparse-aware <see cref="ParameterBuffer{T}"/> +
/// pattern-preserving sparse·dense matmul autograd op.
///
/// Together these unblock <c>SparseLinearLayer</c> integration with
/// <c>TrainWithTape</c> in the AiDotNet repo without paying the
/// O(rows × columns) memory cost of a dense parameter shadow.
/// </summary>
public class SparseAwareParameterBufferTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static SparsityLayout PatternFor(SparseTensor<float> source) =>
        SparsityLayout.FromSparseTensor(source);

    /// <summary>
    /// Buffer-slot sizing for sparse leaves: a 4×4 sparse identity matrix
    /// (NonZeroCount = 4) should occupy 4 buffer slots, NOT 16. The whole
    /// point of the new ctor is to skip the dense O(rows × columns)
    /// allocation that the original API forced for sparse leaves.
    /// </summary>
    [Fact]
    public void Ctor_SparseLeaf_AllocatesNonZeroCountSlotsOnly()
    {
        var pattern = new SparsityLayout(4, 4,
            new[] { 0, 1, 2, 3 }, new[] { 0, 1, 2, 3 });
        var layouts = new[]
        {
            new ParameterLayout(new[] { 4, 4 }, pattern),
        };
        var buffer = new ParameterBuffer<float>(layouts);

        Assert.Equal(4, buffer.TotalSize);
        Assert.True(buffer.IsSparse(0));
        Assert.Equal(4, buffer.GetSparseLayout(0)!.NonZeroCount);
    }

    /// <summary>
    /// Mixed dense + sparse layouts: each occupies its own slot at the
    /// correct buffer offset. Catches an off-by-one in offset arithmetic
    /// when the slot size depends on the layout type.
    /// </summary>
    [Fact]
    public void Ctor_MixedDenseAndSparse_OffsetsAreContiguous()
    {
        var pattern = new SparsityLayout(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var layouts = new[]
        {
            new ParameterLayout(new[] { 2, 5 }),                    // dense slot: 10
            new ParameterLayout(new[] { 3, 3 }, pattern),           // sparse slot: 3
            new ParameterLayout(new[] { 4 }),                       // dense slot: 4
        };
        var buffer = new ParameterBuffer<float>(layouts);

        Assert.Equal(17, buffer.TotalSize); // 10 + 3 + 4
        Assert.Equal(0, buffer.GetOffset(0));
        Assert.Equal(10, buffer.GetOffset(1));
        Assert.Equal(13, buffer.GetOffset(2));
        Assert.False(buffer.IsSparse(0));
        Assert.True(buffer.IsSparse(1));
        Assert.False(buffer.IsSparse(2));
    }

    /// <summary>
    /// Ctor must reject a sparse layout whose dense shape mismatches the
    /// pattern's (rows, columns).
    /// </summary>
    [Fact]
    public void Ctor_SparseShapeMismatch_Throws()
    {
        var pattern = new SparsityLayout(3, 3, new[] { 0 }, new[] { 0 });
        Assert.Throws<ArgumentException>(() =>
            new ParameterLayout(new[] { 4, 4 }, pattern));
    }

    /// <summary>
    /// True zero-copy contract for sparse leaves: writes to the buffer
    /// flow through to a previously-obtained CreateView, AND writes to
    /// the SparseTensor's underlying values vector flow back to the
    /// buffer. Mirrors the dense-leaf contract that callers depend on.
    /// </summary>
    [Fact]
    public void CreateView_SparseLeaf_IsLiveZeroCopyView()
    {
        var pattern = new SparsityLayout(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, pattern),
        });

        // Get a view, then mutate the buffer.
        var view = (SparseTensor<float>)buffer.CreateView(0);
        var bufferSpan = buffer.GetSparseValuesSpan(0);
        bufferSpan[0] = 11f;
        bufferSpan[1] = 22f;
        bufferSpan[2] = 33f;

        // The view sees the new values WITHOUT being re-fetched.
        Assert.Equal(11f, view[0, 0]);
        Assert.Equal(22f, view[1, 1]);
        Assert.Equal(33f, view[2, 2]);

        // Writes through the view's underlying values vector flow back
        // into the buffer.
        view.DataVector.AsWritableSpan()[0] = 100f;
        Assert.Equal(100f, buffer.GetSparseValuesReadOnlySpan(0)[0]);
    }

    /// <summary>
    /// SparsityLayout clones index arrays so external mutations to the
    /// caller's array don't desynchronise the layout from buffer slots
    /// that depend on its indices being immutable.
    /// </summary>
    [Fact]
    public void SparsityLayout_Ctor_ClonesIndexArrays()
    {
        var rowIdx = new[] { 0, 1, 2 };
        var colIdx = new[] { 0, 1, 2 };
        var layout = new SparsityLayout(3, 3, rowIdx, colIdx);

        // Mutate the caller's arrays — must not affect the layout.
        rowIdx[0] = 99;
        colIdx[0] = 99;

        Assert.Equal(0, layout.RowIndices[0]);
        Assert.Equal(0, layout.ColumnIndices[0]);
    }

    /// <summary>
    /// Sparse-only buffer with a very large dense semantic shape: the
    /// dense product would exceed int.MaxValue but the buffer should
    /// allocate based on NonZeroCount only. The whole point of sparse
    /// leaves is to express huge sparse matrices without paying dense
    /// memory cost.
    /// </summary>
    [Fact]
    public void Ctor_SparseLeaf_LargeDenseShape_DoesNotOverflowOnDenseProduct()
    {
        // 50,000 × 50,000 = 2.5e9 (overflows int) — but only 4 non-zeros.
        var pattern = new SparsityLayout(50_000, 50_000,
            new[] { 0, 1, 2, 3 }, new[] { 0, 1, 2, 3 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 50_000, 50_000 }, pattern),
        });

        Assert.Equal(4, buffer.TotalSize);
        Assert.True(buffer.IsSparse(0));
    }

    /// <summary>
    /// CreateView for a sparse leaf returns a SparseTensor whose Values
    /// reflect the buffer's current contents at the layout's pattern
    /// positions, with the recorded RowIndices / ColumnIndices.
    /// </summary>
    [Fact]
    public void CreateView_SparseLeaf_ReturnsSparseTensorWithLayoutPattern()
    {
        var rowIdx = new[] { 0, 1, 2 };
        var colIdx = new[] { 1, 2, 0 };
        var pattern = new SparsityLayout(3, 3, rowIdx, colIdx);
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, pattern),
        });

        // Seed the buffer slot directly via the values span.
        var slot = buffer.GetSparseValuesSpan(0);
        slot[0] = 7f;
        slot[1] = 8f;
        slot[2] = 9f;

        var view = (SparseTensor<float>)buffer.CreateView(0);
        Assert.Equal(3, view.Rows);
        Assert.Equal(3, view.Columns);
        Assert.Equal(3, view.NonZeroCount);
        Assert.Equal(7f, view[0, 1]);
        Assert.Equal(8f, view[1, 2]);
        Assert.Equal(9f, view[2, 0]);
        Assert.Equal(0f, view[0, 0]); // structural zero
    }

    /// <summary>
    /// CopyFrom(IReadOnlyList&lt;Tensor&lt;T&gt;&gt;) for a sparse leaf
    /// must reject inputs whose pattern differs from the layout — the
    /// sparsity pattern is fixed at construction.
    /// </summary>
    [Fact]
    public void CopyFrom_SparseLeaf_RejectsPatternMismatch()
    {
        var pattern = new SparsityLayout(3, 3, new[] { 0, 1 }, new[] { 0, 1 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, pattern),
        });

        // Different pattern — should throw.
        var differentPattern = new SparseTensor<float>(3, 3,
            new[] { 0, 2 }, new[] { 0, 2 }, new[] { 1f, 2f });
        Assert.Throws<ArgumentException>(() =>
            buffer.CopyFrom(new Tensor<float>[] { differentPattern }));
    }

    /// <summary>
    /// CopyFrom for a sparse leaf with matching pattern copies the
    /// Values into the buffer. Round-trip via CreateView.
    /// </summary>
    [Fact]
    public void CopyFrom_SparseLeaf_PatternMatch_CopiesValues()
    {
        var pattern = new SparsityLayout(3, 3, new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, pattern),
        });

        var src = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 10f, 20f, 30f });
        buffer.CopyFrom(new Tensor<float>[] { src });

        var view = (SparseTensor<float>)buffer.CreateView(0);
        Assert.Equal(10f, view[0, 0]);
        Assert.Equal(20f, view[1, 1]);
        Assert.Equal(30f, view[2, 2]);
    }

    /// <summary>
    /// FlattenGradients for a sparse leaf with a matching-pattern sparse
    /// gradient: pulls Values directly into the slot.
    /// </summary>
    [Fact]
    public void FlattenGradients_SparseLeaf_SparseGradMatchingPattern_CopiesValues()
    {
        var pattern = new SparsityLayout(3, 3, new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, pattern),
        });

        // The "parameter" we pass in is the SparseTensor used as the
        // gradient-dictionary key.
        var paramTensor = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 0f, 0f, 0f });
        var gradTensor = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 0.1f, 0.2f, 0.3f });

        var grads = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>
        {
            { paramTensor, gradTensor },
        };

        var flat = buffer.FlattenGradients(new Tensor<float>[] { paramTensor }, grads);
        Assert.Equal(3, flat.Length);
        Assert.Equal(0.1f, flat[0]);
        Assert.Equal(0.2f, flat[1]);
        Assert.Equal(0.3f, flat[2]);
    }

    /// <summary>
    /// FlattenGradients for a sparse leaf with a DENSE gradient (the
    /// PyTorch-default backward form) projects only the values at the
    /// pattern's positions. Values at structural-zero positions are
    /// silently dropped — the layer only trains pattern positions.
    /// </summary>
    [Fact]
    public void FlattenGradients_SparseLeaf_DenseGrad_ProjectsToPattern()
    {
        var pattern = new SparsityLayout(3, 3, new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, pattern),
        });

        var paramTensor = new SparseTensor<float>(3, 3,
            new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 0f, 0f, 0f });
        var denseGrad = new Tensor<float>(new[] { 3, 3 });
        // Fill with a known pattern: G[i,j] = i*3 + j + 1
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                denseGrad[i, j] = i * 3 + j + 1;

        var grads = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>
        {
            { paramTensor, denseGrad },
        };

        var flat = buffer.FlattenGradients(new Tensor<float>[] { paramTensor }, grads);
        // Pattern positions: (0,0), (1,1), (2,2) → 1, 5, 9
        Assert.Equal(1f, flat[0]);
        Assert.Equal(5f, flat[1]);
        Assert.Equal(9f, flat[2]);
    }

    /// <summary>
    /// End-to-end: pattern-preserving sparse·dense matmul forward +
    /// backward via SparseAutograd.SparsePatternPreservingMatMulRecord.
    /// The sparse gradient produced has matching pattern and Values
    /// that ParameterBuffer.FlattenGradients copies into the buffer
    /// slot. Verifies dY/dA[i,j] = sum_k B[j,k] · gradOut[i,k] over
    /// pattern positions only.
    /// </summary>
    [Fact]
    public void SparsePatternPreservingMatMulRecord_BackwardProducesPatternMatchingSparseGrad()
    {
        // A: 2×3 sparse with pattern at (0,0), (1,1), (1,2).
        var aPattern = new SparseTensor<float>(2, 3,
            new[] { 0, 1, 1 }, new[] { 0, 1, 2 }, new[] { 1f, 2f, 3f });

        // B: 3×2 dense
        var b = new Tensor<float>(new[] { 3, 2 });
        b[0, 0] = 0.5f; b[0, 1] = 0.7f;
        b[1, 0] = 1.0f; b[1, 1] = 1.5f;
        b[2, 0] = 2.0f; b[2, 1] = 2.5f;

        using var tape = new GradientTape<float>();
        var output = SparseAutograd.SparsePatternPreservingMatMulRecord(aPattern, b);

        // Force a non-trivial gradient by reducing over output.
        // Here we just feed an explicit gradOut of all-ones to keep the
        // expected math simple to reason about.
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(2, output.Shape[1]);

        // Reduce: sum(output) → scalar. Backward propagates 1.0 to every
        // output position.
        var loss = _engine.ReduceSum(output, axes: null, keepDims: false);
        var grads = tape.ComputeGradients(loss, new Tensor<float>[] { aPattern, b });

        // gradA against aPattern should be a SparseTensor with same pattern.
        Assert.True(grads.ContainsKey(aPattern));
        var gradA = grads[aPattern];
        Assert.True(gradA is SparseTensor<float>, "Expected pattern-preserving sparse gradient.");
        var sparseGradA = (SparseTensor<float>)gradA;
        Assert.Equal(3, sparseGradA.NonZeroCount);

        // For gradOut = 1s everywhere: gradA[i,j] = sum_k B[j,k] · 1 = B[j,0] + B[j,1]
        //   pattern (0,0): B[0,0] + B[0,1] = 0.5 + 0.7 = 1.2
        //   pattern (1,1): B[1,0] + B[1,1] = 1.0 + 1.5 = 2.5
        //   pattern (1,2): B[2,0] + B[2,1] = 2.0 + 2.5 = 4.5
        Assert.Equal(1.2f, sparseGradA[0, 0], 5);
        Assert.Equal(2.5f, sparseGradA[1, 1], 5);
        Assert.Equal(4.5f, sparseGradA[1, 2], 5);
    }

    /// <summary>
    /// Pattern-preserving SpGeMM (sparse · sparse) backward: produces
    /// SPARSE gradients on BOTH sides matching their respective patterns.
    /// Verifies dA and dB values against the analytic formulas.
    ///
    /// A is 2×3 with pattern (0,0), (0,1), (1,2); values 1, 2, 3.
    /// B is 3×2 with pattern (0,0), (1,0), (2,1); values 4, 5, 6.
    /// Forward dense Y = A·B at (0,0) = 1·4 + 2·5 = 14, etc.
    /// </summary>
    [Fact]
    public void SparsePatternPreservingSpGeMMRecord_BothSidesGetPatternMatchingSparseGrads()
    {
        var a = new SparseTensor<float>(2, 3,
            new[] { 0, 0, 1 }, new[] { 0, 1, 2 }, new[] { 1f, 2f, 3f });
        var b = new SparseTensor<float>(3, 2,
            new[] { 0, 1, 2 }, new[] { 0, 0, 1 }, new[] { 4f, 5f, 6f });

        using var tape = new GradientTape<float>();
        var output = SparseAutograd.SparsePatternPreservingSpGeMMRecord(a, b);
        var loss = _engine.ReduceSum(output, axes: null, keepDims: false);
        var grads = tape.ComputeGradients(loss, new Tensor<float>[] { a, b });

        Assert.True(grads.ContainsKey(a));
        Assert.True(grads.ContainsKey(b));
        var gradA = (SparseTensor<float>)grads[a];
        var gradB = (SparseTensor<float>)grads[b];

        Assert.Equal(3, gradA.NonZeroCount);
        Assert.Equal(3, gradB.NonZeroCount);

        // gradOut is all-ones; dA[i,j] = sum_k B[j,k] · 1 = sum_k B[j,k].
        //   pattern (0,0): row 0 of B has only B[0,0]=4 → dA = 4
        //   pattern (0,1): row 1 of B has only B[1,0]=5 → dA = 5
        //   pattern (1,2): row 2 of B has only B[2,1]=6 → dA = 6
        Assert.Equal(4f, gradA[0, 0], 5);
        Assert.Equal(5f, gradA[0, 1], 5);
        Assert.Equal(6f, gradA[1, 2], 5);

        // dB[j,k] = sum_i A[i,j] · 1 = sum over i where (i,j) ∈ A.
        //   pattern (0,0): col 0 of A has only A[0,0]=1 → dB = 1
        //   pattern (1,0): col 1 of A has only A[0,1]=2 → dB = 2
        //   pattern (2,1): col 2 of A has only A[1,2]=3 → dB = 3
        Assert.Equal(1f, gradB[0, 0], 5);
        Assert.Equal(2f, gradB[1, 0], 5);
        Assert.Equal(3f, gradB[2, 1], 5);
    }

    /// <summary>
    /// Variable-sparsity-pattern in-place rebuild: same NonZeroCount,
    /// new positions. Mirrors dynamic-sparse training (SET / RigL /
    /// RigL+ / threshold pruning) where the sparsity ratio stays fixed
    /// but the active connections move between optimizer steps.
    /// </summary>
    [Fact]
    public void RebuildSparsePattern_SameNonZeroCount_InPlaceUpdate()
    {
        var oldPattern = new SparsityLayout(3, 3, new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, oldPattern),
        });

        var span = buffer.GetSparseValuesSpan(0);
        span[0] = 1f; span[1] = 2f; span[2] = 3f;

        // Rebuild pattern: same nnz=3 but different positions.
        var newPattern = new SparsityLayout(3, 3, new[] { 0, 1, 2 }, new[] { 2, 0, 1 });
        buffer.RebuildSparsePattern(0, newPattern, new float[] { 7f, 8f, 9f });

        Assert.Equal(3, buffer.GetSparseLayout(0)!.NonZeroCount);
        var view = (SparseTensor<float>)buffer.CreateView(0);
        Assert.Equal(7f, view[0, 2]);
        Assert.Equal(8f, view[1, 0]);
        Assert.Equal(9f, view[2, 1]);
        // Old positions are now structural zero.
        Assert.Equal(0f, view[0, 0]);
    }

    /// <summary>
    /// RebuildSparsePattern rejects a new pattern whose NonZeroCount
    /// differs — caller must use ResizeSparseLeaf for that.
    /// </summary>
    [Fact]
    public void RebuildSparsePattern_NonZeroCountMismatch_Throws()
    {
        var oldPattern = new SparsityLayout(3, 3, new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, oldPattern),
        });

        var biggerPattern = new SparsityLayout(3, 3,
            new[] { 0, 1, 2, 0 }, new[] { 0, 1, 2, 1 });
        Assert.Throws<ArgumentException>(() =>
            buffer.RebuildSparsePattern(0, biggerPattern));
    }

    /// <summary>
    /// ResizeSparseLeaf — full variable-pattern support: NonZeroCount
    /// can grow or shrink. Buffer is reallocated; subsequent leaves'
    /// offsets shift by the delta. Other leaves' values preserved.
    /// </summary>
    [Fact]
    public void ResizeSparseLeaf_GrowSlot_PreservesOtherLeavesAndShiftsOffsets()
    {
        var pattern0 = new SparsityLayout(3, 3, new[] { 0, 1 }, new[] { 0, 1 });   // nnz=2
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, pattern0),     // sparse, nnz=2
            new ParameterLayout(new[] { 4 }),                  // dense, size=4
        });

        // Seed: sparse leaf 0 with [1, 2], dense leaf 1 with [10, 20, 30, 40].
        var s0 = buffer.GetSparseValuesSpan(0);
        s0[0] = 1f; s0[1] = 2f;
        var d1 = buffer.CreateView(1);
        d1[0] = 10f; d1[1] = 20f; d1[2] = 30f; d1[3] = 40f;

        Assert.Equal(0, buffer.GetOffset(0));   // first leaf always at offset 0
        Assert.Equal(2, buffer.GetOffset(1));   // dense leaf starts after sparse nnz=2

        // Grow the sparse leaf to nnz=5.
        var newPattern = new SparsityLayout(3, 3,
            new[] { 0, 0, 1, 1, 2 }, new[] { 0, 1, 0, 2, 2 });
        buffer.ResizeSparseLeaf(0, newPattern,
            new float[] { 100f, 200f, 300f, 400f, 500f });

        // Leaf 0 now has nnz=5; leaf 1's offset shifts by +3.
        Assert.Equal(5, buffer.GetSparseLayout(0)!.NonZeroCount);
        Assert.Equal(0, buffer.GetOffset(0));
        Assert.Equal(5, buffer.GetOffset(1));

        // Sparse values updated.
        var view0 = (SparseTensor<float>)buffer.CreateView(0);
        Assert.Equal(100f, view0[0, 0]);
        Assert.Equal(200f, view0[0, 1]);
        Assert.Equal(300f, view0[1, 0]);
        Assert.Equal(400f, view0[1, 2]);
        Assert.Equal(500f, view0[2, 2]);

        // Dense leaf preserved across resize.
        var view1 = buffer.CreateView(1);
        Assert.Equal(10f, view1[0]);
        Assert.Equal(20f, view1[1]);
        Assert.Equal(30f, view1[2]);
        Assert.Equal(40f, view1[3]);

        // Total size grew by delta.
        Assert.Equal(9, buffer.TotalSize); // 5 (sparse new) + 4 (dense)
    }

    /// <summary>
    /// ResizeSparseLeaf shrinking case (NonZeroCount decreases).
    /// </summary>
    [Fact]
    public void ResizeSparseLeaf_ShrinkSlot_ShiftsOffsetsBack()
    {
        var pattern0 = new SparsityLayout(3, 3,
            new[] { 0, 0, 1, 2 }, new[] { 0, 1, 1, 2 });   // nnz=4
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 3, 3 }, pattern0),     // sparse, nnz=4
            new ParameterLayout(new[] { 2 }),                  // dense, size=2
        });

        var d1 = buffer.CreateView(1);
        d1[0] = 7f; d1[1] = 9f;

        var newPattern = new SparsityLayout(3, 3, new[] { 0, 2 }, new[] { 0, 2 }); // nnz=2
        buffer.ResizeSparseLeaf(0, newPattern, new float[] { 11f, 22f });

        Assert.Equal(2, buffer.GetSparseLayout(0)!.NonZeroCount);
        Assert.Equal(2, buffer.GetOffset(1)); // shifted back from 4 to 2
        Assert.Equal(4, buffer.TotalSize);    // 2 sparse + 2 dense

        var view0 = (SparseTensor<float>)buffer.CreateView(0);
        Assert.Equal(11f, view0[0, 0]);
        Assert.Equal(22f, view0[2, 2]);

        var view1 = buffer.CreateView(1);
        Assert.Equal(7f, view1[0]);
        Assert.Equal(9f, view1[1]);
    }

    [Fact]
    public void SparsePatternPreservingMatMul_IntegratesWithSparseAwareBuffer()
    {
        var rowIdx = new[] { 0, 1, 1 };
        var colIdx = new[] { 0, 1, 2 };
        var aValues = new[] { 1f, 2f, 3f };
        var aPattern = new SparseTensor<float>(2, 3, rowIdx, colIdx, aValues);

        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 2, 3 }, SparsityLayout.FromSparseTensor(aPattern)),
        });
        buffer.CopyFrom(new Tensor<float>[] { aPattern });

        var b = new Tensor<float>(new[] { 3, 2 });
        b[0, 0] = 0.5f; b[0, 1] = 0.7f;
        b[1, 0] = 1.0f; b[1, 1] = 1.5f;
        b[2, 0] = 2.0f; b[2, 1] = 2.5f;

        using var tape = new GradientTape<float>();
        var output = SparseAutograd.SparsePatternPreservingMatMulRecord(aPattern, b);
        var loss = _engine.ReduceSum(output, axes: null, keepDims: false);
        var grads = tape.ComputeGradients(loss, new Tensor<float>[] { aPattern, b });

        var flat = buffer.FlattenGradients(new Tensor<float>[] { aPattern }, grads);
        Assert.Equal(3, flat.Length);
        // Same expected values as the previous test.
        Assert.Equal(1.2f, flat[0], 5);
        Assert.Equal(2.5f, flat[1], 5);
        Assert.Equal(4.5f, flat[2], 5);
    }
}
