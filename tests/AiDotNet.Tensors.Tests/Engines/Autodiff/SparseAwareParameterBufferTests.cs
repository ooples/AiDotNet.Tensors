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
    /// End-to-end + ParameterBuffer integration: the sparse gradient
    /// produced by SparsePatternPreservingMatMulRecord flows through
    /// ParameterBuffer.FlattenGradients and lands in the right slot
    /// without densification. This is the load-bearing test proving
    /// SparseLinearLayer can train via TrainWithTape with a sparse-aware
    /// ParameterBuffer.
    /// </summary>
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
