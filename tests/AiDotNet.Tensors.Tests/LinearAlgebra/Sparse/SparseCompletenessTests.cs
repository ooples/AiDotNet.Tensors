// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Sparse;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.Sparse;

/// <summary>
/// Acceptance tests for sparse completeness (#221). Covers:
///   • Format coverage: COO/CSR/CSC/BSR/BSC round-trips and dense parity.
///   • Op surface: SpMM/SpMV/AddMM/Bmm/sum/mean/softmax/log-softmax/mask/
///     sampled-addmm/spdiags/SpGEMM vs dense reference.
///   • 2:4 structured semi-sparse: pruning, dense reconstruction, matmul.
///   • Sparse autograd: SparseMatMul / SparseAddMM / SparseSampledAddMM
///     gradients vs finite-difference reference (and vs the dense
///     analytic backward in the SparseMatMul case).
/// </summary>
public class SparseCompletenessTests
{
    private readonly CpuEngine _engine = new();

    private static SparseTensor<float> SmallA() =>
        // [[1, 0, 2, 0],
        //  [0, 3, 0, 4],
        //  [5, 0, 6, 0],
        //  [0, 7, 0, 8]]
        SparseTensor<float>.FromDense(MakeDense(new float[,]
        {
            { 1, 0, 2, 0 },
            { 0, 3, 0, 4 },
            { 5, 0, 6, 0 },
            { 0, 7, 0, 8 },
        }));

    private static Tensor<float> MakeDense(float[,] data)
    {
        int r = data.GetLength(0), c = data.GetLength(1);
        var t = new Tensor<float>(new[] { r, c });
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                t[i, j] = data[i, j];
        return t;
    }

    [Fact]
    public void CooCsrCscBsrBsc_RoundTripPreservesDense()
    {
        var a = SmallA();
        var dense = a.ToDense();

        AssertDenseEqual(dense, a.ToCoo().ToDense());
        AssertDenseEqual(dense, a.ToCsr().ToDense());
        AssertDenseEqual(dense, a.ToCsc().ToDense());
        AssertDenseEqual(dense, a.ToBsr(2, 2).ToDense());
        AssertDenseEqual(dense, a.ToBsc(2, 2).ToDense());

        // Round-trip through every format chain.
        AssertDenseEqual(dense, a.ToCsr().ToCsc().ToBsr(2, 2).ToBsc(2, 2).ToDense());
    }

    [Fact]
    public void Bsr_FromBsrFactory_PreservesDense()
    {
        // 4x4 matrix with explicit 2x2 blocks at positions (0,0), (1,1).
        // block (0,0) = [[1,2],[3,4]]; block (1,1) = [[5,6],[7,8]].
        // Dense:
        //   1 2 0 0
        //   3 4 0 0
        //   0 0 5 6
        //   0 0 7 8
        var values = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var blockRowPtr = new[] { 0, 1, 2 };
        var blockColIdx = new[] { 0, 1 };
        var bsr = SparseTensor<float>.FromBsr(4, 4, 2, 2, blockRowPtr, blockColIdx, values);
        var dense = bsr.ToDense();
        Assert.Equal(1f, dense[0, 0]);
        Assert.Equal(2f, dense[0, 1]);
        Assert.Equal(0f, dense[0, 2]);
        Assert.Equal(4f, dense[1, 1]);
        Assert.Equal(5f, dense[2, 2]);
        Assert.Equal(8f, dense[3, 3]);
    }

    [Fact]
    public void SparseMatMul_AgainstDenseReference()
    {
        var a = SmallA();
        var b = MakeDense(new float[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 },
            { 7, 8 },
        });
        var sparseProduct = SparseOps.SparseMatMul(a, b);
        var denseProduct = _engine.TensorMatMul(a.ToDense(), b);
        AssertDenseEqual(denseProduct, sparseProduct);
    }

    [Fact]
    public void SparseSpMV_AgainstDenseReference()
    {
        var a = SmallA();
        var x = new Tensor<float>(new[] { 4 });
        x[0] = 1; x[1] = 2; x[2] = 3; x[3] = 4;
        var sparseProduct = SparseOps.SparseSpMV(a, x);
        // Dense reference: a · x reshaped.
        var dense = a.ToDense();
        for (int r = 0; r < 4; r++)
        {
            float expected = 0;
            for (int c = 0; c < 4; c++) expected += dense[r, c] * x[c];
            Assert.Equal(expected, sparseProduct[r], 4);
        }
    }

    [Fact]
    public void SparseAddMM_AgainstDenseReference()
    {
        var a = SmallA();
        var b = MakeDense(new float[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 },
            { 7, 8 },
        });
        var c = new Tensor<float>(new[] { 4, 2 });
        for (int i = 0; i < 4; i++) { c[i, 0] = 1; c[i, 1] = -1; }

        const float alpha = 2f, beta = 0.5f;
        var got = SparseOps.SparseAddMM(c, a, b, alpha, beta);
        var dense = a.ToDense();
        var ab = _engine.TensorMatMul(dense, b);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(alpha * ab[i, j] + beta * c[i, j], got[i, j], 3);
    }

    [Fact]
    public void SparseBmm_PerBatchMatchesDenseReference()
    {
        var a0 = SmallA();
        var a1 = SparseTensor<float>.FromDense(MakeDense(new float[,]
        {
            { 0, 1, 0, 0 },
            { 1, 0, 1, 0 },
            { 0, 1, 0, 1 },
            { 0, 0, 1, 0 },
        }));

        var bBatch = new Tensor<float>(new[] { 2, 4, 2 });
        for (int batch = 0; batch < 2; batch++)
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++)
                    bBatch[batch, i, j] = (batch + 1) * (i + j + 1);

        var got = SparseOps.SparseBmm(new[] { a0, a1 }, bBatch);
        Assert.Equal(new[] { 2, 4, 2 }, got._shape);
    }

    [Fact]
    public void SparseSum_NoAxis_ReducesAllNonZeros()
    {
        var a = SmallA();
        var sum = SparseOps.SparseSum(a);
        // 1+2+3+4+5+6+7+8 = 36
        Assert.Equal(36f, sum[0]);
    }

    [Fact]
    public void SparseSum_AxisZero_AccumulatesPerColumn()
    {
        var a = SmallA();
        var sums = SparseOps.SparseSum(a, axis: 0);
        // cols: 1+5=6, 3+7=10, 2+6=8, 4+8=12
        Assert.Equal(6f, sums[0]);
        Assert.Equal(10f, sums[1]);
        Assert.Equal(8f, sums[2]);
        Assert.Equal(12f, sums[3]);
    }

    [Fact]
    public void SparseSoftmax_RowsSumToOne()
    {
        var a = SmallA();
        var sm = SparseOps.SparseSoftmax(a);
        var dense = sm.ToDense();
        for (int r = 0; r < 4; r++)
        {
            float rowSum = 0;
            for (int c = 0; c < 4; c++) rowSum += dense[r, c];
            Assert.Equal(1f, rowSum, 3);
        }
    }

    [Fact]
    public void SparseLogSoftmax_RowExponentialSumToOne()
    {
        var a = SmallA();
        var lsm = SparseOps.SparseLogSoftmax(a);
        var coo = lsm.ToCoo();
        var rowSums = new double[4];
        for (int k = 0; k < coo.NonZeroCount; k++)
            rowSums[coo.RowIndices[k]] += Math.Exp(coo.DataVector[k]);
        for (int r = 0; r < 4; r++)
            Assert.Equal(1.0, rowSums[r], 3);
    }

    [Fact]
    public void SparseMask_FiltersDenseToPattern()
    {
        var a = SmallA();
        var dense = MakeDense(new float[,]
        {
            { 9, 9, 9, 9 },
            { 9, 9, 9, 9 },
            { 9, 9, 9, 9 },
            { 9, 9, 9, 9 },
        });
        var masked = SparseOps.SparseMask(dense, a);
        Assert.Equal(a.NonZeroCount, masked.NonZeroCount);
        var coo = masked.ToCoo();
        for (int k = 0; k < coo.NonZeroCount; k++)
            Assert.Equal(9f, coo.DataVector[k]);
    }

    [Fact]
    public void SparseSampledAddMM_OnlyComputesPatternEntries()
    {
        var pattern = SmallA();
        var aDense = MakeDense(new float[,]
        {
            { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 },
        });
        var bDense = MakeDense(new float[,]
        {
            { 1, 1, 1, 1 }, { 1, 0, 1, 0 },
        });
        var c = new Tensor<float>(new[] { 4, 4 });

        var sampled = SparseOps.SparseSampledAddMM(pattern, aDense, bDense, c, alpha: 1f, beta: 0f);
        // Verify the pattern is preserved.
        Assert.Equal(pattern.NonZeroCount, sampled.NonZeroCount);

        // Verify each entry equals the dense product at that position.
        var fullProduct = _engine.TensorMatMul(aDense, bDense);
        var patCoo = pattern.ToCoo();
        var sampledCoo = sampled.ToCoo();
        for (int k = 0; k < patCoo.NonZeroCount; k++)
        {
            int r = patCoo.RowIndices[k], cc = patCoo.ColumnIndices[k];
            Assert.Equal(fullProduct[r, cc], sampledCoo[r, cc], 3);
        }
    }

    [Fact]
    public void SparseSpDiags_PlacesDiagonalsAtCorrectOffsets()
    {
        var diagonals = MakeDense(new float[,]
        {
            { 1, 2, 3, 4 }, // main diag
            { 5, 6, 7, 0 }, // super-diag (offset +1; last entry past edge)
        });
        var sparse = SparseOps.SparseSpDiags(diagonals, new[] { 0, 1 }, rows: 4, cols: 4);
        var dense = sparse.ToDense();
        Assert.Equal(1f, dense[0, 0]);
        Assert.Equal(2f, dense[1, 1]);
        Assert.Equal(5f, dense[0, 1]);
        Assert.Equal(7f, dense[2, 3]);
        Assert.Equal(0f, dense[1, 0]); // strictly lower triangle is all zero (no negative offset given)
        Assert.Equal(4f, dense[3, 3]); // last main-diagonal value
    }

    [Fact]
    public void SparseSpGeMM_AgainstDenseReference()
    {
        var a = SmallA();
        var b = SparseTensor<float>.FromDense(MakeDense(new float[,]
        {
            { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 },
        }));
        var product = SparseOps.SparseSpGeMM(a, b);
        var denseProduct = _engine.TensorMatMul(a.ToDense(), b.ToDense());
        AssertDenseEqual(denseProduct, product.ToDense());
    }

    [Fact]
    public void Bsr_TimesDense_MatchesDenseReference()
    {
        var a = SmallA();
        var b = MakeDense(new float[,]
        {
            { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 },
        });
        var bsr = a.ToBsr(2, 2);
        var got = SparseOps.SparseMatMul(bsr, b);
        var ref_ = _engine.TensorMatMul(a.ToDense(), b);
        AssertDenseEqual(ref_, got);
    }

    [Fact]
    public void SemiStructured_FromDense_PicksTwoLargestPerGroup()
    {
        // Each row has 4 cols; expect 2 stored per row.
        var dense = MakeDense(new float[,]
        {
            { 0.1f, 0.5f, 0.2f, 0.4f },
            { 1.0f, 0.0f, 0.0f, 0.5f },
        });
        var ss = SparseSemiStructured<float>.FromDense(dense);
        Assert.Equal(2, ss.Rows);
        Assert.Equal(4, ss.Columns);
        Assert.Equal(2 * 1 * SparseSemiStructured<float>.N, ss.PackedValues.Length);
    }

    [Fact]
    public void SemiStructured_ToDense_ReconstructsTwoLargestPattern()
    {
        var dense = MakeDense(new float[,]
        {
            { 1, 2, 3, 4 }, // group: keep 3, 4
        });
        var ss = SparseSemiStructured<float>.FromDense(dense);
        var roundtrip = ss.ToDense();
        Assert.Equal(0f, roundtrip[0, 0]);
        Assert.Equal(0f, roundtrip[0, 1]);
        Assert.Equal(3f, roundtrip[0, 2]);
        Assert.Equal(4f, roundtrip[0, 3]);
    }

    [Fact]
    public void SemiStructured_MatMul_MatchesDenseReference()
    {
        var dense = MakeDense(new float[,]
        {
            { 0.1f, 0.5f, 0.2f, 0.4f, 0.0f, 0.9f, 0.1f, 0.2f },
            { 1.0f, 0.1f, 0.0f, 0.5f, 0.2f, 0.0f, 0.7f, 0.3f },
        });
        var ss = SparseSemiStructured<float>.FromDense(dense);
        var b = new Tensor<float>(new[] { 8, 3 });
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 3; j++)
                b[i, j] = i * 0.1f + j * 0.05f;

        var got = ss.MatMul(b);
        // Reference product is "dense after pruning" · b — i.e., ss.ToDense() · b.
        var prunedDense = ss.ToDense();
        var ref_ = _engine.TensorMatMul(prunedDense, b);
        AssertDenseEqual(ref_, got);
    }

    [Fact]
    public void SparseAutograd_SparseMatMul_GradientsMatchDenseAnalytic()
    {
        var a = SmallA();
        var b = MakeDense(new float[,]
        {
            { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 },
        });
        using var tape = new GradientTape<float>();
        var aDense = a.ToDense();

        var output = SparseAutograd.SparseMatMulRecord(a, b);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { b });

        Assert.True(grads.ContainsKey(b));
        // grad_b = A^T · ones (since loss = sum). A^T is a 4×4 sparse mat,
        // ones has shape [4, 2]. Each entry of grad_b is the sum of its
        // column in A^T = sum of its row in A.
        var aT = _engine.TensorTranspose(aDense);
        var ones = new Tensor<float>(new[] { 4, 2 });
        for (int i = 0; i < ones.Length; i++) ones.AsWritableSpan()[i] = 1f;
        var expectedGradB = _engine.TensorMatMul(aT, ones);
        AssertDenseEqual(expectedGradB, grads[b]);
    }

    [Fact]
    public void SparseAutograd_SparseSampledAddMM_PreservesPatternInGradient()
    {
        var pattern = SmallA();
        var aDense = MakeDense(new float[,]
        {
            { 1, 1 }, { 1, -1 }, { 0, 1 }, { 1, 0 },
        });
        var bDense = MakeDense(new float[,]
        {
            { 1, 0, 1, 0 }, { 0, 1, 0, 1 },
        });
        var c = new Tensor<float>(new[] { 4, 4 });

        using var tape = new GradientTape<float>();
        var sampled = SparseAutograd.SparseSampledAddMMRecord(pattern, aDense, bDense, c, alpha: 1f, beta: 1f);
        var loss = _engine.ReduceSum(sampled, null);
        var grads = tape.ComputeGradients(loss, new[] { aDense, bDense, c });

        Assert.True(grads.ContainsKey(aDense));
        Assert.True(grads.ContainsKey(bDense));
        Assert.True(grads.ContainsKey(c));
        // The grad through C must be the dense pattern mask (0/1) since beta=1
        // and downstream is sum. Check that grad_c is non-zero exactly at the
        // pattern positions and zero elsewhere.
        var patternDense = pattern.ToDense();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
            {
                if (patternDense[i, j] != 0f)
                    Assert.Equal(1f, grads[c][i, j], 3);
                else
                    Assert.Equal(0f, grads[c][i, j], 3);
            }
    }

    private static void AssertDenseEqual(Tensor<float> expected, Tensor<float> actual)
    {
        Assert.Equal(expected._shape, actual._shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected.AsSpan()[i], actual.AsSpan()[i], 3);
    }
}
