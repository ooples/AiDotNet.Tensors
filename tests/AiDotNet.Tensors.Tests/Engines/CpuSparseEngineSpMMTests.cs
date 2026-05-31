using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Verifies <see cref="CpuSparseEngine.SpMM{T}"/> over an <b>unconstrained</b> T —
/// the public API carries no struct/unmanaged constraint (so unconstrained-T
/// consumers such as the neural-network layers can call it). float/double take the
/// SIMD/parallel BLAS fast path; any other numeric T (here decimal) takes the
/// generic CSR scalar fallback. All must agree with a hand-computed reference.
///
/// Regression guard for the #379 constraint leak: SpMM had `where T : unmanaged`,
/// which broke SparseLinearLayer&lt;T&gt; (generic over an open T).
/// </summary>
public class CpuSparseEngineSpMMTests
{
    // A (2×3) with non-zeros (0,0)=2, (0,2)=3, (1,1)=4;  B (3×2) = [[1,2],[3,4],[5,6]].
    //   C = A·B = [[2·1+3·5, 2·2+3·6], [4·3, 4·4]] = [[17,22],[12,16]]
    private static readonly double[,] ExpectedC = { { 17, 22 }, { 12, 16 } };

    private static void CheckSpMM<T>(double tol)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var sparse = new SparseTensor<T>(
            2, 3,
            new[] { 0, 0, 1 },                                  // row indices
            new[] { 0, 2, 1 },                                  // col indices
            new[] { ops.FromDouble(2), ops.FromDouble(3), ops.FromDouble(4) });

        var dense = new Matrix<T>(3, 2);
        double[,] bv = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 2; c++)
                dense[r, c] = ops.FromDouble(bv[r, c]);

        var engine = new CpuSparseEngine();
        Matrix<T> result = engine.SpMM(sparse, dense);

        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 2; c++)
                Assert.True(Math.Abs(ops.ToDouble(result[r, c]) - ExpectedC[r, c]) <= tol,
                    $"[{r},{c}] expected {ExpectedC[r, c]}, got {ops.ToDouble(result[r, c])}");
    }

    [Fact]
    public void SpMM_Float_SimdFastPath_MatchesReference() => CheckSpMM<float>(1e-4);

    [Fact]
    public void SpMM_Double_SimdFastPath_MatchesReference() => CheckSpMM<double>(1e-9);

    [Fact]
    public void SpMM_Decimal_GenericFallback_MatchesReference() => CheckSpMM<decimal>(1e-9);
}
