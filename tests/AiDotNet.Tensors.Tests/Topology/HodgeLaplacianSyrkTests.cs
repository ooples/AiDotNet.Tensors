using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Topology;
using Xunit;

namespace AiDotNet.Tensors.Tests.Topology;

public class HodgeLaplacianSyrkTests
{
    // Independent oracle: L_k = B_kᵀ·B_k + B_{k+1}·B_{k+1}ᵀ, computed via the plain
    // Matrix Transpose/Multiply path. Guards the #379 SYRK rewire of HodgeLaplacian.
    private static Matrix<double> ReferenceHodge(SimplicialComplex sc, int k)
    {
        var kSimplices = sc.GetSimplices(k);
        var result = new Matrix<double>(kSimplices.Count, kSimplices.Count);
        if (k > 0)
        {
            var bK = sc.BoundaryOperator<double>(k);
            var term = (Matrix<double>)bK.Transpose().Multiply(bK);
            result = (Matrix<double>)result.Add(term);
        }
        if (k < sc.MaxDimension)
        {
            var bKPlus = sc.BoundaryOperator<double>(k + 1);
            if (bKPlus.Rows > 0 && bKPlus.Columns > 0)
            {
                var term = (Matrix<double>)bKPlus.Multiply(bKPlus.Transpose());
                result = (Matrix<double>)result.Add(term);
            }
        }
        return result;
    }

    private static SimplicialComplex BuildTriangle()
    {
        var sc = new SimplicialComplex();
        sc.AddSimplex(new Simplex(new[] { 0, 1, 2 }), includeFaces: true); // triangle + edges + vertices
        return sc;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void HodgeLaplacian_MatchesReference_FullSymmetric(int k)
    {
        var sc = BuildTriangle();
        var expected = ReferenceHodge(sc, k);
        var actual = sc.HodgeLaplacian<double>(k);

        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int i = 0; i < expected.Rows; i++)
            for (int j = 0; j < expected.Columns; j++)
                Assert.Equal(expected[i, j], actual[i, j], 10);
    }

    [Fact]
    public void HodgeLaplacian_IsSymmetric()
    {
        // SYRK writes one triangle then symmetrizes — the full result must be symmetric.
        var sc = BuildTriangle();
        var l = sc.HodgeLaplacian<double>(1);
        for (int i = 0; i < l.Rows; i++)
            for (int j = 0; j < l.Columns; j++)
                Assert.Equal(l[i, j], l[j, i], 12);
    }
}
