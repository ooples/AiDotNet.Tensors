using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

public class GenericLinalgAdversarialTests
{
    [Fact]
    public void DecimalQr_LargeReflectors_ReconstructWithoutSquaringTheScale()
    {
        const decimal scale = 100000000000000000000m;
        var values = new[] { 1m, 2m, 3m, 5m, 7m, 11m };
        var data = Array.ConvertAll(values, x => x * scale);
        var (q, r) = Linalg.QR(new Tensor<decimal>(data, new[] { 3, 2 }));
        var qa = q.ToArray(); var ra = r.ToArray();
        int k = q.Shape[1];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
            {
                decimal reconstructed = 0;
                for (int t = 0; t < k; t++) reconstructed += qa[i * k + t] * (ra[t * 2 + j] / scale);
                Assert.True(Math.Abs(reconstructed - values[i * 2 + j]) < 1e-6m);
            }
    }

    [Fact]
    public void HalfQr_LargeReflectors_ReconstructFiniteInput()
    {
        var data = Array.ConvertAll(new[] { 1000f, 2000f, 3000f, 5000f, 7000f, 11000f }, x => (Half)x);
        var (q, r) = Linalg.QR(new Tensor<Half>(data, new[] { 3, 2 }));
        var qa = q.ToArray(); var ra = r.ToArray();
        int k = q.Shape[1];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
            {
                double reconstructed = 0;
                for (int t = 0; t < k; t++) reconstructed += (double)qa[i * k + t] * (double)ra[t * 2 + j];
                Assert.True(Math.Abs(reconstructed - (double)data[i * 2 + j]) < 40);
            }
    }

    [Fact]
    public void DecimalEig_Identity_ReturnsIndependentUnitEigenvectors()
    {
        var a = new[] { 1m, 0m, 0m, 0m, 1m, 0m, 0m, 0m, 1m };
        var (w, v) = Linalg.Eig(new Tensor<decimal>(a, new[] { 3, 3 }));
        AssertEigenResidual(a, w.ToArray(), v.ToArray(), 3, 1e-21m);
        AssertOrthonormal(v.ToArray(), 3, 1e-21m);
    }

    [Fact]
    public void DecimalEig_SeedOrthogonalToAnEigenspace_FindsBothEigenvalues()
    {
        var a = new[] { 1m, -1m, -1m, 1m };
        var (w, v) = Linalg.Eig(new Tensor<decimal>(a, new[] { 2, 2 }));
        AssertEigenResidual(a, w.ToArray(), v.ToArray(), 2, 1e-21m);
        AssertOrthonormal(v.ToArray(), 2, 1e-21m);
    }

    [Fact]
    public void DecimalEig_RepeatedEigenvalueInNonSymmetricMatrix_PreservesItsEigenspace()
    {
        var a = new[] { 1m, 0m, 1m, 0m, 1m, 2m, 0m, 0m, 2m };
        var (w, v) = Linalg.Eig(new Tensor<decimal>(a, new[] { 3, 3 }));
        var wa = w.ToArray(); var va = v.ToArray();
        AssertEigenResidual(a, wa, va, 3, 1e-21m);
        decimal dot = 0;
        for (int i = 0; i < 3; i++) dot += va[(i * 3) * 2] * va[(i * 3 + 1) * 2];
        Assert.True(Math.Abs(dot) < 1e-21m);
    }

    private static void AssertOrthonormal(decimal[] v, int n, decimal tol)
    {
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
            {
                decimal dot = 0;
                for (int i = 0; i < n; i++)
                    dot += v[(i * n + j) * 2] * v[(i * n + k) * 2]
                        + v[(i * n + j) * 2 + 1] * v[(i * n + k) * 2 + 1];
                Assert.True(Math.Abs(dot - (j == k ? 1m : 0m)) < tol, $"columns {j},{k}: {dot}");
            }
    }

    [Fact]
    public void DecimalEig_RepeatedComplexEigenvalues_HaveIndependentVectors()
    {
        var a = new[] { 0m, -1m, 0m, 0m, 1m, 0m, 0m, 0m,
            0m, 0m, 0m, -1m, 0m, 0m, 1m, 0m };
        var (w, v) = Linalg.Eig(new Tensor<decimal>(a, new[] { 4, 4 }));
        AssertEigenResidual(a, w.ToArray(), v.ToArray(), 4, 1e-21m);
        AssertOrthonormal(v.ToArray(), 4, 1e-21m);
    }

    [Fact]
    public void DecimalEig_DefectiveRepeatedEigenvalue_ReturnsEigenvectorsInsteadOfOrthogonalNonEigenvectors()
    {
        var a = new[] { 1m, 1m, 0m, 1m };
        var (w, v) = Linalg.Eig(new Tensor<decimal>(a, new[] { 2, 2 }));
        // A defective Jordan block has only one eigendirection, and is ill-conditioned under inverse iteration.
        AssertEigenResidual(a, w.ToArray(), v.ToArray(), 2, 1e-12m);
    }

    private static void AssertEigenResidual(decimal[] a, decimal[] w, decimal[] v, int n, decimal tol)
    {
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
            {
                decimal rr = 0, ri = 0;
                for (int j = 0; j < n; j++)
                {
                    rr += a[i * n + j] * v[(j * n + k) * 2];
                    ri += a[i * n + j] * v[(j * n + k) * 2 + 1];
                }
                decimal xr = v[(i * n + k) * 2], xi = v[(i * n + k) * 2 + 1];
                Assert.True(Math.Abs(rr - (w[k * 2] * xr - w[k * 2 + 1] * xi)) < tol);
                Assert.True(Math.Abs(ri - (w[k * 2] * xi + w[k * 2 + 1] * xr)) < tol);
            }
    }
}
