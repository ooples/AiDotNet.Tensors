using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

[Collection("EngineCurrentGlobalState")]
public sealed class DirectGpuLinalgRoutingTests
{
    [Fact]
    public void PublicFloatRoutes_FallBackWhenDirectGpuHasNoBackend()
    {
        IEngine previous = AiDotNetEngine.Current;
        using var fallbackEngine = new DirectGpuTensorEngine(null!);
        try
        {
            AiDotNetEngine.Current = fallbackEngine;
            Tensor<float> matrix = PositiveDefinite4x4();
            Tensor<float> rhs = Vector(1, 2, 3, 4);

            var (lowerEigenvalues, lowerEigenvectors) = Linalg.Eigh(matrix, upper: false);
            var (upperEigenvalues, upperEigenvectors) = Linalg.Eigh(matrix, upper: true);
            Tensor<float> valuesOnly = Linalg.Eigvalsh(matrix);
            Assert.Equal(new[] { 4 }, lowerEigenvalues.Shape.ToArray());
            Assert.Equal(new[] { 4, 4 }, lowerEigenvectors.Shape.ToArray());
            Assert.Equal(lowerEigenvalues.GetDataArray(), upperEigenvalues.GetDataArray());
            Assert.Equal(lowerEigenvectors.GetDataArray(), upperEigenvectors.GetDataArray());
            Assert.Equal(lowerEigenvalues.GetDataArray(), valuesOnly.GetDataArray());

            foreach (string mode in new[] { "reduced", "complete", "r" })
            {
                var (q, r) = Linalg.QR(matrix, mode);
                Assert.Equal(new[] { 4, 4 }, r.Shape.ToArray());
                if (mode != "r") Assert.Equal(new[] { 4, 4 }, q.Shape.ToArray());
            }

            var (lu, pivots) = Linalg.LuFactor(matrix);
            AssertSolves(matrix, Linalg.LuSolve(lu, pivots, rhs), rhs);

            var (ld, ldlPivots) = Linalg.LdlFactor(matrix);
            AssertSolves(matrix, Linalg.LdlSolve(ld, ldlPivots, rhs), rhs);

            var (u, singularValues, vh) = Linalg.Svd(matrix, fullMatrices: false);
            Assert.Equal(new[] { 4, 4 }, u.Shape.ToArray());
            Assert.Equal(new[] { 4 }, singularValues.Shape.ToArray());
            Assert.Equal(new[] { 4, 4 }, vh.Shape.ToArray());
            Assert.Equal(singularValues.GetDataArray(), Linalg.SvdVals(matrix).GetDataArray());

            var (lowRankU, lowRankS, lowRankVh) = Linalg.SvdLowRank(matrix, rank: 2);
            Assert.Equal(new[] { 4, 2 }, lowRankU.Shape.ToArray());
            Assert.Equal(new[] { 2 }, lowRankS.Shape.ToArray());
            Assert.Equal(new[] { 2, 4 }, lowRankVh.Shape.ToArray());
            Assert.Throws<ArgumentException>(() => Linalg.SvdLowRank(matrix, rank: 0));

            Tensor<float> solution = Linalg.Solve(matrix, rhs);
            AssertSolves(matrix, solution, rhs);
            var (solutionEx, info) = Linalg.SolveEx(matrix, rhs);
            AssertSolves(matrix, solutionEx, rhs);
            Assert.All(info.GetDataArray(), value => Assert.Equal(0, value));

            Tensor<float> upper = Matrix(
                2, 1, 0, 0,
                0, 3, 1, 0,
                0, 0, 4, 1,
                0, 0, 0, 5);
            AssertSolves(upper, Linalg.SolveTriangular(upper, rhs, upper: true), rhs);

            var leastSquares = Linalg.Lstsq(matrix, rhs, driver: "gelsd");
            AssertSolves(matrix, leastSquares.Solution, rhs);
            Assert.Equal(4, leastSquares.Rank.GetDataArray()[0]);
            Assert.Throws<ArgumentException>(() => Linalg.Lstsq(matrix, rhs, driver: "invalid"));
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [Fact]
    public void TruncateSvd4x4_CopiesTopComponentsForEveryBatch()
    {
        var fullU = new Tensor<float>(Enumerable.Range(0, 32).Select(i => (float)i).ToArray(),
            new[] { 2, 4, 4 });
        var fullS = new Tensor<float>(Enumerable.Range(100, 8).Select(i => (float)i).ToArray(),
            new[] { 2, 4 });
        var fullVh = new Tensor<float>(Enumerable.Range(200, 32).Select(i => (float)i).ToArray(),
            new[] { 2, 4, 4 });

        var (u, s, vh) = Linalg.TruncateSvd4x4(fullU, fullS, fullVh, 2);

        Assert.Equal(new[] { 2, 4, 2 }, u.Shape.ToArray());
        Assert.Equal(new[] { 2, 2 }, s.Shape.ToArray());
        Assert.Equal(new[] { 2, 2, 4 }, vh.Shape.ToArray());
        Assert.Equal(new float[] { 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29 },
            u.GetDataArray());
        Assert.Equal(new float[] { 100, 101, 104, 105 }, s.GetDataArray());
        Assert.Equal(Enumerable.Range(200, 8).Concat(Enumerable.Range(216, 8)).Select(i => (float)i),
            vh.GetDataArray());
    }

    [Fact]
    public void BackwardFloatRoutes_FallBackToTheManagedGradients()
    {
        IEngine previous = AiDotNetEngine.Current;
        using var fallbackEngine = new DirectGpuTensorEngine(null!);
        try
        {
            Tensor<float> matrix = PositiveDefinite4x4();
            Tensor<float> rhs = Vector(1, 2, 3, 4);

            AiDotNetEngine.Current = new CpuEngine();
            var expectedSolve = SolveBackward(matrix, rhs);
            var expectedCholesky = CholeskyBackward(matrix);

            AiDotNetEngine.Current = fallbackEngine;
            var actualSolve = SolveBackward(matrix, rhs);
            var actualCholesky = CholeskyBackward(matrix);

            AssertClose(expectedSolve.GradA, actualSolve.GradA);
            AssertClose(expectedSolve.GradB, actualSolve.GradB);
            AssertClose(expectedCholesky, actualCholesky);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private static (Tensor<float> GradA, Tensor<float> GradB) SolveBackward(
        Tensor<float> matrix, Tensor<float> rhs)
    {
        Tensor<float> solution = Linalg.Solve(matrix, rhs);
        var gradOutput = Vector(1, 1, 1, 1);
        var gradients = new Dictionary<Tensor<float>, Tensor<float>>();
        LinalgBackward.SolveBackward<float>()(
            gradOutput, new[] { matrix, rhs }, solution, Array.Empty<object>(),
            new CpuEngine(), gradients);
        return (gradients[matrix], gradients[rhs]);
    }

    private static Tensor<float> CholeskyBackward(Tensor<float> matrix)
    {
        Tensor<float> factor = Linalg.Cholesky(matrix);
        var gradOutput = new Tensor<float>(Enumerable.Repeat(1f, 16).ToArray(), new[] { 4, 4 });
        var gradients = new Dictionary<Tensor<float>, Tensor<float>>();
        LinalgBackward.CholeskyBackward<float>()(
            gradOutput, new[] { matrix }, factor, new object[] { false },
            new CpuEngine(), gradients);
        return gradients[matrix];
    }

    private static Tensor<float> PositiveDefinite4x4() => Matrix(
        4, 1, 0, 0,
        1, 3, 0, 0,
        0, 0, 2, 0.5f,
        0, 0, 0.5f, 1);

    private static Tensor<float> Matrix(params float[] values) =>
        new(values, new[] { 4, 4 });

    private static Tensor<float> Vector(params float[] values) =>
        new(values, new[] { values.Length });

    private static void AssertSolves(
        Tensor<float> matrix, Tensor<float> solution, Tensor<float> rhs)
    {
        float[] a = matrix.GetDataArray();
        float[] x = solution.GetDataArray();
        float[] b = rhs.GetDataArray();
        for (int row = 0; row < 4; row++)
        {
            float value = 0;
            for (int col = 0; col < 4; col++) value += a[row * 4 + col] * x[col];
            Assert.InRange(Math.Abs(value - b[row]), 0, 1e-4f);
        }
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        float[] expectedData = expected.GetDataArray();
        float[] actualData = actual.GetDataArray();
        for (int i = 0; i < expectedData.Length; i++)
            Assert.InRange(Math.Abs(expectedData[i] - actualData[i]), 0, 1e-5f);
    }
}
