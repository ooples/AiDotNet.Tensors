using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Sums, dot products, matmuls, cumulative sums, norms and softmax denominators over Half and BFloat16 tensors must
/// accumulate in float32 and round once. Accumulating in the element type stalls: Half has an 11-bit significand, so
/// 2048 + 1 rounds back to 2048 and a sum of 4096 ones returns 2048 (BFloat16 stalls at 256). PyTorch 2.x accumulates
/// half-precision sum/mean/var/std/norm/cumsum/matmul/softmax in float32 (<c>at::opmath_type&lt;Half&gt;</c> on CPU,
/// fp32 accumulation in cuBLAS and the CUDA reduction kernels), and NumPy 1.26/2.x sums float16 pairwise in float32;
/// <c>torch.ones(4096, dtype=torch.half).sum()</c> is 4096 and <c>torch.ones(70000, dtype=torch.half).mean()</c> is 1.
/// Expected values are exact in the result type (4096 = 2^12, ln 4096, 2^-12, 64), or computed in double from the
/// rounded inputs and allowed one unit in the last place of the result type.
/// </summary>
public class LowPrecisionAccumulationTests
{
    private const int K = 4096;
    private const int N = 70_000;

    private readonly CpuEngine _engine = new CpuEngine();

    private static Half[] HalfFill(int n, double value)
    {
        var data = new Half[n];
        for (int i = 0; i < n; i++) data[i] = (Half)value;
        return data;
    }

    private static BFloat16[] BFloat16Fill(int n, double value)
    {
        var data = new BFloat16[n];
        for (int i = 0; i < n; i++) data[i] = (BFloat16)value;
        return data;
    }

    private static Half[] HalfData(int n, Func<int, double> f)
    {
        var data = new Half[n];
        for (int i = 0; i < n; i++) data[i] = (Half)f(i);
        return data;
    }

    private static double Signal(int i) => 1.0 + 0.5 * Math.Sin(0.37 * i);

    private static Tensor<Half> HalfOnes(params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        return new Tensor<Half>(HalfFill(n, 1.0), shape);
    }

    private static Tensor<BFloat16> BFloat16Ones(params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        return new Tensor<BFloat16>(BFloat16Fill(n, 1.0), shape);
    }

    private static void AssertAllHalf(double expected, Half[] actual, string what)
    {
        Assert.NotEmpty(actual);
        for (int i = 0; i < actual.Length; i++)
            Assert.True((double)actual[i] == expected, $"{what}[{i}]: got {(double)actual[i]}, expected {expected}");
    }

    private static void AssertAllBFloat16(double expected, BFloat16[] actual, string what)
    {
        Assert.NotEmpty(actual);
        for (int i = 0; i < actual.Length; i++)
            Assert.True((double)actual[i] == expected, $"{what}[{i}]: got {(double)actual[i]}, expected {expected}");
    }

    private static void AssertHalfUlp(double expected, Half actual, string what)
    {
        double a = (double)actual;
        double ulp = Math.Max(Math.Pow(2, -24),
            expected == 0 ? 0 : Math.Pow(2, Math.Floor(Math.Log(Math.Abs(expected), 2)) - 10));
        Assert.True(!double.IsNaN(a) && Math.Abs(a - expected) <= ulp,
            $"{what}: got {a}, expected {expected} (one ulp = {ulp})");
    }

    // ---- Sums ----

    [Fact]
    public void TensorSum_Half_4096Ones_Is4096()
    {
        Assert.Equal(4096.0, (double)_engine.TensorSum(HalfOnes(K)));
    }

    [Fact]
    public void TensorSum_BFloat16_4096Ones_Is4096()
    {
        Assert.Equal(4096.0, (double)_engine.TensorSum(BFloat16Ones(K)));
    }

    [Fact]
    public void VectorSum_Half_4096Ones_Is4096()
    {
        Assert.Equal(4096.0, (double)_engine.Sum(new Vector<Half>(HalfFill(K, 1.0))));
    }

    [Fact]
    public void ReduceSum_Half_AllAxes_LastAxis_LeadingAxis_Are4096()
    {
        AssertAllHalf(K, _engine.ReduceSum(HalfOnes(K), null, false).ToArray(), "ReduceSum all");
        AssertAllHalf(K, _engine.ReduceSum(HalfOnes(3, K), new[] { 1 }, false).ToArray(), "ReduceSum last axis");
        AssertAllHalf(K, _engine.ReduceSum(HalfOnes(K, 3), new[] { 0 }, true).ToArray(), "ReduceSum leading axis");
        AssertAllHalf(K, _engine.ReduceSum(HalfOnes(2, 64, 64), new[] { 1, 2 }, false).ToArray(), "ReduceSum two axes");
    }

    [Fact]
    public void ReduceSum_BFloat16_LastAxis_Is4096()
    {
        AssertAllBFloat16(K, _engine.ReduceSum(BFloat16Ones(3, K), new[] { 1 }, false).ToArray(), "ReduceSum last axis");
    }

    [Fact]
    public void TensorSumOfSquares_Half_4096Ones_Is4096()
    {
        Assert.Equal(4096.0, (double)_engine.TensorSumOfSquares(HalfOnes(K)));
    }

    // ---- Means, variances ----

    [Fact]
    public void Means_Half_70000Ones_AreExactlyOne()
    {
        var ones = HalfOnes(N);
        Assert.Equal(1.0, (double)_engine.TensorMean(ones));
        Assert.Equal(1.0, (double)_engine.Mean(new Vector<Half>(HalfFill(N, 1.0))));
        AssertAllHalf(1.0, _engine.ReduceMean(ones, new[] { 0 }, false).ToArray(), "ReduceMean");
        AssertAllHalf(1.0, _engine.ReduceMean(HalfOnes(2, N), new[] { 1 }, false).ToArray(), "ReduceMean rows");
    }

    [Fact]
    public void TensorNanSum_Half_4096Ones_Is4096()
    {
        AssertAllHalf(K, _engine.TensorNanSum(HalfOnes(K)).ToArray(), "TensorNanSum");
    }

    [Fact]
    public void TensorVarAndStd_Half_Over70000Elements_MatchDoubleReference()
    {
        var data = HalfData(N, Signal);
        double mean = 0;
        foreach (var h in data) mean += (double)h;
        mean /= N;
        double ss = 0;
        foreach (var h in data) { double d = (double)h - mean; ss += d * d; }
        // torch.var / torch.std default to the unbiased (N - 1) estimator; the engine's TensorVar documents its own
        // correction, so the reference is taken from the float64 engine on the same rounded inputs.
        var asDouble = new double[N];
        for (int i = 0; i < N; i++) asDouble[i] = (double)data[i];
        var dEngine = _engine.TensorVar(new Tensor<double>(asDouble, new[] { N })).ToArray()[0];
        var dStd = _engine.TensorStd(new Tensor<double>(asDouble, new[] { N })).ToArray()[0];
        Assert.True(Math.Abs(dEngine - ss / N) < 1e-9 || Math.Abs(dEngine - ss / (N - 1)) < 1e-9,
            "float64 TensorVar must be a population or unbiased variance");

        var t = new Tensor<Half>(data, new[] { N });
        AssertHalfUlp(dEngine, _engine.TensorVar(t).ToArray()[0], "TensorVar");
        AssertHalfUlp(dStd, _engine.TensorStd(t).ToArray()[0], "TensorStd");
    }

    // ---- Norms ----

    [Fact]
    public void Norms_Half_4096Ones_Are64()
    {
        Assert.Equal(64.0, (double)_engine.Norm(new Vector<Half>(HalfFill(K, 1.0))));
        AssertAllHalf(64.0, _engine.TensorNorm(HalfOnes(K)).ToArray(), "TensorNorm()");
        AssertAllHalf(64.0, _engine.TensorNorm(HalfOnes(3, K), 1, false).ToArray(), "TensorNorm(axis 1)");
        AssertAllHalf(64.0, _engine.TensorNorm(HalfOnes(K, 3), 0, false).ToArray(), "TensorNorm(axis 0)");
    }

    // ---- Cumulative sum ----

    [Fact]
    public void TensorCumSum_Half_4096Ones_CountsPast2048()
    {
        var r = _engine.TensorCumSum(HalfOnes(K), 0).ToArray();
        Assert.Equal(K, r.Length);
        for (int i = 0; i < K; i++)
            Assert.True(r[i].Equals((Half)(i + 1)), $"cumsum[{i}]: got {(double)r[i]}, expected {(double)(Half)(i + 1)}");

        var rows = _engine.TensorCumSum(HalfOnes(2, K), 1).ToArray();
        Assert.Equal(4096.0, (double)rows[K - 1]);
        Assert.Equal(4096.0, (double)rows[2 * K - 1]);
        var cols = _engine.TensorCumSum(HalfOnes(K, 2), 0).ToArray();
        Assert.Equal(4096.0, (double)cols[2 * K - 2]);
        Assert.Equal(4096.0, (double)cols[2 * K - 1]);
    }

    // ---- Matmul and dot products ----

    [Fact]
    public void TensorMatMul_Half_InnerDim4096OfOnes_Is4096()
    {
        AssertAllHalf(K, _engine.TensorMatMul(HalfOnes(2, K), HalfOnes(K, 3)).ToArray(), "TensorMatMul 2D");
        AssertAllHalf(K, _engine.TensorMatMul(HalfOnes(2, 2, K), HalfOnes(2, K, 3)).ToArray(), "TensorMatMul batched");
    }

    [Fact]
    public void TensorMatMul_BFloat16_InnerDim4096OfOnes_Is4096()
    {
        AssertAllBFloat16(K, _engine.TensorMatMul(BFloat16Ones(2, K), BFloat16Ones(K, 3)).ToArray(), "TensorMatMul 2D");
    }

    [Fact]
    public void TensorMatMul_Half_BelowGemmThreshold_InnerDim3000OfOnes_Is3000()
    {
        // m*k*n = 3000 is under the BLAS work threshold, so this takes the blocked fallback (and on net471 every
        // Half matmul does). 3000 is exact in Half (spacing 2 in [2048, 4096)).
        AssertAllHalf(3000, _engine.TensorMatMul(HalfOnes(1, 3000), HalfOnes(3000, 1)).ToArray(), "TensorMatMul small");
    }

    [Fact]
    public void BatchMatMul_Half_TransposedViews_InnerDim4096OfOnes_Are4096()
    {
        var a = HalfOnes(K, 2).Transpose(new[] { 1, 0 });
        var b = HalfOnes(3, K).Transpose(new[] { 1, 0 });
        AssertAllHalf(K, _engine.BatchMatMul(a, b).ToArray(), "BatchMatMul strided");
    }

    [Fact]
    public void ReduceSum_Half_TransposedView_Is4096()
    {
        var view = HalfOnes(K, 3).Transpose(new[] { 1, 0 });
        AssertAllHalf(K, _engine.ReduceSum(view, new[] { 1 }, false).ToArray(), "ReduceSum transposed view");
    }

    [Fact]
    public void BatchMatMul_And_TensorBatchMatMul_Half_InnerDim4096OfOnes_Are4096()
    {
        AssertAllHalf(K, _engine.BatchMatMul(HalfOnes(2, 2, K), HalfOnes(2, K, 3)).ToArray(), "BatchMatMul");
        AssertAllHalf(K, _engine.TensorBatchMatMul(HalfOnes(2, 2, K), HalfOnes(2, K, 3)).ToArray(), "TensorBatchMatMul");
    }

    [Fact]
    public void TensorMatMulTransposed_And_MatMulInto_Half_InnerDim4096OfOnes_Are4096()
    {
        AssertAllHalf(K, _engine.TensorMatMulTransposed(HalfOnes(2, K), HalfOnes(3, K)).ToArray(), "TensorMatMulTransposed");
        var destination = new Tensor<Half>(new[] { 2, 3 });
        _engine.MatMulInto(destination, HalfOnes(2, K), HalfOnes(K, 3));
        AssertAllHalf(K, destination.ToArray(), "MatMulInto");
    }

    [Fact]
    public void MatrixMultiply_And_MatrixVectorMultiply_Half_InnerDim4096OfOnes_Are4096()
    {
        var a = new Matrix<Half>(2, K);
        var b = new Matrix<Half>(K, 3);
        for (int i = 0; i < 2; i++) for (int k = 0; k < K; k++) a[i, k] = (Half)1;
        for (int k = 0; k < K; k++) for (int j = 0; j < 3; j++) b[k, j] = (Half)1;
        var c = _engine.MatrixMultiply(a, b);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
            Assert.True((double)c[i, j] == K, $"MatrixMultiply[{i},{j}]: got {(double)c[i, j]}");
        var mv = _engine.MatrixVectorMultiply(a, new Vector<Half>(HalfFill(K, 1.0)));
        for (int i = 0; i < 2; i++) Assert.True((double)mv[i] == K, $"MatrixVectorMultiply[{i}]: got {(double)mv[i]}");
    }

    [Fact]
    public void DotProducts_Half_4096Ones_Are4096()
    {
        var v = new Vector<Half>(HalfFill(K, 1.0));
        Assert.Equal(4096.0, (double)_engine.DotProduct(v, v));
        Assert.Equal(4096.0, (double)_engine.DotProduct(v, v, 0, 1));
        Assert.Equal(4096.0, (double)_engine.TensorVecDot(HalfOnes(K), HalfOnes(K)));
        AssertAllHalf(K, _engine.TensorDot(HalfOnes(2, K), HalfOnes(K, 3), new[] { 1 }, new[] { 0 }).ToArray(), "TensorDot");
    }

    // ---- Softmax and log-sum-exp denominators ----

    [Fact]
    public void Softmax_Half_4096EqualLogits_Are2PowMinus12()
    {
        double expected = 1.0 / K;
        AssertAllHalf(expected, _engine.Softmax(new Vector<Half>(HalfFill(K, 0.0))).ToArray(), "Softmax(Vector)");
        AssertAllHalf(expected, _engine.Softmax(HalfOnes(2, K), -1).ToArray(), "Softmax(axis -1)");
        AssertAllHalf(expected, _engine.Softmax(HalfOnes(K, 2), 0).ToArray(), "Softmax(axis 0)");
        AssertAllHalf(expected, _engine.TensorSoftmax(HalfOnes(2, K), 1).ToArray(), "TensorSoftmax");
        AssertAllHalf(expected, _engine.TensorSoftmaxRows(HalfOnes(2, K)).ToArray(), "TensorSoftmaxRows");
        var destination = new Tensor<Half>(new[] { 2, K });
        _engine.SoftmaxInto(destination, HalfOnes(2, K), 1);
        AssertAllHalf(expected, destination.ToArray(), "SoftmaxInto");
    }

    [Fact]
    public void LogSoftmax_Half_4096EqualLogits_AreMinusLn4096()
    {
        double expected = (double)(Half)(-Math.Log(K));
        AssertAllHalf(expected, _engine.TensorLogSoftmax(HalfOnes(2, K), 1).ToArray(), "TensorLogSoftmax");
        AssertAllHalf(expected, _engine.TensorLogSoftmax(HalfOnes(K, 2), 0).ToArray(), "TensorLogSoftmax axis 0");
        var destination = new Tensor<Half>(new[] { 2, K });
        _engine.LogSoftmaxInto(destination, HalfOnes(2, K), 1);
        AssertAllHalf(expected, destination.ToArray(), "LogSoftmaxInto");
    }

    [Fact]
    public void LogSumExp_Half_4096Zeros_IsLn4096()
    {
        double expected = (double)(Half)Math.Log(K);
        var zeros = new Tensor<Half>(HalfFill(2 * K, 0.0), new[] { 2, K });
        AssertAllHalf(expected, _engine.TensorLogSumExp(zeros, 1, false).ToArray(), "TensorLogSumExp(axis)");
        AssertAllHalf(expected, _engine.TensorLogSumExp(new Tensor<Half>(HalfFill(K, 0.0), new[] { K })).ToArray(),
            "TensorLogSumExp()");
    }
}
