using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Mean-type reductions over more than 65,504 Half elements (a few seconds of 16 kHz audio). Half's largest finite
/// value is 65,504, so putting the element count into T turns the divisor into +Inf (mean 0) and accumulating in T
/// overflows or stalls. PyTorch (2.x, <c>at::opmath_type&lt;Half&gt;</c> / <c>acc_type</c>) accumulates Half and
/// BFloat16 in float32, divides in float32 and rounds once to the storage type; these tests hold the engine to that.
/// <para>Reference values are computed in double from the already-rounded low-precision inputs, so the only error
/// allowed is the final rounding: one unit in the last place of the result type (10 mantissa bits for Half, 7 for
/// BFloat16; Half subnormal spacing 2^-24 below its smallest normal).</para>
/// </summary>
public class LowPrecisionLongReductionTests
{
    private const int N = 70_000;

    private readonly CpuEngine _engine = new CpuEngine();

    private static double Signal(int i) => 1.0 + 0.5 * Math.Sin(0.37 * i);
    private static double Target(int i) => 0.25 * Math.Cos(0.11 * i);

    private static Half[] HalfData(int n, Func<int, double> f)
    {
        var data = new Half[n];
        for (int i = 0; i < n; i++) data[i] = (Half)f(i);
        return data;
    }

    private static BFloat16[] BFloat16Data(int n, Func<int, double> f)
    {
        var data = new BFloat16[n];
        for (int i = 0; i < n; i++) data[i] = (BFloat16)f(i);
        return data;
    }

    private static double[] ToDouble(Half[] data)
    {
        var d = new double[data.Length];
        for (int i = 0; i < data.Length; i++) d[i] = (double)data[i];
        return d;
    }

    private static double[] ToDouble(BFloat16[] data)
    {
        var d = new double[data.Length];
        for (int i = 0; i < data.Length; i++) d[i] = (double)data[i];
        return d;
    }

    private static double Mean(double[] x, int start, int count, int stride)
    {
        double s = 0;
        for (int k = 0; k < count; k++) s += x[start + k * stride];
        return s / count;
    }

    private static double SumSqDev(double[] x, int start, int count, int stride)
    {
        double m = Mean(x, start, count, stride);
        double s = 0;
        for (int k = 0; k < count; k++) { double d = x[start + k * stride] - m; s += d * d; }
        return s;
    }

    private static void AssertHalf(double expected, Half actual, string what) =>
        AssertUlp(expected, (double)actual, 10, Math.Pow(2, -24), what);

    private static void AssertBFloat16(double expected, BFloat16 actual, string what) =>
        AssertUlp(expected, (double)actual, 7, 0.0, what);

    private static void AssertUlp(double expected, double actual, int mantissaBits, double minUlp, string what)
    {
        Assert.False(double.IsNaN(actual) || double.IsInfinity(actual), $"{what}: got {actual}, expected {expected}");
        double ulp = expected == 0 ? minUlp : Math.Pow(2, Math.Floor(Math.Log(Math.Abs(expected), 2)) - mantissaBits);
        ulp = Math.Max(ulp, minUlp);
        Assert.True(Math.Abs(actual - expected) <= ulp,
            $"{what}: got {actual}, expected {expected} (one ulp = {ulp})");
    }

    [Fact]
    public void ReduceMean_Half_FullReduction_Over70000Elements()
    {
        var data = HalfData(N, Signal);
        var x = ToDouble(data);
        var result = _engine.ReduceMean(new Tensor<Half>(data, new[] { N }), new[] { 0 }, keepDims: false);
        AssertHalf(Mean(x, 0, N, 1), result.ToArray()[0], "ReduceMean full");
    }

    [Fact]
    public void ReduceMean_Half_LastAxis_Over70000Elements()
    {
        var data = HalfData(2 * N, i => Signal(i) + (i >= N ? 2.0 : 0.0));
        var x = ToDouble(data);
        var result = _engine.ReduceMean(new Tensor<Half>(data, new[] { 2, N }), new[] { 1 }, keepDims: false).ToArray();
        Assert.Equal(2, result.Length);
        AssertHalf(Mean(x, 0, N, 1), result[0], "ReduceMean row 0");
        AssertHalf(Mean(x, N, N, 1), result[1], "ReduceMean row 1");
    }

    [Fact]
    public void ReduceMean_Half_LeadingAxis_Over70000Elements()
    {
        var data = HalfData(2 * N, i => Signal(i) + (i % 2 == 1 ? 2.0 : 0.0));
        var x = ToDouble(data);
        var result = _engine.ReduceMean(new Tensor<Half>(data, new[] { N, 2 }), new[] { 0 }, keepDims: true);
        Assert.Equal(new[] { 1, 2 }, result.Shape.ToArray());
        var r = result.ToArray();
        AssertHalf(Mean(x, 0, N, 2), r[0], "ReduceMean column 0");
        AssertHalf(Mean(x, 1, N, 2), r[1], "ReduceMean column 1");
    }

    [Fact]
    public void ReduceMean_Half_TransposedView_Over70000Elements()
    {
        var data = HalfData(2 * N, i => Signal(i) + (i >= N ? 2.0 : 0.0));
        var x = ToDouble(data);
        var view = new Tensor<Half>(data, new[] { 2, N }).Transpose(new[] { 1, 0 });
        var r = _engine.ReduceMean(view, new[] { 0 }, keepDims: false).ToArray();
        AssertHalf(Mean(x, 0, N, 1), r[0], "ReduceMean transposed 0");
        AssertHalf(Mean(x, N, N, 1), r[1], "ReduceMean transposed 1");
    }

    [Fact]
    public void ReduceVariance_Half_FullAndAxis_Over70000Elements()
    {
        var data = HalfData(2 * N, i => Signal(i) * (i >= N ? 2.0 : 1.0));
        var x = ToDouble(data);
        var full = _engine.ReduceVariance(new Tensor<Half>(data, new[] { 2 * N }), new[] { 0 }, keepDims: false);
        AssertHalf(SumSqDev(x, 0, 2 * N, 1) / (2 * N), full.ToArray()[0], "ReduceVariance full");

        var rows = _engine.ReduceVariance(new Tensor<Half>(data, new[] { 2, N }), new[] { 1 }, keepDims: false).ToArray();
        AssertHalf(SumSqDev(x, 0, N, 1) / N, rows[0], "ReduceVariance row 0");
        AssertHalf(SumSqDev(x, N, N, 1) / N, rows[1], "ReduceVariance row 1");
    }

    [Fact]
    public void ReduceVariance_Half_TransposedView_Over70000Elements()
    {
        var data = HalfData(2 * N, i => Signal(i) * (i >= N ? 2.0 : 1.0));
        var x = ToDouble(data);
        var view = new Tensor<Half>(data, new[] { 2, N }).Transpose(new[] { 1, 0 });
        var r = _engine.ReduceVariance(view, new[] { 0 }, keepDims: false).ToArray();
        AssertHalf(SumSqDev(x, 0, N, 1) / N, r[0], "ReduceVariance transposed 0");
        AssertHalf(SumSqDev(x, N, N, 1) / N, r[1], "ReduceVariance transposed 1");
    }

    [Fact]
    public void ReduceStd_Half_Over70000Elements()
    {
        var data = HalfData(N, Signal);
        var x = ToDouble(data);
        var r = _engine.ReduceStd(new Tensor<Half>(data, new[] { N }), new[] { 0 }, keepDims: false);
        AssertHalf(Math.Sqrt(SumSqDev(x, 0, N, 1) / N), r.ToArray()[0], "ReduceStd");
    }

    [Fact]
    public void TensorMean_And_TensorMeanDiff_Half_Over70000Elements()
    {
        var data = HalfData(N, Signal);
        var x = ToDouble(data);
        var t = new Tensor<Half>(data, new[] { N });
        AssertHalf(Mean(x, 0, N, 1), _engine.TensorMean(t), "TensorMean");
        AssertHalf(Mean(x, 0, N, 1), _engine.TensorMeanDiff(t).ToArray()[0], "TensorMeanDiff");
    }

    [Fact]
    public void TensorVarMean_Half_Over70000Elements()
    {
        var data = HalfData(N, Signal);
        var x = ToDouble(data);
        var (variance, mean) = _engine.TensorVarMean(new Tensor<Half>(data, new[] { N }), correction: 1);
        AssertHalf(SumSqDev(x, 0, N, 1) / (N - 1), variance.ToArray()[0], "TensorVarMean var");
        AssertHalf(Mean(x, 0, N, 1), mean.ToArray()[0], "TensorVarMean mean");
    }

    [Fact]
    public void VectorMeanAndStdDev_Half_Over70000Elements()
    {
        var data = HalfData(N, Signal);
        var x = ToDouble(data);
        var v = new Vector<Half>(data);
        AssertHalf(Mean(x, 0, N, 1), _engine.Mean(v), "Mean(Vector)");
        AssertHalf(Math.Sqrt(SumSqDev(x, 0, N, 1) / N), _engine.StdDev(v), "StdDev(Vector)");
    }

    [Fact]
    public void ReduceMeanBackward_Half_Over70000Elements()
    {
        var grad = new Tensor<Half>(new[] { (Half)1.0 }, new[] { 1 });
        var g = _engine.ReduceMeanBackward(grad, new[] { N }, new[] { 0 }).ToArray();
        Assert.Equal(N, g.Length);
        AssertHalf(1.0 / N, g[0], "ReduceMeanBackward[0]");
        AssertHalf(1.0 / N, g[N - 1], "ReduceMeanBackward[N-1]");
    }

    [Fact]
    public void ReduceVarianceBackward_Half_Over70000Elements()
    {
        var data = HalfData(N, Signal);
        var x = ToDouble(data);
        double m = Mean(x, 0, N, 1);
        var input = new Tensor<Half>(data, new[] { N });
        var mean = new Tensor<Half>(new[] { (Half)m }, new[] { 1 });
        var grad = new Tensor<Half>(new[] { (Half)1.0 }, new[] { 1 });
        var g = _engine.ReduceVarianceBackward(grad, input, mean, new[] { 0 }).ToArray();
        double mh = (double)(Half)m;
        foreach (int i in new[] { 0, 1, 12345, N - 1 })
            AssertHalf(2.0 * (x[i] - mh) / N, g[i], $"ReduceVarianceBackward[{i}]");
    }

    [Fact]
    public void MeanReductionLosses_Half_Over70000Elements()
    {
        var p = HalfData(N, Signal);
        var t = HalfData(N, Target);
        var px = ToDouble(p);
        var tx = ToDouble(t);
        double mse = 0, l1 = 0, huber = 0;
        for (int i = 0; i < N; i++)
        {
            double d = px[i] - tx[i];
            mse += d * d;
            l1 += Math.Abs(d);
            huber += Math.Abs(d) <= 1.0 ? 0.5 * d * d : Math.Abs(d) - 0.5;
        }
        var pt = new Tensor<Half>(p, new[] { N });
        var tt = new Tensor<Half>(t, new[] { N });
        AssertHalf(mse / N, _engine.TensorMSELoss(pt, tt).ToArray()[0], "TensorMSELoss");
        AssertHalf(mse / N, _engine.MseLoss(pt, tt), "MseLoss");
        AssertHalf(l1 / N, _engine.TensorL1Loss(pt, tt).ToArray()[0], "TensorL1Loss");
        AssertHalf(huber / N, _engine.TensorHuberLoss(pt, tt, 1.0).ToArray()[0], "TensorHuberLoss");
    }

    [Fact]
    public void BceWithLogitsAndKlDivLosses_Half_Over70000Elements()
    {
        var logits = HalfData(N, i => Signal(i) - 1.0);
        var probs = HalfData(N, i => 0.5 + Target(i));
        var lx = ToDouble(logits);
        var px = ToDouble(probs);
        double bce = 0, kl = 0;
        for (int i = 0; i < N; i++)
        {
            double xv = lx[i], tv = px[i];
            bce += Math.Max(xv, 0) - xv * tv + Math.Log(1 + Math.Exp(-Math.Abs(xv)));
            kl += tv * (Math.Log(tv) - xv);
        }
        var lt = new Tensor<Half>(logits, new[] { N });
        var pt = new Tensor<Half>(probs, new[] { N });
        AssertHalf(bce / N, _engine.TensorBCEWithLogitsLoss(lt, pt).ToArray()[0], "TensorBCEWithLogitsLoss");
        AssertHalf(kl / N, _engine.TensorKLDivLoss(lt, pt).ToArray()[0], "TensorKLDivLoss");
    }

    [Fact]
    public void ReduceMeanAndVariance_BFloat16_Over70000Elements()
    {
        var data = BFloat16Data(N, Signal);
        var x = ToDouble(data);
        var t = new Tensor<BFloat16>(data, new[] { N });
        AssertBFloat16(Mean(x, 0, N, 1), _engine.ReduceMean(t, new[] { 0 }, keepDims: false).ToArray()[0], "ReduceMean bf16");
        AssertBFloat16(Mean(x, 0, N, 1), _engine.TensorMean(t), "TensorMean bf16");
        AssertBFloat16(SumSqDev(x, 0, N, 1) / N, _engine.ReduceVariance(t, new[] { 0 }, keepDims: false).ToArray()[0], "ReduceVariance bf16");
    }

    [Fact]
    public void FloatAndDouble_ReduceMean_Unchanged_ForShortInputs()
    {
        // Regression guard for the fast paths: float/double still produce exact sums for small integer data.
        var f = _engine.ReduceMean(new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 }), new[] { 1 }, keepDims: false).ToArray();
        Assert.Equal(new[] { 1.5f, 3.5f }, f);
        var d = _engine.ReduceVariance(new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 4 }), new[] { 0 }, keepDims: false).ToArray();
        Assert.Equal(1.25, d[0]);
    }
}
