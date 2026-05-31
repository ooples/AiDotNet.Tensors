using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused RWKV-4 WKV sequence kernel
/// (<see cref="CpuEngine.Rwkv4WkvForward{T}"/>, issue ooples/AiDotNet#1464).
/// The forward is checked against an independent numerically-stable reference recurrence,
/// and the custom autodiff backward is checked against central finite differences of sum(output)
/// for every input (R, K, V, timeDecay, timeFirst).
/// </summary>
public class Rwkv4WkvTests
{
    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    /// <summary>Independent reference: per-channel numerically-stable RWKV-4 WKV recurrence.</summary>
    private static double[] ReferenceForward(
        double[] R, double[] K, double[] V, double[] timeDecay, double[] timeFirst,
        int batch, int seqLen, int modelDim)
    {
        var outp = new double[R.Length];
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < modelDim; c++)
            {
                double w = -Math.Exp(timeDecay[c]);
                double u = timeFirst[c];
                double aa = 0.0, bb = 0.0, pp = double.NegativeInfinity;
                for (int t = 0; t < seqLen; t++)
                {
                    int off = (b * seqLen + t) * modelDim + c;
                    double k = K[off], v = V[off];
                    double ww = u + k;
                    double q = Math.Max(pp, ww);
                    double e1 = Math.Exp(pp - q);
                    double e2 = Math.Exp(ww - q);
                    outp[off] = Sigmoid(R[off]) * (e1 * aa + e2 * v) / (e1 * bb + e2);

                    double ww2 = pp + w;
                    double q2 = Math.Max(ww2, k);
                    double e1b = Math.Exp(ww2 - q2);
                    double e2b = Math.Exp(k - q2);
                    aa = e1b * aa + e2b * v;
                    bb = e1b * bb + e2b;
                    pp = q2;
                }
            }
        return outp;
    }

    private static double[] Gen(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++)
            arr[i] = Math.Sin(0.7 * (i + 1) + 1.3 * s) * 0.9;
        return arr;
    }

    private static (Tensor<double> r, Tensor<double> k, Tensor<double> v, Tensor<double> td, Tensor<double> tf)
        MakeInputs(int batch, int seqLen, int modelDim, int seed)
    {
        int n = batch * seqLen * modelDim;
        var shape = new[] { batch, seqLen, modelDim };
        return (new Tensor<double>(Gen(n, seed), shape),
                new Tensor<double>(Gen(n, seed + 1), shape),
                new Tensor<double>(Gen(n, seed + 2), shape),
                new Tensor<double>(Gen(modelDim, seed + 3), new[] { modelDim }),
                new Tensor<double>(Gen(modelDim, seed + 4), new[] { modelDim }));
    }

    [Fact]
    public void Forward_MatchesReferenceRecurrence()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6;
        var (r, k, v, td, tf) = MakeInputs(batch, seqLen, modelDim, 10);

        var outp = engine.Rwkv4WkvForward(r, k, v, td, tf);
        var expected = ReferenceForward(
            (double[])(object)r.GetDataArray()!, (double[])(object)k.GetDataArray()!,
            (double[])(object)v.GetDataArray()!, (double[])(object)td.GetDataArray()!,
            (double[])(object)tf.GetDataArray()!, batch, seqLen, modelDim);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-10,
                $"Forward[{i}] = {got[i]} vs reference {expected[i]}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, modelDim = 4;
        var (r, k, v, td, tf) = MakeInputs(batch, seqLen, modelDim, 3);

        Tensor<double> outp;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            outp = engine.Rwkv4WkvForward(r, k, v, td, tf);
            grads = tape.ComputeGradients(outp, new[] { r, k, v, td, tf });
        }

        const double eps = 1e-6;
        var inputs = new[] { r, k, v, td, tf };
        foreach (var x in inputs)
        {
            var data = (double[])(object)x.GetDataArray()!;
            var grad = grads[x];
            for (int i = 0; i < data.Length; i++)
            {
                double orig = data[i];
                data[i] = orig + eps;
                double sumPlus = SumForward(engine, r, k, v, td, tf);
                data[i] = orig - eps;
                double sumMinus = SumForward(engine, r, k, v, td, tf);
                data[i] = orig;
                double numeric = (sumPlus - sumMinus) / (2.0 * eps);
                double analytic = grad.GetFlat(i);
                // Combined absolute + relative tolerance: the WKV ratio can produce
                // larger gradients on timeDecay/timeFirst than the elementwise streams.
                double tol = 1e-5 + 1e-4 * Math.Abs(analytic);
                Assert.True(Math.Abs(numeric - analytic) < tol,
                    $"grad mismatch at element {i}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static double SumForward(
        CpuEngine engine, Tensor<double> r, Tensor<double> k, Tensor<double> v,
        Tensor<double> td, Tensor<double> tf)
    {
        var outp = engine.Rwkv4WkvForward(r, k, v, td, tf);
        var data = (double[])(object)outp.GetDataArray()!;
        double s = 0.0;
        for (int i = 0; i < data.Length; i++) s += data[i];
        return s;
    }
}
