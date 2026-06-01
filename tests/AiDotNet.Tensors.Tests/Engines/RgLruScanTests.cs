using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused RG-LRU scan kernel
/// (<see cref="CpuEngine.RgLruScanForward{T}"/>, issue ooples/AiDotNet#1464).
/// Forward is checked against an independent reference; the custom autodiff backward is checked
/// against central finite differences of sum(output) for every input.
/// </summary>
public class RgLruScanTests
{
    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double[] ReferenceForward(
        double[] V, double[] R, double[] I, double[] decay, int batch, int seqLen, int recDim)
    {
        var outp = new double[V.Length];
        var baseD = new double[recDim];
        for (int c = 0; c < recDim; c++) baseD[c] = Sigmoid(-decay[c]);
        var h = new double[recDim];
        for (int b = 0; b < batch; b++)
        {
            Array.Clear(h, 0, recDim);
            for (int t = 0; t < seqLen; t++)
            {
                int off = (b * seqLen + t) * recDim;
                for (int c = 0; c < recDim; c++)
                {
                    double a = R[off + c] * baseD[c];
                    double om = 1.0 - a * a;
                    double s = om > 0 ? Math.Sqrt(om) : 0.0;
                    h[c] = a * h[c] + s * (I[off + c] * V[off + c]);
                    outp[off + c] = h[c];
                }
            }
        }
        return outp;
    }

    private static double[] Gen(int n, int s, double scale = 0.6)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = Math.Sin(0.55 * (i + 1) + 1.2 * s) * scale;
        return arr;
    }

    // Gates are sigmoid outputs in (0,1).
    private static double[] GenGate(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = Sigmoid(Math.Sin(0.55 * (i + 1) + 1.2 * s));
        return arr;
    }

    private static (Tensor<double> v, Tensor<double> r, Tensor<double> i, Tensor<double> d)
        MakeInputs(int batch, int seqLen, int recDim, int seed)
    {
        int n = batch * seqLen * recDim;
        var shape = new[] { batch, seqLen, recDim };
        return (new Tensor<double>(Gen(n, seed), shape),
                new Tensor<double>(GenGate(n, seed + 1), shape),
                new Tensor<double>(GenGate(n, seed + 2), shape),
                new Tensor<double>(Gen(recDim, seed + 3, 0.5), new[] { recDim }));
    }

    [Fact]
    public void Forward_MatchesReference()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, recDim = 6;
        var (v, r, i, d) = MakeInputs(batch, seqLen, recDim, 11);

        var outp = engine.RgLruScanForward(v, r, i, d);
        var expected = ReferenceForward(
            (double[])(object)v.GetDataArray()!, (double[])(object)r.GetDataArray()!,
            (double[])(object)i.GetDataArray()!, (double[])(object)d.GetDataArray()!,
            batch, seqLen, recDim);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int k = 0; k < expected.Length; k++)
            Assert.True(Math.Abs(got[k] - expected[k]) < 1e-10,
                $"Forward[{k}] = {got[k]} vs reference {expected[k]}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, recDim = 3;
        var (v, r, i, d) = MakeInputs(batch, seqLen, recDim, 4);

        Tensor<double> outp;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            outp = engine.RgLruScanForward(v, r, i, d);
            grads = tape.ComputeGradients(outp, new[] { v, r, i, d });
        }

        const double eps = 1e-6;
        foreach (var input in new[] { v, r, i, d })
        {
            var data = (double[])(object)input.GetDataArray()!;
            var grad = grads[input];
            for (int k = 0; k < data.Length; k++)
            {
                double orig = data[k];
                data[k] = orig + eps;
                double sp = SumForward(engine, v, r, i, d);
                data[k] = orig - eps;
                double sm = SumForward(engine, v, r, i, d);
                data[k] = orig;
                double numeric = (sp - sm) / (2.0 * eps);
                double analytic = grad.GetFlat(k);
                double tol = 1e-5 + 1e-4 * Math.Abs(analytic);
                Assert.True(Math.Abs(numeric - analytic) < tol,
                    $"grad mismatch at element {k}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static double SumForward(
        CpuEngine engine, Tensor<double> v, Tensor<double> r, Tensor<double> i, Tensor<double> d)
    {
        var outp = engine.RgLruScanForward(v, r, i, d);
        var data = (double[])(object)outp.GetDataArray()!;
        double s = 0.0;
        for (int k = 0; k < data.Length; k++) s += data[k];
        return s;
    }
}
