using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused xLSTM (mLSTM) scan kernel
/// (<see cref="CpuEngine.XLstmScanForward{T}"/>, issue ooples/AiDotNet#1464).
/// Forward is checked against an independent reference; the custom autodiff backward is checked
/// against central finite differences of sum(output) for every input.
/// </summary>
public class XLstmScanTests
{
    private static double[] ReferenceForward(
        double[] Q, double[] K, double[] V, double[] I, double[] F, double[] O,
        int batch, int seqLen, int modelDim, int numHeads)
    {
        int headDim = modelDim / numHeads;
        double kappa = 1.0 / Math.Sqrt(headDim);
        var outp = new double[Q.Length];
        var C = new double[headDim * headDim];
        var n = new double[headDim];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                Array.Clear(C, 0, C.Length); Array.Clear(n, 0, n.Length);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int off = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h; // scalar-per-head gates
                    double iv = I[gOff]; if (iv > 4.85e8) iv = 4.85e8;
                    double f = F[gOff], o = O[gOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        n[di] = f * n[di] + iv * (K[off + di] * kappa);
                        double vv = V[off + di];
                        for (int ki = 0; ki < headDim; ki++)
                            C[di * headDim + ki] = f * C[di * headDim + ki] + iv * vv * (K[off + ki] * kappa);
                    }
                    double nq = 0.0;
                    for (int j = 0; j < headDim; j++) nq += n[j] * Q[off + j];
                    double nf = Math.Max(Math.Abs(nq), 1.0);
                    for (int di = 0; di < headDim; di++)
                    {
                        double num = 0.0;
                        for (int ki = 0; ki < headDim; ki++) num += C[di * headDim + ki] * Q[off + ki];
                        outp[off + di] = o * num / nf;
                    }
                }
            }
        return outp;
    }

    private static double[] Gen(int n, int s, double scale = 0.35)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = Math.Sin(0.5 * (i + 1) + 1.5 * s) * scale;
        return arr;
    }

    // i = exp(small), f/o = sigmoid.
    private static double[] GenExp(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = Math.Exp(0.3 * Math.Sin(0.5 * (i + 1) + 1.5 * s));
        return arr;
    }
    private static double[] GenSig(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = 1.0 / (1.0 + Math.Exp(-Math.Sin(0.5 * (i + 1) + 1.5 * s)));
        return arr;
    }

    private static (Tensor<double> q, Tensor<double> k, Tensor<double> v,
                    Tensor<double> i, Tensor<double> f, Tensor<double> o)
        MakeInputs(int batch, int seqLen, int modelDim, int numHeads, int seed)
    {
        int n = batch * seqLen * modelDim;
        var shape = new[] { batch, seqLen, modelDim };
        // Gates are scalar-per-head: [batch, seqLen, numHeads].
        int gn = batch * seqLen * numHeads;
        var gShape = new[] { batch, seqLen, numHeads };
        return (new Tensor<double>(Gen(n, seed), shape), new Tensor<double>(Gen(n, seed + 1), shape),
                new Tensor<double>(Gen(n, seed + 2), shape), new Tensor<double>(GenExp(gn, seed + 3), gShape),
                new Tensor<double>(GenSig(gn, seed + 4), gShape), new Tensor<double>(GenSig(gn, seed + 5), gShape));
    }

    [Fact]
    public void Forward_MatchesReference()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6, numHeads = 2;
        var (q, k, v, i, f, o) = MakeInputs(batch, seqLen, modelDim, numHeads, 17);

        var outp = engine.XLstmScanForward(q, k, v, i, f, o, numHeads);
        var expected = ReferenceForward(
            (double[])(object)q.GetDataArray()!, (double[])(object)k.GetDataArray()!,
            (double[])(object)v.GetDataArray()!, (double[])(object)i.GetDataArray()!,
            (double[])(object)f.GetDataArray()!, (double[])(object)o.GetDataArray()!,
            batch, seqLen, modelDim, numHeads);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int idx = 0; idx < expected.Length; idx++)
            Assert.True(Math.Abs(got[idx] - expected[idx]) < 1e-10,
                $"Forward[{idx}] = {got[idx]} vs reference {expected[idx]}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, modelDim = 4, numHeads = 2;
        var (q, k, v, i, f, o) = MakeInputs(batch, seqLen, modelDim, numHeads, 8);

        Tensor<double> outp;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            outp = engine.XLstmScanForward(q, k, v, i, f, o, numHeads);
            grads = tape.ComputeGradients(outp, new[] { q, k, v, i, f, o });
        }

        const double eps = 1e-6;
        foreach (var input in new[] { q, k, v, i, f, o })
        {
            var data = (double[])(object)input.GetDataArray()!;
            var grad = grads[input];
            for (int idx = 0; idx < data.Length; idx++)
            {
                double orig = data[idx];
                data[idx] = orig + eps;
                double sp = SumForward(engine, q, k, v, i, f, o, numHeads);
                data[idx] = orig - eps;
                double sm = SumForward(engine, q, k, v, i, f, o, numHeads);
                data[idx] = orig;
                double numeric = (sp - sm) / (2.0 * eps);
                double analytic = grad.GetFlat(idx);
                double tol = 1e-5 + 2e-4 * Math.Abs(analytic);
                Assert.True(Math.Abs(numeric - analytic) < tol,
                    $"grad mismatch at element {idx}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static double SumForward(
        CpuEngine engine, Tensor<double> q, Tensor<double> k, Tensor<double> v,
        Tensor<double> i, Tensor<double> f, Tensor<double> o, int numHeads)
    {
        var outp = engine.XLstmScanForward(q, k, v, i, f, o, numHeads);
        var data = (double[])(object)outp.GetDataArray()!;
        double s = 0.0;
        for (int idx = 0; idx < data.Length; idx++) s += data[idx];
        return s;
    }
}
