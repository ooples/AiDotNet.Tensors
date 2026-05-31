using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused Gated DeltaNet scan kernel
/// (<see cref="CpuEngine.GatedDeltaNetScanForward{T}"/>, issue ooples/AiDotNet#1464).
/// Forward is checked against an independent reference delta-rule scan; the custom autodiff backward
/// is checked against central finite differences of sum(output) for every input.
/// </summary>
public class GatedDeltaNetScanTests
{
    private static double[] ReferenceForward(
        double[] Q, double[] K, double[] V, double[] A, double[] B,
        int batch, int seqLen, int modelDim, int numHeads)
    {
        int headDim = modelDim / numHeads;
        double kappa = 1.0 / Math.Sqrt(headDim);
        var outp = new double[Q.Length];
        var S = new double[headDim * headDim];
        var sK = new double[headDim];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                Array.Clear(S, 0, S.Length);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int off = (b * seqLen + t) * modelDim + hOff;
                    int g = (b * seqLen + t) * numHeads + h;
                    double a = A[g], bet = B[g];
                    for (int di = 0; di < headDim; di++)
                    {
                        double s = 0.0;
                        for (int ki = 0; ki < headDim; ki++) s += S[di * headDim + ki] * (K[off + ki] * kappa);
                        sK[di] = s;
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        double delta = V[off + di] - sK[di];
                        for (int ki = 0; ki < headDim; ki++)
                            S[di * headDim + ki] = a * S[di * headDim + ki] + bet * delta * (K[off + ki] * kappa);
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        double o = 0.0;
                        for (int ki = 0; ki < headDim; ki++) o += S[di * headDim + ki] * Q[off + ki];
                        outp[off + di] = o;
                    }
                }
            }
        return outp;
    }

    private static double[] Gen(int n, int s, double scale = 0.4)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = Math.Sin(0.5 * (i + 1) + 1.4 * s) * scale;
        return arr;
    }

    // alpha (forget) in (0,1); beta (write) in (0,1).
    private static double[] GenGate(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = 1.0 / (1.0 + Math.Exp(-Math.Sin(0.5 * (i + 1) + 1.4 * s)));
        return arr;
    }

    private static (Tensor<double> q, Tensor<double> k, Tensor<double> v, Tensor<double> a, Tensor<double> b)
        MakeInputs(int batch, int seqLen, int modelDim, int numHeads, int seed)
    {
        int n = batch * seqLen * modelDim;
        int g = batch * seqLen * numHeads;
        var shape = new[] { batch, seqLen, modelDim };
        var gShape = new[] { batch, seqLen, numHeads };
        return (new Tensor<double>(Gen(n, seed), shape), new Tensor<double>(Gen(n, seed + 1), shape),
                new Tensor<double>(Gen(n, seed + 2), shape),
                new Tensor<double>(GenGate(g, seed + 3), gShape), new Tensor<double>(GenGate(g, seed + 4), gShape));
    }

    [Fact]
    public void Forward_MatchesReference()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6, numHeads = 2;
        var (q, k, v, a, b) = MakeInputs(batch, seqLen, modelDim, numHeads, 13);

        var outp = engine.GatedDeltaNetScanForward(q, k, v, a, b, numHeads);
        var expected = ReferenceForward(
            (double[])(object)q.GetDataArray()!, (double[])(object)k.GetDataArray()!,
            (double[])(object)v.GetDataArray()!, (double[])(object)a.GetDataArray()!,
            (double[])(object)b.GetDataArray()!, batch, seqLen, modelDim, numHeads);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-10,
                $"Forward[{i}] = {got[i]} vs reference {expected[i]}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, modelDim = 4, numHeads = 2;
        var (q, k, v, a, b) = MakeInputs(batch, seqLen, modelDim, numHeads, 6);

        Tensor<double> outp;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            outp = engine.GatedDeltaNetScanForward(q, k, v, a, b, numHeads);
            grads = tape.ComputeGradients(outp, new[] { q, k, v, a, b });
        }

        const double eps = 1e-6;
        foreach (var input in new[] { q, k, v, a, b })
        {
            var data = (double[])(object)input.GetDataArray()!;
            var grad = grads[input];
            for (int i = 0; i < data.Length; i++)
            {
                double orig = data[i];
                data[i] = orig + eps;
                double sp = SumForward(engine, q, k, v, a, b, numHeads);
                data[i] = orig - eps;
                double sm = SumForward(engine, q, k, v, a, b, numHeads);
                data[i] = orig;
                double numeric = (sp - sm) / (2.0 * eps);
                double analytic = grad.GetFlat(i);
                double tol = 1e-5 + 1e-4 * Math.Abs(analytic);
                Assert.True(Math.Abs(numeric - analytic) < tol,
                    $"grad mismatch at element {i}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static double SumForward(
        CpuEngine engine, Tensor<double> q, Tensor<double> k, Tensor<double> v,
        Tensor<double> a, Tensor<double> b, int numHeads)
    {
        var outp = engine.GatedDeltaNetScanForward(q, k, v, a, b, numHeads);
        var data = (double[])(object)outp.GetDataArray()!;
        double s = 0.0;
        for (int i = 0; i < data.Length; i++) s += data[i];
        return s;
    }
}
