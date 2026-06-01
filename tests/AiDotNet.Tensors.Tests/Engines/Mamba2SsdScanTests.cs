using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused Mamba-2 SSD scan kernel
/// (<see cref="CpuEngine.Mamba2SsdScanForward{T}"/>, issue ooples/AiDotNet#1464).
/// Forward is checked against an independent reference; the custom autodiff backward is checked
/// against central finite differences of sum(output) for every input.
/// </summary>
public class Mamba2SsdScanTests
{
    private static double[] ReferenceForward(
        double[] X, double[] delta, double[] aLog, double[] B, double[] C, double[] D,
        int batch, int seqLen, int innerDim, int numHeads, int sd)
    {
        int headDim = innerDim / numHeads;
        var outp = new double[X.Length];
        var negA = new double[numHeads];
        for (int hi = 0; hi < numHeads; hi++) negA[hi] = -Math.Exp(aLog[hi]);
        var h = new double[innerDim * sd];
        for (int b = 0; b < batch; b++)
        {
            Array.Clear(h, 0, h.Length);
            for (int t = 0; t < seqLen; t++)
            {
                int btI = (b * seqLen + t) * innerDim, btS = (b * seqLen + t) * sd, btH = (b * seqLen + t) * numHeads;
                for (int hi = 0; hi < numHeads; hi++)
                {
                    double dt = delta[btH + hi];
                    double aBar = Math.Exp(dt * negA[hi]);
                    for (int di = 0; di < headDim; di++)
                    {
                        int flatD = hi * headDim + di;
                        double xv = X[btI + flatD];
                        int hb = flatD * sd;
                        double y = 0.0;
                        for (int n = 0; n < sd; n++)
                        {
                            h[hb + n] = aBar * h[hb + n] + dt * B[btS + n] * xv;
                            y += C[btS + n] * h[hb + n];
                        }
                        outp[btI + flatD] = y + D[hi] * xv;
                    }
                }
            }
        }
        return outp;
    }

    private static double[] Gen(int n, int s, double scale = 0.4)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = Math.Sin(0.5 * (i + 1) + 1.6 * s) * scale;
        return arr;
    }
    private static double[] GenPos(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = 0.2 + 0.25 * (0.5 + 0.5 * Math.Sin(0.5 * (i + 1) + 1.6 * s));
        return arr;
    }

    private static (Tensor<double> x, Tensor<double> delta, Tensor<double> aLog,
                    Tensor<double> b, Tensor<double> c, Tensor<double> d)
        MakeInputs(int batch, int seqLen, int innerDim, int numHeads, int sd, int seed)
    {
        int nI = batch * seqLen * innerDim, nS = batch * seqLen * sd, nH = batch * seqLen * numHeads;
        return (new Tensor<double>(Gen(nI, seed), new[] { batch, seqLen, innerDim }),
                new Tensor<double>(GenPos(nH, seed + 1), new[] { batch, seqLen, numHeads }),
                new Tensor<double>(Gen(numHeads, seed + 2, 0.3), new[] { numHeads }),
                new Tensor<double>(Gen(nS, seed + 3), new[] { batch, seqLen, sd }),
                new Tensor<double>(Gen(nS, seed + 4), new[] { batch, seqLen, sd }),
                new Tensor<double>(Gen(numHeads, seed + 5, 0.4), new[] { numHeads }));
    }

    [Fact]
    public void Forward_MatchesReference()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, innerDim = 6, numHeads = 3, sd = 4;
        var (x, delta, aLog, b, c, d) = MakeInputs(batch, seqLen, innerDim, numHeads, sd, 21);

        var outp = engine.Mamba2SsdScanForward(x, delta, aLog, b, c, d, numHeads);
        var expected = ReferenceForward(
            (double[])(object)x.GetDataArray()!, (double[])(object)delta.GetDataArray()!,
            (double[])(object)aLog.GetDataArray()!, (double[])(object)b.GetDataArray()!,
            (double[])(object)c.GetDataArray()!, (double[])(object)d.GetDataArray()!,
            batch, seqLen, innerDim, numHeads, sd);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-10,
                $"Forward[{i}] = {got[i]} vs reference {expected[i]}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, innerDim = 4, numHeads = 2, sd = 3;
        var (x, delta, aLog, b, c, d) = MakeInputs(batch, seqLen, innerDim, numHeads, sd, 9);

        Tensor<double> outp;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            outp = engine.Mamba2SsdScanForward(x, delta, aLog, b, c, d, numHeads);
            grads = tape.ComputeGradients(outp, new[] { x, delta, aLog, b, c, d });
        }

        const double eps = 1e-6;
        foreach (var input in new[] { x, delta, aLog, b, c, d })
        {
            var data = (double[])(object)input.GetDataArray()!;
            var grad = grads[input];
            for (int i = 0; i < data.Length; i++)
            {
                double orig = data[i];
                data[i] = orig + eps;
                double sp = SumForward(engine, x, delta, aLog, b, c, d, numHeads);
                data[i] = orig - eps;
                double sm = SumForward(engine, x, delta, aLog, b, c, d, numHeads);
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
        CpuEngine engine, Tensor<double> x, Tensor<double> delta, Tensor<double> aLog,
        Tensor<double> b, Tensor<double> c, Tensor<double> d, int numHeads)
    {
        var outp = engine.Mamba2SsdScanForward(x, delta, aLog, b, c, d, numHeads);
        var data = (double[])(object)outp.GetDataArray()!;
        double s = 0.0;
        for (int i = 0; i < data.Length; i++) s += data[i];
        return s;
    }
}
