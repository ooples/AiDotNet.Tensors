using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused Mamba S6 selective-scan kernel
/// (<see cref="CpuEngine.MambaSelectiveScanForward{T}"/>, issue ooples/AiDotNet#1464).
/// Forward is checked against an independent reference scan; the custom autodiff backward is
/// checked against central finite differences of sum(output) for every input.
/// </summary>
public class MambaScanTests
{
    /// <summary>Independent reference: plain nested-loop selective SSM scan.</summary>
    private static double[] ReferenceForward(
        double[] X, double[] delta, double[] aLog, double[] B, double[] C, double[] D,
        int batch, int seqLen, int innerDim, int stateDim)
    {
        var outp = new double[X.Length];
        var negA = new double[innerDim * stateDim];
        for (int i = 0; i < negA.Length; i++) negA[i] = -Math.Exp(aLog[i]);
        var h = new double[innerDim * stateDim];
        for (int b = 0; b < batch; b++)
        {
            Array.Clear(h, 0, h.Length);
            for (int t = 0; t < seqLen; t++)
            {
                int baseID = (b * seqLen + t) * innerDim;
                int baseSD = (b * seqLen + t) * stateDim;
                for (int di = 0; di < innerDim; di++)
                {
                    double dt = delta[baseID + di], xv = X[baseID + di];
                    int hrow = di * stateDim;
                    double y = 0.0;
                    for (int ni = 0; ni < stateDim; ni++)
                    {
                        double aBar = Math.Exp(dt * negA[hrow + ni]);
                        h[hrow + ni] = aBar * h[hrow + ni] + dt * B[baseSD + ni] * xv;
                        y += C[baseSD + ni] * h[hrow + ni];
                    }
                    outp[baseID + di] = y + D[di] * xv;
                }
            }
        }
        return outp;
    }

    private static double[] Gen(int n, int s, double scale = 0.5)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = Math.Sin(0.6 * (i + 1) + 1.1 * s) * scale;
        return arr;
    }

    // delta must be positive (it is softplus output in Mamba); use a positive generator.
    private static double[] GenPos(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = 0.2 + 0.3 * (0.5 + 0.5 * Math.Sin(0.6 * (i + 1) + 1.1 * s));
        return arr;
    }

    private static (Tensor<double> x, Tensor<double> delta, Tensor<double> aLog,
                    Tensor<double> b, Tensor<double> c, Tensor<double> d)
        MakeInputs(int batch, int seqLen, int innerDim, int stateDim, int seed)
    {
        int n3 = batch * seqLen * innerDim;
        int s3 = batch * seqLen * stateDim;
        return (new Tensor<double>(Gen(n3, seed), new[] { batch, seqLen, innerDim }),
                new Tensor<double>(GenPos(n3, seed + 1), new[] { batch, seqLen, innerDim }),
                new Tensor<double>(Gen(innerDim * stateDim, seed + 2, 0.3), new[] { innerDim, stateDim }),
                new Tensor<double>(Gen(s3, seed + 3), new[] { batch, seqLen, stateDim }),
                new Tensor<double>(Gen(s3, seed + 4), new[] { batch, seqLen, stateDim }),
                new Tensor<double>(Gen(innerDim, seed + 5, 0.4), new[] { innerDim }));
    }

    [Fact]
    public void Forward_MatchesReferenceScan()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, innerDim = 4, stateDim = 3;
        var (x, delta, aLog, b, c, d) = MakeInputs(batch, seqLen, innerDim, stateDim, 7);

        var outp = engine.MambaSelectiveScanForward(x, delta, aLog, b, c, d);
        var expected = ReferenceForward(
            (double[])(object)x.GetDataArray()!, (double[])(object)delta.GetDataArray()!,
            (double[])(object)aLog.GetDataArray()!, (double[])(object)b.GetDataArray()!,
            (double[])(object)c.GetDataArray()!, (double[])(object)d.GetDataArray()!,
            batch, seqLen, innerDim, stateDim);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-10,
                $"Forward[{i}] = {got[i]} vs reference {expected[i]}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, innerDim = 3, stateDim = 2;
        var (x, delta, aLog, b, c, d) = MakeInputs(batch, seqLen, innerDim, stateDim, 2);

        Tensor<double> outp;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            outp = engine.MambaSelectiveScanForward(x, delta, aLog, b, c, d);
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
                double sumPlus = SumForward(engine, x, delta, aLog, b, c, d);
                data[i] = orig - eps;
                double sumMinus = SumForward(engine, x, delta, aLog, b, c, d);
                data[i] = orig;
                double numeric = (sumPlus - sumMinus) / (2.0 * eps);
                double analytic = grad.GetFlat(i);
                double tol = 1e-5 + 1e-4 * Math.Abs(analytic);
                Assert.True(Math.Abs(numeric - analytic) < tol,
                    $"grad mismatch at element {i}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    [Fact]
    public void Forward_ParallelDiPath_MatchesReferenceScan()
    {
        // innerDim well above MambaDiGrain with batch>1 forces the multi-chunk parallel di path.
        var engine = new CpuEngine();
        int batch = 3, seqLen = 12, innerDim = 96, stateDim = 8;
        var (x, delta, aLog, b, c, d) = MakeInputs(batch, seqLen, innerDim, stateDim, 13);

        var outp = engine.MambaSelectiveScanForward(x, delta, aLog, b, c, d);
        var expected = ReferenceForward(
            (double[])(object)x.GetDataArray()!, (double[])(object)delta.GetDataArray()!,
            (double[])(object)aLog.GetDataArray()!, (double[])(object)b.GetDataArray()!,
            (double[])(object)c.GetDataArray()!, (double[])(object)d.GetDataArray()!,
            batch, seqLen, innerDim, stateDim);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-9,
                $"Forward[{i}] = {got[i]} vs reference {expected[i]}");
    }

    [Fact]
    public void Backward_ParallelDiPath_MatchesFiniteDifferences()
    {
        // batch>1, innerDim>grain → exercises the parallel di backward + the cross-di dB/dC reduction.
        var engine = new CpuEngine();
        int batch = 2, seqLen = 6, innerDim = 32, stateDim = 4;
        var (x, delta, aLog, b, c, d) = MakeInputs(batch, seqLen, innerDim, stateDim, 5);

        Tensor<double> outp;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            outp = engine.MambaSelectiveScanForward(x, delta, aLog, b, c, d);
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
                double sumPlus = SumForward(engine, x, delta, aLog, b, c, d);
                data[i] = orig - eps;
                double sumMinus = SumForward(engine, x, delta, aLog, b, c, d);
                data[i] = orig;
                double numeric = (sumPlus - sumMinus) / (2.0 * eps);
                double analytic = grad.GetFlat(i);
                double tol = 1e-5 + 1e-4 * Math.Abs(analytic);
                Assert.True(Math.Abs(numeric - analytic) < tol,
                    $"grad mismatch at element {i}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static double SumForward(
        CpuEngine engine, Tensor<double> x, Tensor<double> delta, Tensor<double> aLog,
        Tensor<double> b, Tensor<double> c, Tensor<double> d)
    {
        var outp = engine.MambaSelectiveScanForward(x, delta, aLog, b, c, d);
        var data = (double[])(object)outp.GetDataArray()!;
        double s = 0.0;
        for (int i = 0; i < data.Length; i++) s += data[i];
        return s;
    }
}
