using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused RWKV-7 sequence kernel
/// (<see cref="CpuEngine.Rwkv7SequenceForward{T}"/>, issue ooples/AiDotNet#1464).
/// The forward is checked against an independent reference recurrence, and the custom
/// autodiff backward is checked against central finite differences of sum(output).
/// </summary>
public class Rwkv7SequenceTests
{
    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    /// <summary>Independent reference: plain nested-loop RWKV-7 recurrence over a sequence.</summary>
    private static double[] ReferenceForward(
        double[] R, double[] K, double[] V, double[] A, double[] B,
        int batch, int seqLen, int modelDim, int numHeads)
    {
        int headDim = modelDim / numHeads;
        var outp = new double[R.Length];
        var S = new double[headDim * headDim];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                Array.Clear(S, 0, S.Length);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int off = (b * seqLen + t) * modelDim + hOff;
                    for (int di = 0; di < headDim; di++)
                    {
                        double ga = Sigmoid(A[off + di]);
                        double gbk = Sigmoid(B[off + di]) * K[off + di];
                        for (int vi = 0; vi < headDim; vi++)
                            S[di * headDim + vi] = ga * S[di * headDim + vi] + gbk * V[off + vi];
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        double sk = 0.0;
                        for (int vi = 0; vi < headDim; vi++)
                            sk += S[di * headDim + vi] * K[off + vi];
                        outp[off + di] = Sigmoid(R[off + di]) * sk;
                    }
                }
            }
        return outp;
    }

    private static (Tensor<double> r, Tensor<double> k, Tensor<double> v, Tensor<double> a, Tensor<double> b)
        MakeInputs(int batch, int seqLen, int modelDim, int seed)
    {
        int n = batch * seqLen * modelDim;
        var shape = new[] { batch, seqLen, modelDim };
        // Deterministic pseudo-random in [-1, 1] (no Random dependency).
        double[] Gen(int s)
        {
            var arr = new double[n];
            for (int i = 0; i < n; i++)
                arr[i] = Math.Sin(0.7 * (i + 1) + 1.3 * s) * 0.9;
            return arr;
        }
        return (new Tensor<double>(Gen(seed), shape), new Tensor<double>(Gen(seed + 1), shape),
                new Tensor<double>(Gen(seed + 2), shape), new Tensor<double>(Gen(seed + 3), shape),
                new Tensor<double>(Gen(seed + 4), shape));
    }

    [Fact]
    public void Forward_MatchesReferenceRecurrence()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6, numHeads = 3;
        var (r, k, v, a, b) = MakeInputs(batch, seqLen, modelDim, 10);

        var outp = engine.Rwkv7SequenceForward(r, k, v, a, b, numHeads);
        var expected = ReferenceForward(
            (double[])(object)r.GetDataArray()!, (double[])(object)k.GetDataArray()!,
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
        var (r, k, v, a, b) = MakeInputs(batch, seqLen, modelDim, 3);

        // Analytic gradients of sum(output) via the custom tape backward.
        Tensor<double> outp;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            outp = engine.Rwkv7SequenceForward(r, k, v, a, b, numHeads);
            grads = tape.ComputeGradients(outp, new[] { r, k, v, a, b });
        }

        const double eps = 1e-6;
        var inputs = new[] { r, k, v, a, b };
        foreach (var x in inputs)
        {
            var data = (double[])(object)x.GetDataArray()!;
            var grad = grads[x];
            for (int i = 0; i < data.Length; i++)
            {
                double orig = data[i];
                data[i] = orig + eps;
                double sumPlus = SumForward(engine, r, k, v, a, b, numHeads);
                data[i] = orig - eps;
                double sumMinus = SumForward(engine, r, k, v, a, b, numHeads);
                data[i] = orig;
                double numeric = (sumPlus - sumMinus) / (2.0 * eps);
                double analytic = grad.GetFlat(i);
                Assert.True(Math.Abs(numeric - analytic) < 1e-5,
                    $"grad mismatch at element {i}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static double SumForward(
        CpuEngine engine, Tensor<double> r, Tensor<double> k, Tensor<double> v,
        Tensor<double> a, Tensor<double> b, int numHeads)
    {
        // No active tape here → pure forward, no recording.
        var outp = engine.Rwkv7SequenceForward(r, k, v, a, b, numHeads);
        var data = (double[])(object)outp.GetDataArray()!;
        double s = 0.0;
        for (int i = 0; i < data.Length; i++) s += data[i];
        return s;
    }
}
