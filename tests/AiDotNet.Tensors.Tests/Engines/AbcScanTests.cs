using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused ABC slot-recurrence kernel
/// (<see cref="CpuEngine.AbcScanForward{T}"/>, issue ooples/AiDotNet#1464).
/// Forward is checked against an independent reference; the custom autodiff backward is checked
/// against central finite differences for every input, including the slot keys — which reach the
/// output by two separate routes (the write scores at each step, and the initial slot state), so a
/// backward that drops either route fails here.
/// </summary>
public class AbcScanTests
{
    private const double InitScale = 0.1;

    private static void SoftmaxRef(double[] x)
    {
        double max = x[0];
        for (int i = 1; i < x.Length; i++) if (x[i] > max) max = x[i];
        double sum = 0.0;
        for (int i = 0; i < x.Length; i++) { x[i] = Math.Exp(x[i] - max); sum += x[i]; }
        for (int i = 0; i < x.Length; i++) x[i] /= sum;
    }

    /// <summary>
    /// Independent transcription of the ABC recurrence, written straight from the layer's documented
    /// formulation rather than sharing any code with the kernel under test.
    /// </summary>
    private static double[] ReferenceForward(
        double[] Q, double[] K, double[] V, double[] FG, double[] SK,
        int batch, int seqLen, int modelDim, int numHeads, int numSlots)
    {
        int headDim = modelDim / numHeads;
        double scale = 1.0 / Math.Sqrt(headDim);
        var outp = new double[Q.Length];

        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;
                int skOff = h * numSlots * headDim;
                var slot = new double[numSlots][];
                for (int s = 0; s < numSlots; s++)
                {
                    slot[s] = new double[headDim];
                    for (int d = 0; d < headDim; d++)
                        slot[s][d] = InitScale * SK[skOff + s * headDim + d];
                }

                for (int t = 0; t < seqLen; t++)
                {
                    int off = (b * seqLen + t) * modelDim + hOff;
                    double fg = FG[(b * seqLen + t) * numHeads + h];

                    var w = new double[numSlots];
                    for (int s = 0; s < numSlots; s++)
                    {
                        double dot = 0.0;
                        for (int d = 0; d < headDim; d++) dot += K[off + d] * SK[skOff + s * headDim + d];
                        w[s] = dot * scale;
                    }
                    SoftmaxRef(w);

                    for (int s = 0; s < numSlots; s++)
                        for (int d = 0; d < headDim; d++)
                            slot[s][d] = fg * slot[s][d] + w[s] * V[off + d];

                    var r = new double[numSlots];
                    for (int s = 0; s < numSlots; s++)
                    {
                        double dot = 0.0;
                        for (int d = 0; d < headDim; d++) dot += Q[off + d] * slot[s][d];
                        r[s] = dot * scale;
                    }
                    SoftmaxRef(r);

                    for (int d = 0; d < headDim; d++)
                    {
                        double o = 0.0;
                        for (int s = 0; s < numSlots; s++) o += r[s] * slot[s][d];
                        outp[off + d] = o;
                    }
                }
            }
        return outp;
    }

    private static double[] Gen(int n, int s, double scale = 0.5)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = Math.Sin(0.5 * (i + 1) + 1.3 * s) * scale;
        return arr;
    }

    // The forget gate is a sigmoid output in (0,1); keep it below 1 for a stable recurrence.
    private static double[] GenGate(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = 1.0 / (1.0 + Math.Exp(-Math.Sin(0.5 * (i + 1) + 1.3 * s)));
        return arr;
    }

    private static (Tensor<double> q, Tensor<double> k, Tensor<double> v, Tensor<double> fg, Tensor<double> sk)
        MakeInputs(int batch, int seqLen, int modelDim, int numHeads, int numSlots, int seed)
    {
        int headDim = modelDim / numHeads;
        int n = batch * seqLen * modelDim;
        var shape = new[] { batch, seqLen, modelDim };
        int gn = batch * seqLen * numHeads;
        int skn = numHeads * numSlots * headDim;
        return (new Tensor<double>(Gen(n, seed), shape),
                new Tensor<double>(Gen(n, seed + 1), shape),
                new Tensor<double>(Gen(n, seed + 2), shape),
                new Tensor<double>(GenGate(gn, seed + 3), new[] { batch, seqLen, numHeads }),
                new Tensor<double>(Gen(skn, seed + 4), new[] { numHeads, numSlots, headDim }));
    }

    [Fact]
    public void Forward_MatchesReference()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 4, modelDim = 6, numHeads = 3, numSlots = 4;
        var (q, k, v, fg, sk) = MakeInputs(batch, seqLen, modelDim, numHeads, numSlots, 9);
        using var qOwner = q;
        using var kOwner = k;
        using var vOwner = v;
        using var fgOwner = fg;
        using var skOwner = sk;

        using var outp = engine.AbcScanForward(q, k, v, fg, sk, numHeads, InitScale);
        var expected = ReferenceForward(
            (double[])(object)q.GetDataArray()!, (double[])(object)k.GetDataArray()!,
            (double[])(object)v.GetDataArray()!, (double[])(object)fg.GetDataArray()!,
            (double[])(object)sk.GetDataArray()!,
            batch, seqLen, modelDim, numHeads, numSlots);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-10,
                $"Forward[{i}] = {got[i]} vs reference {expected[i]}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 3, modelDim = 4, numHeads = 2, numSlots = 3;
        var (q, k, v, fg, sk) = MakeInputs(batch, seqLen, modelDim, numHeads, numSlots, 5);
        using var qOwner = q;
        using var kOwner = k;
        using var vOwner = v;
        using var fgOwner = fg;
        using var skOwner = sk;

        // A NON-uniform output weighting. Under a plain sum(output) the two softmax Jacobians can
        // partially cancel, which would let a wrong softmax backward pass; distinct per-element
        // weights keep every path observable.
        using var weights = new Tensor<double>(Gen(batch * seqLen * modelDim, 77, 1.0),
            new[] { batch, seqLen, modelDim });

        Dictionary<Tensor<double>, Tensor<double>> grads;
        var analytic = new Dictionary<Tensor<double>, double[]>();
        var inputs = new[] { q, k, v, fg, sk };
        using (var tape = new GradientTape<double>())
        {
            using var outp = engine.AbcScanForward(q, k, v, fg, sk, numHeads, InitScale);
            using var weighted = engine.TensorMultiply(outp, weights);
            using var loss = engine.ReduceSum(weighted, new[] { 0, 1, 2 }, keepDims: false);
            grads = tape.ComputeGradients(loss, inputs);
            // Copy inside the tape scope: gradient tensors are pooled buffers and are recycled once
            // the tape is disposed, which would silently fabricate values here.
            foreach (var input in inputs)
            {
                var g = grads[input];
                var copy = new double[g.Length];
                for (int i = 0; i < copy.Length; i++) copy[i] = g.GetFlat(i);
                analytic[input] = copy;
            }
        }

        const double eps = 1e-6;
        var names = new[] { "q", "k", "v", "forgetGate", "slotKeys" };
        for (int ai = 0; ai < inputs.Length; ai++)
        {
            var input = inputs[ai];
            var data = (double[])(object)input.GetDataArray()!;
            var grad = analytic[input];
            for (int i = 0; i < data.Length; i++)
            {
                double orig = data[i];
                data[i] = orig + eps;
                double sp = WeightedForward(engine, q, k, v, fg, sk, numHeads, weights);
                data[i] = orig - eps;
                double sm = WeightedForward(engine, q, k, v, fg, sk, numHeads, weights);
                data[i] = orig;
                double numeric = (sp - sm) / (2.0 * eps);
                double tol = 1e-5 + 1e-4 * Math.Abs(grad[i]);
                Assert.True(Math.Abs(numeric - grad[i]) < tol,
                    $"{names[ai]} grad mismatch at element {i}: analytic={grad[i]}, finite-diff={numeric}");
            }
        }
    }

    [Fact]
    public void GenericPath_MatchesDoublePath()
    {
        // T = float takes the generic implementation rather than the double fast path; the two must
        // agree to float precision, otherwise the layers (which run float) get different numbers.
        var engine = new CpuEngine();
        int batch = 2, seqLen = 3, modelDim = 4, numHeads = 2, numSlots = 3;
        var (q, k, v, fg, sk) = MakeInputs(batch, seqLen, modelDim, numHeads, numSlots, 21);
        using var qOwner = q;
        using var kOwner = k;
        using var vOwner = v;
        using var fgOwner = fg;
        using var skOwner = sk;

        using var expectedTensor = engine.AbcScanForward(q, k, v, fg, sk, numHeads, InitScale);
        var expected = (double[])(object)expectedTensor.GetDataArray()!;

        using var qf = ToFloat(q); using var kf = ToFloat(k); using var vf = ToFloat(v);
        using var fgf = ToFloat(fg); using var skf = ToFloat(sk);
        using var actualTensor = engine.AbcScanForward(qf, kf, vf, fgf, skf, numHeads, InitScale);
        var got = (float[])(object)actualTensor.GetDataArray()!;

        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-5,
                $"generic[{i}] = {got[i]} vs double path {expected[i]}");
    }

    [Fact]
    public void ZeroSlots_FailsWithShapeError()
    {
        var engine = new CpuEngine();
        using var q = new Tensor<double>(new double[4], new[] { 1, 1, 4 });
        using var k = new Tensor<double>(new double[4], new[] { 1, 1, 4 });
        using var v = new Tensor<double>(new double[4], new[] { 1, 1, 4 });
        using var fg = new Tensor<double>(new double[2], new[] { 1, 1, 2 });
        using var slotKeys = new Tensor<double>(Array.Empty<double>(), new[] { 2, 0, 2 });

        var error = Assert.Throws<ArgumentException>(
            () => engine.AbcScanForward(q, k, v, fg, slotKeys, numHeads: 2, slotInitScale: InitScale));
        Assert.Contains("numSlots >= 1", error.Message);
    }

    private static Tensor<float> ToFloat(Tensor<double> t)
    {
        var src = (double[])(object)t.GetDataArray()!;
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
        return new Tensor<float>(dst, t.Shape.ToArray());
    }

    private static double WeightedForward(
        CpuEngine engine, Tensor<double> q, Tensor<double> k, Tensor<double> v,
        Tensor<double> fg, Tensor<double> sk, int numHeads, Tensor<double> weights)
    {
        using var outp = engine.AbcScanForward(q, k, v, fg, sk, numHeads, InitScale);
        var data = (double[])(object)outp.GetDataArray()!;
        var w = (double[])(object)weights.GetDataArray()!;
        double s = 0.0;
        for (int i = 0; i < data.Length; i++) s += data[i] * w[i];
        return s;
    }
}
