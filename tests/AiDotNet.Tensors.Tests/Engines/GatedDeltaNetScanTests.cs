using System;
using System.Linq;
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

    // The paper's rule written as its matrix recurrence (Yang et al., arXiv:2412.06464):
    // S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T, o_t = S_t q_t, keys used as given. Formed with the explicit
    // transition matrix, not the kernels' rank-one update, so it checks their algebra rather than repeating it.
    private static double[] StandardReference(
        double[] Q, double[] K, double[] V, double[] A, double[] B,
        int batch, int seqLen, int modelDim, int numHeads)
    {
        int d = modelDim / numHeads;
        var outp = new double[Q.Length];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                var S = new double[d, d];
                for (int t = 0; t < seqLen; t++)
                {
                    int off = (b * seqLen + t) * modelDim + h * d;
                    int g = (b * seqLen + t) * numHeads + h;
                    var M = new double[d, d];
                    for (int i = 0; i < d; i++)
                        for (int j = 0; j < d; j++)
                            M[i, j] = (i == j ? 1.0 : 0.0) - B[g] * K[off + i] * K[off + j];
                    var next = new double[d, d];
                    for (int i = 0; i < d; i++)
                        for (int j = 0; j < d; j++)
                        {
                            double acc = 0.0;
                            for (int m = 0; m < d; m++) acc += S[i, m] * M[m, j];
                            next[i, j] = A[g] * acc + B[g] * V[off + i] * K[off + j];
                        }
                    S = next;
                    for (int i = 0; i < d; i++)
                    {
                        double o = 0.0;
                        for (int j = 0; j < d; j++) o += S[i, j] * Q[off + j];
                        outp[off + i] = o;
                    }
                }
            }
        return outp;
    }

    // Unit keys per head, as the paper's L2 normalization gives them.
    private static void NormalizeKeysPerHead(Tensor<double> k, int numHeads)
    {
        var data = (double[])(object)k.GetDataArray()!;
        int modelDim = k.Shape[2], d = modelDim / numHeads;
        for (int off = 0; off < data.Length; off += d)
        {
            double ss = 0.0;
            for (int i = 0; i < d; i++) ss += data[off + i] * data[off + i];
            double inv = 1.0 / Math.Sqrt(ss);
            for (int i = 0; i < d; i++) data[off + i] *= inv;
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Standard_Forward_MatchesThePapersMatrixRecurrence(bool unitKeys)
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 6, modelDim = 8, numHeads = 2;
        var (q, k, v, a, b) = MakeInputs(batch, seqLen, modelDim, numHeads, 21);
        if (unitKeys) NormalizeKeysPerHead(k, numHeads);

        var got = (double[])(object)engine.GatedDeltaNetScanForward(q, k, v, a, b, numHeads, GatedDeltaRule.Standard).GetDataArray()!;
        var expected = StandardReference(
            (double[])(object)q.GetDataArray()!, (double[])(object)k.GetDataArray()!,
            (double[])(object)v.GetDataArray()!, (double[])(object)a.GetDataArray()!,
            (double[])(object)b.GetDataArray()!, batch, seqLen, modelDim, numHeads);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-12 * (1 + Math.Abs(expected[i])),
                $"Forward[{i}] = {got[i]:R} vs reference {expected[i]:R}");
    }

    [Fact]
    public void Standard_Differs_FromTheScaledKeysRule()
    {
        // The two rules must not coincide on generic inputs (the decay placement and the key scale both differ).
        var engine = new CpuEngine();
        var (q, k, v, a, b) = MakeInputs(1, 4, 4, 1, 3);
        var standard = engine.GatedDeltaNetScanForward(q, k, v, a, b, 1, GatedDeltaRule.Standard).ToArray();
        var scaled = engine.GatedDeltaNetScanForward(q, k, v, a, b, 1).ToArray();
        Assert.Contains(standard.Zip(scaled, (x, y) => Math.Abs(x - y)), diff => diff > 1e-3);
    }

    [Fact]
    public void Standard_Float_MatchesTheFloat64Reference()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 9, modelDim = 12, numHeads = 3;
        var (q, k, v, a, b) = MakeInputs(batch, seqLen, modelDim, numHeads, 8);
        NormalizeKeysPerHead(k, numHeads);
        // float32 inputs rounded from the float64 ones; the reference runs on the rounded values.
        Tensor<float> F(Tensor<double> x) => new Tensor<float>(x.ToArray().Select(e => (float)e).ToArray(), x.Shape.ToArray());
        double[] R(Tensor<double> x) => x.ToArray().Select(e => (double)(float)e).ToArray();
        var got = engine.GatedDeltaNetScanForward(F(q), F(k), F(v), F(a), F(b), numHeads, GatedDeltaRule.Standard).ToArray();
        var expected = StandardReference(R(q), R(k), R(v), R(a), R(b), batch, seqLen, modelDim, numHeads);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-5 * (1 + Math.Abs(expected[i])),
                $"Forward[{i}] = {got[i]:R} vs reference {expected[i]:R}");
    }

    [Theory]
    [InlineData(GatedDeltaRule.Standard)]
    [InlineData(GatedDeltaRule.ScaledKeys)]
    public void Rule_Backward_MatchesFiniteDifferences(GatedDeltaRule rule)
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, modelDim = 6, numHeads = 2;
        var (q, k, v, a, b) = MakeInputs(batch, seqLen, modelDim, numHeads, 17);
        var w = new Tensor<double>(Gen(batch * seqLen * modelDim, 40), new[] { batch, seqLen, modelDim });

        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var outp = engine.GatedDeltaNetScanForward(q, k, v, a, b, numHeads, rule);
            grads = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(outp, w), null), new[] { q, k, v, a, b });
        }

        double Loss()
        {
            var o = (double[])(object)engine.GatedDeltaNetScanForward(q, k, v, a, b, numHeads, rule).GetDataArray()!;
            var wd = (double[])(object)w.GetDataArray()!;
            double s = 0.0;
            for (int i = 0; i < o.Length; i++) s += o[i] * wd[i];
            return s;
        }

        const double eps = 1e-6;
        foreach (var input in new[] { q, k, v, a, b })
        {
            var data = (double[])(object)input.GetDataArray()!;
            for (int i = 0; i < data.Length; i++)
            {
                double orig = data[i];
                data[i] = orig + eps; double sp = Loss();
                data[i] = orig - eps; double sm = Loss();
                data[i] = orig;
                double numeric = (sp - sm) / (2.0 * eps), analytic = grads[input].GetFlat(i);
                Assert.True(Math.Abs(numeric - analytic) <= 1e-6 + 1e-5 * Math.Abs(analytic),
                    $"{rule} grad mismatch at element {i}: analytic={analytic:R}, finite-diff={numeric:R}");
            }
        }
    }

    [Theory]
    [InlineData(GatedDeltaRule.Standard)]
    [InlineData(GatedDeltaRule.ScaledKeys)]
    public void Rule_CreateGraphGradients_MatchTheFusedBackward(GatedDeltaRule rule)
    {
        // Under create-graph the scan's gradients come from recorded ops (GatedDeltaNetScanGradRecorded); they must
        // equal the fused kernels' gradients.
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6, numHeads = 2;
        var (q, k, v, a, b) = MakeInputs(batch, seqLen, modelDim, numHeads, 29);
        var w = new Tensor<double>(Gen(batch * seqLen * modelDim, 41), new[] { batch, seqLen, modelDim });
        var inputs = new[] { q, k, v, a, b };

        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> Grads(bool createGraph)
        {
            using var tape = new GradientTape<double>();
            var outp = engine.GatedDeltaNetScanForward(q, k, v, a, b, numHeads, rule);
            return tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(outp, w), null), inputs, createGraph: createGraph);
        }

        var fused = Grads(false);
        var recorded = Grads(true);
        foreach (var input in inputs)
        {
            var f = fused[input].ToArray();
            var r = recorded[input].ToArray();
            for (int i = 0; i < f.Length; i++)
                Assert.True(Math.Abs(f[i] - r[i]) <= 1e-12 * (1 + Math.Abs(f[i])), $"{rule} element {i}: fused {f[i]:R}, recorded {r[i]:R}");
        }
    }

    [Fact]
    public void ScaledKeys_Overload_IsTheOriginalRule()
    {
        var engine = new CpuEngine();
        var (q, k, v, a, b) = MakeInputs(2, 5, 6, 2, 13);
        var explicitRule = engine.GatedDeltaNetScanForward(q, k, v, a, b, 2, GatedDeltaRule.ScaledKeys).ToArray();
        var original = engine.GatedDeltaNetScanForward(q, k, v, a, b, 2).ToArray();
        Assert.Equal(original, explicitRule);
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
