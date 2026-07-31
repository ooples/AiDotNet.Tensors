using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused RWKV-7 "Goose" sequence kernel
/// (<see cref="CpuEngine.Rwkv7SequenceForward{T}"/>).
///
/// <para>The kernel implements the paper's generalised delta rule (arXiv:2503.14456, Eq. 17):</para>
/// <code>
///   wkv_t = wkv_{t-1} (diag(w_t) - kappaHat_t^T (a_t (*) kappaHat_t)) + v_t^T kTilde_t
///   o_t   = wkv_t . r_t
/// </code>
/// <para>The forward is checked against an independent reference that builds the FULL transition
/// matrix G_t and multiplies it out (rather than the kernel's fused per-row formulation); the custom
/// autodiff backward is checked against central finite differences of sum(output) on BOTH the double
/// fast path and the generic-T path; and the paper's stated stability property (all eigenvalues of
/// G_t inside the unit interval) is asserted directly.</para>
/// </summary>
public class Rwkv7SequenceTests
{
    // exp(-1/2) — the paper's decay scale in w_t = exp(-e^(-1/2) sigmoid(d_t)).
    private const double DecayScale = 0.60653065971263342;
    private const double NormEps = 1e-12;

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    /// <summary>
    /// Independent reference: materialises G_t = diag(w) - kappaHat^T (a (*) kappaHat) per (b, h, t)
    /// and applies wkv_t = wkv_{t-1} G_t + v_t^T kTilde_t followed by o_t = wkv_t . r_t.
    /// </summary>
    private static double[] ReferenceForward(
        double[] R, double[] KAP, double[] KT, double[] V, double[] D, double[] A,
        int batch, int seqLen, int modelDim, int numHeads)
    {
        int hd = modelDim / numHeads;
        var outp = new double[R.Length];
        var S = new double[hd * hd];
        var Snew = new double[hd * hd];
        var G = new double[hd * hd];
        var kh = new double[hd];
        var w = new double[hd];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                Array.Clear(S, 0, S.Length);
                int hOff = h * hd;
                for (int t = 0; t < seqLen; t++)
                {
                    int off = (b * seqLen + t) * modelDim + hOff;

                    double sumSq = 0.0;
                    for (int i = 0; i < hd; i++) sumSq += KAP[off + i] * KAP[off + i];
                    double n = Math.Sqrt(sumSq + NormEps);
                    for (int i = 0; i < hd; i++)
                    {
                        kh[i] = KAP[off + i] / n;
                        w[i] = Math.Exp(-DecayScale * Sigmoid(D[off + i]));
                    }

                    // G[j, ki] = w[ki]*delta(j,ki) - kappaHat[j] * (a[ki] * kappaHat[ki]).
                    for (int j = 0; j < hd; j++)
                        for (int ki = 0; ki < hd; ki++)
                            G[j * hd + ki] = (j == ki ? w[ki] : 0.0) - kh[j] * A[off + ki] * kh[ki];

                    // Snew = S G + outer(v, kTilde).
                    for (int vi = 0; vi < hd; vi++)
                        for (int ki = 0; ki < hd; ki++)
                        {
                            double acc = 0.0;
                            for (int j = 0; j < hd; j++) acc += S[vi * hd + j] * G[j * hd + ki];
                            Snew[vi * hd + ki] = acc + V[off + vi] * KT[off + ki];
                        }
                    Array.Copy(Snew, S, S.Length);

                    // o[vi] = sum_ki S[vi,ki] * r[ki].
                    for (int vi = 0; vi < hd; vi++)
                    {
                        double acc = 0.0;
                        for (int ki = 0; ki < hd; ki++) acc += S[vi * hd + ki] * R[off + ki];
                        outp[off + vi] = acc;
                    }
                }
            }
        return outp;
    }

    /// <summary>Deterministic pseudo-random values in roughly [-0.9, 0.9] (no Random dependency).</summary>
    private static double[] Gen(int n, int s)
    {
        var arr = new double[n];
        for (int i = 0; i < n; i++)
            arr[i] = Math.Sin(0.7 * (i + 1) + 1.3 * s) * 0.9;
        return arr;
    }

    private readonly record struct Inputs(
        Tensor<double> R, Tensor<double> Kappa, Tensor<double> KTilde,
        Tensor<double> V, Tensor<double> Decay, Tensor<double> Icl);

    private static Inputs MakeInputs(int batch, int seqLen, int modelDim, int seed)
    {
        int n = batch * seqLen * modelDim;
        var shape = new[] { batch, seqLen, modelDim };
        // The in-context learning rate a_t must land in (0,1) — the caller applies the sigmoid.
        var icl = Gen(n, seed + 5);
        for (int i = 0; i < n; i++) icl[i] = Sigmoid(icl[i] * 2.0);
        return new Inputs(
            new Tensor<double>(Gen(n, seed), shape),
            new Tensor<double>(Gen(n, seed + 1), shape),
            new Tensor<double>(Gen(n, seed + 2), shape),
            new Tensor<double>(Gen(n, seed + 3), shape),
            new Tensor<double>(Gen(n, seed + 4), shape),
            new Tensor<double>(icl, shape));
    }

    private static Tensor<double>[] AsArray(in Inputs x)
        => new[] { x.R, x.Kappa, x.KTilde, x.V, x.Decay, x.Icl };

    [Fact]
    public void Forward_MatchesReferenceRecurrence()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6, numHeads = 3;
        var x = MakeInputs(batch, seqLen, modelDim, 10);

        var outp = engine.Rwkv7SequenceForward(
            x.R, x.Kappa, x.KTilde, x.V, x.Decay, x.Icl, numHeads);
        var expected = ReferenceForward(
            (double[])(object)x.R.GetDataArray()!, (double[])(object)x.Kappa.GetDataArray()!,
            (double[])(object)x.KTilde.GetDataArray()!, (double[])(object)x.V.GetDataArray()!,
            (double[])(object)x.Decay.GetDataArray()!, (double[])(object)x.Icl.GetDataArray()!,
            batch, seqLen, modelDim, numHeads);

        var got = (double[])(object)outp.GetDataArray()!;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-10,
                $"Forward[{i}] = {got[i]} vs reference {expected[i]}");
    }

    [Fact]
    public void Forward_GenericPath_MatchesDoublePath()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6, numHeads = 3;
        var x = MakeInputs(batch, seqLen, modelDim, 21);

        var dbl = (double[])(object)engine.Rwkv7SequenceForward(
            x.R, x.Kappa, x.KTilde, x.V, x.Decay, x.Icl, numHeads).GetDataArray()!;

        // float takes the generic-T path (typeof(T) != typeof(double)).
        var flt = (float[])(object)engine.Rwkv7SequenceForward(
            ToFloat(x.R), ToFloat(x.Kappa), ToFloat(x.KTilde),
            ToFloat(x.V), ToFloat(x.Decay), ToFloat(x.Icl), numHeads).GetDataArray()!;

        for (int i = 0; i < dbl.Length; i++)
            Assert.True(Math.Abs(flt[i] - dbl[i]) < 1e-4,
                $"generic-path Forward[{i}] = {flt[i]} vs double path {dbl[i]}");
    }

    private static Tensor<float> ToFloat(Tensor<double> t)
    {
        var src = (double[])(object)t.GetDataArray()!;
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
        return new Tensor<float>(dst, t.Shape.ToArray());
    }

    [Fact]
    public void Backward_DoublePath_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, modelDim = 4, numHeads = 2;
        var x = MakeInputs(batch, seqLen, modelDim, 3);
        var inputs = AsArray(x);

        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var outp = engine.Rwkv7SequenceForward(
                x.R, x.Kappa, x.KTilde, x.V, x.Decay, x.Icl, numHeads);
            grads = tape.ComputeGradients(outp, inputs);
        }

        const double eps = 1e-6;
        foreach (var input in inputs)
        {
            var data = (double[])(object)input.GetDataArray()!;
            var grad = grads[input];
            for (int i = 0; i < data.Length; i++)
            {
                double orig = data[i];
                data[i] = orig + eps;
                double sumPlus = SumForward(engine, x, numHeads);
                data[i] = orig - eps;
                double sumMinus = SumForward(engine, x, numHeads);
                data[i] = orig;
                double numeric = (sumPlus - sumMinus) / (2.0 * eps);
                double analytic = grad.GetFlat(i);
                Assert.True(Math.Abs(numeric - analytic) < 1e-5 + 1e-5 * Math.Abs(numeric),
                    $"grad mismatch at element {i}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static double SumForward(CpuEngine engine, in Inputs x, int numHeads)
    {
        // No active tape here → pure forward, no recording.
        var outp = engine.Rwkv7SequenceForward(
            x.R, x.Kappa, x.KTilde, x.V, x.Decay, x.Icl, numHeads);
        var data = (double[])(object)outp.GetDataArray()!;
        double s = 0.0;
        for (int i = 0; i < data.Length; i++) s += data[i];
        return s;
    }

    /// <summary>
    /// Same finite-difference check on the GENERIC-T code path
    /// (<c>Rwkv7ForwardGeneric</c> / <c>Rwkv7BackwardGeneric</c>). <c>decimal</c> is used rather than
    /// <c>float</c> because it is not the double fast path yet still carries enough precision for
    /// central differences (float would drown the signal in rounding noise, which would only prove
    /// the test is badly conditioned).
    /// </summary>
    [Fact]
    public void Backward_GenericPath_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 3, modelDim = 4, numHeads = 2;
        var d = MakeInputs(batch, seqLen, modelDim, 7);
        var inputs = new[]
        {
            ToDecimal(d.R), ToDecimal(d.Kappa), ToDecimal(d.KTilde),
            ToDecimal(d.V), ToDecimal(d.Decay), ToDecimal(d.Icl),
        };

        System.Collections.Generic.Dictionary<Tensor<decimal>, Tensor<decimal>> grads;
        using (var tape = new GradientTape<decimal>())
        {
            var outp = engine.Rwkv7SequenceForward(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], numHeads);
            grads = tape.ComputeGradients(outp, inputs);
        }

        const decimal eps = 1e-6m;
        foreach (var input in inputs)
        {
            var data = (decimal[])(object)input.GetDataArray()!;
            var grad = grads[input];
            for (int i = 0; i < data.Length; i++)
            {
                decimal orig = data[i];
                data[i] = orig + eps;
                decimal sumPlus = SumForwardDecimal(engine, inputs, numHeads);
                data[i] = orig - eps;
                decimal sumMinus = SumForwardDecimal(engine, inputs, numHeads);
                data[i] = orig;
                double numeric = (double)((sumPlus - sumMinus) / (2m * eps));
                double analytic = (double)grad.GetFlat(i);
                Assert.True(Math.Abs(numeric - analytic) < 1e-4 + 1e-4 * Math.Abs(numeric),
                    $"generic-path grad mismatch at element {i}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static Tensor<decimal> ToDecimal(Tensor<double> t)
    {
        var src = (double[])(object)t.GetDataArray()!;
        var dst = new decimal[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (decimal)src[i];
        return new Tensor<decimal>(dst, t.Shape.ToArray());
    }

    private static decimal SumForwardDecimal(CpuEngine engine, Tensor<decimal>[] x, int numHeads)
    {
        var outp = engine.Rwkv7SequenceForward(x[0], x[1], x[2], x[3], x[4], x[5], numHeads);
        var data = (decimal[])(object)outp.GetDataArray()!;
        decimal s = 0m;
        for (int i = 0; i < data.Length; i++) s += data[i];
        return s;
    }

    /// <summary>
    /// Samples a transition G_t = diag(w) - kappaHat^T (a (*) kappaHat) exactly as the kernel forms it,
    /// plus the symmetric surrogate diag(w) - z z^T with z_i = kappaHat_i * sqrt(a_i).
    /// </summary>
    private static (int Hd, double[] W, double[] Kh, double[] A, double[] G, double[] Sym) SampleTransition(
        Random rng, double minAbsKh, double minA, double maxA)
    {
        int hd = 2 + rng.Next(7); // 2..8
        var kh = new double[hd];
        var w = new double[hd];
        var a = new double[hd];
        double sumSq = 0.0;
        for (int i = 0; i < hd; i++)
        {
            double raw;
            do { raw = rng.NextDouble() * 4.0 - 2.0; } while (Math.Abs(raw) < minAbsKh);
            kh[i] = raw;
            sumSq += raw * raw;
            // Sweep the decay logit across its full range so w covers (exp(-e^(-1/2)), 1).
            w[i] = Math.Exp(-DecayScale * Sigmoid(rng.NextDouble() * 20.0 - 10.0));
            a[i] = minA + rng.NextDouble() * (maxA - minA);
        }
        double n = Math.Sqrt(sumSq + NormEps);
        for (int i = 0; i < hd; i++) kh[i] /= n;

        var g = new double[hd * hd];
        var sym = new double[hd * hd];
        for (int j = 0; j < hd; j++)
            for (int ki = 0; ki < hd; ki++)
            {
                g[j * hd + ki] = (j == ki ? w[ki] : 0.0) - kh[j] * a[ki] * kh[ki];
                double zj = kh[j] * Math.Sqrt(a[j]);
                double zk = kh[ki] * Math.Sqrt(a[ki]);
                sym[j * hd + ki] = (j == ki ? w[ki] : 0.0) - zj * zk;
            }
        return (hd, w, kh, a, g, sym);
    }

    /// <summary>
    /// G_t is NOT symmetric, but because <c>a_i * kappaHat_i^2 &gt;= 0</c> it is diagonally similar to
    /// the symmetric matrix <c>diag(w) - z z^T</c> with <c>z_i = kappaHat_i * sqrt(a_i)</c>: with
    /// <c>P = diag(sqrt(a_i))</c>, <c>P G P^-1 = diag(w) - z z^T</c>. That identity is what makes the
    /// paper's eigenvalue claim provable (real spectrum, Weyl bounds), so it is asserted directly here
    /// before being relied on by <see cref="TransitionMatrix_EigenvaluesWithinUnitInterval"/>.
    /// </summary>
    [Fact]
    public void TransitionMatrix_IsDiagonallySimilarToSymmetricMatrix()
    {
        var rng = new Random(20250727);
        for (int trial = 0; trial < 100; trial++)
        {
            var (hd, _, _, a, g, sym) = SampleTransition(rng, minAbsKh: 0.05, minA: 0.05, maxA: 0.95);
            for (int j = 0; j < hd; j++)
                for (int ki = 0; ki < hd; ki++)
                {
                    // (P G P^-1)[j,ki] = sqrt(a_j) * G[j,ki] / sqrt(a_ki)
                    double conj = Math.Sqrt(a[j]) * g[j * hd + ki] / Math.Sqrt(a[ki]);
                    double expected = sym[j * hd + ki];
                    Assert.True(Math.Abs(conj - expected) < 1e-12 + 1e-12 * Math.Abs(expected),
                        $"similarity mismatch at [{j},{ki}]: {conj} vs {expected}");
                }
        }
    }

    /// <summary>
    /// The paper's stated stability property: the transition
    /// <c>G_t = diag(w_t) - kappaHat_t^T (a_t (*) kappaHat_t)</c> keeps its eigenvalues inside
    /// <c>[-1, 1]</c>. This is what bounds the state — NOT the decay, which the paper deliberately
    /// confines to the WEAKLY contractive range <c>(0.5453, 1)</c>.
    ///
    /// <para>The spectrum is read off the symmetric surrogate (identical eigenvalues, see
    /// <see cref="TransitionMatrix_IsDiagonallySimilarToSymmetricMatrix"/>) via the symmetric
    /// eigensolver, which is accurate to ~1e-14 here. The general non-symmetric QR solver is also run
    /// on G itself as a cross-check that the spectrum is REAL, but only to its own ~1e-3 accuracy —
    /// it reports moduli up to 1.0004 for eigenvalues that are provably &lt;= max(w) &lt; 1.</para>
    /// </summary>
    [Fact]
    public void TransitionMatrix_EigenvaluesWithinUnitInterval()
    {
        var rng = new Random(20250727);
        const int trials = 200;
        double worst = 0.0;
        for (int trial = 0; trial < trials; trial++)
        {
            var (hd, w, kh, a, g, sym) = SampleTransition(rng, minAbsKh: 0.0, minA: 0.0, maxA: 1.0);

            // w must live in (exp(-e^(-1/2)), 1) by construction — the paper's decay range.
            for (int i = 0; i < hd; i++)
                Assert.True(w[i] > 0.5452 && w[i] < 1.0, $"w[{i}] = {w[i]} outside (0.5453, 1)");

            // ||z||^2 = sum_i a_i kappaHat_i^2 <= ||kappaHat||^2 = 1, which is what caps the rank-1
            // removal and yields the Weyl bound [min(w) - ||z||^2, max(w)] on the spectrum.
            double zNormSq = 0.0;
            for (int i = 0; i < hd; i++) zNormSq += a[i] * kh[i] * kh[i];
            Assert.True(zNormSq <= 1.0 + 1e-12, $"||z||^2 = {zNormSq} > 1");

            var evSym = (double[])(object)Linalg.Eigvalsh(
                new Tensor<double>(sym, new[] { hd, hd })).GetDataArray()!;
            for (int i = 0; i < hd; i++)
            {
                double lambda = evSym[i];
                worst = Math.Max(worst, Math.Abs(lambda));
                Assert.True(Math.Abs(lambda) <= 1.0 + 1e-10,
                    $"|eigenvalue| = {Math.Abs(lambda)} > 1 (headDim={hd}, lambda={lambda})");
            }

            // Cross-check on G itself: the spectrum must be real (no complex pairs).
            var evGen = (double[])(object)Linalg.Eigvals(
                new Tensor<double>(g, new[] { hd, hd })).GetDataArray()!; // [hd, 2] (re, im)
            for (int i = 0; i < hd; i++)
            {
                Assert.True(Math.Abs(evGen[i * 2 + 1]) < 1e-6,
                    $"eigenvalue {i} of G is complex (im={evGen[i * 2 + 1]}); the spectrum must be real.");
                Assert.True(Math.Abs(evGen[i * 2]) <= 1.0 + 5e-3,
                    $"|eigenvalue| of G = {Math.Abs(evGen[i * 2])} exceeds 1 beyond QR solver accuracy.");
            }
        }
        // Sanity: the bound is tight (w -> 1 with a -> 0 approaches an eigenvalue of 1), so a
        // vacuous test that never got near the boundary would be suspicious.
        Assert.True(worst > 0.9, $"worst spectral radius over {trials} trials was only {worst}");
    }

    /// <summary>
    /// Regression guard for the failure this recurrence fixes: with the rank-1 removal term in place
    /// the state stays bounded over a long sequence even when every decay channel is pushed to its
    /// least contractive value (w -> 1) — the configuration under which a pure "diagonal decay plus
    /// additive injection" state grows without bound and eventually overflows the readout.
    /// </summary>
    [Fact]
    public void State_StaysBounded_WhenDecayIsSaturated()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 512, modelDim = 8, numHeads = 2;
        int n = batch * seqLen * modelDim;
        var shape = new[] { batch, seqLen, modelDim };

        var decay = new double[n];
        // NEGATIVE, deliberately: w = exp(-e^(-1/2) * sigmoid(D)), so sigmoid(D) -> 0 gives w -> 1,
        // the LEAST contractive end of the paper's (0.5453, 1) range and the only configuration under
        // which "diagonal decay + additive injection" grows without bound. A large POSITIVE decay
        // logit does the opposite: sigmoid(D) -> 1 pins w at exp(-e^(-1/2)) ~ 0.5453, the MOST
        // contractive endpoint, where even the old unbounded recurrence stays small - so the guard
        // would have passed no matter how broken the transition was.
        for (int i = 0; i < n; i++) decay[i] = -30.0;
        var icl = new double[n];
        for (int i = 0; i < n; i++) icl[i] = 0.9; // strong in-context learning rate

        var outp = engine.Rwkv7SequenceForward(
            new Tensor<double>(Gen(n, 31), shape),
            new Tensor<double>(Gen(n, 32), shape),
            new Tensor<double>(Gen(n, 33), shape),
            new Tensor<double>(Gen(n, 34), shape),
            new Tensor<double>(decay, shape),
            new Tensor<double>(icl, shape),
            numHeads);

        var got = (double[])(object)outp.GetDataArray()!;
        double maxAbs = 0.0;
        foreach (var value in got)
        {
            // double.IsFinite does not exist on net471 (netstandard2.0-era BCL), and this suite is
            // multi-targeted — use the NaN/Infinity pair, which is available on both TFMs.
            Assert.True(!double.IsNaN(value) && !double.IsInfinity(value),
                "RWKV-7 readout produced a non-finite value.");
            maxAbs = Math.Max(maxAbs, Math.Abs(value));
        }
        // The injected v (*) kTilde magnitude is O(1) per step; a bounded transition keeps the
        // readout O(10), whereas an unbounded one reaches O(seqLen) or worse.
        Assert.True(maxAbs < 100.0, $"readout magnitude {maxAbs} indicates unbounded state growth.");
    }
}
