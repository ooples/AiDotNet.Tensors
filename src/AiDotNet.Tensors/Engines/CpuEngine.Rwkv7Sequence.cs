using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Scale factor <c>e^(-1/2)</c> in the RWKV-7 decay parameterisation
    /// <c>w_t = exp(-e^(-1/2) * sigmoid(d_t))</c> (arXiv:2503.14456, section 3.1). It bounds
    /// <c>w_t</c> to <c>(exp(-e^(-1/2)), 1) ~ (0.5453, 1)</c>, which the paper reports as better
    /// conditioned than a plain sigmoid decay.
    /// </summary>
    internal const double Rwkv7DecayScale = 0.60653065971263342; // exp(-0.5)

    /// <summary>Guard added to the squared L2 norm before taking the square root of ||kappa||.</summary>
    private const double Rwkv7NormEps = 1e-12;

    /// <summary>
    /// Fused RWKV-7 "Goose" time-mixing WKV recurrence over a whole sequence in a SINGLE op
    /// (forward + custom autodiff backward), replacing the per-timestep tape micro-ops a decomposed
    /// <c>RWKV7Block.TimeMixingForward</c> loop would record (issue #1464). All inputs are the
    /// already-projected, per-position streams <c>[batch, seqLen, modelDim]</c>.
    ///
    /// <para>This implements the paper's generalised delta rule (arXiv:2503.14456 "RWKV-7 Goose",
    /// Eq. 17) exactly — a DIAGONAL decay MINUS a rank-1 removal term, not the RWKV-6/GLA-style
    /// "diagonal decay plus additive injection":</para>
    /// <code>
    ///   wkv_t = wkv_{t-1} (diag(w_t) - kappaHat_t^T (a_t (*) kappaHat_t)) + v_t^T kTilde_t
    ///   o_t   = wkv_t . r_t
    /// </code>
    /// <para>with, per head (modelDim split into <paramref name="numHeads"/> blocks of headDim):</para>
    /// <list type="bullet">
    ///   <item><c>w_t = exp(-e^(-1/2) * sigmoid(decayLogit_t))</c> in <c>(0.5453, 1)</c> — the paper's
    ///   decay parameterisation, applied internally so the recurrence never touches the tape per step.</item>
    ///   <item><c>kappaHat_t = kappa_t / ||kappa_t||_2</c>, L2-normalised PER HEAD (also internal).</item>
    ///   <item><c>a_t</c> = the vector-valued in-context learning rate, already in <c>(0,1)</c>. It is
    ///   passed post-activation (unlike the decay logit) because the caller also needs it to form
    ///   <c>kTilde_t = k_t (*) (1 + (a_t - 1) (*) k_a)</c>; sharing one tensor keeps a single tape node
    ///   for it instead of duplicating the sigmoid.</item>
    /// </list>
    /// <para>The transition <c>G_t = diag(w_t) - kappaHat_t^T (a_t (*) kappaHat_t)</c> is the paper's
    /// "scaled approximation of a Householder matrix". Because <c>a_i * kappaHat_i^2 &gt;= 0</c>, G_t is
    /// diagonally similar to the SYMMETRIC matrix <c>diag(w_t) - z z^T</c> with
    /// <c>z_i = kappaHat_i * sqrt(a_i)</c> and <c>||z||^2 = sum_i a_i kappaHat_i^2 &lt;= 1</c>, so its
    /// eigenvalues are real and (by Weyl) lie in <c>[min(w) - ||z||^2, max(w)] &lt;= [-0.455, 1)</c> —
    /// the stability property the paper states, and the reason the removal term (not the decay) is what
    /// bounds the state. Landing <c>w_t</c> WITHOUT the removal term would be strictly worse than a
    /// plain sigmoid decay, since <c>(0.5453, 1)</c> is less contractive than <c>(0, 1)</c>.</para>
    ///
    /// <para>State orientation follows the paper: <c>wkv</c> is <c>[d_v x d_k]</c>, indexed
    /// <c>S[vi, ki]</c>, and the readout contracts the KEY axis with <c>r_t</c> to produce a
    /// <c>d_v</c>-dimensional output.</para>
    /// </summary>
    /// <param name="rProj">Receptance r_t [batch, seqLen, modelDim]. Used raw (no gate) — the paper
    /// contracts the state with r; any output gating/normalisation belongs to the caller.</param>
    /// <param name="kappa">kappa_t [batch, seqLen, modelDim], PRE-normalisation; L2-normalised per head
    /// inside the kernel. In the reference implementation kappa_t = k_t (*) k_k.</param>
    /// <param name="kTilde">kTilde_t [batch, seqLen, modelDim], the value-injection key
    /// k_t (*) (1 + (a_t - 1) (*) k_a).</param>
    /// <param name="vProj">Value projection v_t [batch, seqLen, modelDim].</param>
    /// <param name="decayLogit">Decay pre-activation d_t [batch, seqLen, modelDim];
    /// w_t = exp(-e^(-1/2) * sigmoid(d_t)) is applied internally.</param>
    /// <param name="iclRate">In-context learning rate a_t [batch, seqLen, modelDim], already in (0,1).</param>
    /// <param name="numHeads">Number of heads; modelDim must be divisible by it.</param>
    /// <returns>The WKV readout o_t [batch, seqLen, modelDim].</returns>
    public virtual Tensor<T> Rwkv7SequenceForward<T>(
        Tensor<T> rProj, Tensor<T> kappa, Tensor<T> kTilde, Tensor<T> vProj,
        Tensor<T> decayLogit, Tensor<T> iclRate,
        int numHeads)
    {
        if (rProj is null) throw new ArgumentNullException(nameof(rProj));
        if (kappa is null) throw new ArgumentNullException(nameof(kappa));
        if (kTilde is null) throw new ArgumentNullException(nameof(kTilde));
        if (vProj is null) throw new ArgumentNullException(nameof(vProj));
        if (decayLogit is null) throw new ArgumentNullException(nameof(decayLogit));
        if (iclRate is null) throw new ArgumentNullException(nameof(iclRate));
        if (numHeads < 1) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (rProj.Rank != 3)
            throw new ArgumentException($"Rwkv7SequenceForward expects rank-3 inputs [batch, seqLen, modelDim]; got rank {rProj.Rank}.", nameof(rProj));

        int batch = rProj.Shape[0];
        int seqLen = rProj.Shape[1];
        int modelDim = rProj.Shape[2];
        if (modelDim % numHeads != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        int headDim = modelDim / numHeads;

        EnsureSameShape(rProj, kappa, nameof(kappa));
        EnsureSameShape(rProj, kTilde, nameof(kTilde));
        EnsureSameShape(rProj, vProj, nameof(vProj));
        EnsureSameShape(rProj, decayLogit, nameof(decayLogit));
        EnsureSameShape(rProj, iclRate, nameof(iclRate));

        var output = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            Rwkv7ForwardDouble(
                (double[])(object)rProj.GetDataArray()!, (double[])(object)kappa.GetDataArray()!,
                (double[])(object)kTilde.GetDataArray()!, (double[])(object)vProj.GetDataArray()!,
                (double[])(object)decayLogit.GetDataArray()!, (double[])(object)iclRate.GetDataArray()!,
                (double[])(object)output.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            Rwkv7ForwardGeneric<T>(
                rProj.GetDataArray()!, kappa.GetDataArray()!, kTilde.GetDataArray()!,
                vProj.GetDataArray()!, decayLogit.GetDataArray()!, iclRate.GetDataArray()!,
                output.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }

        // Record ONE tape node for the whole recurrence with a custom BPTT backward.
        DifferentiableOps.RecordIfActive<T>(
            "Rwkv7Sequence", output,
            new[] { rProj, kappa, kTilde, vProj, decayLogit, iclRate },
            Rwkv7SequenceBackward<T>,
            savedState: new object[] { numHeads });

        return output;
    }

    private static void EnsureSameShape<T>(Tensor<T> reference, Tensor<T> other, string paramName)
    {
        if (other.Rank != reference.Rank)
            throw new ArgumentException($"{paramName} rank ({other.Rank}) must match ({reference.Rank}).", paramName);
        for (int i = 0; i < reference.Rank; i++)
            if (other.Shape[i] != reference.Shape[i])
                throw new ArgumentException($"{paramName} dim {i} ({other.Shape[i]}) must match ({reference.Shape[i]}).", paramName);
    }

    // Validates a scalar-per-head gate tensor shaped [batch, seqLen, numHeads].
    private static void EnsureGateShape<T>(Tensor<T> gate, int batch, int seqLen, int numHeads, string paramName)
    {
        if (gate.Rank != 3 || gate.Shape[0] != batch || gate.Shape[1] != seqLen || gate.Shape[2] != numHeads)
            throw new ArgumentException($"{paramName} must be [batch={batch}, seqLen={seqLen}, numHeads={numHeads}].", paramName);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static double Sig(double x) => 1.0 / (1.0 + Math.Exp(-x));

    // ── Double fast path ────────────────────────────────────────────────────────────────
    private static void Rwkv7ForwardDouble(
        double[] R, double[] KAP, double[] KT, double[] V, double[] D, double[] A, double[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        // Each (batch, head) pair is fully independent (private state, disjoint output); parallelize
        // lock-free over the combined (b*numHeads) axis. See GlaForwardDouble for the pattern.
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var S = new double[hh];
            var kh = new double[headDim];
            var w = new double[headDim];
            var a = new double[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                Array.Clear(S, 0, hh);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;

                    // Per-head gates: kappaHat (L2-normalised), the paper's decay w, and the ICL rate a.
                    double sumSq = 0.0;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        double kap = KAP[baseOff + ki];
                        sumSq += kap * kap;
                    }
                    double invN = 1.0 / Math.Sqrt(sumSq + Rwkv7NormEps);
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        kh[ki] = KAP[baseOff + ki] * invN;
                        w[ki] = Math.Exp(-Rwkv7DecayScale * Sig(D[baseOff + ki]));
                        a[ki] = A[baseOff + ki];
                    }

                    // S_t[vi,ki] = S_{t-1}[vi,ki]*w[ki] - (S_{t-1}[vi,:].kappaHat)*a[ki]*kappaHat[ki]
                    //              + v[vi]*kTilde[ki];   o[vi] = sum_ki S_t[vi,ki]*r[ki].
                    // Row vi of S depends only on row vi of S_{t-1}, so the removal projection, the
                    // update and the readout all fuse into ONE pass per row (no extra scratch).
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        double p = 0.0;
                        for (int ki = 0; ki < headDim; ki++)
                            p += S[srow + ki] * kh[ki];
                        double vv = V[baseOff + vi];
                        double o = 0.0;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            double sv = S[srow + ki] * w[ki] - p * a[ki] * kh[ki] + vv * KT[baseOff + ki];
                            S[srow + ki] = sv;
                            o += sv * R[baseOff + ki];
                        }
                        outp[baseOff + vi] = o;
                    }
                }
            }
        });
    }

    private static void Rwkv7BackwardDouble(
        double[] dOut, double[] R, double[] KAP, double[] KT, double[] V, double[] D, double[] A,
        double[] dR, double[] dKAP, double[] dKT, double[] dV, double[] dD, double[] dA,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        // Lock-free over the independent (b*numHeads) axis with per-chunk scratch; see GlaBackwardDouble.
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var Straj = new double[seqLen * hh]; // S_t (post-update) for every t, reused per (b,h)
            var S = new double[hh];
            var dS = new double[hh];
            var kh = new double[headDim];
            var w = new double[headDim];
            var sD = new double[headDim];   // sigmoid(decayLogit), needed for dw/dd
            var a = new double[headDim];
            var p = new double[headDim];    // p[vi] = S_{t-1}[vi,:] . kappaHat
            var dp = new double[headDim];
            var dwAcc = new double[headDim];
            var mAcc = new double[headDim]; // m[ki] = sum_vi dS_t[vi,ki] * p[vi]
            var dktAcc = new double[headDim];
            var dkhAcc = new double[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;

                // Forward recompute, saving the full state trajectory.
                Array.Clear(S, 0, hh);
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    double sumSq = 0.0;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        double kap = KAP[baseOff + ki];
                        sumSq += kap * kap;
                    }
                    double invN = 1.0 / Math.Sqrt(sumSq + Rwkv7NormEps);
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        kh[ki] = KAP[baseOff + ki] * invN;
                        w[ki] = Math.Exp(-Rwkv7DecayScale * Sig(D[baseOff + ki]));
                    }
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        double pv = 0.0;
                        for (int ki = 0; ki < headDim; ki++)
                            pv += S[srow + ki] * kh[ki];
                        double vv = V[baseOff + vi];
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = S[srow + ki] * w[ki] - pv * A[baseOff + ki] * kh[ki] + vv * KT[baseOff + ki];
                    }
                    Array.Copy(S, 0, Straj, t * hh, hh);
                }

                // Backward sweep over t (dS carries the adjoint of S_t from step t+1).
                Array.Clear(dS, 0, hh);
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int stOff = t * hh;
                    int sprevOff = (t - 1) * hh;

                    // Recompute the per-step gates and p (from S_{t-1}).
                    double sumSq = 0.0;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        double kap = KAP[baseOff + ki];
                        sumSq += kap * kap;
                    }
                    double n = Math.Sqrt(sumSq + Rwkv7NormEps);
                    double invN = 1.0 / n;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        kh[ki] = KAP[baseOff + ki] * invN;
                        double s = Sig(D[baseOff + ki]);
                        sD[ki] = s;
                        w[ki] = Math.Exp(-Rwkv7DecayScale * s);
                        a[ki] = A[baseOff + ki];
                    }
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        double pv = 0.0;
                        if (t > 0)
                            for (int ki = 0; ki < headDim; ki++)
                                pv += Straj[sprevOff + srow + ki] * kh[ki];
                        p[vi] = pv;
                    }

                    // 1) Readout backward: o[vi] = sum_ki S_t[vi,ki]*r[ki].
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        double dov = dOut[baseOff + vi];
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            dR[baseOff + ki] += dov * Straj[stOff + srow + ki];
                            dS[srow + ki] += dov * R[baseOff + ki];
                        }
                    }

                    // 2) Update backward. With q = dS_t:
                    //      dS_{t-1}[vi,ki] (direct) = q[vi,ki]*w[ki]
                    //      dw[ki]  = sum_vi q[vi,ki]*S_{t-1}[vi,ki]
                    //      dp[vi]  = -sum_ki q[vi,ki]*a[ki]*kappaHat[ki]
                    //      m[ki]   = sum_vi q[vi,ki]*p[vi]  ->  da[ki] = -kappaHat[ki]*m[ki],
                    //                                          dkappaHat[ki] -= a[ki]*m[ki]
                    //      dv[vi]  = sum_ki q[vi,ki]*kTilde[ki];  dkTilde[ki] = sum_vi q[vi,ki]*v[vi]
                    Array.Clear(dwAcc, 0, headDim);
                    Array.Clear(mAcc, 0, headDim);
                    Array.Clear(dktAcc, 0, headDim);
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        double pv = p[vi];
                        double vv = V[baseOff + vi];
                        double dpv = 0.0, dvv = 0.0;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            double q = dS[srow + ki];
                            double sprev = t > 0 ? Straj[sprevOff + srow + ki] : 0.0;
                            dwAcc[ki] += q * sprev;
                            mAcc[ki] += q * pv;
                            dpv -= q * a[ki] * kh[ki];
                            dvv += q * KT[baseOff + ki];
                            dktAcc[ki] += q * vv;
                        }
                        dp[vi] = dpv;
                        dV[baseOff + vi] += dvv;
                    }

                    for (int ki = 0; ki < headDim; ki++)
                    {
                        double m = mAcc[ki];
                        dkhAcc[ki] = -a[ki] * m;                      // removal-term contribution
                        dA[baseOff + ki] += -kh[ki] * m;              // a_t arrives post-activation
                        // w = exp(-c*s), s = sigmoid(d)  =>  dw/dd = (-c*w) * s*(1-s)
                        dD[baseOff + ki] += dwAcc[ki] * (-Rwkv7DecayScale * w[ki]) * sD[ki] * (1.0 - sD[ki]);
                        dKT[baseOff + ki] += dktAcc[ki];
                    }

                    // 3) p backward (p[vi] = sum_ki S_{t-1}[vi,ki]*kappaHat[ki]) fused with forming
                    //    dS_{t-1} = q*w + outer(dp, kappaHat).
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        double dpv = dp[vi];
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            if (t > 0) dkhAcc[ki] += dpv * Straj[sprevOff + srow + ki];
                            dS[srow + ki] = dS[srow + ki] * w[ki] + dpv * kh[ki];
                        }
                    }

                    // 4) kappaHat backward: d(kappa_j) = (u_j - kappaHat_j * (u.kappaHat)) / n.
                    double dot = 0.0;
                    for (int ki = 0; ki < headDim; ki++)
                        dot += dkhAcc[ki] * kh[ki];
                    for (int ki = 0; ki < headDim; ki++)
                        dKAP[baseOff + ki] += (dkhAcc[ki] - kh[ki] * dot) * invN;
                }
            }
        });
    }

    // ── Generic-T path (correct for any numeric T; used for non-double) ──────────────────
    private static void Rwkv7ForwardGeneric<T>(
        T[] R, T[] KAP, T[] KT, T[] V, T[] D, T[] A, T[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T normEps = ops.FromDouble(Rwkv7NormEps);
        T negDecayScale = ops.FromDouble(-Rwkv7DecayScale);
        int hh = headDim * headDim;
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var S = new T[hh];
            var kh = new T[headDim];
            var w = new T[headDim];
            var a = new T[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                for (int i = 0; i < hh; i++) S[i] = ops.Zero;
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;

                    T sumSq = normEps;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        T kap = KAP[baseOff + ki];
                        sumSq = ops.Add(sumSq, ops.Multiply(kap, kap));
                    }
                    // Guard the DIVISOR, do not rely on the seeded epsilon. Rwkv7NormEps (1e-12)
                    // flushes to zero under Half/BFloat16, so a zero-kappa head would give
                    // 1/sqrt(0) = +Inf and then kh = 0 * Inf = NaN. Falling back to invN = 0 yields
                    // kappaHat = 0, i.e. the removal term vanishes and the transition degenerates to
                    // diag(w) - the correct limit, since a zero removal key removes nothing.
                    T normK = ops.Sqrt(sumSq);
                    T invN = ops.GreaterThan(normK, ops.Zero) ? ops.Divide(ops.One, normK) : ops.Zero;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        kh[ki] = ops.Multiply(KAP[baseOff + ki], invN);
                        w[ki] = ops.Exp(ops.Multiply(negDecayScale, SigGeneric(ops, D[baseOff + ki])));
                        a[ki] = A[baseOff + ki];
                    }

                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        T pv = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++)
                            pv = ops.Add(pv, ops.Multiply(S[srow + ki], kh[ki]));
                        T vv = V[baseOff + vi];
                        T o = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            T sv = ops.Add(
                                ops.Subtract(
                                    ops.Multiply(S[srow + ki], w[ki]),
                                    ops.Multiply(pv, ops.Multiply(a[ki], kh[ki]))),
                                ops.Multiply(vv, KT[baseOff + ki]));
                            S[srow + ki] = sv;
                            o = ops.Add(o, ops.Multiply(sv, R[baseOff + ki]));
                        }
                        outp[baseOff + vi] = o;
                    }
                }
            }
        });
    }

    private static void Rwkv7BackwardGeneric<T>(
        T[] dOut, T[] R, T[] KAP, T[] KT, T[] V, T[] D, T[] A,
        T[] dR, T[] dKAP, T[] dKT, T[] dV, T[] dD, T[] dA,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hh = headDim * headDim;
        T one = ops.One;
        T normEps = ops.FromDouble(Rwkv7NormEps);
        T negDecayScale = ops.FromDouble(-Rwkv7DecayScale);
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var Straj = new T[seqLen * hh];
            var S = new T[hh];
            var dS = new T[hh];
            var kh = new T[headDim];
            var w = new T[headDim];
            var sD = new T[headDim];
            var a = new T[headDim];
            var p = new T[headDim];
            var dp = new T[headDim];
            var dwAcc = new T[headDim];
            var mAcc = new T[headDim];
            var dktAcc = new T[headDim];
            var dkhAcc = new T[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;

                for (int i = 0; i < hh; i++) S[i] = ops.Zero;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    T sumSq = normEps;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        T kap = KAP[baseOff + ki];
                        sumSq = ops.Add(sumSq, ops.Multiply(kap, kap));
                    }
                    // See the forward's normalisation guard: the seeded epsilon flushes to zero in
                    // Half/BFloat16, so divide only when the norm is genuinely positive.
                    T normK0 = ops.Sqrt(sumSq);
                    T invN0 = ops.GreaterThan(normK0, ops.Zero) ? ops.Divide(one, normK0) : ops.Zero;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        kh[ki] = ops.Multiply(KAP[baseOff + ki], invN0);
                        w[ki] = ops.Exp(ops.Multiply(negDecayScale, SigGeneric(ops, D[baseOff + ki])));
                    }
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        T pv = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++)
                            pv = ops.Add(pv, ops.Multiply(S[srow + ki], kh[ki]));
                        T vv = V[baseOff + vi];
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = ops.Add(
                                ops.Subtract(
                                    ops.Multiply(S[srow + ki], w[ki]),
                                    ops.Multiply(pv, ops.Multiply(A[baseOff + ki], kh[ki]))),
                                ops.Multiply(vv, KT[baseOff + ki]));
                    }
                    Array.Copy(S, 0, Straj, t * hh, hh);
                }

                for (int i = 0; i < hh; i++) dS[i] = ops.Zero;
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int stOff = t * hh;
                    int sprevOff = (t - 1) * hh;

                    T sumSq = normEps;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        T kap = KAP[baseOff + ki];
                        sumSq = ops.Add(sumSq, ops.Multiply(kap, kap));
                    }
                    // See the forward's normalisation guard: the seeded epsilon flushes to zero in
                    // Half/BFloat16, so divide only when the norm is genuinely positive.
                    T normKb = ops.Sqrt(sumSq);
                    T invN = ops.GreaterThan(normKb, ops.Zero) ? ops.Divide(one, normKb) : ops.Zero;
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        kh[ki] = ops.Multiply(KAP[baseOff + ki], invN);
                        T s = SigGeneric(ops, D[baseOff + ki]);
                        sD[ki] = s;
                        w[ki] = ops.Exp(ops.Multiply(negDecayScale, s));
                        a[ki] = A[baseOff + ki];
                    }
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        T pv = ops.Zero;
                        if (t > 0)
                            for (int ki = 0; ki < headDim; ki++)
                                pv = ops.Add(pv, ops.Multiply(Straj[sprevOff + srow + ki], kh[ki]));
                        p[vi] = pv;
                    }

                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        T dov = dOut[baseOff + vi];
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            dR[baseOff + ki] = ops.Add(dR[baseOff + ki], ops.Multiply(dov, Straj[stOff + srow + ki]));
                            dS[srow + ki] = ops.Add(dS[srow + ki], ops.Multiply(dov, R[baseOff + ki]));
                        }
                    }

                    for (int ki = 0; ki < headDim; ki++)
                    {
                        dwAcc[ki] = ops.Zero;
                        mAcc[ki] = ops.Zero;
                        dktAcc[ki] = ops.Zero;
                    }
                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        T pv = p[vi];
                        T vv = V[baseOff + vi];
                        T dpv = ops.Zero, dvv = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            T q = dS[srow + ki];
                            T sprev = t > 0 ? Straj[sprevOff + srow + ki] : ops.Zero;
                            dwAcc[ki] = ops.Add(dwAcc[ki], ops.Multiply(q, sprev));
                            mAcc[ki] = ops.Add(mAcc[ki], ops.Multiply(q, pv));
                            dpv = ops.Subtract(dpv, ops.Multiply(q, ops.Multiply(a[ki], kh[ki])));
                            dvv = ops.Add(dvv, ops.Multiply(q, KT[baseOff + ki]));
                            dktAcc[ki] = ops.Add(dktAcc[ki], ops.Multiply(q, vv));
                        }
                        dp[vi] = dpv;
                        dV[baseOff + vi] = ops.Add(dV[baseOff + vi], dvv);
                    }

                    for (int ki = 0; ki < headDim; ki++)
                    {
                        T m = mAcc[ki];
                        dkhAcc[ki] = ops.Negate(ops.Multiply(a[ki], m));
                        dA[baseOff + ki] = ops.Subtract(dA[baseOff + ki], ops.Multiply(kh[ki], m));
                        T dwdd = ops.Multiply(
                            ops.Multiply(negDecayScale, w[ki]),
                            ops.Multiply(sD[ki], ops.Subtract(one, sD[ki])));
                        dD[baseOff + ki] = ops.Add(dD[baseOff + ki], ops.Multiply(dwAcc[ki], dwdd));
                        dKT[baseOff + ki] = ops.Add(dKT[baseOff + ki], dktAcc[ki]);
                    }

                    for (int vi = 0; vi < headDim; vi++)
                    {
                        int srow = vi * headDim;
                        T dpv = dp[vi];
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            if (t > 0)
                                dkhAcc[ki] = ops.Add(dkhAcc[ki], ops.Multiply(dpv, Straj[sprevOff + srow + ki]));
                            dS[srow + ki] = ops.Add(ops.Multiply(dS[srow + ki], w[ki]), ops.Multiply(dpv, kh[ki]));
                        }
                    }

                    T dot = ops.Zero;
                    for (int ki = 0; ki < headDim; ki++)
                        dot = ops.Add(dot, ops.Multiply(dkhAcc[ki], kh[ki]));
                    for (int ki = 0; ki < headDim; ki++)
                        dKAP[baseOff + ki] = ops.Add(dKAP[baseOff + ki],
                            ops.Multiply(ops.Subtract(dkhAcc[ki], ops.Multiply(kh[ki], dot)), invN));
                }
            }
        });
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static T SigGeneric<T>(INumericOperations<T> ops, T x)
        => ops.Divide(ops.One, ops.Add(ops.One, ops.Exp(ops.Multiply(ops.FromDouble(-1.0), x))));

    private static void Rwkv7SequenceBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int numHeads = (int)savedState[0];
        var rProj = inputs[0];
        var kappa = inputs[1];
        var kTilde = inputs[2];
        var vProj = inputs[3];
        var decayLogit = inputs[4];
        var iclRate = inputs[5];

        int batch = rProj.Shape[0];
        int seqLen = rProj.Shape[1];
        int modelDim = rProj.Shape[2];
        int headDim = modelDim / numHeads;

        var dR = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dKAP = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dKT = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dV = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dD = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dA = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            Rwkv7BackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)rProj.GetDataArray()!, (double[])(object)kappa.GetDataArray()!,
                (double[])(object)kTilde.GetDataArray()!, (double[])(object)vProj.GetDataArray()!,
                (double[])(object)decayLogit.GetDataArray()!, (double[])(object)iclRate.GetDataArray()!,
                (double[])(object)dR.GetDataArray()!, (double[])(object)dKAP.GetDataArray()!,
                (double[])(object)dKT.GetDataArray()!, (double[])(object)dV.GetDataArray()!,
                (double[])(object)dD.GetDataArray()!, (double[])(object)dA.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            Rwkv7BackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                rProj.GetDataArray()!, kappa.GetDataArray()!, kTilde.GetDataArray()!,
                vProj.GetDataArray()!, decayLogit.GetDataArray()!, iclRate.GetDataArray()!,
                dR.GetDataArray()!, dKAP.GetDataArray()!, dKT.GetDataArray()!,
                dV.GetDataArray()!, dD.GetDataArray()!, dA.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }

        DifferentiableOps.AccumulateGrad(grads, rProj, dR, engine);
        DifferentiableOps.AccumulateGrad(grads, kappa, dKAP, engine);
        DifferentiableOps.AccumulateGrad(grads, kTilde, dKT, engine);
        DifferentiableOps.AccumulateGrad(grads, vProj, dV, engine);
        DifferentiableOps.AccumulateGrad(grads, decayLogit, dD, engine);
        DifferentiableOps.AccumulateGrad(grads, iclRate, dA, engine);
    }
}
