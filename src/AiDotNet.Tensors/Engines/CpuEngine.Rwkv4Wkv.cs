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
    /// Fused RWKV-4 time-mixing WKV recurrence over a whole sequence in a SINGLE op
    /// (forward + custom autodiff backward), replacing the ~12 per-timestep tape micro-ops the
    /// decomposed <c>RWKVLayer.TimeMixingForward</c> loop records — the dominant training cost on
    /// long sequences (issue ooples/AiDotNet#1464). The projected receptance/key/value streams
    /// <c>[batch, seqLen, modelDim]</c> are produced by the caller (tape-connected GEMMs); this op
    /// applies the per-position receptance sigmoid internally so the recurrence and its adjoint
    /// never touch the tape per step.
    ///
    /// <para>Paper-faithful RWKV-4 WKV (Peng et al. 2023), the official numerically-stable kernel:
    /// per-CHANNEL scalar state (<c>aa</c> = weighted value sum, <c>bb</c> = weight sum) with a
    /// running max <c>pp</c> so the exponentials never overflow. <c>w[c] = -exp(timeDecay[c])</c>
    /// (the learned static per-channel decay, &lt; 0) and <c>u[c] = timeFirst[c]</c> (the learned
    /// per-channel bonus) are input-independent. For each channel c, with aa=bb=0, pp=-inf:</para>
    /// <code>
    ///   ww = u + k_t;  q = max(pp, ww);  e1 = e^{pp-q};  e2 = e^{ww-q}
    ///   wkv_t = (e1*aa + e2*v_t) / (e1*bb + e2)
    ///   out_t = sigmoid(r_t) * wkv_t
    ///   ww2 = pp + w;  q2 = max(ww2, k_t);  e1b = e^{ww2-q2};  e2b = e^{k_t-q2}
    ///   aa <- e1b*aa + e2b*v_t;  bb <- e1b*bb + e2b;  pp <- q2
    /// </code>
    /// <para>Backward: the running max (<c>q</c>, <c>q2</c>/<c>pp</c>) is a pure stabilizer — it
    /// multiplies numerator and denominator identically and cancels in every ratio, so its gradient
    /// contribution is exactly zero. We therefore treat the max outputs as constants (stop-gradient)
    /// and run clean BPTT through the <c>aa</c>/<c>bb</c> recurrence. This yields the exact gradient
    /// of the (mathematically max-invariant) WKV function, matching the per-timestep tape result.</para>
    /// </summary>
    /// <param name="rProj">Receptance projection [batch, seqLen, modelDim] (pre-sigmoid).</param>
    /// <param name="kProj">Key projection [batch, seqLen, modelDim].</param>
    /// <param name="vProj">Value projection [batch, seqLen, modelDim].</param>
    /// <param name="timeDecay">Per-channel decay parameter [modelDim] (raw; the kernel applies w = -exp(timeDecay)).</param>
    /// <param name="timeFirst">Per-channel bonus parameter [modelDim] (u, applied directly).</param>
    /// <returns>The gated WKV output [batch, seqLen, modelDim] (before the output projection).</returns>
    public virtual Tensor<T> Rwkv4WkvForward<T>(
        Tensor<T> rProj, Tensor<T> kProj, Tensor<T> vProj, Tensor<T> timeDecay, Tensor<T> timeFirst)
    {
        if (rProj is null) throw new ArgumentNullException(nameof(rProj));
        if (kProj is null) throw new ArgumentNullException(nameof(kProj));
        if (vProj is null) throw new ArgumentNullException(nameof(vProj));
        if (timeDecay is null) throw new ArgumentNullException(nameof(timeDecay));
        if (timeFirst is null) throw new ArgumentNullException(nameof(timeFirst));
        if (rProj.Rank != 3)
            throw new ArgumentException($"Rwkv4WkvForward expects rank-3 inputs [batch, seqLen, modelDim]; got rank {rProj.Rank}.", nameof(rProj));

        int batch = rProj.Shape[0];
        int seqLen = rProj.Shape[1];
        int modelDim = rProj.Shape[2];

        EnsureSameShape(rProj, kProj, nameof(kProj));
        EnsureSameShape(rProj, vProj, nameof(vProj));
        if (timeDecay.Length != modelDim)
            throw new ArgumentException($"timeDecay length ({timeDecay.Length}) must equal modelDim ({modelDim}).", nameof(timeDecay));
        if (timeFirst.Length != modelDim)
            throw new ArgumentException($"timeFirst length ({timeFirst.Length}) must equal modelDim ({modelDim}).", nameof(timeFirst));

        var output = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            Rwkv4ForwardDouble(
                (double[])(object)rProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)timeDecay.GetDataArray()!,
                (double[])(object)timeFirst.GetDataArray()!, (double[])(object)output.GetDataArray()!,
                batch, seqLen, modelDim);
        }
        else
        {
            Rwkv4ForwardGeneric<T>(
                rProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!,
                timeDecay.GetDataArray()!, timeFirst.GetDataArray()!, output.GetDataArray()!,
                batch, seqLen, modelDim);
        }

        // Record ONE tape node for the whole recurrence with a custom BPTT backward.
        DifferentiableOps.RecordIfActive<T>(
            "Rwkv4Wkv", output,
            new[] { rProj, kProj, vProj, timeDecay, timeFirst },
            Rwkv4WkvBackward<T>,
            savedState: null);

        return output;
    }

    // ── Double fast path ─────────────────────────────────────────────────────────────────
    private static void Rwkv4ForwardDouble(
        double[] R, double[] K, double[] V, double[] timeDecay, double[] timeFirst, double[] outp,
        int batch, int seqLen, int modelDim)
    {
        var w = new double[modelDim];
        for (int c = 0; c < modelDim; c++) w[c] = -Math.Exp(timeDecay[c]);
        // u = timeFirst (used directly).

        var aa = new double[modelDim];
        var bb = new double[modelDim];
        var pp = new double[modelDim];
        for (int b = 0; b < batch; b++)
        {
            Array.Clear(aa, 0, modelDim);
            Array.Clear(bb, 0, modelDim);
            for (int c = 0; c < modelDim; c++) pp[c] = double.NegativeInfinity;
            for (int t = 0; t < seqLen; t++)
            {
                int baseOff = (b * seqLen + t) * modelDim;
                for (int c = 0; c < modelDim; c++)
                {
                    double k = K[baseOff + c];
                    double v = V[baseOff + c];
                    double u = timeFirst[c];

                    // Output for token t (current key boosted by the time-first bonus u).
                    double ww = u + k;
                    double q = Math.Max(pp[c], ww);
                    double e1 = Math.Exp(pp[c] - q);
                    double e2 = Math.Exp(ww - q);
                    double wkv = (e1 * aa[c] + e2 * v) / (e1 * bb[c] + e2);
                    outp[baseOff + c] = Sig(R[baseOff + c]) * wkv;

                    // State update with the static decay w (no bonus on the carried state).
                    double ww2 = pp[c] + w[c];
                    double q2 = Math.Max(ww2, k);
                    double e1b = Math.Exp(ww2 - q2);
                    double e2b = Math.Exp(k - q2);
                    aa[c] = e1b * aa[c] + e2b * v;
                    bb[c] = e1b * bb[c] + e2b;
                    pp[c] = q2;
                }
            }
        }
    }

    private static void Rwkv4BackwardDouble(
        double[] dOut, double[] R, double[] K, double[] V, double[] timeDecay, double[] timeFirst,
        double[] dR, double[] dK, double[] dV, double[] dTimeDecay, double[] dTimeFirst,
        int batch, int seqLen, int modelDim)
    {
        var w = new double[modelDim];
        for (int c = 0; c < modelDim; c++) w[c] = -Math.Exp(timeDecay[c]);

        // Post-update state trajectories for every t (reused per batch).
        var aaTraj = new double[seqLen * modelDim];
        var bbTraj = new double[seqLen * modelDim];
        var ppTraj = new double[seqLen * modelDim];
        var aa = new double[modelDim];
        var bb = new double[modelDim];
        var pp = new double[modelDim];
        var dAa = new double[modelDim];
        var dBb = new double[modelDim];
        // dw accumulates the grad of w[c] = -exp(timeDecay[c]); converted to dTimeDecay at the end.
        var dw = new double[modelDim];

        for (int b = 0; b < batch; b++)
        {
            // Forward recompute, saving the post-update state trajectory.
            Array.Clear(aa, 0, modelDim);
            Array.Clear(bb, 0, modelDim);
            for (int c = 0; c < modelDim; c++) pp[c] = double.NegativeInfinity;
            for (int t = 0; t < seqLen; t++)
            {
                int baseOff = (b * seqLen + t) * modelDim;
                int tOff = t * modelDim;
                for (int c = 0; c < modelDim; c++)
                {
                    double k = K[baseOff + c];
                    double v = V[baseOff + c];
                    double ww2 = pp[c] + w[c];
                    double q2 = Math.Max(ww2, k);
                    double e1b = Math.Exp(ww2 - q2);
                    double e2b = Math.Exp(k - q2);
                    aa[c] = e1b * aa[c] + e2b * v;
                    bb[c] = e1b * bb[c] + e2b;
                    pp[c] = q2;
                    aaTraj[tOff + c] = aa[c];
                    bbTraj[tOff + c] = bb[c];
                    ppTraj[tOff + c] = pp[c];
                }
            }

            // Reverse sweep. dAa/dBb carry the adjoint of the post-update state from t+1.
            Array.Clear(dAa, 0, modelDim);
            Array.Clear(dBb, 0, modelDim);
            for (int t = seqLen - 1; t >= 0; t--)
            {
                int baseOff = (b * seqLen + t) * modelDim;
                for (int c = 0; c < modelDim; c++)
                {
                    double k = K[baseOff + c];
                    double v = V[baseOff + c];
                    double u = timeFirst[c];

                    // Pre-update state entering step t (post-update state of t-1).
                    double aaPrev = t > 0 ? aaTraj[(t - 1) * modelDim + c] : 0.0;
                    double bbPrev = t > 0 ? bbTraj[(t - 1) * modelDim + c] : 0.0;
                    double ppPrev = t > 0 ? ppTraj[(t - 1) * modelDim + c] : double.NegativeInfinity;
                    double ppCur = ppTraj[t * modelDim + c];

                    // Output recompute (q, pp treated as constants).
                    double ww = u + k;
                    double q = Math.Max(ppPrev, ww);
                    double e1 = Math.Exp(ppPrev - q);
                    double e2 = Math.Exp(ww - q);
                    double den = e1 * bbPrev + e2;
                    double num = e1 * aaPrev + e2 * v;
                    double wkv = num / den;
                    double sr = Sig(R[baseOff + c]);

                    // Output backward: out = sr * wkv.
                    double dO = dOut[baseOff + c];
                    dR[baseOff + c] += dO * wkv * sr * (1.0 - sr);
                    double gWkv = dO * sr;
                    double dNum = gWkv / den;
                    double dDen = -gWkv * wkv / den;
                    // num = e1*aaPrev + e2*v ; den = e1*bbPrev + e2 (e1 constant in pp).
                    double dAaFromOut = dNum * e1;
                    double dBbFromOut = dDen * e1;
                    dV[baseOff + c] += dNum * e2;
                    double dE2 = dNum * v + dDen;          // e2 = exp(u + k - q)
                    dK[baseOff + c] += dE2 * e2;            // de2/dk = e2
                    dTimeFirst[c] += dE2 * e2;              // de2/du = e2

                    // State-update recompute (ww2, q2=ppCur constants).
                    double ww2 = ppPrev + w[c];
                    double e1b = Math.Exp(ww2 - ppCur);
                    double e2b = Math.Exp(k - ppCur);

                    // State-update backward: aa_t = e1b*aaPrev + e2b*v ; bb_t = e1b*bbPrev + e2b.
                    double dAt = dAa[c];
                    double dBt = dBb[c];
                    double dE1b = dAt * aaPrev + dBt * bbPrev;   // e1b = exp(ppPrev + w - ppCur)
                    dw[c] += dE1b * e1b;                          // de1b/dw = e1b
                    dV[baseOff + c] += dAt * e2b;
                    double dE2b = dAt * v + dBt;                 // e2b = exp(k - ppCur)
                    dK[baseOff + c] += dE2b * e2b;                // de2b/dk = e2b

                    // Adjoint flowing to the pre-update state (for step t-1): from the update
                    // (e1b factor) and from this step's output (e1 factor).
                    dAa[c] = dAt * e1b + dAaFromOut;
                    dBb[c] = dBt * e1b + dBbFromOut;
                }
            }
        }

        // Chain w = -exp(timeDecay) -> dTimeDecay = dw * dw/dTimeDecay = dw * (-exp(timeDecay)) = dw * w.
        for (int c = 0; c < modelDim; c++)
            dTimeDecay[c] += dw[c] * w[c];
    }

    // ── Generic-T path (correct for any numeric T; used for non-double) ──────────────────
    private static void Rwkv4ForwardGeneric<T>(
        T[] R, T[] K, T[] V, T[] timeDecay, T[] timeFirst, T[] outp,
        int batch, int seqLen, int modelDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var w = new T[modelDim];
        for (int c = 0; c < modelDim; c++) w[c] = ops.Negate(ops.Exp(timeDecay[c]));
        T negInf = ops.FromDouble(-1e38);

        var aa = new T[modelDim];
        var bb = new T[modelDim];
        var pp = new T[modelDim];
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < modelDim; c++) { aa[c] = ops.Zero; bb[c] = ops.Zero; pp[c] = negInf; }
            for (int t = 0; t < seqLen; t++)
            {
                int baseOff = (b * seqLen + t) * modelDim;
                for (int c = 0; c < modelDim; c++)
                {
                    T k = K[baseOff + c];
                    T v = V[baseOff + c];
                    T ww = ops.Add(timeFirst[c], k);
                    T q = MaxT(ops, pp[c], ww);
                    T e1 = ops.Exp(ops.Subtract(pp[c], q));
                    T e2 = ops.Exp(ops.Subtract(ww, q));
                    T wkv = ops.Divide(
                        ops.Add(ops.Multiply(e1, aa[c]), ops.Multiply(e2, v)),
                        ops.Add(ops.Multiply(e1, bb[c]), e2));
                    outp[baseOff + c] = ops.Multiply(SigGeneric(ops, R[baseOff + c]), wkv);

                    T ww2 = ops.Add(pp[c], w[c]);
                    T q2 = MaxT(ops, ww2, k);
                    T e1b = ops.Exp(ops.Subtract(ww2, q2));
                    T e2b = ops.Exp(ops.Subtract(k, q2));
                    aa[c] = ops.Add(ops.Multiply(e1b, aa[c]), ops.Multiply(e2b, v));
                    bb[c] = ops.Add(ops.Multiply(e1b, bb[c]), e2b);
                    pp[c] = q2;
                }
            }
        }
    }

    private static void Rwkv4BackwardGeneric<T>(
        T[] dOut, T[] R, T[] K, T[] V, T[] timeDecay, T[] timeFirst,
        T[] dR, T[] dK, T[] dV, T[] dTimeDecay, T[] dTimeFirst,
        int batch, int seqLen, int modelDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T one = ops.One;
        var w = new T[modelDim];
        for (int c = 0; c < modelDim; c++) w[c] = ops.Negate(ops.Exp(timeDecay[c]));
        T negInf = ops.FromDouble(-1e38);

        var aaTraj = new T[seqLen * modelDim];
        var bbTraj = new T[seqLen * modelDim];
        var ppTraj = new T[seqLen * modelDim];
        var aa = new T[modelDim];
        var bb = new T[modelDim];
        var pp = new T[modelDim];
        var dAa = new T[modelDim];
        var dBb = new T[modelDim];
        var dw = new T[modelDim];
        for (int c = 0; c < modelDim; c++) dw[c] = ops.Zero;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < modelDim; c++) { aa[c] = ops.Zero; bb[c] = ops.Zero; pp[c] = negInf; }
            for (int t = 0; t < seqLen; t++)
            {
                int baseOff = (b * seqLen + t) * modelDim;
                int tOff = t * modelDim;
                for (int c = 0; c < modelDim; c++)
                {
                    T k = K[baseOff + c];
                    T v = V[baseOff + c];
                    T ww2 = ops.Add(pp[c], w[c]);
                    T q2 = MaxT(ops, ww2, k);
                    T e1b = ops.Exp(ops.Subtract(ww2, q2));
                    T e2b = ops.Exp(ops.Subtract(k, q2));
                    aa[c] = ops.Add(ops.Multiply(e1b, aa[c]), ops.Multiply(e2b, v));
                    bb[c] = ops.Add(ops.Multiply(e1b, bb[c]), e2b);
                    pp[c] = q2;
                    aaTraj[tOff + c] = aa[c];
                    bbTraj[tOff + c] = bb[c];
                    ppTraj[tOff + c] = pp[c];
                }
            }

            for (int c = 0; c < modelDim; c++) { dAa[c] = ops.Zero; dBb[c] = ops.Zero; }
            for (int t = seqLen - 1; t >= 0; t--)
            {
                int baseOff = (b * seqLen + t) * modelDim;
                for (int c = 0; c < modelDim; c++)
                {
                    T k = K[baseOff + c];
                    T v = V[baseOff + c];
                    T aaPrev = t > 0 ? aaTraj[(t - 1) * modelDim + c] : ops.Zero;
                    T bbPrev = t > 0 ? bbTraj[(t - 1) * modelDim + c] : ops.Zero;
                    T ppPrev = t > 0 ? ppTraj[(t - 1) * modelDim + c] : negInf;
                    T ppCur = ppTraj[t * modelDim + c];

                    T ww = ops.Add(timeFirst[c], k);
                    T q = MaxT(ops, ppPrev, ww);
                    T e1 = ops.Exp(ops.Subtract(ppPrev, q));
                    T e2 = ops.Exp(ops.Subtract(ww, q));
                    T den = ops.Add(ops.Multiply(e1, bbPrev), e2);
                    T num = ops.Add(ops.Multiply(e1, aaPrev), ops.Multiply(e2, v));
                    T wkv = ops.Divide(num, den);
                    T sr = SigGeneric(ops, R[baseOff + c]);

                    T dO = dOut[baseOff + c];
                    dR[baseOff + c] = ops.Add(dR[baseOff + c],
                        ops.Multiply(ops.Multiply(dO, wkv), ops.Multiply(sr, ops.Subtract(one, sr))));
                    T gWkv = ops.Multiply(dO, sr);
                    T dNum = ops.Divide(gWkv, den);
                    T dDen = ops.Divide(ops.Negate(ops.Multiply(gWkv, wkv)), den);
                    T dAaFromOut = ops.Multiply(dNum, e1);
                    T dBbFromOut = ops.Multiply(dDen, e1);
                    dV[baseOff + c] = ops.Add(dV[baseOff + c], ops.Multiply(dNum, e2));
                    T dE2 = ops.Add(ops.Multiply(dNum, v), dDen);
                    dK[baseOff + c] = ops.Add(dK[baseOff + c], ops.Multiply(dE2, e2));
                    dTimeFirst[c] = ops.Add(dTimeFirst[c], ops.Multiply(dE2, e2));

                    T ww2 = ops.Add(ppPrev, w[c]);
                    T e1b = ops.Exp(ops.Subtract(ww2, ppCur));
                    T e2b = ops.Exp(ops.Subtract(k, ppCur));
                    T dAt = dAa[c];
                    T dBt = dBb[c];
                    T dE1b = ops.Add(ops.Multiply(dAt, aaPrev), ops.Multiply(dBt, bbPrev));
                    dw[c] = ops.Add(dw[c], ops.Multiply(dE1b, e1b));
                    dV[baseOff + c] = ops.Add(dV[baseOff + c], ops.Multiply(dAt, e2b));
                    T dE2b = ops.Add(ops.Multiply(dAt, v), dBt);
                    dK[baseOff + c] = ops.Add(dK[baseOff + c], ops.Multiply(dE2b, e2b));

                    dAa[c] = ops.Add(ops.Multiply(dAt, e1b), dAaFromOut);
                    dBb[c] = ops.Add(ops.Multiply(dBt, e1b), dBbFromOut);
                }
            }
        }

        for (int c = 0; c < modelDim; c++)
            dTimeDecay[c] = ops.Add(dTimeDecay[c], ops.Multiply(dw[c], w[c]));
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static T MaxT<T>(INumericOperations<T> ops, T a, T b)
        => ops.GreaterThan(a, b) ? a : b;

    private static int[] ShapeOf<T>(Tensor<T> t)
    {
        var s = new int[t.Rank];
        for (int i = 0; i < t.Rank; i++) s[i] = t.Shape[i];
        return s;
    }

    private static void Rwkv4WkvBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var rProj = inputs[0];
        var kProj = inputs[1];
        var vProj = inputs[2];
        var timeDecay = inputs[3];
        var timeFirst = inputs[4];

        int batch = rProj.Shape[0];
        int seqLen = rProj.Shape[1];
        int modelDim = rProj.Shape[2];

        var dR = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dK = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dV = new Tensor<T>(new[] { batch, seqLen, modelDim });
        // Allocate the per-channel param grads with the SAME shape as the inputs
        // (timeDecay is [modelDim]; timeFirst may be [numHeads, headDim] — same
        // flat length, different rank) so AccumulateGrad shape-matches. The
        // kernels index purely by flat channel c, so the backing array layout
        // is identical regardless of rank.
        var dTimeDecay = new Tensor<T>(ShapeOf(timeDecay));
        var dTimeFirst = new Tensor<T>(ShapeOf(timeFirst));

        if (typeof(T) == typeof(double))
        {
            Rwkv4BackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)rProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)timeDecay.GetDataArray()!,
                (double[])(object)timeFirst.GetDataArray()!,
                (double[])(object)dR.GetDataArray()!, (double[])(object)dK.GetDataArray()!,
                (double[])(object)dV.GetDataArray()!, (double[])(object)dTimeDecay.GetDataArray()!,
                (double[])(object)dTimeFirst.GetDataArray()!,
                batch, seqLen, modelDim);
        }
        else
        {
            Rwkv4BackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                rProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!,
                timeDecay.GetDataArray()!, timeFirst.GetDataArray()!,
                dR.GetDataArray()!, dK.GetDataArray()!, dV.GetDataArray()!,
                dTimeDecay.GetDataArray()!, dTimeFirst.GetDataArray()!,
                batch, seqLen, modelDim);
        }

        DifferentiableOps.AccumulateGrad(grads, rProj, dR, engine);
        DifferentiableOps.AccumulateGrad(grads, kProj, dK, engine);
        DifferentiableOps.AccumulateGrad(grads, vProj, dV, engine);
        DifferentiableOps.AccumulateGrad(grads, timeDecay, dTimeDecay, engine);
        DifferentiableOps.AccumulateGrad(grads, timeFirst, dTimeFirst, engine);
    }
}
