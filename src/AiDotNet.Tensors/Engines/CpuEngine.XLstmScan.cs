using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    private const double XLstmIGateClamp = 4.85e8; // exp(20) overflow guard on the exponential input gate.

    /// <summary>
    /// Fused xLSTM (mLSTM) matrix-memory recurrence over a whole sequence in a SINGLE op
    /// (forward + custom autodiff backward), replacing the per-element detached scalar loop the
    /// decomposed <c>ExtendedLSTMLayer</c> ran — both slow AND detached from the autodiff tape
    /// (issue ooples/AiDotNet#1464). Per head, key scale κ = 1/sqrt(headDim), scalar-per-head gates
    /// i (exp input), f (sigmoid forget), o (sigmoid output), matrix cell C and normalizer n at 0:
    /// <code>
    ///   kS[j] = κ * K_t[j]
    ///   n_t[di]    = f_t * n_{t-1}[di] + i_t * kS[di]
    ///   C_t[di,ki] = f_t * C_{t-1}[di,ki] + i_t * V_t[di] * kS[ki]
    ///   num[di]    = sum_ki C_t[di,ki] * Q_t[ki]
    ///   nf         = max(|sum_j n_t[j] * Q_t[j]|, 1)
    ///   h_t[di]    = o_t * num[di] / nf
    /// </code>
    /// The gates are passed post-activation (i = exp(...), f/o = sigmoid(...)) as scalar-per-head
    /// values (and i is clamped to <see cref="XLstmIGateClamp"/>). Records one differentiable tape
    /// node (the BPTT adjoint), so it is safe under an active <c>GradientTape</c>.
    /// </summary>
    /// <param name="qProj">Query projection [batch, seqLen, modelDim].</param>
    /// <param name="kProj">Key projection [batch, seqLen, modelDim] (scaled by 1/sqrt(headDim) internally).</param>
    /// <param name="vProj">Value projection [batch, seqLen, modelDim].</param>
    /// <param name="iGate">Post-exp scalar-per-head input gate [batch, seqLen, numHeads].</param>
    /// <param name="fGate">Post-sigmoid scalar-per-head forget gate [batch, seqLen, numHeads].</param>
    /// <param name="oGate">Post-sigmoid scalar-per-head output gate [batch, seqLen, numHeads].</param>
    /// <param name="numHeads">Number of heads; modelDim must be divisible by it.</param>
    /// <returns>The pre-output-projection hidden state [batch, seqLen, modelDim].</returns>
    public virtual Tensor<T> XLstmScanForward<T>(
        Tensor<T> qProj, Tensor<T> kProj, Tensor<T> vProj,
        Tensor<T> iGate, Tensor<T> fGate, Tensor<T> oGate, int numHeads)
    {
        if (qProj is null) throw new ArgumentNullException(nameof(qProj));
        if (kProj is null) throw new ArgumentNullException(nameof(kProj));
        if (vProj is null) throw new ArgumentNullException(nameof(vProj));
        if (iGate is null) throw new ArgumentNullException(nameof(iGate));
        if (fGate is null) throw new ArgumentNullException(nameof(fGate));
        if (oGate is null) throw new ArgumentNullException(nameof(oGate));
        if (numHeads < 1) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qProj.Rank != 3)
            throw new ArgumentException($"XLstmScanForward expects rank-3 inputs [batch, seqLen, modelDim]; got rank {qProj.Rank}.", nameof(qProj));

        int batch = qProj.Shape[0];
        int seqLen = qProj.Shape[1];
        int modelDim = qProj.Shape[2];
        if (modelDim % numHeads != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        int headDim = modelDim / numHeads;
        EnsureSameShape(qProj, kProj, nameof(kProj));
        EnsureSameShape(qProj, vProj, nameof(vProj));
        // Gates are scalar-per-head: [batch, seqLen, numHeads], not the full modelDim.
        EnsureGateShape(iGate, batch, seqLen, numHeads, nameof(iGate));
        EnsureGateShape(fGate, batch, seqLen, numHeads, nameof(fGate));
        EnsureGateShape(oGate, batch, seqLen, numHeads, nameof(oGate));

        var output = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            XLstmForwardDouble(
                (double[])(object)qProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)iGate.GetDataArray()!,
                (double[])(object)fGate.GetDataArray()!, (double[])(object)oGate.GetDataArray()!,
                (double[])(object)output.GetDataArray()!, batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            XLstmForwardGeneric<T>(
                qProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!,
                iGate.GetDataArray()!, fGate.GetDataArray()!, oGate.GetDataArray()!,
                output.GetDataArray()!, batch, seqLen, modelDim, numHeads, headDim);
        }

        DifferentiableOps.RecordIfActive<T>(
            "XLstmScan", output,
            new[] { qProj, kProj, vProj, iGate, fGate, oGate },
            XLstmScanBackward<T>,
            savedState: new object[] { numHeads });

        return output;
    }

    // ── Double fast path ─────────────────────────────────────────────────────────────────
    private static void XLstmForwardDouble(
        double[] Q, double[] K, double[] V, double[] I, double[] F, double[] O, double[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        double kappa = 1.0 / Math.Sqrt(headDim);
        // Each (batch, head) pair is independent (private state/scratch, disjoint outputs); parallelize
        // lock-free over the combined (b*numHeads) axis. See GlaForwardDouble.
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var C = new double[hh];
            var n = new double[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                Array.Clear(C, 0, hh);
                Array.Clear(n, 0, headDim);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h; // scalar-per-head gate offset
                    double iv = I[gOff]; if (iv > XLstmIGateClamp) iv = XLstmIGateClamp;
                    double f = F[gOff], o = O[gOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        double kSdi = K[baseOff + di] * kappa;
                        n[di] = f * n[di] + iv * kSdi;
                        double vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            C[srow + ki] = f * C[srow + ki] + iv * vv * (K[baseOff + ki] * kappa);
                    }
                    double nq = 0.0;
                    for (int j = 0; j < headDim; j++) nq += n[j] * Q[baseOff + j];
                    double nf = Math.Abs(nq); if (nf < 1.0) nf = 1.0;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double num = 0.0;
                        for (int ki = 0; ki < headDim; ki++) num += C[srow + ki] * Q[baseOff + ki];
                        outp[baseOff + di] = o * num / nf;
                    }
                }
            }
        });
    }

    private static void XLstmBackwardDouble(
        double[] dOut, double[] Q, double[] K, double[] V, double[] I, double[] F, double[] O,
        double[] dQ, double[] dK, double[] dV, double[] dI, double[] dF, double[] dO,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        double kappa = 1.0 / Math.Sqrt(headDim);
        // Lock-free over the independent (b*numHeads) axis with per-chunk scratch; see GlaBackwardDouble.
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var Ctraj = new double[seqLen * hh];        // pre-update C entering step t
            var ntraj = new double[seqLen * headDim];   // pre-update n entering step t
            var C = new double[hh];
            var n = new double[headDim];
            var dC = new double[hh];
            var dN = new double[headDim];
            var dKS = new double[headDim];
            // Both dCp/dNp are fully overwritten each step before being read (per-step scratch).
            var dCp = new double[hh];
            var dNp = new double[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;

                // Forward recompute, saving the pre-update C/n entering each step.
                Array.Clear(C, 0, hh); Array.Clear(n, 0, headDim);
                for (int t = 0; t < seqLen; t++)
                {
                    Array.Copy(C, 0, Ctraj, t * hh, hh);
                    Array.Copy(n, 0, ntraj, t * headDim, headDim);
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h;
                    double iv = I[gOff]; if (iv > XLstmIGateClamp) iv = XLstmIGateClamp;
                    double f = F[gOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        n[di] = f * n[di] + iv * (K[baseOff + di] * kappa);
                        double vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            C[srow + ki] = f * C[srow + ki] + iv * vv * (K[baseOff + ki] * kappa);
                    }
                }

                // Reverse sweep. dC/dN carry the adjoint of the POST-update state from t+1.
                Array.Clear(dC, 0, hh); Array.Clear(dN, 0, headDim);
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h;
                    double iv = I[gOff]; bool iClamped = iv > XLstmIGateClamp; if (iClamped) iv = XLstmIGateClamp;
                    double f = F[gOff], o = O[gOff];
                    int cpOff = t * hh, npOff = t * headDim;

                    // Recompute post-update C_t, n_t, nq, nf for this step.
                    // (Reuse C/n buffers as scratch for the post-update state.)
                    double nq = 0.0;
                    for (int di = 0; di < headDim; di++)
                    {
                        double nCur = f * ntraj[npOff + di] + iv * (K[baseOff + di] * kappa);
                        n[di] = nCur;
                        nq += nCur * Q[baseOff + di];
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        double vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            C[srow + ki] = f * Ctraj[cpOff + srow + ki] + iv * vv * (K[baseOff + ki] * kappa);
                    }
                    double absnq = Math.Abs(nq);
                    double nf = absnq < 1.0 ? 1.0 : absnq;
                    double dNfSign = absnq > 1.0 ? (nq > 0 ? 1.0 : -1.0) : 0.0; // d(nf)/d(nq)

                    // Output backward: h[di] = o*num[di]/nf, num[di] = sum_ki C_t[di,ki]*Q[ki].
                    double dO_acc = 0.0, dnf = 0.0;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double dh = dOut[baseOff + di];
                        double num = 0.0;
                        for (int ki = 0; ki < headDim; ki++) num += C[srow + ki] * Q[baseOff + ki];
                        dO_acc += dh * num / nf;
                        double dNum = dh * o / nf;
                        dnf += dh * o * num * (-1.0 / (nf * nf));
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            dC[srow + ki] += dNum * Q[baseOff + ki];
                            dQ[baseOff + ki] += dNum * C[srow + ki];
                        }
                    }
                    dO[gOff] += dO_acc;

                    // nf = max(|nq|,1); nq = sum_j n_t[j]*Q[j].
                    double dnq = dnf * dNfSign;
                    for (int j = 0; j < headDim; j++)
                    {
                        dN[j] += dnq * Q[baseOff + j];
                        dQ[baseOff + j] += dnq * n[j];
                    }

                    // State-update backward. dC/dN now hold the full adjoint of C_t/n_t.
                    Array.Clear(dKS, 0, headDim);
                    double dF_acc = 0.0, dI_acc = 0.0;
                    // C_t[di,ki] = f*Cp[di,ki] + iv*V[di]*kS[ki].
                    // dCp/dNp hoisted above; every element is written below before
                    // the Array.Copy carry, so no per-step clear is needed.
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double vv = V[baseOff + di];
                        double dVacc = 0.0;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            double dCt = dC[srow + ki];
                            double kSki = K[baseOff + ki] * kappa;
                            dF_acc += dCt * Ctraj[cpOff + srow + ki];
                            dCp[srow + ki] = dCt * f;
                            dI_acc += dCt * vv * kSki;
                            dVacc += dCt * iv * kSki;
                            dKS[ki] += dCt * iv * vv;
                        }
                        dV[baseOff + di] += dVacc;
                    }
                    // n_t[di] = f*np[di] + iv*kS[di].
                    for (int di = 0; di < headDim; di++)
                    {
                        double dNt = dN[di];
                        dF_acc += dNt * ntraj[npOff + di];
                        dNp[di] = dNt * f;
                        dI_acc += dNt * (K[baseOff + di] * kappa);
                        dKS[di] += dNt * iv;
                    }
                    // kS[j] = kappa*K[j].
                    for (int j = 0; j < headDim; j++) dK[baseOff + j] += dKS[j] * kappa;

                    // i gate clamp: gradient passes through only when not clamped.
                    dI[gOff] += iClamped ? 0.0 : dI_acc;
                    dF[gOff] += dF_acc;

                    // Carry adjoints to the previous step.
                    Array.Copy(dCp, 0, dC, 0, hh);
                    Array.Copy(dNp, 0, dN, 0, headDim);
                }
            }
        });
    }

    // ── Generic-T path ───────────────────────────────────────────────────────────────────
    private static void XLstmForwardGeneric<T>(
        T[] Q, T[] K, T[] V, T[] I, T[] F, T[] O, T[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hh = headDim * headDim;
        T kappa = ops.FromDouble(1.0 / Math.Sqrt(headDim));
        T clamp = ops.FromDouble(XLstmIGateClamp);
        T one = ops.One;
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var C = new T[hh];
            var n = new T[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                for (int i = 0; i < hh; i++) C[i] = ops.Zero;
                for (int di = 0; di < headDim; di++) n[di] = ops.Zero;
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h; // scalar-per-head gate offset
                    T iv = I[gOff]; if (ops.GreaterThan(iv, clamp)) iv = clamp;
                    T f = F[gOff], o = O[gOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        T kSdi = ops.Multiply(K[baseOff + di], kappa);
                        n[di] = ops.Add(ops.Multiply(f, n[di]), ops.Multiply(iv, kSdi));
                        T vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            C[srow + ki] = ops.Add(ops.Multiply(f, C[srow + ki]),
                                ops.Multiply(iv, ops.Multiply(vv, ops.Multiply(K[baseOff + ki], kappa))));
                    }
                    T nq = ops.Zero;
                    for (int j = 0; j < headDim; j++) nq = ops.Add(nq, ops.Multiply(n[j], Q[baseOff + j]));
                    T absnq = ops.Abs(nq);
                    T nf = ops.GreaterThan(absnq, one) ? absnq : one;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T num = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++) num = ops.Add(num, ops.Multiply(C[srow + ki], Q[baseOff + ki]));
                        outp[baseOff + di] = ops.Divide(ops.Multiply(o, num), nf);
                    }
                }
            }
        });
    }

    private static void XLstmBackwardGeneric<T>(
        T[] dOut, T[] Q, T[] K, T[] V, T[] I, T[] F, T[] O,
        T[] dQ, T[] dK, T[] dV, T[] dI, T[] dF, T[] dO,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hh = headDim * headDim;
        T kappa = ops.FromDouble(1.0 / Math.Sqrt(headDim));
        T clamp = ops.FromDouble(XLstmIGateClamp);
        T one = ops.One, zero = ops.Zero;
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var Ctraj = new T[seqLen * hh];
            var ntraj = new T[seqLen * headDim];
            var C = new T[hh];
            var n = new T[headDim];
            var dC = new T[hh];
            var dN = new T[headDim];
            var dKS = new T[headDim];
            var dCp = new T[hh];
            var dNp = new T[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;
                for (int i = 0; i < hh; i++) C[i] = zero;
                for (int di = 0; di < headDim; di++) n[di] = zero;
                for (int t = 0; t < seqLen; t++)
                {
                    Array.Copy(C, 0, Ctraj, t * hh, hh);
                    Array.Copy(n, 0, ntraj, t * headDim, headDim);
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h;
                    T iv = I[gOff]; if (ops.GreaterThan(iv, clamp)) iv = clamp;
                    T f = F[gOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        n[di] = ops.Add(ops.Multiply(f, n[di]), ops.Multiply(iv, ops.Multiply(K[baseOff + di], kappa)));
                        T vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            C[srow + ki] = ops.Add(ops.Multiply(f, C[srow + ki]),
                                ops.Multiply(iv, ops.Multiply(vv, ops.Multiply(K[baseOff + ki], kappa))));
                    }
                }

                for (int i = 0; i < hh; i++) dC[i] = zero;
                for (int di = 0; di < headDim; di++) dN[di] = zero;
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h;
                    T iv = I[gOff]; bool iClamped = ops.GreaterThan(iv, clamp); if (iClamped) iv = clamp;
                    T f = F[gOff], o = O[gOff];
                    int cpOff = t * hh, npOff = t * headDim;

                    T nq = zero;
                    for (int di = 0; di < headDim; di++)
                    {
                        T nCur = ops.Add(ops.Multiply(f, ntraj[npOff + di]), ops.Multiply(iv, ops.Multiply(K[baseOff + di], kappa)));
                        n[di] = nCur;
                        nq = ops.Add(nq, ops.Multiply(nCur, Q[baseOff + di]));
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        T vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            C[srow + ki] = ops.Add(ops.Multiply(f, Ctraj[cpOff + srow + ki]),
                                ops.Multiply(iv, ops.Multiply(vv, ops.Multiply(K[baseOff + ki], kappa))));
                    }
                    T absnq = ops.Abs(nq);
                    bool over = ops.GreaterThan(absnq, one);
                    T nf = over ? absnq : one;
                    T dNfSign = over ? (ops.GreaterThan(nq, zero) ? one : ops.FromDouble(-1.0)) : zero;
                    T invNf = ops.Divide(one, nf);
                    T negInvNf2 = ops.Negate(ops.Divide(one, ops.Multiply(nf, nf)));

                    T dO_acc = zero, dnf = zero;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T dh = dOut[baseOff + di];
                        T num = zero;
                        for (int ki = 0; ki < headDim; ki++) num = ops.Add(num, ops.Multiply(C[srow + ki], Q[baseOff + ki]));
                        dO_acc = ops.Add(dO_acc, ops.Multiply(dh, ops.Multiply(num, invNf)));
                        T dNum = ops.Multiply(dh, ops.Multiply(o, invNf));
                        dnf = ops.Add(dnf, ops.Multiply(ops.Multiply(dh, ops.Multiply(o, num)), negInvNf2));
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            dC[srow + ki] = ops.Add(dC[srow + ki], ops.Multiply(dNum, Q[baseOff + ki]));
                            dQ[baseOff + ki] = ops.Add(dQ[baseOff + ki], ops.Multiply(dNum, C[srow + ki]));
                        }
                    }
                    dO[gOff] = ops.Add(dO[gOff], dO_acc);

                    T dnq = ops.Multiply(dnf, dNfSign);
                    for (int j = 0; j < headDim; j++)
                    {
                        dN[j] = ops.Add(dN[j], ops.Multiply(dnq, Q[baseOff + j]));
                        dQ[baseOff + j] = ops.Add(dQ[baseOff + j], ops.Multiply(dnq, n[j]));
                    }

                    for (int j = 0; j < headDim; j++) dKS[j] = zero;
                    for (int i = 0; i < hh; i++) dCp[i] = zero;
                    for (int di = 0; di < headDim; di++) dNp[di] = zero;
                    T dF_acc = zero, dI_acc = zero;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T vv = V[baseOff + di];
                        T dVacc = zero;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            T dCt = dC[srow + ki];
                            T kSki = ops.Multiply(K[baseOff + ki], kappa);
                            dF_acc = ops.Add(dF_acc, ops.Multiply(dCt, Ctraj[cpOff + srow + ki]));
                            dCp[srow + ki] = ops.Multiply(dCt, f);
                            dI_acc = ops.Add(dI_acc, ops.Multiply(dCt, ops.Multiply(vv, kSki)));
                            dVacc = ops.Add(dVacc, ops.Multiply(dCt, ops.Multiply(iv, kSki)));
                            dKS[ki] = ops.Add(dKS[ki], ops.Multiply(dCt, ops.Multiply(iv, vv)));
                        }
                        dV[baseOff + di] = ops.Add(dV[baseOff + di], dVacc);
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        T dNt = dN[di];
                        dF_acc = ops.Add(dF_acc, ops.Multiply(dNt, ntraj[npOff + di]));
                        dNp[di] = ops.Multiply(dNt, f);
                        dI_acc = ops.Add(dI_acc, ops.Multiply(dNt, ops.Multiply(K[baseOff + di], kappa)));
                        dKS[di] = ops.Add(dKS[di], ops.Multiply(dNt, iv));
                    }
                    for (int j = 0; j < headDim; j++) dK[baseOff + j] = ops.Add(dK[baseOff + j], ops.Multiply(dKS[j], kappa));

                    if (!iClamped) dI[gOff] = ops.Add(dI[gOff], dI_acc);
                    dF[gOff] = ops.Add(dF[gOff], dF_acc);

                    Array.Copy(dCp, 0, dC, 0, hh);
                    Array.Copy(dNp, 0, dN, 0, headDim);
                }
            }
        });
    }

    private static void XLstmScanBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int numHeads = (int)savedState[0];
        var qProj = inputs[0];
        var kProj = inputs[1];
        var vProj = inputs[2];
        var iGate = inputs[3];
        var fGate = inputs[4];
        var oGate = inputs[5];

        int batch = qProj.Shape[0];
        int seqLen = qProj.Shape[1];
        int modelDim = qProj.Shape[2];
        int headDim = modelDim / numHeads;

        var dQ = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dK = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dV = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dI = new Tensor<T>(new[] { batch, seqLen, numHeads }); // scalar-per-head gates
        var dF = new Tensor<T>(new[] { batch, seqLen, numHeads });
        var dO = new Tensor<T>(new[] { batch, seqLen, numHeads });

        if (typeof(T) == typeof(double))
        {
            XLstmBackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)qProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)iGate.GetDataArray()!,
                (double[])(object)fGate.GetDataArray()!, (double[])(object)oGate.GetDataArray()!,
                (double[])(object)dQ.GetDataArray()!, (double[])(object)dK.GetDataArray()!,
                (double[])(object)dV.GetDataArray()!, (double[])(object)dI.GetDataArray()!,
                (double[])(object)dF.GetDataArray()!, (double[])(object)dO.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            XLstmBackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                qProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!,
                iGate.GetDataArray()!, fGate.GetDataArray()!, oGate.GetDataArray()!,
                dQ.GetDataArray()!, dK.GetDataArray()!, dV.GetDataArray()!,
                dI.GetDataArray()!, dF.GetDataArray()!, dO.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }

        DifferentiableOps.AccumulateGrad(grads, qProj, dQ, engine);
        DifferentiableOps.AccumulateGrad(grads, kProj, dK, engine);
        DifferentiableOps.AccumulateGrad(grads, vProj, dV, engine);
        DifferentiableOps.AccumulateGrad(grads, iGate, dI, engine);
        DifferentiableOps.AccumulateGrad(grads, fGate, dF, engine);
        DifferentiableOps.AccumulateGrad(grads, oGate, dO, engine);
    }
}
