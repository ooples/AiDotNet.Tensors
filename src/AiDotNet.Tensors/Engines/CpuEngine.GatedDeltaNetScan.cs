using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Fused Gated DeltaNet (delta-rule) recurrence over a whole sequence in a SINGLE op
    /// (forward + custom autodiff backward), replacing the per-element detached scalar loop the
    /// decomposed <c>GatedDeltaNetLayer</c> ran — both slow AND detached from the autodiff tape
    /// (issue ooples/AiDotNet#1464). Per head, with key scaling κ = 1/sqrt(headDim), scalar-per-head
    /// gates α_t (forget) and β_t (write), matrix state S[di,ki] = 0:
    /// <code>
    ///   kS[ki] = κ * K_t[ki]
    ///   sK[di] = sum_ki S_{t-1}[di,ki] * kS[ki]          (current readout for the key)
    ///   delta[di] = V_t[di] - sK[di]                      (delta-rule correction)
    ///   S_t[di,ki] = α_t * S_{t-1}[di,ki] + β_t * delta[di] * kS[ki]
    ///   O_t[di] = sum_ki S_t[di,ki] * Q_t[ki]
    /// </code>
    /// Records one tape node whose backward is the exact BPTT adjoint, so it is differentiable under an
    /// active <c>GradientTape</c>.
    /// </summary>
    /// <param name="qProj">Query projection [batch, seqLen, modelDim].</param>
    /// <param name="kProj">Key projection [batch, seqLen, modelDim] (scaled by 1/sqrt(headDim) internally).</param>
    /// <param name="vProj">Value projection [batch, seqLen, modelDim].</param>
    /// <param name="alpha">Forget gate [batch, seqLen, numHeads] (scalar per head).</param>
    /// <param name="beta">Write gate [batch, seqLen, numHeads] (scalar per head).</param>
    /// <param name="numHeads">Number of heads; modelDim must be divisible by it.</param>
    /// <returns>The delta-rule output [batch, seqLen, modelDim].</returns>
    public virtual Tensor<T> GatedDeltaNetScanForward<T>(
        Tensor<T> qProj, Tensor<T> kProj, Tensor<T> vProj, Tensor<T> alpha, Tensor<T> beta, int numHeads)
    {
        if (qProj is null) throw new ArgumentNullException(nameof(qProj));
        if (kProj is null) throw new ArgumentNullException(nameof(kProj));
        if (vProj is null) throw new ArgumentNullException(nameof(vProj));
        if (alpha is null) throw new ArgumentNullException(nameof(alpha));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (numHeads < 1) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qProj.Rank != 3)
            throw new ArgumentException($"GatedDeltaNetScanForward expects rank-3 q/k/v [batch, seqLen, modelDim]; got rank {qProj.Rank}.", nameof(qProj));

        int batch = qProj.Shape[0];
        int seqLen = qProj.Shape[1];
        int modelDim = qProj.Shape[2];
        if (modelDim % numHeads != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        int headDim = modelDim / numHeads;
        EnsureSameShape(qProj, kProj, nameof(kProj));
        EnsureSameShape(qProj, vProj, nameof(vProj));
        if (alpha.Rank != 3 || alpha.Shape[0] != batch || alpha.Shape[1] != seqLen || alpha.Shape[2] != numHeads)
            throw new ArgumentException($"alpha must be [batch={batch}, seqLen={seqLen}, numHeads={numHeads}].", nameof(alpha));
        if (beta.Rank != 3 || beta.Shape[0] != batch || beta.Shape[1] != seqLen || beta.Shape[2] != numHeads)
            throw new ArgumentException($"beta must be [batch={batch}, seqLen={seqLen}, numHeads={numHeads}].", nameof(beta));

        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var capturedQ = qProj;
            var capturedK = kProj;
            var capturedV = vProj;
            var capturedAlpha = alpha;
            var capturedBeta = beta;
            int capturedNumHeads = numHeads;
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "GatedDeltaNetScan",
                new[] { qProj, kProj, vProj, alpha, beta },
                new[] { batch, seqLen, modelDim },
                (eng, output) =>
                {
                    var result = eng.GatedDeltaNetScanForward(
                        capturedQ, capturedK, capturedV, capturedAlpha, capturedBeta, capturedNumHeads);
                    DirectGpuTensorEngine.CopyResultInto(eng, result, output);
                },
                GatedDeltaNetScanBackward<T>,
                new object[] { numHeads });
        }

        var output = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            GatedDeltaForwardDouble(
                (double[])(object)qProj.GetDataArray(), (double[])(object)kProj.GetDataArray(),
                (double[])(object)vProj.GetDataArray(), (double[])(object)alpha.GetDataArray(),
                (double[])(object)beta.GetDataArray(), (double[])(object)output.GetDataArray(),
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else if (typeof(T) == typeof(float))
        {
            GatedDeltaForwardFloat(
                (float[])(object)qProj.GetDataArray(), (float[])(object)kProj.GetDataArray(),
                (float[])(object)vProj.GetDataArray(), (float[])(object)alpha.GetDataArray(),
                (float[])(object)beta.GetDataArray(), (float[])(object)output.GetDataArray(),
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            GatedDeltaForwardGeneric<T>(
                qProj.GetDataArray(), kProj.GetDataArray(), vProj.GetDataArray(),
                alpha.GetDataArray(), beta.GetDataArray(), output.GetDataArray(),
                batch, seqLen, modelDim, numHeads, headDim);
        }

        DifferentiableOps.RecordIfActive<T>(
            "GatedDeltaNetScan", output,
            new[] { qProj, kProj, vProj, alpha, beta },
            GatedDeltaNetScanBackward<T>,
            savedState: new object[] { numHeads });

        return output;
    }

    // ── Double fast path ─────────────────────────────────────────────────────────────────
    [MethodImpl(Hot)]
    private static void GatedDeltaForwardDouble(
        double[] Q, double[] K, double[] V, double[] A, double[] B, double[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        double kappa = 1.0 / Math.Sqrt(headDim);
        // Each (batch, head) pair is independent (private state/scratch, disjoint outputs); parallelize
        // lock-free over the combined (b*numHeads) axis. See GlaForwardDouble.
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, [MethodImpl(Hot)] (bhStart, bhCount) =>
        {
            var S = new double[hh];
            var kS = new double[headDim];
            var sK = new double[headDim];
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
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    double a = A[gIdx], bet = B[gIdx];
                    for (int ki = 0; ki < headDim; ki++) kS[ki] = K[baseOff + ki] * kappa;

                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double s = 0.0;
                        for (int ki = 0; ki < headDim; ki++) s += S[srow + ki] * kS[ki];
                        sK[di] = s;
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double delta = V[baseOff + di] - sK[di];
                        double bd = bet * delta;
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = a * S[srow + ki] + bd * kS[ki];
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double o = 0.0;
                        for (int ki = 0; ki < headDim; ki++) o += S[srow + ki] * Q[baseOff + ki];
                        outp[baseOff + di] = o;
                    }
                }
            }
        });
    }

    [MethodImpl(Hot)]
    private static void GatedDeltaBackwardDouble(
        double[] dOut, double[] Q, double[] K, double[] V, double[] A, double[] B,
        double[] dQ, double[] dK, double[] dV, double[] dA, double[] dB,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        double kappa = 1.0 / Math.Sqrt(headDim);
        // Lock-free over the independent (b*numHeads) axis with per-chunk scratch; see GlaBackwardDouble.
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, [MethodImpl(Hot)] (bhStart, bhCount) =>
        {
            var Straj = new double[(seqLen + 1) * hh]; // pre-update state at index t = state entering step t
            var S = new double[hh];
            var dS = new double[hh];
            var dSp = new double[hh];
            var kS = new double[headDim];
            var sK = new double[headDim];
            var dKS = new double[headDim];
            var dDelta = new double[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;

                // Forward recompute, saving the PRE-update state entering each step (index t).
                Array.Clear(S, 0, hh);
                for (int t = 0; t < seqLen; t++)
                {
                    Array.Copy(S, 0, Straj, t * hh, hh);
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    double a = A[gIdx], bet = B[gIdx];
                    for (int ki = 0; ki < headDim; ki++) kS[ki] = K[baseOff + ki] * kappa;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double s = 0.0;
                        for (int ki = 0; ki < headDim; ki++) s += S[srow + ki] * kS[ki];
                        double delta = V[baseOff + di] - s;
                        double bd = bet * delta;
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = a * S[srow + ki] + bd * kS[ki];
                    }
                }

                // Reverse sweep. dS = adjoint of the POST-update state S_t.
                Array.Clear(dS, 0, hh);
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    double a = A[gIdx], bet = B[gIdx];
                    var Sp = Straj; int spOff = t * hh; // pre-update state S_{t-1}

                    for (int ki = 0; ki < headDim; ki++) kS[ki] = K[baseOff + ki] * kappa;
                    // Recompute sK and S_t (post-update) for this step.
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double s = 0.0;
                        for (int ki = 0; ki < headDim; ki++) s += Sp[spOff + srow + ki] * kS[ki];
                        sK[di] = s;
                    }

                    // Output backward: O[di] = sum_ki S_t[di,ki]*Q[ki]; S_t[di,ki] = a*Sp + bet*delta[di]*kS[ki].
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double dOutVal = dOut[baseOff + di];
                        double delta = V[baseOff + di] - sK[di];
                        double bd = bet * delta;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            double sT = a * Sp[spOff + srow + ki] + bd * kS[ki];
                            dQ[baseOff + ki] += dOutVal * sT;
                            dS[srow + ki] += dOutVal * Q[baseOff + ki];
                        }
                    }

                    // State-update backward.
                    Array.Clear(dSp, 0, hh);
                    Array.Clear(dKS, 0, headDim);
                    Array.Clear(dDelta, 0, headDim);
                    double dA_acc = 0.0, dB_acc = 0.0;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double delta = V[baseOff + di] - sK[di];
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            double dSt = dS[srow + ki];
                            dA_acc += dSt * Sp[spOff + srow + ki];
                            dSp[srow + ki] += dSt * a;
                            dB_acc += dSt * delta * kS[ki];
                            dDelta[di] += dSt * bet * kS[ki];
                            dKS[ki] += dSt * bet * delta;
                        }
                    }
                    // delta[di] = V[di] - sK[di]; sK[di] = sum_kj Sp[di,kj]*kS[kj].
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        dV[baseOff + di] += dDelta[di];
                        double dSk = -dDelta[di];
                        for (int kj = 0; kj < headDim; kj++)
                        {
                            dSp[srow + kj] += dSk * kS[kj];
                            dKS[kj] += dSk * Sp[spOff + srow + kj];
                        }
                    }
                    // kS[ki] = kappa * K[ki].
                    for (int ki = 0; ki < headDim; ki++) dK[baseOff + ki] += dKS[ki] * kappa;

                    dA[gIdx] += dA_acc;
                    dB[gIdx] += dB_acc;

                    // dSp is the adjoint of S_{t-1}; it becomes dS for step t-1.
                    Array.Copy(dSp, 0, dS, 0, hh);
                }
            }
        });
    }

    // ── Float fast path ──────────────────────────────────────────────────────────────────
    [MethodImpl(Hot)]
    private static void GatedDeltaForwardFloat(
        float[] Q, float[] K, float[] V, float[] A, float[] B, float[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        float kappa = (float)(1.0 / Math.Sqrt(headDim));
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, [MethodImpl(Hot)] (bhStart, bhCount) =>
        {
            var S = new float[hh];
            var kS = new float[headDim];
            var sK = new float[headDim];
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
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    float a = A[gIdx], bet = B[gIdx];
                    for (int ki = 0; ki < headDim; ki++) kS[ki] = K[baseOff + ki] * kappa;

                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float s = 0f;
                        for (int ki = 0; ki < headDim; ki++) s += S[srow + ki] * kS[ki];
                        sK[di] = s;
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float delta = V[baseOff + di] - sK[di];
                        float bd = bet * delta;
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = a * S[srow + ki] + bd * kS[ki];
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float o = 0f;
                        for (int ki = 0; ki < headDim; ki++) o += S[srow + ki] * Q[baseOff + ki];
                        outp[baseOff + di] = o;
                    }
                }
            }
        });
    }

    [MethodImpl(Hot)]
    private static void GatedDeltaBackwardFloat(
        float[] dOut, float[] Q, float[] K, float[] V, float[] A, float[] B,
        float[] dQ, float[] dK, float[] dV, float[] dA, float[] dB,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        float kappa = (float)(1.0 / Math.Sqrt(headDim));
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, [MethodImpl(Hot)] (bhStart, bhCount) =>
        {
            var Straj = new float[(seqLen + 1) * hh];
            var S = new float[hh];
            var dS = new float[hh];
            var dSp = new float[hh];
            var kS = new float[headDim];
            var sK = new float[headDim];
            var dKS = new float[headDim];
            var dDelta = new float[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;

                Array.Clear(S, 0, hh);
                for (int t = 0; t < seqLen; t++)
                {
                    Array.Copy(S, 0, Straj, t * hh, hh);
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    float a = A[gIdx], bet = B[gIdx];
                    for (int ki = 0; ki < headDim; ki++) kS[ki] = K[baseOff + ki] * kappa;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float s = 0f;
                        for (int ki = 0; ki < headDim; ki++) s += S[srow + ki] * kS[ki];
                        float delta = V[baseOff + di] - s;
                        float bd = bet * delta;
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = a * S[srow + ki] + bd * kS[ki];
                    }
                }

                Array.Clear(dS, 0, hh);
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    float a = A[gIdx], bet = B[gIdx];
                    var Sp = Straj; int spOff = t * hh;

                    for (int ki = 0; ki < headDim; ki++) kS[ki] = K[baseOff + ki] * kappa;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float s = 0f;
                        for (int ki = 0; ki < headDim; ki++) s += Sp[spOff + srow + ki] * kS[ki];
                        sK[di] = s;
                    }

                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float dOutVal = dOut[baseOff + di];
                        float delta = V[baseOff + di] - sK[di];
                        float bd = bet * delta;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            float sT = a * Sp[spOff + srow + ki] + bd * kS[ki];
                            dQ[baseOff + ki] += dOutVal * sT;
                            dS[srow + ki] += dOutVal * Q[baseOff + ki];
                        }
                    }

                    Array.Clear(dSp, 0, hh);
                    Array.Clear(dKS, 0, headDim);
                    Array.Clear(dDelta, 0, headDim);
                    float dA_acc = 0f, dB_acc = 0f;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float delta = V[baseOff + di] - sK[di];
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            float dSt = dS[srow + ki];
                            dA_acc += dSt * Sp[spOff + srow + ki];
                            dSp[srow + ki] += dSt * a;
                            dB_acc += dSt * delta * kS[ki];
                            dDelta[di] += dSt * bet * kS[ki];
                            dKS[ki] += dSt * bet * delta;
                        }
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        dV[baseOff + di] += dDelta[di];
                        float dSk = -dDelta[di];
                        for (int kj = 0; kj < headDim; kj++)
                        {
                            dSp[srow + kj] += dSk * kS[kj];
                            dKS[kj] += dSk * Sp[spOff + srow + kj];
                        }
                    }
                    for (int ki = 0; ki < headDim; ki++) dK[baseOff + ki] += dKS[ki] * kappa;

                    dA[gIdx] += dA_acc;
                    dB[gIdx] += dB_acc;

                    Array.Copy(dSp, 0, dS, 0, hh);
                }
            }
        });
    }

    // ── Generic-T path ───────────────────────────────────────────────────────────────────
    private static void GatedDeltaForwardGeneric<T>(
        T[] Q, T[] K, T[] V, T[] A, T[] B, T[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hh = headDim * headDim;
        T kappa = ops.FromDouble(1.0 / Math.Sqrt(headDim));
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var S = new T[hh];
            var kS = new T[headDim];
            var sK = new T[headDim];
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
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    T a = A[gIdx], bet = B[gIdx];
                    for (int ki = 0; ki < headDim; ki++) kS[ki] = ops.Multiply(K[baseOff + ki], kappa);
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T s = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++) s = ops.Add(s, ops.Multiply(S[srow + ki], kS[ki]));
                        sK[di] = s;
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T delta = ops.Subtract(V[baseOff + di], sK[di]);
                        T bd = ops.Multiply(bet, delta);
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = ops.Add(ops.Multiply(a, S[srow + ki]), ops.Multiply(bd, kS[ki]));
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T o = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++) o = ops.Add(o, ops.Multiply(S[srow + ki], Q[baseOff + ki]));
                        outp[baseOff + di] = o;
                    }
                }
            }
        });
    }

    private static void GatedDeltaBackwardGeneric<T>(
        T[] dOut, T[] Q, T[] K, T[] V, T[] A, T[] B,
        T[] dQ, T[] dK, T[] dV, T[] dA, T[] dB,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hh = headDim * headDim;
        T kappa = ops.FromDouble(1.0 / Math.Sqrt(headDim));
        CpuParallelSettings.ParallelForChunks(batch * numHeads, GlaBhGrain, (bhStart, bhCount) =>
        {
            var Straj = new T[(seqLen + 1) * hh];
            var S = new T[hh];
            var dS = new T[hh];
            var dSp = new T[hh];
            var kS = new T[headDim];
            var sK = new T[headDim];
            var dKS = new T[headDim];
            var dDelta = new T[headDim];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;
                for (int i = 0; i < hh; i++) S[i] = ops.Zero;
                for (int t = 0; t < seqLen; t++)
                {
                    Array.Copy(S, 0, Straj, t * hh, hh);
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    T a = A[gIdx], bet = B[gIdx];
                    for (int ki = 0; ki < headDim; ki++) kS[ki] = ops.Multiply(K[baseOff + ki], kappa);
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T s = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++) s = ops.Add(s, ops.Multiply(S[srow + ki], kS[ki]));
                        T delta = ops.Subtract(V[baseOff + di], s);
                        T bd = ops.Multiply(bet, delta);
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = ops.Add(ops.Multiply(a, S[srow + ki]), ops.Multiply(bd, kS[ki]));
                    }
                }

                for (int i = 0; i < hh; i++) dS[i] = ops.Zero;
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    T a = A[gIdx], bet = B[gIdx];
                    int spOff = t * hh;

                    for (int ki = 0; ki < headDim; ki++) kS[ki] = ops.Multiply(K[baseOff + ki], kappa);
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T s = ops.Zero;
                        for (int ki = 0; ki < headDim; ki++) s = ops.Add(s, ops.Multiply(Straj[spOff + srow + ki], kS[ki]));
                        sK[di] = s;
                    }

                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T dOutVal = dOut[baseOff + di];
                        T delta = ops.Subtract(V[baseOff + di], sK[di]);
                        T bd = ops.Multiply(bet, delta);
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            T sT = ops.Add(ops.Multiply(a, Straj[spOff + srow + ki]), ops.Multiply(bd, kS[ki]));
                            dQ[baseOff + ki] = ops.Add(dQ[baseOff + ki], ops.Multiply(dOutVal, sT));
                            dS[srow + ki] = ops.Add(dS[srow + ki], ops.Multiply(dOutVal, Q[baseOff + ki]));
                        }
                    }

                    for (int i = 0; i < hh; i++) dSp[i] = ops.Zero;
                    for (int ki = 0; ki < headDim; ki++) dKS[ki] = ops.Zero;
                    for (int di = 0; di < headDim; di++) dDelta[di] = ops.Zero;
                    T dA_acc = ops.Zero, dB_acc = ops.Zero;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T delta = ops.Subtract(V[baseOff + di], sK[di]);
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            T dSt = dS[srow + ki];
                            dA_acc = ops.Add(dA_acc, ops.Multiply(dSt, Straj[spOff + srow + ki]));
                            dSp[srow + ki] = ops.Add(dSp[srow + ki], ops.Multiply(dSt, a));
                            dB_acc = ops.Add(dB_acc, ops.Multiply(dSt, ops.Multiply(delta, kS[ki])));
                            dDelta[di] = ops.Add(dDelta[di], ops.Multiply(dSt, ops.Multiply(bet, kS[ki])));
                            dKS[ki] = ops.Add(dKS[ki], ops.Multiply(dSt, ops.Multiply(bet, delta)));
                        }
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        dV[baseOff + di] = ops.Add(dV[baseOff + di], dDelta[di]);
                        T dSk = ops.Negate(dDelta[di]);
                        for (int kj = 0; kj < headDim; kj++)
                        {
                            dSp[srow + kj] = ops.Add(dSp[srow + kj], ops.Multiply(dSk, kS[kj]));
                            dKS[kj] = ops.Add(dKS[kj], ops.Multiply(dSk, Straj[spOff + srow + kj]));
                        }
                    }
                    for (int ki = 0; ki < headDim; ki++) dK[baseOff + ki] = ops.Add(dK[baseOff + ki], ops.Multiply(dKS[ki], kappa));

                    dA[gIdx] = ops.Add(dA[gIdx], dA_acc);
                    dB[gIdx] = ops.Add(dB[gIdx], dB_acc);

                    Array.Copy(dSp, 0, dS, 0, hh);
                }
            }
        });
    }

    [MethodImpl(Hot)]
    /// <summary>
    /// The scan's BPTT gradients as recorded engine ops, batched over (batch, head): the forward recomputes each state
    /// <c>S_t = a_t S_{t-1} + b_t (v_t - S_{t-1} k_t) k_tᵀ</c> (keys scaled by κ), then the reverse sweep carries
    /// <c>dS</c> back through <c>o_t = S_t q_t</c>. Every op records on the tape the create-graph backward writes to,
    /// so the returned gradients differentiate again. One small op chain per time step: for create-graph only.
    /// </summary>
    /// <returns>dQ, dK, dV [batch, seqLen, modelDim] and dAlpha, dBeta [batch, seqLen, numHeads].</returns>
    private static Tensor<T>[] GatedDeltaNetScanGradRecorded<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, int numHeads, IEngine engine)
    {
        int batch = inputs[0].Shape[0], seqLen = inputs[0].Shape[1], modelDim = inputs[0].Shape[2];
        int headDim = modelDim / numHeads, bh = batch * numHeads;
        var kappa = MathHelper.GetNumericOperations<T>().FromDouble(1.0 / Math.Sqrt(headDim));

        // [batch, seqLen, modelDim] -> [batch*heads, seqLen, headDim]; [batch, seqLen, heads] -> [batch*heads, seqLen].
        Tensor<T> Heads(Tensor<T> x) => engine.Reshape(
            engine.TensorPermute(engine.Reshape(x, new[] { batch, seqLen, numHeads, headDim }), new[] { 0, 2, 1, 3 }),
            new[] { bh, seqLen, headDim });
        Tensor<T> Gates(Tensor<T> x) => engine.Reshape(engine.TensorPermute(x, new[] { 0, 2, 1 }), new[] { bh, seqLen });
        // Step t as a column [bh, headDim, 1], or a gate as [bh, 1, 1].
        Tensor<T> Column(Tensor<T> x, int t) => engine.Reshape(engine.TensorNarrow(x, 1, t, 1), new[] { bh, headDim, 1 });
        Tensor<T> Gate(Tensor<T> x, int t) => engine.Reshape(engine.TensorNarrow(x, 1, t, 1), new[] { bh, 1, 1 });
        Tensor<T> Transposed(Tensor<T> x) => engine.TensorPermute(x, new[] { 0, 2, 1 });

        var q = Heads(inputs[0]);
        var k = Heads(inputs[1]);
        var v = Heads(inputs[2]);
        var a = Gates(inputs[3]);
        var b = Gates(inputs[4]);
        var g = Heads(gradOutput);

        // states[t] is S_t; states[0] is the zero initial state, so S_{t-1} is states[t] and S_t is states[t + 1].
        var states = new Tensor<T>[seqLen + 1];
        var keys = new Tensor<T>[seqLen];
        var deltas = new Tensor<T>[seqLen];
        states[0] = new Tensor<T>(new[] { bh, headDim, headDim });
        for (int t = 0; t < seqLen; t++)
        {
            keys[t] = engine.TensorMultiplyScalar(Column(k, t), kappa);
            deltas[t] = engine.TensorSubtract(Column(v, t), engine.BatchMatMul(states[t], keys[t]));
            states[t + 1] = engine.TensorAdd(
                engine.TensorMultiply(Gate(a, t), states[t]),
                engine.TensorMultiply(Gate(b, t), engine.BatchMatMul(deltas[t], Transposed(keys[t]))));
        }

        var dQ = new Tensor<T>[seqLen];
        var dK = new Tensor<T>[seqLen];
        var dV = new Tensor<T>[seqLen];
        var dA = new Tensor<T>[seqLen];
        var dB = new Tensor<T>[seqLen];
        var dS = new Tensor<T>(new[] { bh, headDim, headDim });
        var sumAxes = new[] { 1, 2 };
        for (int t = seqLen - 1; t >= 0; t--)
        {
            var gt = Column(g, t);
            var qt = Column(q, t);
            var previous = states[t];
            var betaT = Gate(b, t);
            // o_t = S_t q_t
            dS = engine.TensorAdd(dS, engine.BatchMatMul(gt, Transposed(qt)));
            dQ[t] = engine.BatchMatMul(Transposed(states[t + 1]), gt);
            // S_t = a S_{t-1} + b delta kᵀ with delta = v - S_{t-1} k
            var dSk = engine.BatchMatMul(dS, keys[t]);
            var dDelta = engine.TensorMultiply(betaT, dSk);
            dV[t] = dDelta;
            dB[t] = engine.ReduceSum(engine.TensorMultiply(deltas[t], dSk), sumAxes, keepDims: false);
            dA[t] = engine.ReduceSum(engine.TensorMultiply(dS, previous), sumAxes, keepDims: false);
            var dKey = engine.TensorSubtract(
                engine.TensorMultiply(betaT, engine.BatchMatMul(Transposed(dS), deltas[t])),
                engine.BatchMatMul(Transposed(previous), dDelta));
            dK[t] = engine.TensorMultiplyScalar(dKey, kappa);
            dS = engine.TensorSubtract(
                engine.TensorMultiply(Gate(a, t), dS),
                engine.BatchMatMul(dDelta, Transposed(keys[t])));
        }

        // [batch*heads, seqLen, headDim] back to [batch, seqLen, modelDim]; [batch*heads, seqLen] to [batch, seqLen, heads].
        Tensor<T> FromHeads(Tensor<T>[] steps)
        {
            var columns = new Tensor<T>[seqLen];
            for (int t = 0; t < seqLen; t++) columns[t] = engine.Reshape(steps[t], new[] { bh, headDim });
            var stacked = engine.Reshape(engine.TensorStack(columns, 1), new[] { batch, numHeads, seqLen, headDim });
            return engine.Reshape(engine.TensorPermute(stacked, new[] { 0, 2, 1, 3 }), new[] { batch, seqLen, modelDim });
        }
        Tensor<T> FromGates(Tensor<T>[] steps)
        {
            var stacked = engine.Reshape(engine.TensorStack(steps, 1), new[] { batch, numHeads, seqLen });
            return engine.Reshape(engine.TensorPermute(stacked, new[] { 0, 2, 1 }), new[] { batch, seqLen, numHeads });
        }

        return new[] { FromHeads(dQ), FromHeads(dK), FromHeads(dV), FromGates(dA), FromGates(dB) };
    }

    internal static void GatedDeltaNetScanBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int numHeads = (int)savedState[0];
        // Under create-graph the gradients are themselves differentiated (a second-order meta-learner, a gradient
        // penalty), so they have to come from recorded ops: the fused kernels below record nothing.
        if (DifferentiableOps._isBackwardCreateGraph)
        {
            var recorded = GatedDeltaNetScanGradRecorded(gradOutput, inputs, numHeads, engine);
            for (int i = 0; i < 5; i++)
                DifferentiableOps.AccumulateGrad(grads, inputs[i], recorded[i], engine);
            return;
        }
        if (engine.TryGet<DirectGpuTensorEngine>(out var gpu) && gpu.GatedDeltaNetScanBackwardGpu(gradOutput, inputs, numHeads) is { } gpuGrads)
        {
            for (int i = 0; i < 5; i++)
                DifferentiableOps.AccumulateGrad(grads, inputs[i], gpuGrads[i], engine);
            return;
        }
        var qProj = inputs[0];
        var kProj = inputs[1];
        var vProj = inputs[2];
        var alpha = inputs[3];
        var beta = inputs[4];

        int batch = qProj.Shape[0];
        int seqLen = qProj.Shape[1];
        int modelDim = qProj.Shape[2];
        int headDim = modelDim / numHeads;

        var dQ = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dK = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dV = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dA = new Tensor<T>(new[] { batch, seqLen, numHeads });
        var dB = new Tensor<T>(new[] { batch, seqLen, numHeads });

        if (typeof(T) == typeof(double))
        {
            GatedDeltaBackwardDouble(
                (double[])(object)gradOutput.GetDataArray(),
                (double[])(object)qProj.GetDataArray(), (double[])(object)kProj.GetDataArray(),
                (double[])(object)vProj.GetDataArray(), (double[])(object)alpha.GetDataArray(),
                (double[])(object)beta.GetDataArray(),
                (double[])(object)dQ.GetDataArray(), (double[])(object)dK.GetDataArray(),
                (double[])(object)dV.GetDataArray(), (double[])(object)dA.GetDataArray(),
                (double[])(object)dB.GetDataArray(),
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else if (typeof(T) == typeof(float))
        {
            GatedDeltaBackwardFloat(
                (float[])(object)gradOutput.GetDataArray(),
                (float[])(object)qProj.GetDataArray(), (float[])(object)kProj.GetDataArray(),
                (float[])(object)vProj.GetDataArray(), (float[])(object)alpha.GetDataArray(),
                (float[])(object)beta.GetDataArray(),
                (float[])(object)dQ.GetDataArray(), (float[])(object)dK.GetDataArray(),
                (float[])(object)dV.GetDataArray(), (float[])(object)dA.GetDataArray(),
                (float[])(object)dB.GetDataArray(),
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            GatedDeltaBackwardGeneric<T>(
                gradOutput.GetDataArray(),
                qProj.GetDataArray(), kProj.GetDataArray(), vProj.GetDataArray(),
                alpha.GetDataArray(), beta.GetDataArray(),
                dQ.GetDataArray(), dK.GetDataArray(), dV.GetDataArray(),
                dA.GetDataArray(), dB.GetDataArray(),
                batch, seqLen, modelDim, numHeads, headDim);
        }

        DifferentiableOps.AccumulateGrad(grads, qProj, dQ, engine);
        DifferentiableOps.AccumulateGrad(grads, kProj, dK, engine);
        DifferentiableOps.AccumulateGrad(grads, vProj, dV, engine);
        DifferentiableOps.AccumulateGrad(grads, alpha, dA, engine);
        DifferentiableOps.AccumulateGrad(grads, beta, dB, engine);
    }
}
