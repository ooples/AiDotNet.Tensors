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
    /// Fused Gated Linear Attention (GLA) recurrence over a whole sequence in a SINGLE op
    /// (forward + custom autodiff backward), replacing the per-element detached scalar loop the
    /// decomposed <c>GatedLinearAttentionLayer</c> ran — which was both slow (per-element index-array
    /// allocations) and DETACHED from the autodiff tape (so the q/k/v/gate weights received no gradient
    /// through the recurrence; issue ooples/AiDotNet#1464). Per head (modelDim split into
    /// <paramref name="numHeads"/> blocks of headDim, keyDim = headDim), with matrix state S[di,ki] = 0
    /// and a scalar-per-head gate g_t = gate[..., headStart]:
    /// <code>
    ///   S_t[di,ki] = g_t * S_{t-1}[di,ki] + K_t[ki] * V_t[di]
    ///   O_t[ki]    = sum_di Q_t[di] * S_t[di,ki]
    /// </code>
    /// Records one tape node whose backward is the exact BPTT adjoint, so it is differentiable under an
    /// active <c>GradientTape</c>. Inputs are the already-projected q/k/v/gate streams; the gate is the
    /// post-sigmoid value (the layer's sigmoid records its own tape node).
    /// </summary>
    /// <param name="qProj">Query projection [batch, seqLen, modelDim].</param>
    /// <param name="kProj">Key projection [batch, seqLen, modelDim].</param>
    /// <param name="vProj">Value projection [batch, seqLen, modelDim].</param>
    /// <param name="gate">Post-sigmoid gate [batch, seqLen, modelDim]; only the per-head first element is used.</param>
    /// <param name="numHeads">Number of heads; modelDim must be divisible by it.</param>
    /// <returns>The attention output [batch, seqLen, modelDim].</returns>
    public virtual Tensor<T> GlaScanForward<T>(
        Tensor<T> qProj, Tensor<T> kProj, Tensor<T> vProj, Tensor<T> gate, int numHeads)
    {
        if (qProj is null) throw new ArgumentNullException(nameof(qProj));
        if (kProj is null) throw new ArgumentNullException(nameof(kProj));
        if (vProj is null) throw new ArgumentNullException(nameof(vProj));
        if (gate is null) throw new ArgumentNullException(nameof(gate));
        if (numHeads < 1) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qProj.Rank != 3)
            throw new ArgumentException($"GlaScanForward expects rank-3 inputs [batch, seqLen, modelDim]; got rank {qProj.Rank}.", nameof(qProj));

        int batch = qProj.Shape[0];
        int seqLen = qProj.Shape[1];
        int modelDim = qProj.Shape[2];
        if (modelDim % numHeads != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        int headDim = modelDim / numHeads;
        EnsureSameShape(qProj, kProj, nameof(kProj));
        EnsureSameShape(qProj, vProj, nameof(vProj));
        EnsureSameShape(qProj, gate, nameof(gate));

        var output = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            GlaForwardDouble(
                (double[])(object)qProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)gate.GetDataArray()!,
                (double[])(object)output.GetDataArray()!, batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            GlaForwardGeneric<T>(
                qProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!, gate.GetDataArray()!,
                output.GetDataArray()!, batch, seqLen, modelDim, numHeads, headDim);
        }

        DifferentiableOps.RecordIfActive<T>(
            "GlaScan", output,
            new[] { qProj, kProj, vProj, gate },
            GlaScanBackward<T>,
            savedState: new object[] { numHeads });

        return output;
    }

    // ── Double fast path ─────────────────────────────────────────────────────────────────
    private static void GlaForwardDouble(
        double[] Q, double[] K, double[] V, double[] G, double[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        var S = new double[hh];
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                Array.Clear(S, 0, hh);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    double g = G[baseOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        double vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = g * S[srow + ki] + K[baseOff + ki] * vv;
                    }
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        double o = 0.0;
                        for (int di = 0; di < headDim; di++)
                            o += Q[baseOff + di] * S[di * headDim + ki];
                        outp[baseOff + ki] = o;
                    }
                }
            }
        }
    }

    private static void GlaBackwardDouble(
        double[] dOut, double[] Q, double[] K, double[] V, double[] G,
        double[] dQ, double[] dK, double[] dV, double[] dG,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        var Straj = new double[seqLen * hh];
        var S = new double[hh];
        var dS = new double[hh];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;

                // Forward recompute, saving the post-update state trajectory.
                Array.Clear(S, 0, hh);
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    double g = G[baseOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        double vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = g * S[srow + ki] + K[baseOff + ki] * vv;
                    }
                    Array.Copy(S, 0, Straj, t * hh, hh);
                }

                // Reverse sweep.
                Array.Clear(dS, 0, hh);
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int stOff = t * hh;
                    int sprevOff = (t - 1) * hh;
                    double g = G[baseOff];

                    // Output backward: O[ki] = sum_di Q[di]*S_t[di,ki].
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        double dOutVal = dOut[baseOff + ki];
                        for (int di = 0; di < headDim; di++)
                        {
                            double sval = Straj[stOff + di * headDim + ki];
                            dQ[baseOff + di] += dOutVal * sval;
                            dS[di * headDim + ki] += dOutVal * Q[baseOff + di];
                        }
                    }

                    // State-update backward: S_t[di,ki] = g*S_{t-1}[di,ki] + K[ki]*V[di].
                    double dg = 0.0;
                    for (int di = 0; di < headDim; di++)
                    {
                        double vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            double dStv = dS[srow + ki];
                            double sprev = t > 0 ? Straj[sprevOff + srow + ki] : 0.0;
                            dg += dStv * sprev;
                            dK[baseOff + ki] += dStv * vv;
                            dV[baseOff + di] += dStv * K[baseOff + ki];
                        }
                    }
                    dG[baseOff] += dg; // scalar-per-head gate: only the head-start position

                    // Propagate adjoint to the previous step: dS_{t-1} = g * dS_t.
                    for (int i = 0; i < hh; i++) dS[i] *= g;
                }
            }
        }
    }

    // ── Generic-T path ───────────────────────────────────────────────────────────────────
    private static void GlaForwardGeneric<T>(
        T[] Q, T[] K, T[] V, T[] G, T[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hh = headDim * headDim;
        var S = new T[hh];
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int i = 0; i < hh; i++) S[i] = ops.Zero;
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    T g = G[baseOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        T vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = ops.Add(ops.Multiply(g, S[srow + ki]), ops.Multiply(K[baseOff + ki], vv));
                    }
                    for (int ki = 0; ki < headDim; ki++)
                    {
                        T o = ops.Zero;
                        for (int di = 0; di < headDim; di++)
                            o = ops.Add(o, ops.Multiply(Q[baseOff + di], S[di * headDim + ki]));
                        outp[baseOff + ki] = o;
                    }
                }
            }
        }
    }

    private static void GlaBackwardGeneric<T>(
        T[] dOut, T[] Q, T[] K, T[] V, T[] G,
        T[] dQ, T[] dK, T[] dV, T[] dG,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hh = headDim * headDim;
        var Straj = new T[seqLen * hh];
        var S = new T[hh];
        var dS = new T[hh];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;
                for (int i = 0; i < hh; i++) S[i] = ops.Zero;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    T g = G[baseOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        T vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            S[srow + ki] = ops.Add(ops.Multiply(g, S[srow + ki]), ops.Multiply(K[baseOff + ki], vv));
                    }
                    Array.Copy(S, 0, Straj, t * hh, hh);
                }

                for (int i = 0; i < hh; i++) dS[i] = ops.Zero;
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int stOff = t * hh;
                    int sprevOff = (t - 1) * hh;
                    T g = G[baseOff];

                    for (int ki = 0; ki < headDim; ki++)
                    {
                        T dOutVal = dOut[baseOff + ki];
                        for (int di = 0; di < headDim; di++)
                        {
                            T sval = Straj[stOff + di * headDim + ki];
                            dQ[baseOff + di] = ops.Add(dQ[baseOff + di], ops.Multiply(dOutVal, sval));
                            dS[di * headDim + ki] = ops.Add(dS[di * headDim + ki], ops.Multiply(dOutVal, Q[baseOff + di]));
                        }
                    }

                    T dg = ops.Zero;
                    for (int di = 0; di < headDim; di++)
                    {
                        T vv = V[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            T dStv = dS[srow + ki];
                            T sprev = t > 0 ? Straj[sprevOff + srow + ki] : ops.Zero;
                            dg = ops.Add(dg, ops.Multiply(dStv, sprev));
                            dK[baseOff + ki] = ops.Add(dK[baseOff + ki], ops.Multiply(dStv, vv));
                            dV[baseOff + di] = ops.Add(dV[baseOff + di], ops.Multiply(dStv, K[baseOff + ki]));
                        }
                    }
                    dG[baseOff] = ops.Add(dG[baseOff], dg);

                    for (int i = 0; i < hh; i++) dS[i] = ops.Multiply(dS[i], g);
                }
            }
        }
    }

    private static void GlaScanBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int numHeads = (int)savedState[0];
        var qProj = inputs[0];
        var kProj = inputs[1];
        var vProj = inputs[2];
        var gate = inputs[3];

        int batch = qProj.Shape[0];
        int seqLen = qProj.Shape[1];
        int modelDim = qProj.Shape[2];
        int headDim = modelDim / numHeads;

        var dQ = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dK = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dV = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dG = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            GlaBackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)qProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)gate.GetDataArray()!,
                (double[])(object)dQ.GetDataArray()!, (double[])(object)dK.GetDataArray()!,
                (double[])(object)dV.GetDataArray()!, (double[])(object)dG.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            GlaBackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                qProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!, gate.GetDataArray()!,
                dQ.GetDataArray()!, dK.GetDataArray()!, dV.GetDataArray()!, dG.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }

        DifferentiableOps.AccumulateGrad(grads, qProj, dQ, engine);
        DifferentiableOps.AccumulateGrad(grads, kProj, dK, engine);
        DifferentiableOps.AccumulateGrad(grads, vProj, dV, engine);
        DifferentiableOps.AccumulateGrad(grads, gate, dG, engine);
    }
}
