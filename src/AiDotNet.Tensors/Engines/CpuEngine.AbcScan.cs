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
    /// Fused ABC (Attention with Bounded-memory Control) slot recurrence over a whole sequence in a
    /// SINGLE op (forward + custom autodiff backward), replacing the per-element detached scalar loop
    /// <c>ABCLayer.SlotCompetitionForward</c> ran — which was DETACHED from the autodiff tape, so the
    /// q/k/v and forget-gate weights received no gradient at all through the recurrence (the same
    /// defect class as issue ooples/AiDotNet#1464, of which this is the remaining ABC case).
    /// Per head (modelDim split into <paramref name="numHeads"/> blocks of headDim), with slot state
    /// seeded from the trainable slot keys and scale = 1/sqrt(headDim):
    /// <code>
    ///   slot_-1[s,d] = slotInitScale * SK[h,s,d]
    ///   wScore_t[s]  = scale * sum_d K_t[d] * SK[h,s,d]
    ///   w_t          = softmax_s(wScore_t)                  (competitive write)
    ///   slot_t[s,d]  = fg_t * slot_{t-1}[s,d] + w_t[s] * V_t[d]
    ///   rScore_t[s]  = scale * sum_d Q_t[d] * slot_t[s,d]
    ///   r_t          = softmax_s(rScore_t)                  (competitive read)
    ///   O_t[d]       = sum_s r_t[s] * slot_t[s,d]
    /// </code>
    /// Records one tape node whose backward is the exact BPTT adjoint, so it is differentiable under an
    /// active <c>GradientTape</c> and the gradient reaches q/k/v, the forget gate AND the slot keys —
    /// the slot keys twice over, through the write scores at every step and through the initial state.
    /// The softmax is the standard max-subtracted form with no epsilon in the denominator: after the
    /// row max is subtracted the largest term is exactly 1, so the sum is always at least 1 and the
    /// guard the scalar loop carried could only bias the result.
    /// </summary>
    /// <param name="qProj">Query projection [batch, seqLen, modelDim].</param>
    /// <param name="kProj">Key projection [batch, seqLen, modelDim].</param>
    /// <param name="vProj">Value projection [batch, seqLen, modelDim].</param>
    /// <param name="forgetGate">Post-sigmoid scalar-per-head forget gate [batch, seqLen, numHeads].</param>
    /// <param name="slotKeys">Trainable slot keys [numHeads, numSlots, headDim].</param>
    /// <param name="numHeads">Number of heads; modelDim must be divisible by it.</param>
    /// <param name="slotInitScale">Scale applied to the slot keys to seed the initial slot state.</param>
    /// <returns>The slot-read output [batch, seqLen, modelDim].</returns>
    public virtual Tensor<T> AbcScanForward<T>(
        Tensor<T> qProj, Tensor<T> kProj, Tensor<T> vProj, Tensor<T> forgetGate,
        Tensor<T> slotKeys, int numHeads, double slotInitScale = 0.1)
    {
        if (qProj is null) throw new ArgumentNullException(nameof(qProj));
        if (kProj is null) throw new ArgumentNullException(nameof(kProj));
        if (vProj is null) throw new ArgumentNullException(nameof(vProj));
        if (forgetGate is null) throw new ArgumentNullException(nameof(forgetGate));
        if (slotKeys is null) throw new ArgumentNullException(nameof(slotKeys));
        if (numHeads < 1) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qProj.Rank != 3)
            throw new ArgumentException($"AbcScanForward expects rank-3 inputs [batch, seqLen, modelDim]; got rank {qProj.Rank}.", nameof(qProj));

        int batch = qProj.Shape[0];
        int seqLen = qProj.Shape[1];
        int modelDim = qProj.Shape[2];
        if (modelDim % numHeads != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        int headDim = modelDim / numHeads;
        EnsureSameShape(qProj, kProj, nameof(kProj));
        EnsureSameShape(qProj, vProj, nameof(vProj));
        if (forgetGate.Rank != 3 || forgetGate.Shape[0] != batch || forgetGate.Shape[1] != seqLen || forgetGate.Shape[2] != numHeads)
            throw new ArgumentException($"forgetGate must be [batch={batch}, seqLen={seqLen}, numHeads={numHeads}].", nameof(forgetGate));
        if (slotKeys.Rank != 3 || slotKeys.Shape[0] != numHeads || slotKeys.Shape[2] != headDim)
            throw new ArgumentException($"slotKeys must be [numHeads={numHeads}, numSlots, headDim={headDim}].", nameof(slotKeys));
        int numSlots = slotKeys.Shape[1];
        if (numSlots < 1)
            throw new ArgumentException(
                $"slotKeys must be [numHeads={numHeads}, numSlots >= 1, headDim={headDim}]; got numSlots={numSlots}.",
                nameof(slotKeys));

        var output = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            AbcForwardDouble(
                (double[])(object)qProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)forgetGate.GetDataArray()!,
                (double[])(object)slotKeys.GetDataArray()!, (double[])(object)output.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim, numSlots, slotInitScale);
        }
        else
        {
            AbcForwardGeneric<T>(
                qProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!,
                forgetGate.GetDataArray()!, slotKeys.GetDataArray()!, output.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim, numSlots, slotInitScale);
        }

        DifferentiableOps.RecordIfActive<T>(
            "AbcScan", output,
            new[] { qProj, kProj, vProj, forgetGate, slotKeys },
            AbcScanBackward<T>,
            savedState: new object[] { numHeads, slotInitScale });

        return output;
    }

    // One (batch, head) pair per chunk: slot state and output region are both private to the pair.
    private const int AbcBhGrain = 1;

    // ── Double fast path ─────────────────────────────────────────────────────────────────
    private static void AbcForwardDouble(
        double[] Q, double[] K, double[] V, double[] FG, double[] SK, double[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim, int numSlots, double initScale)
    {
        int sd = numSlots * headDim;
        double scale = 1.0 / Math.Sqrt(headDim);
        // Every (batch, head) pair is fully independent (private slot state, disjoint output region),
        // so the combined (b*numHeads) axis is embarrassingly parallel with no cross-channel reduction.
        CpuParallelSettings.ParallelForChunks(batch * numHeads, AbcBhGrain, (bhStart, bhCount) =>
        {
            var slot = new double[sd];
            var w = new double[numSlots];
            var r = new double[numSlots];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;
                int skOff = h * sd;

                for (int i = 0; i < sd; i++) slot[i] = initScale * SK[skOff + i];

                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    double fg = FG[(b * seqLen + t) * numHeads + h];

                    // Write scores against the slot keys, then softmax over slots.
                    for (int s = 0; s < numSlots; s++)
                    {
                        double dot = 0.0;
                        int krow = skOff + s * headDim;
                        for (int d = 0; d < headDim; d++) dot += K[baseOff + d] * SK[krow + d];
                        w[s] = dot * scale;
                    }
                    SoftmaxInPlaceDouble(w, 0, numSlots);

                    // Forget old content, write new content.
                    for (int s = 0; s < numSlots; s++)
                    {
                        double ws = w[s];
                        int srow = s * headDim;
                        for (int d = 0; d < headDim; d++)
                            slot[srow + d] = fg * slot[srow + d] + ws * V[baseOff + d];
                    }

                    // Read scores against the updated slot content, then softmax over slots.
                    for (int s = 0; s < numSlots; s++)
                    {
                        double dot = 0.0;
                        int srow = s * headDim;
                        for (int d = 0; d < headDim; d++) dot += Q[baseOff + d] * slot[srow + d];
                        r[s] = dot * scale;
                    }
                    SoftmaxInPlaceDouble(r, 0, numSlots);

                    for (int d = 0; d < headDim; d++)
                    {
                        double o = 0.0;
                        for (int s = 0; s < numSlots; s++) o += r[s] * slot[s * headDim + d];
                        outp[baseOff + d] = o;
                    }
                }
            }
        });
    }

    private static void SoftmaxInPlaceDouble(double[] x, int off, int n)
    {
        double max = x[off];
        for (int i = 1; i < n; i++) if (x[off + i] > max) max = x[off + i];
        double sum = 0.0;
        for (int i = 0; i < n; i++) { x[off + i] = Math.Exp(x[off + i] - max); sum += x[off + i]; }
        double inv = 1.0 / sum;
        for (int i = 0; i < n; i++) x[off + i] *= inv;
    }

    // dScore[i] = y[i] * (dy[i] - sum_j dy[j]*y[j]), the standard softmax Jacobian-vector product.
    // The max subtraction cancels exactly in the true softmax, so it contributes nothing here.
    private static void SoftmaxBackwardInPlaceDouble(double[] dy, double[] y, int yOff, int n)
    {
        double dotp = 0.0;
        for (int i = 0; i < n; i++) dotp += dy[i] * y[yOff + i];
        for (int i = 0; i < n; i++) dy[i] = y[yOff + i] * (dy[i] - dotp);
    }

    private static void AbcBackwardDouble(
        double[] dOut, double[] Q, double[] K, double[] V, double[] FG, double[] SK,
        double[] dQ, double[] dK, double[] dV, double[] dFG, double[] dSK,
        int batch, int seqLen, int modelDim, int numHeads, int headDim, int numSlots, double initScale)
    {
        int sd = numSlots * headDim;
        double scale = 1.0 / Math.Sqrt(headDim);
        // Parallelize over HEADS only, not (batch*head): dSK[h] is shared by every batch element, so
        // the batch loop stays sequential inside a head to keep the accumulation lock-free.
        CpuParallelSettings.ParallelForChunks(numHeads, AbcBhGrain, (hStart, hCount) =>
        {
            var slot = new double[sd];
            var slotTraj = new double[seqLen * sd];
            var wTraj = new double[seqLen * numSlots];
            var rTraj = new double[seqLen * numSlots];
            var dSlot = new double[sd];
            var dw = new double[numSlots];
            var dr = new double[numSlots];
            int hEnd = hStart + hCount;
            for (int h = hStart; h < hEnd; h++)
            {
                int hOff = h * headDim;
                int skOff = h * sd;
                for (int b = 0; b < batch; b++)
                {
                    // Forward recompute, saving the slot trajectory and both softmax outputs.
                    for (int i = 0; i < sd; i++) slot[i] = initScale * SK[skOff + i];
                    for (int t = 0; t < seqLen; t++)
                    {
                        int baseOff = (b * seqLen + t) * modelDim + hOff;
                        double fg = FG[(b * seqLen + t) * numHeads + h];
                        int wOff = t * numSlots;
                        for (int s = 0; s < numSlots; s++)
                        {
                            double dot = 0.0;
                            int krow = skOff + s * headDim;
                            for (int d = 0; d < headDim; d++) dot += K[baseOff + d] * SK[krow + d];
                            wTraj[wOff + s] = dot * scale;
                        }
                        SoftmaxInPlaceDouble(wTraj, wOff, numSlots);
                        for (int s = 0; s < numSlots; s++)
                        {
                            double ws = wTraj[wOff + s];
                            int srow = s * headDim;
                            for (int d = 0; d < headDim; d++)
                                slot[srow + d] = fg * slot[srow + d] + ws * V[baseOff + d];
                        }
                        Array.Copy(slot, 0, slotTraj, t * sd, sd);
                        for (int s = 0; s < numSlots; s++)
                        {
                            double dot = 0.0;
                            int srow = s * headDim;
                            for (int d = 0; d < headDim; d++) dot += Q[baseOff + d] * slot[srow + d];
                            rTraj[wOff + s] = dot * scale;
                        }
                        SoftmaxInPlaceDouble(rTraj, wOff, numSlots);
                    }

                    // Reverse sweep.
                    Array.Clear(dSlot, 0, sd);
                    for (int t = seqLen - 1; t >= 0; t--)
                    {
                        int baseOff = (b * seqLen + t) * modelDim + hOff;
                        int gOff = (b * seqLen + t) * numHeads + h;
                        int wOff = t * numSlots;
                        int stOff = t * sd;
                        double fg = FG[gOff];

                        // O_t[d] = sum_s r[s] * slot_t[s,d]
                        for (int s = 0; s < numSlots; s++)
                        {
                            double rs = rTraj[wOff + s];
                            int srow = s * headDim;
                            double acc = 0.0;
                            for (int d = 0; d < headDim; d++)
                            {
                                double dO = dOut[baseOff + d];
                                acc += dO * slotTraj[stOff + srow + d];
                                dSlot[srow + d] += dO * rs;
                            }
                            dr[s] = acc;
                        }

                        // Softmax backward for the read weights, then the read scores.
                        SoftmaxBackwardInPlaceDouble(dr, rTraj, wOff, numSlots);
                        for (int s = 0; s < numSlots; s++)
                        {
                            double drs = dr[s] * scale;
                            int srow = s * headDim;
                            for (int d = 0; d < headDim; d++)
                            {
                                dQ[baseOff + d] += drs * slotTraj[stOff + srow + d];
                                dSlot[srow + d] += drs * Q[baseOff + d];
                            }
                        }

                        // slot_t[s,d] = fg * slot_{t-1}[s,d] + w[s] * V_t[d]
                        double dfg = 0.0;
                        int sprevOff = (t - 1) * sd;
                        for (int s = 0; s < numSlots; s++)
                        {
                            int srow = s * headDim;
                            double accW = 0.0;
                            double ws = wTraj[wOff + s];
                            for (int d = 0; d < headDim; d++)
                            {
                                double ds = dSlot[srow + d];
                                double sprev = t > 0
                                    ? slotTraj[sprevOff + srow + d]
                                    : initScale * SK[skOff + srow + d];
                                dfg += ds * sprev;
                                accW += ds * V[baseOff + d];
                                dV[baseOff + d] += ds * ws;
                            }
                            dw[s] = accW;
                        }
                        dFG[gOff] += dfg;

                        // Carry the state adjoint back one step: dslot_{t-1} = fg * dslot_t.
                        for (int i = 0; i < sd; i++) dSlot[i] *= fg;

                        // Softmax backward for the write weights, then the write scores.
                        SoftmaxBackwardInPlaceDouble(dw, wTraj, wOff, numSlots);
                        for (int s = 0; s < numSlots; s++)
                        {
                            double dws = dw[s] * scale;
                            int krow = skOff + s * headDim;
                            for (int d = 0; d < headDim; d++)
                            {
                                dK[baseOff + d] += dws * SK[krow + d];
                                dSK[krow + d] += dws * K[baseOff + d];
                            }
                        }
                    }

                    // Boundary: slot_-1[s,d] = initScale * SK[h,s,d]; dSlot now holds dL/dslot_-1.
                    for (int i = 0; i < sd; i++) dSK[skOff + i] += initScale * dSlot[i];
                }
            }
        });
    }

    // ── Generic-T path ───────────────────────────────────────────────────────────────────
    private static void AbcForwardGeneric<T>(
        T[] Q, T[] K, T[] V, T[] FG, T[] SK, T[] outp,
        int batch, int seqLen, int modelDim, int numHeads, int headDim, int numSlots, double initScale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int sd = numSlots * headDim;
        T scale = ops.FromDouble(1.0 / Math.Sqrt(headDim));
        T init = ops.FromDouble(initScale);
        // Parallelize lock-free over the independent (b*numHeads) axis; see the double path.
        CpuParallelSettings.ParallelForChunks(batch * numHeads, AbcBhGrain, (bhStart, bhCount) =>
        {
            var slot = new T[sd];
            var w = new T[numSlots];
            var r = new T[numSlots];
            int bhEnd = bhStart + bhCount;
            for (int bh = bhStart; bh < bhEnd; bh++)
            {
                int b = bh / numHeads;
                int h = bh % numHeads;
                int hOff = h * headDim;
                int skOff = h * sd;

                for (int i = 0; i < sd; i++) slot[i] = ops.Multiply(init, SK[skOff + i]);

                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    T fg = FG[(b * seqLen + t) * numHeads + h];

                    for (int s = 0; s < numSlots; s++)
                    {
                        T dot = ops.Zero;
                        int krow = skOff + s * headDim;
                        for (int d = 0; d < headDim; d++)
                            dot = ops.Add(dot, ops.Multiply(K[baseOff + d], SK[krow + d]));
                        w[s] = ops.Multiply(dot, scale);
                    }
                    SoftmaxInPlaceGeneric(w, 0, numSlots, ops);

                    for (int s = 0; s < numSlots; s++)
                    {
                        T ws = w[s];
                        int srow = s * headDim;
                        for (int d = 0; d < headDim; d++)
                            slot[srow + d] = ops.Add(ops.Multiply(fg, slot[srow + d]), ops.Multiply(ws, V[baseOff + d]));
                    }

                    for (int s = 0; s < numSlots; s++)
                    {
                        T dot = ops.Zero;
                        int srow = s * headDim;
                        for (int d = 0; d < headDim; d++)
                            dot = ops.Add(dot, ops.Multiply(Q[baseOff + d], slot[srow + d]));
                        r[s] = ops.Multiply(dot, scale);
                    }
                    SoftmaxInPlaceGeneric(r, 0, numSlots, ops);

                    for (int d = 0; d < headDim; d++)
                    {
                        T o = ops.Zero;
                        for (int s = 0; s < numSlots; s++)
                            o = ops.Add(o, ops.Multiply(r[s], slot[s * headDim + d]));
                        outp[baseOff + d] = o;
                    }
                }
            }
        });
    }

    private static void SoftmaxInPlaceGeneric<T>(T[] x, int off, int n, INumericOperations<T> ops)
    {
        T max = x[off];
        for (int i = 1; i < n; i++) if (ops.GreaterThan(x[off + i], max)) max = x[off + i];
        T sum = ops.Zero;
        for (int i = 0; i < n; i++) { x[off + i] = ops.Exp(ops.Subtract(x[off + i], max)); sum = ops.Add(sum, x[off + i]); }
        T inv = ops.Divide(ops.One, sum);
        for (int i = 0; i < n; i++) x[off + i] = ops.Multiply(x[off + i], inv);
    }

    private static void SoftmaxBackwardInPlaceGeneric<T>(T[] dy, T[] y, int yOff, int n, INumericOperations<T> ops)
    {
        T dotp = ops.Zero;
        for (int i = 0; i < n; i++) dotp = ops.Add(dotp, ops.Multiply(dy[i], y[yOff + i]));
        for (int i = 0; i < n; i++) dy[i] = ops.Multiply(y[yOff + i], ops.Subtract(dy[i], dotp));
    }

    private static void AbcBackwardGeneric<T>(
        T[] dOut, T[] Q, T[] K, T[] V, T[] FG, T[] SK,
        T[] dQ, T[] dK, T[] dV, T[] dFG, T[] dSK,
        int batch, int seqLen, int modelDim, int numHeads, int headDim, int numSlots, double initScale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int sd = numSlots * headDim;
        T scale = ops.FromDouble(1.0 / Math.Sqrt(headDim));
        T init = ops.FromDouble(initScale);
        // Parallelize over HEADS only; dSK[h] is shared across the batch. See the double path.
        CpuParallelSettings.ParallelForChunks(numHeads, AbcBhGrain, (hStart, hCount) =>
        {
            var slot = new T[sd];
            var slotTraj = new T[seqLen * sd];
            var wTraj = new T[seqLen * numSlots];
            var rTraj = new T[seqLen * numSlots];
            var dSlot = new T[sd];
            var dw = new T[numSlots];
            var dr = new T[numSlots];
            int hEnd = hStart + hCount;
            for (int h = hStart; h < hEnd; h++)
            {
                int hOff = h * headDim;
                int skOff = h * sd;
                for (int b = 0; b < batch; b++)
                {
                    for (int i = 0; i < sd; i++) slot[i] = ops.Multiply(init, SK[skOff + i]);
                    for (int t = 0; t < seqLen; t++)
                    {
                        int baseOff = (b * seqLen + t) * modelDim + hOff;
                        T fg = FG[(b * seqLen + t) * numHeads + h];
                        int wOff = t * numSlots;
                        for (int s = 0; s < numSlots; s++)
                        {
                            T dot = ops.Zero;
                            int krow = skOff + s * headDim;
                            for (int d = 0; d < headDim; d++)
                                dot = ops.Add(dot, ops.Multiply(K[baseOff + d], SK[krow + d]));
                            wTraj[wOff + s] = ops.Multiply(dot, scale);
                        }
                        SoftmaxInPlaceGeneric(wTraj, wOff, numSlots, ops);
                        for (int s = 0; s < numSlots; s++)
                        {
                            T ws = wTraj[wOff + s];
                            int srow = s * headDim;
                            for (int d = 0; d < headDim; d++)
                                slot[srow + d] = ops.Add(ops.Multiply(fg, slot[srow + d]), ops.Multiply(ws, V[baseOff + d]));
                        }
                        Array.Copy(slot, 0, slotTraj, t * sd, sd);
                        for (int s = 0; s < numSlots; s++)
                        {
                            T dot = ops.Zero;
                            int srow = s * headDim;
                            for (int d = 0; d < headDim; d++)
                                dot = ops.Add(dot, ops.Multiply(Q[baseOff + d], slot[srow + d]));
                            rTraj[wOff + s] = ops.Multiply(dot, scale);
                        }
                        SoftmaxInPlaceGeneric(rTraj, wOff, numSlots, ops);
                    }

                    for (int i = 0; i < sd; i++) dSlot[i] = ops.Zero;
                    for (int t = seqLen - 1; t >= 0; t--)
                    {
                        int baseOff = (b * seqLen + t) * modelDim + hOff;
                        int gOff = (b * seqLen + t) * numHeads + h;
                        int wOff = t * numSlots;
                        int stOff = t * sd;
                        T fg = FG[gOff];

                        for (int s = 0; s < numSlots; s++)
                        {
                            T rs = rTraj[wOff + s];
                            int srow = s * headDim;
                            T acc = ops.Zero;
                            for (int d = 0; d < headDim; d++)
                            {
                                T dO = dOut[baseOff + d];
                                acc = ops.Add(acc, ops.Multiply(dO, slotTraj[stOff + srow + d]));
                                dSlot[srow + d] = ops.Add(dSlot[srow + d], ops.Multiply(dO, rs));
                            }
                            dr[s] = acc;
                        }

                        SoftmaxBackwardInPlaceGeneric(dr, rTraj, wOff, numSlots, ops);
                        for (int s = 0; s < numSlots; s++)
                        {
                            T drs = ops.Multiply(dr[s], scale);
                            int srow = s * headDim;
                            for (int d = 0; d < headDim; d++)
                            {
                                dQ[baseOff + d] = ops.Add(dQ[baseOff + d], ops.Multiply(drs, slotTraj[stOff + srow + d]));
                                dSlot[srow + d] = ops.Add(dSlot[srow + d], ops.Multiply(drs, Q[baseOff + d]));
                            }
                        }

                        T dfg = ops.Zero;
                        int sprevOff = (t - 1) * sd;
                        for (int s = 0; s < numSlots; s++)
                        {
                            int srow = s * headDim;
                            T accW = ops.Zero;
                            T ws = wTraj[wOff + s];
                            for (int d = 0; d < headDim; d++)
                            {
                                T ds = dSlot[srow + d];
                                T sprev = t > 0
                                    ? slotTraj[sprevOff + srow + d]
                                    : ops.Multiply(init, SK[skOff + srow + d]);
                                dfg = ops.Add(dfg, ops.Multiply(ds, sprev));
                                accW = ops.Add(accW, ops.Multiply(ds, V[baseOff + d]));
                                dV[baseOff + d] = ops.Add(dV[baseOff + d], ops.Multiply(ds, ws));
                            }
                            dw[s] = accW;
                        }
                        dFG[gOff] = ops.Add(dFG[gOff], dfg);

                        for (int i = 0; i < sd; i++) dSlot[i] = ops.Multiply(dSlot[i], fg);

                        SoftmaxBackwardInPlaceGeneric(dw, wTraj, wOff, numSlots, ops);
                        for (int s = 0; s < numSlots; s++)
                        {
                            T dws = ops.Multiply(dw[s], scale);
                            int krow = skOff + s * headDim;
                            for (int d = 0; d < headDim; d++)
                            {
                                dK[baseOff + d] = ops.Add(dK[baseOff + d], ops.Multiply(dws, SK[krow + d]));
                                dSK[krow + d] = ops.Add(dSK[krow + d], ops.Multiply(dws, K[baseOff + d]));
                            }
                        }
                    }

                    for (int i = 0; i < sd; i++)
                        dSK[skOff + i] = ops.Add(dSK[skOff + i], ops.Multiply(init, dSlot[i]));
                }
            }
        });
    }

    private static void AbcScanBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int numHeads = (int)savedState[0];
        double initScale = (double)savedState[1];
        var qProj = inputs[0];
        var kProj = inputs[1];
        var vProj = inputs[2];
        var forgetGate = inputs[3];
        var slotKeys = inputs[4];

        int batch = qProj.Shape[0];
        int seqLen = qProj.Shape[1];
        int modelDim = qProj.Shape[2];
        int headDim = modelDim / numHeads;
        int numSlots = slotKeys.Shape[1];

        var dQ = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dK = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dV = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dFG = new Tensor<T>(new[] { batch, seqLen, numHeads });
        var dSK = new Tensor<T>(new[] { numHeads, numSlots, headDim });

        if (typeof(T) == typeof(double))
        {
            AbcBackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)qProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)forgetGate.GetDataArray()!,
                (double[])(object)slotKeys.GetDataArray()!,
                (double[])(object)dQ.GetDataArray()!, (double[])(object)dK.GetDataArray()!,
                (double[])(object)dV.GetDataArray()!, (double[])(object)dFG.GetDataArray()!,
                (double[])(object)dSK.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim, numSlots, initScale);
        }
        else
        {
            AbcBackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                qProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!,
                forgetGate.GetDataArray()!, slotKeys.GetDataArray()!,
                dQ.GetDataArray()!, dK.GetDataArray()!, dV.GetDataArray()!,
                dFG.GetDataArray()!, dSK.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim, numSlots, initScale);
        }

        DifferentiableOps.AccumulateGrad(grads, qProj, dQ, engine);
        DifferentiableOps.AccumulateGrad(grads, kProj, dK, engine);
        DifferentiableOps.AccumulateGrad(grads, vProj, dV, engine);
        DifferentiableOps.AccumulateGrad(grads, forgetGate, dFG, engine);
        DifferentiableOps.AccumulateGrad(grads, slotKeys, dSK, engine);
    }
}
