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
    /// Fused Real-Gated Linear Recurrent Unit (RG-LRU) scan over a whole sequence in a SINGLE op
    /// (forward + custom autodiff backward), replacing the per-timestep tape micro-ops the decomposed
    /// <c>RealGatedLinearRecurrenceLayer.GatedRecurrenceForward</c> loop records — the core sequence
    /// mixer of Griffin / Hawk / RecurrentGemma (De et al. 2024; issue ooples/AiDotNet#1464). Inputs are
    /// the per-position value/gate streams (the value and gates are projected by the caller); the
    /// per-channel recurrence is, with base = sigmoid(-decay) = exp(-softplus(decay)) and h[ch] = 0:
    /// <code>
    ///   a_t   = recGate_t * base                       (input-modulated decay, in (0,1))
    ///   s_t   = sqrt(max(0, 1 - a_t^2))                (magnitude-preserving input scale)
    ///   h_t   = a_t * h_{t-1} + s_t * (inpGate_t * v_t)
    ///   y_t   = h_t
    /// </code>
    /// Records one tape node whose backward is the exact BPTT adjoint, so it is differentiable under an
    /// active <c>GradientTape</c>.
    /// </summary>
    /// <param name="value">Value stream v [batch, seqLen, recDim] (already x·W_v).</param>
    /// <param name="recGate">Recurrence gate r [batch, seqLen, recDim] (post-sigmoid, in (0,1)).</param>
    /// <param name="inpGate">Input gate i [batch, seqLen, recDim] (post-sigmoid, in (0,1)).</param>
    /// <param name="decay">Per-channel learned decay [recDim] (raw; base = sigmoid(-decay)).</param>
    /// <returns>The recurrence output h [batch, seqLen, recDim].</returns>
    public virtual Tensor<T> RgLruScanForward<T>(
        Tensor<T> value, Tensor<T> recGate, Tensor<T> inpGate, Tensor<T> decay)
    {
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (recGate is null) throw new ArgumentNullException(nameof(recGate));
        if (inpGate is null) throw new ArgumentNullException(nameof(inpGate));
        if (decay is null) throw new ArgumentNullException(nameof(decay));
        if (value.Rank != 3)
            throw new ArgumentException($"RgLruScanForward expects rank-3 inputs [batch, seqLen, recDim]; got rank {value.Rank}.", nameof(value));

        int batch = value.Shape[0];
        int seqLen = value.Shape[1];
        int recDim = value.Shape[2];
        EnsureSameShape(value, recGate, nameof(recGate));
        EnsureSameShape(value, inpGate, nameof(inpGate));
        if (decay.Length != recDim)
            throw new ArgumentException($"decay length ({decay.Length}) must equal recDim ({recDim}).", nameof(decay));

        var output = new Tensor<T>(new[] { batch, seqLen, recDim });

        if (typeof(T) == typeof(double))
        {
            RgLruForwardDouble(
                (double[])(object)value.GetDataArray()!, (double[])(object)recGate.GetDataArray()!,
                (double[])(object)inpGate.GetDataArray()!, (double[])(object)decay.GetDataArray()!,
                (double[])(object)output.GetDataArray()!, batch, seqLen, recDim);
        }
        else
        {
            RgLruForwardGeneric<T>(
                value.GetDataArray()!, recGate.GetDataArray()!, inpGate.GetDataArray()!,
                decay.GetDataArray()!, output.GetDataArray()!, batch, seqLen, recDim);
        }

        DifferentiableOps.RecordIfActive<T>(
            "RgLruScan", output,
            new[] { value, recGate, inpGate, decay },
            RgLruScanBackward<T>,
            savedState: null);

        return output;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static double SigD(double x) => 1.0 / (1.0 + Math.Exp(-x));

    // ── Double fast path ─────────────────────────────────────────────────────────────────
    private static void RgLruForwardDouble(
        double[] V, double[] R, double[] I, double[] decay, double[] outp,
        int batch, int seqLen, int recDim)
    {
        var baseDecay = new double[recDim];
        for (int c = 0; c < recDim; c++) baseDecay[c] = SigD(-decay[c]);

        // Each recurrent channel c evolves an independent scalar state, so c is parallelizable; keep b
        // outer (the per-channel gradients dBase/dDecay accumulate across batch in the backward). Local
        // scalar state + disjoint output writes — fully lock-free.
        for (int b = 0; b < batch; b++)
        {
            int bb = b;
            CpuParallelSettings.ParallelForChunks(recDim, RgLruChannelGrain, (cStart, cCount) =>
            {
                int cEnd = cStart + cCount;
                for (int c = cStart; c < cEnd; c++)
                {
                    double hc = 0.0;
                    double bdc = baseDecay[c];
                    for (int t = 0; t < seqLen; t++)
                    {
                        int off = (bb * seqLen + t) * recDim;
                        double a = R[off + c] * bdc;
                        double oneMinus = 1.0 - a * a;
                        double s = oneMinus > 0.0 ? Math.Sqrt(oneMinus) : 0.0;
                        hc = a * hc + s * (I[off + c] * V[off + c]);
                        outp[off + c] = hc;
                    }
                }
            });
        }
    }

    /// <summary>
    /// Minimum recurrent channels per parallel chunk for the channel-parallel scan kernels (RG-LRU,
    /// RWKV-4). Each channel owns a private scalar state and writes disjoint output/gradient slots.
    /// </summary>
    private const int RgLruChannelGrain = 8;

    private static void RgLruBackwardDouble(
        double[] dOut, double[] V, double[] R, double[] I, double[] decay,
        double[] dV, double[] dR, double[] dI, double[] dDecay,
        int batch, int seqLen, int recDim)
    {
        var baseDecay = new double[recDim];
        for (int c = 0; c < recDim; c++) baseDecay[c] = SigD(-decay[c]);

        var dBase = new double[recDim]; // [recDim], one slot per channel — disjoint across c-threads

        // Parallelize over channel c (independent scalar recurrence); keep b outer so dBase accumulates
        // across batch serially. Each c owns a private scalar state + a private seqLen trajectory.
        for (int b = 0; b < batch; b++)
        {
            int bb = b;
            CpuParallelSettings.ParallelForChunks(recDim, RgLruChannelGrain, (cStart, cCount) =>
            {
                var hTrajC = new double[seqLen]; // this channel's post-update state over time
                int cEnd = cStart + cCount;
                for (int c = cStart; c < cEnd; c++)
                {
                    double bd = baseDecay[c];

                    // Forward recompute for this channel, saving its post-update state trajectory.
                    double hc = 0.0;
                    for (int t = 0; t < seqLen; t++)
                    {
                        int off = (bb * seqLen + t) * recDim;
                        double a = R[off + c] * bd;
                        double oneMinus = 1.0 - a * a;
                        double s = oneMinus > 0.0 ? Math.Sqrt(oneMinus) : 0.0;
                        hc = a * hc + s * (I[off + c] * V[off + c]);
                        hTrajC[t] = hc;
                    }

                    // Reverse sweep. dHc carries the adjoint of h_t from t+1 (and y_t = h_t).
                    double dHc = 0.0;
                    for (int t = seqLen - 1; t >= 0; t--)
                    {
                        int off = (bb * seqLen + t) * recDim;
                        double a = R[off + c] * bd;
                        double oneMinus = 1.0 - a * a;
                        double s = oneMinus > 0.0 ? Math.Sqrt(oneMinus) : 0.0;
                        double iv = I[off + c] * V[off + c];
                        double hPrev = t > 0 ? hTrajC[t - 1] : 0.0;

                        // y_t = h_t.
                        double dh = dHc + dOut[off + c];

                        // h_t = a*hPrev + s*(i*v).
                        double dG = dh * s;
                        dI[off + c] += dG * V[off + c];
                        dV[off + c] += dG * I[off + c];

                        double dS = dh * iv;
                        // s = sqrt(1-a^2) (when positive): ds/da = -a/s.
                        double dA = dh * hPrev;
                        if (s > 1e-12) dA += dS * (-a / s);

                        // a = r * base.
                        dR[off + c] += dA * bd;
                        dBase[c] += dA * R[off + c];

                        // Adjoint to previous step: d(h_{t-1}) = dh * a.
                        dHc = dh * a;
                    }
                }
            });
        }

        // base = sigmoid(-decay): d(base)/d(decay) = -base*(1-base).
        for (int c = 0; c < recDim; c++)
            dDecay[c] += dBase[c] * (-baseDecay[c] * (1.0 - baseDecay[c]));
    }

    // ── Generic-T path ───────────────────────────────────────────────────────────────────
    private static void RgLruForwardGeneric<T>(
        T[] V, T[] R, T[] I, T[] decay, T[] outp, int batch, int seqLen, int recDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T one = ops.One, zero = ops.Zero;
        var baseDecay = new T[recDim];
        for (int c = 0; c < recDim; c++) baseDecay[c] = SigGeneric(ops, ops.Negate(decay[c]));

        // Channel-parallel with b outer; per-channel local scalar state. See RgLruForwardDouble.
        for (int b = 0; b < batch; b++)
        {
            int bb = b;
            CpuParallelSettings.ParallelForChunks(recDim, RgLruChannelGrain, (cStart, cCount) =>
            {
                int cEnd = cStart + cCount;
                for (int c = cStart; c < cEnd; c++)
                {
                    T hc = zero;
                    T bdc = baseDecay[c];
                    for (int t = 0; t < seqLen; t++)
                    {
                        int off = (bb * seqLen + t) * recDim;
                        T a = ops.Multiply(R[off + c], bdc);
                        T oneMinus = ops.Subtract(one, ops.Multiply(a, a));
                        T s = ops.GreaterThan(oneMinus, zero) ? ops.Sqrt(oneMinus) : zero;
                        hc = ops.Add(ops.Multiply(a, hc), ops.Multiply(s, ops.Multiply(I[off + c], V[off + c])));
                        outp[off + c] = hc;
                    }
                }
            });
        }
    }

    private static void RgLruBackwardGeneric<T>(
        T[] dOut, T[] V, T[] R, T[] I, T[] decay,
        T[] dV, T[] dR, T[] dI, T[] dDecay, int batch, int seqLen, int recDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T one = ops.One, zero = ops.Zero;
        T tiny = ops.FromDouble(1e-12);
        var baseDecay = new T[recDim];
        for (int c = 0; c < recDim; c++) baseDecay[c] = SigGeneric(ops, ops.Negate(decay[c]));

        var dBase = new T[recDim];
        for (int c = 0; c < recDim; c++) dBase[c] = zero;

        // Channel-parallel with b outer; per-channel local scalar state + private trajectory. See double.
        for (int b = 0; b < batch; b++)
        {
            int bb = b;
            CpuParallelSettings.ParallelForChunks(recDim, RgLruChannelGrain, (cStart, cCount) =>
            {
                var hTrajC = new T[seqLen];
                int cEnd = cStart + cCount;
                for (int c = cStart; c < cEnd; c++)
                {
                    T bd = baseDecay[c];

                    T hc = zero;
                    for (int t = 0; t < seqLen; t++)
                    {
                        int off = (bb * seqLen + t) * recDim;
                        T a = ops.Multiply(R[off + c], bd);
                        T oneMinus = ops.Subtract(one, ops.Multiply(a, a));
                        T s = ops.GreaterThan(oneMinus, zero) ? ops.Sqrt(oneMinus) : zero;
                        hc = ops.Add(ops.Multiply(a, hc), ops.Multiply(s, ops.Multiply(I[off + c], V[off + c])));
                        hTrajC[t] = hc;
                    }

                    T dHc = zero;
                    for (int t = seqLen - 1; t >= 0; t--)
                    {
                        int off = (bb * seqLen + t) * recDim;
                        T a = ops.Multiply(R[off + c], bd);
                        T oneMinus = ops.Subtract(one, ops.Multiply(a, a));
                        T s = ops.GreaterThan(oneMinus, zero) ? ops.Sqrt(oneMinus) : zero;
                        T iv = ops.Multiply(I[off + c], V[off + c]);
                        T hPrev = t > 0 ? hTrajC[t - 1] : zero;

                        T dh = ops.Add(dHc, dOut[off + c]);
                        T dG = ops.Multiply(dh, s);
                        dI[off + c] = ops.Add(dI[off + c], ops.Multiply(dG, V[off + c]));
                        dV[off + c] = ops.Add(dV[off + c], ops.Multiply(dG, I[off + c]));

                        T dS = ops.Multiply(dh, iv);
                        T dA = ops.Multiply(dh, hPrev);
                        if (ops.GreaterThan(s, tiny))
                            dA = ops.Add(dA, ops.Multiply(dS, ops.Divide(ops.Negate(a), s)));

                        dR[off + c] = ops.Add(dR[off + c], ops.Multiply(dA, bd));
                        dBase[c] = ops.Add(dBase[c], ops.Multiply(dA, R[off + c]));

                        dHc = ops.Multiply(dh, a);
                    }
                }
            });
        }

        for (int c = 0; c < recDim; c++)
            dDecay[c] = ops.Add(dDecay[c],
                ops.Multiply(dBase[c], ops.Negate(ops.Multiply(baseDecay[c], ops.Subtract(one, baseDecay[c])))));
    }

    private static void RgLruScanBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var value = inputs[0];
        var recGate = inputs[1];
        var inpGate = inputs[2];
        var decay = inputs[3];

        int batch = value.Shape[0];
        int seqLen = value.Shape[1];
        int recDim = value.Shape[2];

        var dV = new Tensor<T>(new[] { batch, seqLen, recDim });
        var dR = new Tensor<T>(new[] { batch, seqLen, recDim });
        var dI = new Tensor<T>(new[] { batch, seqLen, recDim });
        var dDecay = new Tensor<T>(ShapeOf(decay));

        if (typeof(T) == typeof(double))
        {
            RgLruBackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)value.GetDataArray()!, (double[])(object)recGate.GetDataArray()!,
                (double[])(object)inpGate.GetDataArray()!, (double[])(object)decay.GetDataArray()!,
                (double[])(object)dV.GetDataArray()!, (double[])(object)dR.GetDataArray()!,
                (double[])(object)dI.GetDataArray()!, (double[])(object)dDecay.GetDataArray()!,
                batch, seqLen, recDim);
        }
        else
        {
            RgLruBackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                value.GetDataArray()!, recGate.GetDataArray()!, inpGate.GetDataArray()!,
                decay.GetDataArray()!,
                dV.GetDataArray()!, dR.GetDataArray()!, dI.GetDataArray()!, dDecay.GetDataArray()!,
                batch, seqLen, recDim);
        }

        DifferentiableOps.AccumulateGrad(grads, value, dV, engine);
        DifferentiableOps.AccumulateGrad(grads, recGate, dR, engine);
        DifferentiableOps.AccumulateGrad(grads, inpGate, dI, engine);
        DifferentiableOps.AccumulateGrad(grads, decay, dDecay, engine);
    }
}
