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
    /// Fused Mamba S6 selective-scan over a whole sequence in a SINGLE op (forward + custom autodiff
    /// backward), replacing the per-timestep tape micro-ops the decomposed
    /// <c>S6Scan.SequentialScanForward</c> loop records — the dominant training cost on Mamba-family
    /// models (issue ooples/AiDotNet#1464). Per the Gu &amp; Dao (2023) selective SSM, for each batch b,
    /// inner channel i, state n, with hidden state h[i,n] = 0:
    /// <code>
    ///   A = -exp(aLog[i,n])
    ///   Abar = exp(delta[b,t,i] * A)                 (ZOH discretization of A)
    ///   Bbar*x = delta[b,t,i] * B[b,t,n] * x[b,t,i]  (Euler discretization of B)
    ///   h[i,n] = Abar * h[i,n] + Bbar*x
    ///   y[b,t,i] = sum_n C[b,t,n] * h[i,n] + D[i] * x[b,t,i]
    /// </code>
    /// The "selective" delta/B/C are input-dependent; A (via aLog) and D are static per-channel. Records
    /// one tape node whose backward is the exact BPTT adjoint (matches the per-timestep result), so it is
    /// differentiable under an active <c>GradientTape</c>.
    /// </summary>
    /// <param name="x">Post-conv/SiLU input [batch, seqLen, innerDim].</param>
    /// <param name="delta">Timestep parameter (post-softplus) [batch, seqLen, innerDim].</param>
    /// <param name="aLog">Log of the (negated) A parameter [innerDim, stateDim]; A = -exp(aLog).</param>
    /// <param name="bParam">Input-dependent B [batch, seqLen, stateDim].</param>
    /// <param name="cParam">Input-dependent C [batch, seqLen, stateDim].</param>
    /// <param name="dParam">Static skip-connection D [innerDim].</param>
    /// <returns>The scan output [batch, seqLen, innerDim].</returns>
    public virtual Tensor<T> MambaSelectiveScanForward<T>(
        Tensor<T> x, Tensor<T> delta, Tensor<T> aLog, Tensor<T> bParam, Tensor<T> cParam, Tensor<T> dParam)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (delta is null) throw new ArgumentNullException(nameof(delta));
        if (aLog is null) throw new ArgumentNullException(nameof(aLog));
        if (bParam is null) throw new ArgumentNullException(nameof(bParam));
        if (cParam is null) throw new ArgumentNullException(nameof(cParam));
        if (dParam is null) throw new ArgumentNullException(nameof(dParam));
        if (x.Rank != 3)
            throw new ArgumentException($"MambaSelectiveScanForward expects rank-3 x [batch, seqLen, innerDim]; got rank {x.Rank}.", nameof(x));
        if (aLog.Rank != 2)
            throw new ArgumentException($"aLog must be rank-2 [innerDim, stateDim]; got rank {aLog.Rank}.", nameof(aLog));

        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int innerDim = x.Shape[2];
        int stateDim = aLog.Shape[1];
        if (aLog.Shape[0] != innerDim)
            throw new ArgumentException($"aLog dim0 ({aLog.Shape[0]}) must equal innerDim ({innerDim}).", nameof(aLog));
        EnsureSameShape(x, delta, nameof(delta));
        if (bParam.Rank != 3 || bParam.Shape[0] != batch || bParam.Shape[1] != seqLen || bParam.Shape[2] != stateDim)
            throw new ArgumentException($"bParam must be [batch={batch}, seqLen={seqLen}, stateDim={stateDim}].", nameof(bParam));
        if (cParam.Rank != 3 || cParam.Shape[0] != batch || cParam.Shape[1] != seqLen || cParam.Shape[2] != stateDim)
            throw new ArgumentException($"cParam must be [batch={batch}, seqLen={seqLen}, stateDim={stateDim}].", nameof(cParam));
        if (dParam.Length != innerDim)
            throw new ArgumentException($"dParam length ({dParam.Length}) must equal innerDim ({innerDim}).", nameof(dParam));

        var output = new Tensor<T>(new[] { batch, seqLen, innerDim });

        if (typeof(T) == typeof(double))
        {
            MambaScanForwardDouble(
                (double[])(object)x.GetDataArray()!, (double[])(object)delta.GetDataArray()!,
                (double[])(object)aLog.GetDataArray()!, (double[])(object)bParam.GetDataArray()!,
                (double[])(object)cParam.GetDataArray()!, (double[])(object)dParam.GetDataArray()!,
                (double[])(object)output.GetDataArray()!, batch, seqLen, innerDim, stateDim);
        }
        else
        {
            MambaScanForwardGeneric<T>(
                x.GetDataArray()!, delta.GetDataArray()!, aLog.GetDataArray()!, bParam.GetDataArray()!,
                cParam.GetDataArray()!, dParam.GetDataArray()!, output.GetDataArray()!,
                batch, seqLen, innerDim, stateDim);
        }

        DifferentiableOps.RecordIfActive<T>(
            "MambaSelectiveScan", output,
            new[] { x, delta, aLog, bParam, cParam, dParam },
            MambaSelectiveScanBackward<T>,
            savedState: null);

        return output;
    }

    // ── Double fast path ─────────────────────────────────────────────────────────────────
    private static void MambaScanForwardDouble(
        double[] X, double[] delta, double[] aLog, double[] B, double[] C, double[] D, double[] outp,
        int batch, int seqLen, int innerDim, int stateDim)
    {
        var negA = new double[innerDim * stateDim];
        for (int i = 0; i < negA.Length; i++) negA[i] = -Math.Exp(aLog[i]);

        var h = new double[innerDim * stateDim];
        for (int b = 0; b < batch; b++)
        {
            Array.Clear(h, 0, h.Length);
            for (int t = 0; t < seqLen; t++)
            {
                int baseID = (b * seqLen + t) * innerDim;
                int baseSD = (b * seqLen + t) * stateDim;
                for (int di = 0; di < innerDim; di++)
                {
                    double dt = delta[baseID + di];
                    double xv = X[baseID + di];
                    int hrow = di * stateDim;
                    double y = 0.0;
                    for (int ni = 0; ni < stateDim; ni++)
                    {
                        double aBar = Math.Exp(dt * negA[hrow + ni]);
                        double hv = aBar * h[hrow + ni] + dt * B[baseSD + ni] * xv;
                        h[hrow + ni] = hv;
                        y += C[baseSD + ni] * hv;
                    }
                    outp[baseID + di] = y + D[di] * xv;
                }
            }
        }
    }

    private static void MambaScanBackwardDouble(
        double[] dOut, double[] X, double[] delta, double[] aLog, double[] B, double[] C, double[] D,
        double[] dX, double[] dDelta, double[] dALog, double[] dB, double[] dC, double[] dD,
        int batch, int seqLen, int innerDim, int stateDim)
    {
        int isd = innerDim * stateDim;
        var negA = new double[isd];
        for (int i = 0; i < isd; i++) negA[i] = -Math.Exp(aLog[i]);

        var hTraj = new double[seqLen * isd];
        var h = new double[isd];
        var dh = new double[isd];

        for (int b = 0; b < batch; b++)
        {
            // Forward recompute, saving the post-update state trajectory.
            Array.Clear(h, 0, isd);
            for (int t = 0; t < seqLen; t++)
            {
                int baseID = (b * seqLen + t) * innerDim;
                int baseSD = (b * seqLen + t) * stateDim;
                for (int di = 0; di < innerDim; di++)
                {
                    double dt = delta[baseID + di];
                    double xv = X[baseID + di];
                    int hrow = di * stateDim;
                    for (int ni = 0; ni < stateDim; ni++)
                    {
                        double aBar = Math.Exp(dt * negA[hrow + ni]);
                        h[hrow + ni] = aBar * h[hrow + ni] + dt * B[baseSD + ni] * xv;
                    }
                }
                Array.Copy(h, 0, hTraj, t * isd, isd);
            }

            // Reverse sweep (dh carries the adjoint state from t+1).
            Array.Clear(dh, 0, isd);
            for (int t = seqLen - 1; t >= 0; t--)
            {
                int baseID = (b * seqLen + t) * innerDim;
                int baseSD = (b * seqLen + t) * stateDim;
                int stOff = t * isd;
                int sprevOff = (t - 1) * isd;
                for (int di = 0; di < innerDim; di++)
                {
                    double dt = delta[baseID + di];
                    double xv = X[baseID + di];
                    double dOutVal = dOut[baseID + di];
                    int hrow = di * stateDim;

                    // D skip connection.
                    dX[baseID + di] += D[di] * dOutVal;
                    dD[di] += xv * dOutVal;

                    double dDeltaAcc = 0.0;
                    for (int ni = 0; ni < stateDim; ni++)
                    {
                        double a = negA[hrow + ni];
                        double dtA = dt * a;
                        double aBar = Math.Exp(dtA);
                        double hCur = hTraj[stOff + hrow + ni];
                        double hPrev = t > 0 ? hTraj[sprevOff + hrow + ni] : 0.0;
                        double cVal = C[baseSD + ni];
                        double bVal = B[baseSD + ni];

                        // Output gradient into the state adjoint; dC from readout.
                        dh[hrow + ni] += cVal * dOutVal;
                        double dhVal = dh[hrow + ni];
                        dC[baseSD + ni] += hCur * dOutVal;

                        // d(Abar) = dh * h_prev  ; chain to delta (via Abar*A) and aLog (via Abar*dt*A).
                        double dAbar = dhVal * hPrev;
                        dDeltaAcc += dAbar * aBar * a;          // A_bar path: dAbar*Abar*A
                        dDeltaAcc += dhVal * bVal * xv;         // B*x path: dh*B*x
                        dB[baseSD + ni] += dhVal * dt * xv;     // dB = dh*delta*x
                        dX[baseID + di] += dhVal * dt * bVal;   // dx (state path) = dh*delta*B
                        dALog[hrow + ni] += dAbar * aBar * dtA; // aLog: dAbar*Abar*(delta*A)

                        // Propagate adjoint to previous step: dh_{t-1} = Abar * dh_t.
                        dh[hrow + ni] = aBar * dhVal;
                    }
                    dDelta[baseID + di] = dDeltaAcc;
                }
            }
        }
    }

    // ── Generic-T path ───────────────────────────────────────────────────────────────────
    private static void MambaScanForwardGeneric<T>(
        T[] X, T[] delta, T[] aLog, T[] B, T[] C, T[] D, T[] outp,
        int batch, int seqLen, int innerDim, int stateDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int isd = innerDim * stateDim;
        var negA = new T[isd];
        for (int i = 0; i < isd; i++) negA[i] = ops.Negate(ops.Exp(aLog[i]));

        var h = new T[isd];
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < isd; i++) h[i] = ops.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int baseID = (b * seqLen + t) * innerDim;
                int baseSD = (b * seqLen + t) * stateDim;
                for (int di = 0; di < innerDim; di++)
                {
                    T dt = delta[baseID + di];
                    T xv = X[baseID + di];
                    int hrow = di * stateDim;
                    T y = ops.Zero;
                    for (int ni = 0; ni < stateDim; ni++)
                    {
                        T aBar = ops.Exp(ops.Multiply(dt, negA[hrow + ni]));
                        T hv = ops.Add(ops.Multiply(aBar, h[hrow + ni]),
                            ops.Multiply(ops.Multiply(dt, B[baseSD + ni]), xv));
                        h[hrow + ni] = hv;
                        y = ops.Add(y, ops.Multiply(C[baseSD + ni], hv));
                    }
                    outp[baseID + di] = ops.Add(y, ops.Multiply(D[di], xv));
                }
            }
        }
    }

    private static void MambaScanBackwardGeneric<T>(
        T[] dOut, T[] X, T[] delta, T[] aLog, T[] B, T[] C, T[] D,
        T[] dX, T[] dDelta, T[] dALog, T[] dB, T[] dC, T[] dD,
        int batch, int seqLen, int innerDim, int stateDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int isd = innerDim * stateDim;
        var negA = new T[isd];
        for (int i = 0; i < isd; i++) negA[i] = ops.Negate(ops.Exp(aLog[i]));

        var hTraj = new T[seqLen * isd];
        var h = new T[isd];
        var dh = new T[isd];

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < isd; i++) h[i] = ops.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int baseID = (b * seqLen + t) * innerDim;
                int baseSD = (b * seqLen + t) * stateDim;
                for (int di = 0; di < innerDim; di++)
                {
                    T dt = delta[baseID + di];
                    T xv = X[baseID + di];
                    int hrow = di * stateDim;
                    for (int ni = 0; ni < stateDim; ni++)
                    {
                        T aBar = ops.Exp(ops.Multiply(dt, negA[hrow + ni]));
                        h[hrow + ni] = ops.Add(ops.Multiply(aBar, h[hrow + ni]),
                            ops.Multiply(ops.Multiply(dt, B[baseSD + ni]), xv));
                    }
                }
                Array.Copy(h, 0, hTraj, t * isd, isd);
            }

            for (int i = 0; i < isd; i++) dh[i] = ops.Zero;
            for (int t = seqLen - 1; t >= 0; t--)
            {
                int baseID = (b * seqLen + t) * innerDim;
                int baseSD = (b * seqLen + t) * stateDim;
                int stOff = t * isd;
                int sprevOff = (t - 1) * isd;
                for (int di = 0; di < innerDim; di++)
                {
                    T dt = delta[baseID + di];
                    T xv = X[baseID + di];
                    T dOutVal = dOut[baseID + di];
                    int hrow = di * stateDim;

                    dX[baseID + di] = ops.Add(dX[baseID + di], ops.Multiply(D[di], dOutVal));
                    dD[di] = ops.Add(dD[di], ops.Multiply(xv, dOutVal));

                    T dDeltaAcc = ops.Zero;
                    for (int ni = 0; ni < stateDim; ni++)
                    {
                        T a = negA[hrow + ni];
                        T dtA = ops.Multiply(dt, a);
                        T aBar = ops.Exp(dtA);
                        T hCur = hTraj[stOff + hrow + ni];
                        T hPrev = t > 0 ? hTraj[sprevOff + hrow + ni] : ops.Zero;
                        T cVal = C[baseSD + ni];
                        T bVal = B[baseSD + ni];

                        dh[hrow + ni] = ops.Add(dh[hrow + ni], ops.Multiply(cVal, dOutVal));
                        T dhVal = dh[hrow + ni];
                        dC[baseSD + ni] = ops.Add(dC[baseSD + ni], ops.Multiply(hCur, dOutVal));

                        T dAbar = ops.Multiply(dhVal, hPrev);
                        dDeltaAcc = ops.Add(dDeltaAcc, ops.Multiply(ops.Multiply(dAbar, aBar), a));
                        dDeltaAcc = ops.Add(dDeltaAcc, ops.Multiply(dhVal, ops.Multiply(bVal, xv)));
                        dB[baseSD + ni] = ops.Add(dB[baseSD + ni], ops.Multiply(dhVal, ops.Multiply(dt, xv)));
                        dX[baseID + di] = ops.Add(dX[baseID + di], ops.Multiply(dhVal, ops.Multiply(dt, bVal)));
                        dALog[hrow + ni] = ops.Add(dALog[hrow + ni], ops.Multiply(ops.Multiply(dAbar, aBar), dtA));

                        dh[hrow + ni] = ops.Multiply(aBar, dhVal);
                    }
                    dDelta[baseID + di] = dDeltaAcc;
                }
            }
        }
    }

    private static void MambaSelectiveScanBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var x = inputs[0];
        var delta = inputs[1];
        var aLog = inputs[2];
        var bParam = inputs[3];
        var cParam = inputs[4];
        var dParam = inputs[5];

        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int innerDim = x.Shape[2];
        int stateDim = aLog.Shape[1];

        var dX = new Tensor<T>(new[] { batch, seqLen, innerDim });
        var dDelta = new Tensor<T>(new[] { batch, seqLen, innerDim });
        var dALog = new Tensor<T>(new[] { innerDim, stateDim });
        var dB = new Tensor<T>(new[] { batch, seqLen, stateDim });
        var dC = new Tensor<T>(new[] { batch, seqLen, stateDim });
        var dD = new Tensor<T>(ShapeOf(dParam));

        if (typeof(T) == typeof(double))
        {
            MambaScanBackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)x.GetDataArray()!, (double[])(object)delta.GetDataArray()!,
                (double[])(object)aLog.GetDataArray()!, (double[])(object)bParam.GetDataArray()!,
                (double[])(object)cParam.GetDataArray()!, (double[])(object)dParam.GetDataArray()!,
                (double[])(object)dX.GetDataArray()!, (double[])(object)dDelta.GetDataArray()!,
                (double[])(object)dALog.GetDataArray()!, (double[])(object)dB.GetDataArray()!,
                (double[])(object)dC.GetDataArray()!, (double[])(object)dD.GetDataArray()!,
                batch, seqLen, innerDim, stateDim);
        }
        else
        {
            MambaScanBackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                x.GetDataArray()!, delta.GetDataArray()!, aLog.GetDataArray()!,
                bParam.GetDataArray()!, cParam.GetDataArray()!, dParam.GetDataArray()!,
                dX.GetDataArray()!, dDelta.GetDataArray()!, dALog.GetDataArray()!,
                dB.GetDataArray()!, dC.GetDataArray()!, dD.GetDataArray()!,
                batch, seqLen, innerDim, stateDim);
        }

        DifferentiableOps.AccumulateGrad(grads, x, dX, engine);
        DifferentiableOps.AccumulateGrad(grads, delta, dDelta, engine);
        DifferentiableOps.AccumulateGrad(grads, aLog, dALog, engine);
        DifferentiableOps.AccumulateGrad(grads, bParam, dB, engine);
        DifferentiableOps.AccumulateGrad(grads, cParam, dC, engine);
        DifferentiableOps.AccumulateGrad(grads, dParam, dD, engine);
    }
}
