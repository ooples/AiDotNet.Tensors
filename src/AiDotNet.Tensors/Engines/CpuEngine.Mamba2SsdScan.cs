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
    /// Fused Mamba-2 SSD (State Space Duality) scan over a whole sequence in a SINGLE op
    /// (forward + custom autodiff backward), replacing the per-element detached scalar loop the
    /// decomposed <c>Mamba2Block.SSDForward</c> ran — both slow AND detached from the autodiff tape
    /// (issue ooples/AiDotNet#1464). Per head (innerDim split into numHeads blocks of headDim) with
    /// PER-HEAD scalar A = -exp(aLog[h]) and per-head scalar delta, shared B/C across heads, and state
    /// h[di,n] = 0:
    /// <code>
    ///   Abar = exp(delta[h] * A[h])
    ///   h_t[di,n] = Abar * h_{t-1}[di,n] + delta[h] * B_t[n] * x_t[di]
    ///   y_t[di]   = sum_n C_t[n] * h_t[di,n] + D[h] * x_t[di]
    /// </code>
    /// The chunked SSD algorithm in the paper is mathematically this recurrence; the fused op computes
    /// it sequentially with one differentiable tape node (BPTT adjoint), safe under a GradientTape.
    /// </summary>
    /// <param name="x">Post-conv/SiLU input [batch, seqLen, innerDim].</param>
    /// <param name="delta">Per-head timestep [batch, seqLen, numHeads] (post-softplus).</param>
    /// <param name="aLog">Per-head log decay [numHeads]; A = -exp(aLog).</param>
    /// <param name="bParam">Shared B [batch, seqLen, stateDim].</param>
    /// <param name="cParam">Shared C [batch, seqLen, stateDim].</param>
    /// <param name="dParam">Per-head skip D [numHeads].</param>
    /// <param name="numHeads">Number of heads; innerDim must be divisible by it.</param>
    /// <returns>The SSD output [batch, seqLen, innerDim].</returns>
    public virtual Tensor<T> Mamba2SsdScanForward<T>(
        Tensor<T> x, Tensor<T> delta, Tensor<T> aLog, Tensor<T> bParam, Tensor<T> cParam, Tensor<T> dParam, int numHeads)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (delta is null) throw new ArgumentNullException(nameof(delta));
        if (aLog is null) throw new ArgumentNullException(nameof(aLog));
        if (bParam is null) throw new ArgumentNullException(nameof(bParam));
        if (cParam is null) throw new ArgumentNullException(nameof(cParam));
        if (dParam is null) throw new ArgumentNullException(nameof(dParam));
        if (numHeads < 1) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (x.Rank != 3)
            throw new ArgumentException($"Mamba2SsdScanForward expects rank-3 x [batch, seqLen, innerDim]; got rank {x.Rank}.", nameof(x));

        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int innerDim = x.Shape[2];
        if (innerDim % numHeads != 0)
            throw new ArgumentException($"innerDim ({innerDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        int headDim = innerDim / numHeads;
        int stateDim = bParam.Shape[2];
        if (delta.Rank != 3 || delta.Shape[0] != batch || delta.Shape[1] != seqLen || delta.Shape[2] != numHeads)
            throw new ArgumentException($"delta must be [batch={batch}, seqLen={seqLen}, numHeads={numHeads}].", nameof(delta));
        if (aLog.Length != numHeads)
            throw new ArgumentException($"aLog length ({aLog.Length}) must equal numHeads ({numHeads}).", nameof(aLog));
        if (cParam.Rank != 3 || cParam.Shape[2] != stateDim)
            throw new ArgumentException($"cParam must be [batch, seqLen, stateDim={stateDim}].", nameof(cParam));
        if (dParam.Length != numHeads)
            throw new ArgumentException($"dParam length ({dParam.Length}) must equal numHeads ({numHeads}).", nameof(dParam));

        var output = new Tensor<T>(new[] { batch, seqLen, innerDim });

        if (typeof(T) == typeof(double))
        {
            Mamba2ForwardDouble(
                (double[])(object)x.GetDataArray()!, (double[])(object)delta.GetDataArray()!,
                (double[])(object)aLog.GetDataArray()!, (double[])(object)bParam.GetDataArray()!,
                (double[])(object)cParam.GetDataArray()!, (double[])(object)dParam.GetDataArray()!,
                (double[])(object)output.GetDataArray()!, batch, seqLen, innerDim, numHeads, headDim, stateDim);
        }
        else
        {
            Mamba2ForwardGeneric<T>(
                x.GetDataArray()!, delta.GetDataArray()!, aLog.GetDataArray()!, bParam.GetDataArray()!,
                cParam.GetDataArray()!, dParam.GetDataArray()!, output.GetDataArray()!,
                batch, seqLen, innerDim, numHeads, headDim, stateDim);
        }

        DifferentiableOps.RecordIfActive<T>(
            "Mamba2SsdScan", output,
            new[] { x, delta, aLog, bParam, cParam, dParam },
            Mamba2SsdScanBackward<T>,
            savedState: new object[] { numHeads });

        return output;
    }

    // ── Double fast path ─────────────────────────────────────────────────────────────────
    private static void Mamba2ForwardDouble(
        double[] X, double[] delta, double[] aLog, double[] B, double[] C, double[] D, double[] outp,
        int batch, int seqLen, int innerDim, int numHeads, int headDim, int sd)
    {
        var negA = new double[numHeads];
        for (int hi = 0; hi < numHeads; hi++) negA[hi] = -Math.Exp(aLog[hi]);
        var h = new double[innerDim * sd];
        for (int b = 0; b < batch; b++)
        {
            Array.Clear(h, 0, h.Length);
            for (int t = 0; t < seqLen; t++)
            {
                int btInner = (b * seqLen + t) * innerDim;
                int btState = (b * seqLen + t) * sd;
                int btHead = (b * seqLen + t) * numHeads;
                for (int hi = 0; hi < numHeads; hi++)
                {
                    double dt = delta[btHead + hi];
                    double aBar = Math.Exp(dt * negA[hi]);
                    double dv = D[hi];
                    int dimStart = hi * headDim;
                    for (int di = 0; di < headDim; di++)
                    {
                        int flatD = dimStart + di;
                        double xv = X[btInner + flatD];
                        int hsBase = flatD * sd;
                        double y = 0.0;
                        for (int n = 0; n < sd; n++)
                        {
                            double hNew = aBar * h[hsBase + n] + dt * B[btState + n] * xv;
                            h[hsBase + n] = hNew;
                            y += C[btState + n] * hNew;
                        }
                        outp[btInner + flatD] = y + dv * xv;
                    }
                }
            }
        }
    }

    private static void Mamba2BackwardDouble(
        double[] dOut, double[] X, double[] delta, double[] aLog, double[] B, double[] C, double[] D,
        double[] dX, double[] dDelta, double[] dALog, double[] dB, double[] dC, double[] dD,
        int batch, int seqLen, int innerDim, int numHeads, int headDim, int sd)
    {
        int isd = innerDim * sd;
        var negA = new double[numHeads];
        for (int hi = 0; hi < numHeads; hi++) negA[hi] = -Math.Exp(aLog[hi]);
        var hTraj = new double[seqLen * isd];
        var h = new double[isd];
        var dh = new double[isd];

        for (int b = 0; b < batch; b++)
        {
            // Forward recompute, saving the post-update state trajectory.
            Array.Clear(h, 0, isd);
            for (int t = 0; t < seqLen; t++)
            {
                int btInner = (b * seqLen + t) * innerDim;
                int btState = (b * seqLen + t) * sd;
                int btHead = (b * seqLen + t) * numHeads;
                for (int hi = 0; hi < numHeads; hi++)
                {
                    double dt = delta[btHead + hi];
                    double aBar = Math.Exp(dt * negA[hi]);
                    int dimStart = hi * headDim;
                    for (int di = 0; di < headDim; di++)
                    {
                        int flatD = dimStart + di;
                        double xv = X[btInner + flatD];
                        int hsBase = flatD * sd;
                        for (int n = 0; n < sd; n++)
                            h[hsBase + n] = aBar * h[hsBase + n] + dt * B[btState + n] * xv;
                    }
                }
                Array.Copy(h, 0, hTraj, t * isd, isd);
            }

            // Reverse sweep.
            Array.Clear(dh, 0, isd);
            for (int t = seqLen - 1; t >= 0; t--)
            {
                int btInner = (b * seqLen + t) * innerDim;
                int btState = (b * seqLen + t) * sd;
                int btHead = (b * seqLen + t) * numHeads;
                int stOff = t * isd, spOff = (t - 1) * isd;
                for (int hi = 0; hi < numHeads; hi++)
                {
                    double dt = delta[btHead + hi];
                    double a = negA[hi];
                    double aBar = Math.Exp(dt * a);
                    int dimStart = hi * headDim;
                    double dDeltaAcc = 0.0, dALogAcc = 0.0, dDAcc = 0.0;
                    for (int di = 0; di < headDim; di++)
                    {
                        int flatD = dimStart + di;
                        double xv = X[btInner + flatD];
                        int hsBase = flatD * sd;
                        double dOutVal = dOut[btInner + flatD];

                        // D skip: y += D[h]*x.
                        dX[btInner + flatD] += D[hi] * dOutVal;
                        dDAcc += xv * dOutVal;

                        for (int n = 0; n < sd; n++)
                        {
                            double hCur = hTraj[stOff + hsBase + n];
                            double hPrev = t > 0 ? hTraj[spOff + hsBase + n] : 0.0;
                            // Output: dh += C*dOut ; dC += h_t*dOut.
                            dh[hsBase + n] += C[btState + n] * dOutVal;
                            dC[btState + n] += hCur * dOutVal;
                            double dhVal = dh[hsBase + n];

                            // h_t = aBar*hPrev + dt*B*x.
                            double dAbar = dhVal * hPrev;
                            dDeltaAcc += dAbar * aBar * a;            // A-path: dAbar*Abar*A
                            dALogAcc += dAbar * aBar * (dt * a);      // aLog: dAbar*Abar*(dt*A)
                            dDeltaAcc += dhVal * B[btState + n] * xv; // B*x path
                            dB[btState + n] += dhVal * dt * xv;
                            dX[btInner + flatD] += dhVal * dt * B[btState + n];

                            // Propagate to previous step.
                            dh[hsBase + n] = dhVal * aBar;
                        }
                    }
                    dDelta[btHead + hi] += dDeltaAcc;
                    dALog[hi] += dALogAcc;
                    dD[hi] += dDAcc;
                }
            }
        }
    }

    // ── Generic-T path ───────────────────────────────────────────────────────────────────
    private static void Mamba2ForwardGeneric<T>(
        T[] X, T[] delta, T[] aLog, T[] B, T[] C, T[] D, T[] outp,
        int batch, int seqLen, int innerDim, int numHeads, int headDim, int sd)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var negA = new T[numHeads];
        for (int hi = 0; hi < numHeads; hi++) negA[hi] = ops.Negate(ops.Exp(aLog[hi]));
        var h = new T[innerDim * sd];
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < h.Length; i++) h[i] = ops.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int btInner = (b * seqLen + t) * innerDim;
                int btState = (b * seqLen + t) * sd;
                int btHead = (b * seqLen + t) * numHeads;
                for (int hi = 0; hi < numHeads; hi++)
                {
                    T dt = delta[btHead + hi];
                    T aBar = ops.Exp(ops.Multiply(dt, negA[hi]));
                    T dv = D[hi];
                    int dimStart = hi * headDim;
                    for (int di = 0; di < headDim; di++)
                    {
                        int flatD = dimStart + di;
                        T xv = X[btInner + flatD];
                        int hsBase = flatD * sd;
                        T y = ops.Zero;
                        for (int n = 0; n < sd; n++)
                        {
                            T hNew = ops.Add(ops.Multiply(aBar, h[hsBase + n]),
                                ops.Multiply(dt, ops.Multiply(B[btState + n], xv)));
                            h[hsBase + n] = hNew;
                            y = ops.Add(y, ops.Multiply(C[btState + n], hNew));
                        }
                        outp[btInner + flatD] = ops.Add(y, ops.Multiply(dv, xv));
                    }
                }
            }
        }
    }

    private static void Mamba2BackwardGeneric<T>(
        T[] dOut, T[] X, T[] delta, T[] aLog, T[] B, T[] C, T[] D,
        T[] dX, T[] dDelta, T[] dALog, T[] dB, T[] dC, T[] dD,
        int batch, int seqLen, int innerDim, int numHeads, int headDim, int sd)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int isd = innerDim * sd;
        var negA = new T[numHeads];
        for (int hi = 0; hi < numHeads; hi++) negA[hi] = ops.Negate(ops.Exp(aLog[hi]));
        var hTraj = new T[seqLen * isd];
        var h = new T[isd];
        var dh = new T[isd];

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < isd; i++) h[i] = ops.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int btInner = (b * seqLen + t) * innerDim;
                int btState = (b * seqLen + t) * sd;
                int btHead = (b * seqLen + t) * numHeads;
                for (int hi = 0; hi < numHeads; hi++)
                {
                    T dt = delta[btHead + hi];
                    T aBar = ops.Exp(ops.Multiply(dt, negA[hi]));
                    int dimStart = hi * headDim;
                    for (int di = 0; di < headDim; di++)
                    {
                        int flatD = dimStart + di;
                        T xv = X[btInner + flatD];
                        int hsBase = flatD * sd;
                        for (int n = 0; n < sd; n++)
                            h[hsBase + n] = ops.Add(ops.Multiply(aBar, h[hsBase + n]),
                                ops.Multiply(dt, ops.Multiply(B[btState + n], xv)));
                    }
                }
                Array.Copy(h, 0, hTraj, t * isd, isd);
            }

            for (int i = 0; i < isd; i++) dh[i] = ops.Zero;
            for (int t = seqLen - 1; t >= 0; t--)
            {
                int btInner = (b * seqLen + t) * innerDim;
                int btState = (b * seqLen + t) * sd;
                int btHead = (b * seqLen + t) * numHeads;
                int stOff = t * isd, spOff = (t - 1) * isd;
                for (int hi = 0; hi < numHeads; hi++)
                {
                    T dt = delta[btHead + hi];
                    T a = negA[hi];
                    T aBar = ops.Exp(ops.Multiply(dt, a));
                    int dimStart = hi * headDim;
                    T dDeltaAcc = ops.Zero, dALogAcc = ops.Zero, dDAcc = ops.Zero;
                    for (int di = 0; di < headDim; di++)
                    {
                        int flatD = dimStart + di;
                        T xv = X[btInner + flatD];
                        int hsBase = flatD * sd;
                        T dOutVal = dOut[btInner + flatD];
                        dX[btInner + flatD] = ops.Add(dX[btInner + flatD], ops.Multiply(D[hi], dOutVal));
                        dDAcc = ops.Add(dDAcc, ops.Multiply(xv, dOutVal));
                        for (int n = 0; n < sd; n++)
                        {
                            T hCur = hTraj[stOff + hsBase + n];
                            T hPrev = t > 0 ? hTraj[spOff + hsBase + n] : ops.Zero;
                            dh[hsBase + n] = ops.Add(dh[hsBase + n], ops.Multiply(C[btState + n], dOutVal));
                            dC[btState + n] = ops.Add(dC[btState + n], ops.Multiply(hCur, dOutVal));
                            T dhVal = dh[hsBase + n];
                            T dAbar = ops.Multiply(dhVal, hPrev);
                            dDeltaAcc = ops.Add(dDeltaAcc, ops.Multiply(ops.Multiply(dAbar, aBar), a));
                            dALogAcc = ops.Add(dALogAcc, ops.Multiply(ops.Multiply(dAbar, aBar), ops.Multiply(dt, a)));
                            dDeltaAcc = ops.Add(dDeltaAcc, ops.Multiply(dhVal, ops.Multiply(B[btState + n], xv)));
                            dB[btState + n] = ops.Add(dB[btState + n], ops.Multiply(dhVal, ops.Multiply(dt, xv)));
                            dX[btInner + flatD] = ops.Add(dX[btInner + flatD], ops.Multiply(dhVal, ops.Multiply(dt, B[btState + n])));
                            dh[hsBase + n] = ops.Multiply(dhVal, aBar);
                        }
                    }
                    dDelta[btHead + hi] = ops.Add(dDelta[btHead + hi], dDeltaAcc);
                    dALog[hi] = ops.Add(dALog[hi], dALogAcc);
                    dD[hi] = ops.Add(dD[hi], dDAcc);
                }
            }
        }
    }

    private static void Mamba2SsdScanBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int numHeads = (int)savedState[0];
        var x = inputs[0];
        var delta = inputs[1];
        var aLog = inputs[2];
        var bParam = inputs[3];
        var cParam = inputs[4];
        var dParam = inputs[5];

        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int innerDim = x.Shape[2];
        int headDim = innerDim / numHeads;
        int stateDim = bParam.Shape[2];

        var dX = new Tensor<T>(new[] { batch, seqLen, innerDim });
        var dDelta = new Tensor<T>(new[] { batch, seqLen, numHeads });
        var dALog = new Tensor<T>(ShapeOf(aLog));
        var dB = new Tensor<T>(new[] { batch, seqLen, stateDim });
        var dC = new Tensor<T>(new[] { batch, seqLen, stateDim });
        var dD = new Tensor<T>(ShapeOf(dParam));

        if (typeof(T) == typeof(double))
        {
            Mamba2BackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)x.GetDataArray()!, (double[])(object)delta.GetDataArray()!,
                (double[])(object)aLog.GetDataArray()!, (double[])(object)bParam.GetDataArray()!,
                (double[])(object)cParam.GetDataArray()!, (double[])(object)dParam.GetDataArray()!,
                (double[])(object)dX.GetDataArray()!, (double[])(object)dDelta.GetDataArray()!,
                (double[])(object)dALog.GetDataArray()!, (double[])(object)dB.GetDataArray()!,
                (double[])(object)dC.GetDataArray()!, (double[])(object)dD.GetDataArray()!,
                batch, seqLen, innerDim, numHeads, headDim, stateDim);
        }
        else
        {
            Mamba2BackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                x.GetDataArray()!, delta.GetDataArray()!, aLog.GetDataArray()!,
                bParam.GetDataArray()!, cParam.GetDataArray()!, dParam.GetDataArray()!,
                dX.GetDataArray()!, dDelta.GetDataArray()!, dALog.GetDataArray()!,
                dB.GetDataArray()!, dC.GetDataArray()!, dD.GetDataArray()!,
                batch, seqLen, innerDim, numHeads, headDim, stateDim);
        }

        DifferentiableOps.AccumulateGrad(grads, x, dX, engine);
        DifferentiableOps.AccumulateGrad(grads, delta, dDelta, engine);
        DifferentiableOps.AccumulateGrad(grads, aLog, dALog, engine);
        DifferentiableOps.AccumulateGrad(grads, bParam, dB, engine);
        DifferentiableOps.AccumulateGrad(grads, cParam, dC, engine);
        DifferentiableOps.AccumulateGrad(grads, dParam, dD, engine);
    }
}
