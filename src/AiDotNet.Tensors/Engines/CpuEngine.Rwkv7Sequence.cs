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
    /// Fused RWKV-7 time-mixing WKV recurrence over a whole sequence in a SINGLE op
    /// (forward + custom autodiff backward), replacing the ~10 per-timestep tape micro-ops the
    /// decomposed <c>RWKV7Block.TimeMixingForward</c> loop records (issue #1464). All inputs are the
    /// already-projected, per-position gate/value streams <c>[batch, seqLen, modelDim]</c>; the
    /// kernel applies the per-position sigmoids internally so the recurrence and its adjoint never
    /// touch the tape per step.
    ///
    /// <para>Per head (modelDim split into <paramref name="numHeads"/> blocks of headDim), with the
    /// matrix-valued state S[di,vi] initialised to zero and stepped for t = 0..seqLen-1:</para>
    /// <code>
    ///   ga = sigmoid(A_t); gb = sigmoid(B_t); gr = sigmoid(R_t)        // [headDim]
    ///   S_t[di,vi] = ga[di]*S_{t-1}[di,vi] + (gb[di]*K_t[di]) * V_t[vi]
    ///   wkv_t[di]  = gr[di] * sum_vi S_t[di,vi] * K_t[vi]
    /// </code>
    /// This matches the decomposed kernel exactly (sigmoid gates, rank-1 injection, diagonal decay,
    /// gated readout), so clone-parity and gradients are preserved — it is purely a per-step
    /// dispatch-overhead reduction.
    /// </summary>
    /// <param name="rProj">Receptance projection [batch, seqLen, modelDim] (pre-sigmoid).</param>
    /// <param name="kProj">Key projection [batch, seqLen, modelDim].</param>
    /// <param name="vProj">Value projection [batch, seqLen, modelDim].</param>
    /// <param name="aProj">Decay (state-evolution a) projection [batch, seqLen, modelDim] (pre-sigmoid).</param>
    /// <param name="bProj">Injection (state-evolution b) projection [batch, seqLen, modelDim] (pre-sigmoid).</param>
    /// <param name="numHeads">Number of heads; modelDim must be divisible by it.</param>
    /// <returns>The gated WKV output [batch, seqLen, modelDim].</returns>
    public virtual Tensor<T> Rwkv7SequenceForward<T>(
        Tensor<T> rProj, Tensor<T> kProj, Tensor<T> vProj, Tensor<T> aProj, Tensor<T> bProj,
        int numHeads)
    {
        if (rProj is null) throw new ArgumentNullException(nameof(rProj));
        if (kProj is null) throw new ArgumentNullException(nameof(kProj));
        if (vProj is null) throw new ArgumentNullException(nameof(vProj));
        if (aProj is null) throw new ArgumentNullException(nameof(aProj));
        if (bProj is null) throw new ArgumentNullException(nameof(bProj));
        if (numHeads < 1) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (rProj.Rank != 3)
            throw new ArgumentException($"Rwkv7SequenceForward expects rank-3 inputs [batch, seqLen, modelDim]; got rank {rProj.Rank}.", nameof(rProj));

        int batch = rProj.Shape[0];
        int seqLen = rProj.Shape[1];
        int modelDim = rProj.Shape[2];
        if (modelDim % numHeads != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        int headDim = modelDim / numHeads;

        EnsureSameShape(rProj, kProj, nameof(kProj));
        EnsureSameShape(rProj, vProj, nameof(vProj));
        EnsureSameShape(rProj, aProj, nameof(aProj));
        EnsureSameShape(rProj, bProj, nameof(bProj));

        var output = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            Rwkv7ForwardDouble(
                (double[])(object)rProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)aProj.GetDataArray()!,
                (double[])(object)bProj.GetDataArray()!, (double[])(object)output.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            Rwkv7ForwardGeneric<T>(
                rProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!,
                aProj.GetDataArray()!, bProj.GetDataArray()!, output.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }

        // Record ONE tape node for the whole recurrence with a custom BPTT backward.
        DifferentiableOps.RecordIfActive<T>(
            "Rwkv7Sequence", output,
            new[] { rProj, kProj, vProj, aProj, bProj },
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
        double[] R, double[] K, double[] V, double[] A, double[] B, double[] outp,
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
                    // State update: S[di,vi] = ga[di]*S[di,vi] + (gb[di]*k[di])*v[vi].
                    for (int di = 0; di < headDim; di++)
                    {
                        double ga = Sig(A[baseOff + di]);
                        double gbk = Sig(B[baseOff + di]) * K[baseOff + di];
                        int srow = di * headDim;
                        for (int vi = 0; vi < headDim; vi++)
                            S[srow + vi] = ga * S[srow + vi] + gbk * V[baseOff + vi];
                    }
                    // Readout: wkv[di] = gr[di] * sum_vi S[di,vi]*k[vi].
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double sk = 0.0;
                        for (int vi = 0; vi < headDim; vi++)
                            sk += S[srow + vi] * K[baseOff + vi];
                        outp[baseOff + di] = Sig(R[baseOff + di]) * sk;
                    }
                }
            }
        }
    }

    private static void Rwkv7BackwardDouble(
        double[] dOut, double[] R, double[] K, double[] V, double[] A, double[] B,
        double[] dR, double[] dK, double[] dV, double[] dA, double[] dB,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        var Straj = new double[seqLen * hh]; // S_t (post-update) for every t, reused per (b,h)
        var S = new double[hh];
        var dS = new double[hh];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;

                // Forward recompute, saving the full state trajectory.
                Array.Clear(S, 0, hh);
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    for (int di = 0; di < headDim; di++)
                    {
                        double ga = Sig(A[baseOff + di]);
                        double gbk = Sig(B[baseOff + di]) * K[baseOff + di];
                        int srow = di * headDim;
                        for (int vi = 0; vi < headDim; vi++)
                            S[srow + vi] = ga * S[srow + vi] + gbk * V[baseOff + vi];
                    }
                    Array.Copy(S, 0, Straj, t * hh, hh);
                }

                // Backward sweep over t (dS carries the adjoint state from t+1).
                Array.Clear(dS, 0, hh);
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int stOff = t * hh;

                    // Readout backward: wkv[di] = gr[di] * sum_vi S_t[di,vi]*k[vi].
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double gr = Sig(R[baseOff + di]);
                        double dwkv = dOut[baseOff + di];
                        double sk = 0.0;
                        for (int vi = 0; vi < headDim; vi++)
                            sk += Straj[stOff + srow + vi] * K[baseOff + vi];
                        dR[baseOff + di] += dwkv * sk * gr * (1.0 - gr); // sigmoid pre-activation grad
                        double g = dwkv * gr;
                        for (int vi = 0; vi < headDim; vi++)
                        {
                            dS[srow + vi] += g * K[baseOff + vi];
                            dK[baseOff + vi] += g * Straj[stOff + srow + vi]; // readout k[vi] term
                        }
                    }

                    // Update backward: S_t[di,vi] = ga[di]*S_{t-1}[di,vi] + gb[di]*k[di]*v[vi].
                    int sprevOff = (t - 1) * hh;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        double ga = Sig(A[baseOff + di]);
                        double gb = Sig(B[baseOff + di]);
                        double kdi = K[baseOff + di];
                        double dgaAcc = 0.0, dgbAcc = 0.0, dkInj = 0.0;
                        for (int vi = 0; vi < headDim; vi++)
                        {
                            double dStv = dS[srow + vi];
                            double sprev = t > 0 ? Straj[sprevOff + srow + vi] : 0.0;
                            double vvi = V[baseOff + vi];
                            dgaAcc += dStv * sprev;
                            dgbAcc += dStv * kdi * vvi;
                            dkInj += dStv * gb * vvi;
                            dV[baseOff + vi] += dStv * gb * kdi;
                        }
                        dA[baseOff + di] += dgaAcc * ga * (1.0 - ga);
                        dB[baseOff + di] += dgbAcc * gb * (1.0 - gb);
                        dK[baseOff + di] += dkInj; // injection k[di] term (adds to readout term above)
                    }

                    // Propagate adjoint to the previous step: dS_{t-1}[di,vi] = dS_t[di,vi]*ga[di].
                    for (int di = 0; di < headDim; di++)
                    {
                        double ga = Sig(A[baseOff + di]);
                        int srow = di * headDim;
                        for (int vi = 0; vi < headDim; vi++)
                            dS[srow + vi] *= ga;
                    }
                }
            }
        }
    }

    // ── Generic-T path (correct for any numeric T; used for non-double) ──────────────────
    private static void Rwkv7ForwardGeneric<T>(
        T[] R, T[] K, T[] V, T[] A, T[] B, T[] outp,
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
                    for (int di = 0; di < headDim; di++)
                    {
                        T ga = SigGeneric(ops, A[baseOff + di]);
                        T gbk = ops.Multiply(SigGeneric(ops, B[baseOff + di]), K[baseOff + di]);
                        int srow = di * headDim;
                        for (int vi = 0; vi < headDim; vi++)
                            S[srow + vi] = ops.Add(ops.Multiply(ga, S[srow + vi]), ops.Multiply(gbk, V[baseOff + vi]));
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T sk = ops.Zero;
                        for (int vi = 0; vi < headDim; vi++)
                            sk = ops.Add(sk, ops.Multiply(S[srow + vi], K[baseOff + vi]));
                        outp[baseOff + di] = ops.Multiply(SigGeneric(ops, R[baseOff + di]), sk);
                    }
                }
            }
        }
    }

    private static void Rwkv7BackwardGeneric<T>(
        T[] dOut, T[] R, T[] K, T[] V, T[] A, T[] B,
        T[] dR, T[] dK, T[] dV, T[] dA, T[] dB,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int hh = headDim * headDim;
        var Straj = new T[seqLen * hh];
        var S = new T[hh];
        var dS = new T[hh];
        T one = ops.One;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;
                for (int i = 0; i < hh; i++) S[i] = ops.Zero;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    for (int di = 0; di < headDim; di++)
                    {
                        T ga = SigGeneric(ops, A[baseOff + di]);
                        T gbk = ops.Multiply(SigGeneric(ops, B[baseOff + di]), K[baseOff + di]);
                        int srow = di * headDim;
                        for (int vi = 0; vi < headDim; vi++)
                            S[srow + vi] = ops.Add(ops.Multiply(ga, S[srow + vi]), ops.Multiply(gbk, V[baseOff + vi]));
                    }
                    Array.Copy(S, 0, Straj, t * hh, hh);
                }

                for (int i = 0; i < hh; i++) dS[i] = ops.Zero;
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int stOff = t * hh;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T gr = SigGeneric(ops, R[baseOff + di]);
                        T dwkv = dOut[baseOff + di];
                        T sk = ops.Zero;
                        for (int vi = 0; vi < headDim; vi++)
                            sk = ops.Add(sk, ops.Multiply(Straj[stOff + srow + vi], K[baseOff + vi]));
                        T dsig = ops.Multiply(ops.Multiply(gr, ops.Subtract(one, gr)), ops.Multiply(dwkv, sk));
                        dR[baseOff + di] = ops.Add(dR[baseOff + di], dsig);
                        T g = ops.Multiply(dwkv, gr);
                        for (int vi = 0; vi < headDim; vi++)
                        {
                            dS[srow + vi] = ops.Add(dS[srow + vi], ops.Multiply(g, K[baseOff + vi]));
                            dK[baseOff + vi] = ops.Add(dK[baseOff + vi], ops.Multiply(g, Straj[stOff + srow + vi]));
                        }
                    }

                    int sprevOff = (t - 1) * hh;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        T ga = SigGeneric(ops, A[baseOff + di]);
                        T gb = SigGeneric(ops, B[baseOff + di]);
                        T kdi = K[baseOff + di];
                        T dgaAcc = ops.Zero, dgbAcc = ops.Zero, dkInj = ops.Zero;
                        for (int vi = 0; vi < headDim; vi++)
                        {
                            T dStv = dS[srow + vi];
                            T sprev = t > 0 ? Straj[sprevOff + srow + vi] : ops.Zero;
                            T vvi = V[baseOff + vi];
                            dgaAcc = ops.Add(dgaAcc, ops.Multiply(dStv, sprev));
                            dgbAcc = ops.Add(dgbAcc, ops.Multiply(dStv, ops.Multiply(kdi, vvi)));
                            dkInj = ops.Add(dkInj, ops.Multiply(dStv, ops.Multiply(gb, vvi)));
                            dV[baseOff + vi] = ops.Add(dV[baseOff + vi], ops.Multiply(dStv, ops.Multiply(gb, kdi)));
                        }
                        dA[baseOff + di] = ops.Add(dA[baseOff + di], ops.Multiply(dgaAcc, ops.Multiply(ga, ops.Subtract(one, ga))));
                        dB[baseOff + di] = ops.Add(dB[baseOff + di], ops.Multiply(dgbAcc, ops.Multiply(gb, ops.Subtract(one, gb))));
                        dK[baseOff + di] = ops.Add(dK[baseOff + di], dkInj);
                    }

                    for (int di = 0; di < headDim; di++)
                    {
                        T ga = SigGeneric(ops, A[baseOff + di]);
                        int srow = di * headDim;
                        for (int vi = 0; vi < headDim; vi++)
                            dS[srow + vi] = ops.Multiply(dS[srow + vi], ga);
                    }
                }
            }
        }
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
        var kProj = inputs[1];
        var vProj = inputs[2];
        var aProj = inputs[3];
        var bProj = inputs[4];

        int batch = rProj.Shape[0];
        int seqLen = rProj.Shape[1];
        int modelDim = rProj.Shape[2];
        int headDim = modelDim / numHeads;

        var dR = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dK = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dV = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dA = new Tensor<T>(new[] { batch, seqLen, modelDim });
        var dB = new Tensor<T>(new[] { batch, seqLen, modelDim });

        if (typeof(T) == typeof(double))
        {
            Rwkv7BackwardDouble(
                (double[])(object)gradOutput.GetDataArray()!,
                (double[])(object)rProj.GetDataArray()!, (double[])(object)kProj.GetDataArray()!,
                (double[])(object)vProj.GetDataArray()!, (double[])(object)aProj.GetDataArray()!,
                (double[])(object)bProj.GetDataArray()!,
                (double[])(object)dR.GetDataArray()!, (double[])(object)dK.GetDataArray()!,
                (double[])(object)dV.GetDataArray()!, (double[])(object)dA.GetDataArray()!,
                (double[])(object)dB.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }
        else
        {
            Rwkv7BackwardGeneric<T>(
                gradOutput.GetDataArray()!,
                rProj.GetDataArray()!, kProj.GetDataArray()!, vProj.GetDataArray()!,
                aProj.GetDataArray()!, bProj.GetDataArray()!,
                dR.GetDataArray()!, dK.GetDataArray()!, dV.GetDataArray()!,
                dA.GetDataArray()!, dB.GetDataArray()!,
                batch, seqLen, modelDim, numHeads, headDim);
        }

        DifferentiableOps.AccumulateGrad(grads, rProj, dR, engine);
        DifferentiableOps.AccumulateGrad(grads, kProj, dK, engine);
        DifferentiableOps.AccumulateGrad(grads, vProj, dV, engine);
        DifferentiableOps.AccumulateGrad(grads, aProj, dA, engine);
        DifferentiableOps.AccumulateGrad(grads, bProj, dB, engine);
    }
}
