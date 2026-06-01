// Copyright (c) AiDotNet. All rights reserved.
// Shared host (CPU) reference implementations of the fused recurrence scans (#1464),
// used as the correct fallback by GPU backends that do not yet ship a native compute
// shader for these ops (Metal / Vulkan / WebGPU follow the same CPU-fallback precedent
// their LstmForwardSequence already uses). The math mirrors the CpuEngine.*Scan kernels
// exactly so GPU-vs-CPU parity holds bit-for-bit on these backends.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Cpu;

internal static class RecurrenceCpuKernels
{
    /// <summary>
    /// Gated Linear Attention forward. q/k/v: [batch, seqLen, modelDim];
    /// gate: [batch, seqLen, numHeads]; output: [batch, seqLen, modelDim].
    ///   S_t[di,ki] = g_t * S_{t-1}[di,ki] + V_t[di] * K_t[ki]
    ///   O_t[di]    = sum_ki S_t[di,ki] * Q_t[ki]
    /// </summary>
    public static void GlaForward(
        float[] q, float[] k, float[] v, float[] gate, float[] output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        var s = new float[hh];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                Array.Clear(s, 0, hh);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    float g = gate[(b * seqLen + t) * numHeads + h];
                    for (int di = 0; di < headDim; di++)
                    {
                        float vd = v[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            s[srow + ki] = g * s[srow + ki] + vd * k[baseOff + ki];
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float o = 0.0f;
                        for (int ki = 0; ki < headDim; ki++)
                            o += s[srow + ki] * q[baseOff + ki];
                        output[baseOff + di] = o;
                    }
                }
            }
    }

    /// <summary>
    /// Gated Linear Attention BPTT backward. dQ/dK/dV: [batch, seqLen, modelDim];
    /// dG: [batch, seqLen, numHeads]. All grad outputs are overwritten (not accumulated).
    /// </summary>
    public static void GlaBackward(
        float[] dOut, float[] q, float[] k, float[] v, float[] gate,
        float[] dQ, float[] dK, float[] dV, float[] dG,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        Array.Clear(dQ, 0, dQ.Length);
        Array.Clear(dK, 0, dK.Length);
        Array.Clear(dV, 0, dV.Length);
        Array.Clear(dG, 0, dG.Length);

        int hh = headDim * headDim;
        var straj = new float[seqLen * hh];
        var s = new float[hh];
        var dS = new float[hh];

        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;

                Array.Clear(s, 0, hh);
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    float g = gate[(b * seqLen + t) * numHeads + h];
                    for (int di = 0; di < headDim; di++)
                    {
                        float vd = v[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            s[srow + ki] = g * s[srow + ki] + vd * k[baseOff + ki];
                    }
                    Array.Copy(s, 0, straj, t * hh, hh);
                }

                Array.Clear(dS, 0, hh);
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h;
                    int stOff = t * hh;
                    int sprevOff = (t - 1) * hh;
                    float g = gate[gOff];

                    for (int di = 0; di < headDim; di++)
                    {
                        float dOutVal = dOut[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            float sval = straj[stOff + srow + ki];
                            dQ[baseOff + ki] += dOutVal * sval;
                            dS[srow + ki] += dOutVal * q[baseOff + ki];
                        }
                    }

                    float dg = 0.0f;
                    for (int di = 0; di < headDim; di++)
                    {
                        float vd = v[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            float dStv = dS[srow + ki];
                            float sprev = t > 0 ? straj[sprevOff + srow + ki] : 0.0f;
                            dg += dStv * sprev;
                            dK[baseOff + ki] += dStv * vd;
                            dV[baseOff + di] += dStv * k[baseOff + ki];
                        }
                    }
                    dG[gOff] += dg;

                    for (int i = 0; i < hh; i++) dS[i] *= g;
                }
            }
    }

    /// <summary>
    /// Gated DeltaNet (delta-rule) forward. q/k/v: [batch, seqLen, modelDim];
    /// alpha (forget) / beta (write): [batch, seqLen, numHeads]; output: [batch, seqLen, modelDim].
    ///   sK[di]     = sum_ki S[di,ki] * (kappa*K[ki])
    ///   S[di,ki]  := alpha * S[di,ki] + beta*(V[di]-sK[di]) * (kappa*K[ki])
    ///   O[di]      = sum_ki S[di,ki] * Q[ki]
    /// </summary>
    public static void GatedDeltaNetForward(
        float[] q, float[] k, float[] v, float[] alpha, float[] beta, float[] output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        float kappa = 1.0f / MathF.Sqrt(headDim);
        var s = new float[hh];
        var sK = new float[headDim];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                Array.Clear(s, 0, hh);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gIdx = (b * seqLen + t) * numHeads + h;
                    float a = alpha[gIdx], bet = beta[gIdx];
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float acc = 0.0f;
                        for (int ki = 0; ki < headDim; ki++) acc += s[srow + ki] * (k[baseOff + ki] * kappa);
                        sK[di] = acc;
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float bd = bet * (v[baseOff + di] - sK[di]);
                        for (int ki = 0; ki < headDim; ki++)
                            s[srow + ki] = a * s[srow + ki] + bd * (k[baseOff + ki] * kappa);
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float o = 0.0f;
                        for (int ki = 0; ki < headDim; ki++) o += s[srow + ki] * q[baseOff + ki];
                        output[baseOff + di] = o;
                    }
                }
            }
    }

    private const float XLstmIGateClamp = 4.85e8f;

    /// <summary>
    /// xLSTM (mLSTM) matrix-memory forward. q/k/v: [batch, seqLen, modelDim];
    /// i/f/o gates: [batch, seqLen, numHeads]; output: [batch, seqLen, modelDim].
    /// </summary>
    public static void XLstmForward(
        float[] q, float[] k, float[] v, float[] iGate, float[] fGate, float[] oGate, float[] output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int hh = headDim * headDim;
        float kappa = 1.0f / MathF.Sqrt(headDim);
        var c = new float[hh];
        var n = new float[headDim];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                Array.Clear(c, 0, hh);
                Array.Clear(n, 0, headDim);
                int hOff = h * headDim;
                for (int t = 0; t < seqLen; t++)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h;
                    float iv = iGate[gOff]; if (iv > XLstmIGateClamp) iv = XLstmIGateClamp;
                    float f = fGate[gOff], o = oGate[gOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        n[di] = f * n[di] + iv * (k[baseOff + di] * kappa);
                        float vv = v[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            c[srow + ki] = f * c[srow + ki] + iv * vv * (k[baseOff + ki] * kappa);
                    }
                    float nq = 0.0f;
                    for (int j = 0; j < headDim; j++) nq += n[j] * q[baseOff + j];
                    float nf = MathF.Abs(nq); if (nf < 1.0f) nf = 1.0f;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float num = 0.0f;
                        for (int ki = 0; ki < headDim; ki++) num += c[srow + ki] * q[baseOff + ki];
                        output[baseOff + di] = o * num / nf;
                    }
                }
            }
    }

    /// <summary>
    /// xLSTM BPTT backward. dQ/dK/dV: [batch, seqLen, modelDim];
    /// dI/dF/dO: [batch, seqLen, numHeads]. All grad outputs are overwritten.
    /// </summary>
    public static void XLstmBackward(
        float[] dOut, float[] q, float[] k, float[] v, float[] iGate, float[] fGate, float[] oGate,
        float[] dQ, float[] dK, float[] dV, float[] dI, float[] dF, float[] dO,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        Array.Clear(dQ, 0, dQ.Length); Array.Clear(dK, 0, dK.Length); Array.Clear(dV, 0, dV.Length);
        Array.Clear(dI, 0, dI.Length); Array.Clear(dF, 0, dF.Length); Array.Clear(dO, 0, dO.Length);

        int hh = headDim * headDim;
        float kappa = 1.0f / MathF.Sqrt(headDim);
        var ctraj = new float[seqLen * hh];
        var ntraj = new float[seqLen * headDim];
        var c = new float[hh];
        var n = new float[headDim];
        var dC = new float[hh];
        var dN = new float[headDim];
        var dKS = new float[headDim];
        var dCp = new float[hh];
        var dNp = new float[headDim];

        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;

                Array.Clear(c, 0, hh); Array.Clear(n, 0, headDim);
                for (int t = 0; t < seqLen; t++)
                {
                    Array.Copy(c, 0, ctraj, t * hh, hh);
                    Array.Copy(n, 0, ntraj, t * headDim, headDim);
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h;
                    float iv = iGate[gOff]; if (iv > XLstmIGateClamp) iv = XLstmIGateClamp;
                    float f = fGate[gOff];
                    for (int di = 0; di < headDim; di++)
                    {
                        n[di] = f * n[di] + iv * (k[baseOff + di] * kappa);
                        float vv = v[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            c[srow + ki] = f * c[srow + ki] + iv * vv * (k[baseOff + ki] * kappa);
                    }
                }

                Array.Clear(dC, 0, hh); Array.Clear(dN, 0, headDim);
                for (int t = seqLen - 1; t >= 0; t--)
                {
                    int baseOff = (b * seqLen + t) * modelDim + hOff;
                    int gOff = (b * seqLen + t) * numHeads + h;
                    float iv = iGate[gOff]; bool iClamped = iv > XLstmIGateClamp; if (iClamped) iv = XLstmIGateClamp;
                    float f = fGate[gOff], o = oGate[gOff];
                    int cpOff = t * hh, npOff = t * headDim;

                    float nq = 0.0f;
                    for (int di = 0; di < headDim; di++)
                    {
                        float nCur = f * ntraj[npOff + di] + iv * (k[baseOff + di] * kappa);
                        n[di] = nCur;
                        nq += nCur * q[baseOff + di];
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        float vv = v[baseOff + di];
                        int srow = di * headDim;
                        for (int ki = 0; ki < headDim; ki++)
                            c[srow + ki] = f * ctraj[cpOff + srow + ki] + iv * vv * (k[baseOff + ki] * kappa);
                    }
                    float absnq = MathF.Abs(nq);
                    float nf = absnq < 1.0f ? 1.0f : absnq;
                    float dNfSign = absnq > 1.0f ? (nq > 0 ? 1.0f : -1.0f) : 0.0f;

                    float dO_acc = 0.0f, dnf = 0.0f;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float dh = dOut[baseOff + di];
                        float num = 0.0f;
                        for (int ki = 0; ki < headDim; ki++) num += c[srow + ki] * q[baseOff + ki];
                        dO_acc += dh * num / nf;
                        float dNum = dh * o / nf;
                        dnf += dh * o * num * (-1.0f / (nf * nf));
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            dC[srow + ki] += dNum * q[baseOff + ki];
                            dQ[baseOff + ki] += dNum * c[srow + ki];
                        }
                    }
                    dO[gOff] += dO_acc;

                    float dnq = dnf * dNfSign;
                    for (int j = 0; j < headDim; j++)
                    {
                        dN[j] += dnq * q[baseOff + j];
                        dQ[baseOff + j] += dnq * n[j];
                    }

                    Array.Clear(dKS, 0, headDim);
                    float dF_acc = 0.0f, dI_acc = 0.0f;
                    for (int di = 0; di < headDim; di++)
                    {
                        int srow = di * headDim;
                        float vv = v[baseOff + di];
                        float dVacc = 0.0f;
                        for (int ki = 0; ki < headDim; ki++)
                        {
                            float dCt = dC[srow + ki];
                            float kSki = k[baseOff + ki] * kappa;
                            dF_acc += dCt * ctraj[cpOff + srow + ki];
                            dCp[srow + ki] = dCt * f;
                            dI_acc += dCt * vv * kSki;
                            dVacc += dCt * iv * kSki;
                            dKS[ki] += dCt * iv * vv;
                        }
                        dV[baseOff + di] += dVacc;
                    }
                    for (int di = 0; di < headDim; di++)
                    {
                        float dNt = dN[di];
                        dF_acc += dNt * ntraj[npOff + di];
                        dNp[di] = dNt * f;
                        dI_acc += dNt * (k[baseOff + di] * kappa);
                        dKS[di] += dNt * iv;
                    }
                    for (int j = 0; j < headDim; j++) dK[baseOff + j] += dKS[j] * kappa;

                    dI[gOff] += iClamped ? 0.0f : dI_acc;
                    dF[gOff] += dF_acc;

                    Array.Copy(dCp, 0, dC, 0, hh);
                    Array.Copy(dNp, 0, dN, 0, headDim);
                }
            }
    }
}
