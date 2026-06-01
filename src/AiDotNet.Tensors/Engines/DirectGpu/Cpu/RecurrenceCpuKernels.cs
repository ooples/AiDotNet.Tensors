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
}
