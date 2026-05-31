// Copyright (c) AiDotNet. All rights reserved.
// #477: per-component breakdown of the float LstmSequenceForward at the AIsEval
// workload, each piece timed in isolation at its real shape with MIN-of-many, so
// the remaining gap to torch (~2.78ms) is attacked where the cycles actually are
// rather than by assumption. Category=Performance => excluded from the normal run.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

[Trait("Category", "Performance")]
public class LstmComponentBreakdownBench
{
    private readonly ITestOutputHelper _out;
    public LstmComponentBreakdownBench(ITestOutputHelper output) => _out = output;

    [Fact]
    public unsafe void LstmComponents_MinTiming()
    {
        const int batch = 128, seq = 32, inF = 32, hidden = 64;
        int gateRows = 4 * hidden;       // 256
        int bh = batch * hidden;         // 8192
        int totalRows = batch * seq;     // 4096
        const int warmup = 200, measured = 2000;

        var rng = new Random(7);
        var input = RandF(totalRows * inF, rng);
        var wIh = RandF(gateRows * inF, rng);
        var wx = new float[totalRows * gateRows];
        var hPrev = RandF(batch * hidden, rng);
        var wHhT = RandF(hidden * gateRows, rng);
        var hh = new float[batch * gateRows];
        var cPrev = RandF(bh, rng);
        var cCurr = new float[bh];
        var hCurr = new float[bh];

        double MinUs(Action call)
        {
            for (int i = 0; i < warmup; i++) call();
            double min = double.MaxValue;
            for (int i = 0; i < measured; i++)
            {
                var sw = Stopwatch.StartNew();
                call();
                sw.Stop();
                double us = sw.Elapsed.TotalMilliseconds * 1000.0;
                if (us < min) min = us;
            }
            return min;
        }

        // 1. Input GEMM: Wx = input @ wIh^T, [4096,32] x [32,256] (transB).
        double tInput = MinUs(() => SimdGemm.Sgemm(input, inF, false, wIh, inF, true, wx, totalRows, inF, gateRows));

        // 2. Recurrent GEMM (per step): h_prev @ wHhT, [128,64] x [64,256].
        double tRec = MinUs(() => SimdGemm.SgemmSequential(hPrev, wHhT, hh, batch, hidden, gateRows));

        // 3. Fused elementwise cell (per step): the #477 fast path no longer
        // de-interleaves into a gate buffer or runs separate activation/cell passes.
        // It reads the four gate pre-activations straight from the NATURAL
        // [b][gateRows] Wx + hh layout, applies sigmoid/tanh, and writes
        // c = f·c_prev + i·g and h = o·tanh(c) in ONE pass. Timing that fused pass
        // (here at t = 0) is what actually catches regressions in the new kernel —
        // the old split sigmoid/tanh + cell-update measurement is retired.
        double tFusedCell = MinUs(() =>
        {
            fixed (float* wxP = wx)
            fixed (float* hhP = hh)
            fixed (float* cp = cPrev)
            fixed (float* cc = cCurr)
            fixed (float* hc = hCurr)
            {
                const int t = 0;
                for (int b = 0; b < batch; b++)
                {
                    int wxRow = (b * seq + t) * gateRows;
                    int hhRow = b * gateRows;
                    int cRow = b * hidden;
                    for (int h = 0; h < hidden; h++)
                    {
                        float ig = Sigmoid(wxP[wxRow + 0 * hidden + h] + hhP[hhRow + 0 * hidden + h]);
                        float fg = Sigmoid(wxP[wxRow + 1 * hidden + h] + hhP[hhRow + 1 * hidden + h]);
                        float gg = MathF.Tanh(wxP[wxRow + 2 * hidden + h] + hhP[hhRow + 2 * hidden + h]);
                        float og = Sigmoid(wxP[wxRow + 3 * hidden + h] + hhP[hhRow + 3 * hidden + h]);
                        float c = fg * cp[cRow + h] + ig * gg;
                        cc[cRow + h] = c;
                        hc[cRow + h] = og * MathF.Tanh(c);
                    }
                }
            }
        });

        double perStep = tRec + tFusedCell;
        double accounted = (tInput + perStep * seq) / 1000.0;
        _out.WriteLine($"LSTM component breakdown [128,32,32]->64 (min of {measured}):");
        _out.WriteLine($"  input GEMM    x1  : {tInput,7:F1} us            -> {tInput / 1000.0,5:F2} ms");
        _out.WriteLine($"  recurrent GEMM x{seq}: {tRec,7:F1} us/step -> {tRec * seq / 1000.0,5:F2} ms");
        _out.WriteLine($"  fused cell     x{seq}: {tFusedCell,7:F1} us/step -> {tFusedCell * seq / 1000.0,5:F2} ms");
        _out.WriteLine($"  accounted sum     : {accounted:F2} ms (rest = per-call overhead + the wHhT transpose + state copies)");

        Assert.True(tInput > 0 && tRec > 0 && tFusedCell > 0);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float[] RandF(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
