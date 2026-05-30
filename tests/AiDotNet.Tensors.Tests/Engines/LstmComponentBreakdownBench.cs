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
        var gate = RandF(4 * bh, rng);   // i|f|g|o blocks, 32768
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

        // 3. Gate activations (per step): sigmoid(i+f) | tanh(g) | sigmoid(o).
        double tAct = MinUs(() =>
        {
            fixed (float* g = gate)
            {
                SimdKernels.SigmoidUnsafe(g + 0 * bh, g + 0 * bh, 2 * bh);
                SimdKernels.TanhUnsafe(g + 2 * bh, g + 2 * bh, bh);
                SimdKernels.SigmoidUnsafe(g + 3 * bh, g + 3 * bh, bh);
            }
        });

        // 4. Cell + hidden update (per step): c=f*cprev+i*g; h=o*tanh(c).
        double tCell = MinUs(() =>
        {
            fixed (float* g = gate)
            fixed (float* cp = cPrev)
            fixed (float* cc = cCurr)
            fixed (float* hc = hCurr)
            {
                float* iB = g; float* fB = g + bh; float* gB = g + 2 * bh; float* oB = g + 3 * bh;
                for (int x = 0; x < bh; x++) cc[x] = fB[x] * cp[x] + iB[x] * gB[x];
                SimdKernels.TanhUnsafe(cc, hc, bh);
                for (int x = 0; x < bh; x++) hc[x] = oB[x] * hc[x];
            }
        });

        double perStep = tRec + tAct + tCell;
        double accounted = (tInput + perStep * seq) / 1000.0;
        _out.WriteLine($"LSTM component breakdown [128,32,32]->64 (min of {measured}):");
        _out.WriteLine($"  input GEMM    x1  : {tInput,7:F1} us            -> {tInput / 1000.0,5:F2} ms");
        _out.WriteLine($"  recurrent GEMM x{seq}: {tRec,7:F1} us/step -> {tRec * seq / 1000.0,5:F2} ms");
        _out.WriteLine($"  activations    x{seq}: {tAct,7:F1} us/step -> {tAct * seq / 1000.0,5:F2} ms");
        _out.WriteLine($"  cell update    x{seq}: {tCell,7:F1} us/step -> {tCell * seq / 1000.0,5:F2} ms");
        _out.WriteLine($"  accounted sum     : {accounted:F2} ms (vs ~3.3ms full; rest = de-interleave + copies + per-call overhead)");

        Assert.True(tInput > 0 && tRec > 0 && tAct > 0 && tCell > 0);
    }

    private static float[] RandF(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
