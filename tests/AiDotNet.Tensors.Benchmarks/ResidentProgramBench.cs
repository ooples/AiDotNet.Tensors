// Copyright (c) AiDotNet. All rights reserved.
// #890 phase 3: the resident program measured end to end.
//
// A declared three-stage chain - per-column affine + ReLU, row softmax, residual add - run two ways on one stream:
// as ONE persistent cooperative launch (CudaBackend.RunResidentProgram) and as the same stages launched one by one
// (RunStagesSeparately, the same device code). The PyTorch composition of the chain is timed by
// tools/parity/resident_chain_torch.py with the same protocol.
//
// Protocol (phase 0.5): interleave the two sides sample by sample, time 50 chains per region with one sync at the end,
// and take the median of the per-sample ratios (median(A)/median(B) let clock drift into the ratio); three runs, P95
// reported beside the median.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Benchmarks;

internal static class ResidentProgramBench
{
    private const int Warmup = 20;
    private const int Samples = 51;
    private const int ChainsPerSample = 50;
    private const int Runs = 3;

    internal static int Run()
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable || !backend.SupportsResidentProgram)
        {
            Console.Error.WriteLine("--resident-program needs CUDA with cooperative launch.");
            return 2;
        }

        Console.WriteLine("#890 phase 3: resident program vs the same stages launched separately (per chain, end to end)");
        Console.WriteLine("chain: affine+ReLU -> row softmax -> residual add, fp32");
        Console.WriteLine("shape        run  resident us   separate us   ratio (median of paired)   P95/med resident");
        foreach (var (rows, cols) in new[] { (64, 256), (128, 512), (256, 1024), (1024, 1024) })
        {
            var rng = new Random(890);
            float[] R(int len) => Enumerable.Range(0, len).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            using var x = backend.AllocateBuffer(R(rows * cols));
            using var scale = backend.AllocateBuffer(R(cols));
            using var bias = backend.AllocateBuffer(R(cols));
            using var residual = backend.AllocateBuffer(R(rows * cols));
            using var hidden = backend.AllocateBuffer(rows * cols);
            using var probs = backend.AllocateBuffer(rows * cols);
            using var output = backend.AllocateBuffer(rows * cols);
            using var barrier = backend.AllocateResidentBarrier();
            var stages = new[]
            {
                new ResidentStage(ResidentOp.AffineRelu, rows, cols, x, scale, bias, hidden),
                new ResidentStage(ResidentOp.RowSoftmax, rows, cols, hidden, null, null, probs),
                new ResidentStage(ResidentOp.Add, rows, cols, probs, residual, null, output),
            };

            for (int i = 0; i < Warmup; i++) { backend.RunResidentProgram(stages, barrier); backend.RunStagesSeparately(stages); }
            backend.Synchronize();

            for (int run = 1; run <= Runs; run++)
            {
                var resident = new List<double>();
                var separate = new List<double>();
                var ratios = new List<double>();
                for (int s = 0; s < Samples; s++)
                {
                    // Alternate which side goes first so neither always follows the other.
                    double r, p;
                    if ((s & 1) == 0) { r = TimeResident(backend, stages, barrier); p = TimeSeparate(backend, stages); }
                    else { p = TimeSeparate(backend, stages); r = TimeResident(backend, stages, barrier); }
                    resident.Add(r); separate.Add(p); ratios.Add(p / r);
                }
                Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                    "{0,-12} {1,3}  {2,11:F2}   {3,11:F2}   {4,24:F3}x   {5,16:F2}",
                    $"{rows}x{cols}", run, Median(resident), Median(separate), Median(ratios),
                    Percentile(resident, 0.95) / Median(resident)));
            }
        }
        return 0;
    }

    private static double TimeResident(CudaBackend backend, ResidentStage[] stages, IGpuBuffer barrier)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < ChainsPerSample; i++) backend.RunResidentProgram(stages, barrier);
        backend.Synchronize();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / ChainsPerSample;
    }

    private static double TimeSeparate(CudaBackend backend, ResidentStage[] stages)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < ChainsPerSample; i++) backend.RunStagesSeparately(stages);
        backend.Synchronize();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / ChainsPerSample;
    }

    private static double Median(List<double> v) { var o = v.OrderBy(x => x).ToList(); return o[o.Count / 2]; }

    private static double Percentile(List<double> v, double q)
    {
        var o = v.OrderBy(x => x).ToList();
        return o[Math.Min(o.Count - 1, (int)Math.Ceiling(q * o.Count) - 1)];
    }
}
