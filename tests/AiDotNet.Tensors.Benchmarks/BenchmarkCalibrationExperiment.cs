// Copyright (c) AiDotNet. All rights reserved.
// Phase 0.5 — calibrate the benchmark harness itself.
//
// Every performance claim downstream depends on the harness being able to resolve a
// 1.10x difference. Today it cannot: five runs of the same convolution comparison
// spanned 0.62x-2.71x, and two kernels with BIT-IDENTICAL output measured 1.57x
// apart purely from sequential A-then-B ordering.
//
// A benchmark is an instrument, so it needs a calibration standard. Two tests, both
// on a device-filling shape:
//
//   NULL TEST      the same kernel against itself. Ground truth is exactly 1.000x.
//                  Any deviation is the harness's own noise floor, and it bounds the
//                  smallest difference that can honestly be claimed.
//
//   KNOWN-RATIO    the same kernel at C=64 vs C=70. The work ratio is exactly
//                  70/64 = 1.09375x, close to the 1.10x gate we must resolve. If the
//                  harness reports that within tolerance it can resolve the gate.
//
// Using the generated kernel for every variant means kernel-implementation
// differences cannot confound the measurement of the instrument.

using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class BenchmarkCalibrationExperiment
{
    // Device-filling: 32*64*56*56 = 6,422,528 threads = 25,088 blocks of 256 on a
    // 68-SM device. The previous shapes were 4 blocks, i.e. ~6% of one GPU.
    private const int N = 32, H = 56, W = 56;
    private const int CBase = 64;
    private const int CMore = 70;              // 70/64 = 1.09375x the work exactly
    private const double KnownRatio = (double)CMore / CBase;

    private const int Warmup = 30;
    private const int Samples = 101;
    private const int Runs = 3;

    /// <summary>
    /// Launches per timed region. Timing ONE launch with a CPU stopwatch around a
    /// synchronize measures launch API + sync latency + OS scheduler jitter, and on
    /// the first calibration that produced a P95/median of 5.5 -- an instrument far
    /// too noisy to resolve the 1.10x gate. Batching amortises the sync over many
    /// launches so the measured quantity is dominated by actual kernel execution.
    /// </summary>
    private const int LaunchesPerSample = 50;

    private readonly record struct Dist(double Median, double P95, double Mean)
    {
        public double Tail => Median > 0 ? P95 / Median : double.NaN;
    }

    internal static void Run()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("bench-calibration-start");
        GpuBenchmarkEnvironment.PrintSnapshot("bench-calibration-start");

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        {
            Console.WriteLine("Calibration requires the experimental SM86 device.");
            return;
        }

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var a = Variant.Create(runtime, CBase);
            using var b = Variant.Create(runtime, CMore);

            Console.WriteLine();
            Console.WriteLine("Phase 0.5 harness calibration");
            Console.WriteLine("  shape A          N" + N + "/C" + CBase + "/H" + H + "/W" + W +
                              "  = " + a.Threads.ToString("N0", CultureInfo.InvariantCulture) +
                              " threads, " + a.Blocks.ToString("N0", CultureInfo.InvariantCulture) + " blocks");
            Console.WriteLine("  shape B          N" + N + "/C" + CMore + "/H" + H + "/W" + W +
                              "  = " + b.Threads.ToString("N0", CultureInfo.InvariantCulture) +
                              " threads, " + b.Blocks.ToString("N0", CultureInfo.InvariantCulture) + " blocks");
            Console.WriteLine("  known work ratio B/A = " + KnownRatio.ToString("F5", CultureInfo.InvariantCulture));
            Console.WriteLine("  protocol         interleaved sample-by-sample, " + Warmup +
                              " warmups, " + Samples + " samples of " + LaunchesPerSample +
                              " launches, " + Runs + " runs");
            Console.WriteLine();

            var nullRatios = new double[Runs];
            var workRatios = new double[Runs];
            double worstTail = 0;

            Console.WriteLine("run   null A/A ratio    work B/A ratio    P95/median (worst)");
            for (int run = 0; run < Runs; run++)
            {
                (Dist n1, Dist n2, double nullRatio) = Interleaved(runtime.Synchronize, a.Launch, a.Launch);
                (Dist wa, Dist wb, double workRatio) = Interleaved(runtime.Synchronize, a.Launch, b.Launch);

                nullRatios[run] = nullRatio;
                workRatios[run] = workRatio;
                double tail = Math.Max(Math.Max(n1.Tail, n2.Tail), Math.Max(wa.Tail, wb.Tail));
                worstTail = Math.Max(worstTail, tail);

                Console.WriteLine("  " + (run + 1).ToString(CultureInfo.InvariantCulture) +
                    nullRatios[run].ToString("F4", CultureInfo.InvariantCulture).PadLeft(17) +
                    workRatios[run].ToString("F4", CultureInfo.InvariantCulture).PadLeft(18) +
                    tail.ToString("F2", CultureInfo.InvariantCulture).PadLeft(22));
            }

            double nullWorst = nullRatios.Select(r => Math.Abs(r - 1.0)).Max();
            double workErr = workRatios.Select(r => Math.Abs(r - KnownRatio) / KnownRatio).Max();

            Console.WriteLine();
            Console.WriteLine("  null-test worst deviation from 1.000x : " +
                (nullWorst * 100).ToString("F2", CultureInfo.InvariantCulture) + "%   (this is the noise floor)");
            Console.WriteLine("  known-ratio worst error vs " + KnownRatio.ToString("F5", CultureInfo.InvariantCulture) +
                "x  : " + (workErr * 100).ToString("F2", CultureInfo.InvariantCulture) + "%");
            Console.WriteLine("  worst P95/median                     : " +
                worstTail.ToString("F2", CultureInfo.InvariantCulture));
            Console.WriteLine();

            // Gate: the instrument must not invent a difference where there is none,
            // must land on a known difference, and must not have a runaway tail.
            bool nullOk = nullWorst <= 0.02;
            bool workOk = workErr <= 0.03;
            bool tailOk = worstTail < 2.0;
            Console.WriteLine("  GATE null   <= 2%   : " + (nullOk ? "PASS" : "FAIL"));
            Console.WriteLine("  GATE ratio  <= 3%   : " + (workOk ? "PASS" : "FAIL"));
            Console.WriteLine("  GATE tail   <  2.0  : " + (tailOk ? "PASS" : "FAIL"));
            Console.WriteLine();
            Console.WriteLine(nullOk && workOk && tailOk
                ? "  RESULT: the harness can resolve a 1.10x difference. Phase 0.5 gate MET."
                : "  RESULT: the harness CANNOT yet resolve 1.10x. Do not trust downstream perf claims.");
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("bench-calibration-end", afterSuite: true);
    }

    /// <summary>One generated depthwise kernel plus its device buffers.</summary>
    private sealed class Variant : IDisposable
    {
        private readonly DirectPtxModule _module;
        private readonly IntPtr _fn;
        private readonly DirectPtxBuffer _in, _w, _b, _out;
        internal long Threads { get; }
        internal uint Blocks { get; }

        private Variant(DirectPtxModule module, IntPtr fn, DirectPtxBuffer i,
                        DirectPtxBuffer w, DirectPtxBuffer b, DirectPtxBuffer o,
                        long threads, uint blocks)
        { _module = module; _fn = fn; _in = i; _w = w; _b = b; _out = o; Threads = threads; Blocks = blocks; }

        internal static Variant Create(DirectPtxRuntime runtime, int channels)
        {
            var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, channels, H, W);
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            long elements = (long)N * channels * H * W;
            var dIn = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
            var dW = runtime.AllocateBytes((nuint)((long)channels * 9 * sizeof(float)));
            var dB = runtime.AllocateBytes((nuint)((long)channels * sizeof(float)));
            var dOut = runtime.AllocateBytes((nuint)(elements * sizeof(float)));

            var host = new float[elements];
            for (long i = 0; i < elements; i++) host[i] = (float)(((i * 37 % 97) - 48) / 64.0);
            dIn.Upload<float>(host);
            var hw = new float[channels * 9];
            for (int i = 0; i < hw.Length; i++) hw[i] = (float)(((i * 53 % 89) - 44) / 128.0);
            dW.Upload<float>(hw);
            var hb = new float[channels];
            for (int i = 0; i < hb.Length; i++) hb[i] = (float)(((i * 29 % 71) - 35) / 256.0);
            dB.Upload<float>(hb);

            return new Variant(module, fn, dIn, dW, dB, dOut, spec.Space.TotalThreads,
                               emitter.LaunchBlocks);
        }

        internal unsafe void Launch()
        {
            IntPtr a = _in.Pointer, b = _w.Pointer, c = _b.Pointer, d = _out.Pointer;
            void** args = stackalloc void*[4];
            args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &d;
            _module.Launch(_fn, Blocks, 1, 1, PtxAffineEmitter.BlockThreads, 1, 1, 0, args);
        }

        public void Dispose()
        {
            _in.Dispose(); _w.Dispose(); _b.Dispose(); _out.Dispose(); _module.Dispose();
        }
    }

    /// <summary>Alternates A and B sample-by-sample so both see identical clock state.</summary>
    private static (Dist A, Dist B, double PairedRatio) Interleaved(Action synchronize, Action a, Action b)
    {
        for (int i = 0; i < Warmup; i++) { a(); b(); }
        synchronize();
        var sa = new double[Samples];
        var sb = new double[Samples];
        for (int i = 0; i < Samples; i++)
        {
            // COUNTERBALANCED. Always timing A in slot 1 and B in slot 2 confounds
            // slot order with variant: the second region of a pair inherits clock
            // state from the first, which showed up as a 2.5% null-test error on a
            // comparison whose true ratio is exactly 1.000x. Alternating the order
            // gives each variant an equal share of both slots so the bias cancels.
            bool aFirst = (i & 1) == 0;
            double first = TimeRegion(synchronize, aFirst ? a : b);
            double second = TimeRegion(synchronize, aFirst ? b : a);
            sa[i] = aFirst ? first : second;
            sb[i] = aFirst ? second : first;
        }
        // PAIRED RATIO. median(A)/median(B) compares two distributions gathered over
        // the whole run, so clock drift during the run leaks straight into the ratio:
        // on an A/A comparison whose true value is exactly 1.000x it produced 2.5-3.8%
        // error. Taking the ratio WITHIN each sample pair -- two regions microseconds
        // apart -- cancels drift, and the median over pairs is robust to outliers.
        var ratios = new double[Samples];
        for (int i = 0; i < Samples; i++) ratios[i] = sb[i] / sa[i];
        Array.Sort(ratios);
        double pairedRatio = Percentile(ratios, 0.5);

        Array.Sort(sa); Array.Sort(sb);
        return (Summarise(sa), Summarise(sb), pairedRatio);
    }

    /// <summary>Times one batched region and returns milliseconds per launch.</summary>
    private static double TimeRegion(Action synchronize, Action launch)
    {
        long start = Stopwatch.GetTimestamp();
        for (int k = 0; k < LaunchesPerSample; k++) launch();
        synchronize();
        return Stopwatch.GetElapsedTime(start).TotalMilliseconds / LaunchesPerSample;
    }

    private static Dist Summarise(double[] sorted) =>
        new(Percentile(sorted, 0.5), Percentile(sorted, 0.95), sorted.Average());

    private static double Percentile(double[] sorted, double fraction)
    {
        double position = (sorted.Length - 1) * fraction;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }
}
