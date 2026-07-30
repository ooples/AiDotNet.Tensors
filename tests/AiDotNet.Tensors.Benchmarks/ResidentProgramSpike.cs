// Copyright (c) AiDotNet. All rights reserved.
// Phase 3 spike: size the prize before building the compiler.
//
// The resident-program idea is that a chain of operators should run as ONE device
// program -- no launch boundary between stages, no intermediate tensor pushed out to
// HBM and pulled back. That is a large amount of compiler work (device-side work
// queues, grid-wide dependencies, halo handling), so the honest first question is not
// "how do we build it" but "how much is there to win".
//
// The first version of this spike point-estimated the two costs: one tiny-kernel
// median for the launch floor, one copy median for the intermediate traffic. Repeating
// it moved the tiny kernel from 13.2 to 32.6 us and the answer from 31.2% to 14.3%,
// so both estimates were noise wearing a decimal point. Phase 0.5 had already shown
// why: at these sizes a single median is dominated by an unstable launch floor.
//
// This version measures the device instead of sampling it once:
//
//   CHARACTERISE   sweep a copy kernel across four orders of magnitude and fit
//                  time = launch + bytes / bandwidth. The intercept is the launch
//                  floor, the slope is achieved bandwidth, and the regression
//                  averages per-point noise rather than inheriting it.
//
//   PAIRED PRIZE   interleave "conv" against "conv then pool" and take the median
//                  per-sample DIFFERENCE, so the marginal cost of the second stage
//                  is measured directly with drift cancelling inside each pair.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class ResidentProgramSpike
{
    private const int Warmup = 20;
    private const int Samples = 51;
    private const int LaunchesPerSample = 50;   // Phase 0.5: amortises sync latency
    private const int Runs = 3;

    // The chain: conv 3x3 over 32 channels -> 64 channels at 28x28, then 2x2 pool.
    private const int N = 8, C = 32, K = 64, H = 28, W = 28;

    internal static void Run()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("resident-spike-start");
        GpuBenchmarkEnvironment.PrintSnapshot("resident-spike-start");

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        {
            Console.WriteLine("The resident spike requires the experimental SM86 device.");
            return;
        }

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            var conv = Stage.Create(runtime, ConvSpec());
            var pool = Stage.Create(runtime, PoolSpec());
            try
            {
                long interElements = (long)N * K * H * W;
                long writeBytes = interElements * sizeof(float);

                Console.WriteLine();
                Console.WriteLine("Phase 3 spike - what a resident program would remove");
                Console.WriteLine("  chain          conv3x3+ReLU N" + N + "/C" + C + "->K" + K +
                                  "/" + H + "x" + W + "  then maxpool 2x2");
                Console.WriteLine("  intermediate   " + interElements.ToString("N0", CultureInfo.InvariantCulture) +
                                  " floats = " + (writeBytes / 1024.0 / 1024.0).ToString("F2", CultureInfo.InvariantCulture) + " MiB");
                Console.WriteLine();

                // ---- 1. Characterise the device.
                Console.WriteLine("copy-kernel sweep (fit: time = launch + bytes / bandwidth)");
                Console.WriteLine("     elements        MiB      us/launch");
                var xs = new List<double>();
                var ys = new List<double>();
                foreach (int elements in new[] { 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216 })
                {
                    using var probe = Stage.Create(runtime, Copy("spike_probe", elements));
                    double us = Median3(runtime, probe) * 1000.0;
                    double bytes = 2.0 * elements * sizeof(float);   // one read + one write
                    xs.Add(bytes);
                    ys.Add(us);
                    Console.WriteLine("  " + elements.ToString("N0", CultureInfo.InvariantCulture).PadLeft(11) +
                        (bytes / 1024.0 / 1024.0).ToString("F2", CultureInfo.InvariantCulture).PadLeft(11) +
                        us.ToString("F2", CultureInfo.InvariantCulture).PadLeft(15));
                }

                // TWO REGIMES, NOT ONE LINE. Fitting all seven points reported 814 GiB/s,
                // which is above this card's ~708 GiB/s peak and therefore impossible for
                // HBM traffic. The cause is the 5 MB L2: every buffer up to ~8 MiB of
                // traffic is largely cache-resident and moves faster per byte than memory
                // can. A single line through both regimes is the wrong model, so the
                // launch floor comes from the small end (traffic negligible) and the
                // bandwidth from the points that exceed L2.
                const double L2Bytes = 8.0 * 1024 * 1024;
                var bigX = new List<double>();
                var bigY = new List<double>();
                var smallY = new List<double>();
                for (int i = 0; i < xs.Count; i++)
                {
                    if (xs[i] > L2Bytes) { bigX.Add(xs[i]); bigY.Add(ys[i]); }
                    else smallY.Add(ys[i]);
                }

                smallY.Sort();
                double launchUs = smallY[smallY.Count / 2];   // median of the cache-resident end
                (double bigIntercept, double usPerByte, double r2) = Fit(bigX, bigY);
                double bandwidthGBs = 1.0 / usPerByte * 1e6 / 1e9;

                Console.WriteLine();
                Console.WriteLine("  launch floor (median, sub-L2 sizes) : " +
                    launchUs.ToString("F2", CultureInfo.InvariantCulture) + " us/launch");
                Console.WriteLine("  HBM bandwidth (fit over >L2 sizes)  : " +
                    bandwidthGBs.ToString("F0", CultureInfo.InvariantCulture) +
                    " GB/s   (card spec ~760 GB/s)");
                Console.WriteLine("  >L2 fit intercept / R^2             : " +
                    bigIntercept.ToString("F1", CultureInfo.InvariantCulture) + " us / " +
                    r2.ToString("F4", CultureInfo.InvariantCulture));
                Console.WriteLine();

                // ---- 2. The prize, measured as a PAIRED marginal cost.
                (double convUs, double chainUs, double marginalUs) = PairedMarginal(runtime, conv, pool);

                Console.WriteLine("measurement                             us/launch");
                Report("conv 3x3 + ReLU alone", convUs);
                Report("chain (conv then pool)", chainUs);
                Report("marginal cost of stage 2, paired", marginalUs);
                Console.WriteLine();

                // Fusion removes stage 2's launch, stage 2's read of the intermediate,
                // and stage 1's write of it. It cannot remove stage 2's arithmetic, the
                // real input read, or the final output write.
                double writeUs = writeBytes * usPerByte;
                double removable = marginalUs + writeUs;
                double bound = removable / chainUs;

                Console.WriteLine("  stage-2 marginal (launch+read+math+write) : " +
                    marginalUs.ToString("F1", CultureInfo.InvariantCulture) + " us");
                Console.WriteLine("  stage-1 write of the intermediate         : " +
                    writeUs.ToString("F1", CultureInfo.InvariantCulture) + " us  (fitted)");
                Console.WriteLine("  UPPER BOUND on the fusion win             : " +
                    (bound * 100).ToString("F1", CultureInfo.InvariantCulture) + "% of " +
                    chainUs.ToString("F1", CultureInfo.InvariantCulture) + " us");
                Console.WriteLine();
                Console.WriteLine("  This is a CEILING that is not reachable: it credits fusion with stage 2's");
                Console.WriteLine("  arithmetic and final write, which a fused kernel still pays, and charges");
                Console.WriteLine("  it nothing for recomputing halo values.");
                Console.WriteLine();
                Console.WriteLine("  Launch floor is " + launchUs.ToString("F1", CultureInfo.InvariantCulture) +
                    " us against a " + chainUs.ToString("F0", CultureInfo.InvariantCulture) + " us chain (" +
                    (launchUs / chainUs * 100).ToString("F1", CultureInfo.InvariantCulture) + "%), but a network");
                Console.WriteLine("  of small ops pays that floor PER OP. That, not this chain, is where a");
                Console.WriteLine("  resident program earns its keep, and it is what to measure next.");
            }
            finally { conv.Dispose(); pool.Dispose(); }
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("resident-spike-end", afterSuite: true);
    }

    private static void Report(string name, double us) =>
        Console.WriteLine("  " + name.PadRight(38) +
            us.ToString("F1", CultureInfo.InvariantCulture).PadLeft(10));

    // ------------------------------------------------------------------ specs

    private static CodegenKernelSpec ConvSpec()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", N), CodegenAxis.Parallel("k", K),
            CodegenAxis.Parallel("oh", H), CodegenAxis.Parallel("ow", W),
            CodegenAxis.Reduce("c", C), CodegenAxis.Reduce("kh", 3), CodegenAxis.Reduce("kw", 3));
        const int AN = 0, AK = 1, AOH = 2, AOW = 3, AC = 4, AKH = 5, AKW = 6;

        var input = new CodegenTensorBinding(0, "input", new[] { N, C, H, W },
            new[]
            {
                CodegenAffineExpr.Axis(AN), CodegenAffineExpr.Axis(AC),
                CodegenAffineExpr.Window(AOH, AKH, 1, 1), CodegenAffineExpr.Window(AOW, AKW, 1, 1)
            });
        var weights = new CodegenTensorBinding(1, "weights", new[] { K, C, 3, 3 },
            new[]
            {
                CodegenAffineExpr.Axis(AK), CodegenAffineExpr.Axis(AC),
                CodegenAffineExpr.Axis(AKH), CodegenAffineExpr.Axis(AKW)
            });
        var bias = new CodegenTensorBinding(2, "bias", new[] { K }, new[] { CodegenAffineExpr.Axis(AK) });
        var output = new CodegenTensorBinding(3, "output", new[] { N, K, H, W },
            new[]
            {
                CodegenAffineExpr.Axis(AN), CodegenAffineExpr.Axis(AK),
                CodegenAffineExpr.Axis(AOH), CodegenAffineExpr.Axis(AOW)
            }, isOutput: true);

        return new CodegenKernelSpec("spike_conv3x3", space, new[] { input, weights, bias }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum, biasInput: 2, activation: CodegenActivationKind.ReLU);
    }

    private static CodegenKernelSpec PoolSpec()
    {
        int oh = H / 2, ow = W / 2;
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", N), CodegenAxis.Parallel("k", K),
            CodegenAxis.Parallel("oh", oh), CodegenAxis.Parallel("ow", ow),
            CodegenAxis.Reduce("ph", 2), CodegenAxis.Reduce("pw", 2));
        const int AN = 0, AK = 1, AOH = 2, AOW = 3, APH = 4, APW = 5;

        var input = new CodegenTensorBinding(0, "input", new[] { N, K, H, W },
            new[]
            {
                CodegenAffineExpr.Axis(AN), CodegenAffineExpr.Axis(AK),
                CodegenAffineExpr.Window(AOH, APH, 2, 0), CodegenAffineExpr.Window(AOW, APW, 2, 0)
            });
        var output = new CodegenTensorBinding(1, "output", new[] { N, K, oh, ow },
            new[]
            {
                CodegenAffineExpr.Axis(AN), CodegenAffineExpr.Axis(AK),
                CodegenAffineExpr.Axis(AOH), CodegenAffineExpr.Axis(AOW)
            }, isOutput: true);

        return new CodegenKernelSpec("spike_pool2x2", space, new[] { input }, output,
            new[] { 0 }, CodegenReduceKind.Max);
    }

    private static CodegenKernelSpec Copy(string name, int elements)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", elements));
        var input = new CodegenTensorBinding(0, "input", new[] { elements },
            new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(1, "output", new[] { elements },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        return new CodegenKernelSpec(name, space, new[] { input }, output,
            new[] { 0 }, CodegenReduceKind.None);
    }

    // ----------------------------------------------------------------- timing

    /// <summary>Median of three run-medians: robust to a single bad run.</summary>
    private static double Median3(DirectPtxRuntime runtime, Stage stage)
    {
        var v = new double[Runs];
        for (int run = 0; run < Runs; run++) v[run] = Median(runtime.Synchronize, stage.Launch);
        Array.Sort(v);
        return v[Runs / 2];
    }

    private static double Median(Action synchronize, Action launch)
    {
        for (int i = 0; i < Warmup; i++) launch();
        synchronize();
        var samples = new double[Samples];
        for (int i = 0; i < Samples; i++)
        {
            long start = Stopwatch.GetTimestamp();
            for (int k = 0; k < LaunchesPerSample; k++) launch();
            synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds / LaunchesPerSample;
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    /// <summary>
    /// Interleaves "A" against "A then B" and returns the median per-sample difference,
    /// so drift cancels inside each pair instead of contaminating a subtraction of two
    /// independently-measured medians. Subtracting independent medians is what made the
    /// pool stage read 28.9 us on one run and 16.7 on the next.
    /// </summary>
    private static (double A, double Chain, double Marginal) PairedMarginal(
        DirectPtxRuntime runtime, Stage a, Stage b)
    {
        void OnlyA() { a.Launch(); }
        void Chain() { a.Launch(); b.Launch(); }

        for (int i = 0; i < Warmup; i++) { OnlyA(); Chain(); }
        runtime.Synchronize();

        var sa = new double[Samples];
        var sc = new double[Samples];
        var diff = new double[Samples];
        for (int i = 0; i < Samples; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            for (int k = 0; k < LaunchesPerSample; k++) OnlyA();
            runtime.Synchronize();
            sa[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds / LaunchesPerSample * 1000.0;

            long t1 = Stopwatch.GetTimestamp();
            for (int k = 0; k < LaunchesPerSample; k++) Chain();
            runtime.Synchronize();
            sc[i] = Stopwatch.GetElapsedTime(t1).TotalMilliseconds / LaunchesPerSample * 1000.0;

            diff[i] = sc[i] - sa[i];
        }
        Array.Sort(sa);
        Array.Sort(sc);
        Array.Sort(diff);
        return (sa[Samples / 2], sc[Samples / 2], diff[Samples / 2]);
    }

    /// <summary>Least-squares fit of y = intercept + slope*x, with R-squared.</summary>
    private static (double Intercept, double Slope, double R2) Fit(List<double> xs, List<double> ys)
    {
        int n = xs.Count;
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
        mx /= n; my /= n;

        double sxy = 0, sxx = 0;
        for (int i = 0; i < n; i++)
        {
            sxy += (xs[i] - mx) * (ys[i] - my);
            sxx += (xs[i] - mx) * (xs[i] - mx);
        }
        double slope = sxy / sxx;
        double intercept = my - slope * mx;

        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = intercept + slope * xs[i];
            ssRes += (ys[i] - pred) * (ys[i] - pred);
            ssTot += (ys[i] - my) * (ys[i] - my);
        }
        return (intercept, slope, ssTot > 0 ? 1.0 - ssRes / ssTot : double.NaN);
    }

    // ------------------------------------------------------------------ stage

    private sealed class Stage : IDisposable
    {
        private readonly DirectPtxModule _module;
        private readonly IntPtr _fn;
        private readonly IntPtr[] _pointers;
        private readonly uint _blocks;
        private readonly uint _blockX, _blockY;
        private readonly List<DirectPtxBuffer> _buffers;

        private Stage(DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks,
                      uint blockX, uint blockY, List<DirectPtxBuffer> buffers)
        { _module = module; _fn = fn; _pointers = pointers; _blocks = blocks;
          _blockX = blockX; _blockY = blockY; _buffers = buffers; }

        internal static Stage Create(DirectPtxRuntime runtime, CodegenKernelSpec spec)
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(
                spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var buffers = new List<DirectPtxBuffer>();
            var pointers = new IntPtr[spec.ParameterCount];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                long count = Elements(spec.Inputs[i].Shape);
                var b = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                var host = new float[count];
                for (long e = 0; e < count; e++) host[e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                b.Upload<float>(host);
                buffers.Add(b);
                pointers[i] = b.Pointer;
            }
            var outBuffer = runtime.AllocateBytes((nuint)(Elements(spec.Output.Shape) * sizeof(float)));
            buffers.Add(outBuffer);
            pointers[spec.Inputs.Count] = outBuffer.Pointer;

            return new Stage(module, fn, pointers, emitter.LaunchBlocks, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY, buffers);
        }

        internal unsafe void Launch()
        {
            fixed (IntPtr* pinned = _pointers)
            {
                void** argv = stackalloc void*[_pointers.Length];
                for (int i = 0; i < _pointers.Length; i++) argv[i] = pinned + i;
                _module.Launch(_fn, _blocks, 1, 1, _blockX, _blockY, 1, 0, argv);
            }
        }

        public void Dispose()
        {
            foreach (var b in _buffers) b.Dispose();
            _module.Dispose();
        }
    }

    private static long Elements(IReadOnlyList<int> shape)
    {
        long total = 1;
        for (int i = 0; i < shape.Count; i++) total *= shape[i];
        return total;
    }
}
