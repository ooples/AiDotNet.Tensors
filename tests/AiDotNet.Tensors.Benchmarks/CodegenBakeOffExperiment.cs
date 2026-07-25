// Copyright (c) AiDotNet. All rights reserved.
// Wall-clock row of the codegen bake-off: the hand-written depthwise Conv2D 3x3
// kernel against the one generated from a CodegenKernelSpec.
//
// The static metrics (registers, SASS instructions, LDG/STG, spills) are already
// measured by ptxas/nvdisasm and are equal or better for the generated kernel.
// This closes the last gate row by measuring the two on an idle device with the
// same protocol the rest of the #863 evidence uses: 30 warmups, 101 samples,
// median and P95.

using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class CodegenBakeOffExperiment
{
    private const int N = 2, C = 8, H = 8, W = 8;

    private readonly record struct Dist(double Mean, double Median, double P95, double P99);

    internal static void Run()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("codegen-bakeoff-start");
        GpuBenchmarkEnvironment.PrintSnapshot("codegen-bakeoff-start");

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        {
            Console.WriteLine("Bake-off requires the experimental SM86 device.");
            return;
        }

        int elements = N * C * H * W;
        var input = new float[elements];
        var weights = new float[C * 9];
        var bias = new float[C];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(((i * 37 % 97) - 48) / 64.0);
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(((i * 53 % 89) - 44) / 128.0);
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(((i * 29 % 71) - 35) / 256.0);

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var dIn = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
            using var dW = runtime.AllocateBytes((nuint)(weights.Length * sizeof(float)));
            using var dB = runtime.AllocateBytes((nuint)(bias.Length * sizeof(float)));
            using var dOutA = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
            using var dOutB = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
            dIn.Upload<float>(input); dW.Upload<float>(weights); dB.Upload<float>(bias);

            // Hand-written kernel.
            using var handWritten = new PtxDepthwiseConv2D3x3Kernel(runtime, N, C, H, W, relu: true);
            var hIn = DirectPtxTensorView.CreateOwned(dIn, handWritten.Blueprint.Tensors[0]);
            var hW = DirectPtxTensorView.CreateOwned(dW, handWritten.Blueprint.Tensors[1]);
            var hB = DirectPtxTensorView.CreateOwned(dB, handWritten.Blueprint.Tensors[2]);
            var hO = DirectPtxTensorView.CreateOwned(dOutA, handWritten.Blueprint.Tensors[3]);
            void LaunchHand() => handWritten.Launch(hIn, hW, hB, hO);

            // Generated kernel, from the spec.
            var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, C, H, W);
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
            using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
            IntPtr fn = module.GetFunction(spec.Name, out DirectPtxFunctionInfo info);
            uint blocks = emitter.LaunchBlocks;
            void LaunchGen() => LaunchFour(module, fn, dIn.Pointer, dW.Pointer, dB.Pointer, dOutB.Pointer, blocks);

            // INTERLEAVED A/B. Measuring all of A then all of B lets clock/thermal
            // drift masquerade as a kernel difference -- at this size the per-call
            // sync dominates, so an ordering artefact is larger than the signal.
            (Dist hand, Dist gen) = MeasureInterleaved(runtime.Synchronize, LaunchHand, LaunchGen);

            // Same-input equality is the precondition for comparing times at all.
            var a = new float[elements];
            var b = new float[elements];
            dOutA.Download<float>(a); dOutB.Download<float>(b);
            double worst = 0;
            for (int i = 0; i < elements; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));

            Console.WriteLine();
            Console.WriteLine("Codegen bake-off - depthwise Conv2D 3x3 + bias + ReLU, N" +
                N.ToString(CultureInfo.InvariantCulture) + "/C" + C.ToString(CultureInfo.InvariantCulture) +
                "/H" + H.ToString(CultureInfo.InvariantCulture) + "/W" + W.ToString(CultureInfo.InvariantCulture));
            Console.WriteLine("agreement hand-written vs generated: max abs " +
                worst.ToString("E3", CultureInfo.InvariantCulture));
            Console.WriteLine("generated: " + info.RegistersPerThread.ToString(CultureInfo.InvariantCulture) +
                " regs, " + emitter.ElidedGuards.ToString(CultureInfo.InvariantCulture) +
                " guards elided by interval analysis");
            Console.WriteLine();
            Console.WriteLine("method            median us     p95 us     mean us");
            Report("hand-written", hand);
            Report("generated", gen);
            Console.WriteLine();
            double ratio = hand.Median / gen.Median;
            Console.WriteLine("generated vs hand-written: " + ratio.ToString("F3", CultureInfo.InvariantCulture) +
                "x  (>1 means the generated kernel is faster)");
            Console.WriteLine("NOTE: this shape is tiny (1024 threads, 4 blocks) so launch overhead");
            Console.WriteLine("dominates; treat the static metrics as the primary signal.");
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("codegen-bakeoff-end");
    }

    private static void Report(string name, Dist d) =>
        Console.WriteLine(name.PadRight(18) +
            (d.Median * 1000.0).ToString("F2", CultureInfo.InvariantCulture).PadLeft(9) +
            (d.P95 * 1000.0).ToString("F2", CultureInfo.InvariantCulture).PadLeft(11) +
            (d.Mean * 1000.0).ToString("F2", CultureInfo.InvariantCulture).PadLeft(12));

    /// <summary>
    /// Alternates the two launches sample-by-sample so both see the same clock and
    /// thermal state. Returns (a, b) distributions.
    /// </summary>
    private static (Dist A, Dist B) MeasureInterleaved(Action synchronize, Action a, Action b)
    {
        for (int warmup = 0; warmup < 30; warmup++) { a(); b(); }
        synchronize();
        var sa = new double[101];
        var sb = new double[101];
        for (int i = 0; i < sa.Length; i++)
        {
            long s0 = Stopwatch.GetTimestamp();
            a();
            synchronize();
            sa[i] = Stopwatch.GetElapsedTime(s0).TotalMilliseconds;

            long s1 = Stopwatch.GetTimestamp();
            b();
            synchronize();
            sb[i] = Stopwatch.GetElapsedTime(s1).TotalMilliseconds;
        }
        Array.Sort(sa); Array.Sort(sb);
        return (new Dist(sa.Average(), Percentile(sa, 0.5), Percentile(sa, 0.95), Percentile(sa, 0.99)),
                new Dist(sb.Average(), Percentile(sb, 0.5), Percentile(sb, 0.95), Percentile(sb, 0.99)));
    }

    private static Dist Measure(Action synchronize, Action launch)
    {
        for (int warmup = 0; warmup < 30; warmup++) launch();
        synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        Array.Sort(samples);
        return new Dist(samples.Average(), Percentile(samples, 0.5), Percentile(samples, 0.95), Percentile(samples, 0.99));
    }

    private static double Percentile(double[] sorted, double fraction)
    {
        double position = (sorted.Length - 1) * fraction;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static unsafe void LaunchFour(
        DirectPtxModule module, IntPtr fn, IntPtr a, IntPtr b, IntPtr c, IntPtr d, uint blocks)
    {
        IntPtr pa = a, pb = b, pc = c, pd = d;
        void** args = stackalloc void*[4];
        args[0] = &pa; args[1] = &pb; args[2] = &pc; args[3] = &pd;
        module.Launch(fn, blocks, 1, 1, PtxAffineEmitter.BlockThreads, 1, 1, 0, args);
    }
}
