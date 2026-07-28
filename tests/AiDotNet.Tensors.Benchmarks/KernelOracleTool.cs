// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Scores every generated kernel against a ceiling derived from its own spec.
/// </summary>
/// <remarks>
/// <para>
/// THE POINT IS THAT NO KERNEL NEEDS A HAND-WRITTEN ORACLE. A spec already carries its
/// operation count and the bytes that must move; the device rates come from
/// <see cref="DeviceCalibration"/>. So the ceiling, the limiter, and the headroom fall out
/// for any kernel the generator can emit -- including ones written after this tool.
/// </para>
/// <para>
/// Comparing against a competitor cannot answer the question this answers. A ratio against
/// cuBLAS conflates "how much is left in this kernel" with "how good is the other
/// implementation": a kernel at 96% of its ceiling is finished whatever the competitor does,
/// and one at 40% has most of its performance still on the table even while winning. Only the
/// second number tells anyone what to do next.
/// </para>
/// <para>
/// The LIMITER column is the actionable one. A memory-bound kernel cannot be helped by a
/// better instruction schedule and a compute-bound one cannot be helped by staging, and this
/// campaign burned two levers on exactly that confusion before it started reading counters.
/// </para>
/// </remarks>
internal static class KernelOracleTool
{
    internal static void Run(string[] args)
    {
        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;

        Console.WriteLine();
        Console.WriteLine("KERNEL ORACLE - every kernel against a ceiling derived from its spec");
        Console.WriteLine("device sm_{0}{1}", major, minor);
        Console.WriteLine();
        Console.WriteLine("calibrating...");

        var rates = DeviceCalibration.Measure(runtime, major, minor);
        var reference = CodegenMachineModel.Rtx3080Locked;
        var machine = DeviceCalibration.ToMachineModel(
            rates, reference.Multiprocessors, reference.ClockHz);

        Console.WriteLine();
        Console.WriteLine("{0,-26} {1,16} {2,16}  {3}", "rate", "measured", "hardcoded", "");
        Report("DRAM GB/s", rates.DramBytesPerSecond / 1e9, reference.DramBytesPerSecond / 1e9);
        Console.WriteLine("{0,-26} {1,16} {2,16}  {3}", "fp32 TFLOP/s", "-",
            (reference.MacsPerSecond * 2 / 1e12).ToString("0.0", CultureInfo.InvariantCulture),
            "architectural; not probed (see DeviceCalibration)");
        Report("tensor-core TFLOP/s", rates.TensorCoreMacsPerSecond * 2 / 1e12,
               reference.TensorCoreMacsPerSecond * 2 / 1e12);

        Console.WriteLine();
        Console.WriteLine(
            "{0,-34} {1,10} {2,9} {3,10} {4,10} {5,9} {6,8}",
            "kernel", "GFLOP", "AI f/B", "measured", "ceiling", "% of max", "limiter");

        foreach (var (label, spec) in Kernels())
        {
            Score(runtime, machine, label, spec, major, minor);
        }

        Console.WriteLine();
        Console.WriteLine("AI = arithmetic intensity, FLOPs per byte at minimum traffic. The limiter");
        Console.WriteLine("is which side of the roofline the kernel sits on, and it decides which");
        Console.WriteLine("class of lever can help: a memory-bound kernel is not helped by a better");
        Console.WriteLine("instruction schedule, and a compute-bound one is not helped by staging.");
    }

    private static void Report(string name, double measured, double hardcoded)
    {
        double delta = hardcoded > 0 ? (measured / hardcoded - 1.0) * 100.0 : 0.0;
        Console.WriteLine("{0,-26} {1,16} {2,16}  {3}",
            name,
            measured.ToString("0.0", CultureInfo.InvariantCulture),
            hardcoded > 0 ? hardcoded.ToString("0.0", CultureInfo.InvariantCulture) : "-",
            hardcoded > 0
                ? (delta >= 0 ? "+" : "") + delta.ToString("0.0", CultureInfo.InvariantCulture) + "%"
                : "(none in the hardcoded model)");
    }

    private static void Score(
        DirectPtxRuntime runtime, CodegenMachineModel machine, string label,
        CodegenKernelSpec spec, int major, int minor)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, major, minor);
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var pointers = new IntPtr[spec.ParameterCount];
            var outputBindings = new List<CodegenTensorBinding> { spec.Output };
            foreach (var extra in spec.ExtraOutputs) outputBindings.Add(extra.Binding);

            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                var binding = spec.Inputs[i];
                var buffer = runtime.AllocateBytes(
                    (nuint)(binding.ElementCount * binding.ElementBytes * spec.Algebra.Components()));
                buffers.Add(buffer);
                pointers[binding.ParameterIndex] = buffer.Pointer;
            }
            foreach (var binding in outputBindings)
            {
                var buffer = runtime.AllocateBytes(
                    (nuint)(binding.ElementCount * binding.ElementBytes * spec.Algebra.Components()));
                buffers.Add(buffer);
                pointers[binding.ParameterIndex] = buffer.Pointer;
            }

            var prediction = CodegenPerformanceModel.Predict(
                spec, spec.Space.TotalThreads, emitter.DynamicLoadsPerThread,
                machine, emitter.LaunchBlockX);

            double measured = Time(runtime, module, fn, pointers,
                (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX);

            // The CEILING is the binding roofline term, not the model's full prediction: the
            // prediction includes issue and occupancy penalties, which are properties of the
            // schedule we chose rather than of the hardware. A ceiling has to be something no
            // schedule can beat.
            double ceiling = Math.Max(prediction.DramMicroseconds, prediction.ComputeMicroseconds);
            double flops = prediction.Macs * 2.0;
            double intensity = prediction.UniqueBytes > 0
                ? flops / prediction.UniqueBytes : double.PositiveInfinity;

            Console.WriteLine(
                "{0,-34} {1,10} {2,9} {3,10} {4,10} {5,9} {6,8}",
                label,
                (flops / 1e9).ToString("0.00", CultureInfo.InvariantCulture),
                intensity.ToString("0.0", CultureInfo.InvariantCulture),
                measured.ToString("0.0", CultureInfo.InvariantCulture) + " us",
                ceiling.ToString("0.0", CultureInfo.InvariantCulture) + " us",
                (ceiling / measured * 100.0).ToString("0.0", CultureInfo.InvariantCulture) + "%",
                prediction.ComputeMicroseconds >= prediction.DramMicroseconds ? "compute" : "memory");
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-34} {1}", label, ex.Message.Replace('\n', ' '));
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }

    private static double Time(
        DirectPtxRuntime runtime, DirectPtxModule module, IntPtr fn, IntPtr[] pointers,
        uint blocks, uint threads)
    {
        for (int i = 0; i < 5; i++) Launch(module, fn, pointers, blocks, threads);
        runtime.Synchronize();

        double best = double.MaxValue;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 50; i++) Launch(module, fn, pointers, blocks, threads);
            runtime.Synchronize();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / 50);
        }
        return best;
    }

    private static unsafe void Launch(
        DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks, uint threads)
    {
        fixed (IntPtr* pinned = pointers)
        {
            void** argv = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) argv[i] = pinned + i;
            module.Launch(fn, blocks, 1, 1, threads, 1, 1, 0, argv);
        }
    }

    /// <summary>
    /// One kernel per family the generator can emit. Deliberately spans both roofline sides,
    /// because a report that only covers compute-bound kernels cannot show that the limiter
    /// column is doing any work.
    /// </summary>
    private static IEnumerable<(string, CodegenKernelSpec)> Kernels()
    {
        var map1 = new[] { CodegenAffineExpr.Axis(0) };

        // Pure streaming: no reduction at all, so this must land memory-bound.
        {
            const int N = 1 << 22;
            var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", N));
            yield return ("elementwise copy 4M",
                new CodegenKernelSpec("orc_copy", space,
                    new[] { new CodegenTensorBinding(0, "x", new[] { N }, map1) },
                    new CodegenTensorBinding(1, "y", new[] { N }, map1, isOutput: true),
                    new[] { 0 }, CodegenReduceKind.None));
        }

        // Streaming with an epilogue: more arithmetic, same bytes.
        {
            const int N = 1 << 22;
            var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", N));
            yield return ("elementwise gelu 4M",
                new CodegenKernelSpec("orc_gelu", space,
                    new[] { new CodegenTensorBinding(0, "x", new[] { N }, map1) },
                    new CodegenTensorBinding(1, "y", new[] { N }, map1, isOutput: true),
                    new[] { 0 }, CodegenReduceKind.None,
                    activation: CodegenActivationKind.Gelu));
        }

        // A long reduction to a small output: reads a lot, writes almost nothing.
        {
            const int Rows = 4096, Inner = 1024;
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("i", Rows), CodegenAxis.Reduce("k", Inner));
            yield return ("row sum 4096x1024",
                new CodegenKernelSpec("orc_rowsum", space,
                    new[]
                    {
                        new CodegenTensorBinding(0, "x", new[] { Rows, Inner },
                            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }),
                    },
                    new CodegenTensorBinding(1, "y", new[] { Rows }, map1, isOutput: true),
                    new[] { 0 }, CodegenReduceKind.Sum));
        }

        // fp32 matmul: the classic compute-bound shape, on the scalar pipe.
        {
            const int Size = 1024;
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("m", Size), CodegenAxis.Parallel("n", Size),
                CodegenAxis.Reduce("k", Size));
            yield return ("fp32 matmul 1024^3",
                new CodegenKernelSpec("orc_gemm32", space,
                    new[]
                    {
                        new CodegenTensorBinding(0, "a", new[] { Size, Size },
                            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) }),
                        new CodegenTensorBinding(1, "b", new[] { Size, Size },
                            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) }),
                    },
                    new CodegenTensorBinding(2, "c", new[] { Size, Size },
                        new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true),
                    new[] { 0, 1 }, CodegenReduceKind.Sum));
        }

        // An embedding gather: pure memory, and data-dependent, so its minimum traffic is not
        // what a naive read of the shapes would suggest.
        {
            const int Tokens = 1 << 20, Vocabulary = 4096, Width = 64;
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("t", Tokens), CodegenAxis.Parallel("e", Width));

            var ids = new CodegenTensorBinding(0, "ids", new[] { Tokens }, map1,
                elementType: CodegenElementType.Int32);
            var table = new CodegenTensorBinding(1, "table", new[] { Vocabulary, Width },
                new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
                indirect: new CodegenIndirectIndex?[]
                {
                    new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), Vocabulary),
                    null,
                });
            var output = new CodegenTensorBinding(2, "out", new[] { Tokens, Width },
                new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

            yield return ("embedding gather 1M x 64",
                new CodegenKernelSpec("orc_gather", space, new[] { ids, table }, output,
                    new[] { 1 }, CodegenReduceKind.None));
        }
    }
}
