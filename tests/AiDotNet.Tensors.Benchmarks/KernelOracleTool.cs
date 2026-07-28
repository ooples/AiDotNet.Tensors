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
            "{0,-34} {1,10} {2,9} {3,14} {4,10} {5,9} {6,8}",
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

                // INDEX TENSORS MUST BE FILLED. An uninitialised buffer is typically zeros, so
                // every row of a gather would hit entry 0 and every scatter would contend on
                // one destination -- which is not a workload, it is the worst case, and
                // optimising against it would chase a number no caller produces. Spread with a
                // stride coprime to the table so the accesses are neither sequential nor
                // degenerate.
                if (binding.IsIndexTensor)
                {
                    long count = binding.ElementCount;
                    var indices = new int[count];
                    int bound = IndexBoundFor(spec, binding);
                    for (long e = 0; e < count; e++) indices[e] = (int)((e * 7919) % bound);
                    buffer.Upload<int>(indices);
                }

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

            var prediction0 = CodegenPerformanceModel.Predict(
                spec, spec.Space.TotalThreads, emitter.DynamicLoadsPerThread,
                machine, emitter.LaunchBlockX);
            var prediction = prediction0;

            var timing = StableTimer.Measure(
                runtime,
                () => Launch(module, fn, pointers,
                             (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX),
                workUnits: Math.Max(prediction0.Macs, prediction0.UniqueBytes));
            double measured = timing.Microseconds;

            // The CEILING is the binding roofline term, not the model's full prediction: the
            // prediction includes issue and occupancy penalties, which are properties of the
            // schedule we chose rather than of the hardware. A ceiling has to be something no
            // schedule can beat.
            double ceiling = Math.Max(prediction.DramMicroseconds, prediction.ComputeMicroseconds);
            double flops = prediction.Macs * 2.0;
            double intensity = prediction.UniqueBytes > 0
                ? flops / prediction.UniqueBytes : double.PositiveInfinity;

            // AN UNSTABLE ROW REPORTS NO PERCENTAGE. Deriving "13.5% of ceiling" from samples
            // that disagree by half is how a ranked work list gets built on noise, which is
            // what happened before this gate existed.
            Console.WriteLine(
                "{0,-34} {1,10} {2,9} {3,14} {4,10} {5,9} {6,8}",
                label,
                (flops / 1e9).ToString("0.00", CultureInfo.InvariantCulture),
                intensity.ToString("0.0", CultureInfo.InvariantCulture),
                timing.Describe(),
                ceiling.ToString("0.0", CultureInfo.InvariantCulture) + " us",
                timing.Stable
                    ? (ceiling / measured * 100.0).ToString("0.0", CultureInfo.InvariantCulture) + "%"
                    : "-",
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

    /// <summary>The extent an index tensor addresses, read from whichever binding consumes it.</summary>
    private static int IndexBoundFor(CodegenKernelSpec spec, CodegenTensorBinding indexTensor)
    {
        foreach (var candidate in AllBindings(spec))
            for (int d = 0; d < candidate.Indirect.Count; d++)
                if (candidate.Indirect[d] is { } indirect
                    && spec.Inputs[indirect.IndexInput] == indexTensor)
                    return indirect.Bound;

        return 1;
    }

    private static IEnumerable<CodegenTensorBinding> AllBindings(CodegenKernelSpec spec)
    {
        foreach (var input in spec.Inputs) yield return input;
        yield return spec.Output;
        foreach (var extra in spec.ExtraOutputs) yield return extra.Binding;
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
    /// <summary>
    /// One kernel per family the generator can emit, spanning both roofline sides.
    /// </summary>
    /// <remarks>
    /// A report that only covers compute-bound kernels cannot show that the limiter column is
    /// doing any work, and one that only covers the families somebody happened to be
    /// optimising cannot find the neglected ones -- which is the whole purpose. Every family
    /// the emitter supports appears here: streaming, epilogues, reductions, contractions,
    /// convolution, gather, scatter, complex arithmetic, mixed precision and multi-output.
    /// </remarks>
    private static IEnumerable<(string, CodegenKernelSpec)> Kernels()
    {
        var map1 = new[] { CodegenAffineExpr.Axis(0) };

        // ---- streaming: no reduction, so these must land memory-bound --------------------
        {
            const int N = 1 << 22;
            var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", N));
            yield return ("elementwise copy 4M",
                new CodegenKernelSpec("orc_copy", space,
                    new[] { new CodegenTensorBinding(0, "x", new[] { N }, map1) },
                    new CodegenTensorBinding(1, "y", new[] { N }, map1, isOutput: true),
                    new[] { 0 }, CodegenReduceKind.None));

            yield return ("elementwise gelu 4M",
                new CodegenKernelSpec("orc_gelu", space,
                    new[] { new CodegenTensorBinding(0, "x", new[] { N }, map1) },
                    new CodegenTensorBinding(1, "y", new[] { N }, map1, isOutput: true),
                    new[] { 0 }, CodegenReduceKind.None,
                    activation: CodegenActivationKind.Gelu));

            // Mixed precision: half the bytes for the same arithmetic, so its ceiling halves.
            yield return ("elementwise copy 4M, fp16",
                new CodegenKernelSpec("orc_copy16", space,
                    new[]
                    {
                        new CodegenTensorBinding(0, "x", new[] { N }, map1,
                            elementType: CodegenElementType.Float16),
                    },
                    new CodegenTensorBinding(1, "y", new[] { N }, map1, isOutput: true,
                        elementType: CodegenElementType.Float16),
                    new[] { 0 }, CodegenReduceKind.None));

            // Three outputs from one pass: the fusion case, where the ceiling counts the
            // bytes ONCE while three separate kernels would move the input three times.
            yield return ("three outputs from one pass 4M",
                new CodegenKernelSpec("orc_multi", space,
                    new[] { new CodegenTensorBinding(0, "x", new[] { N }, map1) },
                    new CodegenTensorBinding(1, "a", new[] { N }, map1, isOutput: true),
                    new[] { 0 }, CodegenReduceKind.None,
                    extraOutputs: new[]
                    {
                        new CodegenExtraOutput(
                            new CodegenTensorBinding(2, "b", new[] { N }, map1, isOutput: true),
                            CodegenExtraOutputKind.AffineOfPrimary, Scale: 2.0),
                        new CodegenExtraOutput(
                            new CodegenTensorBinding(3, "c", new[] { N }, map1, isOutput: true),
                            CodegenExtraOutputKind.AffineOfPrimary, Scale: -1.0),
                    }));
        }

        // ---- reductions ------------------------------------------------------------------
        {
            const int Rows = 4096, Inner = 1024;
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("i", Rows), CodegenAxis.Reduce("k", Inner));
            var x = new CodegenTensorBinding(0, "x", new[] { Rows, Inner },
                new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
            var y = new CodegenTensorBinding(1, "y", new[] { Rows }, map1, isOutput: true);

            var rowSum = new CodegenKernelSpec("orc_rowsum", space, new[] { x }, y,
                new[] { 0 }, CodegenReduceKind.Sum);
            yield return ("row sum 4096x1024", rowSum);

            // THE SAME REDUCTION, SPLIT. One thread per row means consecutive threads read
            // 1024 elements apart, so a warp touches 32 separate cache lines per load instead
            // of four. Splitting the reduction axis gives consecutive threads consecutive k,
            // which is the whole difference. Both halves are reported: a split that only
            // moves cost into the combine pass is not a win.
            var chunked = CodegenSplitReduction.SplitChunked(rowSum, reductionAxis: 1, splitFactor: 32);
            yield return ("  row sum, split x32 (partial)", chunked.Partial);
            yield return ("  row sum, split x32 (combine)", chunked.Combine);

            // THE AUTOMATIC CHOICE, from the planner rather than from a hand-picked factor.
            // If this does not match the manual split, the planner is the thing to fix -- a
            // lever nobody can reach without knowing the factor is not a lever.
            var planned = CodegenSplitReduction.TryPlan(rowSum);
            if (planned is not null)
            {
                yield return ("  row sum, PLANNED (partial)", planned.Partial);
                yield return ("  row sum, PLANNED (combine)", planned.Combine);
            }

            // The same question for the softmax denominator, which carries a transform inside
            // the reduction: a split has to keep that transform in the PARTIAL pass only.
            var softmax = new CodegenKernelSpec("orc_softden_src", space, new[] { x }, y,
                new[] { 0 }, CodegenReduceKind.Sum, preReduce: CodegenPreReduceOp.Exp);
            var softPlan = CodegenSplitReduction.TryPlan(softmax);
            if (softPlan is not null)
            {
                yield return ("  softmax denom, PLANNED (partial)", softPlan.Partial);
                yield return ("  softmax denom, PLANNED (combine)", softPlan.Combine);
            }

            // A softmax denominator: a reduction with a transform inside it.
            yield return ("softmax denom 4096x1024",
                new CodegenKernelSpec("orc_softden", space, new[] { x }, y,
                    new[] { 0 }, CodegenReduceKind.Sum,
                    preReduce: CodegenPreReduceOp.Exp));

            // A max reduction. It CAN be split -- max is associative -- which the splitter
            // used to deny, leaving this the only reduction shape with no available fix.
            var rowMax = new CodegenKernelSpec("orc_rowmax", space, new[] { x }, y,
                new[] { 0 }, CodegenReduceKind.Max);
            yield return ("row max 4096x1024", rowMax);

            var maxPlan = CodegenSplitReduction.TryPlan(rowMax);
            if (maxPlan is not null)
            {
                yield return ("  row max, PLANNED (partial)", maxPlan.Partial);
                yield return ("  row max, PLANNED (combine)", maxPlan.Combine);
            }
        }

        // ---- contractions ----------------------------------------------------------------
        {
            const int Size = 1024;
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("m", Size), CodegenAxis.Parallel("n", Size),
                CodegenAxis.Reduce("k", Size));

            CodegenKernelSpec Gemm(string name, CodegenElementType type) =>
                new(name, space,
                    new[]
                    {
                        new CodegenTensorBinding(0, "a", new[] { Size, Size },
                            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) },
                            elementType: type),
                        new CodegenTensorBinding(1, "b", new[] { Size, Size },
                            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) },
                            elementType: type),
                    },
                    new CodegenTensorBinding(2, "c", new[] { Size, Size },
                        new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true),
                    new[] { 0, 1 }, CodegenReduceKind.Sum);

            yield return ("fp32 matmul 1024^3", Gemm("orc_gemm32", CodegenElementType.Float32));

            // The fp16 shape is scored against the TENSOR-CORE rate, which is the only reason
            // its ceiling is meaningful -- against the fp32 pipe it would read as over 100%.
            yield return ("fp16 matmul 1024^3 (tensor core)",
                Gemm("orc_gemm16", CodegenElementType.Float16));
        }

        // ---- convolution -----------------------------------------------------------------
        {
            const int Batch = 32, Channels = 64, Height = 32, Width = 32, Filters = 64;
            var space = new CodegenIterationSpace(
                CodegenAxis.Parallel("n", Batch), CodegenAxis.Parallel("k", Filters),
                CodegenAxis.Parallel("oh", Height), CodegenAxis.Parallel("ow", Width),
                CodegenAxis.Reduce("c", Channels));

            var input = new CodegenTensorBinding(0, "x", new[] { Batch, Channels, Height, Width },
                new[]
                {
                    CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(4),
                    CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(3),
                });
            var weights = new CodegenTensorBinding(1, "w", new[] { Filters, Channels },
                new[] { CodegenAffineExpr.Axis(1), CodegenAffineExpr.Axis(4) });
            var output = new CodegenTensorBinding(2, "y", new[] { Batch, Filters, Height, Width },
                new[]
                {
                    CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1),
                    CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(3),
                }, isOutput: true);

            yield return ("conv 1x1 32x64x32x32 -> 64",
                new CodegenKernelSpec("orc_conv1x1", space, new[] { input, weights }, output,
                    new[] { 0, 1 }, CodegenReduceKind.Sum,
                    activation: CodegenActivationKind.ReLU));
        }

        // ---- data-dependent indexing -----------------------------------------------------
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
            var gathered = new CodegenTensorBinding(2, "out", new[] { Tokens, Width },
                new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

            yield return ("embedding gather 1M x 64",
                new CodegenKernelSpec("orc_gather", space, new[] { ids, table }, gathered,
                    new[] { 1 }, CodegenReduceKind.None));

            // The backward: same traffic, but every store is an atomic accumulation.
            var grad = new CodegenTensorBinding(1, "grad", new[] { Tokens, Width },
                new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
            var gradTable = new CodegenTensorBinding(2, "grad_table", new[] { Vocabulary, Width },
                new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
                isOutput: true,
                indirect: new CodegenIndirectIndex?[]
                {
                    new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), Vocabulary),
                    null,
                });

            yield return ("embedding scatter 1M x 64 (atomic)",
                new CodegenKernelSpec("orc_scatter", space, new[] { ids, grad }, gradTable,
                    new[] { 1 }, CodegenReduceKind.None));
        }

        // ---- complex arithmetic ----------------------------------------------------------
        {
            const int N = 1 << 21;
            var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", N));

            yield return ("complex elementwise product 2M",
                new CodegenKernelSpec("orc_cmul", space,
                    new[]
                    {
                        new CodegenTensorBinding(0, "a", new[] { N }, map1),
                        new CodegenTensorBinding(1, "b", new[] { N }, map1),
                    },
                    new CodegenTensorBinding(2, "c", new[] { N }, map1, isOutput: true),
                    new[] { 0, 1 }, CodegenReduceKind.None,
                    algebra: CodegenAlgebra.Complex));
        }
    }
}
