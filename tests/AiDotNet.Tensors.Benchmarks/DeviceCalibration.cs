// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Measures a device's achievable rates, so a kernel's ceiling is derived from what the
/// hardware does rather than from what a datasheet claims.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="CodegenMachineModel.Rtx3080Locked"/> is a hand-written constant, and every
/// number in it is an assertion. A DRAM figure that is 15% optimistic makes every
/// memory-bound kernel look 15% short and sends work after headroom that does not exist; one
/// that is pessimistic reports finished kernels and hides real headroom. Neither failure
/// announces itself.
/// </para>
/// <para>
/// Each probe here is generated from a <see cref="CodegenKernelSpec"/> through the same
/// emitters the real kernels use, so the rates are what THIS code generator can extract from
/// THIS device -- not what a hand-tuned vendor kernel could. That is the honest denominator
/// for judging a generated kernel.
/// </para>
/// </remarks>
internal static class DeviceCalibration
{
    /// <summary>Rates measured on a device.</summary>
    internal sealed record Rates(
        double DramBytesPerSecond,
        double TensorCoreMacsPerSecond);

    /// <summary>Measures all three rates and builds a machine model from them.</summary>
    internal static Rates Measure(DirectPtxRuntime runtime, int major, int minor)
    {
        return new Rates(
            MeasureDram(runtime, major, minor),
            MeasureTensorCore(runtime, major, minor));
    }

    /// <summary>Builds a machine model whose rates are the measured ones.</summary>
    internal static CodegenMachineModel ToMachineModel(Rates rates, int multiprocessors, double clockHz)
    {
        var reference = CodegenMachineModel.Rtx3080Locked;

        return new CodegenMachineModel(
            name: $"measured ({multiprocessors} SM @ {clockHz / 1e9:0.00} GHz)",
            multiprocessors: multiprocessors,
            clockHz: clockHz,
            loadInstructionsPerSmPerCycle: reference.LoadInstructionsPerSmPerCycle,

            // THE fp32 LANE COUNT IS AN ARCHITECTURAL FACT AND IS NOT PROBED. A kernel-based
            // probe measures what a kernel ACHIEVED, not what the pipe can issue, and the two
            // are different numbers: the first version of this file timed a mat-vec, which is
            // memory-bound, and reported 0.0 TFLOP/s. Feeding that back as the peak made every
            // compute ceiling astronomically loose -- the report claimed kernels were running
            // at 6000% of their ceiling, which is not a headroom figure, it is a broken model.
            //
            // A ceiling has to be something no schedule can beat, so for the scalar pipe it is
            // the hardware issue rate. The tensor-core rate below IS probed, because there the
            // question is different: whether OUR instruction mix can reach the hardware rate,
            // which the ceiling probe answers directly by running that mix with memory removed.
            fmaLanesPerSm: reference.FmaLanesPerSm,
            dramBytesPerSecond: rates.DramBytesPerSecond,
            tensorCoreMacsPerSmPerCycle: rates.TensorCoreMacsPerSecond / multiprocessors / clockHz);
    }

    /// <summary>
    /// Streams a buffer far larger than L2 through a copy kernel and reports bytes per second.
    /// </summary>
    /// <remarks>
    /// The working set is deliberately past cache capacity: a probe that fits in L2 measures
    /// L2, reports a bandwidth several times the real one, and makes every memory-bound kernel
    /// look like a failure.
    /// </remarks>
    private static double MeasureDram(DirectPtxRuntime runtime, int major, int minor)
    {
        const int Elements = 64 * 1024 * 1024;      // 256 MB in, 256 MB out -- far past L2

        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", Elements));
        var map = new[] { CodegenAffineExpr.Axis(0) };
        var input = new CodegenTensorBinding(0, "x", new[] { Elements }, map);
        var output = new CodegenTensorBinding(1, "y", new[] { Elements }, map, isOutput: true);

        var spec = new CodegenKernelSpec("calib_stream", space, new[] { input }, output,
            new[] { 0 }, CodegenReduceKind.None);

        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(spec, major, minor);
        using var module = runtime.LoadModule(ptx);
        IntPtr fn = module.GetFunction(spec.Name, out _);

        using var src = runtime.AllocateBytes((nuint)((long)Elements * sizeof(float)));
        using var dst = runtime.AllocateBytes((nuint)((long)Elements * sizeof(float)));
        var pointers = new[] { src.Pointer, dst.Pointer };

        double us = Time(runtime, module, fn, pointers,
            (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX, iterations: 20);

        // Read plus write.
        return 2.0 * Elements * sizeof(float) / (us * 1e-6);
    }

    /// <summary>
    /// Measures the tensor-core pipe with the mma ceiling probe: the real staged kernel with
    /// its fragment loads hoisted out of the K loop.
    /// </summary>
    /// <remarks>
    /// This is the honest tensor-core rate for THIS emitter, because it is this emitter's own
    /// instruction stream with only the memory removed. A synthetic peak from a datasheet
    /// would not tell us whether our instruction mix can reach it -- and the whole reason to
    /// have this number is to separate "the mix is short" from "the memory is short".
    /// </remarks>
    private static double MeasureTensorCore(DirectPtxRuntime runtime, int major, int minor)
    {
        if (major < 7) return 0;

        const int Size = 2048;

        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", Size), CodegenAxis.Parallel("n", Size),
            CodegenAxis.Reduce("k", Size));

        var a = new CodegenTensorBinding(0, "a", new[] { Size, Size },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) },
            elementType: CodegenElementType.Float16);
        var b = new CodegenTensorBinding(1, "b", new[] { Size, Size },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) },
            elementType: CodegenElementType.Float16);
        var output = new CodegenTensorBinding(2, "out", new[] { Size, Size },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        var spec = new CodegenKernelSpec("calib_mma", space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        var emitter = new PtxTensorCoreEmitter { MmaCeilingProbe = true };
        if (!PtxTensorCoreEmitter.TryPlan(spec, major, minor, out var plan, out _)) return 0;
        if (plan is null || !emitter.CanStage(plan, out _)) return 0;

        string ptx = emitter.Emit(spec, major, minor);
        using var module = runtime.LoadModule(ptx);
        IntPtr fn = module.GetFunction(emitter.EmittedEntryName, out _);

        using var aBuf = runtime.AllocateBytes((nuint)((long)Size * Size * sizeof(ushort)));
        using var bBuf = runtime.AllocateBytes((nuint)((long)Size * Size * sizeof(ushort)));
        using var oBuf = runtime.AllocateBytes((nuint)((long)Size * Size * sizeof(float)));
        var pointers = new[] { aBuf.Pointer, bBuf.Pointer, oBuf.Pointer };

        double us = Time(runtime, module, fn, pointers,
            (uint)emitter.BlockCount(plan), (uint)emitter.BlockThreads, iterations: 30);

        return (double)Size * Size * Size / (us * 1e-6);
    }

    private static double Time(
        DirectPtxRuntime runtime, DirectPtxModule module, IntPtr fn, IntPtr[] pointers,
        uint blocks, uint threads, int iterations)
    {
        for (int i = 0; i < Math.Max(2, iterations / 5); i++) Launch(module, fn, pointers, blocks, threads);
        runtime.Synchronize();

        double best = double.MaxValue;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) Launch(module, fn, pointers, blocks, threads);
            runtime.Synchronize();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / iterations);
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
}
