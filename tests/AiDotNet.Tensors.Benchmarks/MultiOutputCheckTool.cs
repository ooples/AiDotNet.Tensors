// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Runs multi-output kernels on the device and checks EVERY output buffer.
/// </summary>
/// <remarks>
/// Checking only the primary is how a kernel that writes garbage to its third buffer passes:
/// the first two agree, the summary says PASS, and the defect surfaces later as a training
/// run that will not converge. Every row here downloads and compares all of them.
/// </remarks>
internal static class MultiOutputCheckTool
{
    internal static void Run(string[] args)
    {
        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;

        Console.WriteLine();
        Console.WriteLine("MULTI-OUTPUT - N outputs from one iteration point");
        Console.WriteLine("device sm_{0}{1}", major, minor);
        Console.WriteLine();
        Console.WriteLine("{0,-42} {1,8} {2,12} {3,7}", "kernel", "outputs", "max abs dev", "result");

        int passed = 0, failed = 0;
        foreach (var (label, spec, inputs) in Cases())
        {
            if (CheckOne(runtime, label, spec, inputs, major, minor)) passed++;
            else failed++;
        }

        Console.WriteLine();
        Console.WriteLine("multi-output: {0} passed, {1} failed", passed, failed);
    }

    private static bool CheckOne(
        DirectPtxRuntime runtime, string label, CodegenKernelSpec spec, double[][] inputs,
        int major, int minor)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, major, minor);
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var pointers = new IntPtr[spec.ParameterCount];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                long count = spec.Inputs[i].ElementCount;
                var narrow = new float[count];
                for (long e = 0; e < count; e++) narrow[e] = (float)inputs[i][e];

                var buffer = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                buffer.Upload<float>(narrow);
                buffers.Add(buffer);
                pointers[i] = buffer.Pointer;
            }

            // THE OUTPUT BINDINGS IN PARAMETER ORDER: primary first, then the extras. If this
            // order disagreed with the emitter's, every buffer would be written to the wrong
            // place and the values would still look plausible in isolation.
            var outputBindings = new List<CodegenTensorBinding> { spec.Output };
            foreach (var extra in spec.ExtraOutputs) outputBindings.Add(extra.Binding);

            var outputBuffers = new List<DirectPtxBuffer>();
            foreach (var binding in outputBindings)
            {
                var buffer = runtime.AllocateBytes((nuint)(binding.ElementCount * sizeof(float)));
                buffer.Upload<float>(new float[binding.ElementCount]);
                buffers.Add(buffer);
                outputBuffers.Add(buffer);
                pointers[binding.ParameterIndex] = buffer.Pointer;
            }

            Launch(module, fn, pointers, (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX);
            runtime.Synchronize();

            double[][] want = spec.InterpretAll(inputs);

            // AN ARGMAX BUFFER HOLDS INT32, NOT FLOAT. Downloading it as float reads the
            // index's bit pattern as a denormal near zero, and the comparison then reports a
            // deviation equal to the largest index -- which is what the first run of this
            // tool did, and it looked exactly like a broken kernel.
            var kinds = new List<CodegenExtraOutputKind?> { null };
            foreach (var extra in spec.ExtraOutputs) kinds.Add(extra.Kind);

            double worst = 0;
            for (int o = 0; o < outputBuffers.Count; o++)
            {
                long count = outputBindings[o].ElementCount;

                if (kinds[o] == CodegenExtraOutputKind.ArgMaxIndex)
                {
                    var indices = new int[count];
                    outputBuffers[o].Download<int>(indices);
                    for (long e = 0; e < count; e++)
                        worst = Math.Max(worst, Math.Abs(indices[e] - want[o][e]));
                    continue;
                }

                var got = new float[count];
                outputBuffers[o].Download<float>(got);
                for (long e = 0; e < count; e++)
                    worst = Math.Max(worst, Math.Abs(got[e] - want[o][e]));
            }

            bool ok = worst <= 1e-4;
            Console.WriteLine("{0,-42} {1,8} {2,12} {3,7}",
                label,
                outputBuffers.Count.ToString(CultureInfo.InvariantCulture),
                worst.ToString("0.000E+000", CultureInfo.InvariantCulture),
                ok ? "PASS" : "FAIL");
            return ok;
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-42} {1,8} {2,12} {3,7}   {4}",
                label, "-", "-", "ERROR", ex.Message.Replace("\n", " "));
            return false;
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }

    private static unsafe void Launch(
        DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks, uint blockThreads)
    {
        fixed (IntPtr* pinned = pointers)
        {
            void** argv = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) argv[i] = pinned + i;
            module.Launch(fn, blocks, 1, 1, blockThreads, 1, 1, 0, argv);
        }
    }

    private static double[] Values(long count, int salt)
    {
        var data = new double[count];
        for (long e = 0; e < count; e++) data[e] = ((((e * 37 + salt) % 97) - 48) / 16.0);
        return data;
    }

    private static IEnumerable<(string, CodegenKernelSpec, double[][])> Cases()
    {
        const int Count = 65536;
        var map = new[] { CodegenAffineExpr.Axis(0) };
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", Count));

        // SGD with momentum: two outputs, one of them derived from the other.
        {
            var v = new CodegenTensorBinding(0, "v", new[] { Count }, map);
            var g = new CodegenTensorBinding(1, "g", new[] { Count }, map);
            var p = new CodegenTensorBinding(2, "p", new[] { Count }, map);
            var vOut = new CodegenTensorBinding(3, "v_out", new[] { Count }, map, isOutput: true);
            var pOut = new CodegenTensorBinding(4, "p_out", new[] { Count }, map, isOutput: true);

            yield return ("momentum step (2 outputs)",
                new CodegenKernelSpec("momentum2", space, new[] { v, g, p }, vOut,
                    new[] { 0 }, CodegenReduceKind.None,
                    biasInput: 1, reduceScale: 0.9,
                    extraOutputs: new[]
                    {
                        new CodegenExtraOutput(pOut, CodegenExtraOutputKind.AffineOfPrimary,
                            Scale: -0.01, BiasInput: 2, BiasScale: 1.0),
                    }),
                new[] { Values(Count, 0), Values(Count, 5), Values(Count, 11) });
        }

        // FOUR outputs, which is what the old two-output cap made impossible.
        {
            var x = new CodegenTensorBinding(0, "x", new[] { Count }, map);
            var a = new CodegenTensorBinding(1, "a", new[] { Count }, map, isOutput: true);
            var b = new CodegenTensorBinding(2, "b", new[] { Count }, map, isOutput: true);
            var c = new CodegenTensorBinding(3, "c", new[] { Count }, map, isOutput: true);
            var d = new CodegenTensorBinding(4, "d", new[] { Count }, map, isOutput: true);

            yield return ("four outputs from one pass",
                new CodegenKernelSpec("four_out", space, new[] { x }, a,
                    new[] { 0 }, CodegenReduceKind.None,
                    extraOutputs: new[]
                    {
                        new CodegenExtraOutput(b, CodegenExtraOutputKind.AffineOfPrimary, Scale: 2.0),
                        new CodegenExtraOutput(c, CodegenExtraOutputKind.AffineOfPrimary, Scale: -1.0),
                        new CodegenExtraOutput(d, CodegenExtraOutputKind.AffineOfPrimary, Scale: 0.5),
                    }),
                new[] { Values(Count, 0) });
        }

        // An argmax and an affine extra together: the two kinds coexisting under a real
        // reduction, which is the max-pool-plus-epilogue shape.
        {
            const int Rows = 4096, Inner = 16;
            var reduceSpace = new CodegenIterationSpace(
                CodegenAxis.Parallel("i", Rows), CodegenAxis.Reduce("k", Inner));

            var x = new CodegenTensorBinding(0, "x", new[] { Rows, Inner },
                new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
            var values = new CodegenTensorBinding(1, "out", new[] { Rows }, map, isOutput: true);
            var indices = new CodegenTensorBinding(2, "idx", new[] { Rows }, map, isOutput: true);
            var scaled = new CodegenTensorBinding(3, "scaled", new[] { Rows }, map, isOutput: true);

            yield return ("max + argmax + scaled (3 outputs)",
                new CodegenKernelSpec("max3", reduceSpace, new[] { x }, values,
                    new[] { 0 }, CodegenReduceKind.Max,
                    extraOutputs: new[]
                    {
                        new CodegenExtraOutput(scaled, CodegenExtraOutputKind.AffineOfPrimary,
                            Scale: 3.0),
                    },
                    secondaryOutput: indices, secondaryIndexExpr: CodegenAffineExpr.Axis(1)),
                new[] { Values(Rows * (long)Inner, 3) });
        }
    }
}
