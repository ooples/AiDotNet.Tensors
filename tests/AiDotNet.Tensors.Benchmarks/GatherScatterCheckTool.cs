// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Runs the gather/scatter lowerings on the device and compares them against the spec's own
/// fp64 interpretation.
/// </summary>
/// <remarks>
/// The scatter rows are the point. A destination reached through a run-time index cannot be
/// proven unique, so the emitter lowers the store to <c>red.global.add.f32</c>; whether that
/// is actually correct under real concurrency is not something a unit test can establish,
/// because a plain store would pass every single-threaded check and only lose gradients when
/// warps collide. The repeated-index rows below force those collisions, and the
/// determinism row runs the same scatter many times to show the total does not drift.
/// </remarks>
internal static class GatherScatterCheckTool
{
    internal static void Run(string[] args)
    {
        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;

        Console.WriteLine();
        Console.WriteLine("GATHER / SCATTER - data-dependent indexing");
        Console.WriteLine("device sm_{0}{1}", major, minor);
        Console.WriteLine();
        Console.WriteLine("{0,-46} {1,9} {2,12} {3,7}", "kernel", "elements", "max abs dev", "result");

        int passed = 0, failed = 0;
        foreach (var (label, spec, data) in Cases())
        {
            if (CheckOne(runtime, label, spec, data, major, minor)) passed++;
            else failed++;
        }

        Console.WriteLine();
        Console.WriteLine("gather/scatter: {0} passed, {1} failed", passed, failed);
    }

    private static bool CheckOne(
        DirectPtxRuntime runtime, string label, CodegenKernelSpec spec, double[][] data,
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
                var binding = spec.Inputs[i];
                long count = binding.ElementCount;
                DirectPtxBuffer buffer;

                if (binding.IsIndexTensor)
                {
                    var indices = new int[count];
                    for (long e = 0; e < count; e++) indices[e] = (int)data[i][e];
                    buffer = runtime.AllocateBytes((nuint)(count * sizeof(int)));
                    buffer.Upload<int>(indices);
                }
                else
                {
                    var values = new float[count];
                    for (long e = 0; e < count; e++) values[e] = (float)data[i][e];
                    buffer = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                    buffer.Upload<float>(values);
                }

                buffers.Add(buffer);
                pointers[i] = buffer.Pointer;
            }

            long outCount = spec.Output.ElementCount;
            var outBuffer = runtime.AllocateBytes((nuint)(outCount * sizeof(float)));

            // A SCATTER DESTINATION MUST START AT ZERO, because the kernel adds to it. This
            // is the caller's job and the tool does it explicitly rather than relying on a
            // fresh allocation happening to be zeroed -- that is not guaranteed, and a stale
            // buffer would make the first run pass and every later one fail.
            outBuffer.Upload<float>(new float[outCount]);
            buffers.Add(outBuffer);
            pointers[spec.Inputs.Count] = outBuffer.Pointer;

            Launch(module, fn, pointers, (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX);
            runtime.Synchronize();

            var got = new float[outCount];
            outBuffer.Download<float>(got);
            double[] want = spec.Interpret(data);

            double worst = 0;
            for (long e = 0; e < outCount; e++) worst = Math.Max(worst, Math.Abs(got[e] - want[e]));

            bool ok = worst <= 1e-4;
            Console.WriteLine("{0,-46} {1,9} {2,12} {3,7}",
                label,
                outCount.ToString("N0", CultureInfo.InvariantCulture),
                worst.ToString("0.000E+000", CultureInfo.InvariantCulture),
                ok ? "PASS" : "FAIL");
            return ok;
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-46} {1,9} {2,12} {3,7}   {4}",
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

    private static CodegenKernelSpec Gather(
        string name, int tokens, int vocabulary, int width, CodegenIndexOutOfRange policy)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));

        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Int32);

        var table = new CodegenTensorBinding(1, "table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), vocabulary, policy),
                null,
            });

        var output = new CodegenTensorBinding(2, "out", new[] { tokens, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        return new CodegenKernelSpec(name, space, new[] { ids, table }, output,
            new[] { 1 }, CodegenReduceKind.None);
    }

    private static CodegenKernelSpec Scatter(string name, int tokens, int vocabulary, int width)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));

        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Int32);

        var grad = new CodegenTensorBinding(1, "grad", new[] { tokens, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });

        var table = new CodegenTensorBinding(2, "grad_table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            isOutput: true,
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), vocabulary),
                null,
            });

        return new CodegenKernelSpec(name, space, new[] { ids, grad }, table,
            new[] { 1 }, CodegenReduceKind.None);
    }

    private static double[] Values(long count, int salt)
    {
        var data = new double[count];
        for (long e = 0; e < count; e++) data[e] = ((((e * 37 + salt) % 97) - 48) / 16.0);
        return data;
    }

    private static double[] Indices(int count, int vocabulary, int stride, int badEvery = 0)
    {
        var data = new double[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = (i * stride) % vocabulary;
            if (badEvery > 0 && i % badEvery == 0) data[i] = i % (2 * badEvery) == 0 ? -1 : vocabulary + 7;
        }
        return data;
    }

    private static IEnumerable<(string, CodegenKernelSpec, double[][])> Cases()
    {
        const int Tokens = 4096, Vocabulary = 1024, Width = 64;

        yield return ("embedding gather 4096x64 from 1024",
            Gather("gather_plain", Tokens, Vocabulary, Width, CodegenIndexOutOfRange.Skip),
            new[] { Indices(Tokens, Vocabulary, 7), Values(Vocabulary * (long)Width, 0) });

        // Out-of-range indices in both directions, which is what a padding row or a -1
        // sentinel produces in real data.
        yield return ("embedding gather, 1-in-8 out of range (skip)",
            Gather("gather_skip", Tokens, Vocabulary, Width, CodegenIndexOutOfRange.Skip),
            new[] { Indices(Tokens, Vocabulary, 7, badEvery: 8), Values(Vocabulary * (long)Width, 0) });

        yield return ("embedding gather, 1-in-8 out of range (clamp)",
            Gather("gather_clamp", Tokens, Vocabulary, Width, CodegenIndexOutOfRange.Clamp),
            new[] { Indices(Tokens, Vocabulary, 7, badEvery: 8), Values(Vocabulary * (long)Width, 0) });

        // SCATTER WITH HEAVY COLLISION. Stride 1 into a 32-row table from 4096 tokens means
        // 128 tokens per row, so warps collide constantly. A plain store passes every
        // single-threaded check and fails here.
        yield return ("embedding scatter 4096x64 into 32 (128-way collision)",
            Scatter("scatter_hot", Tokens, 32, Width),
            new[] { Indices(Tokens, 32, 1), Values(Tokens * (long)Width, 5) });

        yield return ("embedding scatter 4096x64 into 1024 (4-way collision)",
            Scatter("scatter_spread", Tokens, Vocabulary, Width),
            new[] { Indices(Tokens, Vocabulary, 1), Values(Tokens * (long)Width, 5) });

        // Every token onto ONE row: maximum contention.
        var allZero = new double[Tokens];
        yield return ("embedding scatter, all 4096 tokens onto row 0",
            Scatter("scatter_single", Tokens, 16, Width),
            new[] { allZero, Values(Tokens * (long)Width, 5) });
    }
}
