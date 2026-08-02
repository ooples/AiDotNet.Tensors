// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Runs complex and quaternion kernels on the device against the spec's fp64 interpretation.
/// </summary>
/// <remarks>
/// The unit tests check the multiplication tables against the defining relations, which is
/// what catches a wrong sign. This checks something different and equally necessary: that the
/// EMITTED code applies those tables to the right registers. A transposed operand pair in the
/// emitter would still satisfy every algebraic identity -- it would just compute <c>ba</c>
/// where <c>ab</c> was asked for, which for quaternions is a different value.
/// </remarks>
internal static class AlgebraCheckTool
{
    internal static void Run(string[] args)
    {
        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;

        Console.WriteLine();
        Console.WriteLine("COMPLEX / QUATERNION - algebra arithmetic");
        Console.WriteLine("device sm_{0}{1}", major, minor);
        Console.WriteLine();
        Console.WriteLine("{0,-44} {1,9} {2,12} {3,7}", "kernel", "floats", "max abs dev", "result");

        int passed = 0, failed = 0;
        foreach (var (label, spec) in Cases())
        {
            if (CheckOne(runtime, label, spec, major, minor)) passed++;
            else failed++;
        }

        Console.WriteLine();
        Console.WriteLine("algebra: {0} passed, {1} failed", passed, failed);
    }

    private static bool CheckOne(
        DirectPtxRuntime runtime, string label, CodegenKernelSpec spec, int major, int minor)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            int components = spec.Algebra.Components();
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, major, minor);
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var pointers = new IntPtr[spec.ParameterCount];
            var hostData = new double[spec.Inputs.Count][];

            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                long floats = spec.Inputs[i].ElementCount * components;
                var wide = new double[floats];
                var narrow = new float[floats];
                for (long e = 0; e < floats; e++)
                {
                    // Dyadic, so fp32 holds every intermediate exactly and the comparison
                    // measures the kernel rather than accumulated rounding.
                    double v = ((((e * 37 + i * 11) % 65) - 32) / 8.0);
                    wide[e] = v;
                    narrow[e] = (float)v;
                }
                var buffer = runtime.AllocateBytes((nuint)(floats * sizeof(float)));
                buffer.Upload<float>(narrow);
                buffers.Add(buffer);
                pointers[i] = buffer.Pointer;
                hostData[i] = wide;
            }

            long outFloats = spec.Output.ElementCount * components;
            var outBuffer = runtime.AllocateBytes((nuint)(outFloats * sizeof(float)));
            outBuffer.Upload<float>(new float[outFloats]);
            buffers.Add(outBuffer);
            pointers[spec.Inputs.Count] = outBuffer.Pointer;

            Launch(module, fn, pointers, (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX);
            runtime.Synchronize();

            var got = new float[outFloats];
            outBuffer.Download<float>(got);
            double[] want = spec.Interpret(hostData);

            double worst = 0;
            for (long e = 0; e < outFloats; e++) worst = Math.Max(worst, Math.Abs(got[e] - want[e]));

            bool ok = worst <= 1e-4;
            Console.WriteLine("{0,-44} {1,9} {2,12} {3,7}",
                label,
                outFloats.ToString("N0", CultureInfo.InvariantCulture),
                worst.ToString("0.000E+000", CultureInfo.InvariantCulture),
                ok ? "PASS" : "FAIL");
            return ok;
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-44} {1,9} {2,12} {3,7}   {4}",
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

    private static CodegenKernelSpec Elementwise(string name, CodegenAlgebra algebra, int count)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var a = new CodegenTensorBinding(0, "a", new[] { count }, new[] { CodegenAffineExpr.Axis(0) });
        var b = new CodegenTensorBinding(1, "b", new[] { count }, new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(2, "out", new[] { count },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        return new CodegenKernelSpec(name, space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.None, algebra: algebra);
    }

    private static CodegenKernelSpec MatVec(string name, CodegenAlgebra algebra, int rows, int inner)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Reduce("k", inner));

        var m = new CodegenTensorBinding(0, "m", new[] { rows, inner },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var v = new CodegenTensorBinding(1, "v", new[] { inner },
            new[] { CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(2, "out", new[] { rows },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        return new CodegenKernelSpec(name, space, new[] { m, v }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum, algebra: algebra);
    }

    private static IEnumerable<(string, CodegenKernelSpec)> Cases()
    {
        yield return ("complex elementwise product 65536",
            Elementwise("complex_mul", CodegenAlgebra.Complex, 65536));
        yield return ("quaternion elementwise product 65536",
            Elementwise("quat_mul", CodegenAlgebra.Quaternion, 65536));

        yield return ("complex mat-vec 1024x64",
            MatVec("complex_matvec", CodegenAlgebra.Complex, 1024, 64));

        // Quaternion contraction: the non-commutative case under a reduction, which is where
        // an operand-order slip in the emitter would show up.
        yield return ("quaternion mat-vec 1024x64",
            MatVec("quat_matvec", CodegenAlgebra.Quaternion, 1024, 64));

        yield return ("complex mat-vec 256x256 (longer contraction)",
            MatVec("complex_matvec_long", CodegenAlgebra.Complex, 256, 256));
    }
}
