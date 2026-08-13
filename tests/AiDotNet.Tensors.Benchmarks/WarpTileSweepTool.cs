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
/// Sweeps the per-warp register tile of the staged tensor-core GEMM, verifying and timing
/// each candidate.
/// </summary>
/// <remarks>
/// <para>
/// The profile said the staged kernel is bound by shared-memory reads -- L1TEX at 87.14%
/// with the tensor pipe at 25.29% -- and the warp tile is the knob that moves that ratio: a
/// warp loads <c>M + N</c> fragments and issues <c>M * N</c> mma instructions from them, so
/// 2x2 is one fragment load per mma and 4x4 is half that.
/// </para>
/// <para>
/// It is SWEPT rather than chosen because the accumulators are <c>M * N * 8</c> fp32 per
/// thread, so a large tile trades shared traffic for registers, and past some point ptxas
/// spills -- at which point a big tile is slower than a small one. Two levers on this branch
/// were predicted from a model and did not pay; the blueprint's rule is that lowerings are
/// picked by measurement.
/// </para>
/// <para>
/// Every candidate is VERIFIED before it is timed. A tile that computes the wrong answer
/// quickly is not a candidate, and an indexing slip in the store -- the column offset using
/// the row tile count, say -- is invisible at 2x2 and wrong at every asymmetric shape.
/// </para>
/// </remarks>
internal static class WarpTileSweepTool
{
    private static readonly (int M, int N)[] Candidates =
    {
        (2, 2),        // the shipped tile: 64x64 block, 1.00 fragment loads per mma
        (2, 4),        // 64x128 block, 0.75
        (4, 2),        // 128x64 block, 0.75
        (4, 4),        // 128x128 block, 0.50 -- and 128 accumulator registers per thread
    };

    /// <summary>Staging forms, swept alongside the tile so the comparison is paired.</summary>
    private static readonly bool[] AsyncForms = { true, false };

    internal static void Run(string[] args)
    {
        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;

        Console.WriteLine();
        Console.WriteLine("WARP TILE SWEEP - staged tensor-core GEMM");
        Console.WriteLine("device sm_{0}{1}", major, minor);
        Console.WriteLine();
        Console.WriteLine(
            "{0,-16} {1,7} {2,10} {3,7} {4,7} {5,9} {6,9} {7,12} {8,10} {9,9}",
            "shape", "tile", "staging", "acc reg", "ld/mma", "shared B", "reference",
            "max abs dev", "us", "TFLOP/s");

        // A single tile can be selected so a profiler can attribute counters to it: the sweep
        // otherwise launches the 2x2 reference between candidates, and every launch of the
        // same kernel name looks alike to ncu.
        string? only = args.Length > 0 && args[0].Contains('x') ? args[0] : null;
        string? shapeOnly = args.Length > 1 ? args[1] : null;
        foreach (var (label, spec, m, n, k) in Shapes())
        {
            if (shapeOnly is not null && !label.StartsWith(shapeOnly, StringComparison.Ordinal)) continue;

            double baseline = 0;
            foreach (var (tm, tn) in Candidates)
            {
                if (only is not null && only != tm + "x" + tn) continue;
                foreach (bool async in AsyncForms)
                {
                    double us = Measure(runtime, spec, major, minor, tm, tn, async,
                                        m, n, k, label, ref baseline);
                    if (us > 0 && baseline == 0) baseline = us;
                }

                // THE CEILING: the same tile and the same mma instructions with the fragment
                // loads hoisted out of the K loop. It computes the wrong answer on purpose --
                // it exists to bound what this instruction mix can reach with memory traffic
                // removed, so progress is measured against a ceiling rather than against a
                // competitor.
                MeasureCeiling(runtime, spec, major, minor, tm, tn, m, n, k, label);
            }
            Console.WriteLine();
        }
    }

    private static double Measure(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, int major, int minor,
        int tileM, int tileN, bool async, int m, int n, int k, string label,
        ref double baseline)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var emitter = new PtxTensorCoreEmitter
            {
                WarpTilesM = tileM, WarpTilesN = tileN, PinWarpTile = true, EnableAsyncCopy = async,
            };

            if (!PtxTensorCoreEmitter.TryPlan(spec, major, minor, out var plan, out string why))
            {
                Report(label, tileM, tileN, emitter, "-", 0, 0, 0, "not a wmma shape: " + why);
                return 0;
            }

            if (!emitter.CanStage(plan!, out string stageWhy))
            {
                Report(label, tileM, tileN, emitter, "-", 0, 0, 0, stageWhy);
                return 0;
            }

            string ptx = emitter.Emit(spec, major, minor);
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var pointers = new IntPtr[spec.ParameterCount];
            var wide = new double[2][];
            for (int i = 0; i < 2; i++)
            {
                long count = spec.Inputs[i].ElementCount;
                var values = new double[count];
                var bits = new ushort[count];
                for (long e = 0; e < count; e++)
                {
                    double v = i == 0
                        ? (((e * 37) % 65) - 32) / 8.0
                        : ReductionFactor((int)(e / n)) * ColumnFactor((int)(e % n));
                    var half = (Half)(float)v;
                    bits[e] = BitConverter.HalfToUInt16Bits(half);
                    values[e] = (float)half;
                }
                var buffer = runtime.AllocateBytes((nuint)(count * sizeof(ushort)));
                buffer.Upload<ushort>(bits);
                buffers.Add(buffer);
                pointers[i] = buffer.Pointer;
                wide[i] = values;
            }

            long outCount = spec.Output.ElementCount;
            var outBuffer = runtime.AllocateBytes((nuint)(outCount * sizeof(float)));
            buffers.Add(outBuffer);
            pointers[2] = outBuffer.Pointer;

            uint blocks = (uint)emitter.BlockCount(plan!);
            uint threads = (uint)emitter.BlockThreads;

            DirectPtxLaunchHelper.Launch(module, fn, pointers, blocks, threads);
            runtime.Synchronize();

            // CORRECTNESS FIRST. Every row is checked against an independent fp64 CPU oracle.
            // The dense right operand is constructed as reductionFactor[k] * columnFactor[n],
            // with exactly representable fp16 factors. This preserves representative dense
            // GPU work while reducing the oracle from O(M*N*K) to O(M*K + M*N).
            double deviation;
            var got = new float[outCount];
            outBuffer.Download<float>(got);
            float[] want = RankOneMatMulReference(wide[0], m, n, k);
            bool agrees = CodegenOutputAgreement.Agrees(
                got, want, 1e-3, out deviation, out _, out _, out _);

            if (!agrees)
            {
                Report(label, tileM, tileN, emitter, "fp64", deviation, 0, 0, "WRONG");
                return 0;
            }

            long macs = (long)m * n * k;
            double us = TimeIt(runtime, module, fn, pointers, blocks, threads, macs);
            Report(label, tileM, tileN, emitter, "fp64", deviation, us, macs, null);
            return us;
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-16} {1,7} {2,10} {3,7} {4,7} {5,9} {6,9} {7,12} {8,10} {9,9}  {10}",
                label, tileM + "x" + tileN, "-", "-", "-", "-", "-", "-", "-", "-",
                ex.Message.Replace("\n", " "));
            return 0;
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }

    private static float[] RankOneMatMulReference(double[] left, int m, int n, int k)
    {
        var rowSums = new double[m];
        for (int row = 0; row < m; row++)
        {
            double sum = 0;
            long rowOffset = (long)row * k;
            for (int reduction = 0; reduction < k; reduction++)
                sum += left[rowOffset + reduction] * ReductionFactor(reduction);
            rowSums[row] = sum;
        }

        var result = new float[(long)m * n];
        for (int row = 0; row < m; row++)
        {
            long rowOffset = (long)row * n;
            for (int column = 0; column < n; column++)
                result[rowOffset + column] = (float)(rowSums[row] * ColumnFactor(column));
        }
        return result;
    }

    private static double ReductionFactor(int reduction) => (reduction % 6) switch
    {
        0 => 0.5,
        1 => 1.0,
        2 => 2.0,
        3 => -0.5,
        4 => -1.0,
        _ => -2.0,
    };

    private static double ColumnFactor(int column)
    {
        double magnitude = ((column * 29) % 63 + 1) / 16.0;
        return (column & 1) == 0 ? magnitude : -magnitude;
    }

    private static void Report(
        string label, int tileM, int tileN, PtxTensorCoreEmitter emitter,
        string correctnessReference, double deviation, double us, long macs, string? note)
    {
        int accRegisters = tileM * tileN * 8;
        double loadsPerMma = (tileM + tileN) / (double)(tileM * tileN);

        Console.WriteLine(
            "{0,-16} {1,7} {2,10} {3,7} {4,7} {5,9} {6,9} {7,12} {8,10} {9,9}{10}",
            label,
            tileM + "x" + tileN,
            emitter.AsyncCopy ? "cp.async" : "registers",
            accRegisters.ToString(CultureInfo.InvariantCulture),
            loadsPerMma.ToString("0.00", CultureInfo.InvariantCulture),
            note is null ? emitter.SharedMemoryBytes.ToString(CultureInfo.InvariantCulture) : "-",
            correctnessReference,
            note is null ? deviation.ToString("0.000E+000", CultureInfo.InvariantCulture) : "-",
            us > 0 ? us.ToString("0.0", CultureInfo.InvariantCulture) + " us" : "-",
            (us > 0 && macs > 0)
                ? (2.0 * macs / us / 1e6).ToString("0.0", CultureInfo.InvariantCulture) : "-",
            note is null ? string.Empty : "  " + note);
    }

    private static double TimeIt(
        DirectPtxRuntime runtime, DirectPtxModule module, IntPtr fn,
        IntPtr[] pointers, uint blocks, uint threads, long macs)
    {
        int iterations = (int)Math.Max(5, Math.Min(200, 20_000_000_000L / Math.Max(1, macs)));
        int warmup = Math.Max(2, iterations / 10);

        for (int i = 0; i < warmup; i++)
            DirectPtxLaunchHelper.Launch(module, fn, pointers, blocks, threads);
        runtime.Synchronize();

        double best = double.MaxValue;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                DirectPtxLaunchHelper.Launch(module, fn, pointers, blocks, threads);
            runtime.Synchronize();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / iterations);
        }
        return best;
    }

    /// <summary>Times the mma ceiling probe: same instructions, no loop-carried memory.</summary>
    private static void MeasureCeiling(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, int major, int minor,
        int tileM, int tileN, int m, int n, int k, string label)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var emitter = new PtxTensorCoreEmitter
            {
                WarpTilesM = tileM, WarpTilesN = tileN, PinWarpTile = true,
                MmaCeilingProbe = true,
            };

            PtxTensorCoreEmitter.TryPlan(spec, major, minor, out var plan, out _);
            if (plan is null || !emitter.CanStage(plan, out _)) return;

            string ptx = emitter.Emit(spec, major, minor);
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(emitter.EmittedEntryName, out _);

            var pointers = new IntPtr[spec.ParameterCount];
            for (int i = 0; i < 2; i++)
            {
                long count = spec.Inputs[i].ElementCount;
                var buffer = runtime.AllocateBytes((nuint)(count * sizeof(ushort)));
                var bits = new ushort[count];
                Array.Fill(bits, BitConverter.HalfToUInt16Bits((Half)1.0f));
                buffer.Upload<ushort>(bits);
                buffers.Add(buffer);
                pointers[i] = buffer.Pointer;
            }
            var outBuffer = runtime.AllocateBytes((nuint)(spec.Output.ElementCount * sizeof(float)));
            buffers.Add(outBuffer);
            pointers[2] = outBuffer.Pointer;

            long macs = (long)m * n * k;
            double us = TimeIt(runtime, module, fn, pointers,
                (uint)emitter.BlockCount(plan), (uint)emitter.BlockThreads, macs);

            Console.WriteLine(
                "{0,-16} {1,7} {2,10} {3,7} {4,7} {5,9} {6,9} {7,12} {8,10} {9,9}",
                label, tileM + "x" + tileN, "CEILING", "-", "0.00", "-", "-", "(no answer)",
                us.ToString("0.0", CultureInfo.InvariantCulture) + " us",
                (2.0 * macs / us / 1e6).ToString("0.0", CultureInfo.InvariantCulture));
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-16} {1,7} {2,10}  ceiling failed: {3}",
                label, tileM + "x" + tileN, "CEILING", ex.Message.Replace('\n', ' '));
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }

    private static CodegenKernelSpec MatMul(string name, int m, int k, int n)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", m), CodegenAxis.Parallel("n", n),
            CodegenAxis.Reduce("k", k));

        var a = new CodegenTensorBinding(0, "a", new[] { m, k },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) },
            elementType: CodegenElementType.Float16);
        var b = new CodegenTensorBinding(1, "b", new[] { k, n },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) },
            elementType: CodegenElementType.Float16);
        var output = new CodegenTensorBinding(2, "out", new[] { m, n },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        return new CodegenKernelSpec(name, space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum);
    }

    private static IEnumerable<(string, CodegenKernelSpec, int, int, int)> Shapes()
    {
        // 256 is divisible by 128, so every rung of the warp-tile ladder (including 4x4,
        // which needs M % 128 == 0) can stage it.
        yield return ("256^3", MatMul("sweep_256", 256, 256, 256), 256, 256, 256);
        yield return ("512^3", MatMul("sweep_512", 512, 512, 512), 512, 512, 512);
        yield return ("1024^3", MatMul("sweep_1024", 1024, 1024, 1024), 1024, 1024, 1024);
        yield return ("2048^3", MatMul("sweep_2048", 2048, 2048, 2048), 2048, 2048, 2048);
        // 256^3 exists so at least ONE shape is checked against the fp64 ORACLE rather than
        // against another GPU lowering. The oracle branch requires M*N*K <= 64Mi, and every other
        // shape here is at least 512^3 = 128Mi, so without this the 2x2 tile was assigned
        // deviation 0 by fiat ("this IS the reference tile") and every larger tile was compared
        // only against that unverified reference - meaning a defect shared by all staged
        // lowerings could be timed and reported as verified. 256 is divisible by 128, so every
        // rung of the warp-tile ladder (including 4x4, which needs M % 128 == 0) can stage it.
        yield return ("256^3", MatMul("sweep_256", 256, 256, 256), 256, 256, 256);
        yield return ("4096^3", MatMul("sweep_4096", 4096, 4096, 4096), 4096, 4096, 4096);
    }
}
