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
/// Assembles, verifies and times the tensor-core lowering against the scalar one.
/// </summary>
/// <remarks>
/// <para>
/// Emitting plausible PTX proves nothing: <c>wmma</c> is warp-collective, so the failure
/// mode is not a compile error but a kernel that runs, produces numbers of the right
/// magnitude, and is wrong. Every row here goes through <c>ptxas</c> (via LoadModule) and
/// then through the device, and is compared against the spec's own fp64 interpretation over
/// the SAME quantised operands the device reads.
/// </para>
/// <para>
/// The timing column is the reason the path exists. The blueprint records dense GEMM as
/// unwinnable at 0.33-0.65x -- but that was measured with the generated kernel on the FP32
/// pipes against a competitor on the tensor cores. This tool measures our two lowerings
/// against each other, so the number says what the tensor cores are worth in OUR emitter,
/// independent of any competitor's tuning.
/// </para>
/// </remarks>
internal static class TensorCoreCheckTool
{
    internal static void Run(string[] args)
    {
        bool timed = Array.IndexOf(args, "--no-time") < 0;

        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;

        Console.WriteLine();
        Console.WriteLine("TENSOR CORES - wmma m16n16k16, fp16 operands, fp32 accumulate");
        Console.WriteLine("device sm_{0}{1}", major, minor);
        Console.WriteLine();
        Console.WriteLine(
            "{0,-40} {1,9} {2,12} {3,7} {4,10} {5,10} {6,8} {7,9}",
            "kernel", "elements", "max rel dev", "result", "wmma", "scalar", "speedup",
            "TFLOP/s");

        int passed = 0, failed = 0;
        foreach (var (label, spec, verify) in Cases())
        {
            if (!PtxTensorCoreEmitter.TryPlan(spec, major, minor, out _, out string why))
            {
                Console.WriteLine("{0,-40} {1,9} {2,12} {3,7}   {4}",
                    label, "-", "-", "SKIP", why);
                continue;
            }

            if (CheckOne(runtime, label, spec, major, minor, timed, verify)) passed++;
            else failed++;
        }

        Console.WriteLine();
        Console.WriteLine("tensor cores: {0} passed, {1} failed", passed, failed);

        // A REJECTION IS ALSO A RESULT. The recogniser deciding "not eligible" silently is
        // indistinguishable from the tensor cores never helping, so the refusals are printed
        // with their reasons.
        Console.WriteLine();
        Console.WriteLine("recogniser refusals (each falls back to the scalar emitter):");
        foreach (var (label, spec) in IneligibleCases())
        {
            PtxTensorCoreEmitter.TryPlan(spec, major, minor, out _, out string why);
            Console.WriteLine("  {0,-34} {1}", label, why);
        }
    }

    private static bool CheckOne(
        DirectPtxRuntime runtime, string label, CodegenKernelSpec spec,
        int major, int minor, bool timed, bool verify)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var emitter = new PtxTensorCoreEmitter();
            string ptx = emitter.Emit(spec, major, minor);

            // ptxas is the first real gate: a malformed fragment list or a wmma variant this
            // architecture lacks fails here rather than at a wrong answer.
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var wide = new double[spec.Inputs.Count][];
            var pointers = new IntPtr[spec.ParameterCount];

            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                long count = spec.Inputs[i].ElementCount;
                var values = new double[count];
                var bits = new ushort[count];

                for (long e = 0; e < count; e++)
                {
                    // Dyadic values, so fp16 holds them exactly and the comparison measures
                    // the kernel rather than the operands' rounding.
                    double v = ((((e * 37 + i * 11) % 65) - 32) / 8.0);
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
            pointers[spec.Inputs.Count] = outBuffer.Pointer;

            PtxTensorCoreEmitter.TryPlan(spec, major, minor, out var plan, out _);
            uint blocks = (uint)emitter.BlockCount(plan!);
            uint blockThreads = (uint)emitter.BlockThreads;

            Launch(module, fn, pointers, blocks, blockThreads);
            runtime.Synchronize();

            // THE ORACLE IS O(M*N*K) ON THE CPU. At 4096^3 that is 68 billion fp64 MACs in
            // a scalar loop -- hours, with the GPU sitting idle, which is exactly what the
            // first attempt at these shapes did. Correctness is established on the shapes
            // below 1024^3, and those exercise every path the large ones use: unrolled and
            // looped K, both B layouts, and every epilogue. The large rows exist only to lift
            // the timing above the launch-overhead floor.
            double deviation = 0;
            if (verify)
            {
                var got = new float[outCount];
                outBuffer.Download<float>(got);
                double[] want = spec.Interpret(wide);

                double worst = 0, scale = 0;
                for (long e = 0; e < outCount; e++)
                {
                    worst = Math.Max(worst, Math.Abs(got[e] - want[e]));
                    scale = Math.Max(scale, Math.Abs(want[e]));
                }
                deviation = scale > 0 ? worst / scale : worst;
            }

            // The tensor cores accumulate in fp32 but their internal ordering is fixed by the
            // hardware and differs from the oracle's sequential walk, so the bound scales
            // with the contraction length exactly as it does on the scalar path.
            double tolerance = Math.Max(1e-6, plan!.K * 1.2e-7);
            bool ok = !verify || deviation <= tolerance;

            long macs = (long)plan.M * plan.N * plan.K;
            double wmmaUs = 0, scalarUs = 0;
            if (timed && ok)
            {
                wmmaUs = TimeIt(runtime, module, fn, pointers, blocks, blockThreads, macs);
                scalarUs = TimeScalar(runtime, spec, major, minor, pointers, macs);
            }

            Console.WriteLine(
                "{0,-40} {1,9} {2,12} {3,7} {4,10} {5,10} {6,8} {7,9}",
                label,
                outCount.ToString("N0", CultureInfo.InvariantCulture),
                verify ? deviation.ToString("0.000E+000", CultureInfo.InvariantCulture) : "timing",
                ok ? (verify ? "PASS" : "-") : "FAIL",
                wmmaUs > 0 ? wmmaUs.ToString("0.0", CultureInfo.InvariantCulture) + " us" : "-",
                scalarUs > 0 ? scalarUs.ToString("0.0", CultureInfo.InvariantCulture) + " us" : "-",
                (wmmaUs > 0 && scalarUs > 0)
                    ? (scalarUs / wmmaUs).ToString("0.00", CultureInfo.InvariantCulture) + "x"
                    : "-",
                wmmaUs > 0
                    ? (2.0 * macs / wmmaUs / 1e6).ToString("0.0", CultureInfo.InvariantCulture)
                    : "-");

            return ok;
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-40} {1,9} {2,12} {3,7}   {4}",
                label, "-", "-", "ERROR", ex.Message.Replace("\n", " "));
            return false;
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }

    /// <summary>
    /// Times the SAME spec through the scalar emitter, which is the honest denominator: it
    /// is what this repository would otherwise have shipped for this operator.
    /// </summary>
    private static double TimeScalar(
        DirectPtxRuntime runtime, CodegenKernelSpec spec, int major, int minor,
        IntPtr[] pointers, long macs)
    {
        // THE SCALAR LOWERING IS CAPPED. Past about 1024^3 it is minutes per row and adds
        // nothing: its ratio is already established at the smaller sizes, and the denominator
        // that decides whether this path is worth shipping is cuBLAS, not our own slower
        // kernel. Timing it at 4096^3 buys a number nobody would act on.
        const long ScalarMacLimit = 1L << 30;
        if (macs > ScalarMacLimit) return 0;

        try
        {
            var scalar = new PtxAffineEmitter();
            string ptx = scalar.Emit(spec, major, minor);
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(spec.Name, out _);
            return TimeIt(runtime, module, fn, pointers,
                (uint)scalar.LaunchBlocks, (uint)scalar.LaunchBlockX, macs);
        }
        catch
        {
            return 0;   // the scalar path may not accept fp16 bindings at this shape
        }
    }

    private static double TimeIt(
        DirectPtxRuntime runtime, DirectPtxModule module, IntPtr fn,
        IntPtr[] pointers, uint blocks, uint blockThreads, long macs)
    {
        // ITERATIONS SCALE WITH THE WORK. A fixed 200 is right at 512^3 and absurd at
        // 4096^3, where the scalar lowering alone is over 100ms a launch -- the run stopped
        // making progress rather than producing a slow number. The floor of 5 keeps the
        // large shapes averaged over enough launches to be stable.
        int iterations = (int)Math.Max(5, Math.Min(200, 20_000_000_000L / Math.Max(1, macs)));
        int warmup = Math.Max(2, iterations / 10);

        for (int i = 0; i < warmup; i++) Launch(module, fn, pointers, blocks, blockThreads);
        runtime.Synchronize();

        // Best of three, matching the protocol the rest of the campaign uses: the minimum is
        // the least contaminated sample, not the most flattering one.
        double best = double.MaxValue;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) Launch(module, fn, pointers, blocks, blockThreads);
            runtime.Synchronize();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / iterations);
        }
        return best;
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

    /// <summary>A fp16 x fp16 -> fp32 matmul, optionally with a fused epilogue.</summary>
    private static CodegenKernelSpec MatMul(
        string name, int m, int k, int n,
        CodegenActivationKind activation = CodegenActivationKind.None,
        bool transposeB = false)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", m), CodegenAxis.Parallel("n", n),
            CodegenAxis.Reduce("k", k));

        var a = new CodegenTensorBinding(0, "a", new[] { m, k },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) },
            elementType: CodegenElementType.Float16);

        var b = transposeB
            ? new CodegenTensorBinding(1, "b", new[] { n, k },
                new[] { CodegenAffineExpr.Axis(1), CodegenAffineExpr.Axis(2) },
                elementType: CodegenElementType.Float16)
            : new CodegenTensorBinding(1, "b", new[] { k, n },
                new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) },
                elementType: CodegenElementType.Float16);

        var output = new CodegenTensorBinding(2, "out", new[] { m, n },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        return new CodegenKernelSpec(name, space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum, activation: activation);
    }

    private static IEnumerable<(string, CodegenKernelSpec, bool)> Cases()
    {
        yield return ("matmul 64x64x64", MatMul("tc_matmul_64", 64, 64, 64), true);
        yield return ("matmul 256x256x256", MatMul("tc_matmul_256", 256, 256, 256), true);
        yield return ("matmul 512x512x512", MatMul("tc_matmul_512", 512, 512, 512), true);

        // A K longer than the unroll limit, so the runtime K loop is exercised on device
        // rather than only in the fully-unrolled form.
        yield return ("matmul 256x2048x256 (looped K)",
            MatMul("tc_matmul_longk", 256, 2048, 256), true);

        // B transposed -- the [N, K] layout a linear layer's weights actually have.
        yield return ("matmul 256x256x256, B transposed",
            MatMul("tc_matmul_bt", 256, 256, 256, transposeB: true), true);

        // THE FUSED EPILOGUE, which is the only structural advantage over cuBLAS.
        yield return ("matmul 512x512x512 + relu",
            MatMul("tc_matmul_relu", 512, 512, 512, CodegenActivationKind.ReLU), true);
        yield return ("matmul 512x512x512 + gelu",
            MatMul("tc_matmul_gelu", 512, 512, 512, CodegenActivationKind.Gelu), true);

        // TIMING ONLY. Below roughly 1024^3 both our harness and the competitor's sit on a
        // launch-submission floor of tens of microseconds under WDDM, which swamps the
        // kernel -- the first attempt at this had 64^3 and 1024^3 costing the same, and had
        // the competitor's fp16->fp32 lane, which does strictly MORE work, coming out faster
        // than fp16->fp16 at every size. These are the shapes a ratio can be read from.
        yield return ("matmul 1024x1024x1024", MatMul("tc_matmul_1024", 1024, 1024, 1024), false);
        yield return ("matmul 2048x2048x2048", MatMul("tc_matmul_2048", 2048, 2048, 2048), false);
        yield return ("matmul 4096x4096x4096", MatMul("tc_matmul_4096", 4096, 4096, 4096), false);
    }

    /// <summary>Specs the recogniser must refuse, each for a different reason.</summary>
    private static IEnumerable<(string, CodegenKernelSpec)> IneligibleCases()
    {
        yield return ("fp32 operands", Fp32MatMul());
        yield return ("K not a multiple of 16", MatMul("tc_bad_k", 64, 24, 64));
        yield return ("M not a multiple of 16", MatMul("tc_bad_m", 40, 64, 64));
    }

    private static CodegenKernelSpec Fp32MatMul()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", 64), CodegenAxis.Parallel("n", 64),
            CodegenAxis.Reduce("k", 64));

        var a = new CodegenTensorBinding(0, "a", new[] { 64, 64 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) });
        var b = new CodegenTensorBinding(1, "b", new[] { 64, 64 },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(2, "out", new[] { 64, 64 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        return new CodegenKernelSpec("tc_fp32", space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum);
    }
}
