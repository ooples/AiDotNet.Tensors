// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). The parity engine + engines fixture + report.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Shared CPU + GPU engines and the parity report, created once for the whole OpParity collection.
/// Forces STRICT fp32 on the GPU (TF32 off) so a parity mismatch means a real logic/numerics bug,
/// not TF32's ~10-bit mantissa (which would legitimately differ ~1e-3 from true fp32 and mask bugs
/// — PyTorch's CUDA-vs-CPU correctness tests disable TF32 identically). Writes a max-ULP-per-op
/// summary on dispose so drift that is within-tolerance-but-worsening is still visible.
/// </summary>
public sealed class OpParityFixture : IDisposable
{
    public CpuEngine Cpu { get; }
    public DirectGpuTensorEngine? Gpu { get; private set; }
    public bool GpuReady { get; }
    public Exception? GpuInitError { get; }

    private readonly List<string> _report = new();
    private readonly object _lock = new();
    private readonly bool _prevAllowTF32;

    public OpParityFixture()
    {
        // Capture + restore (in Dispose) so this collection's strict-fp32 policy does not leak into
        // other test collections that may run afterwards and rely on the process-wide default.
        _prevAllowTF32 = CudaDispatchPolicy.AllowTF32;
        CudaDispatchPolicy.AllowTF32 = false;
        Cpu = new CpuEngine();
        Gpu = new DirectGpuTensorEngine();
        GpuReady = Gpu.IsGpuAvailable;
    }

    /// <summary>
    /// #775 driver-reset: disposes the current GPU engine and creates a fresh one, giving a brand-new
    /// OpenCL context + command queues + compiled kernels. This clears the AMD RX 5500 (RDNA1) driver
    /// state that accumulates across many kernel dispatches and, past a threshold, corrupts some later
    /// kernels (FusedConv3D / NativeMagnitudeAndPhase: correct in isolation, wrong only deep into a long
    /// full-suite run; neither a queue finish nor a queue recycle clears it — only a fresh context does,
    /// which is why the EARLY OpParityTests copy of an op passes while the LATE generated copy fails).
    /// The harness calls this every N ops so accumulated dispatches per context stay well under the
    /// threshold. Safe between ops: each op materializes its result to a host array before returning, so
    /// no live GPU buffer spans the swap. Program binaries are cached, so recreation is cheap.
    /// </summary>
    public void ResetGpuEngine()
    {
        if (!GpuReady) return;
        var old = Gpu;
        var fresh = new DirectGpuTensorEngine();
        Gpu = fresh;
        old?.Dispose();
    }

    /// <summary>Whether GPU parity must run (CI can force it so a missing GPU fails instead of skips).</summary>
    public bool RequireGpu =>
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal);

    public void Record(string line)
    {
        lock (_lock) _report.Add(line);
    }

    public void Dispose()
    {
        Gpu?.Dispose();
        CudaDispatchPolicy.AllowTF32 = _prevAllowTF32;
        lock (_lock)
        {
            if (_report.Count == 0) return;
            try
            {
                var dir = Environment.GetEnvironmentVariable("AIDOTNET_OPPARITY_REPORT_DIR")
                          ?? Path.Combine(Path.GetTempPath(), "aidotnet-opparity");
                Directory.CreateDirectory(dir);
                var path = Path.Combine(dir, "op-parity-report.tsv");
                var sb = new StringBuilder();
                sb.AppendLine(
                    "op\tcategory\tcpu_vs_gpu_ulp\tcpu_vs_gpu_abs\tcpu_vs_gpu_rel\tworst_index\tcpu_value\tgpu_value\t" +
                    "tol_ulp\ttol_abs\ttol_rel\tcpu_vs_oracle_ulp\tgpu_vs_oracle_ulp\tworse_engine");
                foreach (var l in _report.OrderBy(s => s, StringComparer.Ordinal)) sb.AppendLine(l);
                File.WriteAllText(path, sb.ToString());
            }
            catch { /* reporting is best-effort; never fail a run over it */ }
        }
    }
}

[CollectionDefinition("OpParity", DisableParallelization = true)]
public sealed class OpParityCollection : ICollectionFixture<OpParityFixture> { }

/// <summary>The parity checker. For one <see cref="OpCase"/> it runs the float op on CPU and GPU,
/// the double ORACLE on CPU, and asserts: (1) same-engine determinism (bit-exact re-run), (2) all
/// results finite, (3) CPU vs GPU within the op's ULP budget, and (4) each engine bounded against
/// the oracle — recording which engine drifts more (the localization the #775 ViT bug needs).</summary>
public static class OpParityHarness
{
    // #775: the AMD RX 5500 (RDNA1) OpenCL driver accumulates internal state across a large number of
    // kernel dispatches and, past a threshold, returns subtly wrong results for some later kernels
    // (FusedConv3D, NativeMagnitudeAndPhase: correct in isolation, wrong only deep into a long full-suite
    // run). Neither a queue finish nor a queue recycle clears it — only a fresh OpenCL context does. So
    // every N ops we recreate the GPU engine (fresh context), keeping accumulated dispatches per context
    // well under the corruption threshold (~one full legacy suite). Program binaries are cached, so the
    // recreation is cheap. See OpParityFixture.ResetGpuEngine.
    private static int _opsSinceEngineReset;
    private const int ResetEngineEveryNOps = 64;

    internal static void MaybeResetGpuEngine(OpParityFixture fx)
    {
        if (System.Threading.Interlocked.Increment(ref _opsSinceEngineReset) >= ResetEngineEveryNOps)
        {
            System.Threading.Interlocked.Exchange(ref _opsSinceEngineReset, 0);
            fx.ResetGpuEngine();
        }
    }

    public static void CheckForward(OpCase op, OpParityFixture fx)
    {
        if (!fx.GpuReady)
        {
            if (fx.RequireGpu)
                throw new InvalidOperationException(
                    $"{op.Name}: GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.", fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        if (op.GpuUnsafe)
        {
            fx.Record($"{op.Name}:forward\t{op.Category}\tGPU-UNSAFE\t-\t-\t-");
            Skip.If(true, $"GPU-UNSAFE ({op.Name} forward): {op.KnownDivergence ?? "GPU kernel crashes/poisons the host"}. GPU execution skipped so it can't crash the run.");
            return;
        }

        MaybeResetGpuEngine(fx); // may swap fx.Gpu for a fresh engine — fetch AFTER
        var gpu = fx.Gpu!;

        float[] cpuF = op.RunFloat(fx.Cpu).ToArray();
        float[] cpuF2 = op.RunFloat(fx.Cpu).ToArray();
        float[] gpuF = op.RunFloat(gpu).ToArray();
        float[] gpuF2 = op.RunFloat(gpu).ToArray();
        double[] oracleD = op.RunDouble(fx.Cpu).ToArray();

        AssertResults("forward", op, fx, cpuF, cpuF2, gpuF, gpuF2, oracleD, op.Fwd);
    }

    public static void CheckBackward(OpCase op, OpParityFixture fx)
    {
        if (!op.HasBackward) { Skip.If(true, $"{op.Name}: no backward registered."); return; }
        if (!fx.GpuReady)
        {
            if (fx.RequireGpu)
                throw new InvalidOperationException($"{op.Name}: GPU required but unavailable.", fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend available.");
            return;
        }

        if (op.GpuUnsafe)
        {
            fx.Record($"{op.Name}:backward\t{op.Category}\tGPU-UNSAFE\t-\t-\t-");
            Skip.If(true, $"GPU-UNSAFE ({op.Name} backward): {op.KnownDivergence ?? "GPU kernel crashes/poisons the host"}. GPU execution skipped.");
            return;
        }

        MaybeResetGpuEngine(fx); // may swap fx.Gpu for a fresh engine — fetch AFTER
        var gpu = fx.Gpu!;
        float[] cpuF = op.RunFloatGrad!(fx.Cpu).ToArray();
        float[] cpuF2 = op.RunFloatGrad!(fx.Cpu).ToArray();
        float[] gpuF = op.RunFloatGrad!(gpu).ToArray();
        float[] gpuF2 = op.RunFloatGrad!(gpu).ToArray();
        double[] oracleD = op.RunDoubleGrad!(fx.Cpu).ToArray();

        AssertResults("backward", op, fx, cpuF, cpuF2, gpuF, gpuF2, oracleD, op.BwdTol);
    }

    public static void CheckMultipleOutputs(OpCase op, OpParityFixture fx)
    {
        if (!op.HasMultipleOutputs)
        {
            Skip.If(true, $"{op.Name}: no multi-output contract registered.");
            return;
        }

        if (!fx.GpuReady)
        {
            if (fx.RequireGpu)
                throw new InvalidOperationException(
                    $"{op.Name}: GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.", fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        if (op.GpuUnsafe)
        {
            fx.Record($"{op.Name}:outputs\t{op.Category}\tGPU-UNSAFE\t-\t-\t-");
            Skip.If(true, $"GPU-UNSAFE ({op.Name} outputs): {op.KnownDivergence ?? "GPU kernel crashes/poisons the host"}. GPU execution skipped so it can't crash the run.");
            return;
        }

        MaybeResetGpuEngine(fx);
        var gpu = fx.Gpu!;

        Tensor<float>[]? cpu = null;
        Tensor<float>[]? cpuAgain = null;
        Tensor<float>[]? gpuResults = null;
        Tensor<float>[]? gpuAgain = null;
        Tensor<double>[]? oracle = null;
        try
        {
            cpu = op.RunFloatOutputs!(fx.Cpu);
            cpuAgain = op.RunFloatOutputs!(fx.Cpu);
            gpuResults = op.RunFloatOutputs!(gpu);
            gpuAgain = op.RunFloatOutputs!(gpu);
            oracle = op.RunDoubleOutputs!(fx.Cpu);

            Assert.True(cpu.Length > 1, $"{op.Name}: a multi-output contract must expose at least two outputs.");
            Assert.Equal(cpu.Length, cpuAgain.Length);
            Assert.Equal(cpu.Length, gpuResults.Length);
            Assert.Equal(cpu.Length, gpuAgain.Length);
            Assert.Equal(cpu.Length, oracle.Length);

            if (op.TensorOutputComparisons is { } outputComparisons)
                Assert.Equal(cpu.Length, outputComparisons.Count);

            for (int i = 0; i < cpu.Length; i++)
            {
                Assert.NotNull(cpu[i]);
                Assert.NotNull(cpuAgain[i]);
                Assert.NotNull(gpuResults[i]);
                Assert.NotNull(gpuAgain[i]);
                Assert.NotNull(oracle[i]);
                Assert.Equal(cpu[i].Shape.ToArray(), cpuAgain[i].Shape.ToArray());
                Assert.Equal(cpu[i].Shape.ToArray(), gpuResults[i].Shape.ToArray());
                Assert.Equal(cpu[i].Shape.ToArray(), gpuAgain[i].Shape.ToArray());
                Assert.Equal(cpu[i].Shape.ToArray(), oracle[i].Shape.ToArray());

                TensorOutputComparison comparison =
                    op.TensorOutputComparisons?[i] ?? TensorOutputComparison.Numeric;
                if (comparison.Kind == TensorOutputComparisonKind.WrappedRadians)
                {
                    AssertWrappedRadiansResults(
                        $"output[{i}]", op, fx,
                        cpu[i].ToArray(), cpuAgain[i].ToArray(),
                        gpuResults[i].ToArray(), gpuAgain[i].ToArray(),
                        oracle[i].ToArray(), comparison.AbsoluteTolerance);
                }
                else
                {
                    AssertResults(
                        $"output[{i}]", op, fx,
                        cpu[i].ToArray(), cpuAgain[i].ToArray(),
                        gpuResults[i].ToArray(), gpuAgain[i].ToArray(),
                        oracle[i].ToArray(), op.Fwd);
                }
            }
        }
        finally
        {
            DisposeTensors(cpu);
            DisposeTensors(cpuAgain);
            DisposeTensors(gpuResults);
            DisposeTensors(gpuAgain);
            DisposeTensors(oracle);
        }
    }

    public static void CheckHeterogeneousOutputs(OpCase op, OpParityFixture fx)
    {
        if (!op.HasHeterogeneousOutputs)
        {
            Skip.If(true, $"{op.Name}: no heterogeneous output contract registered.");
            return;
        }

        if (!fx.GpuReady)
        {
            if (fx.RequireGpu)
                throw new InvalidOperationException(
                    $"{op.Name}: GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.", fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        if (op.GpuUnsafe)
        {
            fx.Record($"{op.Name}:outputs\t{op.Category}\tGPU-UNSAFE\t-\t-\t-");
            Skip.If(true, $"GPU-UNSAFE ({op.Name} outputs): {op.KnownDivergence ?? "GPU kernel crashes/poisons the host"}.");
            return;
        }

        MaybeResetGpuEngine(fx);
        var gpu = fx.Gpu!;
        HeterogeneousTensorOutputs<float>? cpu = null;
        HeterogeneousTensorOutputs<float>? cpuAgain = null;
        HeterogeneousTensorOutputs<float>? gpuResults = null;
        HeterogeneousTensorOutputs<float>? gpuAgain = null;
        HeterogeneousTensorOutputs<double>? oracle = null;
        try
        {
            cpu = op.RunFloatHeterogeneousOutputs!(fx.Cpu);
            cpuAgain = op.RunFloatHeterogeneousOutputs!(fx.Cpu);
            gpuResults = op.RunFloatHeterogeneousOutputs!(gpu);
            gpuAgain = op.RunFloatHeterogeneousOutputs!(gpu);
            oracle = op.RunDoubleHeterogeneousOutputs!(fx.Cpu);

            Assert.True(cpu.Integers.Length + cpu.Booleans.Length > 0,
                $"{op.Name}: a heterogeneous contract must expose typed metadata outputs.");
            Assert.Equal(cpu.Numeric.Length, cpuAgain.Numeric.Length);
            Assert.Equal(cpu.Numeric.Length, gpuResults.Numeric.Length);
            Assert.Equal(cpu.Numeric.Length, gpuAgain.Numeric.Length);
            Assert.Equal(cpu.Numeric.Length, oracle.Numeric.Length);
            for (int i = 0; i < cpu.Numeric.Length; i++)
            {
                Assert.Equal(cpu.Numeric[i].Shape.ToArray(), cpuAgain.Numeric[i].Shape.ToArray());
                Assert.Equal(cpu.Numeric[i].Shape.ToArray(), gpuResults.Numeric[i].Shape.ToArray());
                Assert.Equal(cpu.Numeric[i].Shape.ToArray(), gpuAgain.Numeric[i].Shape.ToArray());
                Assert.Equal(cpu.Numeric[i].Shape.ToArray(), oracle.Numeric[i].Shape.ToArray());
                AssertResults(
                    $"numeric-output[{i}]", op, fx,
                    cpu.Numeric[i].ToArray(), cpuAgain.Numeric[i].ToArray(),
                    gpuResults.Numeric[i].ToArray(), gpuAgain.Numeric[i].ToArray(),
                    oracle.Numeric[i].ToArray(), op.Fwd);
            }

            AssertExactMetadata(op, "integer", cpu.Integers, cpuAgain.Integers, gpuResults.Integers, gpuAgain.Integers, oracle.Integers);
            AssertExactMetadata(op, "boolean", cpu.Booleans, cpuAgain.Booleans, gpuResults.Booleans, gpuAgain.Booleans, oracle.Booleans);
        }
        finally
        {
            DisposeOutputs(cpu);
            DisposeOutputs(cpuAgain);
            DisposeOutputs(gpuResults);
            DisposeOutputs(gpuAgain);
            DisposeOutputs(oracle);
        }
    }

    private static void AssertExactMetadata<TMetadata>(
        OpCase op,
        string kind,
        Tensor<TMetadata>[] cpu,
        Tensor<TMetadata>[] cpuAgain,
        Tensor<TMetadata>[] gpu,
        Tensor<TMetadata>[] gpuAgain,
        Tensor<TMetadata>[] oracle)
    {
        Assert.Equal(cpu.Length, cpuAgain.Length);
        Assert.Equal(cpu.Length, gpu.Length);
        Assert.Equal(cpu.Length, gpuAgain.Length);
        Assert.Equal(cpu.Length, oracle.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            int[] expectedShape = cpu[i].Shape.ToArray();
            Assert.True(expectedShape.SequenceEqual(cpuAgain[i].Shape.ToArray()),
                $"{op.Name} {kind}[{i}]: repeated CPU shape differs.");
            Assert.True(expectedShape.SequenceEqual(gpu[i].Shape.ToArray()),
                $"{op.Name} {kind}[{i}]: GPU shape differs.");
            Assert.True(expectedShape.SequenceEqual(gpuAgain[i].Shape.ToArray()),
                $"{op.Name} {kind}[{i}]: repeated GPU shape differs.");
            Assert.True(expectedShape.SequenceEqual(oracle[i].Shape.ToArray()),
                $"{op.Name} {kind}[{i}]: oracle shape differs.");
            TMetadata[] expected = cpu[i].ToArray();
            Assert.True(expected.SequenceEqual(cpuAgain[i].ToArray()),
                $"{op.Name} {kind}[{i}]: repeated CPU metadata differs.");
            Assert.True(expected.SequenceEqual(gpu[i].ToArray()),
                $"{op.Name} {kind}[{i}]: GPU metadata differs.");
            Assert.True(expected.SequenceEqual(gpuAgain[i].ToArray()),
                $"{op.Name} {kind}[{i}]: repeated GPU metadata differs.");
            Assert.True(expected.SequenceEqual(oracle[i].ToArray()),
                $"{op.Name} {kind}[{i}]: oracle metadata differs.");
        }
    }

    private static void AssertWrappedRadiansResults(
        string phase, OpCase op, OpParityFixture fx,
        float[] cpu, float[] cpuAgain, float[] gpu, float[] gpuAgain, double[] oracle,
        double absoluteTolerance)
    {
        Assert.True(cpu.Length == cpuAgain.Length && cpu.Length == gpu.Length &&
                    cpu.Length == gpuAgain.Length && cpu.Length == oracle.Length,
            $"{op.Name} {phase}: wrapped-radian output lengths differ.");

        Assert.True(absoluteTolerance > 0.0,
            $"{op.Name} {phase}: wrapped-radian comparison requires a positive absolute tolerance.");

        double maxCpuGpu = 0.0;
        double maxCpuOracle = 0.0;
        double maxGpuOracle = 0.0;
        int worstIndex = -1;
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.False(float.IsNaN(cpu[i]) || float.IsInfinity(cpu[i]), $"{op.Name} {phase}: CPU non-finite {cpu[i]} @[{i}]");
            Assert.False(float.IsNaN(gpu[i]) || float.IsInfinity(gpu[i]), $"{op.Name} {phase}: GPU non-finite {gpu[i]} @[{i}]");
            Assert.False(double.IsNaN(oracle[i]) || double.IsInfinity(oracle[i]), $"{op.Name} {phase}: oracle non-finite {oracle[i]} @[{i}]");

            double cpuGpu = WrappedRadiansDistance(cpu[i], gpu[i]);
            double cpuOracle = WrappedRadiansDistance(cpu[i], oracle[i]);
            double gpuOracle = WrappedRadiansDistance(gpu[i], oracle[i]);
            if (cpuGpu > maxCpuGpu) { maxCpuGpu = cpuGpu; worstIndex = i; }
            if (cpuOracle > maxCpuOracle) maxCpuOracle = cpuOracle;
            if (gpuOracle > maxGpuOracle) maxGpuOracle = gpuOracle;
        }

        Assert.True(ParityMath.BitExact(cpu, cpuAgain, out int cpuDifference),
            $"{op.Name} {phase}: CPU is nondeterministic — differs at [{cpuDifference}] across identical runs.");
        Assert.True(ParityMath.BitExact(gpu, gpuAgain, out int gpuDifference),
            $"{op.Name} {phase}: GPU is nondeterministic — differs at [{gpuDifference}] across identical runs.");

        fx.Record($"{op.Name}:{phase}\t{op.Category}\twrapped-radians\t{maxCpuGpu:R}\t{worstIndex}");
        Assert.True(maxCpuGpu <= absoluteTolerance,
            $"{op.Name} {phase}: circular phase distance {maxCpuGpu:E3} rad exceeded {absoluteTolerance:E3} rad at [{worstIndex}]. " +
            $"Oracle circular drift — CPU {maxCpuOracle:E3} rad, GPU {maxGpuOracle:E3} rad.");
    }

    private static void DisposeTensors<T>(Tensor<T>[]? tensors)
    {
        if (tensors is null) return;
        foreach (Tensor<T>? tensor in tensors)
            tensor?.Dispose();
    }

    private static void DisposeOutputs<T>(HeterogeneousTensorOutputs<T>? outputs)
    {
        if (outputs is null) return;
        DisposeTensors(outputs.Numeric);
        DisposeTensors(outputs.Integers);
        DisposeTensors(outputs.Booleans);
    }

    private static double WrappedRadiansDistance(double left, double right) =>
        Math.Abs(Math.IEEERemainder(left - right, 2.0 * Math.PI));

    private static void AssertResults(
        string phase, OpCase op, OpParityFixture fx,
        float[] cpuF, float[] cpuF2, float[] gpuF, float[] gpuF2, double[] oracleD, ParityTol tol)
    {
        // Quarantined divergence: a confirmed, tracked cross-engine bug (parity gap, nondeterminism,
        // non-finite, OR a hard shape mismatch). Record + SKIP before any hard assert — never fail CI
        // on a known bug.
        if (op.KnownDivergence is { } known)
        {
            fx.Record($"{op.Name}:{phase}\t{op.Category}\tKNOWN-DIVERGENCE\t-\t-\t-");
            Skip.If(true, $"KNOWN DIVERGENCE ({op.Name} {phase}): {known}. cpuLen={cpuF.Length} gpuLen={gpuF.Length}.");
            return;
        }

        // Shape/length agreement.
        Assert.True(cpuF.Length == gpuF.Length && cpuF.Length == oracleD.Length,
            $"{op.Name} {phase}: length mismatch cpu={cpuF.Length} gpu={gpuF.Length} oracle={oracleD.Length}");

        // Bound each engine against the double oracle (rounded to float) — localizes drift.
        var oracleF = ParityMath.ToFloat(oracleD);
        var cpuVsOracle = ParityMath.Compare(cpuF, oracleF);
        var gpuVsOracle = ParityMath.Compare(gpuF, oracleF);
        string worse = cpuVsOracle.MaxUlp > gpuVsOracle.MaxUlp ? "CPU" : (gpuVsOracle.MaxUlp > cpuVsOracle.MaxUlp ? "GPU" : "tie");
        bool ok = ParityMath.Within(cpuF, gpuF, tol, out var cpuVsGpu);

        fx.Record(string.Join("\t", new[]
        {
            $"{op.Name}:{phase}", op.Category,
            cpuVsGpu.MaxUlp.ToString(CultureInfo.InvariantCulture),
            cpuVsGpu.MaxAbs.ToString("R", CultureInfo.InvariantCulture),
            cpuVsGpu.MaxRel.ToString("R", CultureInfo.InvariantCulture),
            cpuVsGpu.WorstIndex.ToString(CultureInfo.InvariantCulture),
            cpuVsGpu.WorstA.ToString("R", CultureInfo.InvariantCulture),
            cpuVsGpu.WorstB.ToString("R", CultureInfo.InvariantCulture),
            tol.Ulps.ToString(CultureInfo.InvariantCulture),
            tol.AbsFloor.ToString("R", CultureInfo.InvariantCulture),
            tol.Rel.ToString("R", CultureInfo.InvariantCulture),
            cpuVsOracle.MaxUlp.ToString(CultureInfo.InvariantCulture),
            gpuVsOracle.MaxUlp.ToString(CultureInfo.InvariantCulture),
            worse,
        }));

        // Finiteness — a non-finite result is a bug regardless of tolerance.
        for (int i = 0; i < cpuF.Length; i++)
        {
            Assert.False(float.IsNaN(cpuF[i]) || float.IsInfinity(cpuF[i]), $"{op.Name} {phase}: CPU non-finite {cpuF[i]} @[{i}]");
            Assert.False(float.IsNaN(gpuF[i]) || float.IsInfinity(gpuF[i]), $"{op.Name} {phase}: GPU non-finite {gpuF[i]} @[{i}]");
        }

        // Same-engine determinism: re-running the identical op must reproduce bit-for-bit.
        Assert.True(ParityMath.BitExact(cpuF, cpuF2, out int cd),
            $"{op.Name} {phase}: CPU is nondeterministic — differs at [{cd}] across identical runs.");
        Assert.True(ParityMath.BitExact(gpuF, gpuF2, out int gd),
            $"{op.Name} {phase}: GPU is nondeterministic — differs at [{gd}] across identical runs.");

        Assert.True(ok,
            $"{op.Name} {phase}: CPU vs GPU exceeded tol {tol}. {cpuVsGpu.Describe()}. " +
            $"Oracle drift — CPU {cpuVsOracle.MaxUlp} ULP, GPU {gpuVsOracle.MaxUlp} ULP (worse: {worse}).");
    }
}
#endif
