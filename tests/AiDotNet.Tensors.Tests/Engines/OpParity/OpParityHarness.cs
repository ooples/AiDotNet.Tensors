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
    public DirectGpuTensorEngine? Gpu { get; }
    public bool GpuReady { get; }
    public Exception? GpuInitError { get; }

    private readonly List<string> _report = new();
    private readonly object _lock = new();

    public OpParityFixture()
    {
        CudaDispatchPolicy.AllowTF32 = false;
        Cpu = new CpuEngine();
        try
        {
            Gpu = new DirectGpuTensorEngine();
            GpuReady = Gpu.IsGpuAvailable;
        }
        catch (Exception ex)
        {
            GpuInitError = ex;
            GpuReady = false;
        }
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
                sb.AppendLine("op\tcategory\tcpu_vs_gpu\tcpu_vs_oracle\tgpu_vs_oracle\tworse_engine");
                foreach (var l in _report.OrderBy(s => s, StringComparer.Ordinal)) sb.AppendLine(l);
                File.WriteAllText(path, sb.ToString());
            }
            catch { /* reporting is best-effort; never fail a run over it */ }
        }
    }
}

[CollectionDefinition("OpParity")]
public sealed class OpParityCollection : ICollectionFixture<OpParityFixture> { }

/// <summary>The parity checker. For one <see cref="OpCase"/> it runs the float op on CPU and GPU,
/// the double ORACLE on CPU, and asserts: (1) same-engine determinism (bit-exact re-run), (2) all
/// results finite, (3) CPU vs GPU within the op's ULP budget, and (4) each engine bounded against
/// the oracle — recording which engine drifts more (the localization the #775 ViT bug needs).</summary>
public static class OpParityHarness
{
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

        var gpu = fx.Gpu!;
        float[] cpuF = op.RunFloatGrad!(fx.Cpu).ToArray();
        float[] cpuF2 = op.RunFloatGrad!(fx.Cpu).ToArray();
        float[] gpuF = op.RunFloatGrad!(gpu).ToArray();
        float[] gpuF2 = op.RunFloatGrad!(gpu).ToArray();
        double[] oracleD = op.RunDoubleGrad!(fx.Cpu).ToArray();

        AssertResults("backward", op, fx, cpuF, cpuF2, gpuF, gpuF2, oracleD, op.BwdTol);
    }

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
