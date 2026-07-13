// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). EMPIRICAL GPU-residency guard.
//
// The reflection guard (BackendCompletenessTests) reports which ops lack a DEDICATED GPU override —
// a SUPERSET, because a composed op (e.g. TensorSquare -> TensorMultiply) has no override of its own
// yet still runs on the GPU through its primitives. That metric can only OVER-count gaps, never prove
// one covered. This guard does the opposite with a MEASUREMENT, not a guess: it runs every registry
// op on the GPU engine while GpuLaunchProbe counts real kernel dispatches, and flags any op that fires
// ZERO kernels. A zero-launch op computed entirely on the host when the caller asked for the GPU
// engine — the TRUE coverage gap. No heuristic can mislabel a real gap as covered here: a kernel
// either launched or it did not.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

[Collection("OpParity")]
public sealed class GpuResidencyProbeTests
{
    private readonly OpParityFixture _fx;
    public GpuResidencyProbeTests(OpParityFixture fx) => _fx = fx;

    // Registry ops that execute ZERO GPU kernels on the GPU engine — i.e. silently run on the CPU.
    // GOAL 0. Ratchets DOWN only: add a kernel / GPU-primitive composition to lower it. Never raise it
    // to hide a regression. Measured in isolation (see class remarks); parallel runs can only observe
    // an equal-or-lower count (a stray cross-collection launch never manufactures a fallback), so the
    // <= assertion cannot false-fail.
    private const int CpuFallbackFloor = 156;

    [SkippableFact]
    public void EveryRegistryOp_ActuallyLaunchesAGpuKernel()
    {
        if (!_fx.GpuReady)
        {
            if (_fx.RequireGpu)
                throw new InvalidOperationException(
                    "GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.", _fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var gpu = _fx.Gpu!;
        var fallback = new SortedSet<string>(StringComparer.Ordinal); // ran on CPU (0 launches)
        var covered = new SortedSet<string>(StringComparer.Ordinal);  // launched >= 1 kernel
        var errored = new SortedSet<string>(StringComparer.Ordinal);  // threw (can't classify)

        foreach (var op in OpParityRegistry.All())
        {
            // A quarantined op makes no residency claim (its GPU path is skipped/known-broken).
            if (op.KnownDivergence is not null || op.GpuUnsafe) continue;

            GpuLaunchProbe.Reset();
            try { _ = op.RunFloat(gpu).ToArray(); }
            catch { errored.Add(op.Name); continue; }

            if (GpuLaunchProbe.Count == 0) fallback.Add($"{op.Name}\t{op.Category}");
            else covered.Add(op.Name);
        }

        WriteReport(covered, fallback, errored);

        Assert.True(fallback.Count <= CpuFallbackFloor,
            $"{fallback.Count} registry ops execute ZERO GPU kernels on the GPU engine (floor " +
            $"{CpuFallbackFloor}) — they silently run on the CPU when the caller selected the GPU engine. " +
            $"See gpu-cpu-fallback.md for the list. Write a kernel or GPU-primitive composition to move " +
            $"them on-device; do not raise the floor.");
    }

    private static void WriteReport(IEnumerable<string> covered, IEnumerable<string> fallback, IEnumerable<string> errored)
    {
        var cov = covered.ToList();
        var fb = fallback.ToList();
        var er = errored.ToList();
        var sb = new StringBuilder();
        sb.AppendLine("# GPU residency probe — measured kernel launches per op (#775)");
        sb.AppendLine($"ran >= 1 GPU kernel (on-device) : {cov.Count}");
        sb.AppendLine($"ran ZERO GPU kernels (CPU fallback, the worklist) : {fb.Count} (goal 0)");
        sb.AppendLine($"threw before classifying : {er.Count}");
        sb.AppendLine();
        sb.AppendLine("## CPU fallback — ran entirely on the host (op\\tcategory)");
        foreach (var n in fb) sb.AppendLine($"  [ ] {n}");
        sb.AppendLine();
        sb.AppendLine("## errored (excluded from the count)");
        foreach (var n in er) sb.AppendLine($"  [!] {n}");
        try
        {
            var dir = Environment.GetEnvironmentVariable("AIDOTNET_OPPARITY_REPORT_DIR")
                      ?? Path.Combine(Path.GetTempPath(), "aidotnet-opparity");
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, "gpu-cpu-fallback.md"), sb.ToString());
        }
        catch { /* best-effort */ }
    }
}
#endif
