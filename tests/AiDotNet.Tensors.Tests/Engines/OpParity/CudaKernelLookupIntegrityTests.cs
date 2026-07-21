// Copyright (c) AiDotNet. All rights reserved.
// Every kernel name CudaBackend looks up must actually be defined in a CUDA kernel source (#775).
//
// WHY THIS EXISTS. CudaBackend.BroadcastMultiplyLastAxis asked the kernel cache for
// "broadcast_multiply_last_axis", a name defined NOWHERE. Every call threw kernel-not-found and
// DirectGpuTensorEngine.TensorBroadcastMultiply swallowed it in a bare catch, returning a correct CPU
// result while doing zero GPU work. It survived because:
//   - the hollow-override probe only counts the THROWING kernel-cache indexer, and this site used
//     TryGetValue, so no KernelMiss was ever recorded; and
//   - broadcast_multiply_FIRST_axis genuinely exists, so the naming looked internally consistent.
// It was found only after the launch-counting fixes cleared the ops that were merely mismeasured.
//
// This guard is static — it reads the sources, needs no GPU, and runs everywhere. It catches the whole
// class at authoring time instead of waiting for an op to be exercised at runtime on CUDA.
#if !NETFRAMEWORK
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

public sealed class CudaKernelLookupIntegrityTests
{
    // Kernel names CudaBackend looks up that have NO definition in any CUDA kernel source. Each one
    // throws the moment its op runs on CUDA. They are NOT renames — verified against the nearest
    // similarly named kernels:
    //   var_axis      : VarAxis passes (input, mean, variance, outer, reduce) = 5 args and wants BOTH
    //                   outputs. variance_axis(input, output, outer, reduce) takes 4 and emits variance
    //                   only. Repointing would reproduce the 15-vs-16 argument truncation fixed in
    //                   3aed217 — a 5-arg launch into a 4-param kernel.
    //   tile_batch    : nearest is tile_axis, different shape contract.
    //   copy_2d_strided, squash, squash_backward, csr_segmented_{max,min,stddev}:
    //                   no similar kernel exists in CUDA or OpenCL. These must be WRITTEN.
    // Callers are live (Squash/SquashBackward from the capsule path, TileBatch from tile/repeat,
    // VarAxis from variance, Copy2DStrided from the concat/slice path), so these are latent defects,
    // not dead API. They are invisible to the op-parity registry because it does not exercise them.
    //
    // RATCHETS DOWN ONLY. Define or correctly repoint a kernel and remove its entry. Never add one.
    private static readonly HashSet<string> KnownDangling = new(StringComparer.Ordinal)
    {
        "copy_2d_strided",
        "csr_segmented_max",
        "csr_segmented_min",
        "csr_segmented_stddev",
        "squash",
        "squash_backward",
        "tile_batch",
        "var_axis",
    };

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "src", "AiDotNet.Tensors")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("repo root not found from " + AppContext.BaseDirectory);
    }

    [Fact]
    public void EveryCudaKernelLookupHasADefinition()
    {
        string cuda = Path.Combine(RepoRoot(), "src", "AiDotNet.Tensors", "Engines", "DirectGpu", "CUDA");
        Skip.If(!Directory.Exists(cuda), "CUDA backend sources not present.");

        string backend = File.ReadAllText(Path.Combine(cuda, "CudaBackend.cs"));
        var lookedUp = new HashSet<string>(
            Regex.Matches(backend, @"_kernelCache(?:\.TryGetValue\(|\[)""([a-z0-9_]+)""")
                 .Select(m => m.Groups[1].Value),
            StringComparer.Ordinal);

        var defined = new HashSet<string>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(Path.Combine(cuda, "Kernels"), "*.cs"))
        {
            string src = File.ReadAllText(file);
            // The declaration frequently wraps, e.g.
            //     extern ""C"" __global__ __launch_bounds__(256)
            //     void normalize_rows_fused(
            // so this must cross newlines. A single-line pattern silently reports ~13 live kernels as
            // missing (normalize_rows_fused, mel_filterbank_apply, bispectrum_gather, ...) — a false
            // alarm that would have been "fixed" by padding the allowlist, hiding the real 8.
            foreach (Match m in Regex.Matches(src, @"__global__[\s\S]{0,120}?\bvoid\s+(\w+)\s*\("))
                defined.Add(m.Groups[1].Value);
            // Names the module registers with cuModuleGetFunction (GetKernelNames arrays) count as
            // defined too — belt and braces for any declaration shape the pattern above misses.
            foreach (Match m in Regex.Matches(src, @"^\s*""([a-z0-9_]+)"",\s*$", RegexOptions.Multiline))
                defined.Add(m.Groups[1].Value);
        }

        Assert.NotEmpty(lookedUp);
        Assert.NotEmpty(defined);

        var dangling = lookedUp.Where(n => !defined.Contains(n)).OrderBy(n => n, StringComparer.Ordinal).ToList();
        var unexpected = dangling.Where(n => !KnownDangling.Contains(n)).ToList();

        Assert.True(unexpected.Count == 0,
            $"CudaBackend looks up {unexpected.Count} kernel name(s) that no CUDA source defines: " +
            $"{string.Join(", ", unexpected)}. Each throws kernel-not-found the moment its op runs and, " +
            "if a caller wraps it in a catch, degrades silently to the CPU with a correct-looking result. " +
            "Define the kernel or point the lookup at the real one — and check the ARGUMENT COUNT matches, " +
            "since a near-name with a different arity reproduces the launch-truncation class.");

        // Ratchet: entries that got fixed must be removed from KnownDangling so it cannot drift upward.
        var stale = KnownDangling.Where(n => !dangling.Contains(n)).OrderBy(n => n, StringComparer.Ordinal).ToList();
        Assert.True(stale.Count == 0,
            $"KnownDangling lists {stale.Count} name(s) that now resolve: {string.Join(", ", stale)}. " +
            "Remove them from the set — this list ratchets DOWN only.");
    }
}
#endif
