// Copyright (c) AiDotNet. All rights reserved.
// Every kernel name HipBackend resolves must be in its kernel cache once the backend is constructed.
//
// WHY THIS EXISTS. HipBackend looked up max_axis, copy_2d_strided, csr_segmented_{max,min,stddev}, four octonion
// kernels, batched_dot_product, tile_batch and broadcast_multiply_last_axis, and no compiled HIP module provided
// them: some were defined in a source HIP never compiled (HipOctonionKernels, HipDotProductKernels,
// CudaMissingLookupKernels), the rest nowhere. Each op threw GpuCapabilityUnavailableException and the engine ran it
// on the CPU, so results stayed correct while the HIP path did no GPU work; TensorAll / TensorAny surfaced it only
// because their tests count fallbacks. A static "is it defined in some source" check (CudaKernelLookupIntegrityTests)
// cannot see a source that is never compiled, so this reads the LIVE cache of a constructed backend.
#if NETCOREAPP
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class HipKernelLookupIntegrityTests
{
    // A kernel name passed to the cache itself, or as the first argument of a helper that resolves one
    // (RequireResidentKernel, LaunchBatch, ParityKernel, GetRequiredFftKernel, LaunchResidentAxis, ...).
    private static readonly Regex LookupPattern = new(
        @"(?:_kernelCache\s*(?:\.TryGetValue|\.ContainsKey)\s*\(\s*|_kernelCache\s*\[\s*|\b(?:\w*Kernel|Launch\w*)\s*\(\s*)""([a-z][a-z0-9_]*)""",
        RegexOptions.None, TimeSpan.FromSeconds(5));

    private static string? RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "src", "AiDotNet.Tensors")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    [SkippableFact]
    public void EveryHipKernelLookupIsCompiled()
    {
        string? root = RepoRoot();
        Skip.If(root is null, "Repository sources not present.");
        string hip = Path.Combine(root!, "src", "AiDotNet.Tensors", "Engines", "DirectGpu", "HIP");
        string[] backendFiles = Directory.GetFiles(hip, "HipBackend*.cs", SearchOption.TopDirectoryOnly);
        Assert.NotEmpty(backendFiles);

        var lookedUp = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string file in backendFiles)
            foreach (Match m in LookupPattern.Matches(File.ReadAllText(file)))
                lookedUp.Add(m.Groups[1].Value);
        // The positive control: the names this guard was written for must be among the ones it reads.
        foreach (string known in new[] { "max_axis", "copy_2d_strided", "csr_segmented_stddev", "octonion_linear_backward_input", "batched_dot_product", "bx_repeat_elements" })
            Assert.Contains(known, lookedUp);

        HipBackend backend;
        try { backend = new HipBackend(); }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException
            or InvalidOperationException)
        {
            Skip.If(true, "HIP runtime not present.");
            return;
        }
        using (backend)
        {
            Skip.IfNot(backend.IsAvailable, "No HIP device.");
            var missing = lookedUp.Where(name => !backend.HasCompiledKernel(name)).ToList();
            Assert.True(missing.Count == 0,
                $"HipBackend resolves {missing.Count} kernel name(s) that no compiled HIP module registered: " +
                $"{string.Join(", ", missing)}. Each op throws GpuCapabilityUnavailableException and the engine runs it " +
                "on the CPU. Compile the source that defines it (and load the name), or point the lookup at the real " +
                "kernel with the same arguments and launch geometry.");
        }
    }
}
#endif