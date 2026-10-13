// Copyright (c) AiDotNet. All rights reserved.
// Every kernel name a backend resolves must be in its kernel cache once the backend is constructed.
//
// WHY THIS EXISTS. HipBackend looked up max_axis, copy_2d_strided, csr_segmented_{max,min,stddev}, four octonion
// kernels, batched_dot_product, tile_batch and broadcast_multiply_last_axis, and no compiled HIP module provided
// them: some were defined in a source HIP never compiled (HipOctonionKernels, HipDotProductKernels,
// CudaMissingLookupKernels), the rest nowhere. OpenClBackend had the same four octonion kernels in a source it never
// compiled. Each op threw GpuCapabilityUnavailableException and the engine ran it on the CPU, so results stayed
// correct while the device path did no work; TensorAll / TensorAny surfaced it only because their tests count
// fallbacks. A static "is it defined in some source" check (CudaKernelLookupIntegrityTests) cannot see a source that
// is never compiled, so these read the LIVE cache of a constructed backend.
#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

internal static class KernelLookupIntegrity
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

    /// <summary>Every name the backend's source files resolve, or null when the sources are not present.</summary>
    internal static SortedSet<string>? LookedUp(string backendFolder, string filePattern)
    {
        string? root = RepoRoot();
        if (root is null) return null;
        string folder = Path.Combine(root, "src", "AiDotNet.Tensors", "Engines", "DirectGpu", backendFolder);
        string[] files = Directory.GetFiles(folder, filePattern, SearchOption.TopDirectoryOnly);
        Assert.NotEmpty(files);
        var names = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string file in files)
            foreach (Match m in LookupPattern.Matches(File.ReadAllText(file)))
                names.Add(m.Groups[1].Value);
        return names;
    }

    internal static void AssertAllCompiled(string backend, IEnumerable<string> lookedUp, Func<string, bool> hasCompiledKernel)
    {
        var missing = lookedUp.Where(name => !hasCompiledKernel(name)).ToList();
        Assert.True(missing.Count == 0,
            $"{backend} resolves {missing.Count} kernel name(s) that no compiled module registered: " +
            $"{string.Join(", ", missing)}. Each op throws GpuCapabilityUnavailableException and the engine runs it " +
            "on the CPU. Compile the source that defines it (and load the name), or point the lookup at the real " +
            "kernel with the same arguments and launch geometry.");
    }
}

[Collection("DirectGpuSerial")]
public sealed class HipKernelLookupIntegrityTests : IClassFixture<HipTensorEngineTestFixture>
{
    private readonly HipTensorEngineTestFixture _fixture;

    public HipKernelLookupIntegrityTests(HipTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void EveryHipKernelLookupIsCompiled()
    {
        var lookedUp = KernelLookupIntegrity.LookedUp("HIP", "HipBackend*.cs");
        Skip.If(lookedUp is null, "Repository sources not present.");
        // The positive control: the names this guard was written for must be among the ones it reads.
        foreach (string known in new[] { "max_axis", "copy_2d_strided", "csr_segmented_stddev", "octonion_linear_backward_input", "batched_dot_product", "bx_repeat_elements" })
            Assert.Contains(known, lookedUp);
        Skip.IfNot(_fixture.IsAvailable, "No HIP device.");
        var backend = _fixture.Engine?.GetBackend() as HipBackend
            ?? throw new InvalidOperationException("The HIP fixture's engine is not on a HipBackend.");
        KernelLookupIntegrity.AssertAllCompiled("HipBackend", lookedUp, backend.HasCompiledKernel);
    }
}

[Collection("OpenClBackend")]
public sealed class OpenClKernelLookupIntegrityTests : IClassFixture<OpenClBackendTestFixture>
{
    private readonly OpenClBackendTestFixture _fixture;

    public OpenClKernelLookupIntegrityTests(OpenClBackendTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void EveryOpenClKernelLookupIsCompiled()
    {
        var lookedUp = KernelLookupIntegrity.LookedUp("OpenCL", "OpenClBackend*.cs");
        Skip.If(lookedUp is null, "Repository sources not present.");
        Assert.Contains("octonion_linear_backward_input", lookedUp);
        Skip.IfNot(_fixture.IsAvailable, "No OpenCL device.");
        KernelLookupIntegrity.AssertAllCompiled("OpenClBackend", lookedUp, _fixture.Backend.HasCompiledKernel);
    }
}
#endif