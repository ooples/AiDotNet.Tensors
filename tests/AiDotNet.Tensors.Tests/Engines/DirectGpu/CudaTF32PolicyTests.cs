// Copyright (c) AiDotNet. All rights reserved.
// PR #333 / issue #337 — verify the user-facing TF32 policy surface
// (AllowTF32 property + AIDOTNET_DISABLE_TF32 env var) before the
// hardware-dependent cuBLAS-mode wiring runs. These tests don't require
// CUDA; they only validate the policy decision plumbing.

using System;
using System.Reflection;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Marker collection that disables xUnit parallel execution for the
/// TF32-policy test class. The tests mutate process-wide state
/// (AIDOTNET_DISABLE_TF32 env var + the static <c>_allowTF32Override</c>
/// field on <see cref="CudaDispatchPolicy"/>), so running them concurrently
/// with each other or with any other test that reads the same state would
/// cause flaky cross-test interference.
/// </summary>
[CollectionDefinition(nameof(CudaTF32PolicyTests), DisableParallelization = true)]
public sealed class CudaTF32PolicyTestsCollection { }

[Collection(nameof(CudaTF32PolicyTests))]
public class CudaTF32PolicyTests : IDisposable
{
    // Snapshot the inherited env var + runtime override on construction so
    // we can restore them in Dispose. Each test runs sequentially under
    // the no-parallel collection, but the env var still leaks into the
    // PROCESS — a subsequent unrelated test that reads AIDOTNET_DISABLE_TF32
    // would observe the wrong value without this save/restore.
    private readonly string? _originalEnvVar;
    private readonly bool? _originalOverride;
    private static readonly FieldInfo _overrideField =
        typeof(CudaDispatchPolicy).GetField("_allowTF32Override",
            BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException(
            "CudaDispatchPolicy._allowTF32Override field not found — refactor broke the test.");

    public CudaTF32PolicyTests()
    {
        _originalEnvVar = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_TF32");
        _originalOverride = (bool?)_overrideField.GetValue(null);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", _originalEnvVar);
        _overrideField.SetValue(null, _originalOverride);
    }

    /// <summary>
    /// Default state: no env var, no runtime override → AllowTF32 = true.
    /// Captures the "TF32 on by default" guarantee that the issue contract
    /// promises Ampere+ users get acceleration without any configuration.
    /// </summary>
    [Fact]
    public void AllowTF32_DefaultIsTrue()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", null);
        // Force null override so the env-var fall-through is exercised.
        _overrideField.SetValue(null, (bool?)null);

        Assert.True(CudaDispatchPolicy.AllowTF32);
    }

    /// <summary>
    /// AIDOTNET_DISABLE_TF32 with any non-empty value forces AllowTF32 = false
    /// when no runtime override is set. Lets ops / reproducibility runs disable
    /// TF32 globally without recompiling.
    /// </summary>
    [Fact]
    public void AllowTF32_EnvVarSet_ReturnsFalse()
    {
        _overrideField.SetValue(null, (bool?)null);
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", "1");

        Assert.False(CudaDispatchPolicy.AllowTF32);
    }

    /// <summary>
    /// Runtime AllowTF32 setter overrides the env var so tests / hot-reload
    /// scenarios can flip TF32 without re-launching the process.
    /// </summary>
    [Fact]
    public void AllowTF32_RuntimeOverride_BeatsEnvVar()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", "1");

        CudaDispatchPolicy.AllowTF32 = true;
        Assert.True(CudaDispatchPolicy.AllowTF32);

        CudaDispatchPolicy.AllowTF32 = false;
        Assert.False(CudaDispatchPolicy.AllowTF32);
    }
}
