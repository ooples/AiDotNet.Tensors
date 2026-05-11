// Copyright (c) AiDotNet. All rights reserved.
// PR #333 / issue #337 — verify the user-facing TF32 policy surface
// (AllowTF32 property + AIDOTNET_DISABLE_TF32 env var) before the
// hardware-dependent cuBLAS-mode wiring runs. These tests don't require
// CUDA; they only validate the policy decision plumbing.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class CudaTF32PolicyTests
{
    /// <summary>
    /// Default state: no env var, no runtime override → AllowTF32 = true.
    /// Captures the "TF32 on by default" guarantee that the issue contract
    /// promises Ampere+ users get acceleration without any configuration.
    /// </summary>
    [Fact]
    public void AllowTF32_DefaultIsTrue()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", null);
        CudaDispatchPolicy.AllowTF32 = true; // explicit reset to clean state
        try
        {
            // Force null override so the env-var fall-through is exercised.
            var prop = typeof(CudaDispatchPolicy).GetField("_allowTF32Override",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            prop!.SetValue(null, (bool?)null);

            Assert.True(CudaDispatchPolicy.AllowTF32);
        }
        finally
        {
            CudaDispatchPolicy.AllowTF32 = true;
        }
    }

    /// <summary>
    /// AIDOTNET_DISABLE_TF32 with any non-empty value forces AllowTF32 = false
    /// when no runtime override is set. Lets ops / reproducibility runs disable
    /// TF32 globally without recompiling.
    /// </summary>
    [Fact]
    public void AllowTF32_EnvVarSet_ReturnsFalse()
    {
        var prop = typeof(CudaDispatchPolicy).GetField("_allowTF32Override",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        prop!.SetValue(null, (bool?)null);

        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", "1");
        try
        {
            Assert.False(CudaDispatchPolicy.AllowTF32);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", null);
        }
    }

    /// <summary>
    /// Runtime AllowTF32 setter overrides the env var so tests / hot-reload
    /// scenarios can flip TF32 without re-launching the process.
    /// </summary>
    [Fact]
    public void AllowTF32_RuntimeOverride_BeatsEnvVar()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", "1");
        try
        {
            CudaDispatchPolicy.AllowTF32 = true;
            Assert.True(CudaDispatchPolicy.AllowTF32);

            CudaDispatchPolicy.AllowTF32 = false;
            Assert.False(CudaDispatchPolicy.AllowTF32);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_TF32", null);
            var prop = typeof(CudaDispatchPolicy).GetField("_allowTF32Override",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            prop!.SetValue(null, (bool?)null);
        }
    }
}
