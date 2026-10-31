// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// Settings the vendor OpenCL runtime reads from the process environment when it loads, at the process's first OpenCL
/// call. Applied from the static constructor of each class that declares an OpenCL entry point, and from the module
/// initializer that creates the GPU engine on .NET 6+, so whichever call comes first finds them set. A variable the
/// user has set is left as it is.
/// </summary>
internal static class OpenClRuntimeDefaults
{
    /// <summary>
    /// AMD's runtime suballocates buffers up to this many KB from heap chunks it reserves on the device, and reserves
    /// the first 64 MB chunk when the first command queue is created, before anything is allocated. Every process that
    /// touched the GPU held 65 MB of dedicated memory, the whole footprint of a small model (PyTorch on DirectML holds
    /// 2-5 MB). At 0 (anything below 8) no chunk is reserved and each buffer is its own allocation, which this library
    /// pools anyway (GpuBufferPool). Other vendors' runtimes do not read it.
    /// </summary>
    internal const string SuballocationVariable = "GPU_MAX_SUBALLOC_SIZE";

    private static int s_applied;

    /// <summary>Sets the defaults once, before the runtime loads; later calls do nothing.</summary>
    internal static void Apply()
    {
        if (Interlocked.Exchange(ref s_applied, 1) != 0) return;
        try
        {
            if (Environment.GetEnvironmentVariable(SuballocationVariable) is null)
                Environment.SetEnvironmentVariable(SuballocationVariable, "0");
        }
        catch (System.Security.SecurityException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"Could not set {SuballocationVariable}: {ex.Message}");
        }
    }
}