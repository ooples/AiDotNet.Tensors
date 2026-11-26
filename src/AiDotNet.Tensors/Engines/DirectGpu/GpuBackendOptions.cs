// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Process-wide backend-selection settings of the GPU engines (<see cref="DirectGpuEngine"/>,
/// <see cref="DirectGpuBackendFactory"/> and the engine <see cref="AiDotNetEngine"/> auto-detects).
/// </summary>
/// <remarks>
/// Backend detection tries CUDA first. When a CUDA backend cannot be created, detection moves on to the next backend
/// (OpenCL, HIP) or to the CPU, and always says so with a warning that names the CUDA failure. Set
/// <see cref="RequiredBackend"/> when running on anything else is an error: an engine that cannot create the
/// required backend then throws, naming why, instead of running on another backend or on the CPU.
/// </remarks>
public static class GpuBackendOptions
{
    private static int _requiredBackend = (int)GpuBackendType.None;

    /// <summary>
    /// The backend a GPU engine must run on, or <see cref="GpuBackendType.None"/> (the default) to take the first that
    /// works. Read when an engine is created; an engine that already exists keeps its backend.
    /// </summary>
    public static GpuBackendType RequiredBackend
    {
        get => (GpuBackendType)System.Threading.Volatile.Read(ref _requiredBackend);
        set
        {
            if (value is not (GpuBackendType.None or GpuBackendType.Cuda or GpuBackendType.OpenCl
                or GpuBackendType.Hip or GpuBackendType.Vulkan))
                throw new ArgumentOutOfRangeException(nameof(value), value,
                    "A GPU engine can be required to run on CUDA, OpenCL, HIP or Vulkan.");
            System.Threading.Volatile.Write(ref _requiredBackend, (int)value);
        }
    }

    internal static string NameOf(GpuBackendType backend) => backend switch
    {
        GpuBackendType.Cuda => "cuda",
        GpuBackendType.OpenCl => "opencl",
        GpuBackendType.Hip => "hip",
        GpuBackendType.Vulkan => "vulkan",
        _ => backend.ToString().ToLowerInvariant(),
    };

    // Says that detection did not land on CUDA where it tried it, wherever it landed: the trace listeners and stderr,
    // since a run that silently trains on another backend (or the CPU) is the failure this exists to surface.
    internal static void WarnCudaFallback(string cudaFailure, string landedOn)
    {
        string message = $"[AiDotNet.Tensors] CUDA backend unavailable ({cudaFailure}); running on {landedOn} instead. "
            + $"Set {nameof(GpuBackendOptions)}.{nameof(RequiredBackend)} = {nameof(GpuBackendType)}.{nameof(GpuBackendType.Cuda)} "
            + "to make this an error.";
        System.Diagnostics.Trace.TraceWarning(message);
        try { Console.Error.WriteLine(message); }
        catch (System.IO.IOException) { }
        catch (ObjectDisposedException) { }
    }
}
