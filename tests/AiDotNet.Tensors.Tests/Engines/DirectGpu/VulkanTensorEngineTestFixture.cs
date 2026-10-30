// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>A fixture that hands a test class the direct-GPU engine it runs on.</summary>
public interface IDirectGpuTensorEngineFixture
{
    /// <summary>The engine, or null when the backend could not be created.</summary>
    DirectGpuTensorEngine? Engine { get; }

    /// <summary>True when <see cref="Engine"/> has a working device.</summary>
    bool IsAvailable { get; }
}

/// <summary>
/// A direct-GPU engine on the Vulkan backend, which the default backend order never selects (cuda, opencl, hip). It is
/// available when the device has the 64-bit shader features the parity kernels compute in.
/// </summary>
public sealed class VulkanTensorEngineTestFixture : IDirectGpuTensorEngineFixture
{
    public DirectGpuTensorEngine? Engine { get; }
    public bool IsAvailable => Engine?.IsGpuAvailable == true;

    public VulkanTensorEngineTestFixture()
    {
        try
        {
            var backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.SupportsTorchParityDoubles || !backend.SupportsTorchParityInt64) return;
            // The backend is a process-wide singleton other test classes share, so nothing here disposes it: the
            // engine is created over a non-owning DirectGpuEngine and left alive.
            Engine = new DirectGpuTensorEngine(new DirectGpuEngine(backend));
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException
            or InvalidOperationException)
        {
            Engine = null;
        }
    }
}
