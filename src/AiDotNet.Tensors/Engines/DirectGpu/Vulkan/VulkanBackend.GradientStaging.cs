// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>No page-locked gradient staging on this backend: the exchange uses its per-buffer download and upload.</summary>
public sealed unsafe partial class VulkanBackend : IGradientHostStagingSource
{
    IGradientHostStaging? IGradientHostStagingSource.TryCreateGradientHostStaging(int floats) => null;

    IGradientHostStaging? IGradientHostStagingSource.TryRegisterGradientHostStaging(System.IntPtr host, int floats,
        out string? failure)
    {
        failure = "this backend has no page-locked host registration";
        return null;
    }
}
