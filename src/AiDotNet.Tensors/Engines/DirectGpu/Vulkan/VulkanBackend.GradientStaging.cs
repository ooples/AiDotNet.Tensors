// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>No page-locked gradient staging on this backend: the exchange uses its per-buffer download and upload.</summary>
public sealed unsafe partial class VulkanBackend : IGradientHostStagingSource
{
    IGradientHostStaging? IGradientHostStagingSource.TryCreateGradientHostStaging(int floats) => null;
}
