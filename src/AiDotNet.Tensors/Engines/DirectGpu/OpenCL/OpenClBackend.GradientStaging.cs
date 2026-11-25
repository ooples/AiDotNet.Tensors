// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>No page-locked gradient staging on this backend: the exchange uses its per-buffer download and upload.</summary>
public sealed partial class OpenClBackend : IGradientHostStagingSource
{
    IGradientHostStaging? IGradientHostStagingSource.TryCreateGradientHostStaging(int floats) => null;
}
