// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DevicePrimitives;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IDeviceMemoryAccounted
{
    /// <inheritdoc/>
    DeviceMemoryAccount IDeviceMemoryAccounted.MemoryAccount => WebGpuBuffer.Account;
}
#endif
