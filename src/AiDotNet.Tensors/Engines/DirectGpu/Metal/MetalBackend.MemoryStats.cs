// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.DevicePrimitives;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IDeviceMemoryAccounted
{
    /// <inheritdoc/>
    DeviceMemoryAccount IDeviceMemoryAccounted.MemoryAccount => _device.MemoryAccount;
}
