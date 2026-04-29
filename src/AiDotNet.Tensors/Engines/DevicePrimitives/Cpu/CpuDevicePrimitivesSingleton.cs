// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// Process-wide singleton for the managed CPU implementation of
/// <see cref="IDevicePrimitives"/>. Used by the default
/// <c>IEngine.DevicePrimitives</c> getter so every engine has a usable
/// primitive surface without each engine allocating its own instance —
/// <see cref="CpuDevicePrimitives"/> holds no per-instance state, so
/// sharing a single instance is correct and avoids per-call allocation
/// at the discovery path.
/// </summary>
internal static class CpuDevicePrimitivesSingleton
{
    /// <summary>The shared instance.</summary>
    public static readonly IDevicePrimitives Instance = new CpuDevicePrimitives();
}
