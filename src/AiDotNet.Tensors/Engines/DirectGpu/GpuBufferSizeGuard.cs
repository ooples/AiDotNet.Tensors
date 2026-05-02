// Copyright (c) AiDotNet. All rights reserved.

using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Issue #285: cross-backend pre-allocation size check. Each backend's
/// buffer wrapper calls this from its constructor to fail fast with a
/// typed <see cref="GpuBufferTooLargeException"/> when the requested
/// allocation exceeds the device's per-allocation cap. The dispatch
/// shim in <see cref="DirectGpuTensorEngine"/> catches this and falls
/// through to chunked or CPU execution.
/// </summary>
internal static class GpuBufferSizeGuard
{
    /// <summary>
    /// Validates that a requested allocation in bytes fits under the
    /// device's per-allocation cap, AFTER applying any user override from
    /// <see cref="GpuFallbackOptionsHolder.Current"/>.
    /// </summary>
    /// <param name="backend">Backend identifier ("OpenCL", "CUDA", …) for diagnostics.</param>
    /// <param name="requestedBytes">Size of the allocation in bytes.</param>
    /// <param name="deviceCap">
    /// Device-reported per-allocation cap in bytes (from
    /// <see cref="IDirectGpuBackend.MaxBufferAllocBytes"/>). Pass 0 when the
    /// cap is unknown / not queryable; the guard then defers to the user
    /// override (<see cref="GpuFallbackOptions.MaxBufferBytes"/>) if any,
    /// or skips validation when no override is set.
    /// </param>
    /// <param name="deviceName">Human-readable device name for diagnostics.</param>
    /// <exception cref="GpuBufferTooLargeException">
    /// Thrown when <paramref name="requestedBytes"/> exceeds the effective cap
    /// (the lower of <paramref name="deviceCap"/> and the user override).
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void EnsureFits(
        string backend,
        long requestedBytes,
        long deviceCap,
        string deviceName)
    {
        // Apply user override (GpuFallbackOptions.MaxBufferBytes). At runtime
        // the override is typically null, so EffectiveMaxBufferBytes returns
        // deviceCap unchanged — single null-coalesce, no allocations.
        long effectiveCap = GpuFallbackOptionsHolder.Current.EffectiveMaxBufferBytes(deviceCap);
        if (effectiveCap > 0 && requestedBytes > 0 && requestedBytes > effectiveCap)
        {
            throw new GpuBufferTooLargeException(
                backend: backend,
                requestedBytes: requestedBytes,
                deviceMaxAllocBytes: effectiveCap,
                deviceName: deviceName);
        }
    }
}
