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
    /// device's per-allocation cap.
    /// </summary>
    /// <param name="backend">Backend identifier ("OpenCL", "CUDA", …) for diagnostics.</param>
    /// <param name="requestedBytes">Size of the allocation in bytes.</param>
    /// <param name="cap">
    /// Per-allocation cap in bytes. Pass 0 when the cap is unknown / not
    /// queryable; the guard then skips validation and the native call's
    /// error (if any) propagates naturally.
    /// </param>
    /// <param name="deviceName">Human-readable device name for diagnostics.</param>
    /// <exception cref="GpuBufferTooLargeException">
    /// Thrown when <paramref name="requestedBytes"/> exceeds <paramref name="cap"/>.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void EnsureFits(
        string backend,
        long requestedBytes,
        long cap,
        string deviceName)
    {
        if (cap > 0 && requestedBytes > 0 && requestedBytes > cap)
        {
            throw new GpuBufferTooLargeException(
                backend: backend,
                requestedBytes: requestedBytes,
                deviceMaxAllocBytes: cap,
                deviceName: deviceName);
        }
    }
}
