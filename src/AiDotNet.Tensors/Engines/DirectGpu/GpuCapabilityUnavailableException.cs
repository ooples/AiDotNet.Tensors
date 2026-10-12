using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A GPU backend lacks a capability the call needs: the kernel is not compiled for this device, the backend or one of its
/// runtimes (a context, a direct-PTX runtime, a shader compiler) is not available on this host, or the backend does not
/// implement the operation for this element type. Unlike a kernel fault, a device error or an out-of-memory condition,
/// this is a deliberate, recoverable gap: <see cref="DirectGpuTensorEngine"/> answers it by running the operation on the
/// CPU and recording the fallback (#1165). Every other exception from a GPU dispatch propagates to the caller.
/// </summary>
public sealed class GpuCapabilityUnavailableException : NotSupportedException
{
    /// <summary>Creates the exception with a message naming the missing capability.</summary>
    public GpuCapabilityUnavailableException(string message) : base(message) { }

    /// <summary>Creates the exception with a message and the failure that revealed the gap.</summary>
    public GpuCapabilityUnavailableException(string message, Exception innerException) : base(message, innerException) { }
}
