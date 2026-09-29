// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The CUDA stream the calling thread's current backend operation runs on (PyTorch's "current stream"). Set by
/// <see cref="CudaBackend"/>'s context scope for the duration of each operation.
/// </summary>
/// <remarks>
/// Every backend on a device shares the device's primary context. A synchronous driver copy or memset runs on the
/// legacy default stream, which implicitly waits on every blocking stream in the context -- so while any other
/// engine (another thread) was capturing a CUDA graph, it failed with CUDA_ERROR_STREAM_CAPTURE_IMPLICIT (906).
/// The legacy-stream wrappers in <see cref="CuBlasNative"/> therefore issue the stream-ordered form on this stream
/// instead, which orders the copy after the backend's own queued work and touches no other stream.
/// </remarks>
internal static class CudaCurrentStream
{
    [ThreadStatic] private static IntPtr t_stream;
    [ThreadStatic] private static IntPtr t_context;

    /// <summary>Makes <paramref name="stream"/> (belonging to <paramref name="context"/>) current; returns the prior pair.</summary>
    internal static (IntPtr Stream, IntPtr Context) Enter(IntPtr stream, IntPtr context)
    {
        var previous = (t_stream, t_context);
        t_stream = stream;
        t_context = context;
        return previous;
    }

    internal static void Restore((IntPtr Stream, IntPtr Context) previous)
    {
        t_stream = previous.Stream;
        t_context = previous.Context;
    }

    /// <summary>
    /// The current stream when it belongs to the thread's current context, else zero (a caller that pushed a
    /// different context, like the direct-PTX runtime's private one, keeps the legacy synchronous behavior).
    /// </summary>
    internal static IntPtr ForCurrentContext()
    {
        var stream = t_stream;
        if (stream == IntPtr.Zero) return IntPtr.Zero;
        return CudaNativeBindings.cuCtxGetCurrent(out var current) == CudaResult.Success && current == t_context
            ? stream
            : IntPtr.Zero;
    }
}
