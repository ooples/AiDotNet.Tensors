using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged.Pool;

/// <summary>
/// Lock-free spin-then-park worker pool specialised for the Streaming GEMM
/// strategy. Designed for sub-µs dispatch overhead at GEMM call rates
/// (~1 dispatch / 10–50 µs). See spec section 3.
/// </summary>
internal static class StreamingWorkerPool
{
    [ThreadStatic]
    private static bool _isExecuting;

    /// <summary>
    /// Run <paramref name="action"/> across <paramref name="numChunks"/>
    /// chunks. Chunk 0 runs on the caller; chunks 1..numChunks-1 dispatch to
    /// worker threads. Returns when all chunks complete. Re-throws the first
    /// worker exception.
    /// </summary>
    public static void Dispatch(int numChunks, Action<int> action)
    {
        if (numChunks <= 0) return;
        if (numChunks == 1 || _isExecuting)
        {
            // Serial fallback: single-chunk OR reentrant call.
            for (int c = 0; c < numChunks; c++) action(c);
            return;
        }

        _isExecuting = true;
        try
        {
            // Placeholder serial implementation; replaced in A.4 with
            // actual lock-free worker dispatch.
            for (int c = 0; c < numChunks; c++) action(c);
        }
        finally { _isExecuting = false; }
    }
}
