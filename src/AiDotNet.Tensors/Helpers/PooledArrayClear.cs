using System;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Zero-fill for rented buffers that honours <c>Rent</c>'s zero-initialization contract without serialising on
/// one core.
/// </summary>
/// <remarks>
/// A buffer handed back to the pool was usually just written by a parallel kernel, so its cache lines sit dirty in
/// the workers' caches, many on other CCXs. Clearing it on the renting thread alone pulled every line across the
/// fabric: renting a [512, 1024] float buffer right after a parallel softmax wrote it took ~270 us in the clear,
/// against 24 us when the lines were local. Above <see cref="ParallelMinBytes"/> the clear is split with the
/// elementwise chunking, so each worker zeroes lines it likely already holds.
/// </remarks>
internal static class PooledArrayClear
{
    /// <summary>Smallest clear, in bytes, that is split across workers.</summary>
    internal const long ParallelMinBytes = 1L << 20;

    [ThreadStatic] private static int t_lastChunks;

    /// <summary>
    /// How many pieces the calling thread's last <see cref="Clear"/> actually ran as: 1 for a serial clear, else the
    /// chunks the dispatcher executed (it runs a nested call serially, as one). Lets tests assert the dispatch, not only
    /// the zeroed range, which a serial clear produces too.
    /// </summary>
    internal static int LastClearChunks => t_lastChunks;

    internal static void Clear<T>(T[] array, int start, int length)
    {
        long bytes = (long)length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        int chunks = bytes >= ParallelMinBytes ? CpuParallelSettings.ElementwiseChunkCount(length) : 1;
        if (chunks < 2)
        {
            Array.Clear(array, start, length);
            t_lastChunks = 1;
            return;
        }
        int grain = (length + chunks - 1) / chunks;
        int ran = 0;
        CpuParallelSettings.ParallelForChunks(length, grain, (s, n) =>
        {
            Array.Clear(array, start + s, n);
            System.Threading.Interlocked.Increment(ref ran);
        });
        t_lastChunks = ran;
    }
}
