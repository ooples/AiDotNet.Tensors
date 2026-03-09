using System;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides configuration settings for CPU parallel operations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This static class holds settings that control how
/// operations are parallelized across CPU cores. The default values are
/// optimized for most modern systems.
/// </remarks>
public static class CpuParallelSettings
{
    /// <summary>
    /// Gets or sets the maximum degree of parallelism for CPU operations.
    /// </summary>
    /// <remarks>
    /// Default is Environment.ProcessorCount, which uses all available cores.
    /// Set to 1 to disable parallelism.
    /// </remarks>
    public static int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// Gets or sets whether SIMD (Single Instruction, Multiple Data) operations are enabled.
    /// </summary>
    /// <remarks>
    /// SIMD allows processing multiple data elements with a single instruction,
    /// significantly speeding up vector and matrix operations on supported hardware.
    /// </remarks>
    public static bool EnableSimd { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum array length before parallelization is applied.
    /// </summary>
    /// <remarks>
    /// For small arrays, the overhead of parallelization may exceed the benefits.
    /// Operations on arrays smaller than this threshold will run sequentially.
    /// </remarks>
    public static int ParallelThreshold { get; set; } = 50000;

    /// <summary>
    /// The minimum chunk size for parallel operations.
    /// </summary>
    /// <remarks>
    /// This ensures each parallel task processes at least this many elements
    /// to avoid excessive task creation overhead.
    /// </remarks>
    public const int MinChunkSize = 8192;

    /// <summary>
    /// Executes a parallel for loop with chunked iterations.
    /// </summary>
    /// <param name="length">Total number of elements to process.</param>
    /// <param name="minChunkSize">Minimum elements per chunk.</param>
    /// <param name="action">Action to execute for each chunk (start index, count).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method divides work into chunks and processes
    /// them in parallel across available CPU cores for better performance.
    /// </remarks>
    public static void ParallelForChunks(int length, int minChunkSize, Action<int, int> action)
    {
        if (length <= 0)
            return;

        if (action is null)
            throw new ArgumentNullException(nameof(action));

        int maxDegree = MaxDegreeOfParallelism;
        if (maxDegree <= 1 || length <= minChunkSize)
        {
            // Single-threaded execution
            action(0, length);
            return;
        }

        // Calculate number of chunks based on length and min chunk size
        int numChunks = Math.Min(maxDegree, (length + minChunkSize - 1) / minChunkSize);
        if (numChunks <= 1)
        {
            action(0, length);
            return;
        }

        int chunkSize = (length + numChunks - 1) / numChunks;

        Parallel.For(0, numChunks, new ParallelOptions { MaxDegreeOfParallelism = maxDegree }, i =>
        {
            int start = i * chunkSize;
            int count = Math.Min(chunkSize, length - start);
            if (count > 0)
            {
                action(start, count);
            }
        });
    }

    /// <summary>
    /// Lightweight parallel chunked execution using ThreadPool directly.
    /// Avoids Parallel.For overhead (~4KB allocation, scheduler state, ParallelOptions).
    /// Uses CountdownEvent for synchronization — ~10x less allocation than Parallel.For.
    /// </summary>
    /// <param name="numChunks">Number of chunks to process in parallel.</param>
    /// <param name="action">Action receiving chunk index (0..numChunks-1).</param>
    public static void LightweightParallel(int numChunks, Action<int> action)
    {
        if (numChunks <= 1)
        {
            action(0);
            return;
        }

        // Execute (numChunks - 1) chunks on ThreadPool, run one chunk on current thread
        using var cde = new CountdownEvent(numChunks - 1);
        for (int c = 1; c < numChunks; c++)
        {
            int chunkIdx = c;
            ThreadPool.UnsafeQueueUserWorkItem(_ =>
            {
                try
                {
                    action(chunkIdx);
                }
                finally
                {
                    cde.Signal();
                }
            }, null);
        }

        // Run chunk 0 on the current thread (no scheduling overhead)
        action(0);

        // Wait for all other chunks to complete
        cde.Wait();
    }
}
