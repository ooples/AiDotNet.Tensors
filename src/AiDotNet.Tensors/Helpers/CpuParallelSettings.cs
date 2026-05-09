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
    /// Gets or sets whether AVX2 hardware gather instructions are used for strided memory access.
    /// </summary>
    /// <remarks>
    /// AVX2 VGATHERDPS/VPGATHERDD can gather 8 floats in a single instruction using an index vector.
    /// This provides significant speedup for wavelet transforms, FFT butterfly patterns, and
    /// interleaved channel separation. However, some AMD processors (pre-Zen 3) have slow
    /// gather implementations that can be slower than scalar loops. Set to false to force
    /// scalar fallback on such hardware.
    /// </remarks>
    public static bool EnableAvx2Gather { get; set; } = true;

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
    /// High-performance parallel execution using pre-spawned worker threads.
    /// Near-zero dispatch overhead — threads are already idle and wake instantly.
    /// Mimics libtorch's OpenMP thread pool pattern for maximum throughput.
    /// </summary>
    /// <param name="numChunks">Number of chunks to process in parallel.</param>
    /// <param name="action">Action receiving chunk index (0..numChunks-1).</param>
    public static void LightweightParallel(int numChunks, Action<int> action)
    {
        PersistentParallelExecutor.Instance.Execute(numChunks, action);
    }

    /// <summary>
    /// Grain-size-aware variant: forwards to
    /// <see cref="PersistentParallelExecutor.Execute(int, long, Action{int})"/>
    /// so callers that know their op's elementwise work can engage
    /// PyTorch-style serial-fallback below
    /// <see cref="PersistentParallelExecutor.DefaultSerialGrainSize"/>.
    /// Issue #319 follow-up to PR #316: kernels that pass a real
    /// totalWork value skip dispatch overhead on small ops.
    /// </summary>
    /// <param name="numChunks">Number of chunks to process in parallel.</param>
    /// <param name="totalWork">Total elementwise work the action will
    /// perform across all chunks combined.</param>
    /// <param name="action">Action receiving chunk index (0..numChunks-1).</param>
    public static void LightweightParallel(int numChunks, long totalWork, Action<int> action)
    {
        PersistentParallelExecutor.Instance.Execute(numChunks, totalWork, action);
    }

    /// <summary>
    /// Grain-size-aware drop-in for <see cref="System.Threading.Tasks.Parallel.For(int, int, Action{int})"/>.
    /// When <paramref name="totalWork"/> is below
    /// <see cref="PersistentParallelExecutor.DefaultSerialGrainSize"/>,
    /// runs the body inline on the calling thread — no ThreadPool
    /// dispatch, no <c>LowLevelLifoSemaphore</c> wait. Above the
    /// threshold, falls through to <c>Parallel.For</c>.
    ///
    /// <para>Issue #319 background: the original profile showed 42.59%
    /// of ViT-Base CPU train wall-clock in
    /// <c>LowLevelLifoSemaphore.WaitForSignal</c> — that's the .NET
    /// ThreadPool primitive used by every <c>Parallel.For</c> call.
    /// Hundreds of small-op call sites in <c>CpuEngine</c> dispatch
    /// to <c>Parallel.For</c> unconditionally even when the work is
    /// smaller than the dispatch overhead itself. This helper is the
    /// migration vehicle: each call site swaps
    /// <c>Parallel.For(0, n, body)</c> for
    /// <c>ParallelForOrSerial(0, n, totalWork, body)</c> and the
    /// JIT eliminates the dispatch on workloads below threshold.</para>
    /// </summary>
    /// <param name="fromInclusive">First index, inclusive.</param>
    /// <param name="toExclusive">One past last index, exclusive.</param>
    /// <param name="totalWork">Total elementwise work the body will
    /// perform across all iterations combined.</param>
    /// <param name="body">Iteration body — same shape as
    /// <c>Parallel.For</c>'s <c>Action&lt;int&gt;</c>.</param>
    public static void ParallelForOrSerial(int fromInclusive, int toExclusive, long totalWork, Action<int> body)
    {
        if (toExclusive <= fromInclusive) return;
        if (totalWork < PersistentParallelExecutor.DefaultSerialGrainSize)
        {
            // Match Parallel.For's exception semantics: capture
            // first thrown exception, finish remaining iterations,
            // re-throw at end. Parallel.For uses
            // AggregateException — for serial-mode we re-throw the
            // raw first exception (consistent with the
            // PersistentParallelExecutor.Execute serial path).
            Exception? firstException = null;
            for (int i = fromInclusive; i < toExclusive; i++)
            {
                try { body(i); }
                catch (Exception ex) { firstException ??= ex; }
            }
            if (firstException is not null) throw firstException;
            return;
        }
        System.Threading.Tasks.Parallel.For(fromInclusive, toExclusive, body);
    }
}
