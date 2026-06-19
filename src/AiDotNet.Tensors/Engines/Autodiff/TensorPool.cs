using System.Collections.Concurrent;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Thread-safe tensor pool for recycling backward tensors during gradient computation.
/// Reduces GC pressure by reusing tensor allocations across backward passes.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>During backward passes, each operation creates gradient tensors that are typically
/// used once then discarded. This pool keeps recently-used tensors and recycles them
/// for the next backward pass, avoiding thousands of small allocations per training step.</para>
/// <para>The pool is thread-local to avoid contention in multi-threaded training.</para>
/// </remarks>
public static class TensorPool<T>
{
    /// <summary>
    /// Per-length pool: maps tensor element count to a stack of reusable tensors.
    /// Thread-local to avoid contention.
    /// </summary>
    [ThreadStatic]
    private static Dictionary<int, Stack<Tensor<T>>>? _pools;

    /// <summary>
    /// Maximum number of tensors to keep per size bucket.
    /// </summary>
    private const int MaxPooledPerSize = 8;

    /// <summary>
    /// Maximum total pooled tensors across all sizes.
    /// </summary>
    private const int MaxTotalPooled = 64;

    [ThreadStatic]
    private static int _totalPooled;

    /// <summary>
    /// Rents a tensor with the specified shape from the pool, or allocates a new one.
    /// The tensor data is NOT zeroed — caller must initialize if needed.
    /// </summary>
    /// <param name="shape">The shape of the tensor to rent.</param>
    /// <returns>A tensor with the specified shape. Data may contain stale values.</returns>
    public static Tensor<T> Rent(int[] shape)
    {
        _pools ??= new Dictionary<int, Stack<Tensor<T>>>();

        int length = 1;
        foreach (int d in shape) length *= d;

        if (_pools.TryGetValue(length, out var stack) && stack.Count > 0)
        {
            var tensor = stack.Pop();
            _totalPooled--;
            // Reshape if shape differs but length matches
            if (!ShapeEquals(tensor._shape, shape))
                tensor = tensor.Reshape(shape);
            return tensor;
        }

        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Rents a tensor and fills it with zeros.
    /// </summary>
    public static Tensor<T> RentZeroed(int[] shape)
    {
        var tensor = Rent(shape);
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = numOps.Zero;
        return tensor;
    }

    /// <summary>
    /// Returns a tensor to the pool for reuse.
    /// </summary>
    /// <param name="tensor">The tensor to return. Should not be used after this call.</param>
    public static void Return(Tensor<T> tensor)
    {
        if (tensor == null || tensor.Length == 0) return;
        // Issue #338 tape-pinning: tensors recorded as inputs (or
        // saved-state) on an active GradientTape must not be reissued as
        // scratch — a later backward op will still consume them. Without
        // this guard, PR #331's per-op pooling attempt regressed
        // Hvp_MatchesHessianTimesVec. DifferentiableOps.Record* sets the
        // pin; the tape's cleanup walk clears it after backward.
        if (tensor._pinnedByTape) return;

        // PR #638: a GPU-RESIDENT tensor (BindResidentBuffer set _gpuBuffer + a deferred host-read materializer)
        // must NOT be pooled — but the GetLiveBackingArrayOrNull() discriminator below calls GetDataArray(), which
        // FIRES that materializer (a DtoH download). During CUDA-graph capture that download is a non-capturable
        // CUDA-900 that aborts the whole-step capture (the LAST blocker after every backward grad went resident:
        // NegateBackward pool-returns its resident negGrad). Reject resident tensors here via the _gpuBuffer FIELD
        // (no materialization) before the downloading check.
        if (tensor._gpuBuffer is not null) return;

        // Issue #338 view-safety: refuse view-tensors whose backing
        // storage is shared with another tensor (strided permute views,
        // reshape views, and offset/sliced views). Recycling such a
        // tensor would let the next Rent caller write through the
        // shared backing into the source tensor — a silent correctness
        // bug exposed by PR #331's attempt to pool TransposeLastTwoDims
        // results (which call TensorPermute on rank-3+ inputs, producing
        // strided views). GetLiveBackingArrayOrNull returns null for
        // non-owned layouts (non-contiguous, _storageOffset != 0, or
        // storage length != logical length), giving exactly the
        // discriminator we need. CPU-only — GPU-resident tensors also
        // fail this check so we don't pool device buffers.
        if (tensor.GetLiveBackingArrayOrNull() is null) return;

        if (_totalPooled >= MaxTotalPooled) return;

        _pools ??= new Dictionary<int, Stack<Tensor<T>>>();

        if (!_pools.TryGetValue(tensor.Length, out var stack))
        {
            stack = new Stack<Tensor<T>>();
            _pools[tensor.Length] = stack;
        }

        if (stack.Count < MaxPooledPerSize)
        {
            stack.Push(tensor);
            _totalPooled++;
        }
    }

    /// <summary>
    /// Clears all pooled tensors. Call at the end of training or when switching models.
    /// </summary>
    /// <summary>
    /// Clears the pool for the calling thread. Each thread has its own pool
    /// (ThreadStatic), so this only affects the current thread's cache.
    /// Call from each worker thread to clear all caches.
    /// </summary>
    public static void Clear()
    {
        _pools?.Clear();
        _totalPooled = 0;
    }

    private static bool ShapeEquals(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }
}
