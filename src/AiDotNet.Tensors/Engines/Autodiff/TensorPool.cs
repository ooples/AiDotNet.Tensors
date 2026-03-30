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
            if (!ShapeEquals(tensor.Shape.ToArray(), shape))
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
