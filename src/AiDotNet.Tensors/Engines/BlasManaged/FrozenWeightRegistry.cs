using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-E (#373): per-process registry of frozen weight tensors that have been
/// pre-packed for fast inference. Callers register a tensor once; subsequent
/// MatMul calls auto-detect the registration and consume the pre-pack handle
/// instead of re-packing the weight every call.
///
/// <para>
/// Storage is a <see cref="ConditionalWeakTable{TKey,TValue}"/> keyed by tensor
/// identity, so registered entries are automatically reclaimed when the tensor
/// becomes unreachable — no explicit unregistration needed for the common
/// "tensor falls out of scope" case. Callers MAY call
/// <see cref="Unregister{T}"/> for prompt cleanup.
/// </para>
///
/// <para>
/// Thread safety: <see cref="ConditionalWeakTable{TKey,TValue}"/> is documented
/// as thread-safe for all members. Registry contents may be read concurrently
/// during inference from worker threads without external locking.
/// </para>
/// </summary>
public static class FrozenWeightRegistry
{
    // Separate tables per element type — Tensor<float> and Tensor<double> have
    // different types, so a single CWT keyed by `object` would still need a
    // dispatch. Two tables keep the API generic and avoid boxing.
    private static readonly ConditionalWeakTable<Tensor<float>, WeightPackHandle> _floatTable
        = new ConditionalWeakTable<Tensor<float>, WeightPackHandle>();
    private static readonly ConditionalWeakTable<Tensor<double>, WeightPackHandle> _doubleTable
        = new ConditionalWeakTable<Tensor<double>, WeightPackHandle>();

    /// <summary>
    /// Register <paramref name="weight"/> as a frozen weight matrix and pre-pack
    /// it into a multi-panel <see cref="WeightPackHandle"/>. Subsequent
    /// <c>CpuEngine.TensorMatMul</c> calls that use <paramref name="weight"/> as
    /// the right-hand-side B operand will automatically consume the pre-pack.
    ///
    /// <para>
    /// <paramref name="weight"/> must be rank-2 row-major [K, N] and float or
    /// double. Existing registration for the same tensor is overwritten.
    /// </para>
    /// </summary>
    /// <typeparam name="T">Element type. Must be float or double.</typeparam>
    /// <param name="weight">The frozen weight tensor. Caller guarantees no in-place mutation; if mutation is needed, call <see cref="MarkDirty{T}"/> first.</param>
    public static void Register<T>(Tensor<T> weight)     {
        if (weight == null) throw new ArgumentNullException(nameof(weight));
        if (weight.Rank != 2)
            throw new ArgumentException($"FrozenWeightRegistry.Register requires rank-2 tensor (got rank {weight.Rank}).", nameof(weight));

        int k = weight._shape[0];
        int n = weight._shape[1];

        if (typeof(T) == typeof(float))
        {
            var data = (float[])(object)weight.GetDataArray();
            var handle = BlasManaged.PrePackB<float>(data, n, transB: false, k: k, n: n);
            var floatWeight = (Tensor<float>)(object)weight;
            // CWT.AddOrUpdate is .NET 6+; for compat AddBefore pattern:
            _floatTable.Remove(floatWeight);
            _floatTable.Add(floatWeight, handle);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            var data = (double[])(object)weight.GetDataArray();
            var handle = BlasManaged.PrePackB<double>(data, n, transB: false, k: k, n: n);
            var doubleWeight = (Tensor<double>)(object)weight;
            _doubleTable.Remove(doubleWeight);
            _doubleTable.Add(doubleWeight, handle);
            return;
        }

        throw new NotSupportedException(
            $"FrozenWeightRegistry supports float and double only (got {typeof(T).Name}).");
    }

    /// <summary>
    /// Return the pre-pack handle for <paramref name="weight"/> if registered,
    /// else null. Consumers (CpuEngine.TensorMatMul) call this on every
    /// candidate B operand — the lookup is O(1) and lock-free under
    /// <see cref="ConditionalWeakTable{TKey,TValue}"/>.
    /// </summary>
    public static WeightPackHandle? TryGetHandle<T>(Tensor<T> weight)     {
        if (weight == null) return null;
        if (typeof(T) == typeof(float))
        {
            var floatWeight = (Tensor<float>)(object)weight;
            if (_floatTable.TryGetValue(floatWeight, out var h)) return h;
            return null;
        }
        if (typeof(T) == typeof(double))
        {
            var doubleWeight = (Tensor<double>)(object)weight;
            if (_doubleTable.TryGetValue(doubleWeight, out var h)) return h;
            return null;
        }
        return null;
    }

    /// <summary>
    /// Mark a registered tensor's pre-pack as stale (e.g., after in-place
    /// modification of the weight buffer). The next consume will fall back to
    /// live pack; a subsequent <see cref="Register{T}"/> rebuilds the pack.
    /// </summary>
    public static void MarkDirty<T>(Tensor<T> weight)     {
        var handle = TryGetHandle(weight);
        handle?.MarkDirty();
    }

    /// <summary>
    /// Drop the registration for <paramref name="weight"/>. After unregister, the
    /// next TensorMatMul call with this tensor as B will use the regular
    /// (non-pre-packed) path.
    /// </summary>
    public static bool Unregister<T>(Tensor<T> weight)     {
        if (weight == null) return false;
        if (typeof(T) == typeof(float))
        {
            return _floatTable.Remove((Tensor<float>)(object)weight);
        }
        if (typeof(T) == typeof(double))
        {
            return _doubleTable.Remove((Tensor<double>)(object)weight);
        }
        return false;
    }
}
