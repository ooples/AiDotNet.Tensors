// Copyright (c) AiDotNet. All rights reserved.
// Saved-tensor hooks: pack/unpack interception for tensors saved by
// backward ops. Used to offload activations to cheaper storage
// (CPU memory, int8 quantization, zstd compression, delta coding).

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Thread-local stack of saved-tensor pack/unpack hook pairs. When
/// a forward op registers a tensor for backward use, the innermost
/// active hook pair transforms it into a compact representation; the
/// backward pass inverts the transformation at use time. Mirrors
/// PyTorch's <c>torch.autograd.graph.saved_tensors_hooks</c>.
/// </summary>
/// <remarks>
/// <para><b>What this solves:</b></para>
/// <para>
/// Large-batch training often runs out of memory holding per-op
/// activations for the backward pass. Saved-tensor hooks let the
/// framework swap that memory for a cheaper alternative — CPU DRAM,
/// quantized int8, delta-from-previous-step, compressed bytes —
/// transparent to the op's backward function. PyTorch ships only
/// <c>save_on_cpu</c>; this API is the integration point for
/// quantization and compression recipes that are tracked as
/// non-parity goals for this library (int1/int2/int3/int4/NF4/FP4).
/// </para>
/// <para><b>Design — delegates, not subclasses:</b></para>
/// <para>
/// Each hook is a pair of <see cref="Func{Tensor, Object}"/> delegates
/// (<c>pack</c>, <c>unpack</c>) rather than a subclass. Delegates
/// compose cleanly, have zero vtable overhead, and let a single
/// thread stack multiple hook pairs — PyTorch's semantics.
/// </para>
/// <para><b>Current integration scope:</b></para>
/// <para>
/// The hook state is maintained here and queryable via
/// <see cref="TryApplyPack{T}"/> / <see cref="ApplyUnpack{T}"/>. The
/// AiDotNet backward functions currently save tensors directly into
/// <see cref="TapeEntry{T}.SavedState"/> for raw-pointer performance;
/// wrapping every such save in a pack call would introduce a branch
/// in every backward kernel. Future work: convert the hottest ops
/// (Conv2D, MatMul, BatchNorm) to route their activation saves through
/// these hooks, which will unlock CPU-offloading and quantized
/// activation recipes without changing the backward rule bodies.
/// </para>
/// </remarks>
public static class SavedTensorHooks
{
    [ThreadStatic]
    private static Stack<(Delegate Pack, Delegate Unpack)>? _stack;

    private static Stack<(Delegate Pack, Delegate Unpack)> Stack
        => _stack ??= new Stack<(Delegate, Delegate)>();

    /// <summary>
    /// Returns true if at least one hook pair is active on this thread.
    /// Intended for op authors to skip hook processing on the hot path
    /// when no user has registered anything — zero cost when inactive.
    /// </summary>
    public static bool IsActive => _stack is { Count: > 0 };

    /// <summary>
    /// Pushes a new hook pair onto the thread-local stack. Matching
    /// <see cref="Pop"/> call restores the previous state. The
    /// returned disposable is the recommended shape — <c>using</c> the
    /// result guarantees the pop even on exception.
    /// </summary>
    /// <typeparam name="T">The tensor element type.</typeparam>
    /// <param name="pack">Called on each saved tensor at forward time —
    /// returns an opaque representation.</param>
    /// <param name="unpack">Inverse of <paramref name="pack"/> — called
    /// at backward time to recover the tensor.</param>
    /// <returns>An <see cref="IDisposable"/> whose
    /// <see cref="IDisposable.Dispose"/> pops the hook pair.</returns>
    /// <exception cref="ArgumentNullException">Thrown if either
    /// delegate is null.</exception>
    public static IDisposable Push<T>(Func<Tensor<T>, object> pack, Func<object, Tensor<T>> unpack)
    {
        if (pack is null) throw new ArgumentNullException(nameof(pack));
        if (unpack is null) throw new ArgumentNullException(nameof(unpack));
        Stack.Push(((Delegate)pack, (Delegate)unpack));
        return new PopOnDispose();
    }

    /// <summary>
    /// Removes the innermost hook pair. Normally called by the
    /// disposable returned from <see cref="Push"/>; exposed for code
    /// paths that can't use <c>using</c> (interop, try/finally
    /// patterns with restart semantics).
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the hook
    /// stack is empty.</exception>
    public static void Pop()
    {
        if (_stack is null || _stack.Count == 0)
            throw new InvalidOperationException(
                "SavedTensorHooks.Pop called with no active hook pair.");
        _stack.Pop();
    }

    /// <summary>
    /// Applies the innermost <c>pack</c> hook to
    /// <paramref name="tensor"/> if any is registered. Op authors
    /// opting into saved-tensor hook integration call this on each
    /// activation they were about to save raw into
    /// <see cref="TapeEntry{T}.SavedState"/>.
    /// </summary>
    /// <typeparam name="T">The tensor element type.</typeparam>
    /// <param name="tensor">Activation tensor to save.</param>
    /// <param name="packed">Receives the opaque packed representation
    /// when a hook is active; otherwise set to null.</param>
    /// <returns>True if a hook packed the tensor; false if the op
    /// should save the tensor directly.</returns>
    public static bool TryApplyPack<T>(Tensor<T> tensor, out object? packed)
    {
        if (_stack is null || _stack.Count == 0) { packed = null; return false; }
        var (pack, _) = _stack.Peek();
        packed = ((Func<Tensor<T>, object>)pack)(tensor);
        return true;
    }

    /// <summary>
    /// Inverts the innermost <c>pack</c> to recover a saved tensor.
    /// Called by backward ops when the object read from
    /// <see cref="TapeEntry{T}.SavedState"/> is a packed handle
    /// (produced by an earlier <see cref="TryApplyPack{T}"/> that
    /// returned true).
    /// </summary>
    /// <typeparam name="T">The tensor element type.</typeparam>
    /// <param name="packed">The packed representation.</param>
    /// <returns>The recovered tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown if no hook
    /// pair is active — the backward cannot recover the tensor
    /// without a matching unpack.</exception>
    public static Tensor<T> ApplyUnpack<T>(object packed)
    {
        if (_stack is null || _stack.Count == 0)
            throw new InvalidOperationException(
                "SavedTensorHooks.ApplyUnpack called with no active hook pair.");
        var (_, unpack) = _stack.Peek();
        return ((Func<object, Tensor<T>>)unpack)(packed);
    }

    private sealed class PopOnDispose : IDisposable
    {
        private bool _disposed;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Pop();
        }
    }
}

/// <summary>
/// Built-in saved-tensor recipes — ready-to-use pack/unpack pairs for
/// the common offloading strategies. Each recipe is a static factory
/// returning the <c>IDisposable</c> guard from
/// <see cref="SavedTensorHooks.Push{T}"/>; <c>using</c> the result
/// activates the recipe for the scope's duration.
/// </summary>
public static class SavedTensorRecipes
{
    /// <summary>
    /// Saves activations as a plain CPU <c>T[]</c> copy. The current
    /// <see cref="Tensor{T}"/> is already CPU-backed, so this is
    /// effectively a defensive copy that prevents mutation of the
    /// original during backward — useful for tapes that outlive the
    /// forward pass. The analogous "offload GPU activations to host
    /// DRAM" variant plugs in here once GPU-backed tensors land a
    /// shared storage interface.
    /// </summary>
    public static IDisposable SaveOnCpu<T>()
        => SavedTensorHooks.Push<T>(
            pack: tensor =>
            {
                var data = new T[tensor.Length];
                tensor.AsSpan().CopyTo(data);
                return new CpuSaved<T>(data, (int[])tensor._shape.Clone());
            },
            unpack: packed =>
            {
                var saved = (CpuSaved<T>)packed;
                return new Tensor<T>(saved.Data, saved.Shape);
            });

    /// <summary>
    /// Activates int8 quantized activation saving for float tensors.
    /// Pack: per-tensor symmetric min-max quantization to a byte[]
    /// plus a scale factor. Unpack: dequantize to float. Achieves a
    /// 4× memory reduction at the cost of ≤0.5% activation error for
    /// typical CNN/transformer activations.
    /// </summary>
    /// <remarks>
    /// Only registered for <c>float</c> tensors — for other element
    /// types the caller should use <see cref="SaveOnCpu{T}"/> or a
    /// type-specific recipe.
    /// </remarks>
    public static IDisposable SaveQuantizedInt8()
        => SavedTensorHooks.Push<float>(
            pack: tensor =>
            {
                var data = tensor.AsSpan();
                float absMax = 0f;
                for (int i = 0; i < data.Length; i++)
                {
                    var v = data[i];
                    if (v < 0) v = -v;
                    if (v > absMax) absMax = v;
                }
                float scale = absMax > 0f ? absMax / 127f : 1f;
                var quant = new sbyte[data.Length];
                float invScale = 1f / scale;
                for (int i = 0; i < data.Length; i++)
                {
                    var q = data[i] * invScale;
                    if (q > 127f) q = 127f; else if (q < -127f) q = -127f;
                    quant[i] = (sbyte)(q >= 0 ? q + 0.5f : q - 0.5f);
                }
                return new QuantizedSaved(quant, scale, (int[])tensor._shape.Clone());
            },
            unpack: packed =>
            {
                var saved = (QuantizedSaved)packed;
                var data = new float[saved.Data.Length];
                for (int i = 0; i < data.Length; i++) data[i] = saved.Data[i] * saved.Scale;
                return new Tensor<float>(data, saved.Shape);
            });

    private sealed class CpuSaved<T>
    {
        public T[] Data { get; }
        public int[] Shape { get; }
        public CpuSaved(T[] data, int[] shape) { Data = data; Shape = shape; }
    }

    private sealed class QuantizedSaved
    {
        public sbyte[] Data { get; }
        public float Scale { get; }
        public int[] Shape { get; }
        public QuantizedSaved(sbyte[] data, float scale, int[] shape)
        {
            Data = data; Scale = scale; Shape = shape;
        }
    }
}
