using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Training;

/// <summary>
/// Multi-slot persistent-input registry for compiled-plan replay. Callers register
/// named slots (each is a <see cref="Tensor{T}"/> allocated once), refresh their
/// data per training step, and the compiled plan's captured references stay stable
/// across steps — so the plan re-reads fresh data on every replay without a recompile.
///
/// <para>Motivating case: batched-per-element diffusion training (Ho et al. 2020,
/// HuggingFace diffusers). Each training step provides three refreshed tensors —
/// the noisy sample, the target noise, and the per-batch-element timesteps — and
/// the compiled plan's forward closure reads all three. The prior two-slot mechanism
/// (input + target only) couldn't support the third tensor without packing tricks
/// that hurt readability and correctness. This registry generalizes to N slots.</para>
///
/// <para>Also useful for models with more than one conditioning stream (text +
/// image + timestep for classifier-free-guidance-style diffusion, or multiple
/// pooled conditioning inputs in TFT-style forecasters).</para>
/// </summary>
/// <typeparam name="T">Numeric type of the persistent tensors.</typeparam>
public sealed class PersistentInputRegistry<T> : IDisposable
{
    // Slot storage. Each entry is the persistent tensor whose reference the compiled
    // plan captures; RefreshSlot copies new data into it in place. Slot indices are
    // stable across the registry's lifetime — callers cache the returned int to
    // refresh a specific slot without a name lookup on the hot path.
    private readonly List<Tensor<T>> _slots = new();
    private bool _disposed;

    /// <summary>Number of slots currently registered.</summary>
    public int SlotCount => _slots.Count;

    /// <summary>
    /// Allocates a new persistent-input slot with the given shape and returns
    /// (a) the persistent tensor reference the caller passes to their forward
    /// closure / plan trace, and (b) the slot index used to refresh the tensor's
    /// data in subsequent steps.
    /// </summary>
    /// <param name="shape">Shape of the tensor. Must not be null or contain
    /// negative dimensions.</param>
    /// <returns>Tuple of (persistent tensor, stable slot index).</returns>
    public (Tensor<T> tensor, int slotIndex) Register(int[] shape)
    {
        ThrowIfDisposed();
        if (shape is null)
            throw new ArgumentNullException(nameof(shape));
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] <= 0)
                throw new ArgumentException(
                    $"Slot shape must have positive dimensions; shape[{i}] = {shape[i]}.",
                    nameof(shape));
        }
        var t = new Tensor<T>(shape);
        int idx = _slots.Count;
        _slots.Add(t);
        return (t, idx);
    }

    /// <summary>
    /// Refreshes slot <paramref name="slotIndex"/> with the data in <paramref name="freshData"/>.
    /// The persistent tensor at the slot keeps its identity — the compiled plan's
    /// captured reference remains valid; only the tensor's backing data is overwritten.
    /// </summary>
    /// <param name="slotIndex">Slot index returned by <see cref="Register"/>.</param>
    /// <param name="freshData">New data. Must have the same total element count as
    /// the slot's tensor; layout is copied contiguously via span.</param>
    /// <exception cref="ArgumentOutOfRangeException">slotIndex out of range.</exception>
    /// <exception cref="ArgumentException">freshData length differs from the slot's element count.</exception>
    public void RefreshSlot(int slotIndex, Tensor<T> freshData)
    {
        ThrowIfDisposed();
        if (slotIndex < 0 || slotIndex >= _slots.Count)
            throw new ArgumentOutOfRangeException(nameof(slotIndex), slotIndex,
                $"Slot index out of range; SlotCount = {SlotCount}.");
        if (freshData is null)
            throw new ArgumentNullException(nameof(freshData));
        var slot = _slots[slotIndex];
        if (freshData.Length != slot.Length)
            throw new ArgumentException(
                $"freshData length {freshData.Length} does not match slot {slotIndex}'s length {slot.Length}.",
                nameof(freshData));
        freshData.AsSpan().CopyTo(slot.AsWritableSpan());
    }

    /// <summary>
    /// Refreshes slot <paramref name="slotIndex"/> from a <see cref="Vector{T}"/> —
    /// same semantics as the tensor overload but avoids an intermediate Tensor
    /// allocation when the caller already has a Vector-shaped source.
    /// </summary>
    public void RefreshSlot(int slotIndex, Vector<T> freshData)
    {
        ThrowIfDisposed();
        if (slotIndex < 0 || slotIndex >= _slots.Count)
            throw new ArgumentOutOfRangeException(nameof(slotIndex), slotIndex,
                $"Slot index out of range; SlotCount = {SlotCount}.");
        if (freshData is null)
            throw new ArgumentNullException(nameof(freshData));
        var slot = _slots[slotIndex];
        if (freshData.Length != slot.Length)
            throw new ArgumentException(
                $"freshData length {freshData.Length} does not match slot {slotIndex}'s length {slot.Length}.",
                nameof(freshData));
        var src = freshData.AsSpan();
        var dst = slot.AsWritableSpan();
        src.CopyTo(dst);
    }

    /// <summary>
    /// Returns the persistent tensor at the given slot. Callers rarely need this
    /// (they cache the tensor reference from <see cref="Register"/>), but it's
    /// available for diagnostics and debugging.
    /// </summary>
    public Tensor<T> GetSlot(int slotIndex)
    {
        ThrowIfDisposed();
        if (slotIndex < 0 || slotIndex >= _slots.Count)
            throw new ArgumentOutOfRangeException(nameof(slotIndex), slotIndex,
                $"Slot index out of range; SlotCount = {SlotCount}.");
        return _slots[slotIndex];
    }

    /// <summary>
    /// Clears all registered slots. After this call every previously-returned slot
    /// index is invalid. Compiled plans holding references to the cleared slots'
    /// tensors will silently read stale data — callers must invalidate their plans
    /// before clearing (mirrors the input-shape-changed invalidation contract).
    /// </summary>
    public void Clear()
    {
        ThrowIfDisposed();
        _slots.Clear();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _slots.Clear();
        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(PersistentInputRegistry<T>));
    }
}
