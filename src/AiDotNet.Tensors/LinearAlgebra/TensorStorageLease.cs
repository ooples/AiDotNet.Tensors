using System;
using System.Threading;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// The CPU array behind a tensor of any element type, for proving when a step's last reader of it has run.
/// </summary>
internal interface IBackingArraySource
{
    /// <summary>
    /// The backing array (null when the data is not array-backed). False while the tensor is a deferred value whose
    /// realization would read other buffers, which this cannot name.
    /// </summary>
    bool TryGetBackingArray(out Array? array);
}

/// <summary>
/// Type-erased source for a reference-counted tensor-storage lease. Lazy graphs use this
/// contract to retain mixed-element-type operands without reflection or element-type strings.
/// </summary>
internal interface ITensorStorageLeaseSource
{
    object StorageIdentity { get; }
    TensorStorageLease AcquireStorageLease();
    void SetTapePinned(bool pinned);
}

/// <summary>
/// Type-safe access to tensor-bearing values hidden inside an opaque saved-state object.
/// </summary>
internal interface ISavedStateTensorContainer
{
    IReadOnlyList<object> SavedStateValues { get; }
}

internal interface ISavedStateTensorVisitor
{
    void Visit(ITensorStorageLeaseSource tensor);
}

/// <summary>
/// Walks the tensor-bearing shapes supported by saved state: direct tensors, nested
/// reference-type arrays, and explicit tensor containers such as AutogradContext.
/// Primitive arrays are deliberately ignored.
/// </summary>
internal static class SavedStateTensorTraversal
{
    internal static void Visit<TVisitor>(object[]? savedState, ref TVisitor visitor)
        where TVisitor : struct, ISavedStateTensorVisitor
    {
        if (savedState is null || savedState.Length == 0) return;
        var pending = new Stack<object>();
        var seen = new HashSet<object>(Engines.Autodiff.ReferenceEqualityComparer<object>.Instance);
        pending.Push(savedState);
        while (pending.Count != 0)
        {
            var value = pending.Pop();
            if (value is ITensorStorageLeaseSource tensor)
            {
                visitor.Visit(tensor);
                continue;
            }
            if (!seen.Add(value)) continue;
            // Explicit worklist avoids stack overflow for both cycles and deeply nested state.
            if (value is Array array && array.GetType().GetElementType() is { IsValueType: false })
            {
                foreach (var item in array)
                    if (item is not null) pending.Push(item);
            }
            else if (value is ISavedStateTensorContainer container)
            {
                foreach (var item in container.SavedStateValues)
                    if (item is not null) pending.Push(item);
            }
        }
    }

}

/// <summary>A type-erased, independently disposable reference to one tensor storage.</summary>
internal abstract class TensorStorageLease : IDisposable
{
    internal abstract object StorageIdentity { get; }
    public abstract void Dispose();
}

/// <summary>Strong reference-counted lease over a closed generic tensor storage.</summary>
internal sealed class TensorStorageLease<T> : TensorStorageLease
{
    private TensorStorage<T>? _storage;

    internal TensorStorageLease(TensorStorage<T> storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        storage.AddRef();
    }

    internal override object StorageIdentity =>
        _storage ?? throw new ObjectDisposedException(nameof(TensorStorageLease<T>));

    public override void Dispose()
    {
        Interlocked.Exchange(ref _storage, null)?.Release();
    }
}
