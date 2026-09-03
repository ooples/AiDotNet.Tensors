using System;
using System.Threading;

namespace AiDotNet.Tensors.LinearAlgebra;

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
        if (savedState is null) return;
        VisitValues(savedState, ref visitor);
    }

    private static void VisitValues<TVisitor>(IReadOnlyList<object> values, ref TVisitor visitor)
        where TVisitor : struct, ISavedStateTensorVisitor
    {
        for (int i = 0; i < values.Count; i++)
            VisitValue(values[i], ref visitor);
    }

    private static void VisitValue<TVisitor>(object? value, ref TVisitor visitor)
        where TVisitor : struct, ISavedStateTensorVisitor
    {
        if (value is ITensorStorageLeaseSource tensor)
        {
            visitor.Visit(tensor);
            return;
        }

        // Tensor<T>[] is covariant with object[]; primitive arrays are not and are skipped.
        if (value is object[] nested)
        {
            VisitValues(nested, ref visitor);
            return;
        }

        if (value is ISavedStateTensorContainer container)
            VisitValues(container.SavedStateValues, ref visitor);
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
