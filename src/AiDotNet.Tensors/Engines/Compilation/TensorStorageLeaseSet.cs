using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Deduplicated ownership of the storages retained by a live graph or compiled plan.
/// Tensor objects are graph metadata; this set independently retains the backing storage so
/// disposing a composite operation's local tensor cannot invalidate a recorded node.
/// </summary>
internal sealed class TensorStorageLeaseSet : IDisposable
{
    private readonly struct LeaseVisitor : ISavedStateTensorVisitor
    {
        private readonly TensorStorageLeaseSet _owner;

        internal LeaseVisitor(TensorStorageLeaseSet owner) => _owner = owner;
        public void Visit(ITensorStorageLeaseSource tensor) => _owner.Add(tensor);
    }

    private sealed class IdentityComparer : IEqualityComparer<object>
    {
        internal static readonly IdentityComparer Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }

    private readonly Dictionary<object, TensorStorageLease> _leases = new(IdentityComparer.Instance);
    private bool _disposed;

    internal void Add(ITensorStorageLeaseSource tensor)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorStorageLeaseSet));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        var lease = tensor.AcquireStorageLease();
        if (!_leases.ContainsKey(lease.StorageIdentity))
        {
            _leases.Add(lease.StorageIdentity, lease);
            return;
        }

        lease.Dispose();
    }

    /// <summary>
    /// Releases this set's ownership of the tensor's current storage. Paging uses this only
    /// for the short interval in which the tensor atomically replaces its sole-owned storage.
    /// </summary>
    internal bool Remove(ITensorStorageLeaseSource tensor)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorStorageLeaseSet));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        object identity = tensor.StorageIdentity;
        if (_leases.TryGetValue(identity, out var lease))
        {
            _leases.Remove(identity);
            lease.Dispose();
            return true;
        }

        return false;
    }

    internal void Add(ILazyNode node)
    {
        if (node is null) throw new ArgumentNullException(nameof(node));
        node.AddStorageLeases(this);
    }

    internal void AddSavedState(object[]? savedState)
    {
        var visitor = new LeaseVisitor(this);
        SavedStateTensorTraversal.Visit(savedState, ref visitor);
    }

    internal void Add<T>(CompiledStep<T> step)
    {
        for (int i = 0; i < step.Inputs.Length; i++) Add(step.Inputs[i]);
        Add(step.OutputBuffer);
        AddSavedState(step.SavedState);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var lease in _leases.Values)
            lease.Dispose();
        _leases.Clear();
    }
}
