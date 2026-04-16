using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Maps <see cref="Tensor{T}"/> object references to sequential integer IDs
/// for serialization. Two references to the same Tensor object get the same ID
/// (identity-based, not value-based). Thread-safe for reads after construction
/// (no concurrent writes during save — the writer is single-threaded).
/// </summary>
internal sealed class TensorIdMap<T>
{
    // ReferenceEqualityComparer uses object.ReferenceEquals, which is what we
    // need: two slices of the same storage are distinct Tensor objects and
    // get distinct IDs.
    private readonly Dictionary<Tensor<T>, int> _map = new(ReferenceEqualityComparer.Instance);
    private readonly List<Tensor<T>> _ordered = new();

    /// <summary>Number of unique tensors registered.</summary>
    internal int Count => _ordered.Count;

    /// <summary>Returns tensors in registration order (same as their IDs).</summary>
    internal IReadOnlyList<Tensor<T>> Ordered => _ordered;

    /// <summary>
    /// Returns the ID for a tensor, registering it on first encounter.
    /// </summary>
    internal int GetOrAdd(Tensor<T> tensor)
    {
        if (_map.TryGetValue(tensor, out int id)) return id;
        id = _ordered.Count;
        _map[tensor] = id;
        _ordered.Add(tensor);
        return id;
    }

    /// <summary>
    /// Returns the ID for a tensor that must already be registered.
    /// Throws if the tensor was never encountered during the scan phase.
    /// </summary>
    internal int GetId(Tensor<T> tensor)
    {
        if (_map.TryGetValue(tensor, out int id)) return id;
        throw new InvalidOperationException(
            "Tensor not found in the ID map — was the scan phase incomplete? " +
            "Every tensor referenced by a step's Inputs or OutputBuffer must " +
            "be registered via GetOrAdd before serialization begins.");
    }

    /// <summary>
    /// Reference-equality comparer for Tensor&lt;T&gt;. Uses
    /// <see cref="RuntimeHelpers.GetHashCode(object)"/> for identity hash.
    /// </summary>
    private sealed class ReferenceEqualityComparer : IEqualityComparer<Tensor<T>>
    {
        internal static readonly ReferenceEqualityComparer Instance = new();
        public bool Equals(Tensor<T>? x, Tensor<T>? y) => ReferenceEquals(x, y);
        public int GetHashCode(Tensor<T> obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
