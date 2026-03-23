using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Immutable, zero-allocation wrapper around tensor dimension metadata.
/// Provides direct indexer access (tensor.Shape[0]) without exposing mutable int[].
/// </summary>
/// <remarks>
/// <para>This is a value type (struct) — no heap allocation, lives on the stack.
/// Internal array is never exposed to consumers, preventing mutation that could
/// corrupt stride/contiguity invariants.</para>
/// </remarks>
public readonly struct TensorShape : IEquatable<TensorShape>
{
    internal readonly int[] _dims;

    /// <summary>
    /// Creates a new TensorShape wrapping the given dimensions array.
    /// The array is NOT copied — caller must not mutate it after construction.
    /// </summary>
    internal TensorShape(int[] dims) => _dims = dims;

    /// <summary>
    /// Gets the size of the specified dimension.
    /// </summary>
    public int this[int index] => _dims[index];

    /// <summary>
    /// Gets the number of dimensions (rank) of the tensor.
    /// </summary>
    public int Length => _dims.Length;

    /// <summary>
    /// Gets a read-only span over the dimensions for zero-copy iteration in hot paths.
    /// </summary>
    public ReadOnlySpan<int> Span => _dims;

    /// <summary>
    /// Returns a new array containing a copy of the dimensions.
    /// Use sparingly — allocates. Prefer Shape[i] or Shape.Span for access.
    /// </summary>
    public int[] ToArray() => (int[])_dims.Clone();

    /// <summary>
    /// Gets the total number of elements (product of all dimensions).
    /// </summary>
    public int Product
    {
        get
        {
            int result = 1;
            for (int i = 0; i < _dims.Length; i++)
                result *= _dims[i];
            return result;
        }
    }

    /// <summary>
    /// Implicit conversion to ReadOnlySpan&lt;int&gt; for interop with span-based APIs.
    /// </summary>
    public static implicit operator ReadOnlySpan<int>(TensorShape shape) => shape._dims;

    /// <summary>
    /// Checks structural equality (same dimensions in same order).
    /// </summary>
    public bool Equals(TensorShape other)
    {
        if (_dims.Length != other._dims.Length) return false;
        for (int i = 0; i < _dims.Length; i++)
        {
            if (_dims[i] != other._dims[i]) return false;
        }
        return true;
    }

    public override bool Equals(object? obj) => obj is TensorShape other && Equals(other);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        for (int i = 0; i < _dims.Length; i++)
            hash.Add(_dims[i]);
        return hash.ToHashCode();
    }

    public static bool operator ==(TensorShape left, TensorShape right) => left.Equals(right);
    public static bool operator !=(TensorShape left, TensorShape right) => !left.Equals(right);

    public override string ToString() => $"[{string.Join(", ", _dims)}]";
}
