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
/// <para>default(TensorShape) is safe — represents a scalar (zero dimensions).</para>
/// </remarks>
public readonly struct TensorShape : IEquatable<TensorShape>
{
    internal readonly int[] _dims;

    /// <summary>
    /// Safe accessor — returns the backing array or empty for default-constructed instances.
    /// </summary>
    private int[] Dims
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _dims ?? Array.Empty<int>();
    }

    /// <summary>
    /// Creates a new TensorShape wrapping the given dimensions array.
    /// The array is NOT copied — caller must not mutate it after construction.
    /// </summary>
    internal TensorShape(int[] dims) => _dims = dims;

    /// <summary>
    /// Gets the size of the specified dimension.
    /// </summary>
    public int this[int index] => Dims[index];

    /// <summary>
    /// Gets the number of dimensions (rank) of the tensor.
    /// </summary>
    public int Length => Dims.Length;

    /// <summary>
    /// Gets a read-only span over the dimensions for zero-copy iteration in hot paths.
    /// </summary>
    public ReadOnlySpan<int> Span => Dims;

    /// <summary>
    /// Returns a new array containing a copy of the dimensions.
    /// Use sparingly — allocates. Prefer Shape[i] or Shape.Span for access.
    /// </summary>
    public int[] ToArray() => (int[])Dims.Clone();

    /// <summary>
    /// Gets the total number of elements (product of all dimensions).
    /// </summary>
    public int Product
    {
        get
        {
            var d = Dims;
            int result = 1;
            for (int i = 0; i < d.Length; i++)
                result *= d[i];
            return result;
        }
    }

    /// <summary>
    /// Implicit conversion to ReadOnlySpan&lt;int&gt; for interop with span-based APIs.
    /// </summary>
    public static implicit operator ReadOnlySpan<int>(TensorShape shape) => shape.Dims;

    /// <summary>
    /// Checks structural equality (same dimensions in same order).
    /// </summary>
    public bool Equals(TensorShape other)
    {
        var a = Dims;
        var b = other.Dims;
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    public override bool Equals(object? obj) => obj is TensorShape other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            var d = Dims;
            int hash = 17;
            for (int i = 0; i < d.Length; i++)
                hash = hash * 31 + d[i];
            return hash;
        }
    }

    public static bool operator ==(TensorShape left, TensorShape right) => left.Equals(right);
    public static bool operator !=(TensorShape left, TensorShape right) => !left.Equals(right);

    public override string ToString() => $"[{string.Join(", ", Dims)}]";
}
