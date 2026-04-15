namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Kernel-input shape signature used by <see cref="AutotuneCache"/> as part of
/// the lookup key. Stored as an immutable copy of the supplied dimensions; two
/// profiles are equal iff their dimension arrays are element-wise equal.
/// </summary>
/// <remarks>
/// Use freely-chosen dimension ordering — whatever the kernel family standardises
/// on. GEMM typically uses <c>[M, N, K]</c>; convolutions often use
/// <c>[batch, in-channels, out-channels, kH, kW]</c>; SDPA uses
/// <c>[batch, seq, head-dim]</c>. The cache treats the array as an opaque
/// vector of integers.
/// </remarks>
public sealed class ShapeProfile : IEquatable<ShapeProfile>
{
    // Defensive copy held privately. `Dimensions` returns a clone so external
    // callers cannot mutate our internal state by writing into the array.
    private readonly int[] _dimensions;

    /// <summary>
    /// Snapshot of the dimension sizes. Each call returns a fresh copy — mutating
    /// the returned array does not affect this profile or its cache identity.
    /// </summary>
    public int[] Dimensions => (int[])_dimensions.Clone();

    /// <summary>Creates a shape profile. The input array is defensively copied.</summary>
    /// <param name="dimensions">Dimension sizes. Must be non-null.</param>
    public ShapeProfile(params int[] dimensions)
    {
        if (dimensions is null) throw new ArgumentNullException(nameof(dimensions));
        _dimensions = (int[])dimensions.Clone();
    }

    /// <summary>
    /// Returns a filesystem-safe string representation, e.g. <c>"256x256x256"</c>.
    /// Used by <see cref="AutotuneCache"/> to build per-shape cache filenames.
    /// </summary>
    public string ToFileStem()
        => _dimensions.Length == 0 ? "scalar" : string.Join("x", _dimensions);

    public bool Equals(ShapeProfile? other)
    {
        if (other is null) return false;
        if (other._dimensions.Length != _dimensions.Length) return false;
        for (int i = 0; i < _dimensions.Length; i++)
            if (_dimensions[i] != other._dimensions[i]) return false;
        return true;
    }

    public override bool Equals(object? obj) => obj is ShapeProfile p && Equals(p);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            for (int i = 0; i < _dimensions.Length; i++)
                hash = hash * 31 + _dimensions[i];
            return hash;
        }
    }

    public override string ToString() => $"[{string.Join(", ", _dimensions)}]";
}
