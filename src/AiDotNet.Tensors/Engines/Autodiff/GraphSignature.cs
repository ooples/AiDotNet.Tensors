using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Compact, hashable representation of a computation graph's structure.
/// Used by <see cref="TrainingGraphCache{T}"/> to detect identical graph structures
/// across training steps without comparing full entry lists.
/// </summary>
internal readonly struct GraphSignature : IEquatable<GraphSignature>
{
    public readonly long Hash;
    public readonly int OpCount;
    public readonly long ShapeFingerprint;

    public GraphSignature(long hash, int opCount, long shapeFingerprint)
    {
        Hash = hash;
        OpCount = opCount;
        ShapeFingerprint = shapeFingerprint;
    }

    public bool Equals(GraphSignature other) =>
        Hash == other.Hash && OpCount == other.OpCount && ShapeFingerprint == other.ShapeFingerprint;

    public override bool Equals(object? obj) => obj is GraphSignature other && Equals(other);

    public override int GetHashCode() => Hash.GetHashCode();
}

/// <summary>
/// Incrementally builds a <see cref="GraphSignature"/> during tape recording.
/// Uses FNV-1a hashing for fast, low-collision fingerprinting of the op sequence.
/// </summary>
internal sealed class GraphSignatureBuilder
{
    // FNV-1a 64-bit constants
    private const long FnvOffsetBasis = unchecked((long)0xcbf29ce484222325);
    private const long FnvPrime = unchecked((long)0x100000001b3);

    private long _hash = FnvOffsetBasis;
    private long _shapeHash = FnvOffsetBasis;
    private int _opCount;

    /// <summary>
    /// Feeds an operation name into the rolling hash.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void FeedOp(string opName)
    {
        _opCount++;
        for (int i = 0; i < opName.Length; i++)
        {
            _hash ^= opName[i];
            _hash *= FnvPrime;
        }
    }

    /// <summary>
    /// Feeds a tensor shape into the shape fingerprint hash.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void FeedShape(int[] shape)
    {
        for (int i = 0; i < shape.Length; i++)
        {
            _shapeHash ^= shape[i];
            _shapeHash *= FnvPrime;
        }
        // Include rank as separator
        _shapeHash ^= shape.Length;
        _shapeHash *= FnvPrime;
    }

    /// <summary>
    /// Builds the final signature from accumulated hash state.
    /// </summary>
    public GraphSignature Build() => new GraphSignature(_hash, _opCount, _shapeHash);

    /// <summary>
    /// Resets the builder for a new graph.
    /// </summary>
    public void Reset()
    {
        _hash = FnvOffsetBasis;
        _shapeHash = FnvOffsetBasis;
        _opCount = 0;
    }
}
