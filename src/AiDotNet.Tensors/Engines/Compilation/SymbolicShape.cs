namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Symbolic shape representation for compiled plans, supporting one or more
/// dynamic dimensions (e.g. batch, sequence length, spatial H/W).
///
/// A <see cref="SymbolicShape"/> marks certain dimensions as dynamic (variable
/// at runtime) while keeping others fixed. The compile cache uses the symbolic
/// key — which ignores dynamic dims — so a single compiled plan serves
/// <b>any</b> runtime combination of the dynamic dimensions.
///
/// <para>Example — <c>[?, ?, 512]</c> (batch + sequence dynamic, feature dim fixed):</para>
/// <code>
///   var sym = SymbolicShape.BatchAndSeqDynamic(new[] { 1, 128, 512 });
///   var plan = cache.GetOrCompileInference(input._shape, forward, sym);
///   // Subsequent calls with shapes [2, 64, 512], [8, 256, 512], [32, 1024, 512]
///   // all hit the same cached plan — no recompile.
/// </code>
///
/// <para>This closes <c>torch.compile</c>'s #1 production gotcha (recompile-per-shape).
/// PyTorch's dynamic-shape support requires marking each symbol explicitly and still
/// frequently retraces; this type's <c>params int[]</c> constructor and the
/// <see cref="BatchAndSeqDynamic"/> / <see cref="AllDynamic"/> helpers make the
/// common transformer case one line.</para>
/// </summary>
public sealed class SymbolicShape
{
    /// <summary>The concrete shape from the trace (used for initial buffer allocation).</summary>
    public int[] ConcreteShape { get; }

    /// <summary>Indices of dimensions that are symbolic (variable at runtime).</summary>
    public int[] SymbolicDimensions { get; }

    /// <summary>
    /// Creates a symbolic shape with the given concrete values and variable dimensions.
    /// The <c>params</c> second parameter accepts dynamic-dimension indices inline
    /// (<c>new SymbolicShape([1, 128, 512], 0, 1)</c>) or as an explicit array
    /// (<c>new SymbolicShape([1, 128, 512], new[] { 0, 1 })</c>). Pass no second
    /// argument for a fully-static shape.
    /// </summary>
    /// <param name="concreteShape">The initial concrete shape values.</param>
    /// <param name="symbolicDims">Indices of dimensions to treat as symbolic (variable).
    /// Must be valid indices into concreteShape (0 to rank-1). Omit for a fully-static shape.</param>
    /// <exception cref="ArgumentOutOfRangeException">If any symbolic dimension index is out of range.</exception>
    public SymbolicShape(int[] concreteShape, params int[] symbolicDims)
    {
        ConcreteShape = (int[])concreteShape.Clone();
        // params guarantees a non-null array — an empty array means "no symbolic dims".
        for (int i = 0; i < symbolicDims.Length; i++)
        {
            if (symbolicDims[i] < 0 || symbolicDims[i] >= concreteShape.Length)
                throw new ArgumentOutOfRangeException(nameof(symbolicDims),
                    $"Symbolic dimension index {symbolicDims[i]} is out of range for shape with rank {concreteShape.Length}.");
        }
        SymbolicDimensions = symbolicDims.Length == 0
            ? Array.Empty<int>()
            : (int[])symbolicDims.Clone();
    }

    /// <summary>
    /// Fluent factory equivalent to <see cref="SymbolicShape(int[], int[])"/>.
    /// </summary>
    /// <param name="concrete">The initial concrete shape values.</param>
    /// <param name="dynamicDims">Indices of dimensions to treat as dynamic.</param>
    public static SymbolicShape From(int[] concrete, params int[] dynamicDims)
        => new SymbolicShape(concrete, dynamicDims);

    /// <summary>
    /// Creates a symbolic shape where dimension 0 (batch) is variable.
    /// This is the most common case for neural network inference/training.
    /// </summary>
    public static SymbolicShape BatchDynamic(int[] shape)
        => new SymbolicShape(shape, new[] { 0 });

    /// <summary>
    /// Creates a symbolic shape where dimensions 0 (batch) AND 1 (sequence length) are
    /// variable — the standard transformer pattern <c>[batch, seq, dim]</c>.
    /// Shape rank must be at least 2.
    /// </summary>
    /// <exception cref="ArgumentException">If <paramref name="shape"/> has fewer than 2 dimensions.</exception>
    public static SymbolicShape BatchAndSeqDynamic(int[] shape)
    {
        if (shape.Length < 2)
            throw new ArgumentException(
                $"BatchAndSeqDynamic requires rank >= 2 (got rank {shape.Length}). " +
                "For rank-1 shapes use BatchDynamic; for 4-D image batches use new SymbolicShape(shape, 0, 2, 3).",
                nameof(shape));
        return new SymbolicShape(shape, new[] { 0, 1 });
    }

    /// <summary>
    /// Creates a symbolic shape where <b>every</b> dimension is variable — useful for
    /// fully shape-polymorphic workloads where the compiled kernel graph is the same
    /// across any input shape (e.g. element-wise pipelines).
    /// </summary>
    public static SymbolicShape AllDynamic(int[] shape)
    {
        var all = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++) all[i] = i;
        return new SymbolicShape(shape, all);
    }

    /// <summary>
    /// Checks whether two shapes match, treating symbolic dimensions as wildcards.
    /// </summary>
    public bool Matches(int[] otherShape)
    {
        if (otherShape.Length != ConcreteShape.Length) return false;

        for (int i = 0; i < ConcreteShape.Length; i++)
        {
            // Skip symbolic dimensions — they match any value
            if (Array.IndexOf(SymbolicDimensions, i) >= 0)
                continue;

            if (otherShape[i] != ConcreteShape[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Computes a shape key that ignores symbolic dimensions. Two shapes with
    /// different values in dynamic positions but identical values in static
    /// positions <b>and</b> identical rank + identical symbolic-position layout
    /// produce the same key.
    /// </summary>
    /// <remarks>
    /// The key mixes in (a) the element rank, (b) a bitmask of which dimension
    /// indices are symbolic, and (c) the position-weighted value of each static
    /// dim. Without (a) and (b), <c>[3, ?]</c> and <c>[?, 3]</c> would produce
    /// the same key (both "one symbolic, one value 3") and collide in the cache
    /// — a latent bug that multi-dim symbolic shapes would surface.
    /// </remarks>
    public long ComputeKey()
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        const long fnvPrime = unchecked((long)0x100000001b3L);

        // (a) Rank — distinguishes shapes of different lengths before we even look at values.
        hash ^= ConcreteShape.Length;
        hash *= fnvPrime;

        // (b) Bitmask of symbolic positions — distinguishes [?, 3] from [3, ?].
        // Shapes with rank up to 63 fit in a single long; for higher rank we fold
        // overflow bits in via XOR (shapes with rank >= 64 are exceedingly rare).
        long symbolicMask = 0;
        for (int i = 0; i < SymbolicDimensions.Length; i++)
        {
            int bit = SymbolicDimensions[i];
            if (bit < 64) symbolicMask |= (1L << bit);
            else          symbolicMask ^= (long)(uint)bit;
        }
        hash ^= symbolicMask;
        hash *= fnvPrime;

        // (c) Position-weighted value of each static dim.
        for (int i = 0; i < ConcreteShape.Length; i++)
        {
            if (Array.IndexOf(SymbolicDimensions, i) >= 0)
                continue; // Skip symbolic dims — their values are dynamic.

            hash ^= (long)i;      // Position — [3, 5] differs from [5, 3].
            hash *= fnvPrime;
            hash ^= ConcreteShape[i];
            hash *= fnvPrime;
        }
        return hash;
    }
}
