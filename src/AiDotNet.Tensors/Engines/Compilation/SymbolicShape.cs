namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Phase 7.2: Symbolic shape representation for compiled plans.
///
/// Allows compiled plans to handle variable batch sizes without full recompilation.
/// A SymbolicShape marks certain dimensions as dynamic (e.g., batch dimension)
/// while keeping others fixed (e.g., feature dimensions).
///
/// Example: [?, 128] means "any batch size, 128 features"
///   - First compile with batch=32 → creates plan with batch=32 buffers
///   - Call with batch=64 → only reallocates buffers, doesn't recompile ops
///
/// Usage with CompiledModelCache:
///   cache.GetOrCompileInference(input._shape, forward, symbolicDims: new[] { 0 });
///   // Dimension 0 is symbolic — plan reused for any batch size
/// </summary>
public sealed class SymbolicShape
{
    /// <summary>The concrete shape from the trace (used for initial buffer allocation).</summary>
    public int[] ConcreteShape { get; }

    /// <summary>Indices of dimensions that are symbolic (variable at runtime).</summary>
    public int[] SymbolicDimensions { get; }

    /// <summary>Creates a symbolic shape with the given concrete values and variable dimensions.</summary>
    public SymbolicShape(int[] concreteShape, int[]? symbolicDims = null)
    {
        ConcreteShape = (int[])concreteShape.Clone();
        SymbolicDimensions = symbolicDims ?? Array.Empty<int>();
    }

    /// <summary>
    /// Creates a symbolic shape where dimension 0 (batch) is variable.
    /// This is the most common case for neural network inference/training.
    /// </summary>
    public static SymbolicShape BatchDynamic(int[] shape)
    {
        return new SymbolicShape(shape, new[] { 0 });
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
    /// Computes a shape key that ignores symbolic dimensions.
    /// Two shapes with different batch sizes but same feature dims get the same key.
    /// </summary>
    public long ComputeKey()
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        for (int i = 0; i < ConcreteShape.Length; i++)
        {
            if (Array.IndexOf(SymbolicDimensions, i) >= 0)
                continue; // Skip symbolic dims in hash

            hash ^= ConcreteShape[i];
            hash *= unchecked((long)0x100000001b3L);
        }
        return hash;
    }
}
