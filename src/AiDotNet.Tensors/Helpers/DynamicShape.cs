namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Represents a tensor shape dimension that can be either a fixed value or a symbolic variable.
/// Enables "compile once, specialize at runtime" for varying batch sizes.
/// </summary>
/// <remarks>
/// <para>
/// When a computation graph is compiled with symbolic dimensions (e.g., batch size),
/// the memory plan can be specialized at runtime without recompilation:
/// - Fixed dims: known at compile time (e.g., channels=256, height=64)
/// - Symbolic dims: resolved at runtime (e.g., batch=N)
/// </para>
/// <para>
/// Usage:
/// <code>
/// var shape = new DynamicShape(
///     DynamicDim.Symbolic("batch"),  // varies at runtime
///     DynamicDim.Fixed(256),         // fixed
///     DynamicDim.Fixed(64),          // fixed
///     DynamicDim.Fixed(64));         // fixed
///
/// // Specialize for batch=4
/// int[] concrete = shape.Resolve(new Dictionary&lt;string, int&gt; { ["batch"] = 4 });
/// // Result: [4, 256, 64, 64]
/// </code>
/// </para>
/// </remarks>
public sealed class DynamicShape
{
    private readonly DynamicDim[] _dims;

    /// <summary>Number of dimensions.</summary>
    public int Rank => _dims.Length;

    /// <summary>Gets the dimension at the given index.</summary>
    public DynamicDim this[int index] => _dims[index];

    /// <summary>Whether this shape contains any symbolic dimensions.</summary>
    public bool HasSymbolicDims
    {
        get
        {
            for (int i = 0; i < _dims.Length; i++)
                if (_dims[i].IsSymbolic) return true;
            return false;
        }
    }

    public DynamicShape(params DynamicDim[] dims)
    {
        _dims = (DynamicDim[])dims.Clone();
    }

    /// <summary>
    /// Creates a DynamicShape from a concrete int[] shape (all fixed dims).
    /// </summary>
    public static DynamicShape FromFixed(int[] shape)
    {
        var dims = new DynamicDim[shape.Length];
        for (int i = 0; i < shape.Length; i++)
            dims[i] = DynamicDim.Fixed(shape[i]);
        return new DynamicShape(dims);
    }

    /// <summary>
    /// Resolves all symbolic dimensions using the provided bindings.
    /// </summary>
    /// <param name="bindings">Map from symbolic name to concrete value.</param>
    /// <returns>Concrete shape with all dimensions resolved.</returns>
    public int[] Resolve(IReadOnlyDictionary<string, int> bindings)
    {
        var result = new int[_dims.Length];
        for (int i = 0; i < _dims.Length; i++)
        {
            if (_dims[i].IsSymbolic)
            {
                if (bindings == null || !bindings.TryGetValue(_dims[i].Name, out int value))
                    throw new InvalidOperationException(
                        $"Symbolic dimension '{_dims[i].Name}' at index {i} has no binding.");
                result[i] = value;
            }
            else
            {
                result[i] = _dims[i].Value;
            }
        }
        return result;
    }

    /// <summary>
    /// Tries to resolve, returning null if any symbolic dim is unbound.
    /// </summary>
    public int[]? TryResolve(IReadOnlyDictionary<string, int>? bindings)
    {
        var result = new int[_dims.Length];
        for (int i = 0; i < _dims.Length; i++)
        {
            if (_dims[i].IsSymbolic)
            {
                if (bindings == null || !bindings.TryGetValue(_dims[i].Name, out int value))
                    return null;
                result[i] = value;
            }
            else
            {
                result[i] = _dims[i].Value;
            }
        }
        return result;
    }

    /// <summary>
    /// Computes the total element count for a given set of bindings.
    /// </summary>
    public int ComputeSize(IReadOnlyDictionary<string, int> bindings)
    {
        var resolved = Resolve(bindings);
        int size = 1;
        foreach (int dim in resolved)
            size = checked(size * dim);
        return size;
    }

    /// <summary>
    /// Gets all unique symbolic dimension names in this shape.
    /// </summary>
    public HashSet<string> GetSymbolicNames()
    {
        var names = new HashSet<string>();
        for (int i = 0; i < _dims.Length; i++)
            if (_dims[i].IsSymbolic)
                names.Add(_dims[i].Name);
        return names;
    }

    public override string ToString()
    {
        return "[" + string.Join(", ", _dims.Select(d => d.ToString())) + "]";
    }
}

/// <summary>
/// A single dimension that is either a fixed integer or a symbolic variable.
/// </summary>
public readonly struct DynamicDim
{
    /// <summary>The fixed value (only valid when IsSymbolic is false).</summary>
    public readonly int Value;

    /// <summary>The symbolic name (only valid when IsSymbolic is true).</summary>
    public readonly string Name;

    /// <summary>Whether this dimension is symbolic (variable at runtime).</summary>
    public readonly bool IsSymbolic;

    private DynamicDim(int value, string name, bool isSymbolic)
    {
        Value = value;
        Name = name;
        IsSymbolic = isSymbolic;
    }

    /// <summary>Creates a fixed dimension with a known value.</summary>
    public static DynamicDim Fixed(int value) => new(value, string.Empty, false);

    /// <summary>Creates a symbolic dimension with a variable name.</summary>
    public static DynamicDim Symbolic(string name) => new(0, name, true);

    public override string ToString() => IsSymbolic ? Name : Value.ToString();
}
