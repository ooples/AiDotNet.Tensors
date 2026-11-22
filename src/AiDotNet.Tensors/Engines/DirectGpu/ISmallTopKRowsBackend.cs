namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Top k of each row of a row-major matrix in one launch, for small k (a mixture-of-experts router's top-2 of 8).</summary>
public interface ISmallTopKRowsBackend
{
    /// <summary>The largest k a launch supports.</summary>
    int MaxSmallTopK { get; }

    /// <summary>
    /// Writes the top <paramref name="k"/> of each of <paramref name="rows"/> rows of <paramref name="input"/>
    /// ([rows, cols]) best-first into <paramref name="values"/> and their column indices (as floats) into
    /// <paramref name="indices"/>, both [rows, k], in the bitonic row sort's order: NaN first, then larger values,
    /// ties to the lower index.
    /// </summary>
    void TopKRowsSmall(IGpuBuffer input, IGpuBuffer values, IGpuBuffer indices, int rows, int cols, int k);
}

internal static class SmallTopKRows
{
    /// <summary>The largest k every backend's kernel supports (its per-thread insertion list).</summary>
    internal const int MaxK = 16;

    /// <summary>Checks the arguments; false when there is no row to launch for.</summary>
    internal static bool Validate(IGpuBuffer input, IGpuBuffer values, IGpuBuffer indices, int rows, int cols, int k)
    {
        if (k <= 0 || k > MaxK) throw new System.ArgumentOutOfRangeException(nameof(k), $"k must be in [1, {MaxK}].");
        if (k > cols) throw new System.ArgumentOutOfRangeException(nameof(k), "k cannot exceed the row length.");
        if (rows < 0) throw new System.ArgumentOutOfRangeException(nameof(rows));
        if (rows == 0) return false;
        GpuBufferCapacity.Require(input, checked(rows * cols), nameof(input));
        GpuBufferCapacity.Require(values, checked(rows * k), nameof(values));
        GpuBufferCapacity.Require(indices, checked(rows * k), nameof(indices));
        return true;
    }
}
