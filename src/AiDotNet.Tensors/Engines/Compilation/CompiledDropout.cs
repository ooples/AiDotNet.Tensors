using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Compiled dropout: pre-generates dropout masks in bulk at compile time
/// and cycles through them during training, avoiding per-step random generation.
///
/// Standard dropout: each forward pass generates a new random mask via RNG (expensive).
/// Compiled dropout: pre-allocate N masks at compile time, cycle index++ per step.
///
/// The masks are deterministic for a given seed, enabling reproducible training.
/// Cycle length is configurable (default: 1024 masks). After cycling through all masks,
/// it wraps around — this is statistically acceptable for training since the masks
/// are independent and the cycle is much longer than typical gradient correlation windows.
///
/// Memory: maskCount * tensorLength * sizeof(float) (1024 masks * 100K elements = ~400MB).
/// For large tensors, use a smaller cycle or generate on-the-fly.
/// </summary>
internal sealed class CompiledDropout
{
    private readonly float[][] _masks;
    private readonly float _scale; // 1 / (1 - dropoutRate) for inverted dropout
    private readonly int _maskLength;
    private int _currentIndex;

    /// <summary>
    /// Creates a compiled dropout with pre-generated masks.
    /// </summary>
    /// <param name="tensorLength">Number of elements per mask.</param>
    /// <param name="dropoutRate">Probability of dropping each element (0-1).</param>
    /// <param name="maskCount">Number of masks to pre-generate.</param>
    /// <param name="seed">Random seed for reproducible mask generation.</param>
    internal CompiledDropout(int tensorLength, float dropoutRate, int maskCount = 128, int seed = 42)
    {
        _maskLength = tensorLength;
        _scale = 1f / (1f - dropoutRate);
        _masks = new float[maskCount][];

        var rng = RandomHelper.CreateSeededRandom(seed);

        for (int m = 0; m < maskCount; m++)
        {
            var mask = new float[tensorLength];
            for (int i = 0; i < tensorLength; i++)
                mask[i] = rng.NextDouble() >= dropoutRate ? _scale : 0f;
            _masks[m] = mask;
        }
    }

    /// <summary>Number of pre-generated masks.</summary>
    internal int MaskCount => _masks.Length;

    /// <summary>
    /// Gets the next dropout mask and advances the cycle index.
    /// Thread-safe via Interlocked increment.
    /// </summary>
    internal float[] GetNextMask()
    {
        int idx = System.Threading.Interlocked.Increment(ref _currentIndex) % _masks.Length;
        if (idx < 0) idx += _masks.Length; // Handle overflow
        return _masks[idx];
    }

    /// <summary>
    /// Applies the next dropout mask to the input tensor in-place.
    /// output[i] = input[i] * mask[i] (where mask is 0 or scale)
    /// </summary>
    internal void ApplyInPlace(float[] data, int length)
    {
        var mask = GetNextMask();
        int len = Math.Min(length, _maskLength);
        for (int i = 0; i < len; i++)
            data[i] *= mask[i];
    }

    /// <summary>Resets the cycle index (e.g., at epoch boundary).</summary>
    internal void Reset() => _currentIndex = 0;

    /// <summary>
    /// Estimates memory usage in bytes for the pre-generated masks.
    /// </summary>
    internal long EstimatedMemoryBytes => (long)_masks.Length * _maskLength * sizeof(float);
}
