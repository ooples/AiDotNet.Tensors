using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Shared counter-based random contract for seeded CPU/GPU tensor operations.
/// </summary>
internal static class StatelessRandom
{
    private const double UInt32Range = 4294967296.0;
    private const float UInt24Inverse = 1.0f / 16777216.0f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Sample(uint seed, uint index)
    {
        uint state = unchecked(index * 747796405u + seed + 2891336453u);
        uint word = unchecked(((state >> (int)((state >> 28) + 4u)) ^ state) * 277803737u);
        return (word >> 22) ^ word;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Uniform01(uint seed, uint index)
        => (Sample(seed, index) >> 8) * UInt24Inverse;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float UniformRange(uint seed, uint index, float minimum, float maximum)
    {
        float range = maximum - minimum;
#if NET7_0_OR_GREATER
        return System.MathF.FusedMultiplyAdd(Uniform01(seed, index), range, minimum);
#else
        return minimum + Uniform01(seed, index) * range;
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double UniformRange(uint seed, uint index, double minimum, double maximum)
    {
        double uniform = Uniform01(seed, index);
        double range = maximum - minimum;
#if NET7_0_OR_GREATER
        return System.Math.FusedMultiplyAdd(uniform, range, minimum);
#else
        return minimum + uniform * range;
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong ProbabilityThreshold(double probability)
    {
        if (!(probability > 0.0)) return 0;
        if (probability >= 1.0) return 1UL << 32;
        return (ulong)System.Math.Ceiling(probability * UInt32Range);
    }
}
