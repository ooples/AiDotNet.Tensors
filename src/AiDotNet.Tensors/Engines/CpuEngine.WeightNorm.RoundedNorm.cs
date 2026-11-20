using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    internal static bool WeightNormRoundedNormEnabled = Environment.GetEnvironmentVariable("AIDOTNET_CPU_WEIGHTNORM_ROUNDED_NORM") == "1";

    // The row norm is an FP32 operation: square/add are separate rounded stages,
    // followed by FP32 sqrt. Keeping the norm, then dividing the gain once, also
    // preserves the identity when gain is that row's norm.
    // Comparison reference: PyTorch 5c488690, ReduceOpsKernel.cpp:204-235.
    private static unsafe float RoundedWeightNormRowNorm(ReadOnlySpan<float> values, float epsilon)
    {
        int width = System.Numerics.Vector<float>.Count;
        Span<float> lanes = stackalloc float[width];
        lanes.Clear();
        int index = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && width == 8)
        {
            var accumulator = Vector256<float>.Zero;
            fixed (float* pointer = values)
                for (; index + 8 <= values.Length; index += 8)
                {
                    var vector = Avx.LoadVector256(pointer + index);
                    accumulator = Avx.Add(accumulator, Avx.Multiply(vector, vector));
                }
            fixed (float* destination = lanes) Avx.Store(destination, accumulator);
        }
        else
#endif
        {
            for (; index + width <= values.Length; index += width)
                for (int lane = 0; lane < width; lane++)
                {
                    float value = values[index + lane];
                    float squared = value * value;
                    lanes[lane] += squared;
                }
        }
        float sum = lanes[0];
        for (int lane = 1; lane < width; lane++) sum += lanes[lane];
        for (; index < values.Length; index++)
        {
            float value = values[index];
            float squared = value * value;
            sum += squared;
        }
        sum += epsilon;
#if NET5_0_OR_GREATER
        return MathF.Sqrt(sum);
#else
        return (float)Math.Sqrt(sum);
#endif
    }
}
