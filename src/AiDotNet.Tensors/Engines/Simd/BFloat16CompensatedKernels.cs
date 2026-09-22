using AiDotNet.Tensors.NumericOperations;
#if NET5_0_OR_GREATER
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>Optional compensated FP32 reductions for precision-sensitive BF16 readout.</summary>
public static class BFloat16CompensatedKernels
{
    /// <summary>
    /// Computes a dot product using BF16 operands and compensated FP32 sums.
    /// This reduces cancellation and rounding error; it does not promise exact
    /// rounding or bitwise equivalence with a particular BLAS implementation.
    /// </summary>
    public static float Dot(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y)
    {
        if (x.Length != y.Length) throw new ArgumentException("Span lengths must match.");
        int i = 0;
        float sum = 0f, correction = 0f;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && x.Length >= 8)
        {
            var sums = Vector256<float>.Zero;
            var corrections = Vector256<float>.Zero;
            var sign = Vector256.Create(-0f);
            for (; i + 8 <= x.Length; i += 8)
            {
                var xb = Unsafe.ReadUnaligned<Vector128<ushort>>(ref Unsafe.As<BFloat16, byte>(ref MemoryMarshal.GetReference(x.Slice(i))));
                var yb = Unsafe.ReadUnaligned<Vector128<ushort>>(ref Unsafe.As<BFloat16, byte>(ref MemoryMarshal.GetReference(y.Slice(i))));
                var xf = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(xb), 16).AsSingle();
                var yf = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(yb), 16).AsSingle();
                var product = Avx.Multiply(xf, yf);
                var next = Avx.Add(sums, product);
                var larger = Avx.Compare(Avx.AndNot(sign, sums), Avx.AndNot(sign, product), FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling);
                var delta = Avx.BlendVariable(Avx.Add(Avx.Subtract(product, next), sums), Avx.Add(Avx.Subtract(sums, next), product), larger);
                corrections = Avx.Add(corrections, delta);
                sums = next;
            }
            for (int lane = 0; lane < 8; lane++)
            {
                if (!Finite(sums.GetElement(lane)) || !Finite(corrections.GetElement(lane))) return Ordinary(x, y);
                Add(sums.GetElement(lane), ref sum, ref correction);
                Add(corrections.GetElement(lane), ref sum, ref correction);
            }
        }
#endif
        for (; i < x.Length; i++) Add((float)x[i] * (float)y[i], ref sum, ref correction);
        return Finite(sum) && Finite(correction) ? sum + correction : Ordinary(x, y);
    }

    private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    private static void Add(float value, ref float sum, ref float correction)
    {
        float next = sum + value;
        correction += Math.Abs(sum) >= Math.Abs(value) ? (sum - next) + value : (value - next) + sum;
        sum = next;
    }

    private static float Ordinary(ReadOnlySpan<BFloat16> x, ReadOnlySpan<BFloat16> y)
    {
        float sum = 0f;
        for (int i = 0; i < x.Length; i++) sum += (float)x[i] * (float)y[i];
        return sum;
    }
}
