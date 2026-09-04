using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Table-driven sigmoid with quadratic interpolation.
///
/// Instead of computing 1/(1+exp(-x)) which requires expensive exp + divide,
/// use a precomputed 514-entry lookup table covering [-16, 16] with quadratic
/// interpolation. Outside this range, sigmoid saturates to 0 or 1 at float32 precision.
///
/// Computation per element: 1 multiply (index) + 1 clamp + table load + 2 FMA
/// Total: ~5 ops vs ~14 for exp+divide path
///
/// Accuracy: max 2.1e-6 absolute error within the interpolation interval.
/// Table: 514 entries = 2056 bytes (fits in L1 cache)
/// </summary>
internal static class TableDrivenSigmoid
{
    private const int TableSize = 512;
    private const float XMin = -16.0f;
    private const float XMax = 16.0f;
    private const float Step = (XMax - XMin) / TableSize; // 0.0625
    private const float InvStep = TableSize / (XMax - XMin); // 16.0

    // Pre-computed sigmoid values at 514 points (512 + 2 for quadratic interpolation boundary)
    private static readonly float[] Table = GenerateTable();

    private static float[] GenerateTable()
    {
        var t = new float[TableSize + 2]; // +2 for quadratic interpolation at boundary
        for (int i = 0; i <= TableSize + 1; i++)
        {
            float x = XMin + i * Step;
            t[i] = 1.0f / (1.0f + MathF.Exp(-x));
        }
        return t;
    }

    /// <summary>
    /// Scalar sigmoid via table lookup + quadratic interpolation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Sigmoid(float x)
    {
        if (float.IsNaN(x)) return float.NaN;
        if (x < XMin) return 0f;
        if (x > XMax) return 1f;

        float t = (x - XMin) * InvStep;
        int idx = (int)t;
        if (idx > TableSize - 2) idx = TableSize - 2;
        float frac = t - idx;

        float y0 = Table[idx], y1 = Table[idx + 1], y2 = Table[idx + 2];
        // Quadratic interpolation: a + b*frac + c*frac^2
        float b = (-3f * y0 + 4f * y1 - y2) * 0.5f;
        float c = (y0 - 2f * y1 + y2) * 0.5f;
        return y0 + frac * (b + frac * c);
    }

    /// <summary>
    /// Vectorized sigmoid: table lookup per-element + quadratic interpolation.
    /// Uses scalar loop since AVX2 gather from float[] requires pinning.
    /// Still faster than exp+divide because no transcendental computation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SigmoidArray(float* input, float* output, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        // AVX2 path: compute index, vpgatherdd from pinned table, quadratic interpolate
        if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
        {
            var vXMin = Vector256.Create(XMin);
            var vInvStep = Vector256.Create(InvStep);
            var vZero = Vector256<float>.Zero;
            var vOne = Vector256.Create(1.0f);
            var vTableSize = Vector256.Create((float)TableSize);
            var vLastQuadraticStart = Vector256.Create((float)(TableSize - 2));
            var vHalf = Vector256.Create(0.5f);
            var vNeg3 = Vector256.Create(-3.0f);
            var vFour = Vector256.Create(4.0f);

            fixed (float* tablePtr = Table)
            {
                int simdLen = length & ~7;
                for (; i < simdLen; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);

                    // Keep the bounded interpolation coordinate separate from the capped gather
                    // index. The final quadratic starts at TableSize-2 and deliberately evaluates
                    // frac in [0, 2]. Clamping t itself to TableSize-2 flattened the entire upper
                    // interval to table[TableSize-2] (sigmoid(7.875) for the 256-entry table).
                    var t = Avx.Multiply(Avx.Subtract(x, vXMin), vInvStep);
                    t = Avx.Max(vZero, Avx.Min(t, vTableSize));
                    var indexCoordinate = Avx.Min(t, vLastQuadraticStart);
                    var idx = Avx.ConvertToVector256Int32WithTruncation(indexCoordinate);
                    var frac = Avx.Subtract(t, Avx.ConvertToVector256Single(idx));

                    // Gather y0, y1, y2 from table
                    var y0 = Avx2.GatherVector256(tablePtr, idx, 4);
                    var idx1 = Avx2.Add(idx, Vector256.Create(1));
                    var y1 = Avx2.GatherVector256(tablePtr, idx1, 4);
                    var idx2 = Avx2.Add(idx, Vector256.Create(2));
                    var y2 = Avx2.GatherVector256(tablePtr, idx2, 4);

                    // Quadratic: b = (-3*y0 + 4*y1 - y2) * 0.5
                    var b = Avx.Multiply(vHalf,
                        Avx.Add(Fma.MultiplyAdd(vNeg3, y0, Avx.Multiply(vFour, y1)),
                            Avx.Subtract(vZero, y2)));
                    // c = (y0 - 2*y1 + y2) * 0.5 (second finite difference)
                    var twoY1 = Avx.Add(y1, y1);
                    var c = Avx.Multiply(vHalf,
                        Avx.Add(Avx.Subtract(y0, twoY1), y2));

                    // result = y0 + frac * (b + frac * c)
                    var result = Fma.MultiplyAdd(frac, Fma.MultiplyAdd(frac, c, b), y0);

                    // Match the scalar saturation contract and propagate NaN. Clamping only the
                    // interpolation coordinate returned sigmoid(-8)/sigmoid(8) for infinities,
                    // while the scalar path returns the exact saturation values.
                    result = Avx.Max(vZero, Avx.Min(vOne, result));
                    var lowMask = Avx.Compare(
                        x, vXMin, FloatComparisonMode.OrderedLessThanSignaling);
                    var highMask = Avx.Compare(
                        x, Vector256.Create(XMax), FloatComparisonMode.OrderedGreaterThanSignaling);
                    var nanMask = Avx.CompareNotEqual(x, x);
                    result = Avx.BlendVariable(result, vZero, lowMask);
                    result = Avx.BlendVariable(result, vOne, highMask);
                    result = Avx.BlendVariable(result, x, nanMask);

                    Avx.Store(output + i, result);
                }
            }
        }
#endif
        // Scalar fallback
        for (; i < length; i++)
            output[i] = Sigmoid(input[i]);
    }
}
