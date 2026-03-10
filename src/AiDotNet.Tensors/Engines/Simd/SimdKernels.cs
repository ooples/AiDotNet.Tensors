using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif
namespace AiDotNet.Tensors.Engines.Simd
{
    /// <summary>
    /// SIMD-optimized kernels for common operations.
    /// Provides hardware-accelerated implementations using AVX/SSE and ARM NEON.
    /// Falls back to scalar operations when intrinsics are unavailable.
    /// </summary>
    public static class SimdKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(result, i, Avx.Add(ReadVector256(a, i), ReadVector256(b, i)));
                    WriteVector256(result, i + 8, Avx.Add(ReadVector256(a, i + 8), ReadVector256(b, i + 8)));
                    WriteVector256(result, i + 16, Avx.Add(ReadVector256(a, i + 16), ReadVector256(b, i + 16)));
                    WriteVector256(result, i + 24, Avx.Add(ReadVector256(a, i + 24), ReadVector256(b, i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(result, i, Avx.Add(ReadVector256(a, i), ReadVector256(b, i)));
                }
            }
            else if (Sse.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(result, i, Sse.Add(ReadVector128(a, i), ReadVector128(b, i)));
                }
            }
            else if (AdvSimd.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(result, i, AdvSimd.Add(ReadVector128(a, i), ReadVector128(b, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// Pointer-based VectorAdd — zero bounds-checking overhead for hot paths.
        /// Caller must ensure pointers are valid and length is correct.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorAddUnsafe(float* a, float* b, float* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                // 4x unrolled AVX2 with software prefetch for large arrays
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(result + i, Avx.Add(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                    Avx.Store(result + i + 8, Avx.Add(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8)));
                    Avx.Store(result + i + 16, Avx.Add(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16)));
                    Avx.Store(result + i + 24, Avx.Add(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(result + i, Avx.Add(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// Pointer-based VectorMultiply — zero bounds-checking overhead for hot paths.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorMultiplyUnsafe(float* a, float* b, float* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(result + i, Avx.Multiply(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                    Avx.Store(result + i + 8, Avx.Multiply(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8)));
                    Avx.Store(result + i + 16, Avx.Multiply(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16)));
                    Avx.Store(result + i + 24, Avx.Multiply(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(result + i, Avx.Multiply(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        /// <summary>
        /// Pointer-based VectorSubtract — zero bounds-checking overhead for hot paths.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorSubtractUnsafe(float* a, float* b, float* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(result + i, Avx.Subtract(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                    Avx.Store(result + i + 8, Avx.Subtract(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8)));
                    Avx.Store(result + i + 16, Avx.Subtract(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16)));
                    Avx.Store(result + i + 24, Avx.Subtract(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(result + i, Avx.Subtract(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                result[i] = a[i] - b[i];
            }
        }

        /// <summary>
        /// Pointer-based ReLU — zero bounds-checking overhead for hot paths.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ReLUUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vzero = Vector256<float>.Zero;
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(output + i, Avx.Max(Avx.LoadVector256(input + i), vzero));
                    Avx.Store(output + i + 8, Avx.Max(Avx.LoadVector256(input + i + 8), vzero));
                    Avx.Store(output + i + 16, Avx.Max(Avx.LoadVector256(input + i + 16), vzero));
                    Avx.Store(output + i + 24, Avx.Max(Avx.LoadVector256(input + i + 24), vzero));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vzero = Vector256<float>.Zero;
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(output + i, Avx.Max(Avx.LoadVector256(input + i), vzero));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] = input[i] > 0 ? input[i] : 0;
            }
        }

        /// <summary>
        /// Pointer-based Sigmoid — zero bounds-checking overhead for hot paths.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void SigmoidUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Fma.IsSupported && Avx.IsSupported && length >= 32)
            {
                // Pre-load constants outside the loop to avoid repeated creation
                var vmin = Vector256.Create(-5.0f);
                var vmax = Vector256.Create(5.0f);
                var vc5 = Vector256.Create(1.5854344e-4f);
                var vc3 = Vector256.Create(-8.9219211e-3f);
                var vc1 = Vector256.Create(2.1562920e-1f);
                var vhalf = Vector256.Create(0.5f);

                // 5th-order odd polynomial: sigmoid(x) ≈ 0.5 + x * (c1 + x² * (c3 + x² * c5))
                // 4x unrolled: 12 FMA + 4 mul + 8 min/max per 32 floats
                // No prefetch (branch in hot loop costs more than it saves)
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    // Load 4 vectors and clamp (ILP-friendly — all loads independent)
                    var x0 = Avx.Min(Avx.Max(Avx.LoadVector256(input + i), vmin), vmax);
                    var x1 = Avx.Min(Avx.Max(Avx.LoadVector256(input + i + 8), vmin), vmax);
                    var x2 = Avx.Min(Avx.Max(Avx.LoadVector256(input + i + 16), vmin), vmax);
                    var x3 = Avx.Min(Avx.Max(Avx.LoadVector256(input + i + 24), vmin), vmax);

                    // x² for all 4 — independent, can execute in parallel on multiple ports
                    var sq0 = Avx.Multiply(x0, x0);
                    var sq1 = Avx.Multiply(x1, x1);
                    var sq2 = Avx.Multiply(x2, x2);
                    var sq3 = Avx.Multiply(x3, x3);

                    // FMA chain step 1: c5*x² + c3
                    var p0 = Fma.MultiplyAdd(sq0, vc5, vc3);
                    var p1 = Fma.MultiplyAdd(sq1, vc5, vc3);
                    var p2 = Fma.MultiplyAdd(sq2, vc5, vc3);
                    var p3 = Fma.MultiplyAdd(sq3, vc5, vc3);

                    // FMA chain step 2: c1 + x²*(c3+c5*x²)
                    p0 = Fma.MultiplyAdd(sq0, p0, vc1);
                    p1 = Fma.MultiplyAdd(sq1, p1, vc1);
                    p2 = Fma.MultiplyAdd(sq2, p2, vc1);
                    p3 = Fma.MultiplyAdd(sq3, p3, vc1);

                    // FMA chain step 3: 0.5 + x*poly
                    Avx.Store(output + i, Fma.MultiplyAdd(x0, p0, vhalf));
                    Avx.Store(output + i + 8, Fma.MultiplyAdd(x1, p1, vhalf));
                    Avx.Store(output + i + 16, Fma.MultiplyAdd(x2, p2, vhalf));
                    Avx.Store(output + i + 24, Fma.MultiplyAdd(x3, p3, vhalf));
                }
            }
            if (Fma.IsSupported && Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(output + i, FastSigmoid256(Avx.LoadVector256(input + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] = 1.0f / (1.0f + MathF.Exp(-input[i]));
            }
        }

        /// <summary>
        /// Pointer-based VectorDivide — zero bounds-checking overhead for hot paths.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorDivideUnsafe(float* a, float* b, float* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(result + i, Avx.Divide(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                    Avx.Store(result + i + 8, Avx.Divide(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8)));
                    Avx.Store(result + i + 16, Avx.Divide(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16)));
                    Avx.Store(result + i + 24, Avx.Divide(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(result + i, Avx.Divide(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                result[i] = a[i] / b[i];
            }
        }

        /// <summary>
        /// Pointer-based Exp — Cephes FastExp256 with zero bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ExpUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(output + i, FastExp256(Avx.LoadVector256(input + i)));
                    Avx.Store(output + i + 8, FastExp256(Avx.LoadVector256(input + i + 8)));
                    Avx.Store(output + i + 16, FastExp256(Avx.LoadVector256(input + i + 16)));
                    Avx.Store(output + i + 24, FastExp256(Avx.LoadVector256(input + i + 24)));
                }
            }
            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(output + i, FastExp256(Avx.LoadVector256(input + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] = MathF.Exp(input[i]);
            }
        }

        /// <summary>
        /// Pointer-based Log — FastLog256 with zero bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void LogUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(output + i, FastLog256(Avx.LoadVector256(input + i)));
                    Avx.Store(output + i + 8, FastLog256(Avx.LoadVector256(input + i + 8)));
                    Avx.Store(output + i + 16, FastLog256(Avx.LoadVector256(input + i + 16)));
                    Avx.Store(output + i + 24, FastLog256(Avx.LoadVector256(input + i + 24)));
                }
            }
            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(output + i, FastLog256(Avx.LoadVector256(input + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] = MathF.Log(input[i]);
            }
        }

        /// <summary>
        /// Pointer-based Sqrt — 4x unrolled AVX with zero bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void SqrtUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(output + i, Avx.Sqrt(Avx.LoadVector256(input + i)));
                    Avx.Store(output + i + 8, Avx.Sqrt(Avx.LoadVector256(input + i + 8)));
                    Avx.Store(output + i + 16, Avx.Sqrt(Avx.LoadVector256(input + i + 16)));
                    Avx.Store(output + i + 24, Avx.Sqrt(Avx.LoadVector256(input + i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(output + i, Avx.Sqrt(Avx.LoadVector256(input + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] = MathF.Sqrt(input[i]);
            }
        }

        /// <summary>
        /// Pointer-based Abs — 4x unrolled AVX AND with sign mask, zero bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void AbsUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var signMask = Vector256.Create(0x7FFFFFFF).AsSingle();
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(output + i, Avx.And(Avx.LoadVector256(input + i), signMask));
                    Avx.Store(output + i + 8, Avx.And(Avx.LoadVector256(input + i + 8), signMask));
                    Avx.Store(output + i + 16, Avx.And(Avx.LoadVector256(input + i + 16), signMask));
                    Avx.Store(output + i + 24, Avx.And(Avx.LoadVector256(input + i + 24), signMask));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var signMask = Vector256.Create(0x7FFFFFFF).AsSingle();
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(output + i, Avx.And(Avx.LoadVector256(input + i), signMask));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] = Math.Abs(input[i]);
            }
        }

        /// <summary>
        /// Pointer-based Tanh — 2*sigmoid(2x)-1 with zero bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void TanhUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Fma.IsSupported && Avx.IsSupported && length >= 32)
            {
                var vtwo = Vector256.Create(2.0f);
                var vone = Vector256.Create(1.0f);
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    // tanh(x) = 2*sigmoid(2x) - 1
                    Avx.Store(output + i, Avx.Subtract(Avx.Multiply(vtwo, FastSigmoid256(Avx.Multiply(vtwo, Avx.LoadVector256(input + i)))), vone));
                    Avx.Store(output + i + 8, Avx.Subtract(Avx.Multiply(vtwo, FastSigmoid256(Avx.Multiply(vtwo, Avx.LoadVector256(input + i + 8)))), vone));
                    Avx.Store(output + i + 16, Avx.Subtract(Avx.Multiply(vtwo, FastSigmoid256(Avx.Multiply(vtwo, Avx.LoadVector256(input + i + 16)))), vone));
                    Avx.Store(output + i + 24, Avx.Subtract(Avx.Multiply(vtwo, FastSigmoid256(Avx.Multiply(vtwo, Avx.LoadVector256(input + i + 24)))), vone));
                }
            }
            if (Fma.IsSupported && Avx.IsSupported && length - i >= 8)
            {
                var vtwo = Vector256.Create(2.0f);
                var vone = Vector256.Create(1.0f);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    Avx.Store(output + i, Avx.Subtract(Avx.Multiply(vtwo, FastSigmoid256(Avx.Multiply(vtwo, Avx.LoadVector256(input + i)))), vone));
                }
            }
#endif
            for (; i < length; i++)
            {
                float ex = MathF.Exp(2f * input[i]);
                output[i] = (ex - 1f) / (ex + 1f);
            }
        }

        /// <summary>
        /// Pointer-based LeakyReLU — max(alpha*x, x) with zero bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void LeakyReLUUnsafe(float* input, float* output, int length, float alpha = 0.01f)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var valpha = Vector256.Create(alpha);
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    var v0 = Avx.LoadVector256(input + i);
                    var v1 = Avx.LoadVector256(input + i + 8);
                    var v2 = Avx.LoadVector256(input + i + 16);
                    var v3 = Avx.LoadVector256(input + i + 24);
                    Avx.Store(output + i, Avx.Max(v0, Avx.Multiply(v0, valpha)));
                    Avx.Store(output + i + 8, Avx.Max(v1, Avx.Multiply(v1, valpha)));
                    Avx.Store(output + i + 16, Avx.Max(v2, Avx.Multiply(v2, valpha)));
                    Avx.Store(output + i + 24, Avx.Max(v3, Avx.Multiply(v3, valpha)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var valpha = Vector256.Create(alpha);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    var v = Avx.LoadVector256(input + i);
                    Avx.Store(output + i, Avx.Max(v, Avx.Multiply(v, valpha)));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] = input[i] >= 0 ? input[i] : alpha * input[i];
            }
        }

        /// <summary>
        /// Pointer-based GELU — fused single-pass with 4x unrolling, zero bounds-checking.
        /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void GELUUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && Fma.IsSupported && length >= 32)
            {
                var vSqrt2OverPi = Vector256.Create(0.7978845608028654f);
                var vCoeff = Vector256.Create(0.044715f);
                var vHalf = Vector256.Create(0.5f);
                var vOne = Vector256.Create(1.0f);
                var vNegOne = Vector256.Create(-1.0f);
                var v27 = Vector256.Create(27.0f);
                var v9 = Vector256.Create(9.0f);

                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    for (int k = 0; k < 32; k += 8)
                    {
                        var x = Avx.LoadVector256(input + i + k);
                        var x_squared = Avx.Multiply(x, x);
                        var x_cubed = Avx.Multiply(x_squared, x);
                        var inner = Fma.MultiplyAdd(vCoeff, x_cubed, x);
                        var tanh_arg = Avx.Multiply(vSqrt2OverPi, inner);
                        var tanh_arg_sq = Avx.Multiply(tanh_arg, tanh_arg);
                        var numerator = Avx.Add(v27, tanh_arg_sq);
                        var denominator = Fma.MultiplyAdd(v9, tanh_arg_sq, v27);
                        var tanh_approx = Avx.Divide(Avx.Multiply(tanh_arg, numerator), denominator);
                        tanh_approx = Avx.Max(vNegOne, Avx.Min(vOne, tanh_approx));
                        Avx.Store(output + i + k, Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(vOne, tanh_approx))));
                    }
                }
            }
            if (Avx.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                var vSqrt2OverPi = Vector256.Create(0.7978845608028654f);
                var vCoeff = Vector256.Create(0.044715f);
                var vHalf = Vector256.Create(0.5f);
                var vOne = Vector256.Create(1.0f);
                var vNegOne = Vector256.Create(-1.0f);
                var v27 = Vector256.Create(27.0f);
                var v9 = Vector256.Create(9.0f);

                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var x_squared = Avx.Multiply(x, x);
                    var x_cubed = Avx.Multiply(x_squared, x);
                    var inner = Fma.MultiplyAdd(vCoeff, x_cubed, x);
                    var tanh_arg = Avx.Multiply(vSqrt2OverPi, inner);
                    var tanh_arg_sq = Avx.Multiply(tanh_arg, tanh_arg);
                    var numerator = Avx.Add(v27, tanh_arg_sq);
                    var denominator = Fma.MultiplyAdd(v9, tanh_arg_sq, v27);
                    var tanh_approx = Avx.Divide(Avx.Multiply(tanh_arg, numerator), denominator);
                    tanh_approx = Avx.Max(vNegOne, Avx.Min(vOne, tanh_approx));
                    Avx.Store(output + i, Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(vOne, tanh_approx))));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                float x3 = x * x * x;
                float inner = x + 0.044715f * x3;
                float tanh_arg = 0.7978845608028654f * inner;
                float tanh_val = MathF.Tanh(tanh_arg);
                output[i] = 0.5f * x * (1f + tanh_val);
            }
        }

        /// <summary>
        /// Pointer-based Mish — fused single-pass, no temp buffer allocation.
        /// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MishUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                var vtwo = Vector256.Create(2.0f);
                var vone = Vector256.Create(1.0f);
                var vthreshold = Vector256.Create(20.0f);

                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    for (int k = 0; k < 32; k += 8)
                    {
                        var x = Avx.LoadVector256(input + i + k);
                        // softplus(x) = ln(1 + exp(x)), or x if x > 20
                        var expx = FastExp256(x);
                        var softplus = FastLog256(Avx.Add(vone, expx));
                        // For large x, softplus ≈ x (avoid log overflow)
                        var mask = Avx.Compare(x, vthreshold, FloatComparisonMode.OrderedGreaterThanSignaling);
                        softplus = Avx.BlendVariable(softplus, x, mask);
                        // tanh(softplus) via 2*sigmoid(2*softplus)-1
                        var tanh_sp = Avx.Subtract(Avx.Multiply(vtwo, FastSigmoid256(Avx.Multiply(vtwo, softplus))), vone);
                        // mish = x * tanh(softplus(x))
                        Avx.Store(output + i + k, Avx.Multiply(x, tanh_sp));
                    }
                }
            }
            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                var vtwo = Vector256.Create(2.0f);
                var vone = Vector256.Create(1.0f);
                var vthreshold = Vector256.Create(20.0f);

                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var expx = FastExp256(x);
                    var softplus = FastLog256(Avx.Add(vone, expx));
                    var mask = Avx.Compare(x, vthreshold, FloatComparisonMode.OrderedGreaterThanSignaling);
                    softplus = Avx.BlendVariable(softplus, x, mask);
                    var tanh_sp = Avx.Subtract(Avx.Multiply(vtwo, FastSigmoid256(Avx.Multiply(vtwo, softplus))), vone);
                    Avx.Store(output + i, Avx.Multiply(x, tanh_sp));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                float sp = x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
                float th = MathF.Tanh(sp);
                output[i] = x * th;
            }
        }

        /// <summary>
        /// Unsafe pointer-based sum with 4-way accumulation. Eliminates Span bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float SumUnsafe(float* data, int length)
        {
            int i = 0;
            float sum = 0f;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vsum0 = Vector256<float>.Zero;
                var vsum1 = Vector256<float>.Zero;
                var vsum2 = Vector256<float>.Zero;
                var vsum3 = Vector256<float>.Zero;
                int simdLength = length & ~31;
                if (Sse.IsSupported && length >= 131072)
                {
                    const int prefetchDistance = 256;
                    for (; i < simdLength; i += 32)
                    {
                        if (i + prefetchDistance < length)
                            Sse.Prefetch0(data + i + prefetchDistance);
                        vsum0 = Avx.Add(vsum0, Avx.LoadVector256(data + i));
                        vsum1 = Avx.Add(vsum1, Avx.LoadVector256(data + i + 8));
                        vsum2 = Avx.Add(vsum2, Avx.LoadVector256(data + i + 16));
                        vsum3 = Avx.Add(vsum3, Avx.LoadVector256(data + i + 24));
                    }
                }
                else
                {
                    for (; i < simdLength; i += 32)
                    {
                        vsum0 = Avx.Add(vsum0, Avx.LoadVector256(data + i));
                        vsum1 = Avx.Add(vsum1, Avx.LoadVector256(data + i + 8));
                        vsum2 = Avx.Add(vsum2, Avx.LoadVector256(data + i + 16));
                        vsum3 = Avx.Add(vsum3, Avx.LoadVector256(data + i + 24));
                    }
                }
                vsum0 = Avx.Add(Avx.Add(vsum0, vsum1), Avx.Add(vsum2, vsum3));
                sum += HorizontalSum(vsum0);
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vsum = Vector256<float>.Zero;
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    vsum = Avx.Add(vsum, Avx.LoadVector256(data + i));
                }
                sum += HorizontalSum(vsum);
            }
#endif
            for (; i < length; i++)
            {
                sum += data[i];
            }
            return sum;
        }

        /// <summary>
        /// Unsafe pointer-based Max with 4-way accumulation. Eliminates Span bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float MaxUnsafe(float* data, int length)
        {
            int i = 0;
            float max = float.NegativeInfinity;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vmax0 = Vector256.Create(float.NegativeInfinity);
                var vmax1 = vmax0; var vmax2 = vmax0; var vmax3 = vmax0;
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    vmax0 = Avx.Max(vmax0, Avx.LoadVector256(data + i));
                    vmax1 = Avx.Max(vmax1, Avx.LoadVector256(data + i + 8));
                    vmax2 = Avx.Max(vmax2, Avx.LoadVector256(data + i + 16));
                    vmax3 = Avx.Max(vmax3, Avx.LoadVector256(data + i + 24));
                }
                vmax0 = Avx.Max(Avx.Max(vmax0, vmax1), Avx.Max(vmax2, vmax3));
                max = HorizontalMax(vmax0);
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vmax = Vector256.Create(max);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    vmax = Avx.Max(vmax, Avx.LoadVector256(data + i));
                }
                max = HorizontalMax(vmax);
            }
#endif
            for (; i < length; i++)
            {
                if (data[i] > max) max = data[i];
            }
            return max;
        }

        /// <summary>
        /// Unsafe pointer-based Min with 4-way accumulation. Eliminates Span bounds-checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float MinUnsafe(float* data, int length)
        {
            int i = 0;
            float min = float.PositiveInfinity;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vmin0 = Vector256.Create(float.PositiveInfinity);
                var vmin1 = vmin0; var vmin2 = vmin0; var vmin3 = vmin0;
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    vmin0 = Avx.Min(vmin0, Avx.LoadVector256(data + i));
                    vmin1 = Avx.Min(vmin1, Avx.LoadVector256(data + i + 8));
                    vmin2 = Avx.Min(vmin2, Avx.LoadVector256(data + i + 16));
                    vmin3 = Avx.Min(vmin3, Avx.LoadVector256(data + i + 24));
                }
                vmin0 = Avx.Min(Avx.Min(vmin0, vmin1), Avx.Min(vmin2, vmin3));
                min = HorizontalMin(vmin0);
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vmin = Vector256.Create(min);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    vmin = Avx.Min(vmin, Avx.LoadVector256(data + i));
                }
                min = HorizontalMin(vmin);
            }
#endif
            for (; i < length; i++)
            {
                if (data[i] < min) min = data[i];
            }
            return min;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorSubtract(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(result, i, Avx.Subtract(ReadVector256(a, i), ReadVector256(b, i)));
                    WriteVector256(result, i + 8, Avx.Subtract(ReadVector256(a, i + 8), ReadVector256(b, i + 8)));
                    WriteVector256(result, i + 16, Avx.Subtract(ReadVector256(a, i + 16), ReadVector256(b, i + 16)));
                    WriteVector256(result, i + 24, Avx.Subtract(ReadVector256(a, i + 24), ReadVector256(b, i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(result, i, Avx.Subtract(ReadVector256(a, i), ReadVector256(b, i)));
                }
            }
            else if (Sse.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(result, i, Sse.Subtract(ReadVector128(a, i), ReadVector128(b, i)));
                }
            }
            else if (AdvSimd.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(result, i, AdvSimd.Subtract(ReadVector128(a, i), ReadVector128(b, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] - b[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorMultiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(result, i, Avx.Multiply(ReadVector256(a, i), ReadVector256(b, i)));
                    WriteVector256(result, i + 8, Avx.Multiply(ReadVector256(a, i + 8), ReadVector256(b, i + 8)));
                    WriteVector256(result, i + 16, Avx.Multiply(ReadVector256(a, i + 16), ReadVector256(b, i + 16)));
                    WriteVector256(result, i + 24, Avx.Multiply(ReadVector256(a, i + 24), ReadVector256(b, i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(result, i, Avx.Multiply(ReadVector256(a, i), ReadVector256(b, i)));
                }
            }
            else if (Sse.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(result, i, Sse.Multiply(ReadVector128(a, i), ReadVector128(b, i)));
                }
            }
            else if (AdvSimd.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(result, i, AdvSimd.Multiply(ReadVector128(a, i), ReadVector128(b, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorDivide(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(result, i, Avx.Divide(ReadVector256(a, i), ReadVector256(b, i)));
                    WriteVector256(result, i + 8, Avx.Divide(ReadVector256(a, i + 8), ReadVector256(b, i + 8)));
                    WriteVector256(result, i + 16, Avx.Divide(ReadVector256(a, i + 16), ReadVector256(b, i + 16)));
                    WriteVector256(result, i + 24, Avx.Divide(ReadVector256(a, i + 24), ReadVector256(b, i + 24)));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(result, i, Avx.Divide(ReadVector256(a, i), ReadVector256(b, i)));
                }
            }
            else if (Sse.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(result, i, Sse.Divide(ReadVector128(a, i), ReadVector128(b, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] / b[i];
            }
        }

        /// <summary>Adds a scalar to each element: result[i] = a[i] + scalar.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
        {
            if (a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(result, i, Avx.Add(ReadVector256(a, i), vs));
                    WriteVector256(result, i + 8, Avx.Add(ReadVector256(a, i + 8), vs));
                    WriteVector256(result, i + 16, Avx.Add(ReadVector256(a, i + 16), vs));
                    WriteVector256(result, i + 24, Avx.Add(ReadVector256(a, i + 24), vs));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(result, i, Avx.Add(ReadVector256(a, i), vs));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] + scalar;
            }
        }

        /// <summary>Multiplies each element by a scalar: result[i] = a[i] * scalar.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
        {
            if (a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(result, i, Avx.Multiply(ReadVector256(a, i), vs));
                    WriteVector256(result, i + 8, Avx.Multiply(ReadVector256(a, i + 8), vs));
                    WriteVector256(result, i + 16, Avx.Multiply(ReadVector256(a, i + 16), vs));
                    WriteVector256(result, i + 24, Avx.Multiply(ReadVector256(a, i + 24), vs));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(result, i, Avx.Multiply(ReadVector256(a, i), vs));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] * scalar;
            }
        }

        /// <summary>Divides each element by a scalar: result[i] = a[i] / scalar.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void DivideScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
        {
            // Multiply by reciprocal for better performance
            MultiplyScalar(a, 1.0f / scalar, result);
        }

        /// <summary>Subtracts a scalar from each element: result[i] = a[i] - scalar.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SubtractScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
        {
            AddScalar(a, -scalar, result);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("Input spans must have the same length.");
            }

            int length = a.Length;
            int i = 0;
            float sum = 0f;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && Fma.IsSupported && length >= 32)
            {
                // 4-way parallel accumulation to hide FMA latency (5 cycles on Zen2)
                var vsum0 = Vector256<float>.Zero;
                var vsum1 = Vector256<float>.Zero;
                var vsum2 = Vector256<float>.Zero;
                var vsum3 = Vector256<float>.Zero;
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    vsum0 = Fma.MultiplyAdd(ReadVector256(a, i), ReadVector256(b, i), vsum0);
                    vsum1 = Fma.MultiplyAdd(ReadVector256(a, i + 8), ReadVector256(b, i + 8), vsum1);
                    vsum2 = Fma.MultiplyAdd(ReadVector256(a, i + 16), ReadVector256(b, i + 16), vsum2);
                    vsum3 = Fma.MultiplyAdd(ReadVector256(a, i + 24), ReadVector256(b, i + 24), vsum3);
                }
                vsum0 = Avx.Add(Avx.Add(vsum0, vsum1), Avx.Add(vsum2, vsum3));
                sum += HorizontalSum(vsum0);
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vsum = Vector256<float>.Zero;
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    var va = ReadVector256(a, i);
                    var vb = ReadVector256(b, i);
                    vsum = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, vsum) : Avx.Add(vsum, Avx.Multiply(va, vb));
                }
                sum += HorizontalSum(vsum);
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    vsum = Sse.Add(vsum, Sse.Multiply(va, vb));
                }

                sum += HorizontalSum(vsum);
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    vsum = AdvSimd.Add(vsum, AdvSimd.Multiply(va, vb));
                }

                sum += HorizontalSum(vsum);
            }
#endif

            for (; i < length; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        /// <summary>
        /// Computes destination[i] = a[i] + b[i] * scalar for double-precision values using SIMD.
        /// Uses FMA (Fused Multiply-Add) when available for better performance and precision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ScalarMultiplyAdd(ReadOnlySpan<double> a, ReadOnlySpan<double> b, double scalar, Span<double> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                var vscalar = Vector256.Create(scalar);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector256Double(a, i);
                    var vb = ReadVector256Double(b, i);
                    var vr = Fma.IsSupported ? Fma.MultiplyAdd(vb, vscalar, va) : Avx.Add(va, Avx.Multiply(vb, vscalar));
                    WriteVector256Double(result, i, vr);
                }
            }
            else if (Sse2.IsSupported && length >= 2)
            {
                var vscalar = Vector128.Create(scalar);
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var va = ReadVector128Double(a, i);
                    var vb = ReadVector128Double(b, i);
                    WriteVector128Double(result, i, Sse2.Add(va, Sse2.Multiply(vb, vscalar)));
                }
            }
            else if (AdvSimd.Arm64.IsSupported && length >= 2)
            {
                var vscalar = Vector128.Create(scalar);
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var va = ReadVector128Double(a, i);
                    var vb = ReadVector128Double(b, i);
                    WriteVector128Double(result, i, AdvSimd.Arm64.Add(va, AdvSimd.Arm64.Multiply(vb, vscalar)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] + scalar * b[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ScalarMultiplyAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float scalar, Span<float> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                var vscalar = Vector256.Create(scalar);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var va = ReadVector256(a, i);
                    var vb = ReadVector256(b, i);
                    var vr = Fma.IsSupported ? Fma.MultiplyAdd(vb, vscalar, va) : Avx.Add(va, Avx.Multiply(vb, vscalar));
                    WriteVector256(result, i, vr);
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vscalar = Vector128.Create(scalar);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    WriteVector128(result, i, Sse.Add(va, Sse.Multiply(vb, vscalar)));
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vscalar = Vector128.Create(scalar);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    WriteVector128(result, i, AdvSimd.Add(va, AdvSimd.Multiply(vb, vscalar)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] + scalar * b[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReLU(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vzero = Vector256<float>.Zero;
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(output, i, Avx.Max(ReadVector256(input, i), vzero));
                    WriteVector256(output, i + 8, Avx.Max(ReadVector256(input, i + 8), vzero));
                    WriteVector256(output, i + 16, Avx.Max(ReadVector256(input, i + 16), vzero));
                    WriteVector256(output, i + 24, Avx.Max(ReadVector256(input, i + 24), vzero));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vzero = Vector256<float>.Zero;
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, Avx.Max(ReadVector256(input, i), vzero));
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vzero = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(output, i, Sse.Max(ReadVector128(input, i), vzero));
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vzero = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(output, i, AdvSimd.Max(ReadVector128(input, i), vzero));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = input[i] > 0f ? input[i] : 0f;
            }
        }

        /// <summary>
        /// Computes element-wise exp(x) using a fast Cephes-style polynomial approximation with AVX2/FMA.
        /// Processes 32 floats per iteration (4x unrolled) for maximum throughput.
        /// Relative error ~0.01% across the valid range [-87.3, 88.7].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exp(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // Use Cephes-style fast exp polynomial with explicit AVX2/FMA intrinsics.
            // This is ~8x faster than scalar MathF.Exp loop for large arrays.
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(output, i, FastExp256(ReadVector256(input, i)));
                    WriteVector256(output, i + 8, FastExp256(ReadVector256(input, i + 8)));
                    WriteVector256(output, i + 16, FastExp256(ReadVector256(input, i + 16)));
                    WriteVector256(output, i + 24, FastExp256(ReadVector256(input, i + 24)));
                }
            }

            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, FastExp256(ReadVector256(input, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Exp(input[i]);
#else
                output[i] = (float)Math.Exp(input[i]);
#endif
            }
        }


        /// <summary>
        /// Computes element-wise exp(x) for double precision using scalar Math.Exp fallback.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exp(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

            // 8x unrolled: helps SVML auto-vectorization on NET8+
            if (length >= 8)
            {
                int unrolled = length & ~7;
                for (; i < unrolled; i += 8)
                {
                    output[i] = Math.Exp(input[i]);
                    output[i + 1] = Math.Exp(input[i + 1]);
                    output[i + 2] = Math.Exp(input[i + 2]);
                    output[i + 3] = Math.Exp(input[i + 3]);
                    output[i + 4] = Math.Exp(input[i + 4]);
                    output[i + 5] = Math.Exp(input[i + 5]);
                    output[i + 6] = Math.Exp(input[i + 6]);
                    output[i + 7] = Math.Exp(input[i + 7]);
                }
            }

            for (; i < length; i++)
            {
                output[i] = Math.Exp(input[i]);
            }
        }

        /// <summary>
        /// Computes element-wise sigmoid: 1/(1+exp(-x)) using fast vectorized exp.
        /// Processes 32 floats per iteration (4x unrolled).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sigmoid(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(output, i, FastSigmoid256(ReadVector256(input, i)));
                    WriteVector256(output, i + 8, FastSigmoid256(ReadVector256(input, i + 8)));
                    WriteVector256(output, i + 16, FastSigmoid256(ReadVector256(input, i + 16)));
                    WriteVector256(output, i + 24, FastSigmoid256(ReadVector256(input, i + 24)));
                }
            }

            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, FastSigmoid256(ReadVector256(input, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = 1.0f / (1.0f + MathF.Exp(-input[i]));
#else
                output[i] = 1.0f / (1.0f + (float)Math.Exp(-input[i]));
#endif
            }
        }

        /// <summary>
        /// Computes element-wise sigmoid for double precision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void Sigmoid(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            fixed (double* pIn = input)
            fixed (double* pOut = output)
            {
                // Vectorized sigmoid using FastExpDouble256: 1/(1+exp(-x))
                if (Avx2.IsSupported && Fma.IsSupported && length >= 16)
                {
                    var one = Vector256.Create(1.0);
                    var negOne = Vector256.Create(-1.0);
                    int simdLen = length & ~15;
                    for (; i < simdLen; i += 16)
                    {
                        // 4x unrolled (4 doubles per vector × 4 = 16 doubles per iteration)
                        var x0 = Avx.Multiply(negOne, Avx.LoadVector256(pIn + i));
                        var x1 = Avx.Multiply(negOne, Avx.LoadVector256(pIn + i + 4));
                        var x2 = Avx.Multiply(negOne, Avx.LoadVector256(pIn + i + 8));
                        var x3 = Avx.Multiply(negOne, Avx.LoadVector256(pIn + i + 12));

                        Avx.Store(pOut + i, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x0))));
                        Avx.Store(pOut + i + 4, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x1))));
                        Avx.Store(pOut + i + 8, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x2))));
                        Avx.Store(pOut + i + 12, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x3))));
                    }
                    // Handle remainder in chunks of 4
                    for (; i + 4 <= length; i += 4)
                    {
                        var x0 = Avx.Multiply(negOne, Avx.LoadVector256(pIn + i));
                        Avx.Store(pOut + i, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x0))));
                    }
                }

                // Scalar remainder
                for (; i < length; i++)
                    pOut[i] = 1.0 / (1.0 + Math.Exp(-pIn[i]));
            }
            return;
#else
            for (; i < length; i++)
            {
                output[i] = 1.0 / (1.0 + Math.Exp(-input[i]));
            }
#endif
        }

        /// <summary>
        /// Pointer-based double Sigmoid — zero bounds-checking overhead, 4x unrolled AVX2+FMA.
        /// Uses FastExpDouble256: sigmoid(x) = 1/(1+exp(-x)) for full double precision accuracy.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void SigmoidUnsafe(double* input, double* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 16)
            {
                var one = Vector256.Create(1.0);
                var negOne = Vector256.Create(-1.0);

                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    var x0 = Avx.Multiply(negOne, Avx.LoadVector256(input + i));
                    var x1 = Avx.Multiply(negOne, Avx.LoadVector256(input + i + 4));
                    var x2 = Avx.Multiply(negOne, Avx.LoadVector256(input + i + 8));
                    var x3 = Avx.Multiply(negOne, Avx.LoadVector256(input + i + 12));

                    Avx.Store(output + i, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x0))));
                    Avx.Store(output + i + 4, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x1))));
                    Avx.Store(output + i + 8, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x2))));
                    Avx.Store(output + i + 12, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x3))));
                }
            }
            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 4)
            {
                var one = Vector256.Create(1.0);
                var negOne = Vector256.Create(-1.0);

                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var x0 = Avx.Multiply(negOne, Avx.LoadVector256(input + i));
                    Avx.Store(output + i, Avx.Divide(one, Avx.Add(one, FastExpDouble256(x0))));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] = 1.0 / (1.0 + Math.Exp(-input[i]));
            }
        }

        /// <summary>
        /// Computes element-wise tanh using fast vectorized exp: tanh(x) = 2*sigmoid(2x) - 1.
        /// Processes 32 floats per iteration (4x unrolled).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Tanh(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                var vone = Vector256.Create(1.0f);
                var vtwo = Vector256.Create(2.0f);
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    // tanh(x) = 2*sigmoid(2x) - 1
                    WriteVector256(output, i, Avx.Subtract(Fma.MultiplyAdd(vtwo, FastSigmoid256(Avx.Multiply(vtwo, ReadVector256(input, i))), Vector256<float>.Zero), vone));
                    WriteVector256(output, i + 8, Avx.Subtract(Fma.MultiplyAdd(vtwo, FastSigmoid256(Avx.Multiply(vtwo, ReadVector256(input, i + 8))), Vector256<float>.Zero), vone));
                    WriteVector256(output, i + 16, Avx.Subtract(Fma.MultiplyAdd(vtwo, FastSigmoid256(Avx.Multiply(vtwo, ReadVector256(input, i + 16))), Vector256<float>.Zero), vone));
                    WriteVector256(output, i + 24, Avx.Subtract(Fma.MultiplyAdd(vtwo, FastSigmoid256(Avx.Multiply(vtwo, ReadVector256(input, i + 24))), Vector256<float>.Zero), vone));
                }
            }

            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                var vone = Vector256.Create(1.0f);
                var vtwo = Vector256.Create(2.0f);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, Avx.Subtract(Avx.Multiply(vtwo, FastSigmoid256(Avx.Multiply(vtwo, ReadVector256(input, i)))), vone));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Tanh(input[i]);
#else
                output[i] = (float)Math.Tanh(input[i]);
#endif
            }
        }

        /// <summary>
        /// Computes element-wise tanh for double precision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Tanh(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Math.Tanh(input[i]);
            }
        }

#if NET5_0_OR_GREATER
        /// <summary>
        /// Fast vectorized exp(x) using Cephes-style 6th-order minimax polynomial approximation.
        /// Range reduction: x = n*ln2 + r, then exp(x) = 2^n * exp(r).
        /// Uses IEEE 754 exponent manipulation for 2^n reconstruction.
        /// Relative error ~0.01% across [-87.3, 88.7].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> FastExp256(Vector256<float> x)
        {
            // Clamp to avoid inf/nan (exp(-87.3) ~ 1e-38, exp(88.7) ~ 3.4e38)
            var clampMin = Vector256.Create(-87.3365f);
            var clampMax = Vector256.Create(88.7228f);
            x = Avx.Max(clampMin, Avx.Min(clampMax, x));

            // Range reduction: n = round(x / ln2)
            var log2e = Vector256.Create(1.44269504088896341f); // 1/ln(2)
            var ln2hi = Vector256.Create(0.693359375f);          // ln(2) high part
            var ln2lo = Vector256.Create(-2.12194440e-4f);       // ln(2) low part

            // n = round(x * log2(e))
            var n = Avx.RoundToNearestInteger(Avx.Multiply(x, log2e));

            // r = x - n * ln2 (using hi/lo split for precision)
            var r = Fma.MultiplyAddNegated(n, ln2hi, x);
            r = Fma.MultiplyAddNegated(n, ln2lo, r);

            // Polynomial: exp(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
            // Horner's form: ((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + c0
            var c0 = Vector256.Create(1.0f);
            var c1 = Vector256.Create(1.0f);
            var c2 = Vector256.Create(0.5f);
            var c3 = Vector256.Create(0.166666666666f);  // 1/6
            var c4 = Vector256.Create(0.041666666666f);  // 1/24
            var c5 = Vector256.Create(0.008333333333f);  // 1/120
            var c6 = Vector256.Create(0.001388888888f);  // 1/720

            var poly = Fma.MultiplyAdd(c6, r, c5);
            poly = Fma.MultiplyAdd(poly, r, c4);
            poly = Fma.MultiplyAdd(poly, r, c3);
            poly = Fma.MultiplyAdd(poly, r, c2);
            poly = Fma.MultiplyAdd(poly, r, c1);
            poly = Fma.MultiplyAdd(poly, r, c0);

            // Reconstruct: exp(x) = 2^n * exp(r)
            // 2^n via IEEE 754: add n to the exponent bits of 1.0f (bias = 127)
            var nInt = Avx.ConvertToVector256Int32(n);
            var pow2n = Avx2.Add(nInt, Vector256.Create(127));
            pow2n = Avx2.ShiftLeftLogical(pow2n, 23); // shift to exponent position
            var scale = pow2n.AsSingle();

            return Avx.Multiply(poly, scale);
        }

        /// <summary>
        /// Cephes-style fast exp for Vector256&lt;double&gt; (4 doubles per vector).
        /// Uses 11th-order minimax polynomial for double precision (~1e-16 relative error).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<double> FastExpDouble256(Vector256<double> x)
        {
            // Clamp to avoid inf/nan
            var clampMin = Vector256.Create(-708.3964185322641);
            var clampMax = Vector256.Create(709.7827128933840);
            x = Avx.Max(clampMin, Avx.Min(clampMax, x));

            // Range reduction: n = round(x / ln2)
            var log2e = Vector256.Create(1.4426950408889634);
            var ln2hi = Vector256.Create(6.93145751953125E-1);
            var ln2lo = Vector256.Create(1.42860682030941723212E-6);

            var n = Avx.RoundToNearestInteger(Avx.Multiply(x, log2e));

            // r = x - n * ln2
            var r = Fma.MultiplyAddNegated(n, ln2hi, x);
            r = Fma.MultiplyAddNegated(n, ln2lo, r);

            // Polynomial approximation of exp(r) - Cephes coefficients for double
            var c0 = Vector256.Create(1.0);
            var c1 = Vector256.Create(1.0);
            var c2 = Vector256.Create(0.5);
            var c3 = Vector256.Create(1.66666666666666019037e-1);
            var c4 = Vector256.Create(4.16666666666666019037e-2);
            var c5 = Vector256.Create(8.33333333333331438267e-3);
            var c6 = Vector256.Create(1.38888888888889347740e-3);
            var c7 = Vector256.Create(1.98412698412699105610e-4);
            var c8 = Vector256.Create(2.48015873015875132200e-5);
            var c9 = Vector256.Create(2.75573192239844089230e-6);
            var c10 = Vector256.Create(2.75573192239332268710e-7);
            var c11 = Vector256.Create(2.50521083854417187751e-8);

            var poly = Fma.MultiplyAdd(c11, r, c10);
            poly = Fma.MultiplyAdd(poly, r, c9);
            poly = Fma.MultiplyAdd(poly, r, c8);
            poly = Fma.MultiplyAdd(poly, r, c7);
            poly = Fma.MultiplyAdd(poly, r, c6);
            poly = Fma.MultiplyAdd(poly, r, c5);
            poly = Fma.MultiplyAdd(poly, r, c4);
            poly = Fma.MultiplyAdd(poly, r, c3);
            poly = Fma.MultiplyAdd(poly, r, c2);
            poly = Fma.MultiplyAdd(poly, r, c1);
            poly = Fma.MultiplyAdd(poly, r, c0);

            // Reconstruct: 2^n * exp(r) using IEEE 754 double exponent manipulation
            // Double bias = 1023, exponent at bits 52-62
            var nLong = Avx2.ConvertToVector256Int64(Avx.ConvertToVector128Int32(n));
            var pow2n = Avx2.Add(nLong, Vector256.Create(1023L));
            pow2n = Avx2.ShiftLeftLogical(pow2n, 52);
            var scale = pow2n.AsDouble();

            return Avx.Multiply(poly, scale);
        }

        /// <summary>
        /// Fast vectorized sigmoid using FastExp256: sigmoid(x) = 1 / (1 + exp(-x)).
        /// Uses Cephes-style exp polynomial (~0.01% relative error) for high accuracy.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> FastSigmoid256(Vector256<float> x)
        {
            // sigmoid(x) = 1 / (1 + exp(-x))
            var negX = Avx.Subtract(Vector256<float>.Zero, x);
            var expNegX = FastExp256(negX);
            return Avx.Divide(Vector256.Create(1.0f), Avx.Add(Vector256.Create(1.0f), expNegX));
        }

        /// <summary>
        /// Fast vectorized natural logarithm using Cephes-style polynomial.
        /// Decomposes x = 2^n * m (1 &lt;= m &lt; 2), then log(x) = n*ln(2) + log(m).
        /// Uses a minimax polynomial to approximate log(m) on [1, 2].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> FastLog256(Vector256<float> x)
        {
            // Extract exponent: n = floor(log2(x))
            // IEEE 754 float: [sign:1][exponent:8][mantissa:23], bias = 127
            var vone = Vector256.Create(1.0f);
            var vzero = Vector256<float>.Zero;

            // Preserve special values: log(0)=-inf, log(negative)=NaN
            var zeroMask = Avx.CompareEqual(x, vzero);
            var negativeMask = Avx.CompareLessThan(x, vzero);
            var minNormPos = Vector256.Create(1.17549435e-38f); // smallest normal float

            // Clamp denormals to minimum normal positive for mantissa extraction
            x = Avx.Max(x, minNormPos);

            // Extract exponent as integer
            var xi = x.AsInt32();
            var exponent = Avx2.ShiftRightArithmetic(xi, 23);
            exponent = Avx2.Subtract(exponent, Vector256.Create(127));
            var e = Avx.ConvertToVector256Single(exponent);

            // Extract mantissa and set exponent to 0 (result in [1, 2))
            var mantissaMask = Vector256.Create(0x007FFFFF);
            var m = Avx2.Or(Avx2.And(xi, mantissaMask), Vector256.Create(0x3F800000)).AsSingle();

            // Adjust range to [0.5, 1) for better polynomial conditioning
            // If m > sqrt(2), divide by 2 and increment exponent
            var sqrt2 = Vector256.Create(1.4142135623730951f);
            var needAdjust = Avx.CompareGreaterThan(m, sqrt2);
            // Conditionally halve m and add 1 to e
            m = Avx.BlendVariable(m, Avx.Multiply(m, Vector256.Create(0.5f)), needAdjust);
            e = Avx.BlendVariable(e, Avx.Add(e, vone), needAdjust);

            // Now m is in [sqrt(2)/2, sqrt(2)] ~= [0.707, 1.414]
            // Compute f = m - 1 (so f is near 0)
            var f = Avx.Subtract(m, vone);

            // Polynomial approximation of log(1+f) using Horner's form
            // Coefficients from Cephes library (minimax on [0, 0.5])
            var p0 = Vector256.Create(7.0376836292e-2f);
            var p1 = Vector256.Create(-1.1514610310e-1f);
            var p2 = Vector256.Create(1.1676998740e-1f);
            var p3 = Vector256.Create(-1.2420140846e-1f);
            var p4 = Vector256.Create(1.4249322787e-1f);
            var p5 = Vector256.Create(-1.6668057665e-1f);
            var p6 = Vector256.Create(2.0000714765e-1f);
            var p7 = Vector256.Create(-2.4999993993e-1f);
            var p8 = Vector256.Create(3.3333331174e-1f);

            var f2 = Avx.Multiply(f, f);

            var poly = Fma.MultiplyAdd(p0, f, p1);
            poly = Fma.MultiplyAdd(poly, f, p2);
            poly = Fma.MultiplyAdd(poly, f, p3);
            poly = Fma.MultiplyAdd(poly, f, p4);
            poly = Fma.MultiplyAdd(poly, f, p5);
            poly = Fma.MultiplyAdd(poly, f, p6);
            poly = Fma.MultiplyAdd(poly, f, p7);
            poly = Fma.MultiplyAdd(poly, f, p8);
            poly = Avx.Multiply(poly, Avx.Multiply(f2, f)); // poly * f^3

            // log(x) = e * ln(2) + f + poly - 0.5 * f^2
            var ln2 = Vector256.Create(0.6931471805599453f);
            var halfF2 = Avx.Multiply(Vector256.Create(0.5f), f2);
            var result = Fma.MultiplyAdd(e, ln2, f);
            result = Avx.Add(result, poly);
            result = Avx.Subtract(result, halfF2);

            // Restore special values: log(0) = -inf, log(negative) = NaN
            result = Avx.BlendVariable(result, Vector256.Create(float.NegativeInfinity), zeroMask);
            result = Avx.BlendVariable(result, Vector256.Create(float.NaN), negativeMask);

            return result;
        }
#endif

        /// <summary>
        /// Computes LeakyReLU element-wise using SIMD: max(alpha * x, x).
        /// Uses AVX/SSE for vectorized comparison and blending when available.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void LeakyReLU(ReadOnlySpan<float> input, float alpha, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                var vzero = Vector256<float>.Zero;
                var valpha = Vector256.Create(alpha);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var v = ReadVector256(input, i);
                    // LeakyReLU: x > 0 ? x : alpha * x
                    var mask = Avx.CompareGreaterThan(v, vzero);
                    var scaled = Avx.Multiply(v, valpha);
                    WriteVector256(output, i, Avx.BlendVariable(scaled, v, mask));
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vzero = Vector128<float>.Zero;
                var valpha = Vector128.Create(alpha);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector128(input, i);
                    var mask = Sse.CompareGreaterThan(v, vzero);
                    var scaled = Sse.Multiply(v, valpha);
                    WriteVector128(output, i, Sse41.IsSupported
                        ? Sse41.BlendVariable(scaled, v, mask)
                        : Sse.Or(Sse.And(mask, v), Sse.AndNot(mask, scaled)));
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vzero = Vector128<float>.Zero;
                var valpha = Vector128.Create(alpha);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector128(input, i);
                    var mask = AdvSimd.CompareGreaterThan(v, vzero);
                    var scaled = AdvSimd.Multiply(v, valpha);
                    WriteVector128(output, i, AdvSimd.BitwiseSelect(mask, v, scaled));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = input[i] > 0f ? input[i] : alpha * input[i];
            }
        }

        /// <summary>
        /// Computes GELU (Gaussian Error Linear Unit) element-wise.
        /// Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        /// Optimized using SIMD vectorization where available.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void GELU(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            // Constants for GELU approximation
            const float sqrt2OverPi = 0.7978845608028654f;
            const float coeff = 0.044715f;
            const float half = 0.5f;

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && Fma.IsSupported && length >= 8)
            {
                var vSqrt2OverPi = Vector256.Create(sqrt2OverPi);
                var vCoeff = Vector256.Create(coeff);
                var vHalf = Vector256.Create(half);
                var vOne = Vector256.Create(1.0f);
                var vNegOne = Vector256.Create(-1.0f);
                // Constants for rational tanh approximation: tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)
                // Accurate to ~0.001 for |x| < 3, which covers typical GELU input ranges
                var v27 = Vector256.Create(27.0f);
                var v9 = Vector256.Create(9.0f);

                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = ReadVector256(input, i);
                    // x^3
                    var x_squared = Avx.Multiply(x, x);
                    var x_cubed = Avx.Multiply(x_squared, x);
                    // x + 0.044715 * x^3
                    var inner = Fma.MultiplyAdd(vCoeff, x_cubed, x);
                    // sqrt(2/pi) * inner = tanh argument
                    var tanh_arg = Avx.Multiply(vSqrt2OverPi, inner);

                    // Vectorized tanh approximation using rational function
                    // tanh(x) ≈ x * (27 + x²) / (27 + 9*x²) for |x| ≤ 3
                    var tanh_arg_sq = Avx.Multiply(tanh_arg, tanh_arg);
                    var numerator = Avx.Add(v27, tanh_arg_sq); // 27 + x²
                    var denominator = Fma.MultiplyAdd(v9, tanh_arg_sq, v27); // 27 + 9*x²
                    var tanh_approx = Avx.Divide(Avx.Multiply(tanh_arg, numerator), denominator);

                    // Clamp to [-1, 1] for |x| > 3 (where approximation is less accurate)
                    tanh_approx = Avx.Max(vNegOne, Avx.Min(vOne, tanh_approx));

                    // GELU = 0.5 * x * (1 + tanh)
                    var one_plus_tanh = Avx.Add(vOne, tanh_approx);
                    var result = Avx.Multiply(vHalf, Avx.Multiply(x, one_plus_tanh));
                    WriteVector256(output, i, result);
                }
            }
#endif

            // Scalar implementation for remaining elements
            for (; i < length; i++)
            {
                float x = input[i];
                float x_cubed = x * x * x;
                float inner = x + coeff * x_cubed;
                float tanh_arg = sqrt2OverPi * inner;
#if NET5_0_OR_GREATER
                float tanh_val = MathF.Tanh(tanh_arg);
#else
                float tanh_val = (float)Math.Tanh(tanh_arg);
#endif
                output[i] = half * x * (1f + tanh_val);
            }
        }

        /// <summary>
        /// Computes Mish activation element-wise: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
        /// Uses our own fast Exp/Log/Tanh kernels for maximum throughput.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mish(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;

            // Use our own fast SIMD kernels: Mish(x) = x * tanh(softplus(x))
            float[] tempBuf = ArrayPool<float>.Shared.Rent(length);
            float[] temp2Buf = ArrayPool<float>.Shared.Rent(length);
            try
            {
                var temp = tempBuf.AsSpan(0, length);
                var temp2 = temp2Buf.AsSpan(0, length);

                // Step 1: temp = exp(x)
                Exp(input, temp);
                // Step 2: temp = 1 + exp(x), then log -> softplus
                for (int i = 0; i < length; i++)
                {
                    temp[i] = input[i] > 20f ? input[i] :
#if NET5_0_OR_GREATER
                        MathF.Log(1f + temp[i]);
#else
                        (float)Math.Log(1.0 + temp[i]);
#endif
                }
                // Step 3: temp2 = tanh(softplus(x))
                Tanh(temp, temp2);
                // Step 4: output = x * tanh(softplus(x))
                VectorMultiply(input, temp2, output);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(tempBuf);
                ArrayPool<float>.Shared.Return(temp2Buf);
            }
        }

        /// <summary>
        /// Computes Swish/SiLU activation element-wise: x * sigmoid(x).
        /// Uses our own fast Sigmoid and VectorMultiply kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Swish(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;

            float[] tempBuf = ArrayPool<float>.Shared.Rent(length);
            try
            {
                var temp = tempBuf.AsSpan(0, length);
                Sigmoid(input, temp);
                VectorMultiply(input, temp, output);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(tempBuf);
            }
        }

        /// <summary>
        /// Computes ELU (Exponential Linear Unit) element-wise: x if x > 0, alpha * (exp(x) - 1) otherwise.
        /// Uses FastExp256 for vectorized exp computation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ELU(ReadOnlySpan<float> input, float alpha, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
            {
                var vzero = Vector256<float>.Zero;
                var valpha = Vector256.Create(alpha);
                var vone = Vector256.Create(1.0f);
                int simdLength = length & ~7;

                for (; i < simdLength; i += 8)
                {
                    var x = ReadVector256(input, i);
                    var expx = FastExp256(x);
                    // negative path: alpha * (exp(x) - 1)
                    var negResult = Avx.Multiply(valpha, Avx.Subtract(expx, vone));
                    // mask: x > 0
                    var mask = Avx.CompareGreaterThan(x, vzero);
                    // blend: positive keeps x, negative gets alpha*(exp(x)-1)
                    WriteVector256(output, i, Avx.BlendVariable(negResult, x, mask));
                }
            }
#endif

            // Scalar fallback
            for (; i < length; i++)
            {
                float x = input[i];
                if (x > 0f)
                {
                    output[i] = x;
                }
                else
                {
#if NET5_0_OR_GREATER
                    output[i] = alpha * (MathF.Exp(x) - 1f);
#else
                    output[i] = alpha * ((float)Math.Exp(x) - 1f);
#endif
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(ReadOnlySpan<float> data)
        {
            int length = data.Length;
            int i = 0;
            float sum = 0f;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                // 4-way parallel accumulation to hide add latency (3 cycles on Zen2)
                var vsum0 = Vector256<float>.Zero;
                var vsum1 = Vector256<float>.Zero;
                var vsum2 = Vector256<float>.Zero;
                var vsum3 = Vector256<float>.Zero;
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    vsum0 = Avx.Add(vsum0, ReadVector256(data, i));
                    vsum1 = Avx.Add(vsum1, ReadVector256(data, i + 8));
                    vsum2 = Avx.Add(vsum2, ReadVector256(data, i + 16));
                    vsum3 = Avx.Add(vsum3, ReadVector256(data, i + 24));
                }
                // Reduce 4 accumulators
                vsum0 = Avx.Add(Avx.Add(vsum0, vsum1), Avx.Add(vsum2, vsum3));
                sum += HorizontalSum(vsum0);
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vsum = Vector256<float>.Zero;
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    vsum = Avx.Add(vsum, ReadVector256(data, i));
                }
                sum += HorizontalSum(vsum);
            }
            else if (Sse.IsSupported && length - i >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    vsum = Sse.Add(vsum, ReadVector128(data, i));
                }

                sum += HorizontalSum(vsum);
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    vsum = AdvSimd.Add(vsum, ReadVector128(data, i));
                }

                sum += HorizontalSum(vsum);
            }
#endif

            for (; i < length; i++)
            {
                sum += data[i];
            }

            return sum;
        }

        /// <summary>
        /// Computes element-wise floor (largest integer less than or equal to each element).
        /// Uses AVX/SSE4.1 intrinsics when available, otherwise falls back to scalar Math.Floor.
        /// </summary>
        /// <param name="input">Source span of float values.</param>
        /// <param name="output">Destination span for floor results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Floor(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var v = ReadVector256(input, i);
                    WriteVector256(output, i, Avx.Floor(v));
                }
            }
            else if (Sse41.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector128(input, i);
                    WriteVector128(output, i, Sse41.Floor(v));
                }
            }
            else if (AdvSimd.Arm64.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector128(input, i);
                    WriteVector128(output, i, AdvSimd.RoundToNegativeInfinity(v));
                }
            }
#endif

            // Scalar fallback for remaining elements or when SIMD not available
            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Floor(input[i]);
#else
                output[i] = (float)Math.Floor(input[i]);
#endif
            }
        }

        /// <summary>
        /// Computes element-wise ceiling (smallest integer greater than or equal to each element).
        /// Uses AVX/SSE4.1 intrinsics when available, otherwise falls back to scalar Math.Ceiling.
        /// </summary>
        /// <param name="input">Source span of float values.</param>
        /// <param name="output">Destination span for ceiling results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Ceiling(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var v = ReadVector256(input, i);
                    WriteVector256(output, i, Avx.Ceiling(v));
                }
            }
            else if (Sse41.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector128(input, i);
                    WriteVector128(output, i, Sse41.Ceiling(v));
                }
            }
            else if (AdvSimd.Arm64.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector128(input, i);
                    WriteVector128(output, i, AdvSimd.RoundToPositiveInfinity(v));
                }
            }
#endif

            // Scalar fallback for remaining elements or when SIMD not available
            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Ceiling(input[i]);
#else
                output[i] = (float)Math.Ceiling(input[i]);
#endif
            }
        }

        /// <summary>
        /// Computes element-wise fractional part (x - floor(x)).
        /// Uses SIMD floor operation and subtraction when available.
        /// </summary>
        /// <param name="input">Source span of float values.</param>
        /// <param name="output">Destination span for fractional part results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Frac(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var v = ReadVector256(input, i);
                    var floored = Avx.Floor(v);
                    WriteVector256(output, i, Avx.Subtract(v, floored));
                }
            }
            else if (Sse41.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector128(input, i);
                    var floored = Sse41.Floor(v);
                    WriteVector128(output, i, Sse.Subtract(v, floored));
                }
            }
            else if (AdvSimd.Arm64.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector128(input, i);
                    var floored = AdvSimd.RoundToNegativeInfinity(v);
                    WriteVector128(output, i, AdvSimd.Subtract(v, floored));
                }
            }
#endif

            // Scalar fallback for remaining elements or when SIMD not available
            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = input[i] - MathF.Floor(input[i]);
#else
                output[i] = input[i] - (float)Math.Floor(input[i]);
#endif
            }
        }

        /// <summary>
        /// Computes element-wise floor for double-precision values.
        /// Uses AVX intrinsics when available, otherwise falls back to scalar Math.Floor.
        /// </summary>
        /// <param name="input">Source span of double values.</param>
        /// <param name="output">Destination span for floor results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Floor(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector256Double(input, i);
                    WriteVector256Double(output, i, Avx.Floor(v));
                }
            }
            else if (Sse41.IsSupported && length >= 2)
            {
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var v = ReadVector128Double(input, i);
                    WriteVector128Double(output, i, Sse41.Floor(v));
                }
            }
            else if (AdvSimd.Arm64.IsSupported && length >= 2)
            {
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var v = ReadVector128Double(input, i);
                    WriteVector128Double(output, i, AdvSimd.Arm64.Floor(v));
                }
            }
#endif

            // Scalar fallback for remaining elements or when SIMD not available
            for (; i < length; i++)
            {
                output[i] = Math.Floor(input[i]);
            }
        }

        /// <summary>
        /// Computes element-wise ceiling for double-precision values.
        /// Uses AVX intrinsics when available, otherwise falls back to scalar Math.Ceiling.
        /// </summary>
        /// <param name="input">Source span of double values.</param>
        /// <param name="output">Destination span for ceiling results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Ceiling(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector256Double(input, i);
                    WriteVector256Double(output, i, Avx.Ceiling(v));
                }
            }
            else if (Sse41.IsSupported && length >= 2)
            {
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var v = ReadVector128Double(input, i);
                    WriteVector128Double(output, i, Sse41.Ceiling(v));
                }
            }
            else if (AdvSimd.Arm64.IsSupported && length >= 2)
            {
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var v = ReadVector128Double(input, i);
                    WriteVector128Double(output, i, AdvSimd.Arm64.Ceiling(v));
                }
            }
#endif

            // Scalar fallback for remaining elements or when SIMD not available
            for (; i < length; i++)
            {
                output[i] = Math.Ceiling(input[i]);
            }
        }

        /// <summary>
        /// Computes element-wise fractional part for double-precision values.
        /// </summary>
        /// <param name="input">Source span of double values.</param>
        /// <param name="output">Destination span for fractional part results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Frac(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector256Double(input, i);
                    var floored = Avx.Floor(v);
                    WriteVector256Double(output, i, Avx.Subtract(v, floored));
                }
            }
            else if (Sse41.IsSupported && length >= 2)
            {
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var v = ReadVector128Double(input, i);
                    var floored = Sse41.Floor(v);
                    WriteVector128Double(output, i, Sse2.Subtract(v, floored));
                }
            }
            else if (AdvSimd.Arm64.IsSupported && length >= 2)
            {
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var v = ReadVector128Double(input, i);
                    var floored = AdvSimd.Arm64.Floor(v);
                    WriteVector128Double(output, i, AdvSimd.Arm64.Subtract(v, floored));
                }
            }
#endif

            // Scalar fallback for remaining elements or when SIMD not available
            for (; i < length; i++)
            {
                output[i] = input[i] - Math.Floor(input[i]);
            }
        }

        /// <summary>
        /// Computes element-wise sine for single-precision values.
        /// </summary>
        /// <param name="input">Source span of float values in radians.</param>
        /// <param name="output">Destination span for sine results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sin(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            for (int i = 0; i < input.Length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Sin(input[i]);
#else
                output[i] = (float)Math.Sin(input[i]);
#endif
            }
        }

        /// <summary>
        /// Computes element-wise cosine for single-precision values.
        /// </summary>
        /// <param name="input">Source span of float values in radians.</param>
        /// <param name="output">Destination span for cosine results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Cos(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            for (int i = 0; i < input.Length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Cos(input[i]);
#else
                output[i] = (float)Math.Cos(input[i]);
#endif
            }
        }

        /// <summary>
        /// Computes element-wise sine and cosine simultaneously for single-precision values.
        /// More efficient than computing sin and cos separately.
        /// </summary>
        /// <param name="input">Source span of float values in radians.</param>
        /// <param name="sinOutput">Destination span for sine results.</param>
        /// <param name="cosOutput">Destination span for cosine results.</param>
        /// <exception cref="ArgumentException">Thrown when span lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SinCos(ReadOnlySpan<float> input, Span<float> sinOutput, Span<float> cosOutput)
        {
            if (input.Length != sinOutput.Length || input.Length != cosOutput.Length)
            {
                throw new ArgumentException("All spans must have the same length.");
            }

            for (int i = 0; i < input.Length; i++)
            {
#if NET5_0_OR_GREATER
                (sinOutput[i], cosOutput[i]) = MathF.SinCos(input[i]);
#else
                sinOutput[i] = (float)Math.Sin(input[i]);
                cosOutput[i] = (float)Math.Cos(input[i]);
#endif
            }
        }

        /// <summary>
        /// Computes element-wise sine for double-precision values.
        /// </summary>
        /// <param name="input">Source span of double values in radians.</param>
        /// <param name="output">Destination span for sine results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sin(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Math.Sin(input[i]);
            }
        }

        /// <summary>
        /// Computes element-wise cosine for double-precision values.
        /// </summary>
        /// <param name="input">Source span of double values in radians.</param>
        /// <param name="output">Destination span for cosine results.</param>
        /// <exception cref="ArgumentException">Thrown when input and output lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Cos(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Math.Cos(input[i]);
            }
        }

        /// <summary>
        /// Computes element-wise sine and cosine simultaneously for double-precision values.
        /// More efficient than computing sin and cos separately.
        /// </summary>
        /// <param name="input">Source span of double values in radians.</param>
        /// <param name="sinOutput">Destination span for sine results.</param>
        /// <param name="cosOutput">Destination span for cosine results.</param>
        /// <exception cref="ArgumentException">Thrown when span lengths differ.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SinCos(ReadOnlySpan<double> input, Span<double> sinOutput, Span<double> cosOutput)
        {
            if (input.Length != sinOutput.Length || input.Length != cosOutput.Length)
            {
                throw new ArgumentException("All spans must have the same length.");
            }

            for (int i = 0; i < input.Length; i++)
            {
#if NET7_0_OR_GREATER
                (sinOutput[i], cosOutput[i]) = Math.SinCos(input[i]);
#else
                sinOutput[i] = Math.Sin(input[i]);
                cosOutput[i] = Math.Cos(input[i]);
#endif
            }
        }

        #region Missing Math Kernels (Log, Sqrt, Abs, Negate, Clamp, Pow, SoftMax, Max, Min)

        /// <summary>Element-wise natural log using SIMD with Cephes-style polynomial approximation.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Log(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // Use FastLog256 on ALL .NET versions — benchmarks prove SVML auto-vectorization
            // does NOT work through Span indexing, so our polynomial is 5-20x faster.
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(output, i, FastLog256(ReadVector256(input, i)));
                    WriteVector256(output, i + 8, FastLog256(ReadVector256(input, i + 8)));
                    WriteVector256(output, i + 16, FastLog256(ReadVector256(input, i + 16)));
                    WriteVector256(output, i + 24, FastLog256(ReadVector256(input, i + 24)));
                }
            }

            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, FastLog256(ReadVector256(input, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Log(input[i]);
#else
                output[i] = (float)Math.Log(input[i]);
#endif
            }
        }

        /// <summary>Element-wise natural log for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Log(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Math.Log(input[i]);
            }
        }

        /// <summary>Element-wise log base 2 using SIMD: log2(x) = log(x) / ln(2).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Log2(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET8_0_OR_GREATER
            // On .NET 8+, the JIT auto-vectorizes MathF.Log2 using SVML
            int unrolled = length & ~3;
            for (; i < unrolled; i += 4)
            {
                output[i] = MathF.Log2(input[i]);
                output[i + 1] = MathF.Log2(input[i + 1]);
                output[i + 2] = MathF.Log2(input[i + 2]);
                output[i + 3] = MathF.Log2(input[i + 3]);
            }
#elif NET5_0_OR_GREATER
            // On .NET 5-7, use our FastLog256 polynomial
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                var log2e = Vector256.Create(1.44269504088896341f); // 1/ln(2) = log2(e)
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    WriteVector256(output, i, Avx.Multiply(FastLog256(ReadVector256(input, i)), log2e));
                    WriteVector256(output, i + 8, Avx.Multiply(FastLog256(ReadVector256(input, i + 8)), log2e));
                    WriteVector256(output, i + 16, Avx.Multiply(FastLog256(ReadVector256(input, i + 16)), log2e));
                    WriteVector256(output, i + 24, Avx.Multiply(FastLog256(ReadVector256(input, i + 24)), log2e));
                }
            }

            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                var log2e = Vector256.Create(1.44269504088896341f);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, Avx.Multiply(FastLog256(ReadVector256(input, i)), log2e));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Log2(input[i]);
#else
                output[i] = (float)(Math.Log(input[i]) / Math.Log(2.0));
#endif
            }
        }

        /// <summary>Element-wise log base 2 for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Log2(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            for (int i = 0; i < input.Length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = Math.Log2(input[i]);
#else
                output[i] = Math.Log(input[i]) / Math.Log(2.0);
#endif
            }
        }

        /// <summary>Element-wise square root using AVX.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sqrt(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, Avx.Sqrt(ReadVector256(input, i)));
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(output, i, Sse.Sqrt(ReadVector128(input, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Sqrt(input[i]);
#else
                output[i] = (float)Math.Sqrt(input[i]);
#endif
            }
        }

        /// <summary>Element-wise square root for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Sqrt(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(output, i, Avx.Sqrt(ReadVector256Double(input, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = Math.Sqrt(input[i]);
            }
        }

        /// <summary>Element-wise absolute value using AVX bitwise AND with sign mask.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Abs(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                // Clear sign bit: AND with 0x7FFFFFFF
                var signMask = Vector256.Create(0x7FFFFFFF).AsSingle();
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, Avx.And(ReadVector256(input, i), signMask));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = Math.Abs(input[i]);
            }
        }

        /// <summary>Element-wise absolute value for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Abs(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                var signMask = Vector256.Create(0x7FFFFFFFFFFFFFFFL).AsDouble();
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(output, i, Avx.And(ReadVector256Double(input, i), signMask));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = Math.Abs(input[i]);
            }
        }

        /// <summary>Element-wise negation using AVX XOR with sign bit.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Negate(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                var vzero = Vector256<float>.Zero;
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, Avx.Subtract(vzero, ReadVector256(input, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = -input[i];
            }
        }

        /// <summary>Element-wise negation for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Negate(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                var vzero = Vector256<double>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(output, i, Avx.Subtract(vzero, ReadVector256Double(input, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = -input[i];
            }
        }

        /// <summary>Element-wise clamp to [min, max] range.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Clamp(ReadOnlySpan<float> input, float min, float max, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                var vmin = Vector256.Create(min);
                var vmax = Vector256.Create(max);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, Avx.Max(vmin, Avx.Min(vmax, ReadVector256(input, i))));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = input[i] < min ? min : (input[i] > max ? max : input[i]);
            }
        }

        /// <summary>Element-wise power: result[i] = base[i] ^ exp[i].</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Pow(ReadOnlySpan<float> baseValues, ReadOnlySpan<float> exponents, Span<float> output)
        {
            if (baseValues.Length != exponents.Length || baseValues.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            for (int i = 0; i < baseValues.Length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Pow(baseValues[i], exponents[i]);
#else
                output[i] = (float)Math.Pow(baseValues[i], exponents[i]);
#endif
            }
        }

        /// <summary>Element-wise power with scalar exponent: result[i] = base[i] ^ exp.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Pow(ReadOnlySpan<float> baseValues, float exponent, Span<float> output)
        {
            if (baseValues.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            for (int i = 0; i < baseValues.Length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Pow(baseValues[i], exponent);
#else
                output[i] = (float)Math.Pow(baseValues[i], exponent);
#endif
            }
        }

        /// <summary>Element-wise clamp to [min, max] range for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Clamp(ReadOnlySpan<double> input, double min, double max, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                var vmin = Vector256.Create(min);
                var vmax = Vector256.Create(max);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(output, i, Avx.Max(vmin, Avx.Min(vmax, ReadVector256Double(input, i))));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = input[i] < min ? min : (input[i] > max ? max : input[i]);
            }
        }

        /// <summary>Element-wise power with scalar exponent for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Pow(ReadOnlySpan<double> baseValues, double exponent, Span<double> output)
        {
            if (baseValues.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            for (int i = 0; i < baseValues.Length; i++)
            {
                output[i] = Math.Pow(baseValues[i], exponent);
            }
        }

        /// <summary>
        /// Computes SoftMax: output[i] = exp(x[i] - max(x)) / sum(exp(x - max(x))).
        /// Numerically stable via max subtraction.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SoftMax(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            // Step 1: Find max for numerical stability
            float maxVal = Max(input);
            // Step 2: Compute exp(x - max)
            int length = input.Length;
            for (int i = 0; i < length; i++)
            {
                output[i] = input[i] - maxVal;
            }
            Exp(output, output);
            // Step 3: Divide by sum
            float sum = Sum(output);
            if (sum > 0)
            {
                MultiplyScalar(output, 1.0f / sum, output);
            }
        }

        /// <summary>Returns the maximum value in the span with 4-way accumulation.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Max(ReadOnlySpan<float> data)
        {
            if (data.Length == 0) throw new ArgumentException("Span must not be empty.");

            int length = data.Length;
            int i = 0;
            float max = float.NegativeInfinity;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vmax0 = Vector256.Create(float.NegativeInfinity);
                var vmax1 = vmax0;
                var vmax2 = vmax0;
                var vmax3 = vmax0;
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    vmax0 = Avx.Max(vmax0, ReadVector256(data, i));
                    vmax1 = Avx.Max(vmax1, ReadVector256(data, i + 8));
                    vmax2 = Avx.Max(vmax2, ReadVector256(data, i + 16));
                    vmax3 = Avx.Max(vmax3, ReadVector256(data, i + 24));
                }
                vmax0 = Avx.Max(Avx.Max(vmax0, vmax1), Avx.Max(vmax2, vmax3));
                max = HorizontalMax(vmax0);
            }

            if (Avx.IsSupported && length - i >= 8)
            {
                var vmax = Vector256.Create(max);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    vmax = Avx.Max(vmax, ReadVector256(data, i));
                }
                max = HorizontalMax(vmax);
            }
#endif

            for (; i < length; i++)
            {
                if (data[i] > max) max = data[i];
            }

            return max;
        }

        /// <summary>Returns the minimum value in the span with 4-way accumulation.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Min(ReadOnlySpan<float> data)
        {
            if (data.Length == 0) throw new ArgumentException("Span must not be empty.");

            int length = data.Length;
            int i = 0;
            float min = float.PositiveInfinity;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vmin0 = Vector256.Create(float.PositiveInfinity);
                var vmin1 = vmin0;
                var vmin2 = vmin0;
                var vmin3 = vmin0;
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    vmin0 = Avx.Min(vmin0, ReadVector256(data, i));
                    vmin1 = Avx.Min(vmin1, ReadVector256(data, i + 8));
                    vmin2 = Avx.Min(vmin2, ReadVector256(data, i + 16));
                    vmin3 = Avx.Min(vmin3, ReadVector256(data, i + 24));
                }
                vmin0 = Avx.Min(Avx.Min(vmin0, vmin1), Avx.Min(vmin2, vmin3));
                min = HorizontalMin(vmin0);
            }

            if (Avx.IsSupported && length - i >= 8)
            {
                var vmin = Vector256.Create(min);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    vmin = Avx.Min(vmin, ReadVector256(data, i));
                }
                min = HorizontalMin(vmin);
            }
#endif

            for (; i < length; i++)
            {
                if (data[i] < min) min = data[i];
            }

            return min;
        }

        /// <summary>Cosine similarity: dot(a,b) / (||a|| * ||b||).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Input spans must have the same length.");

            float dot = DotProduct(a, b);
            float normA = DotProduct(a, a);
            float normB = DotProduct(b, b);

#if NET5_0_OR_GREATER
            float denominator = MathF.Sqrt(normA) * MathF.Sqrt(normB);
#else
            float denominator = (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
#endif
            return denominator > 0 ? dot / denominator : 0f;
        }

        #endregion

        #region Double Precision Arithmetic

        /// <summary>Element-wise addition for double precision with 4x unrolling.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorAdd(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    WriteVector256Double(result, i, Avx.Add(ReadVector256Double(a, i), ReadVector256Double(b, i)));
                    WriteVector256Double(result, i + 4, Avx.Add(ReadVector256Double(a, i + 4), ReadVector256Double(b, i + 4)));
                    WriteVector256Double(result, i + 8, Avx.Add(ReadVector256Double(a, i + 8), ReadVector256Double(b, i + 8)));
                    WriteVector256Double(result, i + 12, Avx.Add(ReadVector256Double(a, i + 12), ReadVector256Double(b, i + 12)));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(result, i, Avx.Add(ReadVector256Double(a, i), ReadVector256Double(b, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>Element-wise subtraction for double precision with 4x unrolling.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorSubtract(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    WriteVector256Double(result, i, Avx.Subtract(ReadVector256Double(a, i), ReadVector256Double(b, i)));
                    WriteVector256Double(result, i + 4, Avx.Subtract(ReadVector256Double(a, i + 4), ReadVector256Double(b, i + 4)));
                    WriteVector256Double(result, i + 8, Avx.Subtract(ReadVector256Double(a, i + 8), ReadVector256Double(b, i + 8)));
                    WriteVector256Double(result, i + 12, Avx.Subtract(ReadVector256Double(a, i + 12), ReadVector256Double(b, i + 12)));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(result, i, Avx.Subtract(ReadVector256Double(a, i), ReadVector256Double(b, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] - b[i];
            }
        }

        /// <summary>Element-wise multiplication for double precision with 4x unrolling.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorMultiply(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    WriteVector256Double(result, i, Avx.Multiply(ReadVector256Double(a, i), ReadVector256Double(b, i)));
                    WriteVector256Double(result, i + 4, Avx.Multiply(ReadVector256Double(a, i + 4), ReadVector256Double(b, i + 4)));
                    WriteVector256Double(result, i + 8, Avx.Multiply(ReadVector256Double(a, i + 8), ReadVector256Double(b, i + 8)));
                    WriteVector256Double(result, i + 12, Avx.Multiply(ReadVector256Double(a, i + 12), ReadVector256Double(b, i + 12)));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(result, i, Avx.Multiply(ReadVector256Double(a, i), ReadVector256Double(b, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        /// <summary>Element-wise division for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorDivide(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    WriteVector256Double(result, i, Avx.Divide(ReadVector256Double(a, i), ReadVector256Double(b, i)));
                    WriteVector256Double(result, i + 4, Avx.Divide(ReadVector256Double(a, i + 4), ReadVector256Double(b, i + 4)));
                    WriteVector256Double(result, i + 8, Avx.Divide(ReadVector256Double(a, i + 8), ReadVector256Double(b, i + 8)));
                    WriteVector256Double(result, i + 12, Avx.Divide(ReadVector256Double(a, i + 12), ReadVector256Double(b, i + 12)));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(result, i, Avx.Divide(ReadVector256Double(a, i), ReadVector256Double(b, i)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] / b[i];
            }
        }

        /// <summary>Adds a scalar to each double element.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
        {
            if (a.Length != result.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    WriteVector256Double(result, i, Avx.Add(ReadVector256Double(a, i), vs));
                    WriteVector256Double(result, i + 4, Avx.Add(ReadVector256Double(a, i + 4), vs));
                    WriteVector256Double(result, i + 8, Avx.Add(ReadVector256Double(a, i + 8), vs));
                    WriteVector256Double(result, i + 12, Avx.Add(ReadVector256Double(a, i + 12), vs));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(result, i, Avx.Add(ReadVector256Double(a, i), vs));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] + scalar;
            }
        }

        /// <summary>Multiplies each double element by a scalar.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
        {
            if (a.Length != result.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    WriteVector256Double(result, i, Avx.Multiply(ReadVector256Double(a, i), vs));
                    WriteVector256Double(result, i + 4, Avx.Multiply(ReadVector256Double(a, i + 4), vs));
                    WriteVector256Double(result, i + 8, Avx.Multiply(ReadVector256Double(a, i + 8), vs));
                    WriteVector256Double(result, i + 12, Avx.Multiply(ReadVector256Double(a, i + 12), vs));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(result, i, Avx.Multiply(ReadVector256Double(a, i), vs));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] * scalar;
            }
        }

        /// <summary>Pointer-based MultiplyScalar for double — zero bounds-checking overhead.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MultiplyScalarUnsafe(double* a, double scalar, double* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    Avx.Store(result + i, Avx.Multiply(Avx.LoadVector256(a + i), vs));
                    Avx.Store(result + i + 4, Avx.Multiply(Avx.LoadVector256(a + i + 4), vs));
                    Avx.Store(result + i + 8, Avx.Multiply(Avx.LoadVector256(a + i + 8), vs));
                    Avx.Store(result + i + 12, Avx.Multiply(Avx.LoadVector256(a + i + 12), vs));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    Avx.Store(result + i, Avx.Multiply(Avx.LoadVector256(a + i), vs));
                }
            }
#endif
            for (; i < length; i++)
            {
                result[i] = a[i] * scalar;
            }
        }

        /// <summary>Pointer-based VectorAdd for double — zero bounds-checking overhead.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorAddUnsafe(double* a, double* b, double* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    Avx.Store(result + i, Avx.Add(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                    Avx.Store(result + i + 4, Avx.Add(Avx.LoadVector256(a + i + 4), Avx.LoadVector256(b + i + 4)));
                    Avx.Store(result + i + 8, Avx.Add(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8)));
                    Avx.Store(result + i + 12, Avx.Add(Avx.LoadVector256(a + i + 12), Avx.LoadVector256(b + i + 12)));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    Avx.Store(result + i, Avx.Add(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>Pointer-based VectorSubtract for double — zero bounds-checking overhead.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorSubtractUnsafe(double* a, double* b, double* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    Avx.Store(result + i, Avx.Subtract(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                    Avx.Store(result + i + 4, Avx.Subtract(Avx.LoadVector256(a + i + 4), Avx.LoadVector256(b + i + 4)));
                    Avx.Store(result + i + 8, Avx.Subtract(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8)));
                    Avx.Store(result + i + 12, Avx.Subtract(Avx.LoadVector256(a + i + 12), Avx.LoadVector256(b + i + 12)));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    Avx.Store(result + i, Avx.Subtract(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                result[i] = a[i] - b[i];
            }
        }

        /// <summary>Pointer-based VectorMultiply for double — zero bounds-checking overhead.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorMultiplyUnsafe(double* a, double* b, double* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    Avx.Store(result + i, Avx.Multiply(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                    Avx.Store(result + i + 4, Avx.Multiply(Avx.LoadVector256(a + i + 4), Avx.LoadVector256(b + i + 4)));
                    Avx.Store(result + i + 8, Avx.Multiply(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8)));
                    Avx.Store(result + i + 12, Avx.Multiply(Avx.LoadVector256(a + i + 12), Avx.LoadVector256(b + i + 12)));
                }
            }
            if (Avx.IsSupported && length - i >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    Avx.Store(result + i, Avx.Multiply(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                }
            }
#endif
            for (; i < length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        /// <summary>Divides each double element by a scalar.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void DivideScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
        {
            MultiplyScalar(a, 1.0 / scalar, result);
        }

        /// <summary>Subtracts a scalar from each double element.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SubtractScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
        {
            AddScalar(a, -scalar, result);
        }

        /// <summary>Sum for double precision with 4-way parallel accumulation.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Sum(ReadOnlySpan<double> data)
        {
            int length = data.Length;
            int i = 0;
            double sum = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                var vsum0 = Vector256<double>.Zero;
                var vsum1 = Vector256<double>.Zero;
                var vsum2 = Vector256<double>.Zero;
                var vsum3 = Vector256<double>.Zero;
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    vsum0 = Avx.Add(vsum0, ReadVector256Double(data, i));
                    vsum1 = Avx.Add(vsum1, ReadVector256Double(data, i + 4));
                    vsum2 = Avx.Add(vsum2, ReadVector256Double(data, i + 8));
                    vsum3 = Avx.Add(vsum3, ReadVector256Double(data, i + 12));
                }
                sum += HorizontalSum(Avx.Add(Avx.Add(vsum0, vsum1), Avx.Add(vsum2, vsum3)));
            }

            if (Avx.IsSupported && length - i >= 4)
            {
                var vsum = Vector256<double>.Zero;
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    vsum = Avx.Add(vsum, ReadVector256Double(data, i));
                }
                sum += HorizontalSum(vsum);
            }
#endif

            for (; i < length; i++)
            {
                sum += data[i];
            }

            return sum;
        }

        /// <summary>Dot product for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double DotProduct(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Input spans must have the same length.");

            int length = a.Length;
            int i = 0;
            double sum = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 16)
            {
                var vsum0 = Vector256<double>.Zero;
                var vsum1 = Vector256<double>.Zero;
                var vsum2 = Vector256<double>.Zero;
                var vsum3 = Vector256<double>.Zero;
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    var va0 = ReadVector256Double(a, i);
                    var vb0 = ReadVector256Double(b, i);
                    var va1 = ReadVector256Double(a, i + 4);
                    var vb1 = ReadVector256Double(b, i + 4);
                    var va2 = ReadVector256Double(a, i + 8);
                    var vb2 = ReadVector256Double(b, i + 8);
                    var va3 = ReadVector256Double(a, i + 12);
                    var vb3 = ReadVector256Double(b, i + 12);
                    vsum0 = Fma.IsSupported ? Fma.MultiplyAdd(va0, vb0, vsum0) : Avx.Add(vsum0, Avx.Multiply(va0, vb0));
                    vsum1 = Fma.IsSupported ? Fma.MultiplyAdd(va1, vb1, vsum1) : Avx.Add(vsum1, Avx.Multiply(va1, vb1));
                    vsum2 = Fma.IsSupported ? Fma.MultiplyAdd(va2, vb2, vsum2) : Avx.Add(vsum2, Avx.Multiply(va2, vb2));
                    vsum3 = Fma.IsSupported ? Fma.MultiplyAdd(va3, vb3, vsum3) : Avx.Add(vsum3, Avx.Multiply(va3, vb3));
                }
                sum += HorizontalSum(Avx.Add(Avx.Add(vsum0, vsum1), Avx.Add(vsum2, vsum3)));
            }

            if (Avx.IsSupported && length - i >= 4)
            {
                var vsum = Vector256<double>.Zero;
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector256Double(a, i);
                    var vb = ReadVector256Double(b, i);
                    vsum = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, vsum) : Avx.Add(vsum, Avx.Multiply(va, vb));
                }
                sum += HorizontalSum(vsum);
            }
#endif

            for (; i < length; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        /// <summary>Max for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Max(ReadOnlySpan<double> data)
        {
            if (data.Length == 0) throw new ArgumentException("Span must not be empty.");
            double max = double.NegativeInfinity;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] > max) max = data[i];
            }
            return max;
        }

        /// <summary>Min for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Min(ReadOnlySpan<double> data)
        {
            if (data.Length == 0) throw new ArgumentException("Span must not be empty.");
            double min = double.PositiveInfinity;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] < min) min = data[i];
            }
            return min;
        }

        /// <summary>Cosine similarity for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double CosineSimilarity(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Input spans must have the same length.");

            double dot = DotProduct(a, b);
            double normA = DotProduct(a, a);
            double normB = DotProduct(b, b);
            double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
            return denominator > 0 ? dot / denominator : 0;
        }

        /// <summary>SoftMax for double precision.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SoftMax(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            double maxVal = Max(input);
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Math.Exp(input[i] - maxVal);
            }
            double sum = Sum(output);
            if (sum > 0)
            {
                MultiplyScalar(output, 1.0 / sum, output);
            }
        }

        #endregion

        #region Double Activation Functions

        /// <summary>
        /// Computes ReLU element-wise using SIMD: max(0, x).
        /// Uses AVX/SSE for vectorized comparison when available.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReLU(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                var vzero = Vector256<double>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector256Double(input, i);
                    WriteVector256Double(output, i, Avx.Max(v, vzero));
                }
            }
            else if (Sse2.IsSupported && length >= 2)
            {
                var vzero = Vector128<double>.Zero;
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var v = ReadVector128Double(input, i);
                    WriteVector128Double(output, i, Sse2.Max(v, vzero));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = input[i] > 0 ? input[i] : 0;
            }
        }

        /// <summary>
        /// Computes LeakyReLU element-wise using SIMD: max(alpha * x, x).
        /// Uses AVX/SSE for vectorized comparison and blending when available.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void LeakyReLU(ReadOnlySpan<double> input, double alpha, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 4)
            {
                var vzero = Vector256<double>.Zero;
                var valpha = Vector256.Create(alpha);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var v = ReadVector256Double(input, i);
                    // LeakyReLU: x > 0 ? x : alpha * x
                    var mask = Avx.CompareGreaterThan(v, vzero);
                    var scaled = Avx.Multiply(v, valpha);
                    WriteVector256Double(output, i, Avx.BlendVariable(scaled, v, mask));
                }
            }
            else if (Sse2.IsSupported && length >= 2)
            {
                var vzero = Vector128<double>.Zero;
                var valpha = Vector128.Create(alpha);
                int simdLength = length & ~1;
                for (; i < simdLength; i += 2)
                {
                    var v = ReadVector128Double(input, i);
                    var mask = Sse2.CompareGreaterThan(v, vzero);
                    var scaled = Sse2.Multiply(v, valpha);
                    WriteVector128Double(output, i, Sse41.IsSupported
                        ? Sse41.BlendVariable(scaled, v, mask)
                        : Sse2.Or(Sse2.And(mask, v), Sse2.AndNot(mask, scaled)));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = input[i] > 0 ? input[i] : alpha * input[i];
            }
        }

        /// <summary>
        /// Computes GELU (Gaussian Error Linear Unit) element-wise for double precision.
        /// Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void GELU(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            // Constants for GELU approximation
            const double sqrt2OverPi = 0.7978845608028654;
            const double coeff = 0.044715;
            const double half = 0.5;

            int length = output.Length;

            // Scalar implementation
            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                double x_cubed = x * x * x;
                double inner = x + coeff * x_cubed;
                double tanh_arg = sqrt2OverPi * inner;
                double tanh_val = Math.Tanh(tanh_arg);
                output[i] = half * x * (1.0 + tanh_val);
            }
        }

        /// <summary>
        /// Computes Mish activation element-wise for double precision: x * tanh(softplus(x)).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mish(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;

            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                double softplus = x > 20.0 ? x : Math.Log(1.0 + Math.Exp(x));
                output[i] = x * Math.Tanh(softplus);
            }
        }

        /// <summary>
        /// Computes Swish/SiLU activation element-wise for double precision: x * sigmoid(x).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Swish(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;

            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
                output[i] = x * sigmoid;
            }
        }

        /// <summary>
        /// Computes ELU (Exponential Linear Unit) element-wise for double precision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ELU(ReadOnlySpan<double> input, double alpha, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && length >= 4)
            {
                var vzero = Vector256<double>.Zero;
                var valpha = Vector256.Create(alpha);
                var vone = Vector256.Create(1.0);
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var vx = ReadVector256Double(input, i);
                    var mask = Avx.Compare(vx, vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    // Scalar exp for negative values (no AVX exp intrinsic for double)
                    var expResult = Vector256.Create(
                        input[i] <= 0 ? Math.Exp(input[i]) : 0.0,
                        input[i + 1] <= 0 ? Math.Exp(input[i + 1]) : 0.0,
                        input[i + 2] <= 0 ? Math.Exp(input[i + 2]) : 0.0,
                        input[i + 3] <= 0 ? Math.Exp(input[i + 3]) : 0.0);
                    var negPart = Avx.Multiply(valpha, Avx.Subtract(expResult, vone));
                    var result = Avx.BlendVariable(negPart, vx, mask);
                    WriteVector256Double(output, i, result);
                }
            }
#endif

            for (; i < length; i++)
            {
                double x = input[i];
                output[i] = x > 0 ? x : alpha * (Math.Exp(x) - 1.0);
            }
        }

        #endregion


#if NET5_0_OR_GREATER
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> ReadVector256(ReadOnlySpan<float> data, int offset)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector256<float>>(ref Unsafe.As<float, byte>(ref element));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteVector256(Span<float> data, int offset, Vector256<float> value)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref element), value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> ReadVector128(ReadOnlySpan<float> data, int offset)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector128<float>>(ref Unsafe.As<float, byte>(ref element));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteVector128(Span<float> data, int offset, Vector128<float> value)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref element), value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum(Vector256<float> v)
        {
            // SIMD shuffle reduction: no stack spill
            // Step 1: Add upper 128 bits to lower 128 bits
            var lo = v.GetLower();
            var hi = Avx.ExtractVector128(v, 1);
            var sum128 = Sse.Add(lo, hi);
            return HorizontalSum(sum128);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalMax(Vector256<float> v)
        {
            var lo = v.GetLower();
            var hi = Avx.ExtractVector128(v, 1);
            var max128 = Sse.Max(lo, hi);
            // [a, b, c, d] -> shuffle to get [c, d, a, b], then max
            var shuf = Sse.Shuffle(max128, max128, 0b_01_00_11_10); // swap hi/lo pairs
            max128 = Sse.Max(max128, shuf);
            shuf = Sse.Shuffle(max128, max128, 0b_10_11_00_01); // swap adjacent
            max128 = Sse.Max(max128, shuf);
            return max128.ToScalar();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalMin(Vector256<float> v)
        {
            var lo = v.GetLower();
            var hi = Avx.ExtractVector128(v, 1);
            var min128 = Sse.Min(lo, hi);
            var shuf = Sse.Shuffle(min128, min128, 0b_01_00_11_10);
            min128 = Sse.Min(min128, shuf);
            shuf = Sse.Shuffle(min128, min128, 0b_10_11_00_01);
            min128 = Sse.Min(min128, shuf);
            return min128.ToScalar();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum(Vector128<float> v)
        {
#if NET8_0_OR_GREATER
            if (AdvSimd.Arm64.IsSupported)
            {
                // ARM NEON: pairwise add reduction
                var pair = AdvSimd.Arm64.AddPairwise(v, v); // [a+b, c+d, a+b, c+d]
                return pair.ToScalar() + pair.GetElement(1);
            }
#endif
            // SIMD shuffle reduction: [a, b, c, d]
            // movehdup: [b, b, d, d], add with [a, b, c, d] -> [a+b, ?, c+d, ?]
            if (Sse3.IsSupported)
            {
                var shuf = Sse3.MoveHighAndDuplicate(v); // [b, b, d, d]
                var sums = Sse.Add(v, shuf);              // [a+b, ?, c+d, ?]
                var hi = Sse.MoveHighToLow(sums, sums);   // [c+d, ?, ?, ?]
                return Sse.AddScalar(sums, hi).ToScalar(); // a+b+c+d
            }
            if (Sse.IsSupported)
            {
                // SSE fallback
                var shuf2 = Sse.Shuffle(v, v, 0b_10_11_00_01); // [b, a, d, c]
                var sums2 = Sse.Add(v, shuf2);                  // [a+b, a+b, c+d, c+d]
                var hi2 = Sse.MoveHighToLow(sums2, sums2);      // [c+d, c+d, ?, ?]
                return Sse.AddScalar(sums2, hi2).ToScalar();     // a+b+c+d
            }
            // Scalar fallback for platforms without SSE or NEON
            return v.GetElement(0) + v.GetElement(1) + v.GetElement(2) + v.GetElement(3);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<double> ReadVector256Double(ReadOnlySpan<double> data, int offset)
        {
            ref double start = ref MemoryMarshal.GetReference(data);
            ref double element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector256<double>>(ref Unsafe.As<double, byte>(ref element));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteVector256Double(Span<double> data, int offset, Vector256<double> value)
        {
            ref double start = ref MemoryMarshal.GetReference(data);
            ref double element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<double, byte>(ref element), value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<double> ReadVector128Double(ReadOnlySpan<double> data, int offset)
        {
            ref double start = ref MemoryMarshal.GetReference(data);
            ref double element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector128<double>>(ref Unsafe.As<double, byte>(ref element));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteVector128Double(Span<double> data, int offset, Vector128<double> value)
        {
            ref double start = ref MemoryMarshal.GetReference(data);
            ref double element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<double, byte>(ref element), value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double HorizontalSum(Vector256<double> v)
        {
            // SIMD shuffle reduction: add upper 128 to lower 128, then reduce 128-bit
            var lo = v.GetLower();
            var hi = Avx.ExtractVector128(v.AsDouble(), 1);
            var sum128 = Sse2.Add(lo, hi);
            return HorizontalSum(sum128);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double HorizontalSum(Vector128<double> v)
        {
#if NET8_0_OR_GREATER
            if (AdvSimd.Arm64.IsSupported)
            {
                return AdvSimd.Arm64.AddPairwiseScalar(v).ToScalar();
            }
#endif
            if (Sse2.IsSupported)
            {
                // [a, b] -> shuffle to [b, a], add -> [a+b, a+b]
                var hi = Sse2.Shuffle(v, v, 0b_01);
                return Sse2.AddScalar(v, hi).ToScalar();
            }
            // Scalar fallback
            return v.GetElement(0) + v.GetElement(1);
        }
#endif

    #region Type Conversion

    /// <summary>
    /// Converts float span to Half span. Uses unrolled loop for better throughput.
    /// </summary>
    public static void ConvertToHalf(ReadOnlySpan<float> source, Span<Half> destination)
    {
        int i = 0;
        int length = source.Length;

        // 4x unrolled scalar conversion (Half has no native SIMD on most hardware)
        for (; i + 4 <= length; i += 4)
        {
            destination[i] = (Half)source[i];
            destination[i + 1] = (Half)source[i + 1];
            destination[i + 2] = (Half)source[i + 2];
            destination[i + 3] = (Half)source[i + 3];
        }

        for (; i < length; i++)
            destination[i] = (Half)source[i];
    }

    /// <summary>
    /// Converts Half span to float span. Uses unrolled loop for better throughput.
    /// </summary>
    public static void ConvertToSingle(ReadOnlySpan<Half> source, Span<float> destination)
    {
        int i = 0;
        int length = source.Length;

        // 4x unrolled scalar conversion
        for (; i + 4 <= length; i += 4)
        {
            destination[i] = (float)source[i];
            destination[i + 1] = (float)source[i + 1];
            destination[i + 2] = (float)source[i + 2];
            destination[i + 3] = (float)source[i + 3];
        }

        for (; i < length; i++)
            destination[i] = (float)source[i];
    }

    /// <summary>
    /// Converts double span to float span using AVX narrowing conversion.
    /// </summary>
    public static void ConvertDoubleToFloat(ReadOnlySpan<double> source, Span<float> destination)
    {
        int i = 0;
        int length = source.Length;

#if NET8_0_OR_GREATER
        if (Avx.IsSupported)
        {
            ref double srcRef = ref MemoryMarshal.GetReference(source);
            ref float dstRef = ref MemoryMarshal.GetReference(destination);

            // Process 4 doubles -> 4 floats per iteration using VCVTPD2PS
            for (; i + 4 <= length; i += 4)
            {
                var doubleVec = Vector256.LoadUnsafe(ref srcRef, (nuint)i);
                var floatVec = Avx.ConvertToVector128Single(doubleVec);
                floatVec.StoreUnsafe(ref dstRef, (nuint)i);
            }
        }
#endif

        // Scalar tail
        for (; i < length; i++)
            destination[i] = (float)source[i];
    }

    #endregion

    #region Softmax

        /// <summary>
        /// Computes softmax over rows of a 2D float array laid out contiguously.
        /// Each row of length <paramref name="axisSize"/> is processed independently.
        /// Uses unsafe pointers for maximum throughput.
        /// </summary>
        public static unsafe void Softmax(ReadOnlySpan<float> input, Span<float> output, int outerSize, int axisSize)
        {
            fixed (float* pIn = input)
            fixed (float* pOut = output)
            {
                for (int row = 0; row < outerSize; row++)
                {
                    float* rowIn = pIn + row * axisSize;
                    float* rowOut = pOut + row * axisSize;
                    SoftmaxRowUnsafe(rowIn, rowOut, axisSize);
                }
            }
        }

        /// <summary>
        /// Computes softmax for a single contiguous row using unsafe pointers.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void SoftmaxRowUnsafe(float* input, float* output, int length)
        {
            int i = 0;

            // Step 1: Find max for numerical stability
            float maxVal = float.NegativeInfinity;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vmax0 = Vector256.Create(float.NegativeInfinity);
                var vmax1 = vmax0;
                var vmax2 = vmax0;
                var vmax3 = vmax0;
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    vmax0 = Avx.Max(vmax0, Avx.LoadVector256(input + i));
                    vmax1 = Avx.Max(vmax1, Avx.LoadVector256(input + i + 8));
                    vmax2 = Avx.Max(vmax2, Avx.LoadVector256(input + i + 16));
                    vmax3 = Avx.Max(vmax3, Avx.LoadVector256(input + i + 24));
                }
                vmax0 = Avx.Max(Avx.Max(vmax0, vmax1), Avx.Max(vmax2, vmax3));
                maxVal = HorizontalMax(vmax0);
            }
#endif
            for (; i < length; i++)
            {
                if (input[i] > maxVal) maxVal = input[i];
            }

            // Fused Step 2+3+4: subtract max, exp, and accumulate sum in ONE pass
            // Halves memory traffic vs 3 separate passes (sub, exp, sum)
            float sumExp = 0f;
            i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                var vmaxBcast = Vector256.Create(maxVal);
                var vsum0 = Vector256<float>.Zero;
                var vsum1 = Vector256<float>.Zero;
                var vsum2 = Vector256<float>.Zero;
                var vsum3 = Vector256<float>.Zero;
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    var e0 = FastExp256(Avx.Subtract(Avx.LoadVector256(input + i), vmaxBcast));
                    var e1 = FastExp256(Avx.Subtract(Avx.LoadVector256(input + i + 8), vmaxBcast));
                    var e2 = FastExp256(Avx.Subtract(Avx.LoadVector256(input + i + 16), vmaxBcast));
                    var e3 = FastExp256(Avx.Subtract(Avx.LoadVector256(input + i + 24), vmaxBcast));
                    Avx.Store(output + i, e0);
                    Avx.Store(output + i + 8, e1);
                    Avx.Store(output + i + 16, e2);
                    Avx.Store(output + i + 24, e3);
                    vsum0 = Avx.Add(vsum0, e0);
                    vsum1 = Avx.Add(vsum1, e1);
                    vsum2 = Avx.Add(vsum2, e2);
                    vsum3 = Avx.Add(vsum3, e3);
                }
                vsum0 = Avx.Add(Avx.Add(vsum0, vsum1), Avx.Add(vsum2, vsum3));
                sumExp = HorizontalSum(vsum0);
            }
#endif
            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                float e = MathF.Exp(input[i] - maxVal);
#else
                float e = (float)Math.Exp(input[i] - maxVal);
#endif
                output[i] = e;
                sumExp += e;
            }

            // Step 5: Divide by sum
            if (sumExp == 0f) return;
            float invSum = 1f / sumExp;
            i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vInvSum = Vector256.Create(invSum);
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    Avx.Store(output + i, Avx.Multiply(Avx.LoadVector256(output + i), vInvSum));
                    Avx.Store(output + i + 8, Avx.Multiply(Avx.LoadVector256(output + i + 8), vInvSum));
                    Avx.Store(output + i + 16, Avx.Multiply(Avx.LoadVector256(output + i + 16), vInvSum));
                    Avx.Store(output + i + 24, Avx.Multiply(Avx.LoadVector256(output + i + 24), vInvSum));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] *= invSum;
            }
        }

        /// <summary>
        /// Computes log_softmax for a single contiguous row using unsafe pointers.
        /// log_softmax(x) = (x - max) - log(sum(exp(x - max)))
        /// Uses inline FastExp256 to avoid per-call overhead.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void LogSoftmaxRowUnsafe(float* input, float* output, int length)
        {
            int i = 0;

            // Step 1: Find max for numerical stability
            float maxVal = float.NegativeInfinity;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vmax0 = Vector256.Create(float.NegativeInfinity);
                var vmax1 = vmax0;
                var vmax2 = vmax0;
                var vmax3 = vmax0;
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    vmax0 = Avx.Max(vmax0, Avx.LoadVector256(input + i));
                    vmax1 = Avx.Max(vmax1, Avx.LoadVector256(input + i + 8));
                    vmax2 = Avx.Max(vmax2, Avx.LoadVector256(input + i + 16));
                    vmax3 = Avx.Max(vmax3, Avx.LoadVector256(input + i + 24));
                }
                vmax0 = Avx.Max(Avx.Max(vmax0, vmax1), Avx.Max(vmax2, vmax3));
                maxVal = HorizontalMax(vmax0);
            }
#endif
            for (; i < length; i++)
            {
                if (input[i] > maxVal) maxVal = input[i];
            }

            // Fused Step 2+3: subtract max, store shifted values, and accumulate sum(exp) in ONE pass
            float sumExp = 0f;
            i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                var vmaxBcast = Vector256.Create(maxVal);
                var vsum0 = Vector256<float>.Zero;
                var vsum1 = Vector256<float>.Zero;
                var vsum2 = Vector256<float>.Zero;
                var vsum3 = Vector256<float>.Zero;
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    var s0 = Avx.Subtract(Avx.LoadVector256(input + i), vmaxBcast);
                    var s1 = Avx.Subtract(Avx.LoadVector256(input + i + 8), vmaxBcast);
                    var s2 = Avx.Subtract(Avx.LoadVector256(input + i + 16), vmaxBcast);
                    var s3 = Avx.Subtract(Avx.LoadVector256(input + i + 24), vmaxBcast);
                    Avx.Store(output + i, s0);
                    Avx.Store(output + i + 8, s1);
                    Avx.Store(output + i + 16, s2);
                    Avx.Store(output + i + 24, s3);
                    vsum0 = Avx.Add(vsum0, FastExp256(s0));
                    vsum1 = Avx.Add(vsum1, FastExp256(s1));
                    vsum2 = Avx.Add(vsum2, FastExp256(s2));
                    vsum3 = Avx.Add(vsum3, FastExp256(s3));
                }
                vsum0 = Avx.Add(Avx.Add(vsum0, vsum1), Avx.Add(vsum2, vsum3));
                sumExp = HorizontalSum(vsum0);
            }
#endif
            for (; i < length; i++)
            {
                float shifted = input[i] - maxVal;
                output[i] = shifted;
#if NET5_0_OR_GREATER
                sumExp += MathF.Exp(shifted);
#else
                sumExp += (float)Math.Exp(shifted);
#endif
            }

            // Step 4: output[i] = shifted[i] - log(sumExp)
            float logSumExp = 0f;
#if NET5_0_OR_GREATER
            logSumExp = MathF.Log(sumExp);
#else
            logSumExp = (float)Math.Log(sumExp);
#endif
            i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vLogSum = Vector256.Create(logSumExp);
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    Avx.Store(output + i, Avx.Subtract(Avx.LoadVector256(output + i), vLogSum));
                    Avx.Store(output + i + 8, Avx.Subtract(Avx.LoadVector256(output + i + 8), vLogSum));
                    Avx.Store(output + i + 16, Avx.Subtract(Avx.LoadVector256(output + i + 16), vLogSum));
                    Avx.Store(output + i + 24, Avx.Subtract(Avx.LoadVector256(output + i + 24), vLogSum));
                }
            }
#endif
            for (; i < length; i++)
            {
                output[i] -= logSumExp;
            }
        }

    #endregion
    }
}
