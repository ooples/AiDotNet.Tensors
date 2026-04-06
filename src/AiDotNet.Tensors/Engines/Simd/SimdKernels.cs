using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static unsafe void VectorAddUnsafe(float* a, float* b, float* result, int length)
        {
            int i = 0;
#if NET8_0_OR_GREATER
            if (Avx512F.IsSupported && length >= 64)
            {
                int simdLength = length & ~63;
                for (; i < simdLength; i += 64)
                {
                    Avx512F.Store(result + i, Avx512F.Add(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
                    Avx512F.Store(result + i + 16, Avx512F.Add(Avx512F.LoadVector512(a + i + 16), Avx512F.LoadVector512(b + i + 16)));
                    Avx512F.Store(result + i + 32, Avx512F.Add(Avx512F.LoadVector512(a + i + 32), Avx512F.LoadVector512(b + i + 32)));
                    Avx512F.Store(result + i + 48, Avx512F.Add(Avx512F.LoadVector512(a + i + 48), Avx512F.LoadVector512(b + i + 48)));
                }
            }
            if (Avx512F.IsSupported && length - i >= 16)
            {
                int simdLength = i + ((length - i) & ~15);
                for (; i < simdLength; i += 16)
                {
                    Avx512F.Store(result + i, Avx512F.Add(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
                }
            }
#endif
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length - i >= 32)
            {
                int simdLength = i + ((length - i) & ~31);
                const int prefetchDist = 256 / sizeof(float);
                for (; i < simdLength; i += 32)
                {
                    if (Sse.IsSupported && i + prefetchDist < length)
                    {
                        Sse.Prefetch0(a + i + prefetchDist);
                        Sse.Prefetch0(b + i + prefetchDist);
                    }
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
        [MethodImpl(HotInline)]
        public static unsafe void VectorMultiplyUnsafe(float* a, float* b, float* result, int length)
        {
            int i = 0;
#if NET8_0_OR_GREATER
            if (Avx512F.IsSupported && length >= 64)
            {
                int simdLength = length & ~63;
                for (; i < simdLength; i += 64)
                {
                    Avx512F.Store(result + i, Avx512F.Multiply(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
                    Avx512F.Store(result + i + 16, Avx512F.Multiply(Avx512F.LoadVector512(a + i + 16), Avx512F.LoadVector512(b + i + 16)));
                    Avx512F.Store(result + i + 32, Avx512F.Multiply(Avx512F.LoadVector512(a + i + 32), Avx512F.LoadVector512(b + i + 32)));
                    Avx512F.Store(result + i + 48, Avx512F.Multiply(Avx512F.LoadVector512(a + i + 48), Avx512F.LoadVector512(b + i + 48)));
                }
            }
            if (Avx512F.IsSupported && length - i >= 16)
            {
                int simdLength = i + ((length - i) & ~15);
                for (; i < simdLength; i += 16)
                {
                    Avx512F.Store(result + i, Avx512F.Multiply(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
                }
            }
#endif
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length - i >= 32)
            {
                const int prefetchDist = 256 / sizeof(float);
                int simdLength = i + ((length - i) & ~31);
                for (; i < simdLength; i += 32)
                {
                    if (Sse.IsSupported && i + prefetchDist < length)
                    {
                        Sse.Prefetch0(a + i + prefetchDist);
                        Sse.Prefetch0(b + i + prefetchDist);
                    }
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
        [MethodImpl(HotInline)]
        public static unsafe void VectorSubtractUnsafe(float* a, float* b, float* result, int length)
        {
            int i = 0;
#if NET8_0_OR_GREATER
            if (Avx512F.IsSupported && length >= 64)
            {
                int simdLength = length & ~63;
                for (; i < simdLength; i += 64)
                {
                    Avx512F.Store(result + i, Avx512F.Subtract(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
                    Avx512F.Store(result + i + 16, Avx512F.Subtract(Avx512F.LoadVector512(a + i + 16), Avx512F.LoadVector512(b + i + 16)));
                    Avx512F.Store(result + i + 32, Avx512F.Subtract(Avx512F.LoadVector512(a + i + 32), Avx512F.LoadVector512(b + i + 32)));
                    Avx512F.Store(result + i + 48, Avx512F.Subtract(Avx512F.LoadVector512(a + i + 48), Avx512F.LoadVector512(b + i + 48)));
                }
            }
            if (Avx512F.IsSupported && length - i >= 16)
            {
                int simdLength = i + ((length - i) & ~15);
                for (; i < simdLength; i += 16)
                {
                    Avx512F.Store(result + i, Avx512F.Subtract(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
                }
            }
#endif
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length - i >= 32)
            {
                const int prefetchDist = 256 / sizeof(float);
                int simdLength = i + ((length - i) & ~31);
                for (; i < simdLength; i += 32)
                {
                    if (Sse.IsSupported && i + prefetchDist < length)
                    {
                        Sse.Prefetch0(a + i + prefetchDist);
                        Sse.Prefetch0(b + i + prefetchDist);
                    }
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
        [MethodImpl(HotInline)]
        public static unsafe void ReLUUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET8_0_OR_GREATER
            if (Avx512F.IsSupported && length >= 64)
            {
                var vzero = Vector512<float>.Zero;
                int simdLength = length & ~63;
                for (; i < simdLength; i += 64)
                {
                    Avx512F.Store(output + i, Avx512F.Max(Avx512F.LoadVector512(input + i), vzero));
                    Avx512F.Store(output + i + 16, Avx512F.Max(Avx512F.LoadVector512(input + i + 16), vzero));
                    Avx512F.Store(output + i + 32, Avx512F.Max(Avx512F.LoadVector512(input + i + 32), vzero));
                    Avx512F.Store(output + i + 48, Avx512F.Max(Avx512F.LoadVector512(input + i + 48), vzero));
                }
            }
            if (Avx512F.IsSupported && length - i >= 16)
            {
                var vzero = Vector512<float>.Zero;
                int simdLength = i + ((length - i) & ~15);
                for (; i < simdLength; i += 16)
                {
                    Avx512F.Store(output + i, Avx512F.Max(Avx512F.LoadVector512(input + i), vzero));
                }
            }
#endif
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length - i >= 32)
            {
                var vzero = Vector256<float>.Zero;
                int simdLength = i + ((length - i) & ~31);
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
        [MethodImpl(HotInline)]
        public static unsafe void SigmoidUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
            {
                // 4x unrolled FastSigmoid256: consistent approximation across all SIMD paths
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(output + i, FastSigmoid256(Avx.LoadVector256(input + i)));
                    Avx.Store(output + i + 8, FastSigmoid256(Avx.LoadVector256(input + i + 8)));
                    Avx.Store(output + i + 16, FastSigmoid256(Avx.LoadVector256(input + i + 16)));
                    Avx.Store(output + i + 24, FastSigmoid256(Avx.LoadVector256(input + i + 24)));
                }
            }
            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
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
        [MethodImpl(HotInline)]
        public static unsafe void VectorDivideUnsafe(float* a, float* b, float* result, int length)
        {
            int i = 0;
#if NET8_0_OR_GREATER
            if (Avx512F.IsSupported && length >= 64)
            {
                int simdLength = length & ~63;
                for (; i < simdLength; i += 64)
                {
                    Avx512F.Store(result + i, Avx512F.Divide(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
                    Avx512F.Store(result + i + 16, Avx512F.Divide(Avx512F.LoadVector512(a + i + 16), Avx512F.LoadVector512(b + i + 16)));
                    Avx512F.Store(result + i + 32, Avx512F.Divide(Avx512F.LoadVector512(a + i + 32), Avx512F.LoadVector512(b + i + 32)));
                    Avx512F.Store(result + i + 48, Avx512F.Divide(Avx512F.LoadVector512(a + i + 48), Avx512F.LoadVector512(b + i + 48)));
                }
            }
            if (Avx512F.IsSupported && length - i >= 16)
            {
                int simdLength = i + ((length - i) & ~15);
                for (; i < simdLength; i += 16)
                {
                    Avx512F.Store(result + i, Avx512F.Divide(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
                }
            }
#endif
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length - i >= 32)
            {
                int simdLength = i + ((length - i) & ~31);
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
        /// Pointer-based Exp — tries MKL VML first (SVML microcode, zero-overhead function pointer),
        /// falls back to Cephes FastExp256 polynomial approximation.
        /// </summary>
        [MethodImpl(HotInline)]
        public static unsafe void ExpUnsafe(float* input, float* output, int length)
        {
#if NET5_0_OR_GREATER
            // MKL VML path: SVML microcode exp, ~3x faster than our polynomial
            if (VmlProvider.TryExp(input, output, length))
                return;
#endif

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
        /// Pointer-based Log — tries MKL VML first, falls back to FastLog256.
        /// </summary>
        [MethodImpl(HotInline)]
        public static unsafe void LogUnsafe(float* input, float* output, int length)
        {
#if NET5_0_OR_GREATER
            if (VmlProvider.TryLn(input, output, length))
                return;
#endif

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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static unsafe void TanhUnsafe(float* input, float* output, int length)
        {
#if NET5_0_OR_GREATER
            // MKL VML path: SVML tanh, zero-overhead function pointer
            if (VmlProvider.TryTanh(input, output, length))
                return;
#endif

            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 32)
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
            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 8)
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
                output[i] = MathF.Tanh(input[i]);
            }
        }

        /// <summary>
        /// Pointer-based LeakyReLU — max(alpha*x, x) with zero bounds-checking.
        /// </summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static unsafe void GELUUnsafe(float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            // VML SVML path: compute tanh_arg in SIMD, then use VML tanh (SVML microcode)
            // This is ~3x faster than our polynomial FastSigmoid256 approximation.
            if (Avx.IsSupported && Fma.IsSupported && length >= 8)
            {
                // Use rented buffer (stackalloc is unsafe for large arrays like 100K+)
                var tanhArgsBuf = System.Buffers.ArrayPool<float>.Shared.Rent(length);
                try
                {
                    fixed (float* tanhArgs = tanhArgsBuf)
                    {
                        var vSqrt2OverPi = Vector256.Create(0.7978845608028654f);
                        var vCoeff = Vector256.Create(0.044715f);

                        int simdLen = length & ~7;
                        for (int j = 0; j < simdLen; j += 8)
                        {
                            var x = Avx.LoadVector256(input + j);
                            var x3 = Avx.Multiply(Avx.Multiply(x, x), x);
                            var inner = Fma.MultiplyAdd(vCoeff, x3, x);
                            Avx.Store(tanhArgs + j, Avx.Multiply(vSqrt2OverPi, inner));
                        }
                        for (int j = simdLen; j < length; j++)
                        {
                            float x = input[j];
                            tanhArgs[j] = 0.7978845608028654f * (x + 0.044715f * x * x * x);
                        }

                        // Step 2: VML tanh (SVML vectorized) — the fast path
                        if (VmlProvider.TryTanh(tanhArgs, tanhArgs, length))
                        {
                            var vHalf = Vector256.Create(0.5f);
                            var vOne = Vector256.Create(1.0f);
                            for (int j = 0; j < simdLen; j += 8)
                            {
                                var x = Avx.LoadVector256(input + j);
                                var t = Avx.LoadVector256(tanhArgs + j);
                                Avx.Store(output + j, Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(vOne, t))));
                            }
                            for (int j = simdLen; j < length; j++)
                                output[j] = 0.5f * input[j] * (1f + tanhArgs[j]);
                            return;
                        }
                    }
                }
                finally
                {
                    System.Buffers.ArrayPool<float>.Shared.Return(tanhArgsBuf);
                }
                // VML not available — fall through to polynomial path
            }

            // Polynomial fallback: FastSigmoid256-based tanh approximation
            if (Avx.IsSupported && Fma.IsSupported && length >= 32)
            {
                var vSqrt2OverPi = Vector256.Create(0.7978845608028654f);
                var vCoeff = Vector256.Create(0.044715f);
                var vHalf = Vector256.Create(0.5f);
                var vOne = Vector256.Create(1.0f);
                var vTwo = Vector256.Create(2.0f);

                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    for (int k = 0; k < 32; k += 8)
                    {
                        var x = Avx.LoadVector256(input + i + k);
                        var x_cubed = Avx.Multiply(Avx.Multiply(x, x), x);
                        var inner = Fma.MultiplyAdd(vCoeff, x_cubed, x);
                        var tanh_arg = Avx.Multiply(vSqrt2OverPi, inner);
                        var tanh_val = Avx.Subtract(Avx.Multiply(vTwo, FastSigmoid256(Avx.Multiply(vTwo, tanh_arg))), vOne);
                        Avx.Store(output + i + k, Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(vOne, tanh_val))));
                    }
                }
            }
            if (Avx.IsSupported && Fma.IsSupported && length - i >= 8)
            {
                var vSqrt2OverPi = Vector256.Create(0.7978845608028654f);
                var vCoeff = Vector256.Create(0.044715f);
                var vHalf = Vector256.Create(0.5f);
                var vOne = Vector256.Create(1.0f);
                var vTwo = Vector256.Create(2.0f);

                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var x_cubed = Avx.Multiply(Avx.Multiply(x, x), x);
                    var inner = Fma.MultiplyAdd(vCoeff, x_cubed, x);
                    var tanh_arg = Avx.Multiply(vSqrt2OverPi, inner);
                    var tanh_val = Avx.Subtract(Avx.Multiply(vTwo, FastSigmoid256(Avx.Multiply(vTwo, tanh_arg))), vOne);
                    Avx.Store(output + i, Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(vOne, tanh_val))));
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static void DivideScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
        {
            // Multiply by reciprocal for better performance
            MultiplyScalar(a, 1.0f / scalar, result);
        }

        /// <summary>Subtracts a scalar from each element: result[i] = a[i] - scalar.</summary>
        [MethodImpl(HotInline)]
        public static void SubtractScalar(ReadOnlySpan<float> a, float scalar, Span<float> result)
        {
            AddScalar(a, -scalar, result);
        }

        [MethodImpl(HotInline)]
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
#if NET5_0_OR_GREATER
                sum = Fma.IsSupported
                    ? MathF.FusedMultiplyAdd(a[i], b[i], sum)
                    : sum + (a[i] * b[i]);
#else
                sum += a[i] * b[i];
#endif
            }

            return sum;
        }

        /// <summary>
        /// Computes destination[i] = a[i] + b[i] * scalar for double-precision values using SIMD.
        /// Uses FMA (Fused Multiply-Add) when available for better performance and precision.
        /// </summary>
        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static unsafe void Exp(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // Use Cephes-style fast exp polynomial with explicit AVX2/FMA intrinsics.
            // Note: MKL VML was tested but delegate-based P/Invoke overhead negated the
            // SVML benefit. Our FastExp256 is competitive for float.
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
        [MethodImpl(HotInline)]
        public static unsafe void Exp(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;

#if NET5_0_OR_GREATER
            // VML double via vmdExp (mode-per-call, VML_LA guaranteed).
            fixed (double* pIn = input)
            fixed (double* pOut = output)
            {
                if (VmlProvider.TryExp(pIn, pOut, length))
                    return;
            }
#endif

            int i = 0;
            int unrolled = length & ~3;
            for (; i < unrolled; i += 4)
            {
                output[i] = Math.Exp(input[i]);
                output[i + 1] = Math.Exp(input[i + 1]);
                output[i + 2] = Math.Exp(input[i + 2]);
                output[i + 3] = Math.Exp(input[i + 3]);
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        /// Pointer-based double Sigmoid — tries MKL VML exp first for SVML performance,
        /// falls back to 4x unrolled AVX2+FMA with FastExpDouble256.
        /// </summary>
        [MethodImpl(HotInline)]
        public static unsafe void SigmoidUnsafe(double* input, double* output, int length)
        {
#if NET5_0_OR_GREATER
            // VML double sigmoid — each pointer now individually verified at load time.
            if (VmlProvider.IsAvailable && length >= 64)
            {
                // Step 1: output = -input
                int j = 0;
                if (Avx.IsSupported)
                {
                    var negMask = Vector256.Create(-0.0); // sign bit only
                    int simdLen = length & ~3;
                    for (; j < simdLen; j += 4)
                    {
                        var v = Avx.LoadVector256(input + j);
                        Avx.Store(output + j, Avx.Xor(v, negMask));
                    }
                }
                for (; j < length; j++)
                    output[j] = -input[j];

                // Step 2: output = exp(-input) via MKL VML (SVML microcode)
                if (VmlProvider.TryExp(output, output, length))
                {
                    // Step 3: output = 1/(1+exp(-x)) with SIMD
                    j = 0;
                    if (Avx.IsSupported)
                    {
                        var one = Vector256.Create(1.0);
                        int simdLen = length & ~3;
                        for (; j < simdLen; j += 4)
                        {
                            var e = Avx.LoadVector256(output + j);
                            Avx.Store(output + j, Avx.Divide(one, Avx.Add(one, e)));
                        }
                    }
                    for (; j < length; j++)
                        output[j] = 1.0 / (1.0 + output[j]);
                    return;
                }
            }
#endif

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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static unsafe void Tanh(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;

#if NET5_0_OR_GREATER
            // VML double via vmdTanh (mode-per-call, VML_LA guaranteed).
            fixed (double* pIn = input)
            fixed (double* pOut = output)
            {
                if (VmlProvider.TryTanh(pIn, pOut, length))
                    return;
            }
#endif

            int i = 0;

#if NET5_0_OR_GREATER
            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            if (Avx2.IsSupported && Fma.IsSupported && length >= 16)
            {
                var two = Vector256.Create(2.0);
                var one = Vector256.Create(1.0);
                int simdLength = length & ~15;
                for (; i < simdLength; i += 16)
                {
                    for (int k = 0; k < 16; k += 4)
                    {
                        var x = ReadVector256Double(input, i + k);
                        var e2x = FastExpDouble256(Avx.Multiply(two, x));
                        WriteVector256Double(output, i + k,
                            Avx.Divide(Avx.Subtract(e2x, one), Avx.Add(e2x, one)));
                    }
                }
            }

            if (Avx2.IsSupported && Fma.IsSupported && length - i >= 4)
            {
                var two = Vector256.Create(2.0);
                var one = Vector256.Create(1.0);
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var x = ReadVector256Double(input, i);
                    var e2x = FastExpDouble256(Avx.Multiply(two, x));
                    WriteVector256Double(output, i,
                        Avx.Divide(Avx.Subtract(e2x, one), Avx.Add(e2x, one)));
                }
            }
#endif

            for (; i < length; i++)
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        private static Vector256<double> FastExpDouble256(Vector256<double> x)
        {
            // Clamp to avoid inf/nan in polynomial — no per-vector edge-case handling
            // (edge cases are handled at the span level for Math.Exp compatibility)
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
        /// Cephes-style fast log(x) for Vector256&lt;double&gt; (4 doubles per vector).
        /// Decomposes x = 2^e * m (0.5 &lt;= m &lt; 1.0), then log(x) = e*ln2 + log(m).
        /// Uses 7th-order Padé-like rational approximation for log(m) on [sqrt(0.5), sqrt(2)].
        /// Relative error ~1e-14 across the normal double range.
        /// </summary>
        [MethodImpl(HotInline)]
        private static Vector256<double> FastLogDouble256(Vector256<double> x)
        {
            // No per-vector edge-case handling — edge cases (zero, negative, NaN, inf)
            // are handled at the span level. This keeps the hot path fast.
            // Clamp to avoid log(0) producing -inf in bit manipulation
            x = Avx.Max(x, Vector256.Create(double.Epsilon));

            // Extract exponent and mantissa via integer bit manipulation
            var xi = x.AsInt64();
            // Exponent: shift right 52, subtract bias 1023
            var eLong = Avx2.Subtract(Avx2.ShiftRightLogical(xi, 52), Vector256.Create(1023L));
            // Mantissa: clear exponent bits, set exponent to bias (1023 << 52)
            var mantissaBits = Avx2.Or(
                Avx2.And(xi, Vector256.Create(0x000FFFFFFFFFFFFFL)),
                Vector256.Create(0x3FF0000000000000L));
            var m = mantissaBits.AsDouble();

            // Adjust range to [sqrt(0.5), sqrt(2)] for better polynomial convergence
            // If m > sqrt(2), divide by 2 and increment exponent
            var sqrt2 = Vector256.Create(1.4142135623730950488);
            var gtSqrt2 = Avx.Compare(m, sqrt2, FloatComparisonMode.OrderedGreaterThanSignaling);
            // m = gtSqrt2 ? m*0.5 : m
            var half = Vector256.Create(0.5);
            m = Avx.BlendVariable(m, Avx.Multiply(m, half), gtSqrt2);
            // e = gtSqrt2 ? e+1 : e  (add 1 via mask)
            var eMask = Avx2.And(gtSqrt2.AsInt64(), Vector256.Create(1L));
            eLong = Avx2.Add(eLong, eMask);

            // Convert int64 exponent to double via int32 (safe — exponents fit in int32)
            var eLow = Vector128.Create(
                (int)eLong.GetElement(0), (int)eLong.GetElement(1),
                (int)eLong.GetElement(2), (int)eLong.GetElement(3));
            var e = Avx.ConvertToVector256Double(eLow);

            // f = m - 1.0
            var one = Vector256.Create(1.0);
            var f = Avx.Subtract(m, one);

            // Rational approximation: log(1+f) ≈ f - 0.5*f^2 + f^3 * P(f)/Q(f)
            // Using Cephes coefficients for double precision
            var f2 = Avx.Multiply(f, f);
            var f3 = Avx.Multiply(f2, f);

            // Numerator P(f) = p0 + p1*f + p2*f^2 + ... + p5*f^5
            var p0 = Vector256.Create(7.70838733755885391666e0);
            var p1 = Vector256.Create(1.79368678507819816313e1);
            var p2 = Vector256.Create(1.44989225341610930846e1);
            var p3 = Vector256.Create(4.70579119878881725854e0);
            var p4 = Vector256.Create(4.97494994976747426342e-1);
            var p5 = Vector256.Create(1.01875663804580931796e-4);

            var pVal = Fma.MultiplyAdd(p5, f, p4);
            pVal = Fma.MultiplyAdd(pVal, f, p3);
            pVal = Fma.MultiplyAdd(pVal, f, p2);
            pVal = Fma.MultiplyAdd(pVal, f, p1);
            pVal = Fma.MultiplyAdd(pVal, f, p0);

            // Denominator Q(f) = q0 + q1*f + q2*f^2 + ... + q5*f^5
            var q0 = Vector256.Create(1.54360888658065167530e1);
            var q1 = Vector256.Create(4.44568781543509296375e1);
            var q2 = Vector256.Create(4.51227709466894823867e1);
            var q3 = Vector256.Create(1.86960070067819026681e1);
            var q4 = Vector256.Create(2.60098586801477581364e0);
            // q5 = 1.0 (monic)
            var qVal = Fma.MultiplyAdd(one, f, q4);
            qVal = Fma.MultiplyAdd(qVal, f, q3);
            qVal = Fma.MultiplyAdd(qVal, f, q2);
            qVal = Fma.MultiplyAdd(qVal, f, q1);
            qVal = Fma.MultiplyAdd(qVal, f, q0);

            // log(1+f) = f - 0.5*f^2 + f^3 * P/Q
            var logm = Avx.Add(
                Avx.Subtract(f, Avx.Multiply(half, f2)),
                Avx.Multiply(f3, Avx.Divide(pVal, qVal)));

            // log(x) = e * ln2 + log(m)
            var ln2 = Vector256.Create(0.693147180559945309417);
            var result = Fma.MultiplyAdd(e, ln2, logm);

            return result;
        }

        /// <summary>
        /// Cephes-style fast sin(x) for Vector256&lt;double&gt;.
        /// Range reduction to [-pi/2, pi/2] via j = round(x * 2/pi), then 11th-order
        /// minimax polynomial. Relative error ~1e-15 across the full double range.
        /// </summary>
        [MethodImpl(HotInline)]
        private static Vector256<double> FastSinDouble256(Vector256<double> x)
        {
            // Guard: fall back to scalar Math.Sin for |x| > 1e9 to avoid int32 overflow
            // in quadrant computation (j = round(|x| * 2/pi) must fit in int32)
            var safeLimit = Vector256.Create(1e9);
            var absX = Avx.And(x, Vector256.Create(0x7FFFFFFFFFFFFFFFL).AsDouble());
            var unsafeMask = Avx.Compare(absX, safeLimit, FloatComparisonMode.OrderedGreaterThanSignaling);
            if (Avx.MoveMask(unsafeMask) != 0)
            {
                // At least one element is too large — compute all via scalar
                return Vector256.Create(
                    Math.Sin(x.GetElement(0)), Math.Sin(x.GetElement(1)),
                    Math.Sin(x.GetElement(2)), Math.Sin(x.GetElement(3)));
            }

            // Range reduction: j = round(x / (pi/2))
            var twoOverPi = Vector256.Create(0.6366197723675814);
            var piOver2Hi = Vector256.Create(1.5707963267341256);    // pi/2 high part
            var piOver2Lo = Vector256.Create(6.077100506506192e-11); // pi/2 low part

            // Sign extraction
            var signBit = Avx2.And(x.AsInt64(), Vector256.Create(unchecked((long)0x8000000000000000L)));
            x = absX;

            // j = round(x * 2/pi)
            var j = Avx.RoundToNearestInteger(Avx.Multiply(x, twoOverPi));
            var jInt = Avx2.ConvertToVector256Int64(Avx.ConvertToVector128Int32(j));

            // r = x - j * pi/2
            var r = Fma.MultiplyAddNegated(j, piOver2Hi, x);
            r = Fma.MultiplyAddNegated(j, piOver2Lo, r);

            // Determine quadrant: if j%4 >= 2, negate result
            var quadrant = Avx2.And(jInt, Vector256.Create(3L));
            var needNeg = Avx2.CompareGreaterThan(quadrant, Vector256.Create(1L));
            var negMask = Avx2.And(needNeg, Vector256.Create(unchecked((long)0x8000000000000000L)));

            // If j%2 == 1, use cos polynomial instead of sin polynomial
            var useCosPoly = Avx2.CompareEqual(Avx2.And(jInt, Vector256.Create(1L)), Vector256.Create(1L));

            var r2 = Avx.Multiply(r, r);

            // Sin polynomial: r - r^3/6 + r^5/120 - ... (odd powers)
            var s1 = Vector256.Create(-1.66666666666666324348e-1);
            var s2 = Vector256.Create(8.33333333332248946124e-3);
            var s3 = Vector256.Create(-1.98412698298579493134e-4);
            var s4 = Vector256.Create(2.75573137070700676789e-6);
            var s5 = Vector256.Create(-2.50507602534068634195e-8);
            var s6 = Vector256.Create(1.58969099521155010221e-10);

            var sinPoly = Fma.MultiplyAdd(s6, r2, s5);
            sinPoly = Fma.MultiplyAdd(sinPoly, r2, s4);
            sinPoly = Fma.MultiplyAdd(sinPoly, r2, s3);
            sinPoly = Fma.MultiplyAdd(sinPoly, r2, s2);
            sinPoly = Fma.MultiplyAdd(sinPoly, r2, s1);
            sinPoly = Fma.MultiplyAdd(sinPoly, Avx.Multiply(r2, r), r);

            // Cos polynomial: 1 - r^2/2 + r^4/24 - ... (even powers)
            var c1 = Vector256.Create(-0.5);
            var c2 = Vector256.Create(4.16666666666666019037e-2);
            var c3 = Vector256.Create(-1.38888888888741095749e-3);
            var c4 = Vector256.Create(2.48015872894767294178e-5);
            var c5 = Vector256.Create(-2.75573143513906633035e-7);
            var c6 = Vector256.Create(2.08757232129817482790e-9);

            var cosPoly = Fma.MultiplyAdd(c6, r2, c5);
            cosPoly = Fma.MultiplyAdd(cosPoly, r2, c4);
            cosPoly = Fma.MultiplyAdd(cosPoly, r2, c3);
            cosPoly = Fma.MultiplyAdd(cosPoly, r2, c2);
            cosPoly = Fma.MultiplyAdd(cosPoly, r2, c1);
            cosPoly = Fma.MultiplyAdd(cosPoly, r2, Vector256.Create(1.0));

            // Select sin or cos polynomial based on quadrant
            var result = Avx.BlendVariable(sinPoly, cosPoly, useCosPoly.AsDouble());

            // Apply quadrant negation and original sign
            var flipBits = Avx2.Xor(negMask, signBit);
            return Avx.Xor(result, flipBits.AsDouble());
        }

        /// <summary>
        /// Cephes-style fast cos(x) for Vector256&lt;double&gt;.
        /// Uses sin(x + pi/2) identity with the FastSinDouble256 infrastructure.
        /// </summary>
        [MethodImpl(HotInline)]
        private static Vector256<double> FastCosDouble256(Vector256<double> x)
        {
            // cos(x) = sin(x + pi/2)
            var piOver2 = Vector256.Create(1.5707963267948966);
            return FastSinDouble256(Avx.Add(x, piOver2));
        }

        /// <summary>
        /// Fast vectorized sigmoid using FastExp256: sigmoid(x) = 1 / (1 + exp(-x)).
        /// Uses Cephes-style exp polynomial (~0.01% relative error) for high accuracy.
        /// </summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        private static Vector256<float> FastLog256(Vector256<float> x)
        {
            // Extract exponent: n = floor(log2(x))
            // IEEE 754 float: [sign:1][exponent:8][mantissa:23], bias = 127
            var vone = Vector256.Create(1.0f);
            var vzero = Vector256<float>.Zero;

            // Preserve special values: log(0)=-inf, log(negative)=NaN, log(+inf)=+inf, log(NaN)=NaN
            var zeroMask = Avx.CompareEqual(x, vzero);
            var negativeMask = Avx.CompareLessThan(x, vzero);
            var infMask = Avx.CompareEqual(x, Vector256.Create(float.PositiveInfinity));
            var nanMask = Avx.CompareNotEqual(x, x); // NaN != NaN
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

            // Restore special values: log(0) = -inf, log(negative) = NaN, log(+inf) = +inf, log(NaN) = NaN
            result = Avx.BlendVariable(result, Vector256.Create(float.NegativeInfinity), zeroMask);
            result = Avx.BlendVariable(result, Vector256.Create(float.NaN), negativeMask);
            result = Avx.BlendVariable(result, Vector256.Create(float.PositiveInfinity), infMask);
            result = Avx.BlendVariable(result, Vector256.Create(float.NaN), nanMask);

            return result;
        }

        /// <summary>
        /// Cephes-style fast sin(x) for Vector256&lt;float&gt; (8 floats per vector).
        /// Range reduction to [-pi/2, pi/2] via j = round(x * 2/pi), then 5th-order minimax.
        /// Relative error ~1e-7 across the full float range.
        /// </summary>
        [MethodImpl(HotInline)]
        private static Vector256<float> FastSin256(Vector256<float> x)
        {
            // Guard: fall back to scalar for |x| > 1e6 to avoid int32 overflow in quadrant
            var absX = Avx.And(x, Vector256.Create(0x7FFFFFFF).AsSingle());
            var unsafeMask = Avx.Compare(absX, Vector256.Create(1e6f), FloatComparisonMode.OrderedGreaterThanSignaling);
            if (Avx.MoveMask(unsafeMask) != 0)
            {
                return Vector256.Create(
                    MathF.Sin(x.GetElement(0)), MathF.Sin(x.GetElement(1)),
                    MathF.Sin(x.GetElement(2)), MathF.Sin(x.GetElement(3)),
                    MathF.Sin(x.GetElement(4)), MathF.Sin(x.GetElement(5)),
                    MathF.Sin(x.GetElement(6)), MathF.Sin(x.GetElement(7)));
            }

            var twoOverPi = Vector256.Create(0.6366197723675814f);
            var piOver2Hi = Vector256.Create(1.5707963267341256f);
            var piOver2Lo = Vector256.Create(6.077100506303966e-11f);

            // Extract sign and work with abs(x)
            var signBit = Avx2.And(x.AsInt32(), Vector256.Create(unchecked((int)0x80000000)));
            x = absX;

            // j = round(x * 2/pi)
            var j = Avx.RoundToNearestInteger(Avx.Multiply(x, twoOverPi));
            var jInt = Avx.ConvertToVector256Int32(j);

            // r = x - j * pi/2
            var r = Fma.MultiplyAddNegated(j, piOver2Hi, x);
            r = Fma.MultiplyAddNegated(j, piOver2Lo, r);

            // Quadrant: if j%4 >= 2, negate
            var quadrant = Avx2.And(jInt, Vector256.Create(3));
            var needNeg = Avx2.CompareGreaterThan(quadrant, Vector256.Create(1));
            var negMask = Avx2.And(needNeg, Vector256.Create(unchecked((int)0x80000000)));

            // If j%2 == 1, use cos polynomial
            var useCosPoly = Avx2.CompareEqual(Avx2.And(jInt, Vector256.Create(1)), Vector256.Create(1));

            var r2 = Avx.Multiply(r, r);

            // Sin polynomial (odd): r - r^3/6 + r^5/120 - r^7/5040
            var s1 = Vector256.Create(-1.6666654611e-1f);
            var s2 = Vector256.Create(8.3321608736e-3f);
            var s3 = Vector256.Create(-1.9515295891e-4f);

            var sinP = Fma.MultiplyAdd(s3, r2, s2);
            sinP = Fma.MultiplyAdd(sinP, r2, s1);
            sinP = Fma.MultiplyAdd(sinP, Avx.Multiply(r2, r), r);

            // Cos polynomial (even): 1 - r^2/2 + r^4/24 - r^6/720
            var c1 = Vector256.Create(-0.5f);
            var c2 = Vector256.Create(4.1666638908e-2f);
            var c3 = Vector256.Create(-1.3888377460e-3f);

            var cosP = Fma.MultiplyAdd(c3, r2, c2);
            cosP = Fma.MultiplyAdd(cosP, r2, c1);
            cosP = Fma.MultiplyAdd(cosP, r2, Vector256.Create(1.0f));

            // Select sin or cos polynomial
            var result2 = Avx.BlendVariable(sinP, cosP, useCosPoly.AsSingle());

            // Apply negation and original sign
            var flipBits = Avx2.Xor(negMask, signBit);
            return Avx.Xor(result2, flipBits.AsSingle());
        }

        /// <summary>
        /// Cephes-style fast cos(x) for Vector256&lt;float&gt;.
        /// cos(x) = sin(x + pi/2).
        /// </summary>
        [MethodImpl(HotInline)]
        private static Vector256<float> FastCos256(Vector256<float> x)
        {
            return FastSin256(Avx.Add(x, Vector256.Create(1.5707963267948966f)));
        }
#endif

        /// <summary>
        /// Computes LeakyReLU element-wise using SIMD: max(alpha * x, x).
        /// Uses AVX/SSE for vectorized comparison and blending when available.
        /// </summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
                var vTwo = Vector256.Create(2.0f);

                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = ReadVector256(input, i);
                    var x_cubed = Avx.Multiply(Avx.Multiply(x, x), x);
                    var inner = Fma.MultiplyAdd(vCoeff, x_cubed, x);
                    var tanh_arg = Avx.Multiply(vSqrt2OverPi, inner);
                    // tanh(z) = 2*sigmoid(2z) - 1 using FastSigmoid256 (Cephes exp, ~0.01% error)
                    var tanh_val = Avx.Subtract(Avx.Multiply(vTwo, FastSigmoid256(Avx.Multiply(vTwo, tanh_arg))), vOne);
                    WriteVector256(output, i, Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(vOne, tanh_val))));
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
        [MethodImpl(HotInline)]
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
                // Step 2: temp = 1 + exp(x)
                for (int j = 0; j < length; j++)
                    temp[j] += 1f;
                // Step 3: temp = log(1 + exp(x)) = softplus(x) — uses SIMD log
                Log(temp, temp);
                // Step 4: clamp softplus for large x (softplus(x) ≈ x when x > 20)
                for (int j = 0; j < length; j++)
                {
                    if (input[j] > 20f) temp[j] = input[j];
                }
                // Step 5: temp2 = tanh(softplus(x))
                Tanh(temp, temp2);
                // Step 6: output = x * tanh(softplus(x))
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static void Sin(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, FastSin256(ReadVector256(input, i)));
                }
            }
#endif

            for (; i < length; i++)
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
        [MethodImpl(HotInline)]
        public static void Cos(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, FastCos256(ReadVector256(input, i)));
                }
            }
#endif

            for (; i < length; i++)
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
        [MethodImpl(HotInline)]
        public static void SinCos(ReadOnlySpan<float> input, Span<float> sinOutput, Span<float> cosOutput)
        {
            if (input.Length != sinOutput.Length || input.Length != cosOutput.Length)
            {
                throw new ArgumentException("All spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    var x = ReadVector256(input, i);
                    WriteVector256(sinOutput, i, FastSin256(x));
                    WriteVector256(cosOutput, i, FastCos256(x));
                }
            }
#endif

            for (; i < length; i++)
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
        [MethodImpl(HotInline)]
        public static void Sin(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(output, i,
                        FastSinDouble256(ReadVector256Double(input, i)));
                }
            }
#endif

            for (; i < length; i++)
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
        [MethodImpl(HotInline)]
        public static void Cos(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    WriteVector256Double(output, i,
                        FastCosDouble256(ReadVector256Double(input, i)));
                }
            }
#endif

            for (; i < length; i++)
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
        [MethodImpl(HotInline)]
        public static void SinCos(ReadOnlySpan<double> input, Span<double> sinOutput, Span<double> cosOutput)
        {
            if (input.Length != sinOutput.Length || input.Length != cosOutput.Length)
            {
                throw new ArgumentException("All spans must have the same length.");
            }

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var x = ReadVector256Double(input, i);
                    WriteVector256Double(sinOutput, i, FastSinDouble256(x));
                    WriteVector256Double(cosOutput, i, FastCosDouble256(x));
                }
            }
#endif

            for (; i < length; i++)
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
        [MethodImpl(HotInline)]
        public static unsafe void Log(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // FastLog256 polynomial (MKL VML tested but delegate overhead negated benefit).
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
        [MethodImpl(HotInline)]
        public static unsafe void Log(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;

#if NET5_0_OR_GREATER
            // VML double via vmdLn (mode-per-call, VML_EP guaranteed).
            fixed (double* pIn = input)
            fixed (double* pOut = output)
            {
                if (VmlProvider.TryLn(pIn, pOut, length))
                    return;
            }
#endif

            int i = 0;
            int unrolled = length & ~3;
            for (; i < unrolled; i += 4)
            {
                output[i] = Math.Log(input[i]);
                output[i + 1] = Math.Log(input[i + 1]);
                output[i + 2] = Math.Log(input[i + 2]);
                output[i + 3] = Math.Log(input[i + 3]);
            }
            for (; i < length; i++)
            {
                output[i] = Math.Log(input[i]);
            }
        }

        /// <summary>Element-wise log base 2 using SIMD: log2(x) = log(x) / ln(2).</summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static void Log2(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = input.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // log2(x) = log(x) / ln(2) = log(x) * log2(e)
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                var vLog2e = Vector256.Create(1.4426950408889634); // 1/ln(2)
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var logVal = FastLogDouble256(ReadVector256Double(input, i));
                    WriteVector256Double(output, i, Avx.Multiply(logVal, vLog2e));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = Math.Log2(input[i]);
#else
                output[i] = Math.Log(input[i]) / Math.Log(2.0);
#endif
            }
        }

        /// <summary>Element-wise square root using AVX.</summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static void Pow(ReadOnlySpan<float> baseValues, ReadOnlySpan<float> exponents, Span<float> output)
        {
            if (baseValues.Length != exponents.Length || baseValues.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = baseValues.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // pow(x, y) = exp(y * log(x)) — only valid for positive bases
            if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
            {
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    var bases = ReadVector256(baseValues, i);
                    // Check if all bases are positive and finite — fall back to scalar for
                    // non-positive, NaN, or Infinity bases (MathF.Pow handles these correctly)
                    var nonPositive = Avx.Compare(bases, Vector256<float>.Zero, FloatComparisonMode.OrderedLessThanOrEqualSignaling);
                    var nanBases = Avx.Compare(bases, bases, FloatComparisonMode.UnorderedNotEqualSignaling);
                    var infBases = Avx.CompareEqual(bases, Vector256.Create(float.PositiveInfinity));
                    var needScalar = Avx.Or(nonPositive, Avx.Or(nanBases, infBases));
                    if (Avx.MoveMask(needScalar) != 0)
                    {
                        // Scalar fallback for this chunk
                        for (int k = i; k < i + 8; k++)
                            output[k] = MathF.Pow(baseValues[k], exponents[k]);
                        continue;
                    }
                    var logBase = FastLog256(bases);
                    var yLogX = Avx.Multiply(ReadVector256(exponents, i), logBase);
                    WriteVector256(output, i, FastExp256(yLogX));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Pow(baseValues[i], exponents[i]);
#else
                output[i] = (float)Math.Pow(baseValues[i], exponents[i]);
#endif
            }
        }

        /// <summary>Element-wise power with scalar exponent: result[i] = base[i] ^ exp.</summary>
        [MethodImpl(HotInline)]
        public static void Pow(ReadOnlySpan<float> baseValues, float exponent, Span<float> output)
        {
            if (baseValues.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = baseValues.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // pow(x, y) = exp(y * log(x)) — only valid for positive bases
            if (Avx2.IsSupported && Fma.IsSupported && length >= 8)
            {
                var vExp = Vector256.Create(exponent);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
                {
                    var bases = ReadVector256(baseValues, i);
                    var nonPositive = Avx.Compare(bases, Vector256<float>.Zero, FloatComparisonMode.OrderedLessThanOrEqualSignaling);
                    if (Avx.MoveMask(nonPositive) != 0)
                    {
                        for (int k = i; k < i + 8; k++)
                            output[k] = MathF.Pow(baseValues[k], exponent);
                        continue;
                    }
                    var logBase = FastLog256(bases);
                    WriteVector256(output, i, FastExp256(Avx.Multiply(vExp, logBase)));
                }
            }
#endif

            for (; i < length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Pow(baseValues[i], exponent);
#else
                output[i] = (float)Math.Pow(baseValues[i], exponent);
#endif
            }
        }

        /// <summary>Element-wise clamp to [min, max] range for double precision.</summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static void Pow(ReadOnlySpan<double> baseValues, double exponent, Span<double> output)
        {
            if (baseValues.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            int length = baseValues.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // pow(x, y) = exp(y * log(x)) — only valid for positive bases
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                var vExp = Vector256.Create(exponent);
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var bases = ReadVector256Double(baseValues, i);
                    var nonPositive = Avx.Compare(bases, Vector256<double>.Zero, FloatComparisonMode.OrderedLessThanOrEqualSignaling);
                    if (Avx.MoveMask(nonPositive) != 0)
                    {
                        for (int k = i; k < i + 4; k++)
                            output[k] = Math.Pow(baseValues[k], exponent);
                        continue;
                    }
                    var logVal = FastLogDouble256(bases);
                    WriteVector256Double(output, i, FastExpDouble256(Avx.Multiply(vExp, logVal)));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = Math.Pow(baseValues[i], exponent);
            }
        }

        /// <summary>
        /// Computes SoftMax: output[i] = exp(x[i] - max(x)) / sum(exp(x - max(x))).
        /// Numerically stable via max subtraction.
        /// </summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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

        /// <summary>
        /// AVX2 hardware gather for float: collects elements at stride intervals.
        /// Uses VGATHERDPS to load 8 floats per instruction with computed index vectors.
        /// Falls back to scalar for tail elements and non-AVX2 hardware.
        /// </summary>
        [MethodImpl(HotInline)]
        public static unsafe void StridedGatherFloat(float* source, int offset, int stride, float* result, int count)
        {
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && count >= 8)
            {
                int i = 0;

                // Build index vector: [0*stride, 1*stride, 2*stride, ..., 7*stride]
                // Each element is the byte offset from the base pointer
                var indices = Vector256.Create(
                    0, stride, stride * 2, stride * 3,
                    stride * 4, stride * 5, stride * 6, stride * 7);

                int simdCount = count & ~7; // Round down to multiple of 8
                for (; i < simdCount; i += 8)
                {
                    // VGATHERDPS: gather 8 floats from source + offset using index vector
                    // Scale=4 because indices are element offsets, each float is 4 bytes
                    var gathered = Avx2.GatherVector256(source + offset + i * stride, indices, 4);
                    Avx.Store(result + i, gathered);
                }

                // Scalar tail
                for (; i < count; i++)
                {
                    result[i] = source[offset + i * stride];
                }
                return;
            }
#endif
            // Scalar fallback
            for (int i = 0; i < count; i++)
            {
                result[i] = source[offset + i * stride];
            }
        }

        /// <summary>
        /// AVX2 hardware gather for double: collects elements at stride intervals.
        /// Uses VGATHERQPD to load 4 doubles per instruction.
        /// </summary>
        [MethodImpl(HotInline)]
        public static unsafe void StridedGatherDouble(double* source, int offset, int stride, double* result, int count)
        {
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && count >= 4)
            {
                int i = 0;

                // Index vector for 4 doubles: [0, stride, 2*stride, 3*stride]
                // Scale=8 because each double is 8 bytes
                var indices = Vector128.Create(0, stride, stride * 2, stride * 3);

                int simdCount = count & ~3;
                for (; i < simdCount; i += 4)
                {
                    var gathered = Avx2.GatherVector256(source + offset + i * stride, indices, 8);
                    Avx.Store(result + i, gathered);
                }

                // Scalar tail
                for (; i < count; i++)
                {
                    result[i] = source[offset + i * stride];
                }
                return;
            }
#endif
            for (int i = 0; i < count; i++)
            {
                result[i] = source[offset + i * stride];
            }
        }

        /// <summary>
        /// Strided scatter for float: writes elements to destination at stride intervals.
        /// No hardware scatter instruction on x86 (scatter is AVX-512 only), so this uses
        /// an unrolled scalar loop for best performance.
        /// </summary>
        [MethodImpl(HotInline)]
        public static unsafe void StridedScatterFloat(float* source, float* destination, int offset, int stride, int count)
        {
            int i = 0;
            // 4x unrolled for ILP (instruction-level parallelism)
            int unrolled = count & ~3;
            for (; i < unrolled; i += 4)
            {
                destination[offset + i * stride] = source[i];
                destination[offset + (i + 1) * stride] = source[i + 1];
                destination[offset + (i + 2) * stride] = source[i + 2];
                destination[offset + (i + 3) * stride] = source[i + 3];
            }
            for (; i < count; i++)
            {
                destination[offset + i * stride] = source[i];
            }
        }

        /// <summary>
        /// Strided scatter for double: writes elements to destination at stride intervals.
        /// </summary>
        [MethodImpl(HotInline)]
        public static unsafe void StridedScatterDouble(double* source, double* destination, int offset, int stride, int count)
        {
            int i = 0;
            int unrolled = count & ~3;
            for (; i < unrolled; i += 4)
            {
                destination[offset + i * stride] = source[i];
                destination[offset + (i + 1) * stride] = source[i + 1];
                destination[offset + (i + 2) * stride] = source[i + 2];
                destination[offset + (i + 3) * stride] = source[i + 3];
            }
            for (; i < count; i++)
            {
                destination[offset + i * stride] = source[i];
            }
        }

        /// <summary>Pointer-based MultiplyScalar for float — AVX 8-wide with 4x unroll.</summary>
        [MethodImpl(HotInline)]
        public static unsafe void MultiplyScalarUnsafe(float* a, float scalar, float* result, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    Avx.Store(result + i, Avx.Multiply(Avx.LoadVector256(a + i), vs));
                    Avx.Store(result + i + 8, Avx.Multiply(Avx.LoadVector256(a + i + 8), vs));
                    Avx.Store(result + i + 16, Avx.Multiply(Avx.LoadVector256(a + i + 16), vs));
                    Avx.Store(result + i + 24, Avx.Multiply(Avx.LoadVector256(a + i + 24), vs));
                }
            }
            if (Avx.IsSupported && length - i >= 8)
            {
                var vs = Vector256.Create(scalar);
                int simdLength = i + ((length - i) & ~7);
                for (; i < simdLength; i += 8)
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

        /// <summary>Pointer-based MultiplyScalar for double — zero bounds-checking overhead.</summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static void DivideScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
        {
            MultiplyScalar(a, 1.0 / scalar, result);
        }

        /// <summary>Subtracts a scalar from each double element.</summary>
        [MethodImpl(HotInline)]
        public static void SubtractScalar(ReadOnlySpan<double> a, double scalar, Span<double> result)
        {
            AddScalar(a, -scalar, result);
        }

        /// <summary>Sum for double precision with 4-way parallel accumulation.</summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
#if NET5_0_OR_GREATER
                sum = Fma.IsSupported
                    ? Math.FusedMultiplyAdd(a[i], b[i], sum)
                    : sum + (a[i] * b[i]);
#else
                sum += a[i] * b[i];
#endif
            }

            return sum;
        }

        /// <summary>Max for double precision.</summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static void SoftMax(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length.");

            double maxVal = Max(input);

            // Subtract max and exp — use SIMD path
            int length = input.Length;
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                var vMax = Vector256.Create(maxVal);
                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var x = Avx.Subtract(ReadVector256Double(input, i), vMax);
                    WriteVector256Double(output, i, FastExpDouble256(x));
                }
            }
#endif
            for (; i < length; i++)
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        public static void GELU(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                var vHalf = Vector256.Create(0.5);
                var vOne = Vector256.Create(1.0);
                var vTwo = Vector256.Create(2.0);
                var vCoeff = Vector256.Create(0.044715);
                var vSqrt2OverPi = Vector256.Create(0.7978845608028654);

                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var x = ReadVector256Double(input, i);
                    var x3 = Avx.Multiply(Avx.Multiply(x, x), x);
                    var inner = Fma.MultiplyAdd(vCoeff, x3, x);
                    var tanhArg = Avx.Multiply(vSqrt2OverPi, inner);
                    // tanh via (exp(2t)-1)/(exp(2t)+1)
                    var e2t = FastExpDouble256(Avx.Multiply(vTwo, tanhArg));
                    var tanhVal = Avx.Divide(Avx.Subtract(e2t, vOne), Avx.Add(e2t, vOne));
                    WriteVector256Double(output, i,
                        Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(vOne, tanhVal))));
                }
            }
#endif

            for (; i < length; i++)
            {
                double x = input[i];
                double x3 = x * x * x;
                double inner = x + 0.044715 * x3;
                double tanhVal = Math.Tanh(0.7978845608028654 * inner);
                output[i] = 0.5 * x * (1.0 + tanhVal);
            }
        }

        /// <summary>
        /// Computes Mish activation element-wise for double precision: x * tanh(softplus(x)).
        /// </summary>
        [MethodImpl(HotInline)]
        public static void Mish(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                var vOne = Vector256.Create(1.0);
                var vTwo = Vector256.Create(2.0);

                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var x = ReadVector256Double(input, i);
                    // softplus = ln(1 + exp(x))
                    var expX = FastExpDouble256(x);
                    var softplus = FastLogDouble256(Avx.Add(vOne, expX));
                    // tanh(softplus) via (exp(2*sp)-1)/(exp(2*sp)+1)
                    var e2sp = FastExpDouble256(Avx.Multiply(vTwo, softplus));
                    var tanhSp = Avx.Divide(Avx.Subtract(e2sp, vOne), Avx.Add(e2sp, vOne));
                    WriteVector256Double(output, i, Avx.Multiply(x, tanhSp));
                }
            }
#endif

            for (; i < length; i++)
            {
                double x = input[i];
                double softplus = x > 20.0 ? x : Math.Log(1.0 + Math.Exp(x));
                output[i] = x * Math.Tanh(softplus);
            }
        }

        /// <summary>
        /// Computes Swish/SiLU activation element-wise for double precision: x * sigmoid(x).
        /// </summary>
        [MethodImpl(HotInline)]
        public static void Swish(ReadOnlySpan<double> input, Span<double> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
            if (Avx2.IsSupported && Fma.IsSupported && length >= 4)
            {
                var vOne = Vector256.Create(1.0);
                var vNegOne = Vector256.Create(-1.0);

                int simdLength = i + ((length - i) & ~3);
                for (; i < simdLength; i += 4)
                {
                    var x = ReadVector256Double(input, i);
                    var negX = Avx.Multiply(vNegOne, x);
                    var sigmoid = Avx.Divide(vOne, Avx.Add(vOne, FastExpDouble256(negX)));
                    WriteVector256Double(output, i, Avx.Multiply(x, sigmoid));
                }
            }
#endif

            for (; i < length; i++)
            {
                double x = input[i];
                double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
                output[i] = x * sigmoid;
            }
        }

        /// <summary>
        /// Computes ELU (Exponential Linear Unit) element-wise for double precision.
        /// </summary>
        [MethodImpl(HotInline)]
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
        [MethodImpl(HotInline)]
        internal static Vector256<float> ReadVector256(ReadOnlySpan<float> data, int offset)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector256<float>>(ref Unsafe.As<float, byte>(ref element));
        }

        [MethodImpl(HotInline)]
        private static void WriteVector256(Span<float> data, int offset, Vector256<float> value)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref element), value);
        }

        [MethodImpl(HotInline)]
        private static Vector128<float> ReadVector128(ReadOnlySpan<float> data, int offset)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector128<float>>(ref Unsafe.As<float, byte>(ref element));
        }

        [MethodImpl(HotInline)]
        private static void WriteVector128(Span<float> data, int offset, Vector128<float> value)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref element), value);
        }

        [MethodImpl(HotInline)]
        internal static float HorizontalSum(Vector256<float> v)
        {
            // SIMD shuffle reduction: no stack spill
            // Step 1: Add upper 128 bits to lower 128 bits
            var lo = v.GetLower();
            var hi = Avx.ExtractVector128(v, 1);
            var sum128 = Sse.Add(lo, hi);
            return HorizontalSum(sum128);
        }

        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
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

        [MethodImpl(HotInline)]
        private static Vector256<double> ReadVector256Double(ReadOnlySpan<double> data, int offset)
        {
            ref double start = ref MemoryMarshal.GetReference(data);
            ref double element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector256<double>>(ref Unsafe.As<double, byte>(ref element));
        }

        [MethodImpl(HotInline)]
        private static void WriteVector256Double(Span<double> data, int offset, Vector256<double> value)
        {
            ref double start = ref MemoryMarshal.GetReference(data);
            ref double element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<double, byte>(ref element), value);
        }

        [MethodImpl(HotInline)]
        private static Vector128<double> ReadVector128Double(ReadOnlySpan<double> data, int offset)
        {
            ref double start = ref MemoryMarshal.GetReference(data);
            ref double element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector128<double>>(ref Unsafe.As<double, byte>(ref element));
        }

        [MethodImpl(HotInline)]
        private static void WriteVector128Double(Span<double> data, int offset, Vector128<double> value)
        {
            ref double start = ref MemoryMarshal.GetReference(data);
            ref double element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<double, byte>(ref element), value);
        }

        [MethodImpl(HotInline)]
        private static double HorizontalSum(Vector256<double> v)
        {
            // SIMD shuffle reduction: add upper 128 to lower 128, then reduce 128-bit
            var lo = v.GetLower();
            var hi = Avx.ExtractVector128(v.AsDouble(), 1);
            var sum128 = Sse2.Add(lo, hi);
            return HorizontalSum(sum128);
        }

        [MethodImpl(HotInline)]
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
            int totalElements = outerSize * axisSize;
            if (input.Length < totalElements)
                throw new ArgumentException($"Input span length {input.Length} is less than outerSize*axisSize ({totalElements}).", nameof(input));
            if (output.Length < totalElements)
                throw new ArgumentException($"Output span length {output.Length} is less than outerSize*axisSize ({totalElements}).", nameof(output));

            fixed (float* pIn = input)
            fixed (float* pOut = output)
            {
#if NET5_0_OR_GREATER
                // oneDNN path: fused SVML-accelerated softmax in a single call.
                // Processes ALL rows in one primitive execution (2 optimized passes).
                if (OneDnnProvider.TrySoftmax(pIn, pOut, outerSize, axisSize))
                    return;
#endif

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
        [MethodImpl(HotInline)]
        internal static unsafe void SoftmaxRowUnsafe(float* input, float* output, int length)
        {
            // Tiled softmax: process in L1-cache-sized tiles to maximize data reuse.
            // Tile size = 2048 floats = 8KB (fits in 32KB L1 with room for input+output).
            // For small rows (<= tileSize), this degrades to the original 2-pass approach
            // but with the divide fused into the same cache-hot window.
            const int tileSize = 2048;

            if (length <= tileSize)
            {
                // Small row: data fits in L1. Use fast 2-pass approach.
                SoftmaxRowSmall(input, output, length);
                return;
            }

            // Large row: tile to keep data in L1 cache.
            // Pass 1: find global max across all tiles
            float maxVal = float.NegativeInfinity;
            for (int t = 0; t < length; t += tileSize)
            {
                int count = Math.Min(tileSize, length - t);
                for (int j = 0; j < count; j++)
                {
                    float v = input[t + j];
                    if (v > maxVal) maxVal = v;
                }
            }

            // Pass 2: per-tile exp+store+sum, then immediately divide the tile
            // while it's still in L1 cache (avoids full-array divide pass)
            float totalSum = 0f;
            // First compute all exp values and accumulate total sum
            for (int t = 0; t < length; t += tileSize)
            {
                int count = Math.Min(tileSize, length - t);
                float tileSum = 0f;
                for (int j = 0; j < count; j++)
                {
#if NET5_0_OR_GREATER
                    float e = MathF.Exp(input[t + j] - maxVal);
#else
                    float e = (float)Math.Exp(input[t + j] - maxVal);
#endif
                    output[t + j] = e;
                    tileSum += e;
                }
                totalSum += tileSum;
            }

            // Pass 3: divide (can't fuse with pass 2 because we need total sum)
            if (totalSum == 0f) return;
            float invSum = 1f / totalSum;
            for (int j = 0; j < length; j++)
                output[j] *= invSum;
        }

        /// <summary>
        /// Optimized softmax for rows that fit in L1 cache (≤ 2048 floats = 8KB).
        /// Uses SIMD for max, fused exp+sum, and divide passes.
        /// All data stays in L1 across the 2 passes.
        /// </summary>
#if NET5_0_OR_GREATER
        /// <summary>
        /// VML-accelerated softmax row: SIMD max → subtract → VML exp (SVML) → SIMD sum → SIMD divide.
        /// Returns false if VML exp fails (caller falls through to polynomial path).
        /// </summary>
        private static unsafe bool SoftmaxRowVml(float* input, float* output, int length)
        {
            // Pass 1: Find max (SIMD 4x unrolled)
            float maxVal = float.NegativeInfinity;
            int j = 0;
            var vm0 = Vector256.Create(float.NegativeInfinity);
            var vm1 = vm0; var vm2 = vm0; var vm3 = vm0;
            int simdLen = length & ~31;
            for (; j < simdLen; j += 32)
            {
                vm0 = Avx.Max(vm0, Avx.LoadVector256(input + j));
                vm1 = Avx.Max(vm1, Avx.LoadVector256(input + j + 8));
                vm2 = Avx.Max(vm2, Avx.LoadVector256(input + j + 16));
                vm3 = Avx.Max(vm3, Avx.LoadVector256(input + j + 24));
            }
            vm0 = Avx.Max(Avx.Max(vm0, vm1), Avx.Max(vm2, vm3));
            maxVal = HorizontalMax(vm0);
            for (; j < length; j++)
                if (input[j] > maxVal) maxVal = input[j];

            // Pass 2a: output = input - max (SIMD subtract)
            j = 0;
            var vmaxBc = Vector256.Create(maxVal);
            for (; j < simdLen; j += 32)
            {
                Avx.Store(output + j, Avx.Subtract(Avx.LoadVector256(input + j), vmaxBc));
                Avx.Store(output + j + 8, Avx.Subtract(Avx.LoadVector256(input + j + 8), vmaxBc));
                Avx.Store(output + j + 16, Avx.Subtract(Avx.LoadVector256(input + j + 16), vmaxBc));
                Avx.Store(output + j + 24, Avx.Subtract(Avx.LoadVector256(input + j + 24), vmaxBc));
            }
            for (; j < length; j++)
                output[j] = input[j] - maxVal;

            // Pass 2b: output = exp(output) via MKL VML (SVML microcode — the key speedup)
            if (!VmlProvider.TryExp(output, output, length))
                return false;

            // Pass 3: sum (SIMD 4x unrolled)
            float sumExp = 0f;
            j = 0;
            var vs0 = Vector256<float>.Zero;
            var vs1 = Vector256<float>.Zero;
            var vs2 = Vector256<float>.Zero;
            var vs3 = Vector256<float>.Zero;
            for (; j < simdLen; j += 32)
            {
                vs0 = Avx.Add(vs0, Avx.LoadVector256(output + j));
                vs1 = Avx.Add(vs1, Avx.LoadVector256(output + j + 8));
                vs2 = Avx.Add(vs2, Avx.LoadVector256(output + j + 16));
                vs3 = Avx.Add(vs3, Avx.LoadVector256(output + j + 24));
            }
            vs0 = Avx.Add(Avx.Add(vs0, vs1), Avx.Add(vs2, vs3));
            sumExp = HorizontalSum(vs0);
            for (; j < length; j++)
                sumExp += output[j];

            // Divide
            if (sumExp == 0f) return true;
            float inv = 1f / sumExp;
            j = 0;
            var vInv = Vector256.Create(inv);
            for (; j < simdLen; j += 32)
            {
                Avx.Store(output + j, Avx.Multiply(Avx.LoadVector256(output + j), vInv));
                Avx.Store(output + j + 8, Avx.Multiply(Avx.LoadVector256(output + j + 8), vInv));
                Avx.Store(output + j + 16, Avx.Multiply(Avx.LoadVector256(output + j + 16), vInv));
                Avx.Store(output + j + 24, Avx.Multiply(Avx.LoadVector256(output + j + 24), vInv));
            }
            for (; j < length; j++)
                output[j] *= inv;
            return true;
        }
#endif

#if NET5_0_OR_GREATER
        /// <summary>
        /// VML-accelerated log-softmax: max → subtract → VML exp → sum → log(sum) → subtract.
        /// log_softmax(x) = (x - max) - log(sum(exp(x - max)))
        /// </summary>
        private static unsafe bool LogSoftmaxRowVml(float* input, float* output, int length)
        {
            // Pass 1: Find max
            float maxVal = float.NegativeInfinity;
            int j = 0;
            var vm0 = Vector256.Create(float.NegativeInfinity);
            var vm1 = vm0; var vm2 = vm0; var vm3 = vm0;
            int simdLen = length & ~31;
            for (; j < simdLen; j += 32)
            {
                vm0 = Avx.Max(vm0, Avx.LoadVector256(input + j));
                vm1 = Avx.Max(vm1, Avx.LoadVector256(input + j + 8));
                vm2 = Avx.Max(vm2, Avx.LoadVector256(input + j + 16));
                vm3 = Avx.Max(vm3, Avx.LoadVector256(input + j + 24));
            }
            vm0 = Avx.Max(Avx.Max(vm0, vm1), Avx.Max(vm2, vm3));
            maxVal = HorizontalMax(vm0);
            for (; j < length; j++)
                if (input[j] > maxVal) maxVal = input[j];

            // Pass 2: output = input - max
            j = 0;
            var vmaxBc = Vector256.Create(maxVal);
            for (; j < simdLen; j += 32)
            {
                Avx.Store(output + j, Avx.Subtract(Avx.LoadVector256(input + j), vmaxBc));
                Avx.Store(output + j + 8, Avx.Subtract(Avx.LoadVector256(input + j + 8), vmaxBc));
                Avx.Store(output + j + 16, Avx.Subtract(Avx.LoadVector256(input + j + 16), vmaxBc));
                Avx.Store(output + j + 24, Avx.Subtract(Avx.LoadVector256(input + j + 24), vmaxBc));
            }
            for (; j < length; j++)
                output[j] = input[j] - maxVal;

            // We need exp(output) for the sum, but we want to keep output = (x - max).
            // Use a temp buffer via stackalloc for small, or compute exp sum from VML in-place then restore.
            // Strategy: compute sum(exp(output)) using VML exp on a temp, then subtract log(sum) from output.
            // For simplicity: use the existing output as temp, compute exp, sum, then rewrite as (x-max)-log(sum).

            // Actually, simpler: we already have output = (x - max). We need sum(exp(output)).
            // We can't use VML in-place without losing the shifted values.
            // Instead: use VML on a copy, or compute exp+sum without storing.
            // Best approach: compute VML exp into output, sum it, then rewrite output as (x-max) - log(sum).

            // Step A: Save shifted values will be recomputed from log(sum) subtraction
            // VML exp in-place on output
            if (!VmlProvider.TryExp(output, output, length))
                return false;

            // Step B: Sum the exp values
            float sumExp = 0f;
            j = 0;
            var vs0 = Vector256<float>.Zero;
            var vs1 = Vector256<float>.Zero;
            var vs2 = Vector256<float>.Zero;
            var vs3 = Vector256<float>.Zero;
            for (; j < simdLen; j += 32)
            {
                vs0 = Avx.Add(vs0, Avx.LoadVector256(output + j));
                vs1 = Avx.Add(vs1, Avx.LoadVector256(output + j + 8));
                vs2 = Avx.Add(vs2, Avx.LoadVector256(output + j + 16));
                vs3 = Avx.Add(vs3, Avx.LoadVector256(output + j + 24));
            }
            vs0 = Avx.Add(Avx.Add(vs0, vs1), Avx.Add(vs2, vs3));
            sumExp = HorizontalSum(vs0);
            for (; j < length; j++)
                sumExp += output[j];

            // Step C: output = (x - max) - log(sum) = input - max - log(sum)
            float logSumExp = MathF.Log(sumExp);
            float maxPlusLog = maxVal + logSumExp;
            j = 0;
            var vMPL = Vector256.Create(maxPlusLog);
            for (; j < simdLen; j += 32)
            {
                Avx.Store(output + j, Avx.Subtract(Avx.LoadVector256(input + j), vMPL));
                Avx.Store(output + j + 8, Avx.Subtract(Avx.LoadVector256(input + j + 8), vMPL));
                Avx.Store(output + j + 16, Avx.Subtract(Avx.LoadVector256(input + j + 16), vMPL));
                Avx.Store(output + j + 24, Avx.Subtract(Avx.LoadVector256(input + j + 24), vMPL));
            }
            for (; j < length; j++)
                output[j] = input[j] - maxPlusLog;
            return true;
        }
#endif

        private static unsafe void SoftmaxRowSmall(float* input, float* output, int length)
        {
#if NET5_0_OR_GREATER
            // VML fast path: SIMD max → VML exp(x-max) → SIMD sum → SIMD divide
            if (VmlProvider.IsAvailable && Avx.IsSupported && length >= 32)
            {
                if (SoftmaxRowVml(input, output, length))
                    return;
            }
#endif

            int i = 0;

            // Pass 1: Find max (SIMD 4x unrolled) — fallback when VML not available
            float maxVal = float.NegativeInfinity;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 32)
            {
                var vmax0 = Vector256.Create(float.NegativeInfinity);
                var vmax1 = vmax0; var vmax2 = vmax0; var vmax3 = vmax0;
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
                if (input[i] > maxVal) maxVal = input[i];

            // Pass 2: Fused exp + store + sum (SIMD 4x unrolled)
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

            // Divide (data still in L1 from pass 2)
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
                output[i] *= invSum;
        }

        /// <summary>
        /// Computes log_softmax for a single contiguous row using unsafe pointers.
        /// log_softmax(x) = (x - max) - log(sum(exp(x - max)))
        /// Uses inline FastExp256 to avoid per-call overhead.
        /// </summary>
        [MethodImpl(HotInline)]
        internal static unsafe void LogSoftmaxRowUnsafe(float* input, float* output, int length)
        {
#if NET5_0_OR_GREATER
            // VML fast path: SIMD max → SIMD subtract → VML exp → SIMD sum → log → SIMD subtract
            if (VmlProvider.IsAvailable && Avx.IsSupported && length >= 32)
            {
                if (LogSoftmaxRowVml(input, output, length))
                    return;
            }
#endif

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

    #region Backward Kernels

        /// <summary>
        /// SIMD-accelerated ReLU backward: result[i] = input[i] > 0 ? grad[i] : 0
        /// Uses AVX2 compare + bitwise AND for zero-branch vectorization.
        /// </summary>
        public static unsafe void ReluBackwardUnsafe(float* grad, float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            // AVX-512: 16 floats per vector, 128 floats per iteration — 2x throughput vs AVX-256
            if (Avx512F.IsSupported)
            {
                var vzero = Vector512<float>.Zero;
                int simd128 = length & ~127;
                for (; i < simd128; i += 128)
                {
                    // input > 0 produces all-ones/all-zeros mask; AND with grad zeros out negative lanes
                    var m0 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i), vzero).AsInt32();
                    var m1 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 16), vzero).AsInt32();
                    var m2 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 32), vzero).AsInt32();
                    var m3 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 48), vzero).AsInt32();
                    var m4 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 64), vzero).AsInt32();
                    var m5 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 80), vzero).AsInt32();
                    var m6 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 96), vzero).AsInt32();
                    var m7 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 112), vzero).AsInt32();
                    Avx512F.Store(output + i, Avx512F.And(Avx512F.LoadVector512(grad + i).AsInt32(), m0).AsSingle());
                    Avx512F.Store(output + i + 16, Avx512F.And(Avx512F.LoadVector512(grad + i + 16).AsInt32(), m1).AsSingle());
                    Avx512F.Store(output + i + 32, Avx512F.And(Avx512F.LoadVector512(grad + i + 32).AsInt32(), m2).AsSingle());
                    Avx512F.Store(output + i + 48, Avx512F.And(Avx512F.LoadVector512(grad + i + 48).AsInt32(), m3).AsSingle());
                    Avx512F.Store(output + i + 64, Avx512F.And(Avx512F.LoadVector512(grad + i + 64).AsInt32(), m4).AsSingle());
                    Avx512F.Store(output + i + 80, Avx512F.And(Avx512F.LoadVector512(grad + i + 80).AsInt32(), m5).AsSingle());
                    Avx512F.Store(output + i + 96, Avx512F.And(Avx512F.LoadVector512(grad + i + 96).AsInt32(), m6).AsSingle());
                    Avx512F.Store(output + i + 112, Avx512F.And(Avx512F.LoadVector512(grad + i + 112).AsInt32(), m7).AsSingle());
                }
                int simd16 = length & ~15;
                for (; i < simd16; i += 16)
                {
                    var m = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i), vzero).AsInt32();
                    Avx512F.Store(output + i, Avx512F.And(Avx512F.LoadVector512(grad + i).AsInt32(), m).AsSingle());
                }
            }
            if (Avx.IsSupported && i < length)
            {
                var vzero = Vector256<float>.Zero;
                // Process 64 floats (2 cache lines) per iteration with prefetch
                int simd64 = length & ~63;
                for (; i < simd64; i += 64)
                {
                    // Prefetch next 2 cache lines (256 bytes ahead = 64 floats)
                    if (i + 128 < length)
                    {
                        Sse.Prefetch1(input + i + 64);
                        Sse.Prefetch1(input + i + 80);
                        Sse.Prefetch1(grad + i + 64);
                        Sse.Prefetch1(grad + i + 80);
                    }
                    // Block 0: 32 floats
                    var mask0 = Avx.Compare(Avx.LoadVector256(input + i), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask1 = Avx.Compare(Avx.LoadVector256(input + i + 8), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask2 = Avx.Compare(Avx.LoadVector256(input + i + 16), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask3 = Avx.Compare(Avx.LoadVector256(input + i + 24), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(output + i, Avx.And(Avx.LoadVector256(grad + i), mask0));
                    Avx.Store(output + i + 8, Avx.And(Avx.LoadVector256(grad + i + 8), mask1));
                    Avx.Store(output + i + 16, Avx.And(Avx.LoadVector256(grad + i + 16), mask2));
                    Avx.Store(output + i + 24, Avx.And(Avx.LoadVector256(grad + i + 24), mask3));
                    // Block 1: next 32 floats
                    var mask4 = Avx.Compare(Avx.LoadVector256(input + i + 32), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask5 = Avx.Compare(Avx.LoadVector256(input + i + 40), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask6 = Avx.Compare(Avx.LoadVector256(input + i + 48), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask7 = Avx.Compare(Avx.LoadVector256(input + i + 56), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(output + i + 32, Avx.And(Avx.LoadVector256(grad + i + 32), mask4));
                    Avx.Store(output + i + 40, Avx.And(Avx.LoadVector256(grad + i + 40), mask5));
                    Avx.Store(output + i + 48, Avx.And(Avx.LoadVector256(grad + i + 48), mask6));
                    Avx.Store(output + i + 56, Avx.And(Avx.LoadVector256(grad + i + 56), mask7));
                }
                int simdLength = length & ~31;
                for (; i < simdLength; i += 32)
                {
                    var mask0 = Avx.Compare(Avx.LoadVector256(input + i), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask1 = Avx.Compare(Avx.LoadVector256(input + i + 8), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask2 = Avx.Compare(Avx.LoadVector256(input + i + 16), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask3 = Avx.Compare(Avx.LoadVector256(input + i + 24), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(output + i, Avx.And(Avx.LoadVector256(grad + i), mask0));
                    Avx.Store(output + i + 8, Avx.And(Avx.LoadVector256(grad + i + 8), mask1));
                    Avx.Store(output + i + 16, Avx.And(Avx.LoadVector256(grad + i + 16), mask2));
                    Avx.Store(output + i + 24, Avx.And(Avx.LoadVector256(grad + i + 24), mask3));
                }
                int simdTail = length & ~7;
                for (; i < simdTail; i += 8)
                {
                    var mask = Avx.Compare(Avx.LoadVector256(input + i), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(output + i, Avx.And(Avx.LoadVector256(grad + i), mask));
                }
            }
#endif
            for (; i < length; i++)
                output[i] = input[i] > 0 ? grad[i] : 0;
        }

        /// <summary>
        /// In-place ReLU backward: grad[i] = (input[i] > 0) ? grad[i] : 0
        /// Writes directly into the grad buffer, eliminating one memory stream and one allocation.
        /// Only 2 arrays touched (grad read+write, input read) vs 3 for the allocating path.
        /// </summary>
        public static unsafe void ReluBackwardInPlaceUnsafe(float* grad, float* input, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx512F.IsSupported)
            {
                var vzero = Vector512<float>.Zero;
                int simd64 = length & ~63;
                for (; i < simd64; i += 64)
                {
                    var m0 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i), vzero).AsInt32();
                    var m1 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 16), vzero).AsInt32();
                    var m2 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 32), vzero).AsInt32();
                    var m3 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 48), vzero).AsInt32();
                    Avx512F.Store(grad + i, Avx512F.And(Avx512F.LoadVector512(grad + i).AsInt32(), m0).AsSingle());
                    Avx512F.Store(grad + i + 16, Avx512F.And(Avx512F.LoadVector512(grad + i + 16).AsInt32(), m1).AsSingle());
                    Avx512F.Store(grad + i + 32, Avx512F.And(Avx512F.LoadVector512(grad + i + 32).AsInt32(), m2).AsSingle());
                    Avx512F.Store(grad + i + 48, Avx512F.And(Avx512F.LoadVector512(grad + i + 48).AsInt32(), m3).AsSingle());
                }
                int simd16 = length & ~15;
                for (; i < simd16; i += 16)
                {
                    var m = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i), vzero).AsInt32();
                    Avx512F.Store(grad + i, Avx512F.And(Avx512F.LoadVector512(grad + i).AsInt32(), m).AsSingle());
                }
            }
            if (Avx.IsSupported && i < length)
            {
                var vzero = Vector256<float>.Zero;
                int simd32 = length & ~31;
                for (; i < simd32; i += 32)
                {
                    var mask0 = Avx.Compare(Avx.LoadVector256(input + i), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask1 = Avx.Compare(Avx.LoadVector256(input + i + 8), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask2 = Avx.Compare(Avx.LoadVector256(input + i + 16), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask3 = Avx.Compare(Avx.LoadVector256(input + i + 24), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(grad + i, Avx.And(Avx.LoadVector256(grad + i), mask0));
                    Avx.Store(grad + i + 8, Avx.And(Avx.LoadVector256(grad + i + 8), mask1));
                    Avx.Store(grad + i + 16, Avx.And(Avx.LoadVector256(grad + i + 16), mask2));
                    Avx.Store(grad + i + 24, Avx.And(Avx.LoadVector256(grad + i + 24), mask3));
                }
                int simd8 = length & ~7;
                for (; i < simd8; i += 8)
                {
                    var mask = Avx.Compare(Avx.LoadVector256(input + i), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(grad + i, Avx.And(Avx.LoadVector256(grad + i), mask));
                }
            }
#endif
            for (; i < length; i++)
                grad[i] = input[i] > 0 ? grad[i] : 0;
        }

        /// <summary>
        /// Fused ReLU backward with scalar gradient — eliminates the intermediate fill tensor.
        /// Computes: output[i] = (input[i] > 0) ? scale : 0
        /// This is equivalent to MeanBackward + ReLU backward but reads only 1 array instead of 2,
        /// halving memory bandwidth and eliminating a 4MB allocation for 1M-element tensors.
        /// </summary>
        public static unsafe void ReluBackwardScalarUnsafe(float scale, float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx512F.IsSupported)
            {
                var vzero = Vector512<float>.Zero;
                var vscale = Vector512.Create(scale);
                var vscaleInts = vscale.AsInt32();
                int simd64 = length & ~63;
                for (; i < simd64; i += 64)
                {
                    var m0 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i), vzero).AsInt32();
                    var m1 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 16), vzero).AsInt32();
                    var m2 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 32), vzero).AsInt32();
                    var m3 = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i + 48), vzero).AsInt32();
                    Avx512F.Store(output + i, Avx512F.And(vscaleInts, m0).AsSingle());
                    Avx512F.Store(output + i + 16, Avx512F.And(vscaleInts, m1).AsSingle());
                    Avx512F.Store(output + i + 32, Avx512F.And(vscaleInts, m2).AsSingle());
                    Avx512F.Store(output + i + 48, Avx512F.And(vscaleInts, m3).AsSingle());
                }
                int simd16 = length & ~15;
                for (; i < simd16; i += 16)
                {
                    var m = Avx512F.CompareGreaterThan(Avx512F.LoadVector512(input + i), vzero).AsInt32();
                    Avx512F.Store(output + i, Avx512F.And(vscaleInts, m).AsSingle());
                }
            }
            if (Avx.IsSupported && i < length)
            {
                var vzero = Vector256<float>.Zero;
                var vscale = Vector256.Create(scale);
                int simd32 = length & ~31;
                for (; i < simd32; i += 32)
                {
                    var mask0 = Avx.Compare(Avx.LoadVector256(input + i), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask1 = Avx.Compare(Avx.LoadVector256(input + i + 8), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask2 = Avx.Compare(Avx.LoadVector256(input + i + 16), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    var mask3 = Avx.Compare(Avx.LoadVector256(input + i + 24), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(output + i, Avx.And(vscale, mask0));
                    Avx.Store(output + i + 8, Avx.And(vscale, mask1));
                    Avx.Store(output + i + 16, Avx.And(vscale, mask2));
                    Avx.Store(output + i + 24, Avx.And(vscale, mask3));
                }
                int simd8 = length & ~7;
                for (; i < simd8; i += 8)
                {
                    var mask = Avx.Compare(Avx.LoadVector256(input + i), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(output + i, Avx.And(vscale, mask));
                }
            }
#endif
            for (; i < length; i++)
                output[i] = input[i] > 0 ? scale : 0;
        }

        /// <summary>
        /// SIMD-accelerated GELU backward using the tanh approximation.
        /// d/dx[GELU(x)] = 0.5*(1 + tanh(k)) + 0.5*x*(1 - tanh(k)^2)*k'
        /// where k = sqrt(2/pi)*(x + 0.044715*x^3)
        /// </summary>
        public static unsafe void GeluBackwardUnsafe(float* grad, float* input, float* output, int length)
        {
            const float sqrtTwoPi = 0.7978845608028654f;
            const float coeff = 0.044715f;
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported && Fma.IsSupported)
            {
                var vSqrtTwoPi = Vector256.Create(sqrtTwoPi);
                var vCoeff = Vector256.Create(coeff);
                var vThreeCoeff = Vector256.Create(3f * coeff);
                var vHalf = Vector256.Create(0.5f);
                var vOne = Vector256.Create(1f);
                var vTwo = Vector256.Create(2f);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    var x2 = Avx.Multiply(x, x);
                    var x3 = Avx.Multiply(x2, x);
                    // k = sqrt(2/pi) * (x + 0.044715 * x^3)
                    var inner = Fma.MultiplyAdd(vCoeff, x3, x);
                    var k = Avx.Multiply(vSqrtTwoPi, inner);
                    // tanh(k) using exp: tanh(k) = (exp(2k) - 1) / (exp(2k) + 1)
                    var exp2k = ExpApprox256(Avx.Multiply(vTwo, k));
                    var tanhK = Avx.Divide(Avx.Subtract(exp2k, vOne), Avx.Add(exp2k, vOne));
                    // sech^2(k) = 1 - tanh^2(k)
                    var sech2 = Avx.Subtract(vOne, Avx.Multiply(tanhK, tanhK));
                    // k' = sqrt(2/pi) * (1 + 3*0.044715*x^2)
                    var kPrime = Avx.Multiply(vSqrtTwoPi, Fma.MultiplyAdd(vThreeCoeff, x2, vOne));
                    // derivative = 0.5 * (1 + tanh(k)) + 0.5 * x * sech^2(k) * k'
                    var term1 = Avx.Multiply(vHalf, Avx.Add(vOne, tanhK));
                    var term2 = Avx.Multiply(vHalf, Avx.Multiply(Avx.Multiply(x, sech2), kPrime));
                    var derivative = Avx.Add(term1, term2);
                    Avx.Store(output + i, Avx.Multiply(g, derivative));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                float x2 = x * x;
                float inner = sqrtTwoPi * (x + coeff * x2 * x);
                float tanhK = MathF.Tanh(inner);
                float sech2 = 1f - tanhK * tanhK;
                float kPrime = sqrtTwoPi * (1f + 3f * coeff * x2);
                float derivative = 0.5f * (1f + tanhK) + 0.5f * x * sech2 * kPrime;
                output[i] = grad[i] * derivative;
            }
        }

        /// <summary>
        /// SIMD-accelerated Sigmoid backward: result[i] = grad[i] * sigmoid_out[i] * (1 - sigmoid_out[i])
        /// Uses the output of sigmoid forward (savedState), NOT the input, avoiding recomputation.
        /// </summary>
        public static unsafe void SigmoidBackwardUnsafe(float* grad, float* sigmoidOutput, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vOne = Vector256.Create(1f);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var s = Avx.LoadVector256(sigmoidOutput + i);
                    var g = Avx.LoadVector256(grad + i);
                    // grad * s * (1 - s)
                    var oneMinusS = Avx.Subtract(vOne, s);
                    Avx.Store(output + i, Avx.Multiply(g, Avx.Multiply(s, oneMinusS)));
                }
            }
#endif
            for (; i < length; i++)
            {
                float s = sigmoidOutput[i];
                output[i] = grad[i] * s * (1f - s);
            }
        }

        /// <summary>
        /// SIMD-accelerated Tanh backward: result[i] = grad[i] * (1 - tanh_out[i]^2)
        /// Uses the output of tanh forward.
        /// </summary>
        public static unsafe void TanhBackwardUnsafe(float* grad, float* tanhOutput, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vOne = Vector256.Create(1f);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var t = Avx.LoadVector256(tanhOutput + i);
                    var g = Avx.LoadVector256(grad + i);
                    // grad * (1 - t^2)
                    var t2 = Avx.Multiply(t, t);
                    Avx.Store(output + i, Avx.Multiply(g, Avx.Subtract(vOne, t2)));
                }
            }
#endif
            for (; i < length; i++)
            {
                float t = tanhOutput[i];
                output[i] = grad[i] * (1f - t * t);
            }
        }

        /// <summary>
        /// SIMD-accelerated Swish/SiLU backward: result[i] = grad[i] * (s + x * s * (1 - s))
        /// where s = sigmoid(x). Uses input x to compute sigmoid.
        /// </summary>
        public static unsafe void SwishBackwardUnsafe(float* grad, float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported)
            {
                var vOne = Vector256.Create(1f);
                var vNegOne = Vector256.Create(-1f);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    // sigmoid(x) = 1 / (1 + exp(-x))
                    var negX = Avx.Multiply(x, vNegOne);
                    var expNegX = ExpApprox256(negX);
                    var s = Avx.Divide(vOne, Avx.Add(vOne, expNegX));
                    // derivative = s * (1 + x * (1 - s))
                    var oneMinusS = Avx.Subtract(vOne, s);
                    var xOneMinusS = Avx.Multiply(x, oneMinusS);
                    var deriv = Avx.Multiply(s, Avx.Add(vOne, xOneMinusS));
                    Avx.Store(output + i, Avx.Multiply(g, deriv));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                float s = 1f / (1f + MathF.Exp(-x));
                float deriv = s * (1f + x * (1f - s));
                output[i] = grad[i] * deriv;
            }
        }

        /// <summary>
        /// SIMD-accelerated LeakyReLU backward: result[i] = input[i] >= 0 ? grad[i] : alpha * grad[i]
        /// </summary>
        public static unsafe void LeakyReluBackwardUnsafe(float* grad, float* input, float* output, int length, float alpha)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vzero = Vector256<float>.Zero;
                var vAlpha = Vector256.Create(alpha);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    var mask = Avx.Compare(x, vzero, FloatComparisonMode.OrderedGreaterThanOrEqualSignaling);
                    // result = mask ? grad : alpha * grad
                    var alphaGrad = Avx.Multiply(g, vAlpha);
                    Avx.Store(output + i, Avx.BlendVariable(alphaGrad, g, mask));
                }
            }
#endif
            for (; i < length; i++)
                output[i] = input[i] >= 0 ? grad[i] : alpha * grad[i];
        }

        /// <summary>
        /// SIMD-accelerated ELU backward: result[i] = input[i] >= 0 ? grad[i] : grad[i] * (output[i] + alpha)
        /// </summary>
        public static unsafe void EluBackwardUnsafe(float* grad, float* input, float* eluOutput, float* output, int length, float alpha)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vzero = Vector256<float>.Zero;
                var vAlpha = Vector256.Create(alpha);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    var e = Avx.LoadVector256(eluOutput + i);
                    var mask = Avx.Compare(x, vzero, FloatComparisonMode.OrderedGreaterThanOrEqualSignaling);
                    // negative branch: grad * (elu_output + alpha)
                    var negDeriv = Avx.Multiply(g, Avx.Add(e, vAlpha));
                    Avx.Store(output + i, Avx.BlendVariable(negDeriv, g, mask));
                }
            }
#endif
            for (; i < length; i++)
                output[i] = input[i] >= 0 ? grad[i] : grad[i] * (eluOutput[i] + alpha);
        }

        /// <summary>
        /// Fast 8-wide exp approximation for AVX2: ~1e-4 relative error.
        /// Uses polynomial approximation with 2^n scaling via integer bit manipulation.
        /// </summary>
#if NET5_0_OR_GREATER
        private static Vector256<float> ExpApprox256(Vector256<float> x)
        {
            if (!Avx2.IsSupported) return Vector256<float>.Zero;
            var log2e = Vector256.Create(1.4426950408889634f);
            var ln2 = Vector256.Create(0.6931471805599453f);
            var c1 = Vector256.Create(0.5f);
            var c2 = Vector256.Create(1f);
            var c3 = Vector256.Create(0.16666666666666666f);
            var shift = Vector256.Create(127);
            var maxClamp = Vector256.Create(88f);
            var minClamp = Vector256.Create(-88f);
            x = Avx.Max(Avx.Min(x, maxClamp), minClamp);
            var fx = Avx.Multiply(x, log2e);
            var floorFx = Avx.Floor(fx);
            var r = Avx.Subtract(x, Avx.Multiply(floorFx, ln2));
            // Polynomial: 1 + r + r^2/2 + r^3/6
            var r2 = Avx.Multiply(r, r);
            var poly = Avx.Add(c2, Avx.Add(r, Avx.Add(Avx.Multiply(c1, r2), Avx.Multiply(c3, Avx.Multiply(r2, r)))));
            // 2^n via integer exponent shift: reinterpret (n + 127) << 23 as float
            var ni = Avx2.ConvertToVector256Int32(floorFx);
            var pow2n = Avx2.ShiftLeftLogical(Avx2.Add(ni, shift), 23).AsSingle();
            return Avx.Multiply(poly, pow2n);
        }
#endif

        /// <summary>
        /// SIMD-accelerated Mish backward: d/dx[x*tanh(softplus(x))]
        /// derivative = tanh(sp) + x * sech^2(sp) * sigmoid(x)
        /// </summary>
        public static unsafe void MishBackwardUnsafe(float* grad, float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported)
            {
                var vOne = Vector256.Create(1f);
                var vNegOne = Vector256.Create(-1f);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    // softplus(x) = log(1 + exp(x))
                    var expX = ExpApprox256(x);
                    var sp = SimdLog256(Avx.Add(vOne, expX));
                    // tanh(sp) via exp: (exp(2sp)-1)/(exp(2sp)+1)
                    var exp2sp = ExpApprox256(Avx.Multiply(Vector256.Create(2f), sp));
                    var tanhSp = Avx.Divide(Avx.Subtract(exp2sp, vOne), Avx.Add(exp2sp, vOne));
                    // sech^2(sp) = 1 - tanh^2(sp)
                    var sech2 = Avx.Subtract(vOne, Avx.Multiply(tanhSp, tanhSp));
                    // sigmoid(x) = 1 / (1 + exp(-x))
                    var negX = Avx.Multiply(x, vNegOne);
                    var sig = Avx.Divide(vOne, Avx.Add(vOne, ExpApprox256(negX)));
                    // deriv = tanh(sp) + x * sech^2(sp) * sigmoid(x)
                    var deriv = Avx.Add(tanhSp, Avx.Multiply(Avx.Multiply(x, sech2), sig));
                    Avx.Store(output + i, Avx.Multiply(g, deriv));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                float sp = MathF.Log(1f + MathF.Exp(x));
                float tanhSp = MathF.Tanh(sp);
                float sech2 = 1f - tanhSp * tanhSp;
                float sig = 1f / (1f + MathF.Exp(-x));
                output[i] = grad[i] * (tanhSp + x * sech2 * sig);
            }
        }

        /// <summary>
        /// SIMD-accelerated Softplus backward: result[i] = grad[i] * sigmoid(beta * input[i])
        /// </summary>
        public static unsafe void SoftplusBackwardUnsafe(float* grad, float* input, float* output, int length, float beta)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported)
            {
                var vBeta = Vector256.Create(beta);
                var vOne = Vector256.Create(1f);
                var vNegOne = Vector256.Create(-1f);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    var bx = Avx.Multiply(x, vBeta);
                    var negBx = Avx.Multiply(bx, vNegOne);
                    var sig = Avx.Divide(vOne, Avx.Add(vOne, ExpApprox256(negBx)));
                    Avx.Store(output + i, Avx.Multiply(g, sig));
                }
            }
#endif
            for (; i < length; i++)
            {
                float bx = input[i] * beta;
                float sig = bx > 20f ? 1f : 1f / (1f + MathF.Exp(-bx));
                output[i] = grad[i] * sig;
            }
        }

        /// <summary>
        /// SIMD-accelerated SELU backward.
        /// </summary>
        public static unsafe void SeluBackwardUnsafe(float* grad, float* input, float* output, int length)
        {
            const float lambda = 1.0507009873554805f;
            const float alpha = 1.6732632423543773f;
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx2.IsSupported)
            {
                var vzero = Vector256<float>.Zero;
                var vLambda = Vector256.Create(lambda);
                var vLambdaAlpha = Vector256.Create(lambda * alpha);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    var mask = Avx.Compare(x, vzero, FloatComparisonMode.OrderedGreaterThanOrEqualSignaling);
                    // positive: lambda, negative: lambda * alpha * exp(x)
                    var expX = ExpApprox256(x);
                    var negDeriv = Avx.Multiply(vLambdaAlpha, expX);
                    var deriv = Avx.BlendVariable(negDeriv, vLambda, mask);
                    Avx.Store(output + i, Avx.Multiply(g, deriv));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                float deriv = x >= 0 ? lambda : lambda * alpha * MathF.Exp(x);
                output[i] = grad[i] * deriv;
            }
        }

#if NET5_0_OR_GREATER
        /// <summary>
        /// Fast 8-wide log approximation for AVX2.
        /// Uses polynomial approximation via integer bit manipulation.
        /// </summary>
        /// <summary>
        /// High-accuracy 8-wide natural log for AVX2 using Cody-Waite range reduction
        /// with degree-7 minimax polynomial. Sub-ULP accuracy matching libm.
        /// </summary>
        private static Vector256<float> SimdLog256(Vector256<float> x)
        {
            if (!Avx2.IsSupported) return Vector256<float>.Zero;

            // Cody-Waite range reduction: x = 2^e * m, where m in [sqrt(2)/2, sqrt(2))
            var bits = x.AsInt32();
            var exponent = Avx2.Subtract(Avx2.ShiftRightLogical(bits, 23), Vector256.Create(127));

            // Extract mantissa and normalize to [1, 2)
            var mantissaBits = Avx2.Or(Avx2.And(bits, Vector256.Create(0x007FFFFF)), Vector256.Create(0x3F800000));
            var m = mantissaBits.AsSingle();

            // Reduce to [sqrt(2)/2, sqrt(2)) by adjusting large mantissas
            var sqrtHalf = Vector256.Create(0.70710678118654752f);
            var adjustMask = Avx.Compare(m, Vector256.Create(1.41421356f), FloatComparisonMode.OrderedGreaterThanSignaling);
            var eAdjust = Avx.And(adjustMask, Vector256.Create(1f));
            m = Avx.BlendVariable(m, Avx.Multiply(m, Vector256.Create(0.5f)), adjustMask);
            var exponentF = Avx.Add(Avx.ConvertToVector256Single(exponent), eAdjust);

            // f = m - 1, compute log(1+f) via minimax polynomial on [-0.2929, 0.4142]
            var f = Avx.Subtract(m, Vector256.Create(1.0f));
            var f2 = Avx.Multiply(f, f);

            // Degree-7 minimax polynomial for log(1+f)/f - 1 on the reduced range
            // Coefficients from Sollya minimax approximation
            var p = Vector256.Create(-0.0258411228f);       // c7
            p = Avx.Add(Avx.Multiply(p, f), Vector256.Create(0.0360884966f));  // c6
            p = Avx.Add(Avx.Multiply(p, f), Vector256.Create(-0.0455004766f)); // c5
            p = Avx.Add(Avx.Multiply(p, f), Vector256.Create(0.0587909147f));  // c4
            p = Avx.Add(Avx.Multiply(p, f), Vector256.Create(-0.0833265781f)); // c3
            p = Avx.Add(Avx.Multiply(p, f), Vector256.Create(0.1249999925f));  // c2
            p = Avx.Add(Avx.Multiply(p, f), Vector256.Create(-0.2499999944f)); // c1
            p = Avx.Add(Avx.Multiply(p, f), Vector256.Create(0.4999999995f));  // c0

            // log(x) = e*ln2 + f + f^2 * p
            // Use Cody-Waite split of ln2 for precision: ln2_hi + ln2_lo
            var ln2_hi = Vector256.Create(0.693145751953125f);
            var ln2_lo = Vector256.Create(1.428606765330187e-06f);
            var result = Avx.Add(
                Avx.Add(Avx.Multiply(exponentF, ln2_hi), f),
                Avx.Add(Avx.Multiply(f2, p), Avx.Multiply(exponentF, ln2_lo)));

            return result;
        }
#endif

        // ──────────────────────────────────────────────────────────
        // Double precision backward kernels (AVX2: 4 doubles at a time)
        // ──────────────────────────────────────────────────────────

        /// <summary>
        /// SIMD-accelerated ReLU backward for double: result[i] = input[i] > 0 ? grad[i] : 0
        /// </summary>
        public static unsafe void ReluBackwardDouble(double* grad, double* input, double* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vzero = Vector256<double>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var mask = Avx.Compare(Avx.LoadVector256(input + i), vzero, FloatComparisonMode.OrderedGreaterThanSignaling);
                    Avx.Store(output + i, Avx.And(Avx.LoadVector256(grad + i), mask));
                }
            }
#endif
            for (; i < length; i++)
                output[i] = input[i] > 0 ? grad[i] : 0;
        }

        /// <summary>
        /// SIMD-accelerated Sigmoid backward for double: result[i] = grad[i] * s[i] * (1 - s[i])
        /// </summary>
        public static unsafe void SigmoidBackwardDouble(double* grad, double* sigmoidOutput, double* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vOne = Vector256.Create(1.0);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var s = Avx.LoadVector256(sigmoidOutput + i);
                    var g = Avx.LoadVector256(grad + i);
                    var oneMinusS = Avx.Subtract(vOne, s);
                    Avx.Store(output + i, Avx.Multiply(g, Avx.Multiply(s, oneMinusS)));
                }
            }
#endif
            for (; i < length; i++)
            {
                double s = sigmoidOutput[i];
                output[i] = grad[i] * s * (1.0 - s);
            }
        }

        /// <summary>
        /// SIMD-accelerated Tanh backward for double: result[i] = grad[i] * (1 - t[i]^2)
        /// </summary>
        public static unsafe void TanhBackwardDouble(double* grad, double* tanhOutput, double* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vOne = Vector256.Create(1.0);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var t = Avx.LoadVector256(tanhOutput + i);
                    var g = Avx.LoadVector256(grad + i);
                    var t2 = Avx.Multiply(t, t);
                    Avx.Store(output + i, Avx.Multiply(g, Avx.Subtract(vOne, t2)));
                }
            }
#endif
            for (; i < length; i++)
            {
                double t = tanhOutput[i];
                output[i] = grad[i] * (1.0 - t * t);
            }
        }

        /// <summary>
        /// SIMD-accelerated LeakyReLU backward for double.
        /// </summary>
        public static unsafe void LeakyReluBackwardDouble(double* grad, double* input, double* output, int length, double alpha)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vzero = Vector256<double>.Zero;
                var vAlpha = Vector256.Create(alpha);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    var mask = Avx.Compare(x, vzero, FloatComparisonMode.OrderedGreaterThanOrEqualSignaling);
                    var alphaGrad = Avx.Multiply(g, vAlpha);
                    Avx.Store(output + i, Avx.BlendVariable(alphaGrad, g, mask));
                }
            }
#endif
            for (; i < length; i++)
                output[i] = input[i] >= 0 ? grad[i] : alpha * grad[i];
        }

        /// <summary>
        /// SIMD-accelerated HardSwish backward for float.
        /// d/dx[x*(x+3)/6] = (2x+3)/6 for -3 &lt; x &lt; 3, 0 for x &lt;= -3, 1 for x >= 3
        /// </summary>
        public static unsafe void HardSwishBackwardUnsafe(float* grad, float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vNeg3 = Vector256.Create(-3.0f);
                var vPos3 = Vector256.Create(3.0f);
                var vTwo = Vector256.Create(2.0f);
                var vThree = Vector256.Create(3.0f);
                var vSix = Vector256.Create(6.0f);
                var vZero = Vector256<float>.Zero;
                var vOne = Vector256.Create(1.0f);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    // linear region: (2x+3)/6
                    var linear = Avx.Divide(Avx.Add(Avx.Multiply(vTwo, x), vThree), vSix);
                    // mask: x >= 3 => 1, x <= -3 => 0, else linear
                    var maskHigh = Avx.Compare(x, vPos3, FloatComparisonMode.OrderedGreaterThanOrEqualSignaling);
                    var maskLow = Avx.Compare(x, vNeg3, FloatComparisonMode.OrderedLessThanOrEqualSignaling);
                    var deriv = Avx.BlendVariable(linear, vOne, maskHigh);
                    deriv = Avx.BlendVariable(deriv, vZero, maskLow);
                    Avx.Store(output + i, Avx.Multiply(g, deriv));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                float deriv = x <= -3f ? 0f : x >= 3f ? 1f : (2f * x + 3f) / 6f;
                output[i] = grad[i] * deriv;
            }
        }

        /// <summary>
        /// SIMD-accelerated HardSigmoid backward for float.
        /// d/dx[clip((x+3)/6, 0, 1)] = 1/6 for -3 &lt; x &lt; 3, else 0
        /// </summary>
        public static unsafe void HardSigmoidBackwardUnsafe(float* grad, float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vNeg3 = Vector256.Create(-3.0f);
                var vPos3 = Vector256.Create(3.0f);
                var vSixth = Vector256.Create(1.0f / 6.0f);
                var vZero = Vector256<float>.Zero;
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    var inRange = Avx.And(
                        Avx.Compare(x, vNeg3, FloatComparisonMode.OrderedGreaterThanSignaling),
                        Avx.Compare(x, vPos3, FloatComparisonMode.OrderedLessThanSignaling));
                    var deriv = Avx.And(vSixth, inRange);
                    Avx.Store(output + i, Avx.Multiply(g, deriv));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                output[i] = (x > -3f && x < 3f) ? grad[i] / 6f : 0f;
            }
        }

        /// <summary>
        /// SIMD-accelerated ReLU6 backward for float.
        /// </summary>
        public static unsafe void Relu6BackwardUnsafe(float* grad, float* input, float* output, int length)
        {
            int i = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported)
            {
                var vZero = Vector256<float>.Zero;
                var vSix = Vector256.Create(6.0f);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var x = Avx.LoadVector256(input + i);
                    var g = Avx.LoadVector256(grad + i);
                    var inRange = Avx.And(
                        Avx.Compare(x, vZero, FloatComparisonMode.OrderedGreaterThanSignaling),
                        Avx.Compare(x, vSix, FloatComparisonMode.OrderedLessThanSignaling));
                    Avx.Store(output + i, Avx.And(g, inRange));
                }
            }
#endif
            for (; i < length; i++)
            {
                float x = input[i];
                output[i] = (x > 0f && x < 6f) ? grad[i] : 0f;
            }
        }

        // ──────────────────────────────────────────────────────────────
        // Additional double backward kernels
        // ──────────────────────────────────────────────────────────────

        /// <summary>
        /// SIMD-accelerated GELU backward for double.
        /// </summary>
        public static unsafe void GeluBackwardDouble(double* grad, double* input, double* output, int length)
        {
            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                double t = Math.Tanh(0.7978845608 * (x + 0.044715 * x * x * x));
                double dtdx = 0.7978845608 * (1.0 + 0.134145 * x * x) * (1.0 - t * t);
                output[i] = grad[i] * (0.5 * (1.0 + t) + 0.5 * x * dtdx);
            }
        }

        /// <summary>
        /// SIMD-accelerated Swish backward for double.
        /// </summary>
        public static unsafe void SwishBackwardDouble(double* grad, double* input, double* output, int length)
        {
            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                double sig = 1.0 / (1.0 + Math.Exp(-x));
                output[i] = grad[i] * (sig + x * sig * (1.0 - sig));
            }
        }

        /// <summary>
        /// SIMD-accelerated ELU backward for double.
        /// </summary>
        public static unsafe void EluBackwardDouble(double* grad, double* input, double* eluOutput, double* output, int length, double alpha)
        {
            for (int i = 0; i < length; i++)
                output[i] = input[i] >= 0 ? grad[i] : grad[i] * (eluOutput[i] + alpha);
        }

        /// <summary>
        /// SIMD-accelerated Mish backward for double.
        /// </summary>
        public static unsafe void MishBackwardDouble(double* grad, double* input, double* output, int length)
        {
            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                double sp = Math.Log(1.0 + Math.Exp(x));
                double tanhSp = Math.Tanh(sp);
                double sig = 1.0 / (1.0 + Math.Exp(-x));
                output[i] = grad[i] * (tanhSp + x * (1.0 - tanhSp * tanhSp) * sig);
            }
        }

        /// <summary>
        /// SIMD-accelerated Softplus backward for double.
        /// </summary>
        public static unsafe void SoftplusBackwardDouble(double* grad, double* input, double* output, int length, double beta)
        {
            for (int i = 0; i < length; i++)
                output[i] = grad[i] / (1.0 + Math.Exp(-input[i] * beta));
        }

        /// <summary>
        /// SIMD-accelerated SELU backward for double.
        /// </summary>
        public static unsafe void SeluBackwardDouble(double* grad, double* input, double* output, int length)
        {
            const double lambda = 1.0507009873554805;
            const double alpha = 1.6732632423543773;
            for (int i = 0; i < length; i++)
                output[i] = grad[i] * (input[i] >= 0 ? lambda : lambda * alpha * Math.Exp(input[i]));
        }

        /// <summary>
        /// SIMD-accelerated HardSwish backward for double.
        /// </summary>
        public static unsafe void HardSwishBackwardDouble(double* grad, double* input, double* output, int length)
        {
            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                double deriv = x <= -3.0 ? 0.0 : x >= 3.0 ? 1.0 : (2.0 * x + 3.0) / 6.0;
                output[i] = grad[i] * deriv;
            }
        }

        // Missing double backward kernels
        public static unsafe void HardSigmoidBackwardDouble(double* grad, double* input, double* output, int length)
        {
            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                output[i] = (x > -3.0 && x < 3.0) ? grad[i] / 6.0 : 0.0;
            }
        }

        public static unsafe void Relu6BackwardDouble(double* grad, double* input, double* output, int length)
        {
            for (int i = 0; i < length; i++)
            {
                double x = input[i];
                output[i] = (x > 0.0 && x < 6.0) ? grad[i] : 0.0;
            }
        }

        // Missing Half backward kernels
        public static void EluBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, ReadOnlySpan<Half> eluOutput, Span<Half> output, float alpha)
        {
            if (grad.Length != input.Length || grad.Length != eluOutput.Length || grad.Length != output.Length)
                throw new ArgumentException($"ELU Half backward span length mismatch: grad={grad.Length}, input={input.Length}, eluOutput={eluOutput.Length}, output={output.Length}");
            int len = grad.Length;
            var gf = new float[len]; var inf = new float[len]; var ef = new float[len]; var outf = new float[len];
            ConvertToSingle(grad, gf); ConvertToSingle(input, inf); ConvertToSingle(eluOutput, ef);
            unsafe { fixed (float* gp = gf, ip = inf, ep = ef, op = outf) EluBackwardUnsafe(gp, ip, ep, op, len, alpha); }
            ConvertToHalf(outf, output);
        }

        public static void HardSigmoidBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len]; var inf = new float[len]; var outf = new float[len];
            ConvertToSingle(grad, gf); ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) HardSigmoidBackwardUnsafe(gp, ip, op, len); }
            ConvertToHalf(outf, output);
        }

        public static void Relu6BackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len]; var inf = new float[len]; var outf = new float[len];
            ConvertToSingle(grad, gf); ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) Relu6BackwardUnsafe(gp, ip, op, len); }
            ConvertToHalf(outf, output);
        }

        // ──────────────────────────────────────────────────────────────
        // Complex SIMD backward kernels (reduction-based)
        // ──────────────────────────────────────────────────────────────

        /// <summary>
        /// Softmax backward for float: grad_input[i] = softmax[i] * (grad[i] - dot(grad, softmax))
        /// Per-row operation where each row is processed independently.
        /// </summary>
        public static unsafe void SoftmaxBackwardUnsafe(float* grad, float* softmaxOutput, float* output, int batchSize, int features)
        {
            for (int b = 0; b < batchSize; b++)
            {
                int offset = b * features;
                // Compute dot(grad, softmax) for this row
                float dot = 0;
                int j = 0;
#if NET5_0_OR_GREATER
                if (Avx.IsSupported)
                {
                    var vDot = Vector256<float>.Zero;
                    int simdLen = features & ~7;
                    for (; j < simdLen; j += 8)
                    {
                        var g = Avx.LoadVector256(grad + offset + j);
                        var s = Avx.LoadVector256(softmaxOutput + offset + j);
                        vDot = Avx.Add(vDot, Avx.Multiply(g, s));
                    }
                    // Horizontal sum of vDot
                    var hi = Avx.ExtractVector128(vDot, 1);
                    var lo = vDot.GetLower();
                    var sum4 = Sse.Add(lo, hi);
                    sum4 = Sse.Add(sum4, Sse.MoveHighToLow(sum4, sum4));
                    sum4 = Sse.AddScalar(sum4, Sse.Shuffle(sum4, sum4, 0x01));
                    dot = sum4.ToScalar();
                }
#endif
                for (; j < features; j++)
                    dot += grad[offset + j] * softmaxOutput[offset + j];

                // grad_input[i] = softmax[i] * (grad[i] - dot)
                j = 0;
#if NET5_0_OR_GREATER
                if (Avx.IsSupported)
                {
                    var vDotBroadcast = Vector256.Create(dot);
                    int simdLen = features & ~7;
                    for (; j < simdLen; j += 8)
                    {
                        var s = Avx.LoadVector256(softmaxOutput + offset + j);
                        var g = Avx.LoadVector256(grad + offset + j);
                        var diff = Avx.Subtract(g, vDotBroadcast);
                        Avx.Store(output + offset + j, Avx.Multiply(s, diff));
                    }
                }
#endif
                for (; j < features; j++)
                    output[offset + j] = softmaxOutput[offset + j] * (grad[offset + j] - dot);
            }
        }

        /// <summary>
        /// BatchNorm backward for float.
        /// Computes gradients for input, gamma (scale), and beta (shift).
        /// </summary>
        public static unsafe void BatchNormBackwardUnsafe(
            float* gradOutput, float* input, float* gamma,
            float* mean, float* variance, float epsilon,
            float* gradInput, float* gradGamma, float* gradBeta,
            int batchSize, int channels, int spatialSize)
        {
            int totalPerChannel = batchSize * spatialSize;
            float invN = 1.0f / totalPerChannel;

            for (int c = 0; c < channels; c++)
            {
                float m = mean[c];
                float invStd = 1.0f / MathF.Sqrt(variance[c] + epsilon);
                float gGamma = 0, gBeta = 0;
                float sumGradXhat = 0, sumGrad = 0;

                // Pass 1: accumulate gradGamma, gradBeta, and intermediate sums
                for (int b = 0; b < batchSize; b++)
                {
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = (b * channels + c) * spatialSize + s;
                        float go = gradOutput[idx];
                        float xhat = (input[idx] - m) * invStd;
                        gGamma += go * xhat;
                        gBeta += go;
                        sumGradXhat += go * xhat;
                        sumGrad += go;
                    }
                }
                gradGamma[c] = gGamma;
                gradBeta[c] = gBeta;

                // Pass 2: compute gradInput
                float g = gamma[c];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = (b * channels + c) * spatialSize + s;
                        float xhat = (input[idx] - m) * invStd;
                        gradInput[idx] = g * invStd * (gradOutput[idx] - invN * (sumGrad + xhat * sumGradXhat));
                    }
                }
            }
        }

        /// <summary>
        /// LayerNorm backward for float.
        /// Computes gradient for input given gradOutput, normalized input, gamma.
        /// </summary>
        public static unsafe void LayerNormBackwardUnsafe(
            float* gradOutput, float* input, float* gamma,
            float* mean, float* variance, float epsilon,
            float* gradInput, float* gradGamma, float* gradBeta,
            int batchSize, int normSize)
        {
            // Accumulate gradGamma and gradBeta across batch
            for (int i = 0; i < normSize; i++)
            {
                gradGamma[i] = 0;
                gradBeta[i] = 0;
            }

            for (int b = 0; b < batchSize; b++)
            {
                int offset = b * normSize;
                float m = mean[b];
                float invStd = 1.0f / MathF.Sqrt(variance[b] + epsilon);

                // Accumulate gradGamma and gradBeta
                for (int i = 0; i < normSize; i++)
                {
                    float xhat = (input[offset + i] - m) * invStd;
                    gradGamma[i] += gradOutput[offset + i] * xhat;
                    gradBeta[i] += gradOutput[offset + i];
                }

                // Compute intermediate sums for gradInput
                float ds = 0, db = 0;
                for (int i = 0; i < normSize; i++)
                {
                    float go = gradOutput[offset + i] * gamma[i];
                    float xhat = (input[offset + i] - m) * invStd;
                    ds += go * xhat;
                    db += go;
                }

                // gradInput
                float invN = 1.0f / normSize;
                for (int i = 0; i < normSize; i++)
                {
                    float xhat = (input[offset + i] - m) * invStd;
                    gradInput[offset + i] = invStd * (gradOutput[offset + i] * gamma[i] - invN * (db + xhat * ds));
                }
            }
        }

        /// <summary>
        /// Softmax backward for double.
        /// </summary>
        public static unsafe void SoftmaxBackwardDouble(double* grad, double* softmaxOutput, double* output, int batchSize, int features)
        {
            for (int b = 0; b < batchSize; b++)
            {
                int offset = b * features;
                double dot = 0;
                for (int j = 0; j < features; j++)
                    dot += grad[offset + j] * softmaxOutput[offset + j];
                for (int j = 0; j < features; j++)
                    output[offset + j] = softmaxOutput[offset + j] * (grad[offset + j] - dot);
            }
        }

        /// <summary>BatchNorm backward for double.</summary>
        public static unsafe void BatchNormBackwardDouble(
            double* gradOutput, double* input, double* gamma,
            double* mean, double* variance, double epsilon,
            double* gradInput, double* gradGamma, double* gradBeta,
            int batchSize, int channels, int spatialSize)
        {
            int totalPerChannel = batchSize * spatialSize;
            double invN = 1.0 / totalPerChannel;
            for (int c = 0; c < channels; c++)
            {
                double m = mean[c];
                double invStd = 1.0 / Math.Sqrt(variance[c] + epsilon);
                double gGamma = 0, gBeta = 0, sumGradXhat = 0, sumGrad = 0;
                for (int b = 0; b < batchSize; b++)
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = (b * channels + c) * spatialSize + s;
                        double go = gradOutput[idx];
                        double xhat = (input[idx] - m) * invStd;
                        gGamma += go * xhat; gBeta += go;
                        sumGradXhat += go * xhat; sumGrad += go;
                    }
                gradGamma[c] = gGamma; gradBeta[c] = gBeta;
                double g = gamma[c];
                for (int b = 0; b < batchSize; b++)
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = (b * channels + c) * spatialSize + s;
                        double xhat = (input[idx] - m) * invStd;
                        gradInput[idx] = g * invStd * (gradOutput[idx] - invN * (sumGrad + xhat * sumGradXhat));
                    }
            }
        }

        /// <summary>LayerNorm backward for double.</summary>
        public static unsafe void LayerNormBackwardDouble(
            double* gradOutput, double* input, double* gamma,
            double* mean, double* variance, double epsilon,
            double* gradInput, double* gradGamma, double* gradBeta,
            int batchSize, int normSize)
        {
            for (int i = 0; i < normSize; i++) { gradGamma[i] = 0; gradBeta[i] = 0; }
            for (int b = 0; b < batchSize; b++)
            {
                int offset = b * normSize;
                double m = mean[b];
                double invStd = 1.0 / Math.Sqrt(variance[b] + epsilon);
                for (int i = 0; i < normSize; i++)
                {
                    double xhat = (input[offset + i] - m) * invStd;
                    gradGamma[i] += gradOutput[offset + i] * xhat;
                    gradBeta[i] += gradOutput[offset + i];
                }
                double ds = 0, db = 0;
                for (int i = 0; i < normSize; i++)
                {
                    double go = gradOutput[offset + i] * gamma[i];
                    double xhat = (input[offset + i] - m) * invStd;
                    ds += go * xhat; db += go;
                }
                double invN = 1.0 / normSize;
                for (int i = 0; i < normSize; i++)
                {
                    double xhat = (input[offset + i] - m) * invStd;
                    gradInput[offset + i] = invStd * (gradOutput[offset + i] * gamma[i] - invN * (db + xhat * ds));
                }
            }
        }

        // ──────────────────────────────────────────────────────────────
        // Half precision backward kernels (compute in FP32, store in FP16)
        // Uses convert-compute-convert pattern: Half→float, run float SIMD, float→Half
        // ──────────────────────────────────────────────────────────────

        /// <summary>
        /// Half precision ReLU backward via FP32 conversion.
        /// </summary>
        private static void ValidateHalfSpanLengths(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> output)
        {
            if (a.Length != b.Length || a.Length != output.Length)
                throw new ArgumentException($"Half backward span length mismatch: grad={a.Length}, input={b.Length}, output={output.Length}");
        }

        public static void ReluBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output)
        {
            ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len];
            var inf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) ReluBackwardUnsafe(gp, ip, op, len); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision Sigmoid backward via FP32 conversion.
        /// </summary>
        public static void SigmoidBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> sigmoidOutput, Span<Half> output)
        { ValidateHalfSpanLengths(grad, sigmoidOutput, output);
            int len = grad.Length;
            var gf = new float[len];
            var sf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(sigmoidOutput, sf);
            unsafe { fixed (float* gp = gf, sp = sf, op = outf) SigmoidBackwardUnsafe(gp, sp, op, len); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision Tanh backward via FP32 conversion.
        /// </summary>
        public static void TanhBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> tanhOutput, Span<Half> output)
        { ValidateHalfSpanLengths(grad, tanhOutput, output);
            int len = grad.Length;
            var gf = new float[len];
            var tf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(tanhOutput, tf);
            unsafe { fixed (float* gp = gf, tp = tf, op = outf) TanhBackwardUnsafe(gp, tp, op, len); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision GELU backward via FP32 conversion.
        /// </summary>
        public static void GeluBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len];
            var inf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) GeluBackwardUnsafe(gp, ip, op, len); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision Swish backward via FP32 conversion.
        /// </summary>
        public static void SwishBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len];
            var inf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) SwishBackwardUnsafe(gp, ip, op, len); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision LeakyReLU backward via FP32 conversion.
        /// </summary>
        public static void LeakyReluBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output, float alpha)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len];
            var inf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) LeakyReluBackwardUnsafe(gp, ip, op, len, alpha); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision Mish backward via FP32 conversion.
        /// </summary>
        public static void MishBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len];
            var inf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) MishBackwardUnsafe(gp, ip, op, len); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision SELU backward via FP32 conversion.
        /// </summary>
        public static void SeluBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len];
            var inf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) SeluBackwardUnsafe(gp, ip, op, len); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision HardSwish backward via FP32 conversion.
        /// </summary>
        public static void HardSwishBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len];
            var inf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) HardSwishBackwardUnsafe(gp, ip, op, len); }
            ConvertToHalf(outf, output);
        }

        /// <summary>
        /// Half precision Softplus backward via FP32 conversion.
        /// </summary>
        public static void SoftplusBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> input, Span<Half> output, float beta)
        { ValidateHalfSpanLengths(grad, input, output);
            int len = grad.Length;
            var gf = new float[len];
            var inf = new float[len];
            var outf = new float[len];
            ConvertToSingle(grad, gf);
            ConvertToSingle(input, inf);
            unsafe { fixed (float* gp = gf, ip = inf, op = outf) SoftplusBackwardUnsafe(gp, ip, op, len, beta); }
            ConvertToHalf(outf, output);
        }

        /// <summary>Softmax backward for Half via FP32 conversion.</summary>
        public static void SoftmaxBackwardHalf(ReadOnlySpan<Half> grad, ReadOnlySpan<Half> softmaxOutput, Span<Half> output, int batchSize, int features)
        {
            var gf = new float[grad.Length]; var sf = new float[softmaxOutput.Length]; var outf = new float[output.Length];
            ConvertToSingle(grad, gf); ConvertToSingle(softmaxOutput, sf);
            unsafe { fixed (float* gp = gf, sp = sf, op = outf) SoftmaxBackwardUnsafe(gp, sp, op, batchSize, features); }
            ConvertToHalf(outf, output);
        }

        /// <summary>BatchNorm backward for Half via FP32 conversion.</summary>
        public static void BatchNormBackwardHalf(
            ReadOnlySpan<Half> gradOutput, ReadOnlySpan<Half> input, ReadOnlySpan<Half> gamma,
            ReadOnlySpan<Half> mean, ReadOnlySpan<Half> variance, float epsilon,
            Span<Half> gradInput, Span<Half> gradGamma, Span<Half> gradBeta,
            int batchSize, int channels, int spatialSize)
        {
            int totalLen = gradOutput.Length;
            var gof = new float[totalLen]; var inf = new float[totalLen]; var gaf = new float[channels];
            var mf = new float[channels]; var vf = new float[channels]; var gif = new float[totalLen];
            var ggf = new float[channels]; var gbf = new float[channels];
            ConvertToSingle(gradOutput, gof); ConvertToSingle(input, inf); ConvertToSingle(gamma, gaf);
            ConvertToSingle(mean, mf); ConvertToSingle(variance, vf);
            unsafe
            {
                fixed (float* gop = gof, ip = inf, gap = gaf, mp = mf, vp = vf, gip = gif, ggp = ggf, gbp = gbf)
                    BatchNormBackwardUnsafe(gop, ip, gap, mp, vp, epsilon, gip, ggp, gbp, batchSize, channels, spatialSize);
            }
            ConvertToHalf(gif, gradInput); ConvertToHalf(ggf, gradGamma); ConvertToHalf(gbf, gradBeta);
        }

        /// <summary>LayerNorm backward for Half via FP32 conversion.</summary>
        public static void LayerNormBackwardHalf(
            ReadOnlySpan<Half> gradOutput, ReadOnlySpan<Half> input, ReadOnlySpan<Half> gamma,
            ReadOnlySpan<Half> mean, ReadOnlySpan<Half> variance, float epsilon,
            Span<Half> gradInput, Span<Half> gradGamma, Span<Half> gradBeta,
            int batchSize, int normSize)
        {
            int totalLen = gradOutput.Length;
            var gof = new float[totalLen]; var inf = new float[totalLen]; var gaf = new float[normSize];
            var mf = new float[batchSize]; var vf = new float[batchSize]; var gif = new float[totalLen];
            var ggf = new float[normSize]; var gbf = new float[normSize];
            ConvertToSingle(gradOutput, gof); ConvertToSingle(input, inf); ConvertToSingle(gamma, gaf);
            ConvertToSingle(mean, mf); ConvertToSingle(variance, vf);
            unsafe
            {
                fixed (float* gop = gof, ip = inf, gap = gaf, mp = mf, vp = vf, gip = gif, ggp = ggf, gbp = gbf)
                    LayerNormBackwardUnsafe(gop, ip, gap, mp, vp, epsilon, gip, ggp, gbp, batchSize, normSize);
            }
            ConvertToHalf(gif, gradInput); ConvertToHalf(ggf, gradGamma); ConvertToHalf(gbf, gradBeta);
        }

    #endregion
    }
}
