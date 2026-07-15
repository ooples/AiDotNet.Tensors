// Math compatibility layer for .NET Framework 4.7.1
// Provides Log2, Cbrt, Acosh, Asinh, Atanh that don't exist in older frameworks

using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Tests.TestHelpers
{
    /// <summary>
    /// Provides math functions compatible with all target frameworks.
    /// Use these instead of Math.Log2/Math.Cbrt/Math.Acosh etc which don't exist in .NET Framework.
    /// </summary>
    public static class MathCompat
    {
        private const double Log2Constant = 0.6931471805599453; // Math.Log(2)

        #region Log2

        /// <summary>
        /// Computes the base-2 logarithm of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Log2(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Log2(x);
#else
            return Math.Log(x) / Log2Constant;
#endif
        }

        /// <summary>
        /// Computes the base-2 logarithm of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Log2(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Log2(x);
#else
            return (float)(Math.Log(x) / Log2Constant);
#endif
        }

        #endregion

        #region Cbrt

        /// <summary>
        /// Computes the cube root of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Cbrt(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Cbrt(x);
#else
            if (x >= 0)
            {
                return Math.Pow(x, 1.0 / 3.0);
            }
            else
            {
                return -Math.Pow(-x, 1.0 / 3.0);
            }
#endif
        }

        /// <summary>
        /// Computes the cube root of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Cbrt(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Cbrt(x);
#else
            if (x >= 0)
            {
                return (float)Math.Pow(x, 1.0 / 3.0);
            }
            else
            {
                return (float)(-Math.Pow(-x, 1.0 / 3.0));
            }
#endif
        }

        #endregion

        #region Acosh

        /// <summary>
        /// Computes the inverse hyperbolic cosine of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Acosh(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Acosh(x);
#else
            // acosh(x) = ln(x + sqrt(x^2 - 1))
            return Math.Log(x + Math.Sqrt(x * x - 1));
#endif
        }

        /// <summary>
        /// Computes the inverse hyperbolic cosine of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Acosh(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Acosh(x);
#else
            return (float)Math.Log(x + Math.Sqrt(x * x - 1));
#endif
        }

        #endregion

        #region Asinh

        /// <summary>
        /// Computes the inverse hyperbolic sine of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Asinh(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Asinh(x);
#else
            // asinh(x) = ln(x + sqrt(x^2 + 1))
            return Math.Log(x + Math.Sqrt(x * x + 1));
#endif
        }

        /// <summary>
        /// Computes the inverse hyperbolic sine of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Asinh(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Asinh(x);
#else
            return (float)Math.Log(x + Math.Sqrt(x * x + 1));
#endif
        }

        #endregion

        #region Atanh

        /// <summary>
        /// Computes the inverse hyperbolic tangent of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Atanh(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Atanh(x);
#else
            // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
            return 0.5 * Math.Log((1 + x) / (1 - x));
#endif
        }

        /// <summary>
        /// Computes the inverse hyperbolic tangent of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Atanh(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Atanh(x);
#else
            return (float)(0.5 * Math.Log((1 + x) / (1 - x)));
#endif
        }

        #endregion

        #region Float bit helpers (net471-safe reinterpret + BitDecrement)

        /// <summary>Reinterprets a float's bits as an int (BitConverter.SingleToInt32Bits is net-core-only).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int SingleToInt32Bits(float value)
        {
#if NET5_0_OR_GREATER
            return BitConverter.SingleToInt32Bits(value);
#else
            return BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
#endif
        }

        /// <summary>Reinterprets an int's bits as a float (BitConverter.Int32BitsToSingle is net-core-only).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Int32BitsToSingle(int value)
        {
#if NET5_0_OR_GREATER
            return BitConverter.Int32BitsToSingle(value);
#else
            return BitConverter.ToSingle(BitConverter.GetBytes(value), 0);
#endif
        }

        /// <summary>Returns the largest float strictly less than <paramref name="x"/>
        /// (float.BitDecrement is net7+ generic-math only). Faithful IEEE-754 polyfill for net471.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float BitDecrement(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.BitDecrement(x);
#else
            int bits = SingleToInt32Bits(x);
            if ((bits & 0x7F800000) >= 0x7F800000)
                return bits == 0x7F800000 ? float.MaxValue : x; // +Inf -> MaxValue; NaN/-Inf -> x
            if (bits == 0)
                return -float.Epsilon;                          // +0.0 -> -epsilon
            bits += bits < 0 ? +1 : -1;                          // negative: away from zero; positive: toward zero
            return Int32BitsToSingle(bits);
#endif
        }

        #endregion
    }
}
