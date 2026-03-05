#if !NET5_0_OR_GREATER
// ReSharper disable once CheckNamespace
namespace System
{
    /// <summary>
    /// MathF polyfill for .NET Framework 4.7.1 where System.MathF is not available.
    /// Delegates all operations to System.Math with float casts.
    /// </summary>
    internal static class MathF
    {
        public const float PI = 3.14159265f;
        public const float E = 2.71828183f;

        public static float Abs(float x) => Math.Abs(x);
        public static float Sqrt(float x) => (float)Math.Sqrt(x);
        public static float Pow(float x, float y) => (float)Math.Pow(x, y);
        public static float Exp(float x) => (float)Math.Exp(x);
        public static float Log(float x) => (float)Math.Log(x);
        public static float Log2(float x) => (float)(Math.Log(x) / Math.Log(2));
        public static float Log10(float x) => (float)Math.Log10(x);
        public static float Sin(float x) => (float)Math.Sin(x);
        public static float Cos(float x) => (float)Math.Cos(x);
        public static float Tan(float x) => (float)Math.Tan(x);
        public static float Asin(float x) => (float)Math.Asin(x);
        public static float Acos(float x) => (float)Math.Acos(x);
        public static float Atan(float x) => (float)Math.Atan(x);
        public static float Atan2(float y, float x) => (float)Math.Atan2(y, x);
        public static float Sinh(float x) => (float)Math.Sinh(x);
        public static float Cosh(float x) => (float)Math.Cosh(x);
        public static float Tanh(float x) => (float)Math.Tanh(x);
        public static float Floor(float x) => (float)Math.Floor(x);
        public static float Ceiling(float x) => (float)Math.Ceiling(x);
        public static float Round(float x) => (float)Math.Round(x);
        public static float Round(float x, int digits) => (float)Math.Round(x, digits);
        public static float Truncate(float x) => (float)Math.Truncate(x);
        public static float Max(float x, float y) => Math.Max(x, y);
        public static float Min(float x, float y) => Math.Min(x, y);
        public static int Sign(float x) => Math.Sign(x);
        public static float Cbrt(float x) => (float)Math.Pow(x, 1.0 / 3.0);

        public static (float Sin, float Cos) SinCos(float x)
        {
            return ((float)Math.Sin(x), (float)Math.Cos(x));
        }

        // Inverse hyperbolic functions
        public static float Asinh(float x) => (float)Math.Log(x + Math.Sqrt(x * x + 1));
        public static float Acosh(float x) => (float)Math.Log(x + Math.Sqrt(x * x - 1));
        public static float Atanh(float x) => (float)(0.5 * Math.Log((1 + x) / (1 - x)));

        // Exp variants
        public static float Exp2(float x) => (float)Math.Pow(2.0, x);

        // Log variants
        public static float Log1P(float x) => (float)Math.Log(1.0 + x);

        // IEEERemainder
        public static float IEEERemainder(float x, float y) => (float)Math.IEEERemainder(x, y);

        // FusedMultiplyAdd - not available on net471, emulate
        public static float FusedMultiplyAdd(float x, float y, float z) => x * y + z;

        // CopySign
        public static float CopySign(float x, float y)
        {
            return Math.Abs(x) * (y >= 0 ? 1f : -1f);
        }

        // BitDecrement/BitIncrement - approximate
        public static float BitDecrement(float x)
        {
            if (float.IsNaN(x) || float.IsNegativeInfinity(x)) return x;
            if (x == 0f) return -float.Epsilon;
            int bits = BitConverter.ToInt32(BitConverter.GetBytes(x), 0);
            bits += (x > 0) ? -1 : 1;
            return BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
        }

        public static float BitIncrement(float x)
        {
            if (float.IsNaN(x) || float.IsPositiveInfinity(x)) return x;
            if (x == 0f) return float.Epsilon;
            int bits = BitConverter.ToInt32(BitConverter.GetBytes(x), 0);
            bits += (x >= 0) ? 1 : -1;
            return BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
        }

        // ScaleB
        public static float ScaleB(float x, int n) => (float)(x * Math.Pow(2, n));

        // Reciprocal estimate
        public static float ReciprocalEstimate(float x) => 1f / x;
        public static float ReciprocalSqrtEstimate(float x) => 1f / (float)Math.Sqrt(x);
    }
}
#endif
