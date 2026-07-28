using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Target-framework compatibility shims for the direct-PTX code so the whole
/// namespace compiles on net471 as well as net8.0+/net10.0. The direct-PTX
/// kernels are emitted as plain strings and dispatched through the CUDA driver
/// P/Invoke surface, both of which are net471-compatible; only a handful of
/// modern BCL convenience APIs (bit reinterpretation, guard helpers, IsFinite,
/// hex encoding, IntPtr/nuint conversion) are missing pre-net5. These shims
/// provide those APIs uniformly on every target so the direct-PTX sources use a
/// single call form with no per-file conditional compilation.
/// </summary>
internal static class PtxCompat
{
    internal static unsafe int SingleToInt32Bits(float value) => *(int*)&value;

    internal static unsafe uint SingleToUInt32Bits(float value) => *(uint*)&value;

    internal static unsafe float Int32BitsToSingle(int value) => *(float*)&value;

    internal static unsafe float UInt32BitsToSingle(uint value) => *(float*)&value;

    internal static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    internal static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    internal static float Abs(float value) => (float)Math.Abs(value);

    internal static float Exp(float value) => (float)Math.Exp(value);

    internal static float Max(float left, float right) => Math.Max(left, right);

    internal static float Sqrt(float value) => (float)Math.Sqrt(value);

    internal static float Tanh(float value) => (float)Math.Tanh(value);

    internal static long GetAllocatedBytesForCurrentThread()
    {
#if NET471
        // The CLR 4.7.1 surface does not expose per-thread allocation counts.
        // Returning a stable zero keeps cross-target contract tests portable;
        // the same tests perform the real measurement on every modern target.
        return 0;
#else
        return GC.GetAllocatedBytesForCurrentThread();
#endif
    }

    internal static void ThrowIfNull(object? argument, string paramName)
    {
        if (argument is null)
            throw new ArgumentNullException(paramName);
    }

    internal static void ThrowIfNullOrWhiteSpace(string? argument, string paramName)
    {
        if (string.IsNullOrWhiteSpace(argument))
            throw new ArgumentException("The value must not be null or white space.", paramName);
    }

    internal static void ThrowIfDisposed(bool disposed, object instance)
    {
        if (disposed)
            throw new ObjectDisposedException(instance?.GetType().FullName);
    }

    internal static string ToHexString(byte[] bytes)
    {
        var builder = new StringBuilder(bytes.Length * 2);
        foreach (byte b in bytes)
            builder.Append(b.ToString("X2", System.Globalization.CultureInfo.InvariantCulture));
        return builder.ToString();
    }

    internal static nuint ToNuint(IntPtr pointer) => unchecked((nuint)(ulong)pointer.ToInt64());

    internal static IntPtr ToIntPtr(nuint value) => new IntPtr(unchecked((long)(ulong)value));

    internal static TValue? GetValueOrDefault<TKey, TValue>(Dictionary<TKey, TValue> dictionary, TKey key)
        where TKey : notnull =>
        dictionary.TryGetValue(key, out TValue? value) ? value : default;

    internal static int Log2(uint value)
    {
        int result = 0;
        while ((value >>= 1) != 0)
            result++;
        return result;
    }

    internal static int TrailingZeroCount(uint value)
    {
        if (value == 0)
            return 32;
        int count = 0;
        while ((value & 1u) == 0)
        {
            value >>= 1;
            count++;
        }
        return count;
    }
}
