using System;
using System.Globalization;
using System.Numerics;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// sin and cos with the argument reduced exactly on every runtime, for the special functions whose large-argument
/// forms take sin/cos of x itself (Hankel's expansion at x = 1e14, the Airy phase at 6.7e20).
/// </summary>
/// <remarks>
/// .NET Core's Math.Sin/Cos reduce any argument exactly, but .NET Framework's do not: at 1e14 they are off by 2e-7
/// relative, and past about 1e19 they hand the argument back unreduced (sin(1e300) came back as 1e300, so
/// sin(x)/x = 1). Below 2^20 every runtime's Math.Sin/Cos is exact enough and is used as is. Above it the argument
/// goes through Payne–Hanek reduction: x = m·2^e, and only the 192 bits of 2/π that can change (x·2/π) mod 4 are
/// multiplied in, which leaves the quadrant and the fraction to 2^-137, far below the closest any double comes to a
/// multiple of π/2 (about 2^-61).
/// </remarks>
internal static class ExactTrig
{
    private const double Threshold = 1 << 20;
    private const int TableBits = 1280;
    private const int WindowBits = 192;

    // floor(2^1280 · 2/π): bit i of 2/π (weight 2^-i) is bit 1280 - i of this integer.
    private static readonly BigInteger TwoOverPi = BigInteger.Parse("0" +
        "a2f9836e4e441529fc2757d1f534ddc0db6295993c439041fe5163abdebbc561" +
        "b7246e3a424dd2e006492eea09d1921cfe1deb1cb129a73ee88235f52ebb4484" +
        "e99c7026b45f7e413991d639835339f49c845f8bbdf9283b1ff897ffde05980f" +
        "ef2f118b5a0a6d1f6d367ecf27cb09b74f463f669e5fea2d7527bac7ebe5f17b" +
        "3d0739f78a5292ea6bfb5fb11f8d5d0856033046fc7b6babf0cfbc209af4361d",
        NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture);

    private static readonly BigInteger WindowMask = (BigInteger.One << WindowBits) - 1;

    public static double Sin(double x)
    {
        double ax = Math.Abs(x);
        if (!(ax >= Threshold) || double.IsInfinity(ax)) return Math.Sin(x);
        double r = Reduce(ax, out int n);
        double s = (n & 1) == 0 ? Math.Sin(r) : Math.Cos(r);
        if ((n & 2) != 0) s = -s;
        return x < 0 ? -s : s;
    }

    public static double Cos(double x)
    {
        double ax = Math.Abs(x);
        if (!(ax >= Threshold) || double.IsInfinity(ax)) return Math.Cos(x);
        double r = Reduce(ax, out int n);
        double c = (n & 1) == 0 ? Math.Cos(r) : Math.Sin(r);
        return ((n + 1) & 2) != 0 ? -c : c;
    }

    // ax = n·π/2 + r, |r| ≤ π/4, for finite ax ≥ 2^20; n is the quadrant mod 4.
    private static double Reduce(double ax, out int n)
    {
        long bits = BitConverter.DoubleToInt64Bits(ax);
        int e = (int)((bits >> 52) & 0x7FF) - 1075;                 // ax = m·2^e, m < 2^53 (ax is normal here)
        long m = (bits & ((1L << 52) - 1)) | (1L << 52);
        // Bits of 2/π with weight 2^-i for i < e - 1 multiply m·2^e into multiples of 4: they cannot change the result.
        int i0 = Math.Max(1, e - 1);
        BigInteger window = (TwoOverPi >> (TableBits - (i0 + WindowBits - 1))) & WindowMask;
        BigInteger product = window * m;                           // ax·2/π = product·2^-shift (mod 4)
        int shift = i0 + WindowBits - 1 - e;
        n = (int)((product >> shift) & 3);
        BigInteger one = BigInteger.One << shift;
        BigInteger fraction = product & (one - 1);
        // Round to the nearest quadrant in integers, so a fraction just under 1 keeps its precision as a small negative.
        if (fraction >= one >> 1)
        {
            fraction -= one;
            n = (n + 1) & 3;
        }
        // The top 128 bits of the fraction to a double: 2^-53 relative, even after the leading zeros of a fraction
        // near 0 (no double comes closer than about 2^-61 to a multiple of π/2).
        BigInteger top = shift > 128 ? fraction >> (shift - 128) : fraction << (128 - shift);
        double f = (double)top * Math.Pow(2, -128);
        return f * (Math.PI / 2);
    }
}
