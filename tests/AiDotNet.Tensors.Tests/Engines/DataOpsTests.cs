using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The data-loading sources on the CPU engine against hand-computed values: <c>TensorFromBytes</c> (each value below is
/// the IEEE-754 round-to-nearest-even result, the same NumPy 2.1 gives for
/// <c>np.frombuffer(buf, src).astype(np.float32)</c>), <c>TensorParse</c> (the framework's own TryParse per token, and
/// the CSV splitting rules), the caller-Random uniform fill (the scalar loop's values and stream consumption), and the
/// stable argsort (LINQ OrderBy / OrderByDescending order for ties).
/// </summary>
public sealed class DataOpsTests
{
    private static readonly CpuEngine Cpu = new();

    private static byte[] Bytes<TV>(TV[] values, Func<TV, byte[]> getBytes, bool littleEndian = true)
    {
        var bytes = new List<byte>();
        foreach (var v in values)
        {
            var b = getBytes(v);
            if (BitConverter.IsLittleEndian != littleEndian) Array.Reverse(b);
            bytes.AddRange(b);
        }
        return bytes.ToArray();
    }

    private static void AssertBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]), $"[{i}] expected NaN, got {actual[i]:R}");
            else Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(expected[i]), 0) == BitConverter.ToInt32(BitConverter.GetBytes(actual[i]), 0),
                $"[{i}] expected {expected[i]:R}, got {actual[i]:R}");
        }
    }

    // ------------------------------------------------------------------ TensorFromBytes

    [Fact]
    public void FromBytes_Float16_IsExact()
    {
        // 1, -2, 65504 (max), 2^-24 (smallest subnormal), 2^-14 (smallest normal), -0, +inf, NaN.
        ushort[] bits = { 0x3C00, 0xC000, 0x7BFF, 0x0001, 0x0400, 0x8000, 0x7C00, 0x7E00 };
        var t = Cpu.TensorFromBytes<float>(Bytes(bits, BitConverter.GetBytes), NumericStorageType.Float16, new[] { 8 });
        AssertBits(new[] { 1f, -2f, 65504f, 5.9604645E-08f, 6.1035156E-05f, -0f, float.PositiveInfinity, float.NaN }, t.ToArray());
    }

    [Fact]
    public void FromBytes_Float64_RoundsToNearestEvenIncludingSubnormals()
    {
        double[] values =
        {
            1.0 / 3.0,                    // 0.333333343f
            16777217.0,                   // tie between 2^24 and 2^24 + 2: to even, 16777216
            16777219.0,                   // tie: to even, 16777220
            Math.Pow(2, -149),            // smallest float subnormal, exact
            Math.Pow(2, -150),            // half of it: tie to even, +0
            1.5 * Math.Pow(2, -150),      // above the tie: 2^-149
            -3.4028235677973366E+38,      // above -float.MaxValue by less than half an ulp: -float.MaxValue
            1e39,                         // beyond float range: +inf
            -0.0,
            double.NegativeInfinity,
            double.NaN,
        };
        var t = Cpu.TensorFromBytes<float>(Bytes(values, BitConverter.GetBytes), NumericStorageType.Float64, new[] { values.Length });
        AssertBits(new[]
        {
            0.333333343f, 16777216f, 16777220f, 1.401298E-45f, 0f, 1.401298E-45f, -float.MaxValue, float.PositiveInfinity,
            -0f, float.NegativeInfinity, float.NaN,
        }, t.ToArray());
    }

    [Fact]
    public void FromBytes_Integers_ConvertAndRoundOnce()
    {
        var i8 = Cpu.TensorFromBytes<float>(new byte[] { 0x80, 0x7F, 0xFF, 0x00 }, NumericStorageType.Int8, new[] { 4 });
        AssertBits(new[] { -128f, 127f, -1f, 0f }, i8.ToArray());
        var u8 = Cpu.TensorFromBytes<float>(new byte[] { 0x80, 0xFF }, NumericStorageType.UInt8, new[] { 2 });
        AssertBits(new[] { 128f, 255f }, u8.ToArray());
        var i16 = Cpu.TensorFromBytes<float>(Bytes(new short[] { short.MinValue, -1, 12345 }, BitConverter.GetBytes),
            NumericStorageType.Int16, new[] { 3 });
        AssertBits(new[] { -32768f, -1f, 12345f }, i16.ToArray());
        var u16 = Cpu.TensorFromBytes<float>(Bytes(new ushort[] { 65535, 1 }, BitConverter.GetBytes), NumericStorageType.UInt16, new[] { 2 });
        AssertBits(new[] { 65535f, 1f }, u16.ToArray());
        var i32 = Cpu.TensorFromBytes<float>(Bytes(new[] { 16777217, 16777219, int.MinValue, int.MaxValue }, BitConverter.GetBytes),
            NumericStorageType.Int32, new[] { 4 });
        AssertBits(new[] { 16777216f, 16777220f, -2147483648f, 2147483648f }, i32.ToArray());
        var u32 = Cpu.TensorFromBytes<float>(Bytes(new[] { uint.MaxValue, 16777217u }, BitConverter.GetBytes), NumericStorageType.UInt32, new[] { 2 });
        AssertBits(new[] { 4294967296f, 16777216f }, u32.ToArray());
        // 2^53 + 1 -> 2^53 (float: 9007199254740992); 2^24 + 1 + 2^40 rounds once (not via double) to 1099528404992.
        var i64 = Cpu.TensorFromBytes<float>(Bytes(new[] { 9007199254740993L, long.MinValue, (1L << 40) + (1L << 24) + 1, -5L },
            BitConverter.GetBytes), NumericStorageType.Int64, new[] { 4 });
        AssertBits(new[] { 9007199254740992f, -9223372036854775808f, 1099528404992f, -5f }, i64.ToArray());
    }

    [Fact]
    public void FromBytes_BigEndian_ReadsEachElementMostSignificantByteFirst()
    {
        var f32 = Cpu.TensorFromBytes<float>(Bytes(new[] { 1.5f, -2.25f }, BitConverter.GetBytes, littleEndian: false),
            NumericStorageType.Float32, new[] { 2 }, littleEndian: false);
        AssertBits(new[] { 1.5f, -2.25f }, f32.ToArray());
        var i16 = Cpu.TensorFromBytes<float>(new byte[] { 0x01, 0x02, 0xFF, 0xFE }, NumericStorageType.Int16, new[] { 2 },
            littleEndian: false);
        AssertBits(new[] { 258f, -2f }, i16.ToArray());
        var f64 = Cpu.TensorFromBytes<double>(Bytes(new[] { Math.PI }, BitConverter.GetBytes, littleEndian: false),
            NumericStorageType.Float64, new[] { 1 }, littleEndian: false);
        Assert.Equal(Math.PI, f64.ToArray()[0]);
    }

    [Fact]
    public void FromBytes_OtherElementTypes_KeepPrecisionAndTruncate()
    {
        // double: int64 2^53 + 1 rounds once to 2^53; float32 widens exactly.
        var d = Cpu.TensorFromBytes<double>(Bytes(new[] { 9007199254740993L }, BitConverter.GetBytes), NumericStorageType.Int64, new[] { 1 });
        Assert.Equal(9007199254740992.0, d.ToArray()[0]);
        var dw = Cpu.TensorFromBytes<double>(Bytes(new[] { 0.1f }, BitConverter.GetBytes), NumericStorageType.Float32, new[] { 1 });
        Assert.Equal((double)0.1f, dw.ToArray()[0]);
        // long keeps every int64 bit; int and long truncate a floating source toward zero (torch .to(int64)).
        var l = Cpu.TensorFromBytes<long>(Bytes(new[] { long.MaxValue, -7L }, BitConverter.GetBytes), NumericStorageType.Int64, new[] { 2 });
        Assert.Equal(new[] { long.MaxValue, -7L }, l.ToArray());
        var n = Cpu.TensorFromBytes<int>(Bytes(new[] { 2.7, -2.7 }, BitConverter.GetBytes), NumericStorageType.Float64, new[] { 2 });
        Assert.Equal(new[] { 2, -2 }, n.ToArray());
        // A generic type goes through the double decode.
        var m = Cpu.TensorFromBytes<decimal>(new byte[] { 200 }, NumericStorageType.UInt8, new[] { 1 });
        Assert.Equal(200m, m.ToArray()[0]);
    }

    [Fact]
    public void FromBytes_ShapeAndLengthAreChecked()
    {
        var t = Cpu.TensorFromBytes<float>(Bytes(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, BitConverter.GetBytes), NumericStorageType.Float32, new[] { 2, 3 });
        Assert.Equal(new[] { 2, 3 }, t.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, t.ToArray());
        Assert.Throws<ArgumentException>(() => Cpu.TensorFromBytes<float>(new byte[7], NumericStorageType.Float32, new[] { 2 }));
        Assert.Throws<ArgumentException>(() => Cpu.TensorFromBytes<float>(new byte[0], NumericStorageType.Float32, new[] { -1 }));
        Assert.Empty(Cpu.TensorFromBytes<float>(ReadOnlySpan<byte>.Empty, NumericStorageType.Float64, new[] { 0, 3 }).ToArray());
    }

    [Fact]
    public void FromBytes_ThroughTheInterface_UsesTheEngine()
    {
        IEngine engine = Cpu;
        var t = engine.TensorFromBytes<float>(Bytes(new[] { 3.0 }, BitConverter.GetBytes), NumericStorageType.Float64, new[] { 1 });
        Assert.Equal(3f, t.ToArray()[0]);
    }

    // ------------------------------------------------------------------ TensorParse

    [Fact]
    public void Parse_Tokens_UseTheTypesTryParse()
    {
        var tokens = new string?[] { "1.5", "-2", "abc", "", "1e3", "  7  ", null, "NaN", "-Infinity", "0.1" };
        var t = Cpu.TensorParse<float>(tokens, new[] { 2, 5 }, NumberStyles.Float,
            CultureInfo.InvariantCulture, -99f);
        AssertBits(new[] { 1.5f, -2f, -99f, -99f, 1000f, 7f, -99f, float.NaN, float.NegativeInfinity, 0.1f }, t.ToArray());

        var d = Cpu.TensorParse<double>(new[] { "0.1", "x" }, new[] { 2 }, NumberStyles.Float, CultureInfo.InvariantCulture, double.NaN);
        Assert.Equal(0.1, d.ToArray()[0]);
        Assert.True(double.IsNaN(d.ToArray()[1]));

        // int: a fractional value does not parse as an integer.
        var n = Cpu.TensorParse<int>(new[] { "42", "4.5", "-3" }, new[] { 3 }, NumberStyles.Integer, CultureInfo.InvariantCulture, -1);
        Assert.Equal(new[] { 42, -1, -3 }, n.ToArray());
    }

    [Fact]
    public void Parse_Text_SplitsOnTheDelimiterAndLineEndings()
    {
        var text = Encoding.UTF8.GetBytes("1,2,3\r\n4,,6e-1\n");
        var t = Cpu.TensorParse<float>(text, (byte)',', new[] { 2, 3 }, NumberStyles.Float, CultureInfo.InvariantCulture, float.NaN);
        AssertBits(new[] { 1f, 2f, 3f, 4f, float.NaN, 0.6f }, t.ToArray());

        // Without a final newline, and with tab separators; a carriage return at the end without a line feed stays in
        // the field (and is trailing white space here).
        var tsv = Encoding.UTF8.GetBytes("0.5\t-1\n2\t3\r");
        var u = Cpu.TensorParse<double>(tsv, (byte)'\t', new[] { 4 }, NumberStyles.Float, CultureInfo.InvariantCulture, double.NaN);
        Assert.Equal(new[] { 0.5, -1, 2, 3 }, u.ToArray());
    }

    [Fact]
    public void Parse_Text_FieldCountMustMatchTheShape()
    {
        var text = Encoding.UTF8.GetBytes("1,2,3");
        Assert.Throws<ArgumentException>(() =>
            Cpu.TensorParse<float>(text, (byte)',', new[] { 2 }, NumberStyles.Float, CultureInfo.InvariantCulture, 0f));
        Assert.Empty(Cpu.TensorParse<float>(Encoding.UTF8.GetBytes("\n"), (byte)',', new[] { 0 }, NumberStyles.Float,
            CultureInfo.InvariantCulture, 0f).ToArray());
    }

    [Fact]
    public void Parse_Text_HonoursTheCulture()
    {
        var text = Encoding.UTF8.GetBytes("1,5;-2,25");
        var t = Cpu.TensorParse<float>(text, (byte)';', new[] { 2 }, NumberStyles.Float, new CultureInfo("de-DE"), float.NaN);
        AssertBits(new[] { 1.5f, -2.25f }, t.ToArray());
    }

    [Fact]
    public void SplitFieldBounds_FollowsTheDocumentedRules()
    {
        var bounds = CpuEngine.SplitFieldBounds(Encoding.UTF8.GetBytes("a,b\r\n\nc\r\n"), (byte)',', out int fields);
        // "a" [0,1), "b" [2,3) (CR dropped), "" [5,5) (blank line), "c" [6,7) (final CRLF dropped).
        Assert.Equal(4, fields);
        Assert.Equal(new[] { 0, 1, 2, 3, 5, 5, 6, 7 }, bounds);
        CpuEngine.SplitFieldBounds(ReadOnlySpan<byte>.Empty, (byte)',', out int none);
        Assert.Equal(0, none);
        CpuEngine.SplitFieldBounds(Encoding.UTF8.GetBytes(","), (byte)',', out int two);
        Assert.Equal(2, two);
    }

#if NETCOREAPP
    // .NET Core 3.0+ parses correctly rounded (IEEE), the reference the fast path must equal bit for bit; .NET Framework's
    // float.Parse is not always correctly rounded, so this runs off .NET Framework only.
    [Fact]
    public void FastPath_DecidesAlmostEveryField_AndEqualsFloatParse()
    {
        var corpus = NumericCorpus(20_000, seed: 7);
        int mask = NumericTextFastPath.StyleMask(NumberStyles.Float, CultureInfo.InvariantCulture);
        Assert.True(mask >= 0);
        int decided = 0;
        foreach (var s in corpus)
        {
            int status = NumericTextFastPath.TryParseSingleBits(Encoding.UTF8.GetBytes(s), mask, out uint bits);
            bool ok = float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out float expected);
            if (status == NumericTextFastPath.Parsed)
            {
                decided++;
                Assert.True(ok, $"'{s}' was decided but does not parse");
                Assert.True(BitConverter.SingleToUInt32Bits(expected) == bits,
                    $"'{s}': float.Parse {expected:R} ({BitConverter.SingleToUInt32Bits(expected):X8}), fast path {bits:X8}");
            }
            else if (status == NumericTextFastPath.Empty)
            {
                Assert.False(ok);
            }
        }
        Assert.True(decided > corpus.Count * 8 / 10, $"the fast path decided only {decided} of {corpus.Count}");

        // The whole op equals float.TryParse per field, decided or not.
        string text = string.Join(";", corpus); // the corpus holds "1,5", one unparsable field
        var t = Cpu.TensorParse<float>(Encoding.UTF8.GetBytes(text), (byte)';', new[] { corpus.Count }, NumberStyles.Float,
            CultureInfo.InvariantCulture, -1234.5f);
        var expectedAll = corpus.Select(s => float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : -1234.5f).ToArray();
        AssertBits(expectedAll, t.ToArray());
    }
#endif

    /// <summary>Numeric text the loaders meet: round-trip floats and doubles, integers, scientific notation, padding,
    /// signs, long mantissas, ties, and a few non-numbers.</summary>
    internal static List<string> NumericCorpus(int count, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var list = new List<string>
        {
            "0", "-0", "+0", "0.0", "1", "-1", "+1", "1.", ".5", "-.5", "1e5", "1E-5", "1e+5", "1.5e-3", "16777217",
            "16777219", "0.1", "3.14159265358979", "2147483647", "2147483648", "1e10", "1e-10", "123456789012345678",
            "0.000000001", "999999999", "9999999999", " 4 ", "\t-6.25", "", " ", "abc", "1,5", "--1", "1e", "1e+",
            "NaN", "Infinity", "-Infinity", "1.2.3", "0x10", "7d", ".", "+", "-", "e5",
        };
        while (list.Count < count)
        {
            switch (rng.Next(8))
            {
                case 0: list.Add(((float)(rng.NextDouble() * 2000 - 1000)).ToString("R", CultureInfo.InvariantCulture)); break;
                case 1: list.Add((rng.NextDouble() * 2 - 1).ToString("R", CultureInfo.InvariantCulture)); break;
                case 2: list.Add(rng.Next(-100000, 100000).ToString(CultureInfo.InvariantCulture)); break;
                case 3: list.Add((rng.NextDouble() * 10).ToString("E3", CultureInfo.InvariantCulture)); break;
                case 4: list.Add((rng.NextDouble() * 100).ToString("F" + rng.Next(0, 9), CultureInfo.InvariantCulture)); break;
                case 5:
                {
                    // Exact ties of float rounding: an odd multiple of half an ulp at a random magnitude.
                    double m = (1 << 23) + rng.Next(1 << 23);
                    double v = (2 * m + 1) * Math.Pow(2, rng.Next(-30, -24));
                    list.Add(v.ToString("R", CultureInfo.InvariantCulture));
                    break;
                }
                case 6: list.Add(((float)Math.Pow(10, rng.Next(-12, 12)) * (float)rng.NextDouble()).ToString("R", CultureInfo.InvariantCulture)); break;
                default: list.Add(new string('9', rng.Next(1, 14)) + "." + new string('5', rng.Next(0, 12))); break;
            }
        }
        return list;
    }

    // ------------------------------------------------------------------ TensorRandomUniformRange (caller's Random)

    [Fact]
    public void RandomUniform_MatchesTheScalarLoop_AndConsumesTheStreamExactly()
    {
        var rng = RandomHelper.CreateSeededRandom(1234);
        var reference = RandomHelper.CreateSeededRandom(1234);
        var t = Cpu.TensorRandomUniformRange<float>(rng, new[] { 3, 5 }, -1.0, 1.0);
        var expected = new float[15];
        for (int i = 0; i < 15; i++) expected[i] = (float)(reference.NextDouble() * 2 - 1);   // the consumers' loop
        AssertBits(expected, t.ToArray());
        Assert.Equal(reference.NextDouble(), rng.NextDouble());

        var d = ((IEngine)Cpu).TensorRandomUniformRange<double>(rng, new[] { 4 }, -5.0, 5.0);
        for (int i = 0; i < 4; i++) Assert.Equal(reference.NextDouble() * 10.0 - 5.0, d.ToArray()[i]);
        Assert.Equal(reference.Next(), rng.Next());
    }

    [Fact]
    public void RandomUniformInto_FillsAStridedViewInLogicalOrder()
    {
        var rng = RandomHelper.CreateSeededRandom(5);
        var reference = RandomHelper.CreateSeededRandom(5);
        var baseTensor = new Tensor<double>(new[] { 3, 2 });
        var view = baseTensor.Transpose(new[] { 1, 0 });   // [2, 3], not contiguous
        Cpu.TensorRandomUniformRangeInto(rng, view, 0.0, 1.0);
        var logical = view.ToArray();
        for (int i = 0; i < 6; i++) Assert.Equal(reference.NextDouble(), logical[i]);
    }

    // ------------------------------------------------------------------ stable argsort

    public static IEnumerable<object[]> SortCases()
    {
        foreach (bool descending in new[] { false, true })
            foreach (int n in new[] { 1, 7, 64, 1000 })
                yield return new object[] { n, descending };
    }

    [Theory]
    [MemberData(nameof(SortCases))]
    public void Argsort_KeepsTiesInIndexOrder_LikeLinq(int n, bool descending)
    {
        var rng = RandomHelper.CreateSeededRandom(n * 2 + (descending ? 1 : 0));
        var keys = Enumerable.Range(0, n).Select(_ => (double)rng.Next(0, 5)).ToArray();   // many ties
        var expected = (descending
            ? Enumerable.Range(0, n).OrderByDescending(i => keys[i])
            : Enumerable.Range(0, n).OrderBy(i => keys[i])).ToArray();

        var d = new Tensor<double>(keys, new[] { n });
        Assert.Equal(expected, Cpu.TensorArgsort(d, -1, descending).ToArray());
        Assert.Equal(expected, ((IEngine)Cpu).ArgSort(d, -1, descending).ToArray());
        Assert.Equal(expected, Cpu.TensorSort(d, -1, descending).Indices.ToArray());
        var f = new Tensor<float>(keys.Select(k => (float)k).ToArray(), new[] { n });
        Assert.Equal(expected, ((IEngine)Cpu).ArgSort(f, 0, descending).ToArray());
        // A type without a radix key (the comparison path).
        var ints = new Tensor<int>(keys.Select(k => (int)k).ToArray(), new[] { n });
        Assert.Equal(expected, ((IEngine)Cpu).ArgSort(ints, -1, descending).ToArray());
    }

    [Fact]
    public void Argsort_AlongTheFirstAxis_IsStablePerColumn()
    {
        // Column 0: 2, 1, 2, 1 -> ascending 1, 3, 0, 2; descending 0, 2, 1, 3. Column 1: all equal -> 0, 1, 2, 3.
        var t = new Tensor<float>(new[] { 2f, 5f, 1f, 5f, 2f, 5f, 1f, 5f }, new[] { 4, 2 });
        Assert.Equal(new[] { 1, 0, 3, 1, 0, 2, 2, 3 }, ((IEngine)Cpu).ArgSort(t, 0, false).ToArray());
        Assert.Equal(new[] { 0, 0, 2, 1, 1, 2, 3, 3 }, ((IEngine)Cpu).ArgSort(t, 0, true).ToArray());
    }

    [Fact]
    public void Argsort_PutsNaNLastAscendingAndFirstDescending_AndEqualsSignedZeros()
    {
        var t = new Tensor<double>(new[] { double.NaN, 0.0, -0.0, -1.0, double.NaN }, new[] { 5 });
        Assert.Equal(new[] { 3, 1, 2, 0, 4 }, ((IEngine)Cpu).ArgSort(t, -1, false).ToArray());
        Assert.Equal(new[] { 0, 4, 1, 2, 3 }, ((IEngine)Cpu).ArgSort(t, -1, true).ToArray());
    }
}
