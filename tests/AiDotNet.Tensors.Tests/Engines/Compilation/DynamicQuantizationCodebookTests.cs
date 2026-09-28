using System;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The block-wise dynamic quantization codebook shared by every int8 optimizer-state path (CPU and all GPU backends).
/// </summary>
public class DynamicQuantizationCodebookTests
{
    public static TheoryData<bool> BothMaps => new() { true, false };

    [Theory]
    [MemberData(nameof(BothMaps))]
    public void Codebook_Has256AscendingEntriesSpanningTheUnitRange(bool signed)
    {
        float[] code = signed ? DynamicQuantizationCodebook.Signed : DynamicQuantizationCodebook.Unsigned;
        Assert.Equal(256, code.Length);
        for (int i = 1; i < code.Length; i++) Assert.True(code[i] >= code[i - 1], $"entry {i} is out of order");
        Assert.Equal(1f, code[255]);
        if (signed)
        {
            Assert.True(code[0] < -0.99f && code[0] > -1f, $"signed minimum {code[0]}");
            Assert.Equal(0f, code[DynamicQuantizationCodebook.SignedZeroIndex]);
        }
        else
        {
            Assert.Equal(0f, code[0]);
        }

        // Dynamic, not linear: the smallest non-zero magnitude is many orders below the largest, which is what
        // keeps small second moments from rounding to zero.
        float smallest = code.Where(v => v != 0f).Select(Math.Abs).Min();
        Assert.True(smallest < 1e-6f, $"smallest non-zero entry {smallest} is not in the dynamic range");
    }

    [Theory]
    [MemberData(nameof(BothMaps))]
    public void Encode_ReturnsTheNearestEntry(bool signed)
    {
        float[] code = signed ? DynamicQuantizationCodebook.Signed : DynamicQuantizationCodebook.Unsigned;
        var rng = new Random(11);
        for (int n = 0; n < 20000; n++)
        {
            // Log-uniform magnitudes so the dense region near zero is exercised as much as the top decade.
            float magnitude = (float)Math.Pow(10, -8 + 8 * rng.NextDouble());
            float x = signed && rng.Next(2) == 0 ? -magnitude : magnitude;
            byte q = DynamicQuantizationCodebook.Encode(x, code);
            float best = code.Select(c => Math.Abs(c - x)).Min();
            Assert.True(Math.Abs(code[q] - x) == best, $"x={x:R} encoded to {code[q]:R}, nearest distance {best:R}");
        }

        Assert.Equal(255, DynamicQuantizationCodebook.Encode(2f, code));
        Assert.Equal(0, DynamicQuantizationCodebook.Encode(float.NaN, code));
    }

    [Theory]
    [InlineData((int)DynamicQuantizationCodebook.KernelLanguage.CudaHip)]
    [InlineData((int)DynamicQuantizationCodebook.KernelLanguage.OpenCl)]
    [InlineData((int)DynamicQuantizationCodebook.KernelLanguage.Metal)]
    [InlineData((int)DynamicQuantizationCodebook.KernelLanguage.Glsl)]
    [InlineData((int)DynamicQuantizationCodebook.KernelLanguage.Wgsl)]
    public void KernelPrelude_EmbedsBitIdenticalTables(int languageValue)
    {
        var language = (DynamicQuantizationCodebook.KernelLanguage)languageValue;
        // A GPU decodes with the literals in this prelude; if any printed value did not parse back to the exact
        // float, a state written on one device would decode differently on another.
        string prelude = DynamicQuantizationCodebook.KernelPrelude(language);
        foreach (var (name, code) in new[] { ("ADAM8_S", DynamicQuantizationCodebook.Signed), ("ADAM8_U", DynamicQuantizationCodebook.Unsigned) })
        {
            var match = Regex.Match(prelude, name + @"[^(\{]*[(\{](?<vals>[^)\}]*)[)\}]");
            Assert.True(match.Success, $"{language}: table {name} not found");
            float[] parsed = match.Groups["vals"].Value
                .Split(',')
                .Select(v => float.Parse(v.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture))
                .ToArray();
            Assert.Equal(code, parsed);
        }

        foreach (string fn in new[] { "adam8_dec_s", "adam8_dec_u", "adam8_enc_s", "adam8_enc_u" })
            Assert.Contains(fn, prelude);
    }
}
