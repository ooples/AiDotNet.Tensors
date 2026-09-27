using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// The 8-bit dynamic quantization data type of Dettmers et al., the single source of truth for every block-wise
/// int8 optimizer-state encoding in this library: the CPU fused kernels, the host fallback and every GPU backend.
/// </summary>
/// <remarks>
/// <para>
/// A block is normalized by its absolute maximum and each value is stored as the index of the nearest entry in a fixed
/// 256-entry codebook that is dense near zero, so small values keep their relative precision. Linear absmax
/// quantization (<c>round(x / (absmax / 255))</c>) rounds every second moment below about 1/510 of its block maximum to
/// zero; Adam's denominator then collapses and the update explodes. The first moment uses the signed map and the second
/// moment the unsigned map, exactly as the reference 8-bit optimizers do.
/// </para>
/// <para>
/// The entries follow <c>create_dynamic_map(signed, max_exponent_bits=7, total_bits=8)</c> from bitsandbytes, computed
/// through float32 like the reference. GPU kernels embed the SAME float values (see <see cref="EmitArray"/>), so a
/// state encoded on any device decodes identically on every other.
/// </para>
/// <para><b>Reference:</b> T. Dettmers, M. Lewis, S. Shleifer, L. Zettlemoyer, "8-bit Optimizers via Block-wise
/// Quantization", ICLR 2022.</para>
/// </remarks>
internal static class DynamicQuantizationCodebook
{
    /// <summary>Codebook for signed values (first moment), ascending.</summary>
    internal static readonly float[] Signed = Create(signed: true);

    /// <summary>Codebook for non-negative values (second moment), ascending; entry 0 is 0.</summary>
    internal static readonly float[] Unsigned = Create(signed: false);

    /// <summary>The index of 0 in <see cref="Signed"/>.</summary>
    internal static readonly byte SignedZeroIndex = (byte)Array.IndexOf(Signed, 0f);

    private static float[] Create(bool signed, int maxExponentBits = 7, int totalBits = 8)
    {
        var data = new List<float>(1 << totalBits);
        int nonSignBits = totalBits - 1;
        int additionalItems = (1 << (nonSignBits - maxExponentBits)) - 1;
        int i;
        for (i = 0; i < maxExponentBits; i++)
        {
            int fractionItems = signed
                ? (1 << (i + nonSignBits - maxExponentBits)) + 1
                : (1 << (i + nonSignBits - maxExponentBits + 1)) + 1;
            AddMeans(data, fractionItems, (float)Math.Pow(10, -(maxExponentBits - 1) + i), signed);
        }

        if (additionalItems > 0)
            AddMeans(data, additionalItems + 1, (float)Math.Pow(10, -(maxExponentBits - 1) + i - 1), signed);

        data.Add(0f);
        data.Add(1f);
        if (data.Count != 1 << totalBits)
            throw new InvalidOperationException($"Dynamic quantization codebook has {data.Count} entries, expected 256.");
        data.Sort();
        return data.ToArray();
    }

    private static void AddMeans(List<float> data, int count, float magnitude, bool signed)
    {
        var boundaries = new float[count];
        for (int k = 0; k < count; k++)
            boundaries[k] = count == 1 ? 0.1f : 0.1f + (1f - 0.1f) * k / (count - 1);
        for (int k = 0; k < count - 1; k++)
        {
            float mean = (boundaries[k] + boundaries[k + 1]) / 2f;
            data.Add(magnitude * mean);
            if (signed) data.Add(-magnitude * mean);
        }
    }

    /// <summary>Encodes a value already divided by its block scale as the nearest codebook index.</summary>
    internal static byte Encode(float normalized, float[] code)
    {
        if (!(normalized > code[0])) return 0;                           // also maps NaN to the lowest entry
        if (normalized >= code[code.Length - 1]) return (byte)(code.Length - 1);
        int lo = 0, hi = code.Length - 1;                                // code[lo] <= normalized < code[hi]
        while (hi - lo > 1)
        {
            int mid = (lo + hi) >> 1;
            if (code[mid] <= normalized) lo = mid; else hi = mid;
        }
        return (byte)(normalized - code[lo] <= code[hi] - normalized ? lo : hi);
    }

    /// <summary>The GPU shading languages a <see cref="KernelPrelude"/> can be emitted in.</summary>
    internal enum KernelLanguage
    {
        /// <summary>CUDA and HIP (C++ with <c>__constant__</c> / <c>__device__</c>).</summary>
        CudaHip,
        /// <summary>OpenCL C.</summary>
        OpenCl,
        /// <summary>Metal Shading Language.</summary>
        Metal,
        /// <summary>GLSL (Vulkan compute).</summary>
        Glsl,
        /// <summary>WGSL (WebGPU).</summary>
        Wgsl,
    }

    /// <summary>
    /// Source to prepend to a GPU kernel: both codebooks as device-constant tables plus <c>adam8_dec_s/u(q)</c> (index
    /// to normalized value) and <c>adam8_enc_s/u(x)</c> (normalized value to nearest index). The encoder follows
    /// <see cref="Encode"/> step for step (same comparisons, same tie-break to the lower entry), and the tables hold
    /// the same float values, so every device produces the same bytes as the CPU.
    /// </summary>
    internal static string KernelPrelude(KernelLanguage language)
    {
        string table, fn, u32, f32, arrS, arrU;
        switch (language)
        {
            case KernelLanguage.CudaHip:
                table = "__constant__ float"; fn = "__device__ __forceinline__"; u32 = "unsigned int"; f32 = "float"; break;
            case KernelLanguage.OpenCl:
                table = "__constant float"; fn = "inline"; u32 = "uint"; f32 = "float"; break;
            case KernelLanguage.Metal:
                table = "constant float"; fn = "inline"; u32 = "uint"; f32 = "float"; break;
            case KernelLanguage.Glsl:
                table = "const float"; fn = ""; u32 = "uint"; f32 = "float"; break;
            case KernelLanguage.Wgsl:
                return WgslPrelude();
            default:
                throw new ArgumentOutOfRangeException(nameof(language));
        }

        arrS = language == KernelLanguage.Glsl ? GlslArray("ADAM8_S", Signed) : EmitArray(table, "ADAM8_S", Signed);
        arrU = language == KernelLanguage.Glsl ? GlslArray("ADAM8_U", Unsigned) : EmitArray(table, "ADAM8_U", Unsigned);
        var sb = new StringBuilder();
        sb.AppendLine("// Block-wise dynamic quantization codebooks (Dettmers et al., ICLR 2022), generated from");
        sb.AppendLine("// AiDotNet.Tensors DynamicQuantizationCodebook so every backend decodes the same bytes identically.");
        sb.AppendLine(arrS);
        sb.AppendLine(arrU);
        foreach (var (suffix, arr) in new[] { ("s", "ADAM8_S"), ("u", "ADAM8_U") })
        {
            sb.AppendLine($"{fn} {f32} adam8_dec_{suffix}({u32} q) {{ return {arr}[q]; }}".TrimStart());
            sb.AppendLine($"{fn} {u32} adam8_enc_{suffix}({f32} x) {{".TrimStart());
            sb.AppendLine($"    if (!(x > {arr}[0])) return 0u;");
            sb.AppendLine($"    if (x >= {arr}[255]) return 255u;");
            sb.AppendLine($"    {u32} lo = 0u; {u32} hi = 255u;");
            sb.AppendLine($"    while (hi - lo > 1u) {{ {u32} mid = (lo + hi) >> 1; if ({arr}[mid] <= x) lo = mid; else hi = mid; }}");
            sb.AppendLine($"    return (x - {arr}[lo] <= {arr}[hi] - x) ? lo : hi;");
            sb.AppendLine("}");
        }
        return sb.ToString();
    }

    private static string GlslArray(string name, float[] code)
    {
        var sb = new StringBuilder($"const float {name}[{code.Length}] = float[{code.Length}](");
        for (int k = 0; k < code.Length; k++) { if (k > 0) sb.Append(", "); sb.Append(FormatFloat(code[k])); }
        return sb.Append(");").ToString();
    }

    private static string WgslPrelude()
    {
        var sb = new StringBuilder();
        sb.AppendLine("// Block-wise dynamic quantization codebooks (Dettmers et al., ICLR 2022), generated from");
        sb.AppendLine("// AiDotNet.Tensors DynamicQuantizationCodebook so every backend decodes the same bytes identically.");
        foreach (var (suffix, name, code) in new[] { ("s", "ADAM8_S", Signed), ("u", "ADAM8_U", Unsigned) })
        {
            // var<private>, not const: WGSL only guarantees runtime indexing of an addressable array.
            var arr = new StringBuilder($"var<private> {name}: array<f32, {code.Length}> = array<f32, {code.Length}>(");
            for (int k = 0; k < code.Length; k++) { if (k > 0) arr.Append(", "); arr.Append(FormatFloat(code[k])); }
            sb.AppendLine(arr.Append(");").ToString());
            sb.AppendLine($"fn adam8_dec_{suffix}(q: u32) -> f32 {{ return {name}[q]; }}");
            sb.AppendLine($"fn adam8_enc_{suffix}(x: f32) -> u32 {{");
            sb.AppendLine($"    if (!(x > {name}[0])) {{ return 0u; }}");
            sb.AppendLine($"    if (x >= {name}[255]) {{ return 255u; }}");
            sb.AppendLine("    var lo: u32 = 0u; var hi: u32 = 255u;");
            sb.AppendLine($"    loop {{ if (hi - lo <= 1u) {{ break; }} let mid = (lo + hi) >> 1u; if ({name}[mid] <= x) {{ lo = mid; }} else {{ hi = mid; }} }}");
            sb.AppendLine($"    return select(hi, lo, x - {name}[lo] <= {name}[hi] - x);");
            sb.AppendLine("}");
        }
        return sb.ToString();
    }

    /// <summary>Emits <paramref name="code"/> as a C-style constant array declaration (CUDA/HIP/OpenCL/Metal).</summary>
    internal static string EmitArray(string qualifier, string name, float[] code)
    {
        var sb = new StringBuilder($"{qualifier} {name}[{code.Length}] = {{");
        for (int k = 0; k < code.Length; k++) { if (k > 0) sb.Append(", "); sb.Append(FormatFloat(code[k])); }
        return sb.Append("};").ToString();
    }

    private static string FormatFloat(float value)
    {
        // "R" guarantees round-trip; force a decimal point and an f32 suffix-free literal every language accepts.
        string s = value.ToString("R", CultureInfo.InvariantCulture);
        if (s.IndexOf('E') >= 0 || s.IndexOf('e') >= 0)
            s = ((double)value).ToString("0.0###########################", CultureInfo.InvariantCulture);
        if (s.IndexOf('.') < 0) s += ".0";
        return s;
    }
}
