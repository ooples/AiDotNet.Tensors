#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Helpers a WGSL source may call without defining, added when it does. WGSL rejects a constant expression that
/// evaluates to infinity or NaN (bitcast&lt;f32&gt;(0x7F800000u), 1.0 / 0.0), so such values come from
/// <c>wgsl_f32_bits</c>, a function call and therefore a runtime value; and WGSL has no isNan builtin.
/// </summary>
internal static class WebGpuShaderPrelude
{
    private const string F32Bits = "fn wgsl_f32_bits(bits: u32) -> f32 { return bitcast<f32>(bits); }\n";
    private const string IsNan = "fn wgsl_is_nan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7FFFFFFFu) > 0x7F800000u; }\n";

    public static string Prepare(string source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        string helpers = "";
        if (source.Contains("wgsl_f32_bits(") && !source.Contains("fn wgsl_f32_bits(")) helpers += F32Bits;
        if (source.Contains("wgsl_is_nan(") && !source.Contains("fn wgsl_is_nan(")) helpers += IsNan;
        if (helpers.Length == 0) return source;
        // Directives (enable / requires / diagnostic) must precede every declaration.
        int insert = 0;
        while (true)
        {
            int lineStart = insert;
            while (lineStart < source.Length && (source[lineStart] == ' ' || source[lineStart] == '\t' || source[lineStart] == '\r' || source[lineStart] == '\n')) lineStart++;
            if (!(Starts(source, lineStart, "enable ") || Starts(source, lineStart, "requires ") || Starts(source, lineStart, "diagnostic"))) break;
            int end = source.IndexOf(';', lineStart);
            if (end < 0) break;
            insert = end + 1;
        }
        return source.Substring(0, insert) + "\n" + helpers + source.Substring(insert);
    }

    private static bool Starts(string s, int at, string token) => string.CompareOrdinal(s, at, token, 0, token.Length) == 0;
}
#endif