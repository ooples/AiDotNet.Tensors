#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Helpers a WGSL source may call without defining, added when it does. WGSL rejects a constant expression that
/// evaluates to infinity or NaN (bitcast&lt;f32&gt;(0x7F800000u), 1.0 / 0.0), so such values come from
/// <c>wgsl_f32_bits</c>, a function call and therefore a runtime value; and WGSL has no isNan builtin.
/// <para>WGSL bounds its tanh only through sinh / cosh and its log / log2 only absolutely (2^-21 on [0.5, 2]), and
/// Dawn's D3D12 lowering shows it: tanh about 1e-3 relative near zero, log2 and log 2e-5 relative near one, where the
/// answers are small (a GELU's tanh(0.08), a log-probability near log 1). So every f32 module's builtin <c>tanh</c>,
/// <c>log</c> and <c>log2</c> calls are routed to <c>wgsl_tanh</c> / <c>wgsl_log</c> / <c>wgsl_log2</c>, which hold
/// a few ULP relative everywhere as
/// the other backends' libm does: the hand-written kernels, the codegen emitter's and the CUDA-dialect translator's
/// alike. A module that enables f16 keeps the builtins (the helpers are f32-only).</para>
/// </summary>
internal static class WebGpuShaderPrelude
{
    private const string F32Bits = "fn wgsl_f32_bits(bits: u32) -> f32 { return bitcast<f32>(bits); }\n";
    private const string IsNan = "fn wgsl_is_nan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7FFFFFFFu) > 0x7F800000u; }\n";

    // tanh: below 0.55 its odd Taylor series to x^15 (the first omitted term is under 2e-8 of the answer there);
    // above, 1 - 2 / (exp(2|x|) + 1), whose cancellation is at most a factor of two and whose exp overflow gives 1.
    private const string Tanh = @"fn wgsl_tanh(x: f32) -> f32 {
    let a = abs(x);
    if (a < 0.55) {
        let z = x * x;
        let p = -0.3333333333 + z * (0.1333333333 + z * (-0.0539682540 + z * (0.0218694885
              + z * (-0.0088632355 + z * (0.0035921280 + z * -0.0014558344)))));
        return x + x * z * p;
    }
    let t = 1.0 - 2.0 / (exp(2.0 * a) + 1.0);
    return select(-t, t, x >= 0.0);
}
";

    // log and log2 with relative accuracy near one: x = m 2^e with m in [sqrt(1/2), sqrt(2)), ln m = 2 atanh(s) for
    // s = (m - 1) / (m + 1) (|s| <= 0.1716; m - 1 is exact there), summed to s^11. log2 is exact at powers of two;
    // log adds e ln 2 as a high part exact in f32 for any exponent plus a low part. Zero, negatives, subnormals,
    // infinity and NaN keep the builtin's answer (wgsl_log_special).
    private const string LogSplit = @"fn wgsl_log_special(x: f32) -> bool { return !(x >= 1.17549435e-38) || bitcast<u32>(x) >= 0x7F800000u; }
fn wgsl_log_split(x: f32) -> vec2<f32> {
    let parts = frexp(x);
    var m = parts.fract;
    var e = f32(parts.exp);
    if (m < 0.70710678) { m = m * 2.0; e = e - 1.0; }
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let lnm = 2.0 * s + 2.0 * s * z * (0.3333333333 + z * (0.2 + z * (0.1428571429 + z * (0.1111111111 + z * 0.0909090909))));
    return vec2<f32>(e, lnm);
}
";

    private const string Log2 = @"fn wgsl_log2(x: f32) -> f32 {
    if (wgsl_log_special(x)) { return log2(x); }
    let p = wgsl_log_split(x);
    return p.x + p.y * 1.4426950409;
}
";

    private const string Log = @"fn wgsl_log(x: f32) -> f32 {
    if (wgsl_log_special(x)) { return log(x); }
    let p = wgsl_log_split(x);
    return p.x * 0.693145751953125 + (p.x * 1.4286068203094172e-6 + p.y);
}
";

    // A builtin call: the name not preceded by an identifier character or a member access (fast_tanh, x.tanh) and
    // followed by its argument list (tanh_derivative is another name).
    private static readonly Regex BuiltinTanh = new(@"(?<![\w.])tanh\s*\(", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    private static readonly Regex BuiltinLog2 = new(@"(?<![\w.])log2\s*\(", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    private static readonly Regex BuiltinLog = new(@"(?<![\w.])log\s*\(", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));

    public static string Prepare(string source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        // Rewritten before the helpers are added, so the helpers' own builtin calls stay builtin; a source that already
        // carries a helper (prepared once) is not rewritten again, or wgsl_log2's builtin fallback would call itself.
        if (!source.Contains("enable f16"))
        {
            if (!source.Contains("fn wgsl_tanh(")) source = BuiltinTanh.Replace(source, "wgsl_tanh(");
            if (!source.Contains("fn wgsl_log2(")) source = BuiltinLog2.Replace(source, "wgsl_log2(");
            if (!source.Contains("fn wgsl_log(")) source = BuiltinLog.Replace(source, "wgsl_log(");
        }
        string helpers = "";
        if (source.Contains("wgsl_f32_bits(") && !source.Contains("fn wgsl_f32_bits(")) helpers += F32Bits;
        if (source.Contains("wgsl_is_nan(") && !source.Contains("fn wgsl_is_nan(")) helpers += IsNan;
        if (source.Contains("wgsl_tanh(") && !source.Contains("fn wgsl_tanh(")) helpers += Tanh;
        bool log2 = source.Contains("wgsl_log2(") && !source.Contains("fn wgsl_log2(");
        bool log = source.Contains("wgsl_log(") && !source.Contains("fn wgsl_log(");
        if ((log2 || log) && !source.Contains("fn wgsl_log_split(")) helpers += LogSplit;
        if (log2) helpers += Log2;
        if (log) helpers += Log;
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
