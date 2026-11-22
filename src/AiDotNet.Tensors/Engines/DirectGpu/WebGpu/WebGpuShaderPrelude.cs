#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Linq;
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
/// alike. The helpers are scalar f32, and WGSL has no overloading, so three kinds of call keep the builtin: every call in a
/// module that enables f16, a call whose argument is a vector (a vector constructor, a name or function declared as a
/// vector, a multi-component swizzle), and a call inside a const or override declaration (a constant expression may not
/// call a user function).</para>
/// </summary>
internal static class WebGpuShaderPrelude
{
    private const string F32Bits = "fn wgsl_f32_bits(bits: u32) -> f32 { return bitcast<f32>(bits); }\n";
    private const string IsNan = "fn wgsl_is_nan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7FFFFFFFu) > 0x7F800000u; }\n";

    // tanh: below 0.55 its odd Taylor series to x^15 (the first omitted term is under 5e-8 of the answer there);
    // above, 1 - 2 / (exp(2|x|) + 1), whose cancellation is at most a factor of two. From 9.02 up tanh rounds to
    // exactly one in f32 (1 - tanh(9.02) is under 3e-8, less than half the spacing below one), so the sign is
    // returned before exp is evaluated: exp(2|x|) overflows past |x| = 44.4 and WGSL leaves an overflowed result
    // indeterminate. A NaN fails both comparisons and flows through the exp branch as NaN.
    private const string Tanh = @"fn wgsl_tanh(x: f32) -> f32 {
    let a = abs(x);
    if (a >= 9.02) { return select(-1.0, 1.0, x >= 0.0); }
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

    // Names the source declares as vectors: parameters, fields and typed declarations (name: vecN / name: vecNf),
    // declarations initialized from a constructor (let name = vecN...), and functions returning one (fn name(...) -> vecN).
    private static readonly Regex VectorNames = new(
        @"(\w+)\s*:\s*vec[234]|(?:let|var)\s+(\w+)\s*=\s*vec[234]|fn\s+(\w+)\s*\([^)]*\)\s*->\s*vec[234]",
        RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    private static readonly Regex Identifier = new(@"[A-Za-z_]\w*", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    private static readonly Regex VectorSyntax = new(@"\bvec[234]|\.[xyzwrgba]{2,4}\b", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    private static readonly Regex SingleComponent = new(@"\G\.[xyzwrgba]\b", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));

    public static string Prepare(string source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        // Rewritten before the helpers are added, so the helpers' own builtin calls stay builtin; a source that already
        // carries a helper (prepared once) is not rewritten again, or wgsl_log2's builtin fallback would call itself.
        DirectivesEnd(source, out bool enablesF16);
        if (!enablesF16)
        {
            HashSet<string>? vectors = null;
            if (!source.Contains("fn wgsl_tanh(")) source = Rewrite(source, BuiltinTanh, "wgsl_tanh(", ref vectors);
            if (!source.Contains("fn wgsl_log2(")) source = Rewrite(source, BuiltinLog2, "wgsl_log2(", ref vectors);
            if (!source.Contains("fn wgsl_log(")) source = Rewrite(source, BuiltinLog, "wgsl_log(", ref vectors);
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
        int insert = DirectivesEnd(source, out _);
        return source.Substring(0, insert) + "\n" + helpers + source.Substring(insert);
    }

    // The offset just past the leading directives, skipping whitespace and comments between them; and whether an enable
    // among them lists f16.
    private static int DirectivesEnd(string source, out bool enablesF16)
    {
        enablesF16 = false;
        int end = 0;
        while (true)
        {
            int at = SkipTrivia(source, end);
            bool enable = StartsWord(source, at, "enable");
            if (!(enable || StartsWord(source, at, "requires") || StartsWord(source, at, "diagnostic"))) return end;
            int semicolon = source.IndexOf(';', at);
            if (semicolon < 0) return end;
            if (enable)
            {
                // Tokens, not text: comments and any whitespace may stand between enable, the names and the commas.
                if (source.Substring(at + 6, semicolon - at - 6).Split(',')
                    .Any(extension => StartsWord(extension, SkipTrivia(extension, 0), "f16")))
                    enablesF16 = true;
            }
            end = semicolon + 1;
        }
    }

    private static int SkipTrivia(string s, int at)
    {
        while (at < s.Length)
        {
            if (char.IsWhiteSpace(s[at])) at++;
            else if (Starts(s, at, "//")) { int line = s.IndexOf('\n', at); at = line < 0 ? s.Length : line + 1; }
            else if (Starts(s, at, "/*"))
            {
                int depth = 0;   // WGSL block comments nest
                do
                {
                    if (Starts(s, at, "/*")) { depth++; at += 2; }
                    else if (Starts(s, at, "*/")) { depth--; at += 2; }
                    else at++;
                } while (depth > 0 && at < s.Length);
            }
            else break;
        }
        return at;
    }

    private static string Rewrite(string source, Regex builtin, string helper, ref HashSet<string>? vectors)
    {
        var calls = builtin.Matches(source);
        if (calls.Count == 0) return source;
        if (vectors is null)
        {
            vectors = new HashSet<string>(StringComparer.Ordinal);
            foreach (Match m in VectorNames.Matches(source))
                for (int g = 1; g <= 3; g++)
                    if (m.Groups[g].Success) vectors.Add(m.Groups[g].Value);
        }
        var result = new System.Text.StringBuilder(source.Length + calls.Count * 5);
        int copied = 0;
        foreach (Match call in calls)
        {
            int open = call.Index + call.Length - 1;
            int close = MatchingParen(source, open);
            if (close < 0 || InConstantDeclaration(source, call.Index)
                || IsVector(source.Substring(open + 1, close - open - 1), vectors))
                continue;
            result.Append(source, copied, call.Index - copied).Append(helper);
            copied = open + 1;
        }
        return copied == 0 ? source : result.Append(source, copied, source.Length - copied).ToString();
    }

    private static int MatchingParen(string s, int open)
    {
        int depth = 0;
        for (int i = open; i < s.Length; i++)
        {
            if (s[i] == '(') depth++;
            else if (s[i] == ')' && --depth == 0) return i;
        }
        return -1;
    }

    // The statement holding the call (back to the previous ';', '{' or '}') is a const or override declaration.
    private static bool InConstantDeclaration(string s, int at)
    {
        int start = at;
        while (start > 0 && s[start - 1] != ';' && s[start - 1] != '{' && s[start - 1] != '}') start--;
        start = SkipTrivia(s, start);
        while (start < at && s[start] == '@')   // an attribute such as @id(0) before override
        {
            var attribute = Identifier.Match(s, start + 1);
            if (!attribute.Success) break;
            start = attribute.Index + attribute.Length;
            if (start < s.Length && s[start] == '(') start = MatchingParen(s, start) + 1;
            start = SkipTrivia(s, start);
        }
        return StartsWord(s, start, "const") || StartsWord(s, start, "override");
    }

    // An argument that is, or is built from, a vector: a constructor, a swizzle of two or more components, or a name
    // the source declares as a vector, unless that name is indexed or takes a single component (v.x, v[i]).
    private static bool IsVector(string argument, HashSet<string> vectors)
    {
        if (VectorSyntax.IsMatch(argument)) return true;
        foreach (Match name in Identifier.Matches(argument).Cast<Match>().Where(name => vectors.Contains(name.Value)))
        {
            int after = SkipTrivia(argument, name.Index + name.Length);
            bool component = after < argument.Length
                && (argument[after] == '[' || SingleComponent.IsMatch(argument, after));
            if (!component) return true;
        }
        return false;
    }

    private static bool StartsWord(string s, int at, string word)
        => Starts(s, at, word)
           && (at + word.Length >= s.Length || !(char.IsLetterOrDigit(s[at + word.Length]) || s[at + word.Length] == '_'));

    private static bool Starts(string s, int at, string token)
        => at >= 0 && at + token.Length <= s.Length && string.CompareOrdinal(s, at, token, 0, token.Length) == 0;
}
#endif
