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
    // Names declared as arrays of vectors: an element of one is a vector.
    private static readonly Regex VectorArrayNames = new(
        @"(\w+)\s*:\s*array\s*<\s*vec[234]", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    // Functions the source declares with a scalar result: a call to one is a scalar whatever its arguments are.
    private static readonly Regex ScalarFunctionNames = new(
        @"fn\s+(\w+)\s*\([^)]*\)\s*->\s*(?:f32|f16|i32|u32|bool)\b", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    // A declaration whose type is inferred from its initializer (let u = v;).
    private static readonly Regex InferredDeclaration = new(
        @"(?:let|var)\s+(\w+)\s*=\s*([^;]+);", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
    // Builtins that reduce vectors to a scalar, and the scalar conversions.
    private static readonly string[] ScalarBuiltins =
        { "dot", "length", "distance", "determinant", "f32", "i32", "u32", "bool", "wgsl_tanh", "wgsl_log", "wgsl_log2" };

    // What the source says about its names' types, gathered once per Prepare.
    private sealed class SourceTypes
    {
        public HashSet<string> Vectors { get; } = new(StringComparer.Ordinal);
        public HashSet<string> VectorArrays { get; } = new(StringComparer.Ordinal);
        public HashSet<string> ScalarFunctions { get; } = new(ScalarBuiltins, StringComparer.Ordinal);
    }
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
            if (!source.Contains("fn wgsl_tanh(")) source = Rewrite(source, BuiltinTanh, "tanh", "wgsl_tanh(");
            if (!source.Contains("fn wgsl_log2(")) source = Rewrite(source, BuiltinLog2, "log2", "wgsl_log2(");
            if (!source.Contains("fn wgsl_log(")) source = Rewrite(source, BuiltinLog, "log", "wgsl_log(");
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

    // The source with every comment's text replaced by spaces (line breaks kept), so its offsets are the source's and
    // structure - a directive's terminator, a statement's start, brace depth, a call - is read from code only. WGSL has
    // no string literals, so comments are the only text that is not code.
    private static string MaskComments(string s)
    {
        System.Text.StringBuilder? masked = null;
        int at = 0;
        while (at < s.Length)
        {
            if (!Starts(s, at, "//") && !Starts(s, at, "/*")) { at++; continue; }
            int end = SkipComment(s, at);
            masked ??= new System.Text.StringBuilder(s);
            for (int k = at; k < end; k++)
                if (masked[k] != '\n') masked[k] = ' ';
            at = end;
        }
        return masked?.ToString() ?? s;
    }

    // The offset just past the comment that starts at `at`.
    private static int SkipComment(string s, int at)
    {
        if (Starts(s, at, "//"))
        {
            int line = s.IndexOf('\n', at);
            return line < 0 ? s.Length : line + 1;
        }
        int depth = 0;   // WGSL block comments nest
        do
        {
            if (Starts(s, at, "/*")) { depth++; at += 2; }
            else if (Starts(s, at, "*/")) { depth--; at += 2; }
            else at++;
        } while (depth > 0 && at < s.Length);
        return at;
    }

    // The offset just past the leading directives, skipping whitespace and comments between them; and whether an enable
    // among them lists f16. Read from the comment-masked source: a ';' or a name inside a comment is not the
    // directive's.
    private static int DirectivesEnd(string source, out bool enablesF16)
    {
        enablesF16 = false;
        string code = MaskComments(source);
        int end = 0;
        while (true)
        {
            int at = SkipTrivia(code, end);
            bool enable = StartsWord(code, at, "enable");
            if (!(enable || StartsWord(code, at, "requires") || StartsWord(code, at, "diagnostic"))) return end;
            int semicolon = code.IndexOf(';', at);
            if (semicolon < 0) return end;
            if (enable && code.Substring(at + 6, semicolon - at - 6).Split(',').Any(extension => extension.Trim() == "f16"))
                enablesF16 = true;
            end = semicolon + 1;
        }
    }

    private static int SkipTrivia(string s, int at)
    {
        while (at < s.Length)
        {
            if (char.IsWhiteSpace(s[at])) at++;
            else if (Starts(s, at, "//") || Starts(s, at, "/*")) at = SkipComment(s, at);
            else break;
        }
        return at;
    }

    private static SourceTypes ReadTypes(string code)
    {
        var types = new SourceTypes();
        foreach (Match m in VectorNames.Matches(code))
            for (int g = 1; g <= 3; g++)
                if (m.Groups[g].Success) types.Vectors.Add(m.Groups[g].Value);
        foreach (Match m in VectorArrayNames.Matches(code)) types.VectorArrays.Add(m.Groups[1].Value);
        foreach (Match m in ScalarFunctionNames.Matches(code)) types.ScalarFunctions.Add(m.Groups[1].Value);
        // Inferred declarations take their initializer's type (let u = v;), which may itself be inferred: to a fixed
        // point. Each pass adds at least one name or ends the loop, and there are finitely many declarations.
        var inferred = InferredDeclaration.Matches(code).Cast<Match>().ToList();
        bool added = true;
        while (added)
        {
            added = false;
            foreach (var declaration in inferred.Where(d => !types.Vectors.Contains(d.Groups[1].Value)
                                                            && IsVector(d.Groups[2].Value, types)))
            {
                types.Vectors.Add(declaration.Groups[1].Value);
                added = true;
            }
        }
        return types;
    }

    // A function declaration: its name, parameter list and body, by offset into the comment-masked source.
    private readonly struct FunctionSpan
    {
        public FunctionSpan(string name, int parametersOpen, int parametersClose, int bodyOpen, int bodyClose)
        {
            Name = name;
            ParametersOpen = parametersOpen;
            ParametersClose = parametersClose;
            BodyOpen = bodyOpen;
            BodyClose = bodyClose;
        }

        public string Name { get; }
        public int ParametersOpen { get; }
        public int ParametersClose { get; }
        public int BodyOpen { get; }
        public int BodyClose { get; }
    }

    private static readonly Regex FunctionHeader = new(@"\bfn\s+(\w+)\s*\(", RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));

    private static List<FunctionSpan> ReadFunctions(string code)
    {
        var functions = new List<FunctionSpan>();
        foreach (Match header in FunctionHeader.Matches(code))
        {
            int open = header.Index + header.Length - 1;
            int close = Matching(code, open, '(', ')');
            int body = close < 0 ? -1 : code.IndexOf('{', close);
            int end = body < 0 ? -1 : Matching(code, body, '{', '}');
            if (end >= 0) functions.Add(new FunctionSpan(header.Groups[1].Value, open, close, body, end));
        }
        return functions;
    }

    // The source as the function `visible` sees it: module-scope declarations and every function's signature, with
    // the parameters and body of every OTHER function blanked. Parameters and locals are function-scoped in WGSL, so
    // a name declared a vector in one function says nothing about the same name in another.
    private static string ScopeOf(string code, List<FunctionSpan> functions, int visible)
    {
        var scoped = new System.Text.StringBuilder(code);
        for (int f = 0; f < functions.Count; f++)
        {
            if (f == visible) continue;
            Blank(scoped, functions[f].ParametersOpen + 1, functions[f].ParametersClose);
            Blank(scoped, functions[f].BodyOpen + 1, functions[f].BodyClose);
        }
        return scoped.ToString();
    }

    private static void Blank(System.Text.StringBuilder text, int from, int to)
    {
        for (int k = from; k < to; k++)
            if (text[k] != '\n') text[k] = ' ';
    }

    private static string Rewrite(string source, Regex builtin, string name, string helper)
    {
        string code = MaskComments(source);
        var calls = builtin.Matches(code);
        if (calls.Count == 0) return source;
        var functions = ReadFunctions(code);
        // WGSL lets a module function hide a builtin: calls to that name are calls to the module's function.
        if (functions.Exists(function => function.Name == name)) return source;
        var typesByFunction = new SourceTypes?[functions.Count];
        var result = new System.Text.StringBuilder(source.Length + calls.Count * 5);
        int copied = 0;
        foreach (Match call in calls)
        {
            int open = call.Index + call.Length - 1;
            int close = Matching(code, open, '(', ')');
            int enclosing = functions.FindIndex(function => function.BodyOpen < call.Index && call.Index < function.BodyClose);
            // Outside every function body the call is at module scope, where only const-expressions are allowed.
            if (close < 0 || enclosing < 0 || IsEarlyEvaluated(code, call.Index)) continue;
            var types = typesByFunction[enclosing] ??= ReadTypes(ScopeOf(code, functions, enclosing));
            if (IsVector(code.Substring(open + 1, close - open - 1), types)) continue;
            result.Append(source, copied, call.Index - copied).Append(helper);
            copied = open + 1;
        }
        return copied == 0 ? source : result.Append(source, copied, source.Length - copied).ToString();
    }

    private static int MatchingParen(string s, int open) => Matching(s, open, '(', ')');

    private static int Matching(string s, int open, char opening, char closing)
    {
        int depth = 0;
        for (int i = open; i < s.Length; i++)
        {
            if (s[i] == opening) depth++;
            else if (s[i] == closing && --depth == 0) return i;
        }
        return -1;
    }

    // Whether the call at `at` sits where WGSL requires a const- or override-expression, which a call to a user
    // function (the helper) is not: anywhere at module scope (a const, override or var initializer, an attribute, a
    // global's array count), and inside a function a const declaration, an attribute's arguments or an array's
    // element count. `code` is the comment-masked source.
    private static bool IsEarlyEvaluated(string code, int at)
    {
        if (BraceDepth(code, at) <= 0) return true;
        int statement = at;
        while (statement > 0 && code[statement - 1] != ';' && code[statement - 1] != '{' && code[statement - 1] != '}') statement--;
        return IsConstantDeclaration(code, statement, at) || InAttributeArguments(code, statement, at)
               || InArrayElementCount(code, statement, at);
    }

    private static int BraceDepth(string code, int at)
    {
        int braces = 0;
        for (int i = 0; i < at; i++)
        {
            if (code[i] == '{') braces++;
            else if (code[i] == '}') braces--;
        }
        return braces;
    }

    // The statement starting at `statement` is a const or override declaration (after any attributes).
    private static bool IsConstantDeclaration(string code, int statement, int at)
    {
        int start = SkipTrivia(code, statement);
        while (start < at && code[start] == '@')   // an attribute such as @id(0) before override
        {
            var attribute = Identifier.Match(code, start + 1);
            if (!attribute.Success) break;
            start = attribute.Index + attribute.Length;
            if (start < code.Length && code[start] == '(') start = MatchingParen(code, start) + 1;
            start = SkipTrivia(code, start);
        }
        return StartsWord(code, start, "const") || StartsWord(code, start, "override");
    }

    // Inside an attribute's argument list: an unclosed '(' before the call that follows @name.
    private static bool InAttributeArguments(string code, int statement, int at)
    {
        int depth = 0;
        for (int i = at - 1; i >= statement; i--)
        {
            if (code[i] == ')') { depth++; continue; }
            if (code[i] != '(') continue;
            if (depth > 0) { depth--; continue; }
            if (FollowsAttributeName(code, statement, i)) return true;
        }
        return false;
    }

    // The '(' at `open` comes straight after "@name".
    private static bool FollowsAttributeName(string code, int statement, int open)
    {
        int name = open;
        while (name > statement && char.IsWhiteSpace(code[name - 1])) name--;
        int word = name;
        while (word > statement && (char.IsLetterOrDigit(code[word - 1]) || code[word - 1] == '_')) word--;
        return word < name && word > statement && code[word - 1] == '@';
    }

    // Inside an array type's template list (its element count): an "array<" not yet closed before the call.
    private static bool InArrayElementCount(string code, int statement, int at)
    {
        int array = code.IndexOf("array", statement, at - statement, StringComparison.Ordinal);
        while (array >= 0)
        {
            int angle = SkipTrivia(code, array + 5);
            bool wholeWord = array == 0 || !(char.IsLetterOrDigit(code[array - 1]) || code[array - 1] == '_');
            if (wholeWord && angle < at && code[angle] == '<' && !ClosesBefore(code, angle, at)) return true;
            int next = array + 5;
            array = next < at ? code.IndexOf("array", next, at - next, StringComparison.Ordinal) : -1;
        }
        return false;
    }

    // The template list opened at `angle` is closed before `at`.
    private static bool ClosesBefore(string code, int angle, int at)
    {
        int open = 1;
        for (int i = angle + 1; i < at && open > 0; i++)
        {
            if (code[i] == '<') open++;
            else if (code[i] == '>') open--;
        }
        return open == 0;
    }

    // Whether an argument's VALUE is a vector. Calls with a scalar result (dot, length, a function declared -> f32)
    // are scalars whatever they are given, so they are set aside first; what is left is a vector when it holds a
    // vector constructor, a swizzle of two or more components, an element of an array of vectors, or a name the
    // source declares (or infers) as a vector - unless that name is indexed or takes a single component (v.x, v[i]).
    private static bool IsVector(string argument, SourceTypes types)
    {
        argument = WithoutScalarCalls(argument, types);
        if (VectorSyntax.IsMatch(argument)) return true;
        foreach (Match name in Identifier.Matches(argument))
        {
            int after = SkipTrivia(argument, name.Index + name.Length);
            if (types.Vectors.Contains(name.Value))
            {
                bool component = after < argument.Length
                    && (argument[after] == '[' || SingleComponent.IsMatch(argument, after));
                if (!component) return true;
            }
            else if (types.VectorArrays.Contains(name.Value) && after < argument.Length && argument[after] == '[')
            {
                int close = Matching(argument, after, '[', ']');
                if (close < 0) continue;
                int next = SkipTrivia(argument, close + 1);
                bool component = next < argument.Length
                    && (argument[next] == '[' || SingleComponent.IsMatch(argument, next));
                if (!component) return true;
            }
        }
        return false;
    }

    // The argument with every call to a scalar-valued function replaced by a scalar literal, outermost first.
    private static string WithoutScalarCalls(string argument, SourceTypes types)
    {
        System.Text.StringBuilder? result = null;
        int copied = 0;
        // Names of scalar-valued functions that are not members (x.length is a field, not a call).
        foreach (Match name in Identifier.Matches(argument).Cast<Match>().Where(name =>
                     types.ScalarFunctions.Contains(name.Value) && (name.Index == 0 || argument[name.Index - 1] != '.')))
        {
            if (name.Index < copied) continue;   // inside a call already replaced
            int open = SkipTrivia(argument, name.Index + name.Length);
            if (open >= argument.Length || argument[open] != '(') continue;
            int close = Matching(argument, open, '(', ')');
            if (close < 0) continue;
            result ??= new System.Text.StringBuilder(argument.Length);
            result.Append(argument, copied, name.Index - copied).Append("0.0");
            copied = close + 1;
        }
        return result is null ? argument : result.Append(argument, copied, argument.Length - copied).ToString();
    }

    private static bool StartsWord(string s, int at, string word)
        => Starts(s, at, word)
           && (at + word.Length >= s.Length || !(char.IsLetterOrDigit(s[at + word.Length]) || s[at + word.Length] == '_'));

    private static bool Starts(string s, int at, string token)
        => at >= 0 && at + token.Length <= s.Length && string.CompareOrdinal(s, at, token, 0, token.Length) == 0;
}
#endif
