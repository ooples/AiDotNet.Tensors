#if NET7_0_OR_GREATER
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The prelude routes an f32 module's builtin tanh, log and log2 to the accurate helpers (Dawn's D3D12 builtins are about
/// 1e-3 and 2e-5 relative off where the answer is small); WebGpuShaderAccuracyTests checks the helpers' values on Dawn.
/// </summary>
public sealed class WebGpuShaderPreludeTests
{
    private const string Kernel = @"
fn fast_tanh(x: f32) -> f32 { return x; }
fn tanh_derivative(t: f32) -> f32 { return 1.0 - t * t; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let a = tanh(1.0) + tanh (2.0) + fast_tanh(3.0) + tanh_derivative(4.0) + log2(5.0) + log(6.0) + log1p(7.0);
}
";

    private static int Count(string source, string pattern) => Regex.Matches(source, pattern).Count;

    [Fact]
    public void Builtin_tanh_log_and_log2_calls_go_to_the_helpers_and_nothing_else_is_renamed()
    {
        string prepared = WebGpuShaderPrelude.Prepare(Kernel);
        Assert.Equal(2, Count(prepared, @"[^\w]wgsl_tanh\s*\(") - 1);   // two call sites plus the helper's declaration
        Assert.Contains("fn wgsl_tanh(", prepared);
        Assert.Contains("fn wgsl_log2(", prepared);
        Assert.Contains("fast_tanh(3.0)", prepared);
        Assert.Contains("tanh_derivative(4.0)", prepared);
        Assert.Contains("wgsl_log2(5.0)", prepared);
        Assert.Contains("wgsl_log(6.0)", prepared);
        Assert.Contains("log1p(7.0)", prepared);
        Assert.DoesNotContain("wgsl_log1p", prepared);
        // The helper's own fallback for zero, negatives and non-finite values is the builtin.
        Assert.Contains("return log2(x);", prepared);
        Assert.Contains("return log(x);", prepared);
    }

    [Fact]
    public void Preparing_twice_changes_nothing_more()
    {
        string once = WebGpuShaderPrelude.Prepare(Kernel);
        Assert.Equal(once, WebGpuShaderPrelude.Prepare(once));
    }

    [Fact]
    public void A_module_that_enables_f16_keeps_the_builtins()
    {
        string source = "enable f16;\n" + Kernel;
        string prepared = WebGpuShaderPrelude.Prepare(source);
        Assert.DoesNotContain("wgsl_tanh", prepared);
        Assert.DoesNotContain("wgsl_log2", prepared);
        Assert.DoesNotContain("wgsl_log(", prepared);
    }

    [Theory]
    [InlineData("enable subgroups, f16;\n")]
    [InlineData("enable  f16 ;\n")]
    [InlineData("// header\n/* a /* nested */ comment */\nenable f16;\n")]
    [InlineData("enable\nf16;\n")]
    [InlineData("enable\t\tf16;\n")]
    [InlineData("enable /* half floats */ f16;\n")]
    [InlineData("enable subgroups, // wide ops\n f16;\n")]
    public void Any_enable_that_lists_f16_keeps_the_builtins(string directives)
    {
        string prepared = WebGpuShaderPrelude.Prepare(directives + Kernel);
        Assert.DoesNotContain("wgsl_tanh", prepared);
        Assert.DoesNotContain("wgsl_log", prepared);
    }

    // exp(2|x|) overflows f32 past |x| = 44.4 and WGSL leaves an overflowed result indeterminate, so the helper must
    // answer the saturated value before it evaluates exp; and the cutoff must be one at which tanh already rounds to 1.
    [Fact]
    public void The_tanh_helper_saturates_before_its_exponential_can_overflow()
    {
        string prepared = WebGpuShaderPrelude.Prepare(Kernel);
        int helper = prepared.IndexOf("fn wgsl_tanh(", System.StringComparison.Ordinal);
        int saturate = prepared.IndexOf("if (a >= 9.02) { return select(-1.0, 1.0, x >= 0.0); }", helper, System.StringComparison.Ordinal);
        int exponential = prepared.IndexOf("exp(2.0 * a)", helper, System.StringComparison.Ordinal);
        Assert.True(saturate > helper, "The tanh helper has no saturation branch.");
        Assert.True(exponential > saturate, "The saturation branch must come before exp.");
        Assert.Equal(1.0f, (float)System.Math.Tanh(9.02));
        Assert.True(System.Math.Exp(2.0 * 9.02) < float.MaxValue, "exp is still finite at the cutoff.");
    }

    [Theory]
    [InlineData("// enable f16;\n")]
    [InlineData("enable f16x;\n")]
    [InlineData("enable subgroups;\n")]
    public void An_enable_that_does_not_list_f16_still_rewrites(string directives)
    {
        Assert.Contains("fn wgsl_tanh(", WebGpuShaderPrelude.Prepare(directives + Kernel));
    }

    // A let or var takes its initializer's type, an element of an array of vectors is a vector, and neither may reach
    // the scalar helper.
    [Fact]
    public void Inferred_vectors_and_vector_array_elements_keep_the_builtin()
    {
        const string source = @"
fn f(v: vec2<f32>, a: array<vec3<f32>, 4>) -> f32 {
    let u = v;
    var w = u * 2.0;
    let t = tanh(u) + tanh(w);
    let e = tanh(a[1]);
    return tanh(u.x) + tanh(a[1].y) + tanh(a[2][0]) + t.x + e.x;
}";
        string prepared = WebGpuShaderPrelude.Prepare(source);
        foreach (var kept in new[] { "tanh(u) ", "tanh(w)", "= tanh(a[1]);" })
            Assert.Contains(kept, prepared);
        foreach (var rewritten in new[] { "wgsl_tanh(u.x)", "wgsl_tanh(a[1].y)", "wgsl_tanh(a[2][0])" })
            Assert.Contains(rewritten, prepared);
    }

    // dot and length reduce their vector operands to a scalar, as does a function declared with a scalar result: the
    // accurate path must apply to them.
    [Fact]
    public void Scalar_results_of_vector_operands_are_rewritten()
    {
        const string source = @"
fn norm(v: vec3<f32>) -> f32 { return v.x; }
fn f(v: vec3<f32>) -> f32 {
    return tanh(dot(v, v)) + log(length(v)) + log2(norm(v)) + tanh(v).x;
}";
        string prepared = WebGpuShaderPrelude.Prepare(source);
        Assert.Contains("wgsl_tanh(dot(v, v))", prepared);
        Assert.Contains("wgsl_log(length(v))", prepared);
        Assert.Contains("wgsl_log2(norm(v))", prepared);
        Assert.Contains(" tanh(v).x", prepared);
    }

    // A ';' inside a comment does not end the directive: the helpers must land after the real one.
    [Fact]
    public void A_semicolon_inside_a_comment_does_not_end_a_directive()
    {
        const string source = "enable /* ; */ subgroups;\nfn f(x: f32) -> f32 { return tanh(x); }";
        string prepared = WebGpuShaderPrelude.Prepare(source);
        int directive = prepared.IndexOf("subgroups;", System.StringComparison.Ordinal);
        int helper = prepared.IndexOf("fn wgsl_tanh(", System.StringComparison.Ordinal);
        Assert.True(helper > directive, "The helper was inserted before the directive ended.");
        Assert.Contains("return wgsl_tanh(x);", prepared);
        Assert.StartsWith("enable /* ; */ subgroups;", prepared);
        // An extension named only inside a comment is not enabled.
        Assert.Contains("fn wgsl_tanh(", WebGpuShaderPrelude.Prepare("enable /* f16, */ subgroups;\n" + Kernel));
    }

    // Module-scope initializers, attributes and array counts need const-expressions; a helper call is not one.
    [Fact]
    public void Calls_that_must_be_evaluated_early_keep_the_builtin()
    {
        const string source = @"
var<private> scale: f32 = log(2.0);
const half: f32 = log2(1.5);
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(u32(log2(64.0)))
fn main() {
    var local: array<f32, u32(log2(16.0))>;
    const c = tanh(0.5);
    data[0] = log(data[1]) + tanh(data[2]) + local[0] + c + scale + half;
}";
        string prepared = WebGpuShaderPrelude.Prepare(source);
        foreach (var kept in new[]
                 {
                     "scale: f32 = log(2.0);", "half: f32 = log2(1.5);", "@workgroup_size(u32(log2(64.0)))",
                     "array<f32, u32(log2(16.0))>", "const c = tanh(0.5);",
                 })
            Assert.Contains(kept, prepared);
        Assert.Contains("wgsl_log(data[1]) + wgsl_tanh(data[2])", prepared);
    }

    [Fact]
    public void Mentioning_f16_in_a_comment_does_not_turn_the_rewrite_off()
    {
        string prepared = WebGpuShaderPrelude.Prepare("// TODO enable f16 later\n" + Kernel);
        Assert.Contains("wgsl_tanh(1.0)", prepared);
    }

    [Fact]
    public void Helpers_go_after_directives_that_follow_a_comment()
    {
        string prepared = WebGpuShaderPrelude.Prepare("// header\nenable subgroups;\n" + Kernel);
        Assert.True(prepared.IndexOf("enable subgroups;", System.StringComparison.Ordinal)
                    < prepared.IndexOf("fn wgsl_tanh(", System.StringComparison.Ordinal));
    }

    [Fact]
    public void Vector_arguments_keep_the_builtin_and_their_components_do_not()
    {
        const string source = @"
struct S { pos: vec3<f32>, w: f32 }
fn dir() -> vec2<f32> { return vec2<f32>(1.0, 2.0); }
fn f(v: vec4<f32>, s: S) -> f32 {
    let u = vec3<f32>(1.0);
    let a = tanh(v) + tanh(v * 2.0) + tanh(vec2<f32>(1.0)) + tanh(u.xy) + tanh(dir()) + tanh(s.pos);
    return tanh(v.x) + log(v[1]) + log2(s.w) + tanh(u.z);
}
";
        string prepared = WebGpuShaderPrelude.Prepare(source);
        foreach (var kept in new[] { "tanh(v)", "tanh(v * 2.0)", "tanh(vec2<f32>(1.0))", "tanh(u.xy)", "tanh(dir())", "tanh(s.pos)" })
            Assert.Contains(" " + kept, prepared);
        foreach (var rewritten in new[] { "wgsl_tanh(v.x)", "wgsl_log(v[1])", "wgsl_log2(s.w)", "wgsl_tanh(u.z)" })
            Assert.Contains(rewritten, prepared);
    }

    [Fact]
    public void Constant_declarations_keep_the_builtin()
    {
        const string source = @"
const INV_LN2 = 1.0 / log(2.0);
@id(0) override SCALE: f32 = tanh(0.5);
fn f(x: f32) -> f32 { const C = log2(8.0); return tanh(x) * C * INV_LN2 * SCALE; }
";
        string prepared = WebGpuShaderPrelude.Prepare(source);
        Assert.Contains("1.0 / log(2.0)", prepared);
        Assert.Contains("= tanh(0.5)", prepared);
        Assert.Contains("= log2(8.0)", prepared);
        Assert.Contains("return wgsl_tanh(x)", prepared);
    }

    [Fact]
    public void A_source_with_none_of_them_is_returned_unchanged()
    {
        const string source = "@compute @workgroup_size(1) fn main() { let a = exp(1.0); }";
        Assert.Same(source, WebGpuShaderPrelude.Prepare(source));
    }
}
#endif
