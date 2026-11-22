#if NET7_0_OR_GREATER
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The prelude routes an f32 module's builtin tanh, log and log2 to the accurate helpers (Dawn's D3D12 builtins are about
/// 1e-3 and 2e-5 relative off where the answer is small); the WebGPU kernel tests check the values on a device.
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

    [Fact]
    public void A_source_with_none_of_them_is_returned_unchanged()
    {
        const string source = "@compute @workgroup_size(1) fn main() { let a = exp(1.0); }";
        Assert.Same(source, WebGpuShaderPrelude.Prepare(source));
    }
}
#endif
