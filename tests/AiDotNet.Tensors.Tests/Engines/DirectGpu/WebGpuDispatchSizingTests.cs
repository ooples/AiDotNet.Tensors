#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// WebGPU dispatch sizing. A shader covers one element per invocation, so a dispatch must never be clamped: that
/// leaves the tail of its output uncomputed (PR #1231 review). A 1-D dispatch past the device's workgroups per dimension
/// (65,535 by default, WebGPU's default when uninitialized) folds onto a 2-D grid instead (WebGpuDispatchFold); only
/// work past the folded grid is refused.
/// </summary>
public sealed class WebGpuDispatchSizingTests
{
    [Fact]
    public void OneDimensionalDispatch_PastTheLimit_KeepsEveryWorkgroup()
    {
        var device = WebGpuDevice.Instance;
        int limit = device.MaxWorkgroupsPerDimension;
        Assert.Equal((limit, 256), device.CalculateWorkgroups1D(limit * 256));
        Assert.Equal((1, 256), device.CalculateWorkgroups1D(1));
        Assert.Equal((limit + 1, 256), device.CalculateWorkgroups1D(limit * 256 + 1));
        Assert.Equal((int.MaxValue / 256 + 1, 256), device.CalculateWorkgroups1D(int.MaxValue));
    }

    [Fact]
    public void TwoDimensionalDispatch_RefusesAnAxisBeyondTheLimit()
    {
        var device = WebGpuDevice.Instance;
        var (x, y, sx, sy) = device.CalculateWorkgroups2D(64, 64);
        Assert.True(x * sx >= 64 && y * sy >= 64);
        Assert.Throws<GpuCapabilityUnavailableException>(() => device.CalculateWorkgroups2D(int.MaxValue / 2, 1));
    }

    // The folded grid covers every workgroup, each dimension within the limit, with fewer than Y surplus groups.
    [Theory]
    [InlineData(1L)]
    [InlineData(65535L)]
    [InlineData(65536L)]
    [InlineData(131071L)]
    [InlineData(8388608L)]                 // int.MaxValue elements at 256 per group
    [InlineData(65535L * 65535L)]
    public void Grid_CoversEveryWorkgroup_WithinTheLimit(long groups)
    {
        const int limit = 65535;
        var (x, y) = WebGpuDispatchFold.Grid(groups, limit);
        Assert.InRange(x, 1, limit);
        Assert.InRange(y, 1, limit);
        Assert.True((long)x * y >= groups, $"{x} x {y} < {groups}");
        Assert.True((long)x * y - groups < y, $"{x} x {y} leaves {(long)x * y - groups} surplus groups");
        Assert.Throws<GpuCapabilityUnavailableException>(() => WebGpuDispatchFold.Grid(65535L * 65535L + 1, limit));
    }

    private const string Elementwise = @"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&b)) { return; }
    b[idx] = max(a[idx], 0.0);
}
@compute @workgroup_size(64)
fn other(@builtin(global_invocation_id) gid: vec3<u32>) { b[gid.x] = 0.0; }
";

    [Fact]
    public void FoldSource_GivesTheBuiltinsTheirOneDimensionalValues()
    {
        string? folded = WebGpuDispatchFold.FoldSource(Elementwise, "relu");
        Assert.NotNull(folded);
        string text = folded ?? string.Empty;
        // The entry point takes the workgroup builtins and rebuilds its own from the flattened workgroup index.
        Assert.Contains("fn relu(@builtin(local_invocation_id) aidn_fold_lid: vec3<u32>, @builtin(workgroup_id) aidn_fold_wid: vec3<u32>, @builtin(num_workgroups) aidn_fold_nwg: vec3<u32>) {", text);
        Assert.Contains("let aidn_fold_flat = aidn_fold_wid.x + aidn_fold_wid.y * aidn_fold_nwg.x;", text);
        Assert.Contains("if (aidn_fold_flat >= aidn_fold_groups.x) { return; }", text);
        Assert.Contains("let gid = vec3<u32>(aidn_fold_flat * u32(256) + aidn_fold_lid.x, aidn_fold_lid.y, aidn_fold_lid.z);", text);
        Assert.EndsWith(WebGpuDispatchFold.GroupsDeclaration + "\n", text);
        // The other entry point and the body are untouched.
        Assert.Contains("fn other(@builtin(global_invocation_id) gid: vec3<u32>) { b[gid.x] = 0.0; }", text);
        Assert.Contains("    let idx = gid.x;", text);
    }

    [Fact]
    public void FoldSource_RebuildsWorkgroupIdAndCount_AndKeepsTheShadersLocalId()
    {
        const string rows = @"
@compute @workgroup_size(WG, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) n: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wid.x;
}";
        string text = WebGpuDispatchFold.FoldSource(rows, "main") ?? string.Empty;
        Assert.Contains("fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) aidn_fold_wid: vec3<u32>, @builtin(num_workgroups) aidn_fold_nwg: vec3<u32>) {", text);
        Assert.Contains("let wid = vec3<u32>(aidn_fold_flat, 0u, 0u);", text);
        Assert.Contains("let n = vec3<u32>(aidn_fold_groups.x, 1u, 1u);", text);
        Assert.DoesNotContain("aidn_fold_lid", text);
    }

    [Theory]
    [InlineData("missing")]   // no such entry point
    [InlineData("helper")]    // a function that is not @compute
    [InlineData("st")]        // a parameter that is not a builtin
    public void FoldSource_RefusesWhatItCannotFold(string entry)
    {
        const string source = @"
struct In { @builtin(global_invocation_id) gid: vec3<u32> }
fn helper(x: u32) -> u32 { return x; }
@compute @workgroup_size(64)
fn st(input: In) { }
";
        Assert.Null(WebGpuDispatchFold.FoldSource(source, entry));
    }
}
#endif
