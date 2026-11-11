using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The direct-PTX emitters write <c>.version 7.1</c> with the device's own <c>.target</c>; PTX 7.1 stops at sm_86, so on
/// Ada (sm_89) ptxas rejected every module ("PTX .version 7.1 does not support .target sm_89"). The loader raises the
/// version to the lowest ISA that knows the target, and leaves every module it already covers byte-for-byte unchanged
/// (the embedded sm_86 cubins are keyed by the PTX text).
/// </summary>
public sealed class DirectPtxIsaVersionTests
{
    private static string Module(string version, string target) =>
        $".version {version}\n.target {target}\n.address_size 64\n.visible .entry k() {{ ret; }}\n";

    [Theory]
    [InlineData("sm_75")]
    [InlineData("sm_80")]
    [InlineData("sm_86")]
    public void A_target_the_7_1_isa_covers_is_left_unchanged(string target)
    {
        string ptx = Module("7.1", target);
        Assert.Same(ptx, DirectPtxPtxText.EnsureIsaCoversTarget(ptx));
    }

    [Theory]
    [InlineData("sm_87", "7.4")]
    [InlineData("sm_88", "9.0")]
    [InlineData("sm_89", "7.8")]
    [InlineData("sm_90", "7.8")]
    [InlineData("sm_90a", "7.8")]
    [InlineData("sm_100", "8.6")]
    [InlineData("sm_101", "8.6")]
    [InlineData("sm_103", "8.8")]
    [InlineData("sm_110", "9.0")]
    [InlineData("sm_120", "8.7")]
    [InlineData("sm_121", "8.8")]
    public void A_newer_target_raises_the_version_to_its_minimum_isa(string target, string expected)
    {
        string raised = DirectPtxPtxText.EnsureIsaCoversTarget(Module("7.1", target));
        Assert.Equal(Module(expected, target), raised);
    }

    [Fact]
    public void An_unknown_future_target_throws_instead_of_emitting_an_unverified_isa()
    {
        Assert.Throws<System.NotSupportedException>(() =>
            DirectPtxPtxText.EnsureIsaCoversTarget(Module("7.1", "sm_199")));
    }

    [Fact]
    public void A_version_already_high_enough_is_never_lowered()
    {
        string ptx = Module("8.0", "sm_89");
        Assert.Same(ptx, DirectPtxPtxText.EnsureIsaCoversTarget(ptx));
    }

    [Fact]
    public void Text_without_a_header_is_returned_as_is()
    {
        const string fragment = "// no header\n.visible .entry k() { ret; }\n";
        Assert.Same(fragment, DirectPtxPtxText.EnsureIsaCoversTarget(fragment));
    }
}
