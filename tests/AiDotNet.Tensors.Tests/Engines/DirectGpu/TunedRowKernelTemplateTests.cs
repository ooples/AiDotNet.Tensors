using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The tuned-kernel template (#1116) emits one definition as OpenCL C and as Metal Shading Language. These checks run on
/// every host, Metal or not: each dialect declares every kernel, neither leaks the other's syntax, and each Metal kernel
/// binds exactly the buffer slots MetalBackend.TunedKernels fills, in order. The OpenCL instance runs on a device in
/// OpenClTunedKernelTests, so the shared bodies are exercised there.
/// </summary>
public sealed class TunedRowKernelTemplateTests
{
    private static readonly System.TimeSpan RegexTimeout = System.TimeSpan.FromSeconds(5);

    private static string Metal => MetalTunedRowKernels.GetSource();
    private static string OpenCl => TunedRowKernelTemplate.GetSource(TunedKernelDialect.OpenCl);

    [Fact]
    public void EveryKernel_IsDeclaredInBothDialects()
    {
        foreach (string name in TunedRowKernelTemplate.GetKernelNames())
        {
            Assert.Matches(new Regex(@"__kernel\b[^{]*\bvoid " + name + @"\(", RegexOptions.None, RegexTimeout), OpenCl);
            Assert.Contains("kernel void " + name + "(", Metal);
        }
        foreach (string reference in new[]
                 {
                     MetalTunedRowKernels.LayerNormBackwardInputReference, MetalTunedRowKernels.LayerNormGradParamsReference,
                     MetalTunedRowKernels.RmsNormBackwardInputReference, MetalTunedRowKernels.RmsNormGradGammaReference,
                 })
            Assert.Contains("kernel void " + reference + "(", Metal);
    }

    [Theory]
    [InlineData("__global")]
    [InlineData("__local")]
    [InlineData("__kernel")]
    [InlineData("get_global_id")]
    [InlineData("get_local_id")]
    [InlineData("get_group_id")]
    [InlineData("CLK_LOCAL_MEM_FENCE")]
    [InlineData("reqd_work_group_size")]
    public void Metal_HasNoOpenClSyntax(string token) => Assert.DoesNotContain(token, Metal);

    [Theory]
    [InlineData("[[")]
    [InlineData("threadgroup")]
    [InlineData("metal_stdlib")]
    [InlineData("device ")]
    public void OpenCl_HasNoMetalSyntax(string token) => Assert.DoesNotContain(token, OpenCl);

    // Buffers then scalars, as MetalBackend.TunedKernels / TunedConvolution bind them.
    public static IEnumerable<object[]> MetalBindings()
    {
        foreach (int l in TunedRowKernelTemplate.RowLanes)
        {
            yield return new object[] { TunedRowKernelTemplate.SoftmaxName(l), 2 + 2 };
            yield return new object[] { TunedRowKernelTemplate.SoftmaxBackwardName(l), 3 + 2 };
            yield return new object[] { TunedRowKernelTemplate.LayerNormName(l), 6 + 3 };
            yield return new object[] { TunedRowKernelTemplate.LayerNormBackwardName(l), 6 + 2 };
        }
        foreach (int r in TunedRowKernelTemplate.ColumnRowLanes)
        {
            yield return new object[] { TunedRowKernelTemplate.LayerNormGradParamsName(r), 6 + 2 };
            yield return new object[] { TunedRowKernelTemplate.RmsNormGradGammaName(r), 4 + 2 };
        }
        yield return new object[] { TunedRowKernelTemplate.Im2ColName, 2 + 14 };
        yield return new object[] { MetalTunedRowKernels.LayerNormBackwardInputReference, 6 + 2 };
        yield return new object[] { MetalTunedRowKernels.LayerNormGradParamsReference, 6 + 2 };
        yield return new object[] { MetalTunedRowKernels.RmsNormBackwardInputReference, 5 + 2 };
        yield return new object[] { MetalTunedRowKernels.RmsNormGradGammaReference, 4 + 2 };
    }

    [Theory]
    [MemberData(nameof(MetalBindings))]
    public void MetalKernel_BindsContiguousBufferSlots_MatchingTheHost(string kernel, int bindings)
    {
        string source = Metal;
        int start = source.IndexOf("kernel void " + kernel + "(", System.StringComparison.Ordinal);
        Assert.True(start >= 0, kernel + " is not declared");
        int end = source.IndexOf(')', source.IndexOf("[[thread_position_in", start, System.StringComparison.Ordinal));
        string signature = source.Substring(start, end - start);
        int[] slots = Regex.Matches(signature, @"\[\[buffer\((\d+)\)\]\]", RegexOptions.None, RegexTimeout)
            .Cast<Match>().Select(m => int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        Assert.Equal(Enumerable.Range(0, bindings), slots);
    }
}
