// Copyright (c) AiDotNet. All rights reserved.
// Convolution reaches the front end, and means the same thing it always did.
//
// CodegenOpKind had no convolution at all, so a convolution arrived as Opaque or not at
// all. The index-map layer has expressed convolution exactly since the catalog was
// written -- Window(spatial, tap, stride, pad) -- and CodegenAdjoint already derives the
// backward maps from it. What was missing was any way for a GRAPH to ask for one, which
// left all thirteen measured catalog kernels reachable only as hand-written specs.
//
// The bar here is the strongest one available: a spec TRANSLATED FROM A GRAPH must agree,
// element for element in fp64, with the hand-written catalog spec that has already been
// verified on the device at 0.000E+000. Anything weaker would pass for a translation that
// swapped a stride for a pad, which reads a shifted window and still emits.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenGraphConvolutionTests
{
    private static double[] Fill(long count, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++) v[i] = (((i * 37 + salt * 101) % 97) - 48) / 64.0;
        return v;
    }

    private static long Elements(IReadOnlyList<int> shape)
    {
        long total = 1;
        foreach (int d in shape) total *= d;
        return total;
    }

    /// <summary>
    /// Interprets a graph-translated spec and the catalog spec on the same operands and
    /// requires them to agree exactly.
    /// </summary>
    private static void AssertMatchesCatalog(CodegenGraph graph, CodegenKernelSpec reference)
    {
        Assert.True(CodegenGraphToSpec.TryTranslate(graph, "conv", out var spec, out string reason), reason);

        Assert.Equal(reference.Output.Shape.Count, spec!.Output.Shape.Count);
        for (int d = 0; d < reference.Output.Shape.Count; d++)
            Assert.Equal(reference.Output.Shape[d], spec.Output.Shape[d]);
        Assert.Equal(reference.Inputs.Count, spec.Inputs.Count);

        var operands = new double[reference.Inputs.Count][];
        for (int i = 0; i < reference.Inputs.Count; i++)
        {
            Assert.Equal(Elements(reference.Inputs[i].Shape), Elements(spec.Inputs[i].Shape));
            operands[i] = Fill(Elements(reference.Inputs[i].Shape), i + 1);
        }

        double[] want = reference.Interpret(operands);
        double[] got = spec.Interpret(operands);
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], got[i], 9);
    }

    /// <summary>
    /// The bake-off kernel: depthwise 3x3 + bias + ReLU, built as a graph, must equal the
    /// hand-written catalog spec it has been measured against all along.
    /// </summary>
    [Fact]
    public void DepthwiseConv3x3BiasRelu_FromAGraph_MatchesTheCatalogSpec()
    {
        const int N = 2, C = 8, H = 16, W = 16;
        var graph = CodegenLowering.LowerConv2D<float>(
            CodegenOpKind.DepthwiseConv2D,
            new[] { N, C, H, W }, new[] { C, 3, 3 },
            CodegenConvAttributes.Same3x3, withBias: true, withRelu: true);

        AssertMatchesCatalog(graph, CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, C, H, W));
    }

    /// <summary>The same convolution without an epilogue, isolating the gather and reduce.</summary>
    [Fact]
    public void DepthwiseConv3x3_WithoutEpilogue_MatchesTheCatalogSpec()
    {
        const int N = 2, C = 8, H = 16, W = 16;
        var graph = CodegenLowering.LowerConv2D<float>(
            CodegenOpKind.DepthwiseConv2D,
            new[] { N, C, H, W }, new[] { C, 3, 3 }, CodegenConvAttributes.Same3x3);

        var entry = CodegenKernelCatalog.Find("depthwise_conv2d_3x3");
        Assert.NotNull(entry);
        AssertMatchesCatalog(graph, entry!.Verify);
    }

    /// <summary>
    /// Dense 1x1 reduces over input channels and has no spatial window, so it exercises
    /// the channel-reduction path rather than the window path.
    /// </summary>
    [Fact]
    public void DenseConv1x1BiasRelu_FromAGraph_MatchesTheCatalogSpec()
    {
        const int N = 2, C = 8, K = 8, H = 16, W = 16;
        var graph = CodegenLowering.LowerConv2D<float>(
            CodegenOpKind.Conv2D,
            new[] { N, C, H, W }, new[] { K, C, 1, 1 },
            CodegenConvAttributes.Valid, withBias: true, withRelu: true);

        var entry = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu");
        Assert.NotNull(entry);
        AssertMatchesCatalog(graph, entry!.Verify);
    }

    /// <summary>Dense 3x3 reduces over C and both taps at once -- all three at once.</summary>
    [Fact]
    public void DenseConv3x3BiasRelu_FromAGraph_MatchesTheCatalogSpec()
    {
        const int N = 2, C = 8, K = 8, H = 16, W = 16;
        var graph = CodegenLowering.LowerConv2D<float>(
            CodegenOpKind.Conv2D,
            new[] { N, C, H, W }, new[] { K, C, 3, 3 },
            CodegenConvAttributes.Same3x3, withBias: true, withRelu: true);

        var entry = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu");
        Assert.NotNull(entry);
        AssertMatchesCatalog(graph, entry!.Verify);
    }

    /// <summary>
    /// The transposed convolution is the one whose index map divides, so an exact-division
    /// requirement rides on it. Getting the adjoint window wrong reads a shifted slice.
    /// </summary>
    [Fact]
    public void ConvTranspose3x3Stride2_FromAGraph_MatchesTheCatalogSpec()
    {
        const int N = 2, C = 8, H = 16, W = 16;
        var entry = CodegenKernelCatalog.Find("conv_transpose2d_3x3_stride2");
        Assert.NotNull(entry);

        var graph = CodegenLowering.LowerConv2D<float>(
            CodegenOpKind.ConvTranspose2D,
            new[] { N, C, H, W }, new[] { C, 3, 3 },
            new CodegenConvAttributes(2, 2, 1, 1));

        AssertMatchesCatalog(graph, entry!.Verify);
    }

    /// <summary>
    /// A convolution node without attributes must decline. Guessing stride and padding
    /// would silently change which operator was compiled.
    /// </summary>
    [Fact]
    public void ConvolutionWithoutAttributes_Declines()
    {
        var g = new CodegenGraph();
        int input = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 1, 4, 8, 8 }));
        int weights = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4, 3, 3 }));
        int conv = g.AddNode(new CodegenNode(CodegenOpKind.DepthwiseConv2D,
            new[] { input, weights }, CodegenElementType.Float32, new[] { 1, 4, 8, 8 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { conv },
            CodegenElementType.Float32, new[] { 1, 4, 8, 8 }));

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "noattr", out _, out string reason));
        Assert.Contains("CodegenConvAttributes", reason, StringComparison.Ordinal);
    }

    /// <summary>A TryTranslate path reports malformed geometry instead of throwing.</summary>
    [Fact]
    public void ConvolutionWithInvalidAttributes_Declines()
    {
        var g = new CodegenGraph();
        int input = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 1, 4, 8, 8 }));
        int weights = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4, 3, 3 }));
        int conv = g.AddNode(new CodegenNode(CodegenOpKind.DepthwiseConv2D,
            new[] { input, weights }, CodegenElementType.Float32, new[] { 1, 4, 8, 8 },
            new CodegenConvAttributes(0, 1, 1, 1)));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { conv },
            CodegenElementType.Float32, new[] { 1, 4, 8, 8 }));

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "badattrs", out _, out string reason));
        Assert.Contains("invalid convolution geometry", reason, StringComparison.Ordinal);
    }

    /// <summary>A malformed activation node declines instead of indexing a missing operand.</summary>
    [Fact]
    public void ActivationWithoutAnInput_Declines()
    {
        var g = new CodegenGraph();
        int relu = g.AddNode(new CodegenNode(CodegenOpKind.ReLU, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 8 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { relu },
            CodegenElementType.Float32, new[] { 8 }));

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "badrelu", out _, out string reason));
        Assert.Contains("exactly one input", reason, StringComparison.Ordinal);
    }

    /// <summary>
    /// An output extent the stride and padding cannot produce must be refused, not
    /// emitted against a shifted window.
    /// </summary>
    [Fact]
    public void ConvolutionWithAnImpossibleOutputShape_Declines()
    {
        var g = new CodegenGraph();
        int input = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 1, 4, 8, 8 }));
        int weights = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4, 3, 3 }));
        // 3x3 valid on 8x8 gives 6x6, not 8x8.
        int conv = g.AddNode(new CodegenNode(CodegenOpKind.DepthwiseConv2D,
            new[] { input, weights }, CodegenElementType.Float32, new[] { 1, 4, 8, 8 },
            CodegenConvAttributes.Valid));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { conv },
            CodegenElementType.Float32, new[] { 1, 4, 8, 8 }));

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "badshape", out _, out string reason));
        Assert.Contains("does not match", reason, StringComparison.Ordinal);
    }

    /// <summary>Mismatched channel counts must be refused rather than contracted wrongly.</summary>
    [Fact]
    public void DenseConvolutionWithMismatchedChannels_Declines()
    {
        var g = new CodegenGraph();
        int input = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 1, 4, 8, 8 }));
        int weights = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 6, 5, 1, 1 }));   // expects 5 input channels
        int conv = g.AddNode(new CodegenNode(CodegenOpKind.Conv2D,
            new[] { input, weights }, CodegenElementType.Float32, new[] { 1, 6, 8, 8 },
            CodegenConvAttributes.Valid));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { conv },
            CodegenElementType.Float32, new[] { 1, 6, 8, 8 }));

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "badchannels", out _, out string reason));
        Assert.Contains("input channels", reason, StringComparison.Ordinal);
    }

    /// <summary>
    /// A geometry that produces nothing must be rejected at lowering, before a graph
    /// carrying an empty output exists.
    /// </summary>
    [Fact]
    public void LowerConv2D_RefusesAGeometryThatProducesNoOutput()
    {
        Assert.Throws<ArgumentException>(() => CodegenLowering.LowerConv2D<float>(
            CodegenOpKind.DepthwiseConv2D,
            new[] { 1, 4, 2, 2 }, new[] { 4, 5, 5 }, CodegenConvAttributes.Valid));
    }

    /// <summary>Every convolution form must emit, not merely translate.</summary>
    [Theory]
    [InlineData(CodegenOpKind.DepthwiseConv2D)]
    [InlineData(CodegenOpKind.Conv2D)]
    [InlineData(CodegenOpKind.ConvTranspose2D)]
    public void TranslatedConvolution_Emits(CodegenOpKind op)
    {
        bool depthwise = op == CodegenOpKind.DepthwiseConv2D;
        var graph = CodegenLowering.LowerConv2D<float>(
            op, new[] { 2, 16, 28, 28 },
            depthwise ? new[] { 16, 3, 3 } : new[] { 16, 16, 3, 3 },
            op == CodegenOpKind.ConvTranspose2D
                ? new CodegenConvAttributes(2, 2, 1, 1)
                : CodegenConvAttributes.Same3x3,
            withBias: true, withRelu: true);

        var emitter = new PtxGraphEmitter();
        var result = emitter.Emit(graph, CodegenElementType.Float32);

        Assert.NotNull(result.Source);
        Assert.Contains(".visible .entry", result.Source!, StringComparison.Ordinal);
        Assert.NotNull(emitter.LastSpec);
        Assert.True(emitter.LastLaunchBlocks > 0);
    }

    /// <summary>
    /// The backward pass must be derivable from a graph-built convolution, since the
    /// whole point of reaching the front end is that training can use these kernels.
    /// </summary>
    [Fact]
    public void GraphBuiltConvolution_SupportsTheDerivedAdjoint()
    {
        var graph = CodegenLowering.LowerConv2D<float>(
            CodegenOpKind.DepthwiseConv2D,
            new[] { 2, 8, 16, 16 }, new[] { 8, 3, 3 }, CodegenConvAttributes.Same3x3);

        Assert.True(CodegenGraphToSpec.TryTranslate(graph, "dw", out var spec, out string reason), reason);

        var backwardData = CodegenAdjoint.BackwardData(spec!, 0);
        var backwardWeights = CodegenAdjoint.BackwardWeights(spec!, 1);

        // The gradient wrt data has the shape of the data; the gradient wrt weights has
        // the shape of the weights.
        Assert.Equal(Elements(spec!.Inputs[0].Shape), Elements(backwardData.Output.Shape));
        Assert.Equal(Elements(spec.Inputs[1].Shape), Elements(backwardWeights.Output.Shape));
    }
}
