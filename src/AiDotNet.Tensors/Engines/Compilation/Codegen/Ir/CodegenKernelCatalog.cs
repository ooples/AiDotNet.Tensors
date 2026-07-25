// Copyright (c) AiDotNet. All rights reserved.
// The assembly line's input: every kernel the conveyor knows how to verify,
// release and benchmark, declared once as a spec rather than as hand-written PTX.
//
// Adding a kernel means adding an entry here. The three conveyor stages
// (--kernel-verify, --kernel-release, --kernel-bench) then apply to it without
// any per-kernel code, which is the whole point: 800 kernels cannot be carried
// through three gates each by hand, but they can be carried through by a loop
// over this catalog.
//
// Each entry declares the SAME kernel at two shapes, because the two shapes
// answer different questions:
//
//   Verify shape   small, so the fp64 CPU reference is cheap and the on-device
//                  comparison is fast. Correctness does not need a large shape.
//   Bench shape    device-filling. Phase 0.5 measured that small shapes report
//                  launch and synchronisation latency, not kernel quality: at
//                  4 blocks on a 68-SM device two bit-identical kernels differed
//                  by 1.57x from measurement ordering alone.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>One catalog row: a kernel declared at a correctness shape and a timing shape.</summary>
public sealed class CodegenCatalogEntry
{
    /// <summary>Creates a catalog entry.</summary>
    public CodegenCatalogEntry(string name, string summary, CodegenKernelSpec verify, CodegenKernelSpec bench)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Summary = summary ?? throw new ArgumentNullException(nameof(summary));
        Verify = verify ?? throw new ArgumentNullException(nameof(verify));
        Bench = bench ?? throw new ArgumentNullException(nameof(bench));
    }

    /// <summary>Stable catalog identifier, used on the command line and in manifests.</summary>
    public string Name { get; }

    /// <summary>One line describing what the kernel computes.</summary>
    public string Summary { get; }

    /// <summary>Small shape: cheap fp64 reference, used by the verify stage.</summary>
    public CodegenKernelSpec Verify { get; }

    /// <summary>Device-filling shape, used by the bench stage.</summary>
    public CodegenKernelSpec Bench { get; }
}

/// <summary>The set of kernels the conveyor operates on.</summary>
public static class CodegenKernelCatalog
{
    /// <summary>All catalog entries, in stable order.</summary>
    public static IReadOnlyList<CodegenCatalogEntry> All { get; } = Build();

    /// <summary>Finds one entry by name, or null.</summary>
    public static CodegenCatalogEntry? Find(string name)
    {
        foreach (var entry in All)
            if (string.Equals(entry.Name, name, StringComparison.OrdinalIgnoreCase))
                return entry;
        return null;
    }

    private static CodegenCatalogEntry[] Build() =>
    [
        new("depthwise_conv2d_3x3_bias_relu",
            "depthwise 3x3 + bias + ReLU (the bake-off kernel)",
            CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(2, 8, 8, 8),
            CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(32, 64, 56, 56)),

        new("depthwise_conv2d_3x3",
            "depthwise 3x3, no epilogue (isolates the gather+reduce)",
            Depthwise(2, 8, 8, 8, bias: false, relu: false),
            Depthwise(32, 64, 56, 56, bias: false, relu: false)),

        new("conv2d_1x1_bias_relu",
            "dense 1x1 + bias + ReLU (reduction over input channels)",
            Conv2D1x1(2, 8, 8, 8, 8),
            Conv2D1x1(16, 64, 64, 28, 28)),

        new("conv2d_3x3_bias_relu",
            "dense 3x3 + bias + ReLU (reduction over C and both taps)",
            Conv2D3x3(2, 8, 8, 8, 8),   // C=8 -> 72 trips: strip-mined, like the bench shape
            Conv2D3x3(8, 32, 64, 28, 28)),

        new("maxpool2d_2x2",
            "2x2/stride-2 max pool (max reduction, no weights)",
            MaxPool2x2(2, 8, 16, 16),
            MaxPool2x2(32, 64, 112, 112)),

        new("conv_transpose2d_3x3_stride2",
            "transposed 3x3 stride 2 (exact-division index map)",
            ConvTranspose2D3x3Stride2(2, 8, 8, 8),
            ConvTranspose2D3x3Stride2(16, 64, 28, 28)),
    ];

    /// <summary>Depthwise 3x3 with optional bias and ReLU.</summary>
    private static CodegenKernelSpec Depthwise(int n, int c, int h, int w, bool bias, bool relu)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("c", c),
            CodegenAxis.Parallel("oh", h), CodegenAxis.Parallel("ow", w),
            CodegenAxis.Reduce("kh", 3), CodegenAxis.Reduce("kw", 3));
        const int N = 0, C = 1, OH = 2, OW = 3, KH = 4, KW = 5;

        var input = new CodegenTensorBinding(0, "input", [n, c, h, w],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.Window(OH, KH, 1, 1), CodegenAffineExpr.Window(OW, KW, 1, 1)
        ]);
        var weights = new CodegenTensorBinding(1, "weights", [c, 3, 3],
            [CodegenAffineExpr.Axis(C), CodegenAffineExpr.Axis(KH), CodegenAffineExpr.Axis(KW)]);
        var output = new CodegenTensorBinding(bias ? 3 : 2, "output", [n, c, h, w],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
        ], isOutput: true);

        if (!bias)
            return new CodegenKernelSpec("dwconv2d_3x3", space, [input, weights], output,
                [0, 1], CodegenReduceKind.Sum,
                activation: relu ? CodegenActivationKind.ReLU : CodegenActivationKind.None);

        var biasBinding = new CodegenTensorBinding(2, "bias", [c], [CodegenAffineExpr.Axis(C)]);
        return new CodegenKernelSpec("dwconv2d_3x3_bias", space, [input, weights, biasBinding], output,
            [0, 1], CodegenReduceKind.Sum, biasInput: 2,
            activation: relu ? CodegenActivationKind.ReLU : CodegenActivationKind.None);
    }

    /// <summary>Dense 1x1 convolution: reduce over input channels only.</summary>
    private static CodegenKernelSpec Conv2D1x1(int n, int c, int k, int h, int w)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("k", k),
            CodegenAxis.Parallel("oh", h), CodegenAxis.Parallel("ow", w),
            CodegenAxis.Reduce("c", c));
        const int N = 0, K = 1, OH = 2, OW = 3, C = 4;

        var input = new CodegenTensorBinding(0, "input", [n, c, h, w],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
        ]);
        var weights = new CodegenTensorBinding(1, "weights", [k, c],
            [CodegenAffineExpr.Axis(K), CodegenAffineExpr.Axis(C)]);
        var bias = new CodegenTensorBinding(2, "bias", [k], [CodegenAffineExpr.Axis(K)]);
        var output = new CodegenTensorBinding(3, "output", [n, k, h, w],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(K),
            CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
        ], isOutput: true);

        return new CodegenKernelSpec("conv2d_1x1_bias_relu", space, [input, weights, bias], output,
            [0, 1], CodegenReduceKind.Sum, biasInput: 2, activation: CodegenActivationKind.ReLU);
    }

    /// <summary>Dense 3x3 convolution: reduce over input channels and both taps.</summary>
    private static CodegenKernelSpec Conv2D3x3(int n, int c, int k, int h, int w)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("k", k),
            CodegenAxis.Parallel("oh", h), CodegenAxis.Parallel("ow", w),
            CodegenAxis.Reduce("c", c), CodegenAxis.Reduce("kh", 3), CodegenAxis.Reduce("kw", 3));
        const int N = 0, K = 1, OH = 2, OW = 3, C = 4, KH = 5, KW = 6;

        var input = new CodegenTensorBinding(0, "input", [n, c, h, w],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.Window(OH, KH, 1, 1), CodegenAffineExpr.Window(OW, KW, 1, 1)
        ]);
        var weights = new CodegenTensorBinding(1, "weights", [k, c, 3, 3],
        [
            CodegenAffineExpr.Axis(K), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.Axis(KH), CodegenAffineExpr.Axis(KW)
        ]);
        var bias = new CodegenTensorBinding(2, "bias", [k], [CodegenAffineExpr.Axis(K)]);
        var output = new CodegenTensorBinding(3, "output", [n, k, h, w],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(K),
            CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
        ], isOutput: true);

        return new CodegenKernelSpec("conv2d_3x3_bias_relu", space, [input, weights, bias], output,
            [0, 1], CodegenReduceKind.Sum, biasInput: 2, activation: CodegenActivationKind.ReLU);
    }

    /// <summary>2x2 stride-2 max pool: exercises the Max reduction with no weight operand.</summary>
    private static CodegenKernelSpec MaxPool2x2(int n, int c, int h, int w)
    {
        int oh = h / 2, ow = w / 2;
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("c", c),
            CodegenAxis.Parallel("oh", oh), CodegenAxis.Parallel("ow", ow),
            CodegenAxis.Reduce("kh", 2), CodegenAxis.Reduce("kw", 2));
        const int N = 0, C = 1, OH = 2, OW = 3, KH = 4, KW = 5;

        var input = new CodegenTensorBinding(0, "input", [n, c, h, w],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.Window(OH, KH, 2, 0), CodegenAffineExpr.Window(OW, KW, 2, 0)
        ]);
        var output = new CodegenTensorBinding(1, "output", [n, c, oh, ow],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
        ], isOutput: true);

        return new CodegenKernelSpec("maxpool2d_2x2", space, [input], output,
            [0], CodegenReduceKind.Max);
    }

    /// <summary>
    /// Transposed 3x3 stride 2. The input index is <c>(oh + pad - kh) / 2</c>, which
    /// only contributes when the division is exact -- the case the affine layer models
    /// with <see cref="CodegenAffineExpr.TransposedWindow"/> and the emitter must guard.
    /// </summary>
    private static CodegenKernelSpec ConvTranspose2D3x3Stride2(int n, int c, int inH, int inW)
    {
        int oh = inH * 2, ow = inW * 2;
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("c", c),
            CodegenAxis.Parallel("oh", oh), CodegenAxis.Parallel("ow", ow),
            CodegenAxis.Reduce("kh", 3), CodegenAxis.Reduce("kw", 3));
        const int N = 0, C = 1, OH = 2, OW = 3, KH = 4, KW = 5;

        var input = new CodegenTensorBinding(0, "input", [n, c, inH, inW],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.TransposedWindow(OH, KH, 2, 1),
            CodegenAffineExpr.TransposedWindow(OW, KW, 2, 1)
        ]);
        var weights = new CodegenTensorBinding(1, "weights", [c, 3, 3],
            [CodegenAffineExpr.Axis(C), CodegenAffineExpr.Axis(KH), CodegenAffineExpr.Axis(KW)]);
        var output = new CodegenTensorBinding(2, "output", [n, c, oh, ow],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
            CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
        ], isOutput: true);

        return new CodegenKernelSpec("convtranspose2d_3x3_s2", space, [input, weights], output,
            [0, 1], CodegenReduceKind.Sum);
    }
}
