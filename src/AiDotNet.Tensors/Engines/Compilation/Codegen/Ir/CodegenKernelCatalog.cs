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

/// <summary>
/// One catalog row for a multi-pass program: an operator whose passes communicate through
/// intermediates (a normalisation's statistics), declared at the same two shapes.
/// </summary>
public sealed class CodegenCatalogProgramEntry
{
    /// <summary>Creates a program catalog entry.</summary>
    public CodegenCatalogProgramEntry(string name, string summary, CodegenProgram verify, CodegenProgram bench)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Summary = summary ?? throw new ArgumentNullException(nameof(summary));
        Verify = verify ?? throw new ArgumentNullException(nameof(verify));
        Bench = bench ?? throw new ArgumentNullException(nameof(bench));
        if (verify.Passes.Count != bench.Passes.Count)
            throw new ArgumentException(
                "Program '" + name + "' has " + verify.Passes.Count + " passes at the verify " +
                "shape and " + bench.Passes.Count + " at the bench shape; they must be the same program.");
    }

    /// <summary>Stable catalog identifier, used on the command line and in manifests.</summary>
    public string Name { get; }

    /// <summary>One line describing what the program computes.</summary>
    public string Summary { get; }

    /// <summary>Small shape: cheap fp64 reference.</summary>
    public CodegenProgram Verify { get; }

    /// <summary>Device-filling shape, used by the bench stage.</summary>
    public CodegenProgram Bench { get; }
}

/// <summary>The set of kernels the conveyor operates on.</summary>
public static class CodegenKernelCatalog
{
    /// <summary>All catalog entries, in stable order.</summary>
    public static IReadOnlyList<CodegenCatalogEntry> All { get; } = Build();

    /// <summary>
    /// The multi-pass programs: normalisations and softmax, whose passes are chained by the
    /// names of the statistics they produce.
    /// </summary>
    public static IReadOnlyList<CodegenCatalogProgramEntry> Programs { get; } = BuildPrograms();

    /// <summary>
    /// Every bench-shape kernel the static ratchet holds: each single-kernel entry under
    /// its own name, and each program pass as <c>program/pass</c>.
    /// </summary>
    public static IEnumerable<(string Name, CodegenKernelSpec Spec)> RatchetedKernels()
    {
        foreach (var entry in All) yield return (entry.Name, entry.Bench);
        foreach (var program in Programs)
            foreach (var pass in program.Bench.Passes)
                yield return (program.Name + "/" + pass.Name, pass);
    }

    /// <summary>Finds one program entry by name, or null.</summary>
    public static CodegenCatalogProgramEntry? FindProgram(string name)
    {
        foreach (var entry in Programs)
            if (string.Equals(entry.Name, name, StringComparison.OrdinalIgnoreCase))
                return entry;
        return null;
    }

    // Bench shapes follow the hand-written kernels' release cells where they exist (rows
    // 8192 of a 64-wide row for the row normalisations; B8/C64 for BatchNorm; B32/C64/G8
    // for GroupNorm), widened where that cell would not fill the device.
    private static CodegenCatalogProgramEntry[] BuildPrograms() =>
    [
        new("softmax_rows",
            "row softmax: negated max, reciprocal exp-sum, normalise",
            CodegenFusedStatistics.Softmax(16, 96),
            CodegenFusedStatistics.Softmax(8192, 1024)),

        new("layernorm",
            "LayerNorm over the last axis, saving negmean and rstd",
            CodegenFusedStatistics.LayerNorm(16, 96),
            CodegenFusedStatistics.LayerNorm(8192, 1024)),

        new("layernorm_d64",
            "LayerNorm at the hand-written kernel's D64 release cell",
            CodegenFusedStatistics.LayerNorm(16, 64),
            CodegenFusedStatistics.LayerNorm(8192, 64)),

        new("rmsnorm",
            "RMSNorm over the last axis, saving rstd",
            CodegenFusedStatistics.RmsNorm(16, 96),
            CodegenFusedStatistics.RmsNorm(8192, 1024)),

        new("batchnorm2d",
            "training BatchNorm over NCHW: per-channel statistics over (n, h, w)",
            CodegenFusedStatistics.BatchNorm2D(2, 8, 6, 6),
            CodegenFusedStatistics.BatchNorm2D(32, 64, 56, 56)),

        new("layernorm_backward",
            "LayerNorm backward: dx (row-fused) and dgamma/dbeta (column reductions)",
            CodegenFusedStatistics.LayerNormBackward(16, 96),
            CodegenFusedStatistics.LayerNormBackward(8192, 1024)),

        new("rmsnorm_backward",
            "RMSNorm backward: dx (row-fused) and dgamma (column reduction)",
            CodegenFusedStatistics.RmsNormBackward(16, 96),
            CodegenFusedStatistics.RmsNormBackward(8192, 1024)),

        new("softmax_backward",
            "row softmax backward: dx = y * (dy - sum_j dy*y)",
            CodegenFusedStatistics.SoftmaxBackward(16, 96),
            CodegenFusedStatistics.SoftmaxBackward(8192, 1024)),

        new("groupnorm_backward",
            "GroupNorm backward: dx (row-fused per group), dgamma/dbeta per channel",
            CodegenFusedStatistics.GroupNormBackward(2, 8, 4, 36),
            CodegenFusedStatistics.GroupNormBackward(32, 64, 8, 32 * 32)),

        new("batchnorm2d_backward",
            "training BatchNorm backward: both channel sums in one streamed sweep, then dx",
            CodegenFusedStatistics.BatchNorm2DBackward(2, 8, 6, 6),
            CodegenFusedStatistics.BatchNorm2DBackward(32, 64, 56, 56)),

        new("paged_attention_decode",
            "paged attention decode: scores and the value product through a block table, with a row softmax between",
            CodegenBackendPrograms.PagedAttentionDecode(4, 4, 8, 4, 3, 7, 0.125),
            CodegenBackendPrograms.PagedAttentionDecode(32, 8, 128, 16, 64, 4096, 0.08838834764831845)),

        new("paged_attention_prefill",
            "paged attention prefill: the same cache through a block table, causal over the query's own position",
            CodegenBackendPrograms.PagedAttentionPrefill(4, 4, 8, 4, 3, 7, 3, 0.125),
            CodegenBackendPrograms.PagedAttentionPrefill(32, 8, 128, 16, 64, 4096, 64, 0.08838834764831845)),

        new("groupnorm",
            "GroupNorm: statistics per (sample, group), per-channel affine",
            CodegenFusedStatistics.GroupNorm(2, 8, 4, 36),
            CodegenFusedStatistics.GroupNorm(32, 64, 8, 32 * 32)),
    ];

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
            CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(2, 8, 16, 16),
            CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(32, 64, 56, 56)),

        new("depthwise_conv2d_3x3",
            "depthwise 3x3, no epilogue (isolates the gather+reduce)",
            Depthwise(2, 8, 16, 16, bias: false, relu: false),
            Depthwise(32, 64, 56, 56, bias: false, relu: false)),

        new("conv2d_1x1_bias_relu",
            "dense 1x1 + bias + ReLU (reduction over input channels)",
            Conv2D1x1(2, 8, 8, 16, 16),
            Conv2D1x1(16, 64, 64, 28, 28)),

        new("conv2d_3x3_bias_relu",
            "dense 3x3 + bias + ReLU (reduction over C and both taps)",
            Conv2D3x3(2, 8, 8, 16, 16),   // C=8 -> 72 trips: strip-mined, like the bench shape
            Conv2D3x3(8, 32, 64, 28, 28)),

        new("maxpool2d_2x2",
            "2x2/stride-2 max pool (max reduction, no weights)",
            MaxPool2x2(2, 8, 32, 32),
            MaxPool2x2(32, 64, 112, 112)),

        new("conv_transpose2d_3x3_stride2",
            "transposed 3x3 stride 2 (exact-division index map)",
            ConvTranspose2D3x3Stride2(2, 8, 16, 16),
            ConvTranspose2D3x3Stride2(16, 64, 28, 28)),

        // Backward kernels are DERIVED from the forward spec, never authored. Each of
        // these is CodegenAdjoint.BackwardData applied to the linear forward operator
        // above it, so it cannot drift from the forward pass it differentiates.
        new("depthwise_conv2d_3x3_bwd_data",
            "gradient wrt data of depthwise 3x3 (derived adjoint)",
            CodegenAdjoint.BackwardData(Depthwise(2, 8, 16, 16, bias: false, relu: false), 0),
            CodegenAdjoint.BackwardData(Depthwise(32, 64, 56, 56, bias: false, relu: false), 0)),

        new("conv2d_1x1_bwd_data",
            "gradient wrt data of dense 1x1 (contracted axis moves into the reduction)",
            CodegenAdjoint.BackwardData(Conv2D1x1Linear(2, 8, 8, 16, 16), 0),
            CodegenAdjoint.BackwardData(Conv2D1x1Linear(16, 64, 64, 28, 28), 0)),

        new("conv2d_3x3_bwd_data",
            "gradient wrt data of dense 3x3 (moved axis AND two inverted windows)",
            CodegenAdjoint.BackwardData(Conv2D3x3Linear(2, 8, 8, 16, 16), 0),
            CodegenAdjoint.BackwardData(Conv2D3x3Linear(8, 32, 64, 28, 28), 0)),

        // WEIGHT GRADIENTS, also derived. Without these the generated set could not
        // train: only the data gradient existed. The same transform produces them --
        // it differentiates with respect to whichever operand it is given -- but the
        // reduction is over batch and spatial axes rather than channels and taps.
        new("depthwise_conv2d_3x3_bwd_weights",
            "gradient wrt weights of depthwise 3x3 (reduces over n, oh, ow)",
            CodegenAdjoint.BackwardWeights(Depthwise(2, 8, 16, 16, bias: false, relu: false), 1),
            CodegenAdjoint.BackwardWeights(Depthwise(32, 64, 56, 56, bias: false, relu: false), 1)),

        new("conv2d_1x1_bwd_weights",
            "gradient wrt weights of dense 1x1",
            CodegenAdjoint.BackwardWeights(Conv2D1x1Linear(2, 8, 8, 16, 16), 1),
            CodegenAdjoint.BackwardWeights(Conv2D1x1Linear(16, 64, 64, 28, 28), 1)),

        new("conv2d_3x3_bwd_weights",
            "gradient wrt weights of dense 3x3 (compound map survives renumbering)",
            CodegenAdjoint.BackwardWeights(Conv2D3x3Linear(2, 8, 8, 16, 16), 1),
            CodegenAdjoint.BackwardWeights(Conv2D3x3Linear(8, 32, 64, 28, 28), 1)),

        // ATTACKS A MEASURED STRUCTURAL WEAKNESS. Profiling showed cuDNN's "one
        // convolution" is five kernels: an NCHW->NHWC transform, the implicit GEMM, the
        // transform back, and a separate elementwise pass for bias -- 9.47 us of 41.06
        // (23%) spent on work we do not do at all. PyTorch cannot fuse through a cuDNN
        // call, so EVERY extra elementwise op costs it a full kernel launch plus a full
        // round trip of the tensor through memory, while costing us one instruction
        // inside a loop we are already running.
        //
        // This kernel exists to make that scaling visible: same convolution, deeper
        // epilogue. Our cost should be flat; theirs should grow by a memory pass per op.
        new("conv2d_1x1_deep_epilogue",
            "dense 1x1 + bias + scale + ReLU in ONE kernel (cuDNN needs four)",
            Conv2D1x1DeepEpilogue(2, 8, 8, 16, 16),
            Conv2D1x1DeepEpilogue(16, 64, 64, 28, 28)),

        // GENERAL CONVOLUTION (#890 phase 4b): stride, dilation and groups through one
        // builder, so each is an index map rather than a kernel. Shapes are the ones the
        // common networks use them at: ResNet's stride-2 downsample, a DeepLab-style
        // dilated 3x3, and a ResNeXt-style grouped 3x3. Each derives its data and weight
        // gradients from the same linear operator.
        new("conv2d_3x3_stride2",
            "dense 3x3 stride 2 + bias + ReLU (the ResNet downsample)",
            Conv2D(2, 8, 8, 16, 16, 3, stride: 2, pad: 1, dilation: 1, groups: 1, epilogue: true),
            Conv2D(8, 64, 128, 56, 56, 3, stride: 2, pad: 1, dilation: 1, groups: 1, epilogue: true)),

        new("conv2d_3x3_dilation2",
            "dense 3x3 dilation 2 + bias + ReLU (a dilated / atrous convolution)",
            Conv2D(2, 8, 8, 16, 16, 3, stride: 1, pad: 2, dilation: 2, groups: 1, epilogue: true),
            Conv2D(8, 32, 32, 28, 28, 3, stride: 1, pad: 2, dilation: 2, groups: 1, epilogue: true)),

        new("conv2d_3x3_grouped",
            "grouped 3x3 (8 groups) + bias + ReLU (ResNeXt-style)",
            // 8 channels a group -> 72 trips: strip-mined, like the bench shape
            Conv2D(2, 16, 16, 16, 16, 3, stride: 1, pad: 1, dilation: 1, groups: 2, epilogue: true),
            Conv2D(8, 64, 64, 28, 28, 3, stride: 1, pad: 1, dilation: 1, groups: 8, epilogue: true)),

        new("conv2d_3x3_stride2_bwd_data",
            "gradient wrt data of the stride-2 3x3 (derived adjoint: exact-division windows)",
            CodegenAdjoint.BackwardData(Conv2D(2, 8, 8, 16, 16, 3, 2, 1, 1, 1, epilogue: false), 0),
            CodegenAdjoint.BackwardData(Conv2D(8, 64, 128, 56, 56, 3, 2, 1, 1, 1, epilogue: false), 0)),

        new("conv2d_3x3_stride2_bwd_weights",
            "gradient wrt weights of the stride-2 3x3 (derived)",
            // 32x32 input: enough (n, oh, ow) trips that two reduction axes loop, like the bench shape
            CodegenAdjoint.BackwardWeights(Conv2D(2, 8, 8, 32, 32, 3, 2, 1, 1, 1, epilogue: false), 1),
            CodegenAdjoint.BackwardWeights(Conv2D(8, 64, 128, 56, 56, 3, 2, 1, 1, 1, epilogue: false), 1)),

        new("conv2d_3x3_dilation2_bwd_data",
            "gradient wrt data of the dilated 3x3 (derived)",
            CodegenAdjoint.BackwardData(Conv2D(2, 8, 8, 16, 16, 3, 1, 2, 2, 1, epilogue: false), 0),
            CodegenAdjoint.BackwardData(Conv2D(8, 32, 32, 28, 28, 3, 1, 2, 2, 1, epilogue: false), 0)),

        new("conv2d_3x3_grouped_bwd_data",
            "gradient wrt data of the grouped 3x3 (derived)",
            // 8 channels a group -> 72 trips: strip-mined, like the bench shape
            CodegenAdjoint.BackwardData(Conv2D(2, 16, 16, 16, 16, 3, 1, 1, 1, 2, epilogue: false), 0),
            CodegenAdjoint.BackwardData(Conv2D(8, 64, 64, 28, 28, 3, 1, 1, 1, 8, epilogue: false), 0)),

        new("conv2d_3x3_grouped_bwd_weights",
            "gradient wrt weights of the grouped 3x3 (derived)",
            CodegenAdjoint.BackwardWeights(Conv2D(2, 16, 16, 16, 16, 3, 1, 1, 1, 4, epilogue: false), 1),
            CodegenAdjoint.BackwardWeights(Conv2D(8, 64, 64, 28, 28, 3, 1, 1, 1, 8, epilogue: false), 1)),

        // POINTWISE (#890 phase 4a). No reduction axis, so each is one thread per element
        // and the question the bench answers is purely how close to DRAM bandwidth it runs.
        new("bias_gelu",
            "x + bias[col] -> GELU (the transformer MLP epilogue)",
            BiasActivation(16, 96, CodegenActivationKind.Gelu),
            BiasActivation(8192, 1024, CodegenActivationKind.Gelu)),

        new("bias_swish",
            "x + bias[col] -> Swish/SiLU",
            BiasActivation(16, 96, CodegenActivationKind.Swish),
            BiasActivation(8192, 1024, CodegenActivationKind.Swish)),

        new("residual_add_relu",
            "relu(x + residual), the residual-block join",
            ResidualAdd(16 * 96, CodegenActivationKind.ReLU),
            ResidualAdd(8192 * 1024, CodegenActivationKind.ReLU)),

        new("multiply",
            "x * y elementwise",
            Multiply(16 * 96),
            Multiply(8192 * 1024)),

        new("sigmoid",
            "sigmoid(x)",
            Unary(16 * 96, CodegenActivationKind.Sigmoid),
            Unary(8192 * 1024, CodegenActivationKind.Sigmoid)),

        new("tanh",
            "tanh(x)",
            Unary(16 * 96, CodegenActivationKind.Tanh),
            Unary(8192 * 1024, CodegenActivationKind.Tanh)),

        new("batchnorm_inference_relu",
            "relu(x * scale[c] + shift[c]): inference BatchNorm folded to a channel affine",
            ChannelAffine(2, 8, 6, 6, CodegenActivationKind.ReLU),
            ChannelAffine(32, 64, 56, 56, CodegenActivationKind.ReLU)),

        new("scatter_add_deterministic",
            "deterministic scatter-add: each destination reduces over the sources that target it, no atomic",
            CodegenBackendPrograms.ScatterAddDeterministic(96, 16, 4).Passes[0],
            CodegenBackendPrograms.ScatterAddDeterministic(4096, 256, 64).Passes[0]),

        new("scatter_add_accumulate_deterministic",
            "the same inversion onto the destination's existing value, which the caller binds as the seed",
            CodegenBackendPrograms.ScatterAddAccumulateDeterministic(96, 16, 4).Passes[0],
            CodegenBackendPrograms.ScatterAddAccumulateDeterministic(4096, 256, 64).Passes[0]),

        new("segment_max",
            "deterministic segment max over unsorted ids; a sparse matrix's row indices make it the CSR one",
            CodegenBackendPrograms.SegmentMax(96, 16, 4).Passes[0],
            CodegenBackendPrograms.SegmentMax(4096, 256, 64).Passes[0]),

        new("segment_min",
            "deterministic segment min: the maximum over negated terms, negated again",
            CodegenBackendPrograms.SegmentMin(96, 16, 4).Passes[0],
            CodegenBackendPrograms.SegmentMin(4096, 256, 64).Passes[0]),
    ];

    /// <summary>
    /// General 2D convolution over NCHW: square kernel, uniform stride, padding and dilation,
    /// and channel groups. With groups the operands are bound as [n, G, C/G, h, w],
    /// [G, K/G, C/G, kh, kw] and [n, G, K/G, oh, ow] -- the same bytes as the NCHW and OIHW
    /// layouts -- so a group is an axis rather than a division, every map is a single axis or
    /// a window, and the adjoint differentiates it like any other convolution. With
    /// epilogue it adds a per-output-channel bias and ReLU; without, it is the linear
    /// operator the adjoint differentiates.
    /// </summary>
    internal static CodegenKernelSpec Conv2D(
        int n, int c, int k, int h, int w, int kernel, int stride, int pad, int dilation, int groups, bool epilogue)
    {
        if (groups < 1 || c % groups != 0 || k % groups != 0)
            throw new ArgumentException("Groups must divide both channel counts.", nameof(groups));
        int cg = c / groups, kg = k / groups;
        int oh = (h + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        int ow = (w + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        string name = "conv2d_" + kernel + "x" + kernel + "_s" + stride + "_d" + dilation + "_g" + groups;

        static CodegenAffineExpr A(int axis) => CodegenAffineExpr.Axis(axis);
        static CodegenAffineExpr Win(int spatial, int tap, int stride, int dilation, int pad) =>
            new(new[] { new CodegenAffineTerm(spatial, stride), new CodegenAffineTerm(tap, dilation) }, -pad);

        CodegenIterationSpace space;
        CodegenTensorBinding input, weights;
        CodegenAffineExpr[] outMap, biasMap;
        int[] outShape, biasShape;
        if (groups > 1)
        {
            space = new CodegenIterationSpace(
                CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("g", groups), CodegenAxis.Parallel("kg", kg),
                CodegenAxis.Parallel("oh", oh), CodegenAxis.Parallel("ow", ow),
                CodegenAxis.Reduce("cg", cg), CodegenAxis.Reduce("kh", kernel), CodegenAxis.Reduce("kw", kernel));
            const int N = 0, G = 1, KG = 2, OH = 3, OW = 4, CG = 5, KH = 6, KW = 7;
            input = new CodegenTensorBinding(0, "input", [n, groups, cg, h, w],
                [A(N), A(G), A(CG), Win(OH, KH, stride, dilation, pad), Win(OW, KW, stride, dilation, pad)]);
            weights = new CodegenTensorBinding(1, "weights", [groups, kg, cg, kernel, kernel],
                [A(G), A(KG), A(CG), A(KH), A(KW)]);
            outShape = [n, groups, kg, oh, ow];
            outMap = [A(N), A(G), A(KG), A(OH), A(OW)];
            biasShape = [groups, kg];
            biasMap = [A(G), A(KG)];
        }
        else
        {
            space = new CodegenIterationSpace(
                CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("k", k),
                CodegenAxis.Parallel("oh", oh), CodegenAxis.Parallel("ow", ow),
                CodegenAxis.Reduce("c", c), CodegenAxis.Reduce("kh", kernel), CodegenAxis.Reduce("kw", kernel));
            const int N = 0, K = 1, OH = 2, OW = 3, C = 4, KH = 5, KW = 6;
            input = new CodegenTensorBinding(0, "input", [n, c, h, w],
                [A(N), A(C), Win(OH, KH, stride, dilation, pad), Win(OW, KW, stride, dilation, pad)]);
            weights = new CodegenTensorBinding(1, "weights", [k, c, kernel, kernel], [A(K), A(C), A(KH), A(KW)]);
            outShape = [n, k, oh, ow];
            outMap = [A(N), A(K), A(OH), A(OW)];
            biasShape = [k];
            biasMap = [A(K)];
        }

        if (!epilogue)
        {
            var output = new CodegenTensorBinding(2, "output", outShape, outMap, isOutput: true);
            return new CodegenKernelSpec(name, space, [input, weights], output, [0, 1], CodegenReduceKind.Sum);
        }
        var bias = new CodegenTensorBinding(2, "bias", biasShape, biasMap);
        var withBias = new CodegenTensorBinding(3, "output", outShape, outMap, isOutput: true);
        return new CodegenKernelSpec(name + "_bias_relu", space, [input, weights, bias], withBias, [0, 1],
            CodegenReduceKind.Sum, biasInput: 2, activation: CodegenActivationKind.ReLU);
    }

    private static CodegenKernelSpec BiasActivation(int rows, int columns, CodegenActivationKind activation)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", rows), CodegenAxis.Parallel("j", columns));
        var x = new CodegenTensorBinding(0, "x", [rows, columns],
            [CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1)]);
        var bias = new CodegenTensorBinding(1, "bias", [columns], [CodegenAffineExpr.Axis(1)]);
        var y = new CodegenTensorBinding(2, "y", [rows, columns],
            [CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1)], isOutput: true);
        return new CodegenKernelSpec("bias_" + activation.ToString().ToLowerInvariant(), space,
            [x, bias], y, [0], CodegenReduceKind.None, biasInput: 1, activation: activation);
    }

    private static CodegenKernelSpec ResidualAdd(int count, CodegenActivationKind activation)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var x = new CodegenTensorBinding(0, "x", [count], [CodegenAffineExpr.Axis(0)]);
        var residual = new CodegenTensorBinding(1, "residual", [count], [CodegenAffineExpr.Axis(0)]);
        var y = new CodegenTensorBinding(2, "y", [count], [CodegenAffineExpr.Axis(0)], isOutput: true);
        return new CodegenKernelSpec("residual_add", space, [x, residual], y, [0],
            CodegenReduceKind.None, activation: activation, preBiasInput: 1);
    }

    private static CodegenKernelSpec Multiply(int count)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var x = new CodegenTensorBinding(0, "x", [count], [CodegenAffineExpr.Axis(0)]);
        var other = new CodegenTensorBinding(1, "other", [count], [CodegenAffineExpr.Axis(0)]);
        var y = new CodegenTensorBinding(2, "y", [count], [CodegenAffineExpr.Axis(0)], isOutput: true);
        return new CodegenKernelSpec("multiply", space, [x, other], y, [0, 1], CodegenReduceKind.None);
    }

    private static CodegenKernelSpec Unary(int count, CodegenActivationKind activation)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", count));
        var x = new CodegenTensorBinding(0, "x", [count], [CodegenAffineExpr.Axis(0)]);
        var y = new CodegenTensorBinding(1, "y", [count], [CodegenAffineExpr.Axis(0)], isOutput: true);
        return new CodegenKernelSpec(activation.ToString().ToLowerInvariant(), space, [x], y, [0],
            CodegenReduceKind.None, activation: activation);
    }

    private static CodegenKernelSpec ChannelAffine(int n, int c, int h, int w, CodegenActivationKind activation)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", n), CodegenAxis.Parallel("c", c),
            CodegenAxis.Parallel("h", h), CodegenAxis.Parallel("w", w));
        CodegenAffineExpr[] full =
            [CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1), CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(3)];
        var x = new CodegenTensorBinding(0, "x", [n, c, h, w], full);
        var scale = new CodegenTensorBinding(1, "scale", [c], [CodegenAffineExpr.Axis(1)]);
        var shift = new CodegenTensorBinding(2, "shift", [c], [CodegenAffineExpr.Axis(1)]);
        var y = new CodegenTensorBinding(3, "y", [n, c, h, w], full, isOutput: true);
        return new CodegenKernelSpec("channel_affine", space, [x, scale, shift], y, [0],
            CodegenReduceKind.None, activation: activation,
            postScaleInput: 1, postBiasInput: 2);
    }

    /// <summary>Dense 1x1 with a three-stage epilogue: bias, then scale, then ReLU.</summary>
    private static CodegenKernelSpec Conv2D1x1DeepEpilogue(int n, int c, int k, int h, int w)
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
        var scale = new CodegenTensorBinding(3, "scale", [k], [CodegenAffineExpr.Axis(K)]);
        var output = new CodegenTensorBinding(4, "output", [n, k, h, w],
        [
            CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(K),
            CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
        ], isOutput: true);

        return new CodegenKernelSpec("conv2d_1x1_deep_epilogue", space,
            [input, weights, bias, scale], output,
            [0, 1], CodegenReduceKind.Sum,
            biasInput: 2, scaleInput: 3, activation: CodegenActivationKind.ReLU);
    }

    /// <summary>Bias-free, activation-free 1x1: the linear operator an adjoint needs.</summary>
    private static CodegenKernelSpec Conv2D1x1Linear(int n, int c, int k, int h, int w)
    {
        var full = Conv2D1x1(n, c, k, h, w);
        var output = new CodegenTensorBinding(2, "output", [n, k, h, w],
        [
            CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1),
            CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(3)
        ], isOutput: true);
        return new CodegenKernelSpec("conv2d_1x1", full.Space,
            new[] { full.Inputs[0], full.Inputs[1] }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum);
    }

    /// <summary>Bias-free, activation-free 3x3: the linear operator an adjoint needs.</summary>
    private static CodegenKernelSpec Conv2D3x3Linear(int n, int c, int k, int h, int w)
    {
        var full = Conv2D3x3(n, c, k, h, w);
        var output = new CodegenTensorBinding(2, "output", [n, k, h, w],
        [
            CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1),
            CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(3)
        ], isOutput: true);
        return new CodegenKernelSpec("conv2d_3x3", full.Space,
            new[] { full.Inputs[0], full.Inputs[1] }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum);
    }

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
        int oh = CodegenConvAttributes.TransposedExtent(inH, 3, 2, 1);
        int ow = CodegenConvAttributes.TransposedExtent(inW, 3, 2, 1);
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
