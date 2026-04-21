using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Policy for inserting NCHW ↔ NCHWc reorders during ONNX translation or
/// ahead-of-time graph lowering. Captures a small oneDNN-style rule set:
/// which ops prefer the packed layout, which ops break it, and what the
/// cost threshold is for standing up a reorder.
///
/// <para>Consumed by the ONNX translator (task C10) and by the compilation
/// fusion passes (task C9). Not a general graph-rewrite engine — just the
/// per-op decision function.</para>
/// </summary>
public static class LayoutPlanner
{
    /// <summary>
    /// Channel-block size for NCHWc on the current dispatch path. Always 8
    /// today (AVX2+FMA). Will become 16 once the AVX-512 Conv kernel lands
    /// (task B4) and we flip <c>cbDefault</c>.
    /// </summary>
    public static int PreferredCBlock => 8;

    /// <summary>
    /// Returns <c>true</c> if the op has a dedicated NCHWc8 fast path and
    /// will benefit from the channel-packed layout at this shape.
    /// Conservative by default — only lists ops with shipped parity tests.
    /// </summary>
    public static bool PrefersNchwc(string opType, int[] inputShape)
    {
        if (inputShape == null || inputShape.Length != 4) return false;
        int C = inputShape[1];
        if (C % PreferredCBlock != 0) return false;
        return opType switch
        {
            "Conv"                => true,  // dedicated NCHWc Conv2D kernel
            "BatchNormalization"  => true,  // fused inference BN
            "Relu"                => true,  // elementwise, layout-propagating
            "Sigmoid"             => true,
            "Tanh"                => true,
            "LeakyRelu"           => true,
            "Gelu"                => true,
            "Mish"                => true,
            "HardSwish"           => true,
            "Silu"                => true,
            "MaxPool"             => true,
            "AveragePool"         => true,
            "Concat"              => true,  // axis=1 verified; caller must check axis
            "Split"               => true,  // split % cBlock == 0 verified; caller must check
            _ => false,
        };
    }

    /// <summary>
    /// Ops that require NCHW input and break the packed layout.
    /// Consumers must emit a ReorderToNchw before these.
    /// </summary>
    public static bool RequiresNchw(string opType) => opType switch
    {
        // Reductions / global pools that flatten the spatial dim: output is
        // 1×1 so packed layout carries no further benefit.
        "GlobalAveragePool" => true,
        "GlobalMaxPool"     => true,
        // Shape-changing ops: physical channel permutation would be wrong
        // on packed layout.
        "Flatten"           => true,
        "Reshape"           => true,
        "Transpose"         => true,
        "Squeeze"           => true,
        "Unsqueeze"         => true,
        // Matmul-family: takes NCHW 2-D / N-D input, no per-channel axis.
        "Gemm"              => true,
        "MatMul"            => true,
        "FullyConnected"    => true,
        // Normalisation forms that reduce across channels (not per-channel):
        // the Nchwc lane boundary would split the reduction axis.
        "LayerNormalization" => true,
        "InstanceNormalization" => true,
        "Softmax"           => true,
        _ => false,
    };

    /// <summary>
    /// Decide the output layout of <paramref name="opType"/> given its
    /// input layout and shape. Matches the behaviour already implemented
    /// in the engine's per-op NCHWc fast paths.
    /// </summary>
    public static TensorLayout PropagateLayout(string opType, TensorLayout inputLayout, int[] outputShape)
    {
        // Layout-breaking ops always emit NCHW.
        if (RequiresNchw(opType)) return TensorLayout.Nchw;

        // If input is NCHW, the output stays NCHW unless the op emits a new
        // packed tensor (only Conv does that via the engine's NCHWc gate).
        if (inputLayout == TensorLayout.Nchw) return TensorLayout.Nchw;

        // Input is packed — the output stays packed for every op in the
        // PrefersNchwc set, and for pointwise elementwise ops (which
        // propagate Layout automatically via the C5 wrappers).
        // If the output shape's channel dim is no longer divisible by cBlock
        // (can happen if someone passes a hand-crafted weird Conv),
        // bail to Nchw so the downstream consumer doesn't blow up.
        if (outputShape != null && outputShape.Length == 4 &&
            outputShape[1] % PreferredCBlock != 0)
        {
            return TensorLayout.Nchw;
        }

        return inputLayout;
    }

    /// <summary>
    /// Heuristic cost threshold: only stand up a ReorderToNchwc if the
    /// downstream op chain is expected to run at least <paramref name="minOps"/>
    /// NCHWc-preferring ops. One pack + one unpack is ≈ 2×H×W×C float
    /// reads/writes, amortised over the kernel work — cheap vs a Conv but
    /// wasteful for a single Add.
    /// </summary>
    public static bool WorthReordering(int downstreamOpCount, int minOps = 2) => downstreamOpCount >= minOps;
}
