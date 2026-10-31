// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: a whole multilayer perceptron's forward for ONE input row in one launch. At batch 1 every layer
/// was a GEMV, a reduce and an epilogue pass, so a three-layer MLP's forward was about seven launches around a few hundred
/// thousand multiply-adds: launch-bound on any device with a launch cost in the tens of microseconds.
/// </summary>
[BackendSpecificRoute("OpenCL batch-1 MLP forward in one launch; every backend runs it as a FusedLinear per layer",
    nameof(IDirectGpuBackend.GemmBias), nameof(IDirectGpuBackend.GemmBiasRelu))]
internal interface IRowMlpForwardBackend
{
    /// <summary>
    /// y = act_out(... act_hidden(x W0 + b0) ... W[L-1] + b[L-1]) for one row: <paramref name="widths"/> holds the
    /// L + 1 layer widths (input first), weight l is [widths[l], widths[l + 1]] row-major, and a null bias adds nothing.
    /// The activations are <see cref="FusedActivationType.None"/> or <see cref="FusedActivationType.ReLU"/>. False
    /// (nothing launched) when the depth, a width or an activation is not supported.
    /// </summary>
    bool TryRowMlpForward(IGpuBuffer input, IGpuBuffer[] weights, IGpuBuffer?[] biases, int[] widths,
        FusedActivationType hiddenActivation, FusedActivationType outputActivation, IGpuBuffer output);
}