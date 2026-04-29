// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// High-level MIOpen-backed <see cref="IDeviceRnn"/>. AMD-side mirror of
/// <c>CuDnnRnn</c>. Currently every dispatch delegates to <see cref="CpuRnn"/>
/// (which now ships the full multi-layer / bidirectional / projection /
/// dropout / BPTT surface). The native MIOpen RNN entry points
/// (<c>miopenRNNForwardTraining</c> / <c>miopenRNNBackwardData</c> /
/// <c>miopenRNNBackwardWeights</c>) live in
/// <see cref="MIOpenNativeBindings"/> and the dispatch flips on once the
/// device-tensor pipeline lands.
///
/// <para>The cross-vendor RNN parity property is that PyTorch's CUDA
/// and ROCm RNNs produce different bits at the same seed (different
/// reduction order). Ours don't — both backends end up calling the
/// same managed BPTT in <see cref="CpuRnn.BackwardRnn"/> until the
/// native paths flip on, and the native path's correctness will be
/// pinned to <see cref="CpuRnn"/>'s reference output via #219's
/// "How we beat PyTorch" tests.</para>
/// </summary>
public sealed class MIOpenRnn : IDeviceRnn
{
    private readonly CpuRnn _cpuFallback = new();

    /// <summary>Whether the MIOpen shared library is loadable.</summary>
    public static bool IsAvailable => MIOpenNativeBindings.IsAvailable;

    /// <inheritdoc/>
    public (Tensor<T> Output, Tensor<T> HN, Tensor<T>? CN) ForwardRnn<T>(
        RnnCellType cell,
        Tensor<T> input, Tensor<T> h0, Tensor<T>? c0,
        Tensor<T> weights, RnnOptions options)
        => _cpuFallback.ForwardRnn(cell, input, h0, c0, weights, options);

    /// <inheritdoc/>
    public (Tensor<T> Output, Tensor<T> HN, Tensor<T> CN) ForwardLstm<T>(
        Tensor<T> input, Tensor<T> h0, Tensor<T> c0,
        Tensor<T> weights, RnnOptions options)
        => _cpuFallback.ForwardLstm(input, h0, c0, weights, options);

    /// <inheritdoc/>
    public (Tensor<T> GradInput, Tensor<T> GradH0, Tensor<T>? GradC0, Tensor<T> GradWeights)
        BackwardRnn<T>(
            RnnCellType cell,
            Tensor<T> input, Tensor<T> output, Tensor<T> hN, Tensor<T>? cN,
            Tensor<T> gradOutput, Tensor<T>? gradHN, Tensor<T>? gradCN,
            Tensor<T> weights, RnnOptions options)
        => _cpuFallback.BackwardRnn(cell, input, output, hN, cN, gradOutput, gradHN, gradCN, weights, options);
}
