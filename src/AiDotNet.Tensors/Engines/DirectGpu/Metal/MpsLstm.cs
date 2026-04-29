// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// MPS-backed <see cref="IDeviceRnn"/>. Currently every dispatch
/// delegates to <see cref="CpuRnn"/> (which ships the full
/// multi-layer / bidirectional / projection / dropout / managed-BPTT
/// surface). The native MPS LSTM kernel
/// (<c>MPSCNNLSTM</c> / <c>MPSGraph.LSTM</c>) is wired through
/// <see cref="MpsLstmNative"/>; the device-pointer plumbing flips on
/// once the device-tensor pipeline lands.
/// </summary>
public sealed class MpsLstm : IDeviceRnn
{
    private readonly CpuRnn _cpuFallback = new();

    /// <summary>Whether the MPS LSTM kernel is loadable.</summary>
    public static bool IsAvailable => MpsLstmNative.IsAvailable;

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
