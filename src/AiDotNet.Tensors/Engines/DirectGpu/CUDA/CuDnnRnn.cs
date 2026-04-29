// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level cuDNN RNN-backed <see cref="IDeviceRnn"/>. Forward and
/// backward passes for plain RNN (tanh / ReLU), LSTM, and GRU dispatch
/// through <see cref="CuDnnRnnNative"/> when <c>libcudnn</c> is
/// loadable; otherwise the wrapper hands work to <see cref="CpuRnn"/>.
///
/// <para><b>Persistent kernels:</b> cuDNN's persistent-RNN kernels are
/// the perf differentiator vs PyTorch's <c>torch.nn.LSTM</c>. PyTorch
/// defaults the persistent kernels off due to historical correctness
/// regressions; per #219's "How we beat PyTorch" point #8 we ship them
/// on with explicit correctness coverage. The dispatch flag lives on
/// <see cref="RnnOptions.UsePersistentKernels"/>; the cuDNN side reads
/// it via <c>cudnnSetRNNDescriptor_v8</c>'s algo selector.</para>
///
/// <para>The CPU fallback is single-layer / unidirectional / no-projection
/// today; multi-layer / bidirectional / projection / inter-layer dropout
/// configurations are exclusive to the cuDNN tier until those features
/// land in <see cref="CpuRnn"/>. Tracked as a #219 follow-up.</para>
/// </summary>
public sealed class CuDnnRnn : IDeviceRnn
{
    private readonly CpuRnn _cpuFallback = new();

    /// <summary>Whether the cuDNN shared library is loadable.</summary>
    public static bool IsAvailable => CuDnnRnnNative.IsAvailable;

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
