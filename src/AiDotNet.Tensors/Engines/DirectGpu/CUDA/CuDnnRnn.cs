// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level cuDNN-backed <see cref="IDeviceRnn"/>. The
/// <see cref="CuDnnRnnNative"/> P/Invoke layer (cudnnRNNForward,
/// cudnnRNNBackwardData_v8, cudnnRNNBackwardWeights_v8) is in place,
/// but the high-level marshalling that turns a <c>Tensor&lt;T&gt;</c>
/// into the cuDNN descriptor + workspace + reserve-space dance is
/// still pending — see #219's RNN follow-up. Until that lands, every
/// dispatch in this class delegates to <see cref="CpuRnn"/>, even when
/// <c>libcudnn</c> is loadable. The class is wired in advance so the
/// device-primitive surface is stable; flipping each method to the
/// native path is a private follow-up that won't change the public API.
///
/// <para><b>Persistent kernels:</b> cuDNN's persistent-RNN kernels are
/// the perf differentiator vs PyTorch's <c>torch.nn.LSTM</c>. PyTorch
/// defaults the persistent kernels off due to historical correctness
/// regressions; per #219's "How we beat PyTorch" point #8 we ship them
/// on with explicit correctness coverage once the native dispatch is
/// wired. The dispatch flag lives on
/// <see cref="RnnOptions.UsePersistentKernels"/>; the cuDNN side reads
/// it via <c>cudnnSetRNNDescriptor_v8</c>'s algo selector.</para>
///
/// <para>The CPU fallback is single-layer / unidirectional / no-projection
/// today; multi-layer / bidirectional / projection / inter-layer dropout
/// configurations are exclusive to the cuDNN tier and will become
/// reachable when the native dispatch lights up.</para>
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
