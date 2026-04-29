// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>RNN cell type. cuDNN RNN supports all four.</summary>
public enum RnnCellType
{
    /// <summary>Plain RNN, tanh activation.</summary>
    RnnTanh,
    /// <summary>Plain RNN, ReLU activation.</summary>
    RnnRelu,
    /// <summary>Long short-term memory.</summary>
    Lstm,
    /// <summary>Gated recurrent unit.</summary>
    Gru,
}

/// <summary>
/// Recurrent-network primitive surface. CUDA backend wraps cuDNN RNN;
/// HIP wraps MIOpen; Metal wraps MPS LSTM; CPU implements the unrolled
/// forward + backward via the existing
/// <c>BackwardFunctions&lt;T&gt;.LstmBackward</c> path. <see cref="ForwardLstm"/>
/// is a convenience for the LSTM-specific case (most common cell type
/// in production); <see cref="ForwardRnn"/> is the generic dispatcher.
///
/// <para><b>Persistent kernels:</b> cuDNN's persistent-RNN kernels are
/// gated behind <see cref="RnnOptions.UsePersistentKernels"/>. PyTorch
/// turned this off by default due to historical bugs; #219's
/// "How we beat PyTorch" point #8 commits us to shipping it on with
/// correctness tests covering the bug classes that motivated PyTorch's
/// regression.</para>
/// </summary>
public interface IDeviceRnn
{
    /// <summary>
    /// Forward pass for any cell type. <paramref name="input"/> shape:
    /// <c>[seqLen, batch, inputSize]</c>. Returns <c>(output, hN, cN)</c>
    /// where <c>output</c> has shape <c>[seqLen, batch, hiddenSize *
    /// numDirections]</c>; <c>hN</c> is <c>[numLayers * numDirections,
    /// batch, hiddenSize]</c>; <c>cN</c> is null for non-LSTM cells.
    /// </summary>
    (Tensor<T> Output, Tensor<T> HN, Tensor<T>? CN) ForwardRnn<T>(
        RnnCellType cell,
        Tensor<T> input, Tensor<T> h0, Tensor<T>? c0,
        Tensor<T> weights, RnnOptions options);

    /// <summary>LSTM-specific forward — equivalent to
    /// <see cref="ForwardRnn{T}"/> with <c>cell = Lstm</c>, but the
    /// signature surfaces the LSTM cell-state explicitly so callers
    /// don't deal with the optional cN. Convenience.</summary>
    (Tensor<T> Output, Tensor<T> HN, Tensor<T> CN) ForwardLstm<T>(
        Tensor<T> input, Tensor<T> h0, Tensor<T> c0,
        Tensor<T> weights, RnnOptions options);

    /// <summary>Backward pass — produces gradients for input, h0, c0, and
    /// the weight bundle. Pass the forward's saved tensors back in.</summary>
    (Tensor<T> GradInput, Tensor<T> GradH0, Tensor<T>? GradC0, Tensor<T> GradWeights)
        BackwardRnn<T>(
            RnnCellType cell,
            Tensor<T> input, Tensor<T> output, Tensor<T> hN, Tensor<T>? cN,
            Tensor<T> gradOutput, Tensor<T>? gradHN, Tensor<T>? gradCN,
            Tensor<T> weights, RnnOptions options);
}

/// <summary>RNN dispatch options.</summary>
public sealed class RnnOptions
{
    /// <summary>Hidden state size per direction.</summary>
    public int HiddenSize { get; set; }

    /// <summary>Number of stacked RNN layers.</summary>
    public int NumLayers { get; set; } = 1;

    /// <summary>True for bidirectional (the cuDNN <c>numDirections</c>
    /// becomes 2). Output and hN/cN's layer dimension scales by 2.</summary>
    public bool Bidirectional { get; set; }

    /// <summary>Dropout probability between stacked layers.
    /// 0 ⇒ no dropout. cuDNN parity.</summary>
    public double Dropout { get; set; }

    /// <summary>Optional projection size. When &gt; 0, the cell's hidden
    /// state is projected down to this many features before the next
    /// timestep. cuDNN LSTM projection.</summary>
    public int ProjSize { get; set; }

    /// <summary>Engage cuDNN's persistent kernels when shape is supported.
    /// PyTorch defaults this to false; we default true with explicit
    /// correctness coverage.</summary>
    public bool UsePersistentKernels { get; set; } = true;

    /// <summary>If true, weights are interpreted as packed (cuDNN
    /// <c>cudnnRNNDataLayout_t</c> packed format) — saves memory at the
    /// cost of a one-time pack/unpack on weight load.</summary>
    public bool PackedWeights { get; set; }
}
