// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// Reference CPU implementation of <see cref="IDeviceRnn"/>. Mirrors
/// the layout PyTorch's <c>torch.nn.LSTM</c> / <c>torch.nn.GRU</c> /
/// <c>torch.nn.RNN</c> use so weights produced by either runtime are
/// portable between CPU and GPU code paths.
///
/// <para><b>Weight packing (single-layer, unidirectional):</b>
/// <c>weights</c> is a flat tensor of layout
/// <c>[W_ih (G·H·I) | W_hh (G·H·H) | b_ih (G·H) | b_hh (G·H)]</c>
/// where <c>G</c> is the gate count (4 for LSTM, 3 for GRU, 1 for plain
/// RNN) and the gate order is <c>i, f, g, o</c> for LSTM and
/// <c>r, z, n</c> for GRU — matching PyTorch's fused weight layout.
/// Multi-layer / bidirectional / projection configurations dispatch to
/// the cuDNN tier; the CPU reference returns <see cref="NotSupportedException"/>
/// for those today and the cuDNN wrapper in
/// <see cref="DirectGpu.CUDA.CuDnnRnn"/> exercises the full surface
/// once the device-tensor pipeline lands.</para>
/// </summary>
public sealed class CpuRnn : IDeviceRnn
{
    /// <inheritdoc/>
    public (Tensor<T> Output, Tensor<T> HN, Tensor<T>? CN) ForwardRnn<T>(
        RnnCellType cell,
        Tensor<T> input, Tensor<T> h0, Tensor<T>? c0,
        Tensor<T> weights, RnnOptions options)
    {
        EnsureSimpleConfig(options);
        return cell switch
        {
            RnnCellType.Lstm => ForwardLstmCore(input, h0, RequireC0(c0), weights, options.HiddenSize),
            RnnCellType.RnnTanh => ForwardPlainRnn(input, h0, weights, options.HiddenSize, useTanh: true),
            RnnCellType.RnnRelu => ForwardPlainRnn(input, h0, weights, options.HiddenSize, useTanh: false),
            RnnCellType.Gru => ForwardGru(input, h0, weights, options.HiddenSize),
            _ => throw new ArgumentOutOfRangeException(nameof(cell)),
        };
    }

    /// <inheritdoc/>
    public (Tensor<T> Output, Tensor<T> HN, Tensor<T> CN) ForwardLstm<T>(
        Tensor<T> input, Tensor<T> h0, Tensor<T> c0,
        Tensor<T> weights, RnnOptions options)
    {
        EnsureSimpleConfig(options);
        var (output, hN, cN) = ForwardLstmCore(input, h0, c0, weights, options.HiddenSize);
        return (output, hN, cN!);
    }

    /// <inheritdoc/>
    public (Tensor<T> GradInput, Tensor<T> GradH0, Tensor<T>? GradC0, Tensor<T> GradWeights)
        BackwardRnn<T>(
            RnnCellType cell,
            Tensor<T> input, Tensor<T> output, Tensor<T> hN, Tensor<T>? cN,
            Tensor<T> gradOutput, Tensor<T>? gradHN, Tensor<T>? gradCN,
            Tensor<T> weights, RnnOptions options)
    {
        // The reference backward path is finite-difference-backed for
        // correctness coverage; production GPU dispatch goes through
        // the cuDNN backward kernel. The CPU path is currently
        // wired to throw rather than ship a numerically-fragile
        // managed BPTT — the cuDNN path is what production training
        // hits, and the CPU forward is what's exercised in offline
        // inference. A fully-managed BPTT lives behind #219's
        // follow-up "CPU BPTT for RNN backward parity" task.
        _ = cell; _ = input; _ = output; _ = hN; _ = cN;
        _ = gradOutput; _ = gradHN; _ = gradCN;
        _ = weights; _ = options;
        throw new NotSupportedException(
            "Managed BPTT for RNN backward isn't wired in the reference CPU tier; " +
            "use the GPU backend (cuDNN / MIOpen / MPS) for backward, or call the " +
            "existing IEngine LstmBackward / GruBackward kernels directly. Tracked as a #219 follow-up.");
    }

    private static void EnsureSimpleConfig(RnnOptions options)
    {
        if (options.NumLayers != 1)
            throw new NotSupportedException("Multi-layer RNN goes through the cuDNN tier; CPU reference is single-layer.");
        if (options.Bidirectional)
            throw new NotSupportedException("Bidirectional RNN goes through the cuDNN tier; CPU reference is unidirectional.");
        if (options.ProjSize != 0)
            throw new NotSupportedException("LSTM projection goes through the cuDNN tier; CPU reference doesn't project.");
        if (options.Dropout != 0.0)
            throw new NotSupportedException("Inter-layer dropout goes through the cuDNN tier; CPU reference is single-layer (no dropout).");
    }

    private static Tensor<T> RequireC0<T>(Tensor<T>? c0) =>
        c0 ?? throw new ArgumentException("LSTM requires a non-null cell-state c0.");

    private static (Tensor<T> Output, Tensor<T> HN, Tensor<T>? CN) ForwardLstmCore<T>(
        Tensor<T> input, Tensor<T> h0, Tensor<T> c0, Tensor<T> weights, int hidden)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var (seqLen, batch, inputSize) = ParseInputShape(input);
        EnsureHiddenShape(h0, batch, hidden);
        EnsureHiddenShape(c0, batch, hidden);

        const int G = 4;
        var (wIh, wHh, bIh, bHh) = SplitWeights(weights, G, hidden, inputSize);
        var output = new Tensor<T>(new[] { seqLen, batch, hidden });
        var outSpan = output.AsWritableSpan();

        var hPrev = new T[batch * hidden];
        var cPrev = new T[batch * hidden];
        var hNew = new T[batch * hidden];
        var cNew = new T[batch * hidden];
        h0.AsSpan().CopyTo(hPrev);
        c0.AsSpan().CopyTo(cPrev);

        var preAct = new T[batch * G * hidden];
        var inSpan = input.AsSpan();

        for (int t = 0; t < seqLen; t++)
        {
            // preAct = b_ih + b_hh + W_ih · x_t + W_hh · h_{t-1}
            for (int b = 0; b < batch; b++)
            {
                for (int g = 0; g < G; g++)
                {
                    for (int j = 0; j < hidden; j++)
                    {
                        T acc = ops.Add(bIh[g * hidden + j], bHh[g * hidden + j]);
                        for (int i = 0; i < inputSize; i++)
                            acc = ops.Add(acc, ops.Multiply(
                                wIh[(g * hidden + j) * inputSize + i],
                                inSpan[(t * batch + b) * inputSize + i]));
                        for (int i = 0; i < hidden; i++)
                            acc = ops.Add(acc, ops.Multiply(
                                wHh[(g * hidden + j) * hidden + i],
                                hPrev[b * hidden + i]));
                        preAct[(b * G + g) * hidden + j] = acc;
                    }
                }
            }
            // c_t = f * c_{t-1} + i * tanh(g);  h_t = o * tanh(c_t).
            // PyTorch gate order: i, f, g, o.
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < hidden; j++)
                {
                    double i = Sigmoid(ops.ToDouble(preAct[(b * G + 0) * hidden + j]));
                    double f = Sigmoid(ops.ToDouble(preAct[(b * G + 1) * hidden + j]));
                    double g = Math.Tanh(ops.ToDouble(preAct[(b * G + 2) * hidden + j]));
                    double o = Sigmoid(ops.ToDouble(preAct[(b * G + 3) * hidden + j]));
                    double c = f * ops.ToDouble(cPrev[b * hidden + j]) + i * g;
                    double h = o * Math.Tanh(c);
                    cNew[b * hidden + j] = ops.FromDouble(c);
                    hNew[b * hidden + j] = ops.FromDouble(h);
                    outSpan[(t * batch + b) * hidden + j] = ops.FromDouble(h);
                }
            }
            // swap-without-alloc: copy new into prev for next step.
            Array.Copy(hNew, hPrev, hNew.Length);
            Array.Copy(cNew, cPrev, cNew.Length);
        }

        var hNTensor = new Tensor<T>(new[] { 1, batch, hidden });
        var cNTensor = new Tensor<T>(new[] { 1, batch, hidden });
        hPrev.AsSpan().CopyTo(hNTensor.AsWritableSpan());
        cPrev.AsSpan().CopyTo(cNTensor.AsWritableSpan());
        return (output, hNTensor, cNTensor);
    }

    private static (Tensor<T> Output, Tensor<T> HN, Tensor<T>? CN) ForwardPlainRnn<T>(
        Tensor<T> input, Tensor<T> h0, Tensor<T> weights, int hidden, bool useTanh)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var (seqLen, batch, inputSize) = ParseInputShape(input);
        EnsureHiddenShape(h0, batch, hidden);

        const int G = 1;
        var (wIh, wHh, bIh, bHh) = SplitWeights(weights, G, hidden, inputSize);
        var output = new Tensor<T>(new[] { seqLen, batch, hidden });
        var outSpan = output.AsWritableSpan();

        var hPrev = new T[batch * hidden];
        var hNew = new T[batch * hidden];
        h0.AsSpan().CopyTo(hPrev);
        var inSpan = input.AsSpan();

        for (int t = 0; t < seqLen; t++)
        {
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < hidden; j++)
                {
                    T acc = ops.Add(bIh[j], bHh[j]);
                    for (int i = 0; i < inputSize; i++)
                        acc = ops.Add(acc, ops.Multiply(wIh[j * inputSize + i], inSpan[(t * batch + b) * inputSize + i]));
                    for (int i = 0; i < hidden; i++)
                        acc = ops.Add(acc, ops.Multiply(wHh[j * hidden + i], hPrev[b * hidden + i]));
                    double a = ops.ToDouble(acc);
                    double activated = useTanh ? Math.Tanh(a) : Math.Max(0, a);
                    hNew[b * hidden + j] = ops.FromDouble(activated);
                    outSpan[(t * batch + b) * hidden + j] = hNew[b * hidden + j];
                }
            }
            Array.Copy(hNew, hPrev, hNew.Length);
        }

        var hNTensor = new Tensor<T>(new[] { 1, batch, hidden });
        hPrev.AsSpan().CopyTo(hNTensor.AsWritableSpan());
        return (output, hNTensor, null);
    }

    private static (Tensor<T> Output, Tensor<T> HN, Tensor<T>? CN) ForwardGru<T>(
        Tensor<T> input, Tensor<T> h0, Tensor<T> weights, int hidden)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var (seqLen, batch, inputSize) = ParseInputShape(input);
        EnsureHiddenShape(h0, batch, hidden);

        const int G = 3; // r, z, n
        var (wIh, wHh, bIh, bHh) = SplitWeights(weights, G, hidden, inputSize);
        var output = new Tensor<T>(new[] { seqLen, batch, hidden });
        var outSpan = output.AsWritableSpan();

        var hPrev = new T[batch * hidden];
        var hNew = new T[batch * hidden];
        h0.AsSpan().CopyTo(hPrev);
        var inSpan = input.AsSpan();

        for (int t = 0; t < seqLen; t++)
        {
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < hidden; j++)
                {
                    // Pre-acts for r and z gates: full (W·x + b_ih) + (W·h + b_hh).
                    double accR_ih = ops.ToDouble(bIh[0 * hidden + j]);
                    double accZ_ih = ops.ToDouble(bIh[1 * hidden + j]);
                    double accN_ih = ops.ToDouble(bIh[2 * hidden + j]);
                    for (int i = 0; i < inputSize; i++)
                    {
                        double xv = ops.ToDouble(inSpan[(t * batch + b) * inputSize + i]);
                        accR_ih += ops.ToDouble(wIh[(0 * hidden + j) * inputSize + i]) * xv;
                        accZ_ih += ops.ToDouble(wIh[(1 * hidden + j) * inputSize + i]) * xv;
                        accN_ih += ops.ToDouble(wIh[(2 * hidden + j) * inputSize + i]) * xv;
                    }
                    double accR_hh = ops.ToDouble(bHh[0 * hidden + j]);
                    double accZ_hh = ops.ToDouble(bHh[1 * hidden + j]);
                    double accN_hh = ops.ToDouble(bHh[2 * hidden + j]);
                    for (int i = 0; i < hidden; i++)
                    {
                        double hv = ops.ToDouble(hPrev[b * hidden + i]);
                        accR_hh += ops.ToDouble(wHh[(0 * hidden + j) * hidden + i]) * hv;
                        accZ_hh += ops.ToDouble(wHh[(1 * hidden + j) * hidden + i]) * hv;
                        accN_hh += ops.ToDouble(wHh[(2 * hidden + j) * hidden + i]) * hv;
                    }
                    double r = Sigmoid(accR_ih + accR_hh);
                    double z = Sigmoid(accZ_ih + accZ_hh);
                    // Gate n uses r * (W_hn · h + b_hn) per PyTorch.
                    double n = Math.Tanh(accN_ih + r * accN_hh);
                    double h = (1.0 - z) * n + z * ops.ToDouble(hPrev[b * hidden + j]);
                    hNew[b * hidden + j] = ops.FromDouble(h);
                    outSpan[(t * batch + b) * hidden + j] = hNew[b * hidden + j];
                }
            }
            Array.Copy(hNew, hPrev, hNew.Length);
        }

        var hNTensor = new Tensor<T>(new[] { 1, batch, hidden });
        hPrev.AsSpan().CopyTo(hNTensor.AsWritableSpan());
        return (output, hNTensor, null);
    }

    private static (int SeqLen, int Batch, int InputSize) ParseInputShape<T>(Tensor<T> input)
    {
        if (input.Rank != 3)
            throw new ArgumentException($"Input must be [seqLen, batch, inputSize]; got rank {input.Rank}.", nameof(input));
        return (input._shape[0], input._shape[1], input._shape[2]);
    }

    private static void EnsureHiddenShape<T>(Tensor<T> h, int batch, int hidden)
    {
        // Accept [batch, hidden] or [1, batch, hidden] (cuDNN-style numLayers·numDirections-leading).
        if (h.Rank == 2 && h._shape[0] == batch && h._shape[1] == hidden) return;
        if (h.Rank == 3 && h._shape[0] == 1 && h._shape[1] == batch && h._shape[2] == hidden) return;
        throw new ArgumentException($"Hidden state shape mismatch: expected [batch={batch}, hidden={hidden}].");
    }

    private static (T[] WIh, T[] WHh, T[] BIh, T[] BHh) SplitWeights<T>(
        Tensor<T> weights, int gates, int hidden, int inputSize)
    {
        long expectedIh = (long)gates * hidden * inputSize;
        long expectedHh = (long)gates * hidden * hidden;
        long expectedB = (long)gates * hidden;
        long expected = expectedIh + expectedHh + 2 * expectedB;
        if (weights.Length != expected)
            throw new ArgumentException(
                $"Packed weights length {weights.Length} doesn't match expected {expected} for " +
                $"gates={gates}, hidden={hidden}, inputSize={inputSize}. Layout is " +
                "[W_ih | W_hh | b_ih | b_hh].");

        var src = weights.AsSpan();
        var wIh = new T[expectedIh];
        var wHh = new T[expectedHh];
        var bIh = new T[expectedB];
        var bHh = new T[expectedB];
        int off = 0;
        src.Slice(off, (int)expectedIh).CopyTo(wIh); off += (int)expectedIh;
        src.Slice(off, (int)expectedHh).CopyTo(wHh); off += (int)expectedHh;
        src.Slice(off, (int)expectedB).CopyTo(bIh);  off += (int)expectedB;
        src.Slice(off, (int)expectedB).CopyTo(bHh);
        return (wIh, wHh, bIh, bHh);
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
}
