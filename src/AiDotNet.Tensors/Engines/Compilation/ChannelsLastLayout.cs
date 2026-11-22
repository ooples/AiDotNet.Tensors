// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Channels-last activations inside a compiled plan's convolution stacks: the fused convolution nodes that can run with
// NHWC activations carry a ChannelsLastSlot; ChannelsLastLayoutPass decides, per activation, whether its producer writes
// it channels-last and its consumers read it so.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// The layout of a fused convolution node's activation input and output. Both start NCHW;
/// <see cref="ChannelsLastLayoutPass"/> sets them before the plan first runs, and the node's forward and backward read
/// them on every step. A channels-last tensor keeps its logical NCHW shape; only its storage order differs.
/// </summary>
internal sealed class ChannelsLastSlot
{
    internal ChannelsLastSlot(LazyNode<float> conv, ChannelsLastConvolution convolution, Tensor<float> input,
        bool outputEligible)
    {
        Convolution = convolution;
        Input = input;
        OutputEligible = outputEligible;
        _ = conv;
    }

    /// <summary>The node's convolution.</summary>
    internal ChannelsLastConvolution Convolution { get; }

    /// <summary>The convolution's activation input (x).</summary>
    internal Tensor<float> Input { get; }

    /// <summary>Whether the node's epilogue can write its output channels-last (its BatchNorm or epilogue takes it).</summary>
    internal bool OutputEligible { get; }

    /// <summary>x is channels-last.</summary>
    internal bool InputNhwc { get; set; }

    /// <summary>The node's output (and the convolution's output inside it) is channels-last.</summary>
    internal bool OutputNhwc { get; set; }

    internal bool Any => InputNhwc || OutputNhwc;

    /// <summary>The slot a fused node's saved state carries, or null.</summary>
    internal static ChannelsLastSlot? Of(ILazyNode node)
    {
        if (node is not LazyNode<float> { SavedState: { } state }) return null;
        foreach (var item in state)
            if (item is ChannelsLastSlot slot) return slot;
        return null;
    }
}

/// <summary>
/// A recorded <c>Conv2D</c> or <c>ConvTranspose2D</c> node's geometry, and its passes with a layout per activation
/// through <see cref="DirectGpuTensorEngine.TryConvolutionChannelsLast"/>. A transposed convolution is run as the passes
/// of the convolution it transposes: its forward that convolution's backward data, its input gradient that
/// convolution's forward, its kernel gradient that convolution's backward filter.
/// </summary>
internal sealed class ChannelsLastConvolution
{
    private readonly ChannelsLastConvGeometry _g;
    private readonly bool _transposed;
    private readonly int[] _inputShape, _outputShape, _kernelShape;

    private ChannelsLastConvolution(ChannelsLastConvGeometry g, bool transposed, int[] inputShape, int[] outputShape,
        int[] kernelShape)
    {
        _g = g;
        _transposed = transposed;
        _inputShape = inputShape;
        _outputShape = outputShape;
        _kernelShape = kernelShape;
    }

    /// <summary>The geometry of a recorded convolution node, or null for any other node or an unsupported form.</summary>
    internal static ChannelsLastConvolution? Describe(LazyNode<float> conv)
    {
        if (conv.GetInputsArray() is not { Length: 2 } inputs || inputs[0] is not { Rank: 4 } x
            || inputs[1] is not { Rank: 4 } w || conv.Output is not { Rank: 4 } z || conv.SavedState is not { } state
            || state.Length < 2 || state[0] is not int[] { Length: 2 } stride || state[1] is not int[] { Length: 2 } pad)
            return null;
        var xd = x.Shape._dims;
        var wd = w.Shape._dims;
        var zd = z.Shape._dims;
        if (conv.OpName == "Conv2D" && conv.OpType == LazyNodeType.Conv2D)
        {
            int[] dilation = state.Length > 2 && state[2] is int[] { Length: 2 } d ? d : new[] { 1, 1 };
            if (wd[1] != xd[1] || wd[0] != zd[1] || zd[0] != xd[0]) return null;
            return new ChannelsLastConvolution(new ChannelsLastConvGeometry(xd[0], xd[1], xd[2], xd[3], wd[0], wd[2], wd[3],
                    zd[2], zd[3], pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1]), false,
                x.Shape.ToArray(), z.Shape.ToArray(), w.Shape.ToArray());
        }
        if (conv.OpName == "ConvTranspose2D")
        {
            // x [n, cin, h, w], w [cin, cout, r, s], z [n, cout, oh, ow]: the convolution z -> x it transposes.
            if (wd[0] != xd[1] || wd[1] != zd[1] || zd[0] != xd[0]) return null;
            if ((zd[2] + 2 * pad[0] - wd[2]) / stride[0] + 1 != xd[2]
                || (zd[3] + 2 * pad[1] - wd[3]) / stride[1] + 1 != xd[3])
                return null;
            return new ChannelsLastConvolution(new ChannelsLastConvGeometry(zd[0], zd[1], zd[2], zd[3], xd[1], wd[2], wd[3],
                    xd[2], xd[3], pad[0], pad[1], stride[0], stride[1], 1, 1), true,
                x.Shape.ToArray(), z.Shape.ToArray(), w.Shape.ToArray());
        }
        return null;
    }

    /// <summary>The node's output channels (z's).</summary>
    internal int OutputChannels => _outputShape[1];

    /// <summary>The node's input channels (x's).</summary>
    internal int InputChannels => _inputShape[1];

    /// <summary>z = conv(x, w), x and z in the slot's layouts; null when the backend does not take it.</summary>
    internal Tensor<float>? Forward(DirectGpuTensorEngine gpu, Tensor<float> x, Tensor<float> w, bool inputNhwc,
        bool outputNhwc)
        => _transposed
            ? gpu.TryConvolutionChannelsLast(1, w, x, _outputShape, _g, inputNhwc: outputNhwc, outputNhwc: inputNhwc)
            : gpu.TryConvolutionChannelsLast(0, x, w, _outputShape, _g, inputNhwc, outputNhwc);

    /// <summary>The input gradient from z's (layouts as the forward's), or null.</summary>
    internal Tensor<float>? InputGradient(DirectGpuTensorEngine gpu, Tensor<float> gradZ, Tensor<float> w, bool inputNhwc,
        bool outputNhwc)
        => _transposed
            ? gpu.TryConvolutionChannelsLast(0, gradZ, w, _inputShape, _g, inputNhwc: outputNhwc, outputNhwc: inputNhwc)
            : gpu.TryConvolutionChannelsLast(1, w, gradZ, _inputShape, _g, inputNhwc, outputNhwc);

    /// <summary>The kernel gradient from x and z's gradient, or null.</summary>
    internal Tensor<float>? KernelGradient(DirectGpuTensorEngine gpu, Tensor<float> x, Tensor<float> gradZ, bool inputNhwc,
        bool outputNhwc)
        => _transposed
            ? gpu.TryConvolutionChannelsLast(2, gradZ, x, _kernelShape, _g, inputNhwc: outputNhwc, outputNhwc: inputNhwc)
            : gpu.TryConvolutionChannelsLast(2, x, gradZ, _kernelShape, _g, inputNhwc, outputNhwc);

    /// <summary>
    /// The convolution's backward through the channels-last passes: x's and the kernel's gradients where required,
    /// accumulated into <paramref name="grads"/>. A pass the backend refuses throws: the plan chose the layout because
    /// the backend accepted it, so a refusal mid-step is a defect, not a fallback.
    /// </summary>
    internal void Backward(DirectGpuTensorEngine gpu, Tensor<float> gradZ, Tensor<float> x, Tensor<float> w,
        ChannelsLastSlot slot, IEngine engine, Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        // Both passes read dY: under autocast the second reads the 16-bit copy the first narrowed (one backward function
        // is one compiled action, so the copy cannot go stale or be overwritten inside it).
        using var share = new DirectGpuTensorEngine.BackwardConvPrecisionScope(null, shareNarrowedOperands: true);
        if (DifferentiableOps.IsGradientRequired(x))
            DifferentiableOps.AccumulateGrad(grads, x,
                InputGradient(gpu, gradZ, w, slot.InputNhwc, slot.OutputNhwc)
                ?? throw new System.InvalidOperationException("The channels-last convolution input gradient did not run."),
                engine);
        if (DifferentiableOps.IsGradientRequired(w))
            DifferentiableOps.AccumulateGrad(grads, w,
                KernelGradient(gpu, x, gradZ, slot.InputNhwc, slot.OutputNhwc)
                ?? throw new System.InvalidOperationException("The channels-last convolution kernel gradient did not run."),
                engine);
    }
}

/// <summary>
/// Chooses the layout of every activation a channels-last-capable fused convolution node writes: channels-last when the
/// engine prefers it (cuDNN's tensor-core convolutions), the node can write it, and every reader of the activation reads
/// it so -- another such node's convolution input, or a chain of elementwise unary ops (<see cref="TransparentOps"/>)
/// whose last output only such inputs read -- and nothing outside the plan observes it. The readers' inputs are then
/// set channels-last too. Everything else keeps NCHW, so a plan output, a host read or any other op never sees the
/// other layout.
/// </summary>
internal sealed class ChannelsLastLayoutPass : ILazyGraphOptimizationPass
{
    private readonly HashSet<ILazyNode>? _observableOutputs;

    /// <summary>Elementwise unary ops (and their backwards) that do not depend on the layout.</summary>
    internal static readonly HashSet<LazyNodeType> TransparentOps = new() { LazyNodeType.Tanh };

    public ChannelsLastLayoutPass(HashSet<ILazyNode>? observableOutputs = null) => _observableOutputs = observableOutputs;

    public string Name => "ChannelsLastLayout";

    // False keeps every activation NCHW (tests compare the two). Whether a plan goes channels-last is otherwise the
    // backend's call (IChannelsLastBackend.PrefersChannelsLast).
    internal static bool s_enabled = true;

    /// <summary>Activations a plan compiled channels-last (tests).</summary>
    internal static long ChannelsLastActivations;

    public List<ILazyNode> Run(List<ILazyNode> nodes)
    {
        if (!s_enabled) return nodes;
        var consumers = new Dictionary<Tensor<float>, List<ILazyNode>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        var outputs = new HashSet<Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        foreach (var node in nodes)
        {
            if (node is not LazyNode<float> typed) continue;
            outputs.Add(typed.Output);
            foreach (var input in typed.GetInputsArray())
            {
                if (!consumers.TryGetValue(input, out var list)) consumers[input] = list = new List<ILazyNode>(2);
                if (!list.Contains(node)) list.Add(node);
            }
        }
        var observed = new HashSet<Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        if (_observableOutputs is not null)
            foreach (var node in _observableOutputs)
                if (node is LazyNode<float> typed) observed.Add(typed.Output);

        foreach (var node in nodes)
        {
            if (ChannelsLastSlot.Of(node) is not { OutputEligible: true } slot || node is not LazyNode<float> producer
                || !producer.RecordingEngine.TryGet<DirectGpuTensorEngine>(out var gpu) || !gpu.PrefersChannelsLast
                || !gpu.AcceptsChannelsLast(slot.Convolution.InputChannels, slot.Convolution.OutputChannels))
                continue;
            var readers = new List<ChannelsLastSlot>();
            var chain = new List<Tensor<float>>();
            if (!ReadersAllChannelsLast(producer.Output, consumers, observed, readers, chain, 0)) continue;
            slot.OutputNhwc = true;
            foreach (var reader in readers) reader.InputNhwc = true;
            System.Threading.Interlocked.Increment(ref ChannelsLastActivations);
        }
        return nodes;
    }

    // Every reader of `tensor` is a slot's convolution input, or a transparent op whose output's readers all are.
    private static bool ReadersAllChannelsLast(Tensor<float> tensor,
        Dictionary<Tensor<float>, List<ILazyNode>> consumers, HashSet<Tensor<float>> observed,
        List<ChannelsLastSlot> readers, List<Tensor<float>> chain, int depth)
    {
        if (depth > 8 || observed.Contains(tensor) || !consumers.TryGetValue(tensor, out var list) || list.Count == 0)
            return false;
        chain.Add(tensor);
        foreach (var reader in list)
        {
            if (ChannelsLastSlot.Of(reader) is { } slot && reader is LazyNode<float> typed)
            {
                // Only as the convolution's activation input: any other operand position reads NCHW.
                var inputs = typed.GetInputsArray();
                for (int i = 1; i < inputs.Length; i++)
                    if (ReferenceEquals(inputs[i], tensor)) return false;
                if (!ReferenceEquals(slot.Input, tensor)) return false;
                readers.Add(slot);
                continue;
            }
            if (reader is LazyNode<float> unary && TransparentOps.Contains(unary.OpType)
                && unary.GetInputsArray() is { Length: 1 } && unary.BackwardFn is not null
                && ReadersAllChannelsLast(unary.Output, consumers, observed, readers, chain, depth + 1))
                continue;
            return false;
        }
        return true;
    }
}
