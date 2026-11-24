// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: convolution -> per-channel bias / inference-BatchNorm affine -> activation as one epilogue pass.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a <c>Conv2D</c> or <c>ConvTranspose2D</c> whose output runs through a chain of per-channel steps -- a
/// <c>TensorChannelBiasAdd</c>, then a <c>TensorBroadcastMultiply</c> and a <c>TensorBroadcastAdd</c> by per-channel
/// [1, C, 1, 1] operands (an inference BatchNorm's scale and shift), then a ReLU or a LeakyReLU of positive slope, each
/// step optional -- and emits one node: the bare convolution, then all the steps as one pass over its output
/// (<see cref="DirectGpuTensorEngine.TryChannelEpilogue"/>), and one pass back (<c>dz = act'(y) * dy * scale</c>).
/// </summary>
/// <remarks>
/// <para>A GAN generator step's frozen discriminator runs each of its layers as conv, bias, multiply, add, LeakyReLU:
/// four passes over the activation after the convolution in each direction, one with this fusion. A discriminator's first
/// layer (conv, bias, LeakyReLU) saves one. A chain of only a bias and a ReLU is left to
/// <see cref="ConvBiasFusionPattern"/>, whose convolution applies them in its own output kernel.</para>
/// <para>The values are the separate steps' (each rounded on its own); the parameter gradients are theirs too: the
/// shift's is the sum of the pre-activation gradient, the scale's the sum of it times the scaled input, the bias's the sum
/// of the input gradient z receives, each per channel and only where required.</para>
/// </remarks>
internal sealed class ConvChannelEpilogueFusionPattern : IFusionPattern
{
    public string Name => "ConvChannelEpilogue";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records ConvChannelEpilogueBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> conv || !conv.RecordingEngine.Is<DirectGpuTensorEngine>()
            || !(conv.OpName == "Conv2D" && conv.OpType == LazyNodeType.Conv2D || conv.OpName == "ConvTranspose2D")
            || conv.BackwardFn is null || conv.SavedState is null || conv.Output is not { Rank: 4 } z
            || conv.GetInputsArray() is not { Length: 2 } convInputs)
            return false;
        int channels = z.Shape._dims[1];
        var absorbed = new List<LazyNode<float>>(4);
        LazyNode<float> last = conv;
        Tensor<float>? bias = null, scale = null, shift = null;
        int activation = ChannelEpilogueActivation.Identity;
        float slope = 0f;

        LazyNode<float>? Next(LazyNode<float> producer)
            => PatternHelpers.FanOutAtMostOne(producer, consumerCounts) ? SoleConsumer(nodes, index, producer, alreadyRemoved)
                : null;

        var next = Next(last);
        if (next is { OpName: "TensorChannelBiasAdd" } biasAdd && ReferenceEquals(biasAdd.Input0, last.Output)
            && biasAdd.Input1 is { Rank: 1 } b && b.Length == channels && PatternHelpers.SameShape(biasAdd.Output, z))
        {
            bias = b;
            absorbed.Add(biasAdd);
            last = biasAdd;
            next = Next(last);
        }
        if (next is { OpName: "TensorBroadcastMultiply" } multiply
            && ChannelOperand(multiply, last.Output, channels) is { } s && PatternHelpers.SameShape(multiply.Output, z))
        {
            scale = s;
            absorbed.Add(multiply);
            last = multiply;
            next = Next(last);
        }
        if (next is { OpName: "TensorBroadcastAdd" } add
            && ChannelOperand(add, last.Output, channels) is { } t && PatternHelpers.SameShape(add.Output, z))
        {
            shift = t;
            absorbed.Add(add);
            last = add;
            next = Next(last);
        }
        if (next is { OpType: LazyNodeType.ReLU } relu && ReferenceEquals(relu.Input0, last.Output))
        {
            activation = ChannelEpilogueActivation.Relu;
            absorbed.Add(relu);
            last = relu;
        }
        else if (next is { OpName: "LeakyReLU", SavedState: { Length: 1 } leakyState } leaky
                 && ReferenceEquals(leaky.Input0, last.Output) && leakyState[0] is double ls
                 && ChannelEpilogueActivation.IsSlope((float)ls) && ls <= float.MaxValue)
        {
            activation = ChannelEpilogueActivation.LeakyRelu;
            slope = (float)ls;
            absorbed.Add(leaky);
            last = leaky;
        }
        // Worth a node of its own past what the convolution's own epilogue fuses (a bias and a ReLU), or where the input
        // comes from a node that can write it channels-last (this node can read it so; ConvBiasFusionPattern's cannot).
        bool upstreamChannelsLast = ChannelsLastLayoutPass.s_enabled
            && convInputs[0].LazySource is { } source && ChannelsLastSlot.Of(source) is not null
            && conv.RecordingEngine.TryGet<DirectGpuTensorEngine>(out var recording) && recording.PrefersChannelsLast;
        if (scale is null && shift is null && activation != ChannelEpilogueActivation.LeakyRelu && !upstreamChannelsLast)
            return false;
        var description = ChannelsLastConvolution.Describe(conv);
        var slot = description is null ? null : new ChannelsLastSlot(conv, description, convInputs[0], outputEligible: true);
        var x = convInputs[0];
        var kernel = convInputs[1];

        var inputs = new List<Tensor<float>>(5) { convInputs[0], convInputs[1] };
        int biasIndex = bias is null ? -1 : Add(inputs, bias);
        int scaleIndex = scale is null ? -1 : Add(inputs, scale);
        int shiftIndex = shift is null ? -1 : Add(inputs, shift);
        var convExecute = conv.Execute;
        int act = activation;
        float sl = slope;
        var node = new LazyNode<float>(LazyNodeType.Custom, Name, inputs.ToArray(), last.Output,
            (eng, result) =>
            {
                if (slot is { Any: true })
                {
                    // Channels-last activations (ChannelsLastLayoutPass): the convolution and the epilogue in the slot's
                    // layouts. The plan chose them because the backend takes them, so a refusal is a defect.
                    var device = eng.TryGet<DirectGpuTensorEngine>(out var g) ? g
                        : throw new System.InvalidOperationException("A channels-last plan step ran on a non-GPU engine.");
                    var convolved = slot.Convolution.Forward(device, x, kernel, slot.InputNhwc, slot.OutputNhwc)
                        ?? throw new System.InvalidOperationException("The channels-last convolution did not run.");
                    DirectGpuTensorEngine.CopyResultInto(eng, convolved, z);
                    var epilogue = device.TryChannelEpilogue(z, bias, scale, shift, act, sl, slot.OutputNhwc)
                        ?? throw new System.InvalidOperationException("The channels-last epilogue did not run.");
                    DirectGpuTensorEngine.CopyResultInto(eng, epilogue, result);
                    return;
                }
                convExecute(eng, z);   // the convolution alone; its epilogue below
                Tensor<float>? computed = eng.TryGet<DirectGpuTensorEngine>(out var gpu)
                    ? gpu.TryChannelEpilogue(z, bias, scale, shift, act, sl)
                    : null;
                if (computed is null)
                {
                    // The steps as recorded.
                    computed = z;
                    if (bias is not null) computed = eng.TensorChannelBiasAdd(computed, bias);
                    var broadcast = eng.TryGet<CpuEngine>(out var host) ? host : null;
                    if (scale is not null) computed = broadcast!.TensorBroadcastMultiply(computed, scale);
                    if (shift is not null) computed = broadcast!.TensorBroadcastAdd(computed, shift);
                    if (act == ChannelEpilogueActivation.Relu) computed = eng.ReLU(computed);
                    else if (act == ChannelEpilogueActivation.LeakyRelu) computed = eng.LeakyReLU(computed, sl);
                }
                DirectGpuTensorEngine.CopyResultInto(eng, computed, result);
            },
            BackwardFunctions<float>.ConvChannelEpilogueBackward,
            new object?[] { conv.BackwardFn, conv.SavedState, z, act, (double)sl, biasIndex, scaleIndex, shiftIndex, slot }!)
        {
            RecordingEngine = conv.RecordingEngine,
        };
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(conv);
        foreach (var step in absorbed) alreadyRemoved.Add(step);
        return true;
    }

    private static int Add(List<Tensor<float>> inputs, Tensor<float> tensor)
    {
        inputs.Add(tensor);
        return inputs.Count - 1;
    }

    // The per-channel operand of a broadcast binary node whose other input is `activation`: C values laid out along the
    // channel axis of an NCHW activation ([1, C, 1, 1] or [C, 1, 1]); null for any other broadcast.
    private static Tensor<float>? ChannelOperand(LazyNode<float> node, Tensor<float> activation, int channels)
    {
        var operands = node.GetInputsArray();
        if (operands.Length != 2) return null;
        Tensor<float> other;
        if (ReferenceEquals(operands[0], activation) && !ReferenceEquals(operands[1], activation)) other = operands[1];
        else if (ReferenceEquals(operands[1], activation) && !ReferenceEquals(operands[0], activation)) other = operands[0];
        else return null;
        var d = other.Shape._dims;
        bool channelAxis = other.Rank == 4 ? d[0] == 1 && d[1] == channels && d[2] == 1 && d[3] == 1
            : other.Rank == 3 && d[0] == channels && d[1] == 1 && d[2] == 1;
        return channelAxis && other.Length == channels ? other : null;
    }

    /// <summary>The one node after <paramref name="index"/> that reads <paramref name="producer"/>, or null.</summary>
    private static LazyNode<float>? SoleConsumer(IReadOnlyList<ILazyNode> nodes, int index, ILazyNode producer,
        HashSet<ILazyNode> alreadyRemoved)
    {
        LazyNode<float>? found = null;
        for (int j = index + 1; j < nodes.Count; j++)
        {
            if (alreadyRemoved.Contains(nodes[j]) || !PatternHelpers.IsConsumerOf(nodes[j], producer)) continue;
            if (found is not null || nodes[j] is not LazyNode<float> typed) return null;
            found = typed;
        }
        return found;
    }
}
