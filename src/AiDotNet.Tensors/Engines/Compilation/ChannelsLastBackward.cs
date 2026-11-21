// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// The backwards of <see cref="ConvChannelEpilogueFusionPattern"/>'s and <see cref="ConvBiasBatchNormFusionPattern"/>'s
/// nodes when <see cref="ChannelsLastLayoutPass"/> made an activation of theirs channels-last: the same gradients, through
/// the channels-last kernels. A channels-last activation of logical shape [n, c, h, w] is, as storage, a [n * h * w, c]
/// matrix, so a per-channel reduction is a column sum of that view.
/// </summary>
internal static class ChannelsLastBackward
{
    private static DirectGpuTensorEngine Gpu(IEngine engine)
        => engine.TryGet<DirectGpuTensorEngine>(out var gpu) ? gpu
            : throw new System.InvalidOperationException("A channels-last plan step ran on a non-GPU engine.");

    private static T Ran<T>(T? value, string what) where T : class
        => value ?? throw new System.InvalidOperationException("The channels-last " + what + " did not run.");

    // The per-channel sum of an activation (over n, h, w) in its layout, shaped [c].
    internal static Tensor<float> ChannelSums(IEngine engine, Tensor<float> t, bool channelsLast)
    {
        int c = t.Shape._dims[1];
        return channelsLast
            ? engine.Reshape(engine.ReduceSum(engine.Reshape(t, new[] { t.Length / c, c }), new[] { 0 }, keepDims: false),
                new[] { c })
            : engine.ReduceSum(t, new[] { 0, 2, 3 }, keepDims: false);
    }

    // t * v[c] (or t + v[c]) for an activation in its layout and a per-channel operand of C values.
    private static Tensor<float> ChannelBroadcast(CpuEngine host, IEngine engine, Tensor<float> t, Tensor<float> v,
        bool channelsLast, bool multiply)
    {
        var shape = t.Shape.ToArray();
        int c = shape[1];
        if (!channelsLast)
        {
            var channelShaped = engine.Reshape(v, new[] { 1, c, 1, 1 });
            return multiply ? host.TensorBroadcastMultiply(t, channelShaped) : host.TensorBroadcastAdd(t, channelShaped);
        }
        var rows = engine.Reshape(t, new[] { t.Length / c, c });
        var row = engine.Reshape(v, new[] { 1, c });
        var r = multiply ? host.TensorBroadcastMultiply(rows, row) : host.TensorBroadcastAdd(rows, row);
        return engine.Reshape(r, shape);
    }

    /// <summary>As <see cref="BackwardFunctions{T}.ConvChannelEpilogueBackward"/>, the layouts the slot's.</summary>
    internal static void ConvChannelEpilogue(Tensor<float> gradOutput, Tensor<float>[] inputs, Tensor<float> output,
        object[] savedState, ChannelsLastSlot slot, IEngine engine, Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var gpu = Gpu(engine);
        var z = (Tensor<float>)savedState[2];
        int activation = (int)savedState[3];
        float slope = (float)(double)savedState[4];
        int biasIndex = (int)savedState[5], scaleIndex = (int)savedState[6], shiftIndex = (int)savedState[7];
        var bias = biasIndex >= 0 ? inputs[biasIndex] : null;
        var scale = scaleIndex >= 0 ? inputs[scaleIndex] : null;
        var shift = shiftIndex >= 0 ? inputs[shiftIndex] : null;
        bool scaleGrad = scale is not null && DifferentiableOps.IsGradientRequired(scale);
        bool shiftGrad = shift is not null && DifferentiableOps.IsGradientRequired(shift);
        bool nhwc = slot.OutputNhwc;

        Tensor<float> gradZ;
        if (!scaleGrad && !shiftGrad)
        {
            gradZ = Ran(gpu.TryChannelEpilogueBackward(gradOutput, output, scale, activation, slope, nhwc), "epilogue backward");
        }
        else
        {
            var host = engine.TryGet<CpuEngine>(out var cpu) ? cpu
                : throw new System.NotSupportedException("ConvChannelEpilogue backward needs a CpuEngine-derived engine.");
            var gradPre = activation == ChannelEpilogueActivation.Relu ? engine.ReluBackward(gradOutput, output)
                : activation == ChannelEpilogueActivation.LeakyRelu ? engine.LeakyReluBackward(gradOutput, output, slope)
                : gradOutput;
            if (shiftGrad)
                DifferentiableOps.AccumulateGrad(grads, shift!,
                    engine.Reshape(ChannelSums(engine, gradPre, nhwc), shift!._shape), engine);
            if (scaleGrad)
            {
                var scaledInput = bias is null ? z : ChannelBroadcast(host, engine, z, bias, nhwc, multiply: false);
                DifferentiableOps.AccumulateGrad(grads, scale!,
                    engine.Reshape(ChannelSums(engine, engine.TensorMultiply(gradPre, scaledInput), nhwc), scale!._shape),
                    engine);
            }
            gradZ = scale is null ? gradPre : ChannelBroadcast(host, engine, gradPre, scale, nhwc, multiply: true);
        }
        if (bias is not null && DifferentiableOps.IsGradientRequired(bias))
            DifferentiableOps.AccumulateGrad(grads, bias, ChannelSums(engine, gradZ, nhwc), engine);
        slot.Convolution.Backward(gpu, gradZ, inputs[0], inputs[1], slot, engine, grads);
    }

    /// <summary>As <see cref="BackwardFunctions{T}.ConvBiasBatchNormBackward"/>, the layouts the slot's.</summary>
    internal static void ConvBiasBatchNorm(Tensor<float> gradOutput, Tensor<float>[] inputs, Tensor<float> output,
        object[] savedState, ChannelsLastSlot slot, IEngine engine, Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var gpu = Gpu(engine);
        var z = (Tensor<float>)savedState[4];
        Tensor<float>? gradZ = null;
        if (slot.OutputNhwc)
        {
            var inner = (object[])savedState[3];
            var mean = (Tensor<float>)inner[0];
            var variance = (Tensor<float>)inner[1];
            double epsilon = (double)inner[2];
            int act = (int)savedState[6];
            float slope = (float)(double)savedState[7];
            var gamma = inputs[3];
            var beta = inputs[4];
            gradZ = Ran(gpu.TryBatchNormChannelsLastBackward(gradOutput, act != 0 ? output : null, z, gamma, mean, variance,
                epsilon, act, slope, out var gradGamma, out var gradBeta), "BatchNorm backward");
            if (DifferentiableOps.IsGradientRequired(gamma))
                DifferentiableOps.AccumulateGrad(grads, gamma, engine.Reshape(gradGamma!, gamma._shape), engine);
            if (DifferentiableOps.IsGradientRequired(beta))
                DifferentiableOps.AccumulateGrad(grads, beta, engine.Reshape(gradBeta!, beta._shape), engine);
        }
        else
        {
            var innerBackward = (BackwardFunction<float>)savedState[2];
            innerBackward(gradOutput, new[] { z, inputs[3], inputs[4] }, output, (object[])savedState[3], engine, grads);
            if (grads.TryGetValue(z, out var g))
            {
                grads.Remove(z);
                gradZ = g;
            }
        }
        if (gradZ is not null) slot.Convolution.Backward(gpu, gradZ, inputs[0], inputs[1], slot, engine, grads);
        if (DifferentiableOps.IsGradientRequired(inputs[2]))
            DifferentiableOps.AccumulateGrad(grads, inputs[2], new Tensor<float>((int[])inputs[2]._shape.Clone()), engine);
    }
}
