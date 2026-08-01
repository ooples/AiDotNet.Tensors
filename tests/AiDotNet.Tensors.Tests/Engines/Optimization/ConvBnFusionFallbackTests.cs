// Copyright (c) AiDotNet. All rights reserved.

using System.Reflection;
using System.Runtime.ExceptionServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

public sealed class ConvBnFusionFallbackTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonCpuFallback_BroadcastsFoldedBiasAcrossNchwChannels(bool depthwise)
    {
        CompiledStep<float> fused = CreateFusedStep(depthwise, withActivation: false);
        var proxy = CreateForwardingEngine(out _);

        fused.Execute(proxy, fused.OutputBuffer);

        Assert.Equal(new[] { 1, 2, 2, 3 }, fused.OutputBuffer._shape);
        float[] actual = fused.OutputBuffer.AsSpan().ToArray();
        Assert.All(actual[..6], value => Assert.Equal(10f, value));
        Assert.All(actual[6..], value => Assert.Equal(20f, value));
    }

    [Fact]
    public void ActivatedNonCpuFallback_KeepsRawChannelBiasForFusedConv2D()
    {
        CompiledStep<float> fused = CreateFusedStep(depthwise: false, withActivation: true);
        IEngine proxy = CreateForwardingEngine(out ForwardingEngineProxy forwarding);

        fused.Execute(proxy, fused.OutputBuffer);

        Assert.Equal(new[] { 2 }, forwarding.FusedConvBiasShape);
        float[] actual = fused.OutputBuffer.AsSpan().ToArray();
        Assert.All(actual[..6], value => Assert.Equal(10f, value));
        Assert.All(actual[6..], value => Assert.Equal(20f, value));
    }

    private static CompiledStep<float> CreateFusedStep(bool depthwise, bool withActivation)
    {
        int[] inputShape = depthwise ? [1, 2, 2, 3] : [1, 1, 2, 3];
        var input = new Tensor<float>(new float[inputShape.Aggregate(1, (a, b) => a * b)], inputShape);
        var weights = new Tensor<float>(new float[2], [2, 1, 1, 1]);
        var convOutput = new Tensor<float>([1, 2, 2, 3]);
        var bnOutput = new Tensor<float>([1, 2, 2, 3]);
        var finalOutput = withActivation ? new Tensor<float>([1, 2, 2, 3]) : bnOutput;
        var gamma = new Tensor<float>([1f, 1f], [2]);
        var beta = new Tensor<float>([10f, 20f], [2]);
        var mean = new Tensor<float>([0f, 0f], [2]);
        var variance = new Tensor<float>([1f, 1f], [2]);

        var conv = new CompiledStep<float>(
            depthwise ? "DepthwiseConv2D" : "Conv2D",
            static (_, _) => { },
            convOutput,
            [input, weights],
            savedState: [new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }]);
        var batchNorm = new CompiledStep<float>(
            "BatchNorm",
            static (_, _) => { },
            bnOutput,
            [convOutput, gamma, beta, mean, variance],
            savedState: [0d]);

        var steps = new List<CompiledStep<float>> { conv, batchNorm };
        if (withActivation)
        {
            steps.Add(new CompiledStep<float>(
                "ReLU",
                static (_, _) => { },
                finalOutput,
                [bnOutput]));
        }

        MethodInfo matcher = typeof(ConvBnFusionPass)
            .GetMethod("TryMatchConvBn", BindingFlags.Static | BindingFlags.NonPublic)!
            .MakeGenericMethod(typeof(float));
        object[] arguments = [steps.ToArray(), 0, null!, 0];
        bool matched = (bool)matcher.Invoke(null, arguments)!;

        Assert.True(matched);
        Assert.Equal(withActivation ? 3 : 2, (int)arguments[3]);
        return Assert.IsType<CompiledStep<float>>(arguments[2]);
    }

    private static IEngine CreateForwardingEngine(out ForwardingEngineProxy forwarding)
    {
        IEngine engine = DispatchProxy.Create<IEngine, ForwardingEngineProxy>();
        forwarding = (ForwardingEngineProxy)(object)engine;
        forwarding.Inner = new CpuEngine();
        return engine;
    }

    private class ForwardingEngineProxy : DispatchProxy
    {
        internal CpuEngine Inner { get; set; } = null!;
        internal int[]? FusedConvBiasShape { get; private set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod is null || args is null)
                throw new InvalidOperationException("Missing forwarded engine invocation metadata.");

            if (targetMethod.Name == nameof(IEngine.FusedConv2D) && args[2] is Tensor<float> bias)
                FusedConvBiasShape = (int[])bias._shape.Clone();

            try
            {
                return targetMethod.Invoke(Inner, args);
            }
            catch (TargetInvocationException exception) when (exception.InnerException is not null)
            {
                ExceptionDispatchInfo.Capture(exception.InnerException).Throw();
                throw;
            }
        }
    }
}
