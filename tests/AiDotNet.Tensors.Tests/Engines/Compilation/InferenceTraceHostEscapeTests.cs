// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A forward that reads a traced value on the host and computes from it outside the graph cannot be an inference
/// plan: the plan would replay that computation's trace-time result as a constant. Found by the AiDotNet model
/// sweep (HippoMemoryCellLayer's LegS solve reads its right-hand side through GetDataArray).
/// </summary>
public sealed class InferenceTraceHostEscapeTests
{
    private static Tensor<float> HostDouble(Tensor<float> x)
    {
        var data = x.DataVector.GetDataArray();       // a host read of a traced value
        var output = new Tensor<float>(x.Shape.ToArray());
        var dst = output.DataVector.GetDataArray();
        for (int i = 0; i < x.Length; i++) dst[i] = 2 * data[i];
        return output;
    }

    [Fact]
    public void AHostReadOfATracedValue_RefusesToCompile()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        using var cache = new CompiledModelCache<float>();
        var ex = Assert.Throws<GraphCaptureNotSupportedException>(() =>
            cache.GetOrCompileInference(input._shape, () => engine.TensorAddScalar(HostDouble(engine.ReLU(input)), 1f)));
        Assert.Equal(GraphCaptureLimitation.HostBoundary, ex.Limitation);
    }

    [Fact]
    public void ReadingTheInputLeaf_IsNotAnEscape()
    {
        // A leaf is not a traced value: a forward may validate its input on the host and still compile.
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[] { 1, -2, 3, -4 }, new[] { 2, 2 });
        using var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileInference(input._shape, () =>
        {
            Assert.Equal(1f, input.DataVector.GetDataArray()[0]);
            return engine.ReLU(input);
        });
        var request = new Tensor<float>(new float[] { -1, 5, -3, 7 }, new[] { 2, 2 });
        plan.SetInputs(new[] { request });
        Assert.Equal(new[] { 0f, 5f, 0f, 7f }, plan.Execute().AsSpan().ToArray());
    }
}
