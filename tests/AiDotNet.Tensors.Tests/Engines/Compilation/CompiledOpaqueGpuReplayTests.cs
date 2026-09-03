// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class CompiledOpaqueGpuReplayTests
{
    [Fact]
    public void OpaqueInferenceCapture_RecordsThroughBase_AndReplaysOnGpuBackend()
    {
        var state = new MockBackendState();
        IDirectGpuBackend backend = MockDirectGpuBackend.Create(state);
        using var directGpu = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(directGpu);
        using var cache = new CompiledModelCache<float>();
        var input = new Tensor<float>(new[] { -1.25f, -0.2f, 0.4f, 1.8f }, new[] { 2, 2 });

        ICompiledPlan<float> plan = cache.GetOrCompileInference(
            input, () => engine.NativeTanh(input));

        // The outer GPU override must route the active trace to CpuEngine so the graph node is
        // recorded. Its one suspended materialization still uses the selected GPU backend to
        // establish the exact output shape and proves that ordinary GPU dispatch remains enabled.
        Assert.Equal(1, state.TanhCalls);

        AssertTanh(input, plan.Execute());
        Assert.Equal(2, state.TanhCalls);

        input[0] = 0.75f;
        plan.SetInputs(new[] { input });
        AssertTanh(input, plan.Execute());
        Assert.Equal(3, state.TanhCalls);
    }

    private static void AssertTanh(Tensor<float> input, Tensor<float> actual)
    {
        float[] values = actual.ToArray();
        Assert.Equal(input.Length, values.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.InRange(values[i], MathF.Tanh(input[i]) - 1e-6f, MathF.Tanh(input[i]) + 1e-6f);
    }
}
