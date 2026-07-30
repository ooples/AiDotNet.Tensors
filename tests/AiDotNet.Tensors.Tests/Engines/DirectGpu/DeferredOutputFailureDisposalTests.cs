// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("EngineCurrentGlobalState")]
public sealed class DeferredOutputFailureDisposalTests
{
    [Theory]
    [InlineData("GLU")]
    [InlineData("GeGLU")]
    [InlineData("ReGLU")]
    [InlineData("SwiGLU")]
    [InlineData("Upsample")]
    [InlineData("TensorSoftmaxBackward")]
    public void DeferredOutput_IsDisposedWhenGpuDispatchFails(string operation)
    {
        using var scope = new MockGpuEngineScope();

        Tensor<float> result = RunOperation(scope.Engine, operation);

        Assert.NotNull(result);
        Assert.NotEmpty(scope.State.AllocatedBuffers);
        MockGpuBuffer output = scope.State.AllocatedBuffers[scope.State.AllocatedBuffers.Count - 1];
        Assert.Equal(1, output.DisposeCount);
    }

    private static Tensor<float> RunOperation(IEngine engine, string operation)
    {
        var gatedInput = new Tensor<float>(
            new[] { -0.8f, -0.4f, 0.2f, 0.6f, 0.3f, -0.1f, 0.7f, -0.5f },
            new[] { 1, 8 });

        switch (operation)
        {
            case "GLU": return engine.GLU(gatedInput, -1);
            case "GeGLU": return engine.GeGLU(gatedInput, -1);
            case "ReGLU": return engine.ReGLU(gatedInput, -1);
            case "SwiGLU": return engine.SwiGLU(gatedInput, -1);
            case "Upsample":
                return engine.Upsample(
                    new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 1, 1, 2, 2 }), 2, 2);
            case "TensorSoftmaxBackward":
                return engine.TensorSoftmaxBackward(
                    new Tensor<float>(new[] { 0.1f, 0.2f, 0.3f, 0.4f }, new[] { 1, 4 }),
                    new Tensor<float>(new[] { 1f, -1f, 0.5f, -0.5f }, new[] { 1, 4 }), -1);
            default:
                throw new ArgumentOutOfRangeException(nameof(operation), operation, null);
        }
    }

    private sealed class MockGpuEngineScope : IDisposable
    {
        private readonly IEngine _priorEngine;
        private readonly string _priorBackendEnv;
        public MockBackendState State { get; } = new();
        public DirectGpuTensorEngine Engine { get; }

        public MockGpuEngineScope()
        {
            _priorEngine = AiDotNetEngine.Current;
            _priorBackendEnv = Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS");

            DirectGpuEngine directGpu;
            try
            {
                Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "none");
                directGpu = new DirectGpuEngine();
            }
            finally
            {
                Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", _priorBackendEnv);
            }

            IDirectGpuBackend backend = MockDirectGpuBackend.Create(State);
            SetField(directGpu, "_backend", backend);
            SetField(directGpu, "_isAvailable", true);

            Engine = new DirectGpuTensorEngine(directGpu);
            AiDotNetEngine.Current = Engine;
        }

        public void Dispose()
        {
            AiDotNetEngine.Current = _priorEngine;
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", _priorBackendEnv);
            Engine.Dispose();
        }

        private static void SetField(object target, string name, object value)
        {
            FieldInfo field = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException($"Field not found: {name}");
            field.SetValue(target, value);
        }
    }
}
