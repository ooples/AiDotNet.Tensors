// Copyright (c) AiDotNet. All rights reserved.

#if !NETFRAMEWORK
#nullable disable

using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class GpuOptimizerResidencyMockTests
{
    [Fact]
    public void DenseOptimizerWrappers_MarkEveryMutatedTensorCurrent()
    {
        using var ctx = new MockGpuEngineScope();

        RunDenseCase(ctx, "AdamUpdate",
            t => GpuOptimizer.TryAdamStep(t.P, t.G, t.M, t.V, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.V);
        RunDenseCase(ctx, "AdamUpdate",
            t => GpuOptimizer.TryAdamWStep(t.P, t.G, t.M, t.V, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1),
            t => t.P, t => t.M, t => t.V);
        RunDenseCase(ctx, "SgdUpdate",
            t => GpuOptimizer.TrySgdStep(t.P, t.G, 0.01f),
            t => t.P);
        RunDenseCase(ctx, "SgdMomentumUpdate",
            t => GpuOptimizer.TrySgdMomentumStep(t.P, t.G, t.Velocity, 0.01f, 0.9f, 0f),
            t => t.P, t => t.Velocity);
        RunDenseCase(ctx, "RmspropUpdate",
            t => GpuOptimizer.TryRmspropStep(t.P, t.G, t.SquaredAvg, 0.01f, 0.99f, 1e-8f, 0f),
            t => t.P, t => t.SquaredAvg);
        RunDenseCase(ctx, "AdagradUpdate",
            t => GpuOptimizer.TryAdagradStep(t.P, t.G, t.Accum, 0.01f, 1e-8f, 0f),
            t => t.P, t => t.Accum);
        RunDenseCase(ctx, "NagUpdate",
            t => GpuOptimizer.TryNagStep(t.P, t.G, t.Velocity, 0.01f, 0.9f, 0f),
            t => t.P, t => t.Velocity);
        RunDenseCase(ctx, "LarsUpdate",
            t => GpuOptimizer.TryLarsStep(t.P, t.G, t.Velocity, 0.01f, 0.9f, 0f, 0.001f),
            t => t.P, t => t.Velocity);
        RunDenseCase(ctx, "LambUpdate",
            t => GpuOptimizer.TryLambStep(t.P, t.G, t.M, t.V, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.V);
        RunDenseCase(ctx, "AdadeltaUpdate",
            t => GpuOptimizer.TryAdadeltaStep(t.P, t.G, t.AccumGrad, t.AccumUpdate, 0.95f, 1e-6f, 0f),
            t => t.P, t => t.AccumGrad, t => t.AccumUpdate);
        RunDenseCase(ctx, "AmsgradUpdate",
            t => GpuOptimizer.TryAmsgradStep(t.P, t.G, t.M, t.V, t.VMax, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.V, t => t.VMax);
        RunDenseCase(ctx, "AdamaxUpdate",
            t => GpuOptimizer.TryAdamaxStep(t.P, t.G, t.M, t.U, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.U);
        RunDenseCase(ctx, "LionUpdate",
            t => GpuOptimizer.TryLionStep(t.P, t.G, t.M, 0.01f, 0.9f, 0.99f, 0f),
            t => t.P, t => t.M);
        RunDenseCase(ctx, "NadamUpdate",
            t => GpuOptimizer.TryNadamStep(t.P, t.G, t.M, t.V, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.V);
        RunDenseCase(ctx, "FtrlUpdate",
            t => GpuOptimizer.TryFtrlStep(t.P, t.G, t.Z, t.N, 0.01f, 0f, 0f, 0f),
            t => t.P, t => t.Z, t => t.N);
    }

    [Fact]
    public void SparseOptimizerWrappers_MarkEveryMutatedTensorCurrent()
    {
        using var ctx = new MockGpuEngineScope();

        RunSparseCase(ctx, "SparseAdamUpdate",
            t => GpuOptimizer.TryAdamStepSparse(t.P, t.M, t.V, t.Indices, t.Values, 2, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.V);
        RunSparseCase(ctx, "SparseAdamWUpdate",
            t => GpuOptimizer.TryAdamWStepSparse(t.P, t.M, t.V, t.Indices, t.Values, 2, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1),
            t => t.P, t => t.M, t => t.V);
        RunSparseCase(ctx, "SparseSgdUpdate",
            t => GpuOptimizer.TrySgdStepSparse(t.P, t.Indices, t.Values, 2, 0.01f),
            t => t.P);
        RunSparseCase(ctx, "SparseSgdMomentumUpdate",
            t => GpuOptimizer.TrySgdMomentumStepSparse(t.P, t.Velocity, t.Indices, t.Values, 2, 0.01f, 0.9f, 0f),
            t => t.P, t => t.Velocity);
        RunSparseCase(ctx, "SparseRmspropUpdate",
            t => GpuOptimizer.TryRmspropStepSparse(t.P, t.SquaredAvg, t.Indices, t.Values, 2, 0.01f, 0.99f, 1e-8f, 0f),
            t => t.P, t => t.SquaredAvg);
        RunSparseCase(ctx, "SparseAdagradUpdate",
            t => GpuOptimizer.TryAdagradStepSparse(t.P, t.Accum, t.Indices, t.Values, 2, 0.01f, 1e-8f, 0f),
            t => t.P, t => t.Accum);
        RunSparseCase(ctx, "SparseNagUpdate",
            t => GpuOptimizer.TryNagStepSparse(t.P, t.Velocity, t.Indices, t.Values, 2, 0.01f, 0.9f, 0f),
            t => t.P, t => t.Velocity);
        RunSparseCase(ctx, "SparseAdadeltaUpdate",
            t => GpuOptimizer.TryAdadeltaStepSparse(t.P, t.AccumGrad, t.AccumUpdate, t.Indices, t.Values, 2, 0.95f, 1e-6f, 0f),
            t => t.P, t => t.AccumGrad, t => t.AccumUpdate);
        RunSparseCase(ctx, "SparseAmsgradUpdate",
            t => GpuOptimizer.TryAmsgradStepSparse(t.P, t.M, t.V, t.VMax, t.Indices, t.Values, 2, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.V, t => t.VMax);
        RunSparseCase(ctx, "SparseAdamaxUpdate",
            t => GpuOptimizer.TryAdamaxStepSparse(t.P, t.M, t.U, t.Indices, t.Values, 2, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.U);
        RunSparseCase(ctx, "SparseLionUpdate",
            t => GpuOptimizer.TryLionStepSparse(t.P, t.M, t.Indices, t.Values, 2, 0.01f, 0.9f, 0.99f, 0f),
            t => t.P, t => t.M);
        RunSparseCase(ctx, "SparseNadamUpdate",
            t => GpuOptimizer.TryNadamStepSparse(t.P, t.M, t.V, t.Indices, t.Values, 2, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.V);
        RunSparseCase(ctx, "SparseFtrlUpdate",
            t => GpuOptimizer.TryFtrlStepSparse(t.P, t.Z, t.N, t.Indices, t.Values, 2, 0.01f, 0f, 0f, 0f),
            t => t.P, t => t.Z, t => t.N);
        RunSparseCase(ctx, "SparseProximalL1Update",
            t => GpuOptimizer.TryProximalL1StepSparse(t.P, t.Indices, t.Values, 2, 0.01f, 0.1f),
            t => t.P);
    }

    private static void RunDenseCase(
        MockGpuEngineScope ctx,
        string expectedBackendCall,
        Func<DenseTensors, bool> run,
        params Func<DenseTensors, Tensor<float>>[] updatedSelectors)
    {
        var tensors = new DenseTensors(ctx.Backend);
        RunCase(ctx, expectedBackendCall, () => run(tensors), tensors.G, tensors.All, Select(tensors, updatedSelectors));
    }

    private static void RunSparseCase(
        MockGpuEngineScope ctx,
        string expectedBackendCall,
        Func<SparseTensors, bool> run,
        params Func<SparseTensors, Tensor<float>>[] updatedSelectors)
    {
        var tensors = new SparseTensors(ctx.Backend);
        RunCase(ctx, expectedBackendCall, () => run(tensors), tensors.Values, tensors.All, Select(tensors, updatedSelectors));
    }

    private static void RunCase(
        MockGpuEngineScope ctx,
        string expectedBackendCall,
        Func<bool> run,
        Tensor<float> gradientTensor,
        IReadOnlyList<Tensor<float>> allTensors,
        IReadOnlyList<Tensor<float>> expectedUpdated)
    {
        ctx.State.OptimizerCalls.Clear();
        var before = new Dictionary<Tensor<float>, int>();
        foreach (var tensor in allTensors)
            before[tensor] = tensor.Version;

        Assert.True(run(), $"GpuOptimizer wrapper for {expectedBackendCall} should run against the mock GPU backend.");
        Assert.Contains(expectedBackendCall, ctx.State.OptimizerCalls);

        var updatedSet = new HashSet<Tensor<float>>(expectedUpdated);
        foreach (var tensor in expectedUpdated)
            AssertGpuMarkedCurrent(tensor, before[tensor], expectedBackendCall);

        Assert.DoesNotContain(gradientTensor, updatedSet);
        Assert.Equal(before[gradientTensor], gradientTensor.Version);
    }

    private static Tensor<float>[] Select<TState>(TState state, Func<TState, Tensor<float>>[] selectors)
    {
        var tensors = new Tensor<float>[selectors.Length];
        for (int i = 0; i < selectors.Length; i++)
            tensors[i] = selectors[i](state);
        return tensors;
    }

    private static void AssertGpuMarkedCurrent(Tensor<float> tensor, int oldVersion, string caseName)
    {
        Assert.True(tensor.Version > oldVersion, $"{caseName}: tensor Version should advance.");
        Assert.Equal(tensor.Version, tensor._gpuBufferVersion);
        Assert.NotNull(tensor.LastWriteSync);
        Assert.True(tensor.LastWriteSync!.IsComplete, $"{caseName}: mock backend write sync should be complete.");
    }

    private static Tensor<float> GpuFloat(IDirectGpuBackend backend, int length = 4)
        => Tensor<float>.FromGpuBuffer(backend, new MockGpuBuffer(new float[length]), new[] { length });

    private static Tensor<int> GpuInt(IDirectGpuBackend backend, int length = 4)
        => Tensor<int>.FromGpuBuffer(backend, new MockGpuBuffer(new float[length]), new[] { length });

    private sealed class DenseTensors
    {
        public Tensor<float> P { get; }
        public Tensor<float> G { get; }
        public Tensor<float> M { get; }
        public Tensor<float> V { get; }
        public Tensor<float> U { get; }
        public Tensor<float> Z { get; }
        public Tensor<float> N { get; }
        public Tensor<float> Velocity { get; }
        public Tensor<float> Accum { get; }
        public Tensor<float> SquaredAvg { get; }
        public Tensor<float> AccumGrad { get; }
        public Tensor<float> AccumUpdate { get; }
        public Tensor<float> VMax { get; }
        public Tensor<float>[] All { get; }

        public DenseTensors(IDirectGpuBackend backend)
        {
            P = GpuFloat(backend);
            G = GpuFloat(backend);
            M = GpuFloat(backend);
            V = GpuFloat(backend);
            U = GpuFloat(backend);
            Z = GpuFloat(backend);
            N = GpuFloat(backend);
            Velocity = GpuFloat(backend);
            Accum = GpuFloat(backend);
            SquaredAvg = GpuFloat(backend);
            AccumGrad = GpuFloat(backend);
            AccumUpdate = GpuFloat(backend);
            VMax = GpuFloat(backend);
            All = new[] { P, G, M, V, U, Z, N, Velocity, Accum, SquaredAvg, AccumGrad, AccumUpdate, VMax };
        }
    }

    private sealed class SparseTensors
    {
        public Tensor<float> P { get; }
        public Tensor<float> M { get; }
        public Tensor<float> V { get; }
        public Tensor<float> U { get; }
        public Tensor<float> Z { get; }
        public Tensor<float> N { get; }
        public Tensor<float> Velocity { get; }
        public Tensor<float> Accum { get; }
        public Tensor<float> SquaredAvg { get; }
        public Tensor<float> AccumGrad { get; }
        public Tensor<float> AccumUpdate { get; }
        public Tensor<float> VMax { get; }
        public Tensor<int> Indices { get; }
        public Tensor<float> Values { get; }
        public Tensor<float>[] All { get; }

        public SparseTensors(IDirectGpuBackend backend)
        {
            P = GpuFloat(backend);
            M = GpuFloat(backend);
            V = GpuFloat(backend);
            U = GpuFloat(backend);
            Z = GpuFloat(backend);
            N = GpuFloat(backend);
            Velocity = GpuFloat(backend);
            Accum = GpuFloat(backend);
            SquaredAvg = GpuFloat(backend);
            AccumGrad = GpuFloat(backend);
            AccumUpdate = GpuFloat(backend);
            VMax = GpuFloat(backend);
            Indices = GpuInt(backend);
            Values = GpuFloat(backend);
            All = new[] { P, M, V, U, Z, N, Velocity, Accum, SquaredAvg, AccumGrad, AccumUpdate, VMax, Values };
        }
    }

    private sealed class MockGpuEngineScope : IDisposable
    {
        private readonly IEngine _priorEngine;
        private readonly string? _priorBackendEnv;
        public MockBackendState State { get; } = new();
        public IDirectGpuBackend Backend { get; }
        public DirectGpuTensorEngine Engine { get; }

        public MockGpuEngineScope()
        {
            _priorEngine = AiDotNetEngine.Current;
            _priorBackendEnv = Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS");
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "mock-unavailable");
            var directGpu = new DirectGpuEngine();
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", _priorBackendEnv);

            Backend = MockDirectGpuBackend.Create(State);
            SetField(directGpu, "_backend", Backend);
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

        private static void SetField(object target, string name, object? value)
        {
            var field = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException($"Field not found: {name}");
            field.SetValue(target, value);
        }
    }
}
#endif
