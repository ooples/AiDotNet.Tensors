// Copyright (c) AiDotNet. All rights reserved.

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

[Collection("EngineCurrentGlobalState")]
public sealed class GpuOptimizerResidencyMockTests
{
    [Fact]
    public void DenseOptimizerWrappers_MarkEveryMutatedTensorCurrent()
    {
        using var ctx = new MockGpuEngineScope();

        RunDenseCase(ctx, "AdamUpdate",
            t => GpuOptimizer.TryAdamStep(t.P, t.G, t.M, t.V, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
            t => t.P, t => t.M, t => t.V);
        RunDenseCase(ctx, "AdamWUpdate",
            t => GpuOptimizer.TryAdamWStep(t.P, t.G, t.M, t.V, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1),
            t => t.P, t => t.M, t => t.V);
        RunDenseCase(ctx, "SgdUpdate",
            t => GpuOptimizer.TrySgdStep(t.P, t.G, 0.01f),
            t => t.P);
        RunDenseCase(ctx, "ProximalL1Update",
            t => GpuOptimizer.TryProximalL1Step(t.P, t.G, 0.01f, 0.1f),
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

    [Fact]
    public void SparseDispatchValidation_RejectsDuplicateIndicesBeforeDispatch()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var indices = new MockGpuBuffer(new[] { 1f, 1f });

        WithSparseIndexValidation(enabled: true, () =>
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                GpuOptimizer.EnsureUniqueSparseDispatchIndices(backend, indices, 2, 4));

            Assert.Contains("duplicate index 1", ex.Message);
            Assert.Empty(state.OptimizerCalls);
        });
    }

    [Fact]
    public void SparseDispatchValidation_RejectsOutOfRangeIndicesBeforeDispatch()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var indices = new MockGpuBuffer(new[] { 0f, 4f });

        WithSparseIndexValidation(enabled: true, () =>
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                GpuOptimizer.EnsureUniqueSparseDispatchIndices(backend, indices, 2, 4));
            Assert.Empty(state.OptimizerCalls);
        });
    }

    [Fact]
    public void SparseDispatchValidation_SkipsHostDownloadWhenDisabled()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var duplicateIndices = new MockGpuBuffer(new[] { 1f, 1f });

        WithSparseIndexValidation(enabled: false, () =>
            GpuOptimizer.EnsureUniqueSparseDispatchIndices(backend, duplicateIndices, 2, 4));

        Assert.Equal(0, state.DownloadBufferCalls);
        Assert.Empty(state.OptimizerCalls);
    }

    [Fact]
    public void SparseDispatchValidation_AllowsUniqueIndicesWhenEnabled()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var indices = new MockGpuBuffer(new[] { 0f, 2f });

        WithSparseIndexValidation(enabled: true, () =>
            GpuOptimizer.EnsureUniqueSparseDispatchIndices(backend, indices, 2, 4));

        Assert.Equal(1, state.DownloadBufferCalls);
        Assert.Empty(state.OptimizerCalls);
    }

    [Fact]
    public void SparseDispatchValidation_AllowsZeroNnzWithoutHostDownload()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var indices = new MockGpuBuffer(Array.Empty<float>());

        WithSparseIndexValidation(enabled: true, () =>
            GpuOptimizer.EnsureUniqueSparseDispatchIndices(backend, indices, 0, 4));

        Assert.Equal(0, state.DownloadBufferCalls);
        Assert.Empty(state.OptimizerCalls);
    }

    [Fact]
    public void SparseDispatchValidation_RejectsShortIndexBufferWithoutHostDownload()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var indices = new MockGpuBuffer(new[] { 0f });

        WithSparseIndexValidation(enabled: false, () =>
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                GpuOptimizer.EnsureUniqueSparseDispatchIndices(backend, indices, 2, 4));

            Assert.Contains("sparseIndices buffer holds 1 elements but nnz=2", ex.Message);
        });

        Assert.Equal(0, state.DownloadBufferCalls);
        Assert.Empty(state.OptimizerCalls);
    }

    [Fact]
    public void SparsePrepare_RejectsShortDeviceBuffersBeforeDispatch()
    {
        using var ctx = new MockGpuEngineScope();
        var param = GpuFloat(ctx.Backend, 4);
        var values = GpuFloat(ctx.Backend, 2);
        var shortIndices = Tensor<int>.FromGpuBuffer(ctx.Backend, new MockGpuBuffer(new[] { 0f }), new[] { 2 });

        var indexEx = Assert.Throws<ArgumentException>(() =>
            GpuOptimizer.TrySgdStepSparse(param, shortIndices, values, 2, 0.01f));
        Assert.Contains("sparseIndices buffer holds 1 elements but nnz=2", indexEx.Message);
        Assert.Empty(ctx.State.OptimizerCalls);

        var indices = GpuInt(ctx.Backend, 2);
        var shortValues = Tensor<float>.FromGpuBuffer(ctx.Backend, new MockGpuBuffer(new[] { 1f }), new[] { 2 });

        var valueEx = Assert.Throws<ArgumentException>(() =>
            GpuOptimizer.TrySgdStepSparse(param, indices, shortValues, 2, 0.01f));
        Assert.Contains("sparseValues buffer holds 1 elements but nnz=2", valueEx.Message);
        Assert.Empty(ctx.State.OptimizerCalls);
    }

    [Fact]
    public void DenseProximalL1_RejectsInvalidInputsBeforeDispatch()
    {
        using var ctx = new MockGpuEngineScope();
        var p = GpuFloat(ctx.Backend, 4);
        var g = GpuFloat(ctx.Backend, 4);
        var shortGrad = GpuFloat(ctx.Backend, 3);

        Assert.Throws<ArgumentException>(() => GpuOptimizer.TryProximalL1Step(p, shortGrad, 0.01f, 0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuOptimizer.TryProximalL1Step(p, g, -0.01f, 0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuOptimizer.TryProximalL1Step(p, g, 0.01f, -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuOptimizer.TryProximalL1Step(p, g, float.NaN, 0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuOptimizer.TryProximalL1Step(p, g, 0.01f, float.PositiveInfinity));
        Assert.Empty(ctx.State.OptimizerCalls);
    }

    private static void RunDenseCase(
        MockGpuEngineScope ctx,
        string expectedBackendCall,
        Func<DenseTensors, bool> run,
        params Func<DenseTensors, Tensor<float>>[] updatedSelectors)
    {
        var tensors = new DenseTensors(ctx.Backend);
        RunCase(ctx, expectedBackendCall, () => run(tensors), tensors.All, Select(tensors, updatedSelectors));
    }

    private static void RunSparseCase(
        MockGpuEngineScope ctx,
        string expectedBackendCall,
        Func<SparseTensors, bool> run,
        params Func<SparseTensors, Tensor<float>>[] updatedSelectors)
    {
        var tensors = new SparseTensors(ctx.Backend);
        RunCase(ctx, expectedBackendCall, () => run(tensors), tensors.All, Select(tensors, updatedSelectors));
    }

    private static void RunCase(
        MockGpuEngineScope ctx,
        string expectedBackendCall,
        Func<bool> run,
        IReadOnlyList<TrackedTensor> allTensors,
        IReadOnlyList<Tensor<float>> expectedUpdated)
    {
        ctx.State.OptimizerCalls.Clear();
        var before = new TensorSnapshot[allTensors.Count];
        for (int i = 0; i < allTensors.Count; i++)
            before[i] = allTensors[i].Snapshot();

        Assert.True(run(), $"GpuOptimizer wrapper for {expectedBackendCall} should run against the mock GPU backend.");
        Assert.Contains(expectedBackendCall, ctx.State.OptimizerCalls);

        foreach (var tensor in expectedUpdated)
        {
            int trackedIndex = IndexOfTracked(allTensors, tensor);
            Assert.True(trackedIndex >= 0, $"{expectedBackendCall}: expected updated tensor was not tracked.");
            AssertGpuMarkedCurrent(tensor, before[trackedIndex].Version, expectedBackendCall);
        }

        for (int i = 0; i < allTensors.Count; i++)
        {
            if (ContainsReference(expectedUpdated, allTensors[i].Instance)) continue;
            Assert.Equal(before[i].Version, allTensors[i].Version);
            Assert.Equal(before[i].GpuBufferVersion, allTensors[i].GpuBufferVersion);
            Assert.Same(before[i].LastWriteSync, allTensors[i].LastWriteSync);
        }
    }

    private static int IndexOfTracked(IReadOnlyList<TrackedTensor> tensors, object expected)
    {
        for (int i = 0; i < tensors.Count; i++)
        {
            if (ReferenceEquals(tensors[i].Instance, expected))
                return i;
        }

        return -1;
    }

    private static bool ContainsReference(IReadOnlyList<Tensor<float>> tensors, object expected)
        => IndexOfTracked(Select(tensors), expected) >= 0;

    private static TrackedTensor[] Select(IReadOnlyList<Tensor<float>> tensors)
    {
        var tracked = new TrackedTensor[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
            tracked[i] = Track(tensors[i]);
        return tracked;
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

    private static TrackedTensor Track<T>(Tensor<T> tensor) => new TrackedTensor(
        tensor,
        () => tensor.Version,
        () => tensor._gpuBufferVersion,
        () => tensor.LastWriteSync);

    private sealed class TrackedTensor
    {
        private readonly Func<int> _version;
        private readonly Func<int> _gpuBufferVersion;
        private readonly Func<object> _lastWriteSync;

        public TrackedTensor(object instance, Func<int> version, Func<int> gpuBufferVersion, Func<object> lastWriteSync)
        {
            Instance = instance;
            _version = version;
            _gpuBufferVersion = gpuBufferVersion;
            _lastWriteSync = lastWriteSync;
        }

        public object Instance { get; }
        public int Version => _version();
        public int GpuBufferVersion => _gpuBufferVersion();
        public object LastWriteSync => _lastWriteSync();
        public TensorSnapshot Snapshot() => new TensorSnapshot(Version, GpuBufferVersion, LastWriteSync);
    }

    private readonly struct TensorSnapshot
    {
        public TensorSnapshot(int version, int gpuBufferVersion, object lastWriteSync)
        {
            Version = version;
            GpuBufferVersion = gpuBufferVersion;
            LastWriteSync = lastWriteSync;
        }

        public int Version { get; }
        public int GpuBufferVersion { get; }
        public object LastWriteSync { get; }
    }

    private static Tensor<float> GpuFloat(IDirectGpuBackend backend, int length = 4)
        => Tensor<float>.FromGpuBuffer(backend, new MockGpuBuffer(new float[length]), new[] { length });

    private static Tensor<int> GpuInt(IDirectGpuBackend backend, int length = 4)
    {
        var data = new float[length];
        for (int i = 0; i < data.Length; i++)
            data[i] = i;
        return Tensor<int>.FromGpuBuffer(backend, new MockGpuBuffer(data), new[] { length });
    }

    private static void WithSparseIndexValidation(bool enabled, Action action)
    {
        bool prior = GpuOptimizer.ValidateSparseIndexUniqueness;
        try
        {
            GpuOptimizer.ValidateSparseIndexUniqueness = enabled;
            action();
        }
        finally
        {
            GpuOptimizer.ValidateSparseIndexUniqueness = prior;
        }
    }

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
        public TrackedTensor[] All { get; }

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
            All = new[] { Track(P), Track(G), Track(M), Track(V), Track(U), Track(Z), Track(N), Track(Velocity), Track(Accum), Track(SquaredAvg), Track(AccumGrad), Track(AccumUpdate), Track(VMax) };
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
        public TrackedTensor[] All { get; }

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
            All = new[] { Track(P), Track(M), Track(V), Track(U), Track(Z), Track(N), Track(Velocity), Track(Accum), Track(SquaredAvg), Track(AccumGrad), Track(AccumUpdate), Track(VMax), Track(Indices), Track(Values) };
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
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "none");
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
