using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledOwnedLeafInputReuseTests
{
    private sealed class Policy : IDisposable
    {
        private readonly string? _previous = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_OWNED_LEAF_INPUT_REUSE");
        internal Policy(bool enabled = true) => Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_OWNED_LEAF_INPUT_REUSE", enabled ? "1" : "0");
        public void Dispose() => Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_OWNED_LEAF_INPUT_REUSE", _previous);
    }

    private static DirectGpuTensorEngine Cuda()
    {
        var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "Actual CUDA required.");
        return gpu;
    }

    private static Tensor<float> Values(float start)
    {
        var values = new float[16];
        for (int i = 0; i < values.Length; i++) values[i] = start + .01f * i;
        return new Tensor<float>(values, new[] { 4, 4 });
    }

    private static (CompiledTrainingPlan<float> Plan, Tensor<float> Input, Tensor<float> Weight) Compile(IEngine engine)
    {
        var input = Values(.1f); var weight = Values(.2f);
        CompiledTrainingPlan<float> plan;
        using (var graph = GraphMode.Enable())
        {
            var output = engine.TensorMultiply(input, weight);
            engine.ReduceSum(engine.TensorMultiply(output, output), null);
            plan = graph.CompileTraining(new[] { weight });
        }
        plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: .001f);
        plan.DisableGraphStep();
        return (plan, input, weight);
    }

    private static Dictionary<Tensor<float>, IGpuBuffer> Owned(CompiledTrainingPlan<float> plan)
        => (Dictionary<Tensor<float>, IGpuBuffer>)typeof(CompiledTrainingPlan<float>)
            .GetField("_ownedLeafBuffers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(plan)!;

    private static void Detach(Tensor<float> input)
    {
        // A host batch rebind can leave this metadata without its previous plan-owned binding.
        input._gpuBuffer = null; input._gpuBackend = null; input._gpuBufferVersion = -1;
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DetachedLeafReusesItsPlanAllocationAndTracksChangingBatchesLikeCpu(bool enabled)
    {
        using var policy = new Policy(enabled);
        using var gpu = Cuda();
        var actual = Compile(gpu); var expected = Compile(new CpuEngine());
        using (actual.Input) using (actual.Weight) using (expected.Input) using (expected.Weight)
        using (actual.Plan) using (expected.Plan)
        {
            actual.Plan.Step(); expected.Plan.Step();
            Assert.True(Owned(actual.Plan).TryGetValue(actual.Input, out var owned));
            Assert.NotEqual(IntPtr.Zero, owned!.Handle);
            for (int step = 0; step < 8; step++)
            {
                using var next = Values(.05f * (step + 1));
                actual.Plan.SetInput(next); expected.Plan.SetInput(next);
                Detach(actual.Input);
                actual.Plan.Step(); expected.Plan.Step();
                if (enabled) Assert.Same(owned, Owned(actual.Plan)[actual.Input]);
                else Assert.NotSame(owned, Owned(actual.Plan)[actual.Input]);
                var a = actual.Weight.ToArray(); var e = expected.Weight.ToArray();
                for (int i = 0; i < a.Length; i++)
                    Assert.True(Math.Abs(a[i] - e[i]) <= 2e-5f + Math.Abs(e[i]) * 1e-4f, $"step={step} index={i}: {a[i]} vs {e[i]}");
            }
        }
    }

    [SkippableFact]
    public void ReleasedPlanWrapperCannotOverwriteItsNewPoolTenant()
    {
        using var policy = new Policy();
        using var gpu = Cuda();
        var state = Compile(gpu);
        using (state.Input) using (state.Weight) using (state.Plan)
        {
            state.Plan.Step();
            var owned = Owned(state.Plan)[state.Input];
            IntPtr address = owned.Handle;
            owned.Dispose(); Detach(state.Input);
            using var tenant = gpu.GetBackend().AllocateBuffer(16);
            Assert.Equal(address, tenant.Handle);
            var sentinel = new float[16]; Array.Fill(sentinel, 7f);
            gpu.GetBackend().UploadBufferInPlace(sentinel, tenant);
            using var next = Values(.3f); state.Plan.SetInput(next); state.Plan.Step();
            Assert.NotSame(owned, Owned(state.Plan)[state.Input]);
            Assert.Equal(sentinel, gpu.GetBackend().DownloadBuffer(tenant));
        }
    }

    [SkippableFact]
    public void RefreshDeclinesForeignBackendAndExistingInputBindings()
    {
        using var gpu = Cuda(); using var foreign = new CudaBackend();
        Skip.IfNot(foreign.IsAvailable, "Second actual CUDA backend required.");
        using var input = Values(.1f);
        using var owned = gpu.GetBackend().AllocateBuffer(16);
        Assert.False(gpu.TryRefreshPlanOwnedLeafInput(input, foreign, owned));
        Assert.Null(input._gpuBuffer);
        gpu.EnsureResidentInput(input);
        var callerBuffer = input._gpuBuffer;
        Assert.False(gpu.TryRefreshPlanOwnedLeafInput(input, gpu.GetBackend(), owned));
        Assert.Same(callerBuffer, input._gpuBuffer);
        Assert.Equal(ValuesArray(.1f), input.ToArray());
        gpu.ReleaseDeadDeviceStorage(input, afterQueuedWork: true);
    }

    private static float[] ValuesArray(float start)
    {
        var values = new float[16];
        for (int i = 0; i < values.Length; i++) values[i] = start + .01f * i;
        return values;
    }
}
