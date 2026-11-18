using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class PrivateGradientInferenceDemandTests
{
    [Fact]
    public void Cache_invalidation_marks_weight_version_without_exposing_private_host_payload()
    {
        var cpu = new CpuEngine();
        var array = new[] { 1f, 2f, 3f, 4f };
        var owner = ResultBufferTracker.TrackFreshPrivateArray(array, 0);
        using var tensor = Tensor<float>.FromMemory(array, new[] { 2, 2 });
        using var peer = Assert.IsType<Tensor<float>>(tensor.CloneShared());
        object storage = tensor.StorageIdentity;
        long version = AiDotNet.Tensors.Engines.Simd.SimdGemm.WeightArrayVersionOf(array);
        int downloads = 0;
        HostSync.Register(array, value => { downloads++; Array.Fill((float[])value, 17f); });
        try
        {
            cpu.InvalidatePersistentTensor(tensor);
            cpu.InvalidatePersistentTensor(peer);
            Assert.Equal(version + 2, AiDotNet.Tensors.Engines.Simd.SimdGemm.WeightArrayVersionOf(array));
            Assert.Same(storage, tensor.StorageIdentity);
            Assert.Same(storage, peer.StorageIdentity);
            Assert.False(owner.State.Escaped);
            Assert.Equal(0, downloads);
            Assert.True(HostSync.IsPending(array));
            Assert.Equal(new[] { 1f, 2f, 3f, 4f }, array);
            // Genuine caller raw access still materializes the current payload and permanently protects its owner.
            Assert.Equal(new[] { 17f, 17f, 17f, 17f }, tensor.AsSpan().ToArray());
            Assert.Equal(1, downloads);
            Assert.True(owner.State.Escaped);
            Assert.Equal(new[] { 17f, 17f, 17f, 17f }, array);
            Assert.Same(storage, peer.StorageIdentity);
        }
        finally { HostSync.Remove(array); }
        GC.KeepAlive(owner);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Native_return_protects_unread_public_array_alias_even_with_a_different_native_root(bool sameHostArray, bool rawPrivateArrayEscaped)
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var plan = Compile(engine, out _, out _);
        PrepareCompletedMetadata(plan, engine);
        var publicGradient = plan.Gradients[0];
        var gradients = (Tensor<float>[])Field(plan, "_preAllocatedGrads")!;
        var privateGradient = gradients.First(gradient => gradient.Length == 4 && !ReferenceEquals(gradient, publicGradient));
        var publicArray = new float[4];
        var privateArray = sameHostArray ? publicArray : new float[4];
        var publicOwner = ResultBufferTracker.TrackFreshPrivateArray(publicArray, 0);
        var privateOwner = sameHostArray ? publicOwner : ResultBufferTracker.TrackFreshPrivateArray(privateArray, 0);
        using (var owner = Tensor<float>.FromMemory(publicArray, new[] { 2, 2 })) publicGradient.RebindStorageFrom(owner);
        using (var owner = Tensor<float>.FromMemory(privateArray, new[] { 2, 2 })) privateGradient.RebindStorageFrom(owner);
        Assert.Same(publicArray, publicGradient.GraphBackingArrayIdentity);
        Assert.Same(privateArray, privateGradient.GraphBackingArrayIdentity);
        var escaped = rawPrivateArrayEscaped ? privateGradient.GetDataArray() : null;
        if (rawPrivateArrayEscaped) { Assert.Same(privateArray, escaped); Assert.True(privateOwner.State.Escaped); }
        var publicBuffer = new NativeCell(4);
        var privateBuffer = new NativeCell(4);
        Bind(publicGradient, publicBuffer, backend);
        Bind(privateGradient, privateBuffer, backend);
        plan.RecordPrivateGradientAllocation(privateBuffer);
        int downloads = 0;
        HostSync.Register(publicArray, value => { downloads++; Array.Fill((float[])value, 37f); });
        using var publicView = publicGradient.Reshape(4);
        Assert.NotSame(publicGradient.StorageIdentity, privateGradient.StorageIdentity);
        Assert.NotSame(publicBuffer, privateBuffer);
        plan.RegisterPrivateGradientDemandOwner(engine, backend);
        using (InferenceWorkspaceDemandScope.Enter(engine))
            InferenceWorkspaceDemandScope.BeforeAllocation(backend, 5);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Assert.Equal(sameHostArray || rawPrivateArrayEscaped ? 0 : 1, privateBuffer.Releases);
        Assert.Equal(0, publicBuffer.Releases);
        Assert.Equal(0, publicBuffer.Disposes);
        Assert.Equal(0, downloads);
        Assert.True(HostSync.IsPending(publicArray));
        Assert.Equal(new[] { 37f, 37f, 37f, 37f }, publicView.ToArray());
        Assert.Equal(1, downloads);
        Assert.Equal(sameHostArray || rawPrivateArrayEscaped ? 0L : 16L, plan.PrivateGradientNativeReleaseRequestedBytes);
        GC.KeepAlive(escaped);
        GC.KeepAlive(publicOwner);
        GC.KeepAlive(privateOwner);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_host_wrapper_outside_the_plan_protects_the_global_download(bool registerFreshOwnership)
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var plan = Compile(engine, out _, out _);
        PrepareCompletedMetadata(plan, engine);
        var publicGradient = plan.Gradients[0];
        var gradients = (Tensor<float>[])Field(plan, "_preAllocatedGrads")!;
        var privateGradient = gradients.First(gradient => gradient.Length == 4 && !ReferenceEquals(gradient, publicGradient));
        var array = new float[4];
        var owner = registerFreshOwnership ? ResultBufferTracker.TrackFreshPrivateArray(array, 0) : null;
        using (var initial = Tensor<float>.FromMemory(array, new[] { 2, 2 })) privateGradient.RebindStorageFrom(initial);
        // Deliberately absent from the plan's known tensor census. It has its own Vector and TensorStorage.
        using var foreignPublic = Tensor<float>.FromMemory(array, new[] { 2, 2 });
        Assert.NotSame(foreignPublic.StorageIdentity, privateGradient.StorageIdentity);
        Assert.NotSame(foreignPublic.GraphVectorIdentity, privateGradient.GraphVectorIdentity);
        var privateBuffer = new NativeCell(4);
        Bind(privateGradient, privateBuffer, backend);
        plan.RecordPrivateGradientAllocation(privateBuffer);
        int downloads = 0;
        HostSync.Register(array, value => { downloads++; Array.Fill((float[])value, 53f); });
        plan.RegisterPrivateGradientDemandOwner(engine, backend);
        using (InferenceWorkspaceDemandScope.Enter(engine))
            InferenceWorkspaceDemandScope.BeforeAllocation(backend, 5);
        GC.Collect(); GC.WaitForPendingFinalizers();
        Assert.Equal(0, privateBuffer.Releases);
        Assert.Equal(0, downloads);
        Assert.True(HostSync.IsPending(array));
        Assert.Equal(new[] { 53f, 53f, 53f, 53f }, foreignPublic.ToArray());
        Assert.Equal(1, downloads);
        Assert.Equal(0, plan.PrivateGradientNativeReleaseRequestedBytes);
        GC.KeepAlive(owner);
    }

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Cuda_TrainPredictTrain_preserves_unread_results_and_Adam_after_private_native_release(bool inputGradients, bool forceFallback)
    {
        using var gpu = CreateGpu();
        var previous = AiDotNetEngine.Current;
        bool? priorInputGradients = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        bool priorTraceEscapes = ResultBufferTracker.TraceEscapes;
        string? priorPolicy = Environment.GetEnvironmentVariable("AIDOTNET_INFERENCE_PRIVATE_GRADIENT_RETURN");
        try
        {
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = inputGradients;
            var baseline = Trajectory(gpu, false, inputGradients, forceFallback, out long baselineReleased, out _);
            if (inputGradients) ResultBufferTracker.TraceEscapes = true;
            var retired = Trajectory(gpu, true, inputGradients, forceFallback, out long released, out string evidence);
            Assert.Equal(0, baselineReleased);
            Assert.True(released > 0, "The fixture must actually retire private accumulator native buffers.\n" + evidence);
            Assert.Equal(baseline.Count, retired.Count);
            for (int sample = 0; sample < baseline.Count; sample++)
                Assert.Equal(baseline[sample].Select(BitConverter.SingleToInt32Bits),
                    retired[sample].Select(BitConverter.SingleToInt32Bits));
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_INFERENCE_PRIVATE_GRADIENT_RETURN", priorPolicy);
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = priorInputGradients;
            ResultBufferTracker.TraceEscapes = priorTraceEscapes;
            AiDotNetEngine.Current = previous;
        }
    }

    private static List<float[]> Trajectory(DirectGpuTensorEngine engine, bool release, bool inputGradients, bool forceFallback, out long released, out string evidence)
    {
        AiDotNetEngine.Current = engine;
        Environment.SetEnvironmentVariable("AIDOTNET_INFERENCE_PRIVATE_GRADIENT_RETURN", release ? null : "0");
        using var plan = Compile(engine, out var input, out var weight);
        plan.DisableGraphStep();
        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: .001);
        var samples = new List<float[]>();
        var eligibility = new List<string>();
        for (int iteration = 0; iteration < 5; iteration++)
        {
            input[0] = .1f + iteration * .01f;
            if (forceFallback && iteration == 2) plan.FailPrivateGradientResidencyAtForTesting = 1;
            var loss = plan.Step();
            if (forceFallback && iteration >= 2) Assert.Equal(true, Field(plan, "_residentStepFailed"));
            var gradient = plan.Gradients[0];
            using var gradientView = gradient.Reshape(4);
            var inputGradient = input.Grad;
            var publicBuffer = gradient._gpuBuffer;
            using (new NoGradScope<float>())
            using (InferenceWorkspaceDemandScope.Enter(engine))
            using (var prediction = engine.TensorAdd(new Tensor<float>(new[] { 4096 }), new Tensor<float>(new[] { 4096 })))
                Assert.Equal(4096, prediction.Length);
            if (release && inputGradients && iteration < 2) eligibility.Add(DescribeRetirementEligibility(plan, iteration));
            GC.Collect();
            GC.WaitForPendingFinalizers();
            Assert.Same(publicBuffer, gradient._gpuBuffer);
            if (publicBuffer is not null) Assert.NotEqual(IntPtr.Zero, gradient._gpuBuffer!.Handle);
            samples.Add(loss.ToArray());
            samples.Add(gradientView.ToArray());
            samples.Add(weight.ToArray());
            if (inputGradients) samples.Add(Assert.IsType<Tensor<float>>(inputGradient).ToArray());
        }
        released = plan.PrivateGradientNativeReleaseRequestedBytes;
        evidence = string.Join("\n", eligibility);
        return samples;
    }

    private static string DescribeRetirementEligibility(CompiledTrainingPlan<float> plan, int iteration)
    {
        var facts = new List<string> { "iteration=" + iteration };
        facts.Add("firstEscapeSites=" + string.Join(",", ResultBufferTracker.TopEscapeSites(12).Select(site => site.Key + ":" + site.Value)));
        foreach (string name in new[] { "_privateGradientCompletedStep", "_privateGradientRegistered", "_residentBodyRanLast", "_residentStepFailed", "_graphStepEligible", "_workspaceExplicitGraphDisable", "_graphStepDisabled", "_graphPreparationStarted" })
            facts.Add(name + "=" + Field(plan, name));
        var tensors = Assert.IsType<HashSet<Tensor<float>>>(Field(plan, "_workspaceTensors"));
        var protectedRoots = Assert.IsType<HashSet<IGpuBuffer>>(Field(plan, "_workspaceProtectedRoots"));
        var protectedStorage = Assert.IsType<HashSet<object>>(Field(plan, "_workspaceProtectedStorage"));
        var protectedVectors = Assert.IsType<HashSet<object>>(Field(plan, "_privateProtectedVectors"));
        var protectedArrays = Assert.IsType<HashSet<object>>(Field(plan, "_privateProtectedArrays"));
        var allocations = Assert.IsType<System.Runtime.CompilerServices.ConditionalWeakTable<IGpuBuffer, object>>(Field(plan, "_privateGradientAllocations"));
        foreach (var gradient in Assert.IsType<Tensor<float>[]>(Field(plan, "_preAllocatedGrads")))
        {
            if (gradient._gpuBuffer is not { } buffer) continue;
            var root = GpuBufferViewRoot.Of(buffer);
            var vector = (Vector<float>)gradient.GraphVectorIdentity;
            object? array = gradient.GraphBackingArrayIdentity;
            int knownVectors = tensors.Where(t => ReferenceEquals(t.GraphBackingArrayIdentity, array)).Select(t => t.GraphVectorIdentity).Distinct().Count();
            facts.Add($"gradient[{gradient.Length}] fresh={allocations.TryGetValue(root, out _)} protectedRoot={protectedRoots.Contains(root)} protectedStorage={protectedStorage.Contains(gradient.StorageIdentity)} protectedVector={protectedVectors.Contains(vector)} protectedArray={array is not null && protectedArrays.Contains(array)} wrappers={vector.ResultOwner?.State.Wrappers} knownVectors={knownVectors} escaped={vector.ResultOwner?.State.Escaped}");
        }
        return string.Join("; ", facts);
    }

    private static CompiledTrainingPlan<float> Compile(IEngine engine, out Tensor<float> input, out Tensor<float> weight)
    {
        input = new Tensor<float>(new[] { .1f, .2f, -.3f, .4f }, new[] { 2, 2 });
        weight = new Tensor<float>(new[] { .3f, -.2f, .1f, .4f }, new[] { 2, 2 });
        using var graph = GraphMode.EnableTraining(new[] { weight });
        var value = engine.TensorMatMul(input, weight);
        for (int layer = 0; layer < 4; layer++) value = engine.Tanh(engine.TensorMultiplyScalar(value, .9f));
        var loss = engine.ReduceMean(engine.TensorMultiply(value, value), new[] { 0, 1 }, false);
        return Assert.IsType<CompiledTrainingPlan<float>>(graph.CompileTraining(new[] { weight }, loss));
    }

    private static void PrepareCompletedMetadata(CompiledTrainingPlan<float> plan, DirectGpuTensorEngine engine)
    {
        Assert.True(plan.IsGraphStepEligible);
        plan.DisableGraphStep();
        typeof(CompiledTrainingPlan<float>).GetField("_residentBodyRanLast", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(plan, true);
        typeof(CompiledTrainingPlan<float>).GetField("_scratchPool", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(plan, engine.CreateScratchPool());
    }

    private static object? Field(object owner, string name) => owner.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(owner);

    private static void Bind(Tensor<float> tensor, IGpuBuffer buffer, IDirectGpuBackend backend)
    { tensor._gpuBuffer = buffer; tensor._gpuBackend = backend; tensor._gpuBufferVersion = tensor.GpuCacheVersion; }

    private static DirectGpuTensorEngine CreateGpu()
    {
        DirectGpuTensorEngine engine;
        try { engine = new DirectGpuTensorEngine(); }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or DllNotFoundException)
        { Skip.If(true, "CUDA unavailable: " + e.GetType().Name); throw; }
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "CUDA required.");
        return engine;
    }

    private sealed class NativeCell : IGpuBuffer, IPoolableGpuBuffer
    {
        private static int s_next;
        internal NativeCell(int size) { Size = size; Handle = (IntPtr)Interlocked.Increment(ref s_next); }
        public int Size { get; }
        public int Capacity => Size;
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle { get; private set; }
        internal int Releases { get; private set; }
        internal int Disposes { get; private set; }
        public void MarkRented(int size) => throw new InvalidOperationException();
        public void Release() { Releases++; Handle = IntPtr.Zero; }
        public void Dispose() { Disposes++; Handle = IntPtr.Zero; }
    }
}
