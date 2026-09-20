using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Dropping a tensor's GPU buffer must also return it to <see cref="TensorDevice.CPU"/>.
///
/// <para><b>The invariant.</b> <c>Tensor.Gpu()</c> sets <c>_gpuBuffer</c>, <c>_gpuBackend</c> and
/// <c>_device</c> together, and <c>Tensor.Cpu()</c> is its inverse: it downloads only when
/// <c>_gpuBuffer is not null &amp;&amp; _gpuBackend is not null &amp;&amp; _device != CPU</c>, then sets
/// <c>_device = CPU</c> unconditionally. Device and buffer are meant to move as a pair.</para>
///
/// <para><b>The defect.</b> <see cref="DirectGpuTensorEngine.InvalidateResidentWeightBuffer{T}"/> clears
/// <c>_gpuBuffer</c>, <c>_gpuBackend</c> and <c>_gpuBufferVersion</c> but leaves <c>_device</c> alone. It is
/// called exactly when the HOST copy was just updated in place and is authoritative — its own summary says
/// "the just-written host weights authoritative — the next forward re-uploads them" — so the data is on the
/// host while every accessor asking "where does this live?" is answered CUDA.</para>
///
/// <para><b>This is reachable from ordinary training, not a synthetic state.</b>
/// <c>CompiledTapeTrainingStep</c> runs <c>foreach (var p in parameters) p.Gpu();</c> before fused training
/// whenever <c>AIDOTNET_GPU_RESIDENT_PARAMS != "0"</c> — which is the default — so every parameter of a
/// GPU-trained model is explicitly marked CUDA-resident, and the fused CPU-side optimizer then invalidates
/// those buffers in place.</para>
///
/// <para><b>Why it matters.</b> The resulting state is invisible to both GPU predicates in
/// <c>CompiledTrainingPlan</c> and rejected by the CPU one:</para>
/// <list type="bullet">
/// <item><description><c>ConfigureOptimizer</c> derives <c>hasGpuParams</c> from
/// <c>TryGetGpuBuffer() is not null</c> — now false, so it configures the CPU fused closures.</description></item>
/// <item><description>The per-parameter dispatch takes the GPU branch only when
/// <c>TryGetGpuBuffer() is not null &amp;&amp; _gpuBackend is not null</c> — also false, so the parameter
/// falls to the CPU branch.</description></item>
/// <item><description><c>BindCpuOptimizerTensor</c> then throws on <c>Device != TensorDevice.CPU</c>:
/// "A fused CPU optimizer cannot stage tensor shape [...] from device CUDA without an attached backend
/// buffer."</description></item>
/// </list>
///
/// <para>Nothing catches it at configure time, so it surfaces mid-training from a plan that has already
/// stepped successfully, which AiDotNet reports as the unrecoverable "Fused compiled training has already
/// run successfully, but the current step cannot engage the fused path". Observed in production as PPO
/// bake-off runs failing at <c>status=fit</c> with zero return and zero trades.</para>
/// </summary>
public sealed class ResidentWeightInvalidationDeviceStateTests
{
    private static Tensor<float> Weights(int rows, int cols, float scale)
    {
        var t = new Tensor<float>([rows, cols]);
        for (int i = 0; i < t.Length; i++)
        {
            t[i] = scale * (((i * 37) % 19) - 9);
        }

        return t;
    }

    /// <summary>
    /// Marks the tensor GPU-resident the same way <c>CompiledTapeTrainingStep</c> does, and skips if this
    /// box cannot actually reach that state — a silently-passing test here would assert nothing.
    /// </summary>
    private static Tensor<float> GpuResidentWeights(DirectGpuTensorEngine gpu, float scale)
    {
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        var weights = Weights(16, 8, scale);
        weights.Gpu();

        Skip.If(
            !weights.IsGpuResident || weights.TryGetGpuBuffer() is null,
            "Tensor did not become GPU-resident; the invalidation path under test is unreachable here.");

        // The precondition the production path establishes: device AND buffer both set.
        Assert.NotEqual(TensorDevice.CPU, weights.Device);
        Assert.NotNull(weights.TryGetGpuBuffer());

        return weights;
    }

    /// <summary>
    /// THE REGRESSION. After invalidation the tensor must be bindable by SOMETHING: either it still holds a
    /// GPU buffer, or it reports CPU residency. Claiming a GPU device with no buffer is the one state no
    /// optimizer-binding path accepts.
    /// </summary>
    [SkippableFact]
    public void InvalidatingAResidentWeightBufferLeavesTheTensorBindable()
    {
        using var gpu = new DirectGpuTensorEngine();
        var weights = GpuResidentWeights(gpu, 0.25f);

        gpu.InvalidateResidentWeightBuffer(weights);

        // The buffer is gone — that part already works and is the whole point of the call.
        Assert.Null(weights.TryGetGpuBuffer());

        // ...so the tensor must stop claiming to live on a GPU. This is what fails before the fix.
        Assert.Equal(TensorDevice.CPU, weights.Device);
        Assert.False(weights.IsGpuResident);
    }

    /// <summary>
    /// THE SAME REGRESSION, WITH NO GPU. The arm above is a <see cref="SkippableFactAttribute"/> gated on
    /// <c>IsGpuAvailable</c>, so on CI — which has no device — it skips, and the invariant it guards ships
    /// unverified; codecov measured 22% of this PR's diff hit for exactly that reason. Nothing in
    /// <c>InvalidateResidentWeightBuffer</c> touches hardware: it drops a deferred-download registration,
    /// clears the cache entries and nulls four fields. Only the PRECONDITION needed a device, and
    /// <c>Tensor.FromGpuBuffer</c> establishes it against the mock backend — <c>DeviceType</c> is
    /// <see cref="TensorDevice.OpenCL"/>, so the tensor reports a GPU device while holding a buffer,
    /// which is the state the production path starts from.
    /// </summary>
    [Fact]
    public void InvalidatingAMockResidentWeightBufferLeavesTheTensorBindable()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var buffer = new MockGpuBuffer(new float[16 * 8]);
        var weights = Tensor<float>.FromGpuBuffer(backend, buffer, new[] { 16, 8 });

        // The precondition, asserted rather than assumed: a silently-CPU tensor here would make the
        // rest of this test vacuous, which is how the hardware arm would have to be read on CI.
        Assert.NotEqual(TensorDevice.CPU, weights.Device);
        Assert.NotNull(weights.TryGetGpuBuffer());

        using var gpu = new DirectGpuTensorEngine();
        gpu.InvalidateResidentWeightBuffer(weights);

        Assert.Null(weights.TryGetGpuBuffer());
        Assert.Equal(TensorDevice.CPU, weights.Device);
        Assert.False(weights.IsGpuResident);
    }

    /// <summary>
    /// The same consequence stated in the terms the fused optimizer actually uses: after invalidation the
    /// CPU binder's precondition must hold, so a compiled plan binds the parameter instead of throwing.
    /// </summary>
    [SkippableFact]
    public void AnInvalidatedWeightExposesWritableCpuStorageForTheFusedOptimizer()
    {
        using var gpu = new DirectGpuTensorEngine();
        var weights = GpuResidentWeights(gpu, 0.5f);

        gpu.InvalidateResidentWeightBuffer(weights);

        // GetCpuBackingForContiguousWrite returns null for any non-CPU device — it deliberately does not
        // force materialization — and BindCpuOptimizerTensor throws before even reaching it. Both are
        // satisfied only once the device state is honest.
        var backing = weights.GetCpuBackingForContiguousWrite(out int offset);

        Assert.NotNull(backing);
        float[] cpuBacking = backing;
        Assert.True(
            offset + weights.Length <= cpuBacking.Length,
            $"backing too small: offset={offset}, length={weights.Length}, backing={cpuBacking.Length}");
    }

    /// <summary>
    /// The split-complex marker records that the DEVICE buffer holds a
    /// <c>[real plane][imaginary plane]</c> layout rather than one element per slot, so it describes the
    /// buffer and not the tensor. Invalidation drops the buffer, and nothing on the way back up —
    /// <c>Tensor.Gpu()</c> included — ever clears the marker, so leaving it set here arms it against the
    /// next ordinary interleaved upload: <c>GetOrAllocateSplitComplexBuffers</c> would then read that
    /// buffer's first half as the real plane and its second as the imaginary one. Needs no GPU: the
    /// marker is plain tensor state and the assertion is about who owns it.
    /// </summary>
    [Fact]
    public void InvalidatingAResidentWeightBufferClearsTheSplitComplexMarker()
    {
        using var gpu = new DirectGpuTensorEngine();
        var weights = Weights(16, 8, 0.25f);
        weights._gpuBufferIsSplitComplex = true;

        gpu.InvalidateResidentWeightBuffer(weights);

        Assert.False(weights._gpuBufferIsSplitComplex);
        Assert.Equal(TensorDevice.CPU, weights.Device);
    }

    /// <summary>
    /// The second marker on the same buffer, with the same lifetime and the same failure mode.
    /// <c>_gpuBufferContainsRawInt32</c> says the DEVICE buffer holds raw int32 index bits instead of
    /// the usual one-float-per-element encoding — the only thing that reads it,
    /// <c>DirectGpuTensorEngine.GetOrAllocateInt32IndexBuffer</c>, forwards such a buffer to its
    /// consumer UNCONVERTED rather than routing it through <c>ConvertNumericIndicesToInt32</c>.
    /// Invalidation drops the buffer, so a marker left set arms that shortcut against whatever comes
    /// next: index 3 would reach the kernel as 0x40400000.
    /// </summary>
    [Fact]
    public void InvalidatingAResidentWeightBufferClearsTheRawInt32Marker()
    {
        using var gpu = new DirectGpuTensorEngine();
        var indices = new Tensor<int>([8]);
        indices._gpuBufferContainsRawInt32 = true;

        gpu.InvalidateResidentWeightBuffer(indices);

        Assert.False(indices._gpuBufferContainsRawInt32);
    }

    /// <summary>
    /// Storage replacement, which is the path CodeRabbit's finding named. Unlike
    /// <c>InvalidateResidentWeightBuffer</c>, <c>RebindStorageFrom</c> leaves <c>_device</c> untouched,
    /// so the tensor stays GPU-resident with no buffer — and that is exactly the state
    /// <c>GetOrAllocateInt32IndexBuffer</c> branches on: <c>HasResidentIndexStorage</c> is satisfied by
    /// <c>IsGpuResident</c> alone, and the raw-int32 shortcut is taken before anything re-examines the
    /// buffer. The marker therefore has to die with the storage it described, not with the device flag.
    /// </summary>
    /// <remarks>
    /// Asserted on the marker rather than by calling the consumer because the consumer needs a live
    /// backend; this pins the precondition it reads, on any box, with no GPU.
    /// </remarks>
    [Fact]
    public void ReplacingStorageClearsTheRawInt32Marker()
    {
        var indices = new Tensor<int>([8]);
        var replacement = new Tensor<int>([8]);
        for (int i = 0; i < replacement.Length; i++)
        {
            replacement[i] = i;
        }

        indices._gpuBufferContainsRawInt32 = true;
        indices.RebindStorageFrom(replacement);

        Assert.False(indices._gpuBufferContainsRawInt32);
    }
}
