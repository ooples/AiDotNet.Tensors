using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
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
        Assert.True(
            offset + weights.Length <= backing!.Length,
            $"backing too small: offset={offset}, length={weights.Length}, backing={backing.Length}");
    }
}
