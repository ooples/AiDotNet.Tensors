// Copyright (c) AiDotNet. All rights reserved.
// Capability parity (#1142): seeded Gumbel-softmax (the IGpuBatchExecution kernel), uninitialized allocation and
// multi-tensor Adam/AdamW (per-tensor native kernels).
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IFftBackend, IGenericFftBackend, IMeshgridBackend, ISeededGumbelSoftmaxBackend, IUninitializedGpuAllocation, IMultiTensorGpuOptimizerBackend, IDeviceReadbackFactory
{
    // A non-blocking read queued behind the submitted work; the host waits only on Read (OpenClBackend.Readback.cs).
    IDeviceReadback IDeviceReadbackFactory.CreateReadback(int byteCapacity) => new OpenClDeviceReadback(this, byteCapacity);

    /// <remarks>OpenCL keeps its power-of-two FFT kernel.</remarks>
    bool IGenericFftBackend.PrefersGenericFftForPowerOfTwo => false;

    /// <summary>
    /// This backend's allocation does no separate fill that could be skipped, so the plain allocation already meets
    /// the undefined-contents contract.
    /// </summary>
    public IGpuBuffer AllocateBufferUninitialized(int size) => AllocateBuffer(size);
    /// <summary>
    /// The guarded updates read the guard in their kernels (<c>*_update_guarded</c>, OptimizerKernels): a step whose
    /// gradients are not finite writes nothing, and the host learns it from a deferred readback, so the step never
    /// waits on the device for the verdict (#1502).
    /// </summary>
    public bool GuardsFiniteOnDevice => true;

    public void AdamMultiTensorUpdate(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
        => PerTensorMultiTensorOptimizer.Update(this, false, parameters, gradients, firstMoments, secondMoments,
            sizes, learningRate, beta1, beta2, epsilon, weightDecay, step);

    public void AdamWMultiTensorUpdate(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step)
        => PerTensorMultiTensorOptimizer.Update(this, true, parameters, gradients, firstMoments, secondMoments,
            sizes, learningRate, beta1, beta2, epsilon, weightDecay, step);

    public void AdamMultiTensorUpdateGuarded(bool decoupledWeightDecay,
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step, IGpuBuffer finiteGuard)
    {
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateAdam(parameters, gradients, firstMoments, secondMoments, sizes, epsilon, step);
        var k = _kernelCache[decoupledWeightDecay ? "adamw_update_guarded" : "adam_update_guarded"];
        var guard = ((DirectOpenClGpuBuffer)finiteGuard).Buffer.Handle;
        for (int t = 0; t < parameters.Count; t++)
        {
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)parameters[t]).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradients[t]).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)firstMoments[t]).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)secondMoments[t]).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, step);
            k.SetArg(arg++, sizes[t]);
            k.SetArg(arg++, guard);
            k.Execute1D(sizes[t], Math.Min(256, sizes[t]));
        }
    }

    public void SgdMultiTensorUpdate(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        float learningRate, float weightDecay)
        => PerTensorMultiTensorOptimizer.Sgd(this, parameters, gradients, sizes, learningRate, weightDecay);

    public void SgdMultiTensorUpdateGuarded(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        float learningRate, float weightDecay, IGpuBuffer finiteGuard)
    {
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateSgd(parameters, gradients, sizes);
        var k = _kernelCache["sgd_update_guarded"];
        var guard = ((DirectOpenClGpuBuffer)finiteGuard).Buffer.Handle;
        for (int t = 0; t < parameters.Count; t++)
        {
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)parameters[t]).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)gradients[t]).Buffer.Handle);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, weightDecay);
            k.SetArg(arg++, sizes[t]);
            k.SetArg(arg++, guard);
            k.Execute1D(sizes[t], Math.Min(256, sizes[t]));
        }
    }

    public void Meshgrid2DPair(IGpuBuffer source0, IGpuBuffer source1, IGpuBuffer output0, IGpuBuffer output1,
        int n0, int n1, bool xy)
    {
        if (n0 <= 0) throw new ArgumentOutOfRangeException(nameof(n0));
        if (n1 <= 0) throw new ArgumentOutOfRangeException(nameof(n1));
        LaunchBatch("bx_meshgrid2d", checked(n0 * n1), source0, source1, output0, output1, n0, n1, xy ? 1 : 0);
    }

    bool IGenericFftBackend.SupportsFftElementType(FftElementType type) => PortableGenericFft.Supports(type);

    void IGenericFftBackend.ConvertFloatToFftStorage(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type)
        => PortableGenericFft.ToStorage(this, LaunchBatch, source, destination, count, type);

    void IGenericFftBackend.ConvertFftStorageToFloat(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type)
        => PortableGenericFft.FromStorage(this, LaunchBatch, source, destination, count, type);

    void IGenericFftBackend.LaunchFftGeneric(IGpuBuffer real, IGpuBuffer imaginary, int batch, int n, bool inverse, FftElementType type)
        => PortableGenericFft.Launch(this, LaunchBatch, real, imaginary, batch, n, inverse, type);

    public void LaunchFft(IGpuBuffer buffer, int batchCount, int n, bool inverse)
        => PortableGenericFft.LaunchInterleaved(this, this, buffer, batchCount, n, inverse);
}

