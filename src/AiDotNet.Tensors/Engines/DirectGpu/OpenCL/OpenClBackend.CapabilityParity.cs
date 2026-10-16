// Copyright (c) AiDotNet. All rights reserved.
// Capability parity (#1142): seeded Gumbel-softmax (the IGpuBatchExecution kernel), uninitialized allocation and
// multi-tensor Adam/AdamW (per-tensor native kernels).
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IFftBackend, IGenericFftBackend, IMeshgridBackend, ISeededGumbelSoftmaxBackend, IUninitializedGpuAllocation, IMultiTensorGpuOptimizerBackend, IDeviceReadbackFactory
{
    // No asynchronous device-to-host copy is wired on this backend: the readback downloads when it is queued.
    IDeviceReadback IDeviceReadbackFactory.CreateReadback(int byteCapacity) => new SynchronousDeviceReadback(this, byteCapacity);

    /// <remarks>OpenCL keeps its power-of-two FFT kernel.</remarks>
    bool IGenericFftBackend.PrefersGenericFftForPowerOfTwo => false;

    /// <summary>
    /// This backend's allocation does no separate fill that could be skipped, so the plain allocation already meets
    /// the undefined-contents contract.
    /// </summary>
    public IGpuBuffer AllocateBufferUninitialized(int size) => AllocateBuffer(size);
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
        => PerTensorMultiTensorOptimizer.UpdateGuarded(this, decoupledWeightDecay, parameters, gradients, firstMoments,
            secondMoments, sizes, learningRate, beta1, beta2, epsilon, weightDecay, step, finiteGuard);

    public void SgdMultiTensorUpdate(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        float learningRate, float weightDecay)
        => PerTensorMultiTensorOptimizer.Sgd(this, parameters, gradients, sizes, learningRate, weightDecay);

    public void SgdMultiTensorUpdateGuarded(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        float learningRate, float weightDecay, IGpuBuffer finiteGuard)
        => PerTensorMultiTensorOptimizer.SgdGuarded(this, parameters, gradients, sizes, learningRate, weightDecay, finiteGuard);

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

