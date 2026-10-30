// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Capability parity (#1142): seeded Gumbel-softmax (the IGpuBatchExecution kernel), uninitialized allocation and
// multi-tensor Adam/AdamW (per-tensor native kernels).
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IGenericFftBackend, IMeshgridBackend, ISeededGumbelSoftmaxBackend, IUninitializedGpuAllocation, IMultiTensorGpuOptimizerBackend, IDeviceReadbackFactory
{
    // A blit committed without waiting; the host waits for its command buffer only on Read (#1520).
    IDeviceReadback IDeviceReadbackFactory.CreateReadback(int byteCapacity) => new MetalDeviceReadback(this, byteCapacity);

    /// <remarks>Metal keeps its power-of-two FFT kernel.</remarks>
    bool IGenericFftBackend.PrefersGenericFftForPowerOfTwo => false;

    /// <summary>
    /// This backend's allocation does no separate fill that could be skipped, so the plain allocation already meets
    /// the undefined-contents contract.
    /// </summary>
    public IGpuBuffer AllocateBufferUninitialized(int size) => AllocateBuffer(size);
    /// <summary>
    /// The guarded updates read the guard on the device (optimizer_update_guarded) and write nothing when it is not
    /// finite (#1520); without the resident library they read it on the host.
    /// </summary>
    public bool GuardsFiniteOnDevice => _residentLibrary != IntPtr.Zero;

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
        if (!GuardsFiniteOnDevice)
        {
            PerTensorMultiTensorOptimizer.UpdateGuarded(this, decoupledWeightDecay, parameters, gradients, firstMoments,
                secondMoments, sizes, learningRate, beta1, beta2, epsilon, weightDecay, step, finiteGuard);
            return;
        }
        ThrowIfDisposed();
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateAdam(parameters, gradients, firstMoments, secondMoments, sizes, epsilon, step);
        // AdamUpdate's / AdamWUpdate's operation codes (2 / 3) and arguments.
        for (int t = 0; t < parameters.Count; t++)
            DispatchOptimizerMetalGuarded(decoupledWeightDecay ? 3u : 2u, parameters[t], gradients[t], firstMoments[t],
                secondMoments[t], finiteGuard, sizes[t], step, learningRate, beta1, beta2, epsilon, weightDecay);
    }

    public void SgdMultiTensorUpdate(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        float learningRate, float weightDecay)
        => PerTensorMultiTensorOptimizer.Sgd(this, parameters, gradients, sizes, learningRate, weightDecay);

    public void SgdMultiTensorUpdateGuarded(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        float learningRate, float weightDecay, IGpuBuffer finiteGuard)
    {
        if (!GuardsFiniteOnDevice)
        {
            PerTensorMultiTensorOptimizer.SgdGuarded(this, parameters, gradients, sizes, learningRate, weightDecay, finiteGuard);
            return;
        }
        ThrowIfDisposed();
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateSgd(parameters, gradients, sizes);
        // SgdUpdate's operation code (0) and arguments.
        for (int t = 0; t < parameters.Count; t++)
            DispatchOptimizerMetalGuarded(0u, parameters[t], gradients[t], null, null, finiteGuard, sizes[t], 0,
                learningRate, weightDecay);
    }

    public void Meshgrid2DPair(IGpuBuffer source0, IGpuBuffer source1, IGpuBuffer output0, IGpuBuffer output1,
        int n0, int n1, bool xy)
        => Portable.Meshgrid2DPair(source0, source1, output0, output1, n0, n1, xy);

    private GenericFftReadiness? _fftReadiness;

    // The capability holds only once every FFT pipeline has been built from the batch library (#963).
    internal GenericFftReadiness FftReadiness => _fftReadiness ??= new GenericFftReadiness(CompileBatchKernel);

    private string? CompileBatchKernel(string name)
    {
        try
        {
            GetBatchPipeline(name);
            return null;
        }
        catch (InvalidOperationException ex)
        {
            // The Metal compiler's error becomes the capability's reason, reported when the transform is requested.
            return ex.Message;
        }
    }

    bool IGenericFftBackend.SupportsFftElementType(FftElementType type) => FftReadiness.IsReady(type);

    void IGenericFftBackend.ConvertFloatToFftStorage(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type)
    {
        FftReadiness.Require(type, "Metal");
        PortableGenericFft.ToStorage(this, LaunchBatch, source, destination, count, type);
    }

    void IGenericFftBackend.ConvertFftStorageToFloat(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type)
    {
        FftReadiness.Require(type, "Metal");
        PortableGenericFft.FromStorage(this, LaunchBatch, source, destination, count, type);
    }

    void IGenericFftBackend.LaunchFftGeneric(IGpuBuffer real, IGpuBuffer imaginary, int batch, int n, bool inverse, FftElementType type)
    {
        FftReadiness.Require(type, "Metal");
        PortableGenericFft.Launch(this, LaunchBatch, real, imaginary, batch, n, inverse, type);
    }
}
