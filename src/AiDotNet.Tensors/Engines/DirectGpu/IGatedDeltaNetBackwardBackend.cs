// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Internal capability interface implemented by GPU backends that provide a fused device-resident
/// BPTT backward pass for <see cref="IDirectGpuBackend.GatedDeltaNetScanForward"/>.
/// Keeping this internal avoids breaking external <see cref="IDirectGpuBackend"/> implementers
/// while allowing <see cref="DirectGpuTensorEngine"/> to keep GatedDeltaNet training 100% on-device.
/// </summary>
internal interface IGatedDeltaNetBackwardBackend
{
    void GatedDeltaNetScanBackward(
        IGpuBuffer dOut,
        IGpuBuffer q,
        IGpuBuffer k,
        IGpuBuffer v,
        IGpuBuffer alpha,
        IGpuBuffer beta,
        IGpuBuffer dQ,
        IGpuBuffer dK,
        IGpuBuffer dV,
        IGpuBuffer dAlpha,
        IGpuBuffer dBeta,
        int batch,
        int seqLen,
        int modelDim,
        int numHeads,
        int headDim);
}

internal static class GatedDeltaNetBackwardResolver
{
    /// <summary>The largest head every backend's scan kernels hold in their per-thread state rows.</summary>
    public const int MaxHeadDim = 256;

    /// <summary>The physical backend's fused backward, or null when it has none or may not launch it now.</summary>
    public static IGatedDeltaNetBackwardBackend? Resolve(IDirectGpuBackend backend)
    {
        while (backend is Gpu.DelegatingGpuBackend wrapper)
        {
            // A pure deferred scope queues its ops for later replay; launching the backward on the physical backend
            // here would run it now, ahead of the queued work that produces its inputs. Decline so the managed
            // backward runs instead.
            if (wrapper is Gpu.RecordingGpuBackend { IsRecording: true, ExecuteWhileRecording: false })
                return null;
            backend = wrapper.InnerBackend;
        }
        return backend as IGatedDeltaNetBackwardBackend;
    }

    /// <summary>
    /// Rejects buffers smaller than the scan's shapes before any kernel reads or writes them: dOut, q, k, v, dQ, dK
    /// and dV hold [batch, seqLen, modelDim], alpha, beta, dAlpha and dBeta [batch, seqLen, numHeads].
    /// </summary>
    public static void ValidateBuffers(
        IGpuBuffer dOut, IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer alpha, IGpuBuffer beta,
        IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, IGpuBuffer dAlpha, IGpuBuffer dBeta,
        int batch, int seqLen, int modelDim, int numHeads)
    {
        long sequence = (long)batch * seqLen * modelDim, gates = (long)batch * seqLen * numHeads;
        Require(dOut, sequence, nameof(dOut)); Require(q, sequence, nameof(q)); Require(k, sequence, nameof(k));
        Require(v, sequence, nameof(v)); Require(dQ, sequence, nameof(dQ)); Require(dK, sequence, nameof(dK));
        Require(dV, sequence, nameof(dV));
        Require(alpha, gates, nameof(alpha)); Require(beta, gates, nameof(beta));
        Require(dAlpha, gates, nameof(dAlpha)); Require(dBeta, gates, nameof(dBeta));
    }

    private static void Require(IGpuBuffer buffer, long elements, string name)
    {
        if (buffer is null) throw new ArgumentNullException(name);
        if (buffer.Size < elements)
            throw new ArgumentException($"GatedDeltaNet buffer {name} holds {buffer.Size} elements; the scan needs {elements}.", name);
    }
}
