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
        int headDim,
        float keyScale,
        bool decayFirst);
}

/// <summary>
/// Internal capability interface of GPU backends whose fused Gated DeltaNet scan forward takes the rule terms:
/// <paramref name="keyScale"/> multiplies every key, and <paramref name="decayFirst"/> makes the prediction read the
/// decayed state (<see cref="GatedDeltaRule.Standard"/>); <see cref="IDirectGpuBackend.GatedDeltaNetScanForward"/> is
/// the original rule, keys scaled by 1/sqrt(headDim) and the prediction read before the decay.
/// </summary>
internal interface IGatedDeltaRuleForwardBackend
{
    void GatedDeltaNetScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer alpha, IGpuBuffer beta, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, bool decayFirst);
}

internal static class GatedDeltaNetBackwardResolver
{
    /// <summary>The largest head every backend's scan kernels hold in their per-thread state rows.</summary>
    public const int MaxHeadDim = 256;

    /// <summary>The physical backend's fused backward, or null when it has none or may not launch it now.</summary>
    public static IGatedDeltaNetBackwardBackend? Resolve(IDirectGpuBackend backend)
        => Physical(backend) as IGatedDeltaNetBackwardBackend;

    private static IDirectGpuBackend? Physical(IDirectGpuBackend backend)
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
        return backend;
    }

    /// <summary>The kernels' terms of the original rule: keys scaled by 1/sqrt(headDim), prediction before the decay.</summary>
    public static float OriginalKeyScale(int headDim) => (float)(1.0 / Math.Sqrt(headDim));

    /// <summary>A float's bit pattern, for backends whose kernel scalars are ints (the kernel bit-casts it back).</summary>
    public static int FloatBits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

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
