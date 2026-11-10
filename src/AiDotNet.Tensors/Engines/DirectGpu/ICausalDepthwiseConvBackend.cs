// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Internal capability interface for the channels-last causal depthwise convolution
/// <see cref="CpuEngine.CausalDepthwiseConv1D{T}"/>: <c>y[r, t, c] = sum_{j &lt; K} w[j, c] * x[r, t - j, c]</c> over
/// <c>rows</c> independent sequences of <c>seqLen</c> steps and <c>channels</c> channels (zero before the sequence
/// start). All three kernels write every output element, so no output needs pre-zeroing, and none uses atomics: the
/// kernel gradient sums each (tap, channel) over all rows and steps in a fixed order, so it is deterministic.
/// Internal so external <see cref="IDirectGpuBackend"/> implementers are not broken.
/// </summary>
internal interface ICausalDepthwiseConvBackend
{
    /// <summary>output[r, t, c] = sum_j kernel[j, c] * input[r, t - j, c].</summary>
    void CausalDepthwiseConv1DForward(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int rows, int seqLen, int channels, int taps);

    /// <summary>gradInput[r, t, c] = sum_j kernel[j, c] * gradOutput[r, t + j, c].</summary>
    void CausalDepthwiseConv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int rows, int seqLen, int channels, int taps);

    /// <summary>gradKernel[j, c] = sum_{r, t} gradOutput[r, t, c] * input[r, t - j, c].</summary>
    void CausalDepthwiseConv1DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int rows, int seqLen, int channels, int taps);
}

internal static class CausalDepthwiseConvResolver
{
    /// <summary>The physical backend's kernels, or null when it has none or may not launch them now.</summary>
    public static ICausalDepthwiseConvBackend? Resolve(IDirectGpuBackend backend)
    {
        while (backend is Gpu.DelegatingGpuBackend wrapper)
        {
            // A pure deferred scope queues its ops for later replay; launching on the physical backend here would
            // run ahead of the queued work that produces the inputs. Decline so the managed path runs instead.
            if (wrapper is Gpu.RecordingGpuBackend { IsRecording: true, ExecuteWhileRecording: false })
                return null;
            backend = wrapper.InnerBackend;
        }
        return backend as ICausalDepthwiseConvBackend;
    }
}
