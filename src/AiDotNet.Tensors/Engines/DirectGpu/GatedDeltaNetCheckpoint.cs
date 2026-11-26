namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Buffer plan for the checkpointed Gated DeltaNet backward kernel (<c>gated_delta_scan_backward_ckpt</c>) the
/// HIP, OpenCL, Metal, Vulkan and WebGPU backends run. One work-group per (batch, head) saves the state entering each
/// chunk of <c>chunk</c> steps, recomputes one chunk of states at a time and runs BPTT over it, so backward holds
/// <c>(nChunks + chunk + 1) * headDim^2 + 2 * chunk * headDim</c> floats per (batch, head) instead of the
/// <c>seqLen * headDim^2</c> state trajectory. <c>chunk = ceil(sqrt(seqLen))</c> (<see cref="GlaScanCheckpoint.Chunk"/>).
/// </summary>
/// <remarks>
/// Each head's workspace is laid out as Ckpt <c>[nChunks][d][d]</c> | Scratch <c>[chunk][d][d]</c> | dS <c>[d][d]</c> |
/// Part <c>[chunk][2][d]</c>. The kernel sums dQ, dK, dA and dB over the state rows in ascending order behind
/// work-group barriers, so the gradients are deterministic, need no float atomics and no pre-zeroed outputs.
/// </remarks>
internal static class GatedDeltaNetCheckpoint
{
    /// <summary>The kernel's name on every backend that runs it.</summary>
    internal const string KernelName = "gated_delta_scan_backward_ckpt";

    /// <summary>The chunk length, each head's workspace stride and the whole workspace's float count.</summary>
    internal static (int Chunk, int HeadStride, int WorkLength) Plan(int batch, int seqLen, int numHeads, int headDim)
    {
        int chunk = GlaScanCheckpoint.Chunk(seqLen);
        int nChunks = (seqLen + chunk - 1) / chunk;
        long hh = (long)headDim * headDim;
        long stride = checked((nChunks + chunk + 1L) * hh + 2L * chunk * headDim);
        long work = checked(stride * batch * numHeads);
        if (work > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batch),
                "GatedDeltaNet backward checkpoint workspace is too large for a single allocation.");
        return (chunk, (int)stride, (int)work);
    }

    /// <summary>Threads per work-group: one per state row up to <paramref name="limit"/>, a multiple of 32.</summary>
    internal static int GroupSize(int headDim, int limit)
    {
        int rows = (headDim + 31) / 32 * 32;
        int cap = Math.Max(1, Math.Min(256, limit));
        return Math.Min(rows, cap);
    }
}
