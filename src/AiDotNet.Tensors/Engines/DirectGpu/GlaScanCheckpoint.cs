namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Buffer plan for the checkpointed GLA backward kernels every backend runs: the state entering each chunk of
/// <c>chunk</c> steps is saved, and one chunk of states is recomputed at a time, so backward holds
/// <c>batch * heads * (nChunks + chunk) * headDim^2</c> floats instead of the <c>batch * heads * seqLen * headDim^2</c>
/// state trajectory. <c>chunk = ceil(sqrt(seqLen))</c> minimises <c>nChunks + chunk</c>.
/// </summary>
internal static class GlaScanCheckpoint
{
    /// <summary>Steps per checkpoint chunk: <c>ceil(sqrt(seqLen))</c>, clamped to <c>[1, seqLen]</c>.</summary>
    internal static int Chunk(int seqLen)
    {
        int c = (int)Math.Ceiling(Math.Sqrt(seqLen));
        return Math.Max(1, Math.Min(seqLen, c));
    }

    /// <summary>The chunk length and the float counts of the checkpoint and scratch buffers.</summary>
    internal static (int Chunk, int CheckpointLength, int ScratchLength) Plan(int batch, int seqLen, int numHeads, int headDim)
    {
        int chunk = Chunk(seqLen);
        int nChunks = (seqLen + chunk - 1) / chunk;
        long rows = checked((long)batch * numHeads * headDim * headDim);
        long ckpt = checked(rows * nChunks);
        long scratch = checked(rows * chunk);
        if (ckpt > int.MaxValue || scratch > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA backward checkpoint buffer is too large for a single allocation.");
        return (chunk, (int)ckpt, (int)scratch);
    }
}
