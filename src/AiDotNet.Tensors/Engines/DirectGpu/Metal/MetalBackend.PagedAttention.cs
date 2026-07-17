// Copyright (c) AiDotNet. All rights reserved.
// Paged-attention decode dispatch (P1) for the Metal backend.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    /// <summary>
    /// Paged-attention decode (P1): out[heads*headDim] = softmax(scale·Q·K)·V over the sequence,
    /// reading K/V from the physical block pool [maxBlocks, blockSize, heads, headDim] via
    /// <paramref name="blockTable"/> (an int buffer of physical block ids). headDim &lt;= 256.
    /// Matches a standard-attention CPU oracle.
    /// </summary>
    public IGpuBuffer PagedAttentionDecode(IGpuBuffer q, IGpuBuffer kcache, IGpuBuffer vcache, IGpuBuffer blockTable,
        int heads, int headDim, int blockSize, int seqLen, float scale)
    {
        ThrowIfDisposed();
        GpuKernelGuards.Attention(heads, headDim, blockSize, seqLen, nameof(PagedAttentionDecode));
        var output = AllocateBuffer(heads * headDim);
        var pipeline = GetPipeline("PagedAttention", _pagedAttnLibrary, "paged_attention_decode");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(heads);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)q, 0);
        encoder.SetBuffer((MetalGpuBuffer)kcache, 1);
        encoder.SetBuffer((MetalGpuBuffer)vcache, 2);
        encoder.SetBuffer((MetalGpuBuffer)blockTable, 3);
        encoder.SetBuffer((MetalGpuBuffer)output, 4);
        encoder.SetBytes(heads, 5);
        encoder.SetBytes(headDim, 6);
        encoder.SetBytes(blockSize, 7);
        encoder.SetBytes(seqLen, 8);
        encoder.SetBytes(scale, 9);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        return output;
    }

    /// <summary>
    /// Prefill / multi-query paged attention (P1, causal): out[numQueries,heads,headDim]; query qi
    /// (logical position startPos+qi) attends to key positions 0..(startPos+qi). headDim &lt;= 256.
    /// </summary>
    public IGpuBuffer PagedAttentionPrefill(IGpuBuffer q, IGpuBuffer kcache, IGpuBuffer vcache, IGpuBuffer blockTable,
        int heads, int headDim, int blockSize, int numQueries, int startPos, float scale)
    {
        ThrowIfDisposed();
        GpuKernelGuards.Attention(heads, headDim, blockSize, numQueries, nameof(PagedAttentionPrefill));
        if (startPos < 0) throw new ArgumentOutOfRangeException(nameof(startPos));
        var output = AllocateBuffer(numQueries * heads * headDim);
        var pipeline = GetPipeline("PagedAttention", _pagedAttnLibrary, "paged_attention_prefill");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(numQueries * heads);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)q, 0);
        encoder.SetBuffer((MetalGpuBuffer)kcache, 1);
        encoder.SetBuffer((MetalGpuBuffer)vcache, 2);
        encoder.SetBuffer((MetalGpuBuffer)blockTable, 3);
        encoder.SetBuffer((MetalGpuBuffer)output, 4);
        encoder.SetBytes(heads, 5);
        encoder.SetBytes(headDim, 6);
        encoder.SetBytes(blockSize, 7);
        encoder.SetBytes(numQueries, 8);
        encoder.SetBytes(startPos, 9);
        encoder.SetBytes(scale, 10);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        return output;
    }

    /// <summary>
    /// GQA decode (P1): like <see cref="PagedAttentionDecode"/> but query head h shares KV head
    /// h/(heads/kvHeads); K/V pool is [maxBlocks, blockSize, kvHeads, headDim]. headDim &lt;= 256.
    /// </summary>
    public IGpuBuffer PagedAttentionDecodeGqa(IGpuBuffer q, IGpuBuffer kcache, IGpuBuffer vcache, IGpuBuffer blockTable,
        int heads, int kvHeads, int headDim, int blockSize, int seqLen, float scale)
    {
        ThrowIfDisposed();
        GpuKernelGuards.Attention(heads, headDim, blockSize, seqLen, nameof(PagedAttentionDecodeGqa));
        GpuKernelGuards.Gqa(heads, kvHeads, nameof(PagedAttentionDecodeGqa));
        var output = AllocateBuffer(heads * headDim);
        var pipeline = GetPipeline("PagedAttention", _pagedAttnLibrary, "paged_attention_decode_gqa");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(heads);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)q, 0);
        encoder.SetBuffer((MetalGpuBuffer)kcache, 1);
        encoder.SetBuffer((MetalGpuBuffer)vcache, 2);
        encoder.SetBuffer((MetalGpuBuffer)blockTable, 3);
        encoder.SetBuffer((MetalGpuBuffer)output, 4);
        encoder.SetBytes(heads, 5);
        encoder.SetBytes(kvHeads, 6);
        encoder.SetBytes(headDim, 7);
        encoder.SetBytes(blockSize, 8);
        encoder.SetBytes(seqLen, 9);
        encoder.SetBytes(scale, 10);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        return output;
    }

    /// <summary>
    /// GQA prefill (P1, causal): like <see cref="PagedAttentionPrefill"/> but query head h shares KV head
    /// h/(heads/kvHeads); K/V pool is [maxBlocks, blockSize, kvHeads, headDim]. headDim &lt;= 256.
    /// </summary>
    public IGpuBuffer PagedAttentionPrefillGqa(IGpuBuffer q, IGpuBuffer kcache, IGpuBuffer vcache, IGpuBuffer blockTable,
        int heads, int kvHeads, int headDim, int blockSize, int numQueries, int startPos, float scale)
    {
        ThrowIfDisposed();
        GpuKernelGuards.Attention(heads, headDim, blockSize, numQueries, nameof(PagedAttentionPrefillGqa));
        GpuKernelGuards.Gqa(heads, kvHeads, nameof(PagedAttentionPrefillGqa));
        if (startPos < 0) throw new ArgumentOutOfRangeException(nameof(startPos));
        var output = AllocateBuffer(numQueries * heads * headDim);
        var pipeline = GetPipeline("PagedAttention", _pagedAttnLibrary, "paged_attention_prefill_gqa");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(numQueries * heads);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)q, 0);
        encoder.SetBuffer((MetalGpuBuffer)kcache, 1);
        encoder.SetBuffer((MetalGpuBuffer)vcache, 2);
        encoder.SetBuffer((MetalGpuBuffer)blockTable, 3);
        encoder.SetBuffer((MetalGpuBuffer)output, 4);
        encoder.SetBytes(heads, 5);
        encoder.SetBytes(kvHeads, 6);
        encoder.SetBytes(headDim, 7);
        encoder.SetBytes(blockSize, 8);
        encoder.SetBytes(numQueries, 9);
        encoder.SetBytes(startPos, 10);
        encoder.SetBytes(scale, 11);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        return output;
    }

    /// <summary>
    /// Fused decode attention (P2, FlashDecoding): single-query attention over contiguous K/V
    /// [seqLen, kvHeads, headDim], split across threads and merged by an online-softmax reduction.
    /// GQA via kvHead = h/(heads/kvHeads); pass <paramref name="kvHeads"/> == heads for MHA. headDim &lt;= 256.
    /// </summary>
    public IGpuBuffer FlashDecode(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v,
        int heads, int kvHeads, int headDim, int seqLen, float scale, int splits = 0)
    {
        ThrowIfDisposed();
        GpuKernelGuards.FlashDecode(heads, kvHeads, headDim, seqLen, nameof(FlashDecode));
        GpuKernelGuards.Capacity(q, (long)heads * headDim, nameof(q), nameof(FlashDecode));
        GpuKernelGuards.Capacity(k, (long)seqLen * kvHeads * headDim, nameof(k), nameof(FlashDecode));
        GpuKernelGuards.Capacity(v, (long)seqLen * kvHeads * headDim, nameof(v), nameof(FlashDecode));
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        int effSplits = splits > 0 ? splits : System.Math.Min(seqLen, 8);
        if (effSplits > seqLen) effSplits = seqLen;
        int splitLen = (seqLen + effSplits - 1) / effSplits;

        var output = AllocateBuffer(heads * headDim);
        var partialM = AllocateBuffer(heads * effSplits);
        var partialL = AllocateBuffer(heads * effSplits);
        var partialAcc = AllocateBuffer(heads * effSplits * headDim);
        try
        {
            var partPipe = GetPipeline("FlashDecode", _flashDecodeLibrary, "flash_decode_partial");
            var (pg, ptpg) = partPipe.Calculate1DDispatch(heads * effSplits);
            using (var enc = _commandQueue.CreateScopedComputeEncoder())
            {
                enc.SetPipelineState(partPipe.Handle);
                enc.SetBuffer((MetalGpuBuffer)q, 0);
                enc.SetBuffer((MetalGpuBuffer)k, 1);
                enc.SetBuffer((MetalGpuBuffer)v, 2);
                enc.SetBuffer((MetalGpuBuffer)partialM, 3);
                enc.SetBuffer((MetalGpuBuffer)partialL, 4);
                enc.SetBuffer((MetalGpuBuffer)partialAcc, 5);
                enc.SetBytes(heads, 6);
                enc.SetBytes(kvHeads, 7);
                enc.SetBytes(headDim, 8);
                enc.SetBytes(seqLen, 9);
                enc.SetBytes(effSplits, 10);
                enc.SetBytes(splitLen, 11);
                enc.SetBytes(scale, 12);
                enc.DispatchThreadgroups(pg, ptpg);
            }

            var reducePipe = GetPipeline("FlashDecode", _flashDecodeLibrary, "flash_decode_reduce");
            var (rg, rtpg) = reducePipe.Calculate1DDispatch(heads);
            using (var enc = _commandQueue.CreateScopedComputeEncoder())
            {
                enc.SetPipelineState(reducePipe.Handle);
                enc.SetBuffer((MetalGpuBuffer)partialM, 0);
                enc.SetBuffer((MetalGpuBuffer)partialL, 1);
                enc.SetBuffer((MetalGpuBuffer)partialAcc, 2);
                enc.SetBuffer((MetalGpuBuffer)output, 3);
                enc.SetBytes(heads, 4);
                enc.SetBytes(headDim, 5);
                enc.SetBytes(effSplits, 6);
                enc.DispatchThreadgroups(rg, rtpg);
            }
            return output;
        }
        catch { output.Dispose(); throw; }
        finally { partialM.Dispose(); partialL.Dispose(); partialAcc.Dispose(); }
    }
}
