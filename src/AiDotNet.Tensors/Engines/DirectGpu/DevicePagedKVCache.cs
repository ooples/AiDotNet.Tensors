// Copyright (c) AiDotNet. All rights reserved.
// On-device paged KV cache (P1): the GPU-resident sibling of the CPU
// AiDotNet.Tensors.Engines.Autodiff.PagedKVCache reference.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// On-device block-based KV cache à la vLLM's PagedAttention. The physical
/// block pool lives entirely in GPU memory as two flat device buffers
/// (<see cref="KeyBlocks"/> / <see cref="ValueBlocks"/>, laid out
/// <c>[MaxBlocks, BlockSize, Heads, HeadDim]</c>); per-sequence block-table
/// bookkeeping and the free list stay host-side, exactly as vLLM's block
/// manager does. Appends copy device-to-device into the pool (no host
/// round-trip on the resident path) and the block table is uploaded on
/// demand as an int buffer that feeds straight into the paged-attention
/// kernels (<c>PagedAttentionDecode</c> / <c>Prefill</c> and their GQA
/// variants) on any backend.
///
/// <para>Contract mirrors the CPU <c>PagedKVCache&lt;float&gt;</c> reference:
/// append + block table + free + prefix-share, so results are validated
/// against the same standard-attention oracle. Backend-agnostic — it only
/// uses <see cref="IDirectGpuBackend"/> buffer primitives (AllocateBuffer,
/// AllocateIntBuffer, device Copy at offsets), so it runs on all six GPU
/// backends.</para>
/// </summary>
public sealed class DevicePagedKVCache : IDisposable
{
    /// <summary>Number of tokens per block — classic vLLM default is 16.</summary>
    public int BlockSize { get; }

    public int MaxBlocks { get; }
    public int Heads { get; }
    public int HeadDim { get; }

    /// <summary>Device-resident key pool, flat [MaxBlocks*BlockSize*Heads*HeadDim].</summary>
    public IGpuBuffer KeyBlocks { get; }

    /// <summary>Device-resident value pool, flat [MaxBlocks*BlockSize*Heads*HeadDim].</summary>
    public IGpuBuffer ValueBlocks { get; }

    private readonly IDirectGpuBackend _backend;
    private readonly int _stepStride;               // Heads * HeadDim, one token's KV span.

    private readonly Stack<int> _free;              // Available physical block ids.
    private readonly Dictionary<int, List<int>> _blockTables = new();
    private readonly Dictionary<int, int> _lengths = new();
    private bool _disposed;

    public DevicePagedKVCache(IDirectGpuBackend backend, int maxBlocks, int blockSize, int heads, int headDim)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (maxBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(maxBlocks));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (heads <= 0) throw new ArgumentOutOfRangeException(nameof(heads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));

        MaxBlocks = maxBlocks;
        BlockSize = blockSize;
        Heads = heads;
        HeadDim = headDim;
        _stepStride = heads * headDim;

        int poolLen = maxBlocks * blockSize * heads * headDim;
        KeyBlocks = backend.AllocateBuffer(poolLen);
        ValueBlocks = backend.AllocateBuffer(poolLen);

        _free = new Stack<int>(maxBlocks);
        // Push in reverse so the first allocation hands out block id 0.
        for (int i = maxBlocks - 1; i >= 0; i--) _free.Push(i);
    }

    /// <summary>Number of physical blocks currently allocated.</summary>
    public int AllocatedBlocks => MaxBlocks - _free.Count;

    /// <summary>Current token length of sequence <paramref name="seqId"/>.</summary>
    public int GetLength(int seqId) => _lengths.TryGetValue(seqId, out var l) ? l : 0;

    /// <summary>Read-only view of the block table for <paramref name="seqId"/> (physical block ids).</summary>
    public IReadOnlyList<int> GetBlockTable(int seqId)
        => _blockTables.TryGetValue(seqId, out var t) ? t : (IReadOnlyList<int>)Array.Empty<int>();

    /// <summary>
    /// Append <paramref name="newLen"/> new (K, V) tokens for <paramref name="seqId"/>, already resident
    /// in device buffers of shape <c>[newLen, Heads, HeadDim]</c>. New physical blocks are allocated on
    /// demand as the sequence crosses block boundaries; each maximal run that lands contiguously inside a
    /// single block is copied device-to-device in one shot (no host round-trip).
    /// </summary>
    public void Append(int seqId, IGpuBuffer newKeys, IGpuBuffer newValues, int newLen)
    {
        ThrowIfDisposed();
        if (newKeys is null) throw new ArgumentNullException(nameof(newKeys));
        if (newValues is null) throw new ArgumentNullException(nameof(newValues));
        if (newLen < 0) throw new ArgumentOutOfRangeException(nameof(newLen));
        if (newLen == 0) return;

        if (!_blockTables.TryGetValue(seqId, out var table))
        {
            table = new List<int>();
            _blockTables[seqId] = table;
            _lengths[seqId] = 0;
        }

        int len = _lengths[seqId];
        // Atomic exhaustion check: reserve all blocks up front so a mid-append shortfall can't leave the
        // sequence partially extended (blocks popped + data copied but _lengths not advanced).
        int blocksNeeded = (len + newLen + BlockSize - 1) / BlockSize - (len + BlockSize - 1) / BlockSize;
        if (blocksNeeded > _free.Count)
            throw new InvalidOperationException(
                $"Device paged KV cache exhausted: appending {newLen} token(s) needs {blocksNeeded} more " +
                $"block(s) but only {_free.Count} of {MaxBlocks} are free.");

        int i = 0;
        while (i < newLen)
        {
            int posInBlock = len % BlockSize;
            if (posInBlock == 0)
            {
                table.Add(_free.Pop()); // guaranteed available by the up-front reservation check
            }
            int blockId = table[table.Count - 1];
            int runTokens = Math.Min(BlockSize - posInBlock, newLen - i);
            int srcOffset = i * _stepStride;
            int dstOffset = (blockId * BlockSize + posInBlock) * _stepStride;
            int runElems = runTokens * _stepStride;
            _backend.Copy(newKeys, srcOffset, KeyBlocks, dstOffset, runElems);
            _backend.Copy(newValues, srcOffset, ValueBlocks, dstOffset, runElems);
            i += runTokens;
            len += runTokens;
        }
        _lengths[seqId] = len;
    }

    /// <summary>
    /// Convenience overload that uploads host K/V (flat <c>[newLen*Heads*HeadDim]</c>) to a temporary
    /// device buffer, then appends. Prefer the device-buffer overload on the hot path.
    /// </summary>
    public void Append(int seqId, float[] newKeys, float[] newValues)
    {
        ThrowIfDisposed();
        if (newKeys is null) throw new ArgumentNullException(nameof(newKeys));
        if (newValues is null) throw new ArgumentNullException(nameof(newValues));
        if (newKeys.Length % _stepStride != 0 || newValues.Length != newKeys.Length)
            throw new ArgumentException($"K/V length must be a multiple of Heads*HeadDim ({_stepStride}) and equal.");
        int newLen = newKeys.Length / _stepStride;
        if (newLen == 0) return;

        var kTmp = _backend.AllocateBuffer(newKeys);
        var vTmp = _backend.AllocateBuffer(newValues);
        try { Append(seqId, kTmp, vTmp, newLen); }
        finally { kTmp.Dispose(); vTmp.Dispose(); }
    }

    /// <summary>
    /// Upload the block table for <paramref name="seqId"/> as an int device buffer of physical block ids,
    /// ready to pass to the paged-attention kernels. Caller owns the returned buffer and disposes it.
    /// </summary>
    public IGpuBuffer GetBlockTableBuffer(int seqId)
    {
        ThrowIfDisposed();
        var table = _blockTables.TryGetValue(seqId, out var t) ? t : null;
        int count = table?.Count ?? 0;
        var ids = new int[Math.Max(count, 1)]; // avoid zero-length device allocation
        for (int i = 0; i < count; i++) ids[i] = table![i];
        return _backend.AllocateIntBuffer(ids);
    }

    /// <summary>
    /// Release every physical block owned by <paramref name="seqId"/> back to the free list (unless the
    /// block is shared — see <see cref="ShareBlocks"/>). The sequence is removed from the block table map.
    /// </summary>
    public void Free(int seqId)
    {
        ThrowIfDisposed();
        if (_blockTables.TryGetValue(seqId, out var table))
        {
            foreach (var blockId in table) _free.Push(blockId);
            _blockTables.Remove(seqId);
            _lengths.Remove(seqId);
        }
    }

    /// <summary>
    /// Share the first <paramref name="prefixLen"/> tokens of <paramref name="sourceSeqId"/> with
    /// <paramref name="targetSeqId"/> — the target's block table points at the source's first blocks
    /// (prefix-dedup for long shared system prompts).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This mirrors the CPU <c>PagedKVCache</c> reference's simple <b>non-refcounted</b> contract, which
    /// imposes two caller obligations:
    /// </para>
    /// <list type="bullet">
    /// <item><description><b>Read-only reuse only.</b> Shared blocks are aliased, not copy-on-write:
    /// appending to <paramref name="targetSeqId"/> after sharing writes through into the source's blocks
    /// (and, when <paramref name="prefixLen"/> is not a multiple of <see cref="BlockSize"/>, into the
    /// partially-filled last shared block). Treat a shared prefix as immutable, or only continue the
    /// sequence that owns it.</description></item>
    /// <item><description><b>Free a shared block once.</b> Because there is no refcount, calling
    /// <see cref="Free"/> on both the source and a sharer returns the same physical block ids to the free
    /// list twice — a later allocation would hand the same block to two sequences. Free only one holder of
    /// a shared prefix (or drop refcounting in a future revision if independent lifetimes are needed).
    /// </description></item>
    /// </list>
    /// </remarks>
    public void ShareBlocks(int sourceSeqId, int targetSeqId, int prefixLen)
    {
        ThrowIfDisposed();
        if (!_blockTables.TryGetValue(sourceSeqId, out var src))
            throw new ArgumentException("Source sequence has no blocks.", nameof(sourceSeqId));
        if (_blockTables.ContainsKey(targetSeqId))
            throw new ArgumentException("Target sequence already has blocks.", nameof(targetSeqId));
        int srcLen = _lengths[sourceSeqId];
        if (prefixLen < 0 || prefixLen > srcLen)
            throw new ArgumentOutOfRangeException(nameof(prefixLen));
        int blocksNeeded = (prefixLen + BlockSize - 1) / BlockSize;
        var targetTable = new List<int>(blocksNeeded);
        for (int i = 0; i < blocksNeeded; i++) targetTable.Add(src[i]);
        _blockTables[targetSeqId] = targetTable;
        _lengths[targetSeqId] = prefixLen;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(DevicePagedKVCache));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        KeyBlocks.Dispose();
        ValueBlocks.Dispose();
    }
}
