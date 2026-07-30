// Copyright (c) AiDotNet. All rights reserved.
// On-device paged KV cache (P1): the GPU-resident sibling of the CPU
// AiDotNet.Tensors.Engines.Autodiff.PagedKVCache reference.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// On-device block-based KV cache à la vLLM's PagedAttention. The physical
/// block pool lives entirely in GPU memory as two flat device buffers
/// (<see cref="KeyBlocks"/> / <see cref="ValueBlocks"/>, laid out
/// <c>[MaxBlocks, BlockSize, Heads, HeadDim]</c>); per-sequence block-table
/// bookkeeping, the free list, and per-block reference counts stay host-side,
/// exactly as vLLM's block manager does. Appends copy device-to-device into the
/// pool (no host round-trip on the resident path) and the block table is
/// uploaded on demand as an int buffer that feeds straight into the
/// paged-attention kernels (<c>PagedAttentionDecode</c> / <c>Prefill</c> and
/// their GQA variants) on any backend.
///
/// <para>Sharing is reference-counted with copy-on-write: <see cref="ShareBlocks"/>
/// aliases a prefix and bumps the shared blocks' refcounts; <see cref="Free"/>
/// only returns a block to the pool when its last holder releases it; and an
/// <see cref="Append"/> that would write into a shared partially-filled tail
/// block first copies it to a private block. This makes independent sequence
/// lifetimes safe (no double-free, no cross-sequence corruption).</para>
///
/// <para>Backend-agnostic — it only uses <see cref="IDirectGpuBackend"/> buffer
/// primitives (AllocateBuffer, AllocateIntBuffer, device Copy at offsets), so it
/// runs on all six GPU backends. Results are validated against the same
/// standard-attention oracle as the CPU <c>PagedKVCache&lt;float&gt;</c>.</para>
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
    private readonly Dictionary<int, int> _refCount = new(); // physical block id -> holders
    private bool _disposed;

    public DevicePagedKVCache(IDirectGpuBackend backend, int maxBlocks, int blockSize, int heads, int headDim)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (maxBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(maxBlocks));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (heads <= 0) throw new ArgumentOutOfRangeException(nameof(heads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));

        // Pool size is computed in 64-bit and range-checked: the flat buffer is addressed with int
        // element offsets, so the whole pool must fit in a positive int to stay copy-safe.
        long stepStride = (long)heads * headDim;
        long poolLen = (long)maxBlocks * blockSize * stepStride;
        if (poolLen > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(maxBlocks),
                $"Pool size {poolLen} (maxBlocks*blockSize*heads*headDim) exceeds int.MaxValue; " +
                "reduce maxBlocks/blockSize/headDim.");

        MaxBlocks = maxBlocks;
        BlockSize = blockSize;
        Heads = heads;
        HeadDim = headDim;
        _stepStride = (int)stepStride;

        // Allocate into locals and dispose the first pool if the second allocation throws, so a
        // partially-constructed cache never leaks a native buffer.
        var keyBlocks = backend.AllocateBuffer((int)poolLen);
        try
        {
            ValueBlocks = backend.AllocateBuffer((int)poolLen);
        }
        catch
        {
            keyBlocks.Dispose();
            throw;
        }
        KeyBlocks = keyBlocks;

        _free = new Stack<int>(maxBlocks);
        // Push in reverse so the first allocation hands out block id 0.
        for (int i = maxBlocks - 1; i >= 0; i--) _free.Push(i);
    }

    /// <summary>Number of physical blocks currently allocated (not on the free list).</summary>
    public int AllocatedBlocks => MaxBlocks - _free.Count;

    /// <summary>Current token length of sequence <paramref name="seqId"/>.</summary>
    public int GetLength(int seqId) => _lengths.TryGetValue(seqId, out var l) ? l : 0;

    /// <summary>Immutable snapshot of the block table for <paramref name="seqId"/> (physical block ids).</summary>
    public IReadOnlyList<int> GetBlockTable(int seqId)
        => _blockTables.TryGetValue(seqId, out var t) ? t.ToArray() : Array.Empty<int>();

    /// <summary>
    /// Append <paramref name="newLen"/> new (K, V) tokens for <paramref name="seqId"/>, already resident
    /// in device buffers of shape <c>[newLen, Heads, HeadDim]</c>. New physical blocks are allocated on
    /// demand as the sequence crosses block boundaries; each maximal run that lands contiguously inside a
    /// single block is copied device-to-device in one shot. If the sequence's current tail block is shared
    /// (refcount &gt; 1) it is copied to a private block first (copy-on-write). The operation is atomic:
    /// on exhaustion or a failed device copy, all reservations made in this call are rolled back and the
    /// sequence is left exactly as it was.
    /// </summary>
    public void Append(int seqId, IGpuBuffer newKeys, IGpuBuffer newValues, int newLen)
    {
        ThrowIfDisposed();
        if (newKeys is null) throw new ArgumentNullException(nameof(newKeys));
        if (newValues is null) throw new ArgumentNullException(nameof(newValues));
        if (newLen < 0) throw new ArgumentOutOfRangeException(nameof(newLen));
        if (newLen == 0) return;

        // Validate source capacity before any device copy: IGpuBuffer exposes only size metadata, and
        // some backends forward Copy straight to native calls, so an oversized append would read OOB.
        long required = (long)newLen * _stepStride;
        if (newKeys.Size < required || newValues.Size < required)
            throw new ArgumentException(
                $"newKeys/newValues must each hold at least newLen*Heads*HeadDim = {required} elements " +
                $"(got {newKeys.Size}/{newValues.Size}).");

        if (!_blockTables.TryGetValue(seqId, out var table))
        {
            table = new List<int>();
            _blockTables[seqId] = table;
            _lengths[seqId] = 0;
        }

        int len = _lengths[seqId];

        // Capacity: the whole sequence must fit the pool.
        if ((long)len + newLen > (long)MaxBlocks * BlockSize)
            throw new InvalidOperationException(
                $"Device paged KV cache: sequence would reach {(long)len + newLen} tokens, exceeding " +
                $"pool capacity {(long)MaxBlocks * BlockSize} (MaxBlocks*BlockSize).");

        int posInBlock0 = len % BlockSize;
        bool needCow = posInBlock0 != 0 && table.Count > 0 && _refCount[table[table.Count - 1]] > 1;
        int growthBlocks = (len + newLen + BlockSize - 1) / BlockSize - (len + BlockSize - 1) / BlockSize;
        int needed = growthBlocks + (needCow ? 1 : 0);
        if (needed > _free.Count)
            throw new InvalidOperationException(
                $"Device paged KV cache exhausted: appending {newLen} token(s) needs {needed} more " +
                $"block(s) but only {_free.Count} of {MaxBlocks} are free.");

        // Track everything mutated so the whole append can roll back atomically on any failure.
        int originalTableCount = table.Count;
        var reserved = new List<int>();      // blocks popped from the free list this call
        int cowIndex = -1, cowOldBlock = -1; // tail block replaced by copy-on-write, if any

        try
        {
            if (needCow)
            {
                int oldBlk = table[table.Count - 1];
                int nb = _free.Pop();
                reserved.Add(nb);
                int copyElems = posInBlock0 * _stepStride;
                int oldBase = oldBlk * BlockSize * _stepStride;
                int nbBase = nb * BlockSize * _stepStride;
                _backend.Copy(KeyBlocks, oldBase, KeyBlocks, nbBase, copyElems);
                _backend.Copy(ValueBlocks, oldBase, ValueBlocks, nbBase, copyElems);
                _refCount[oldBlk]--;   // release this sequence's reference to the shared block
                _refCount[nb] = 1;
                cowIndex = table.Count - 1;
                cowOldBlock = oldBlk;
                table[cowIndex] = nb;
            }

            int i = 0;
            while (i < newLen)
            {
                int posInBlock = len % BlockSize;
                if (posInBlock == 0)
                {
                    int nb = _free.Pop();
                    reserved.Add(nb);
                    table.Add(nb);
                    _refCount[nb] = 1;
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
        catch
        {
            // Roll back: drop growth blocks appended this call, undo the COW swap, and return every
            // reserved physical block to the free list. _lengths was not yet advanced.
            if (table.Count > originalTableCount)
                table.RemoveRange(originalTableCount, table.Count - originalTableCount);
            if (cowIndex >= 0)
            {
                table[cowIndex] = cowOldBlock;
                _refCount[cowOldBlock]++; // undo the decrement
            }
            foreach (var b in reserved) { _refCount.Remove(b); _free.Push(b); }
            throw;
        }
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

        // Dispose the first temp buffer even if the second allocation throws.
        var kTmp = _backend.AllocateBuffer(newKeys);
        IGpuBuffer? vTmp = null;
        try
        {
            vTmp = _backend.AllocateBuffer(newValues);
            Append(seqId, kTmp, vTmp, newLen);
        }
        finally
        {
            kTmp.Dispose();
            vTmp?.Dispose();
        }
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
    /// Release sequence <paramref name="seqId"/>. Each of its physical blocks is returned to the free
    /// list only when its reference count reaches zero, so freeing one holder of a shared prefix leaves
    /// the blocks live for the others.
    /// </summary>
    public void Free(int seqId)
    {
        ThrowIfDisposed();
        if (_blockTables.TryGetValue(seqId, out var table))
        {
            foreach (var blockId in table)
            {
                if (--_refCount[blockId] <= 0)
                {
                    _refCount.Remove(blockId);
                    _free.Push(blockId);
                }
            }
            _blockTables.Remove(seqId);
            _lengths.Remove(seqId);
        }
    }

    /// <summary>
    /// Share the first <paramref name="prefixLen"/> tokens of <paramref name="sourceSeqId"/> with
    /// <paramref name="targetSeqId"/> — the target's block table points at the source's first blocks
    /// (prefix-dedup for long shared system prompts), bumping each shared block's reference count.
    /// </summary>
    /// <remarks>
    /// Sharing is safe for independent lifetimes: <see cref="Free"/> is reference-counted (a shared block
    /// is returned to the pool only when its last holder frees it), and a later <see cref="Append"/> to
    /// either sequence copies a shared partially-filled tail block to a private block before writing
    /// (copy-on-write), so continuations never corrupt the other sequence.
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
        for (int i = 0; i < blocksNeeded; i++)
        {
            int blk = src[i];
            targetTable.Add(blk);
            _refCount[blk]++;
        }
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
