using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Block-based KV cache à la vLLM's PagedAttention — the cache memory
/// is a pool of fixed-size blocks (block size 16/32/64 tokens typical),
/// and each sequence carries a <i>block table</i> mapping logical
/// positions to physical block ids.
///
/// <para>Benefit: separate sequences running concurrently can share
/// block storage (prompt prefix deduplication) or have their blocks
/// freed independently when a sequence ends, without fragmenting the
/// cache. Virtual-memory-style paging for the KV cache.</para>
///
/// <para>This implementation ships the CPU reference: an array-backed
/// block pool, a per-sequence block list, append + materialize APIs.
/// The GPU variant follows the same contract with on-device buffers
/// and a cudaMemcpy fan-out on materialize.</para>
/// </summary>
public sealed class PagedKVCache<T>
    where T : unmanaged
{
    /// <summary>Number of tokens per block — classic vLLM default is 16.</summary>
    public int BlockSize { get; }

    public int MaxBlocks { get; }
    public int Heads { get; }
    public int HeadDim { get; }

    // Physical block pool — contiguous [MaxBlocks, BlockSize, Heads, HeadDim]
    // for keys and values respectively. The block id is the first axis.
    private readonly Tensor<T> _keyBlocks;
    private readonly Tensor<T> _valueBlocks;

    // Free list: stack of available block ids.
    private readonly Stack<int> _free;

    // Block table per sequence id → list of physical block ids, in order.
    private readonly Dictionary<int, List<int>> _blockTables = new();
    // Token count per sequence.
    private readonly Dictionary<int, int> _lengths = new();

    public PagedKVCache(int maxBlocks, int blockSize, int heads, int headDim)
    {
        if (maxBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(maxBlocks));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (heads <= 0) throw new ArgumentOutOfRangeException(nameof(heads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        MaxBlocks = maxBlocks;
        BlockSize = blockSize;
        Heads = heads;
        HeadDim = headDim;
        _keyBlocks = new Tensor<T>(new[] { maxBlocks, blockSize, heads, headDim });
        _valueBlocks = new Tensor<T>(new[] { maxBlocks, blockSize, heads, headDim });
        _free = new Stack<int>(maxBlocks);
        // Initialize free-list with every block id in reverse so
        // allocations start at id 0.
        for (int i = maxBlocks - 1; i >= 0; i--) _free.Push(i);
    }

    /// <summary>Number of physical blocks currently allocated.</summary>
    public int AllocatedBlocks => MaxBlocks - _free.Count;

    /// <summary>Current token length of sequence <paramref name="seqId"/>.</summary>
    public int GetLength(int seqId) => _lengths.TryGetValue(seqId, out var l) ? l : 0;

    /// <summary>Read-only view of the block table for <paramref name="seqId"/>.</summary>
    public IReadOnlyList<int> GetBlockTable(int seqId)
        => _blockTables.TryGetValue(seqId, out var t) ? t : Array.Empty<int>();

    /// <summary>
    /// Append <paramref name="newLen"/> new (K, V) tokens for
    /// <paramref name="seqId"/>. Allocates new physical blocks on
    /// demand as the sequence crosses block boundaries.
    /// </summary>
    public void Append(int seqId, Tensor<T> newKeys, Tensor<T> newValues)
    {
        if (newKeys is null) throw new ArgumentNullException(nameof(newKeys));
        if (newValues is null) throw new ArgumentNullException(nameof(newValues));
        if (newKeys.Rank != 3 || newKeys._shape[1] != Heads || newKeys._shape[2] != HeadDim)
            throw new ArgumentException(
                $"newKeys must be [newLen, {Heads}, {HeadDim}].", nameof(newKeys));
        int newLen = newKeys._shape[0];
        if (newValues._shape[0] != newLen || newValues._shape[1] != Heads || newValues._shape[2] != HeadDim)
            throw new ArgumentException("newValues shape mismatch.", nameof(newValues));

        if (!_blockTables.TryGetValue(seqId, out var table))
        {
            table = new List<int>();
            _blockTables[seqId] = table;
            _lengths[seqId] = 0;
        }

        int len = _lengths[seqId];
        int stepStride = Heads * HeadDim;
        var kSrc = newKeys.AsSpan();
        var vSrc = newValues.AsSpan();
        var kDst = _keyBlocks.AsWritableSpan();
        var vDst = _valueBlocks.AsWritableSpan();

        for (int i = 0; i < newLen; i++)
        {
            int posInBlock = len % BlockSize;
            if (posInBlock == 0)
            {
                // Cross block boundary — allocate a new physical block.
                if (_free.Count == 0)
                    throw new InvalidOperationException(
                        $"Paged KV cache exhausted: all {MaxBlocks} blocks in use.");
                table.Add(_free.Pop());
            }
            int blockId = table[table.Count - 1];
            int blockBase = (blockId * BlockSize + posInBlock) * stepStride;
            kSrc.Slice(i * stepStride, stepStride).CopyTo(kDst.Slice(blockBase, stepStride));
            vSrc.Slice(i * stepStride, stepStride).CopyTo(vDst.Slice(blockBase, stepStride));
            len++;
        }
        _lengths[seqId] = len;
    }

    /// <summary>
    /// Materialize the live K / V slice for <paramref name="seqId"/>
    /// into contiguous tensors of shape <c>[currentLen, heads,
    /// headDim]</c>. Walks the block table and copies block-by-block.
    /// </summary>
    public (Tensor<T> K, Tensor<T> V) Materialize(int seqId)
    {
        int len = GetLength(seqId);
        var shape = new[] { len, Heads, HeadDim };
        var k = new Tensor<T>(shape);
        var v = new Tensor<T>(shape);
        if (len == 0) return (k, v);

        var table = _blockTables[seqId];
        int stepStride = Heads * HeadDim;
        var kSrc = _keyBlocks.AsSpan();
        var vSrc = _valueBlocks.AsSpan();
        var kDst = k.AsWritableSpan();
        var vDst = v.AsWritableSpan();
        int copied = 0;
        for (int b = 0; b < table.Count && copied < len; b++)
        {
            int blockId = table[b];
            int tokensInBlock = Math.Min(BlockSize, len - copied);
            int srcBase = blockId * BlockSize * stepStride;
            int dstBase = copied * stepStride;
            kSrc.Slice(srcBase, tokensInBlock * stepStride).CopyTo(kDst.Slice(dstBase));
            vSrc.Slice(srcBase, tokensInBlock * stepStride).CopyTo(vDst.Slice(dstBase));
            copied += tokensInBlock;
        }
        return (k, v);
    }

    /// <summary>
    /// Release every physical block owned by <paramref name="seqId"/>
    /// back to the free list. The sequence is removed from the block
    /// table map — further <see cref="GetLength"/> calls return 0.
    /// </summary>
    public void Free(int seqId)
    {
        if (_blockTables.TryGetValue(seqId, out var table))
        {
            foreach (var blockId in table) _free.Push(blockId);
            _blockTables.Remove(seqId);
            _lengths.Remove(seqId);
        }
    }

    /// <summary>
    /// Share the first <paramref name="prefixLen"/> tokens of
    /// <paramref name="sourceSeqId"/> with <paramref name="targetSeqId"/>
    /// — the target's block table points at the source's first blocks.
    /// The prefix-dedup pattern vLLM exploits to avoid re-computing KV
    /// for long shared system prompts.
    /// </summary>
    public void ShareBlocks(int sourceSeqId, int targetSeqId, int prefixLen)
    {
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
}
