using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Pre-allocated KV cache for autoregressive attention —
/// <c>[maxBatch, maxSeq, heads, headDim]</c> keys + values plus a
/// per-batch length cursor. <c>Append</c> writes new tokens at the
/// current cursor, <c>Slice</c> returns the live prefix the attention
/// kernel consumes.
///
/// <para><b>Why a first-class primitive:</b> decode-time attention
/// reads <i>every past key and value</i> at each step. Re-allocating
/// and re-copying the whole past for every token is O(seq²) — the KV
/// cache makes it O(seq) by writing new entries in place. The cache
/// also lets <see cref="FusedAttention{T}.Forward"/> with
/// <see cref="FlashAttentionConfig.QueryOffset"/> work unchanged: the
/// consumer passes a length-1 query plus a <c>[maxSeq, …]</c> K/V
/// slice.</para>
///
/// <para>This is the dense contiguous variant. The block-based paged
/// layout used by vLLM ships as <see cref="PagedKVCache{T}"/>.</para>
/// </summary>
public sealed class KVCache<T>
    where T : unmanaged
{
    private readonly Tensor<T> _keys;
    private readonly Tensor<T> _values;
    private readonly int[] _lengths; // length per batch row

    public int MaxBatch { get; }
    public int MaxSeq { get; }
    public int Heads { get; }
    public int HeadDim { get; }

    /// <summary>Backing K tensor. Shape <c>[maxBatch, maxSeq, heads, headDim]</c>.</summary>
    public Tensor<T> Keys => _keys;
    /// <summary>Backing V tensor. Shape <c>[maxBatch, maxSeq, heads, headDim]</c>.</summary>
    public Tensor<T> Values => _values;

    public KVCache(int maxBatch, int maxSeq, int heads, int headDim)
    {
        if (maxBatch <= 0) throw new ArgumentOutOfRangeException(nameof(maxBatch));
        if (maxSeq <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeq));
        if (heads <= 0) throw new ArgumentOutOfRangeException(nameof(heads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        MaxBatch = maxBatch; MaxSeq = maxSeq; Heads = heads; HeadDim = headDim;
        _keys = new Tensor<T>(new[] { maxBatch, maxSeq, heads, headDim });
        _values = new Tensor<T>(new[] { maxBatch, maxSeq, heads, headDim });
        _lengths = new int[maxBatch];
    }

    /// <summary>Current sequence length at <paramref name="batchIdx"/>.</summary>
    public int GetLength(int batchIdx)
    {
        if ((uint)batchIdx >= (uint)MaxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));
        return _lengths[batchIdx];
    }

    /// <summary>
    /// Append <paramref name="newLen"/> new (K, V) tokens to batch row
    /// <paramref name="batchIdx"/>. Both input tensors must have shape
    /// <c>[newLen, heads, headDim]</c>.
    /// </summary>
    public void Append(int batchIdx, Tensor<T> newKeys, Tensor<T> newValues)
    {
        if ((uint)batchIdx >= (uint)MaxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));
        if (newKeys is null) throw new ArgumentNullException(nameof(newKeys));
        if (newValues is null) throw new ArgumentNullException(nameof(newValues));
        if (newKeys.Rank != 3 || newKeys._shape[1] != Heads || newKeys._shape[2] != HeadDim)
            throw new ArgumentException(
                $"newKeys must be [newLen, {Heads}, {HeadDim}]; got [{string.Join(", ", newKeys._shape)}].",
                nameof(newKeys));
        if (newValues.Rank != 3 || newValues._shape[1] != Heads || newValues._shape[2] != HeadDim)
            throw new ArgumentException(
                $"newValues must match newKeys shape; got [{string.Join(", ", newValues._shape)}].",
                nameof(newValues));
        int newLen = newKeys._shape[0];
        if (newValues._shape[0] != newLen)
            throw new ArgumentException("newKeys and newValues must agree on newLen.", nameof(newValues));

        int start = _lengths[batchIdx];
        if (start + newLen > MaxSeq)
            throw new InvalidOperationException(
                $"KVCache overflow: batch {batchIdx} length {start} + {newLen} exceeds MaxSeq {MaxSeq}.");

        // Copy into the [start, start+newLen) slot of the batch row.
        int stepStride = Heads * HeadDim;
        var kSrc = newKeys.AsSpan();
        var vSrc = newValues.AsSpan();
        var kDst = _keys.AsWritableSpan();
        var vDst = _values.AsWritableSpan();
        int dstBase = (batchIdx * MaxSeq + start) * stepStride;
        kSrc.CopyTo(kDst.Slice(dstBase, newLen * stepStride));
        vSrc.CopyTo(vDst.Slice(dstBase, newLen * stepStride));
        _lengths[batchIdx] = start + newLen;
    }

    /// <summary>
    /// Reset the length of batch row <paramref name="batchIdx"/> to zero.
    /// Does not wipe the underlying buffer — just drops the cursor.
    /// </summary>
    public void Reset(int batchIdx)
    {
        if ((uint)batchIdx >= (uint)MaxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));
        _lengths[batchIdx] = 0;
    }

    /// <summary>Reset every batch row's length to zero.</summary>
    public void ResetAll()
    {
        for (int i = 0; i < _lengths.Length; i++) _lengths[i] = 0;
    }

    /// <summary>
    /// Returns the live K / V slices for <paramref name="batchIdx"/> —
    /// shape <c>[currentLen, heads, headDim]</c> each. These are fresh
    /// tensors (contiguous copies) since Tensor slicing doesn't
    /// currently support arbitrary rank-reducing views.
    /// </summary>
    public (Tensor<T> K, Tensor<T> V) Slice(int batchIdx)
    {
        if ((uint)batchIdx >= (uint)MaxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));
        int len = _lengths[batchIdx];
        int stepStride = Heads * HeadDim;
        var shape = new[] { len, Heads, HeadDim };
        var k = new Tensor<T>(shape);
        var v = new Tensor<T>(shape);
        if (len > 0)
        {
            int srcBase = batchIdx * MaxSeq * stepStride;
            _keys.AsSpan().Slice(srcBase, len * stepStride).CopyTo(k.AsWritableSpan());
            _values.AsSpan().Slice(srcBase, len * stepStride).CopyTo(v.AsWritableSpan());
        }
        return (k, v);
    }
}
