using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> ScaledDotProductAttention<T>(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, Tensor<bool>? mask = null, double? scale = null,
        bool isCausal = false, double softcap = 0.0)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            throw new ArgumentException("query, key and value must be [batch, heads, sequence, dim].");
        int batch = query._shape[0], heads = query._shape[1], seqQ = query._shape[2], headDim = query._shape[3];
        int seqK = key._shape[2], valueDim = value._shape[3];
        if (key._shape[0] != batch || key._shape[1] != heads || key._shape[3] != headDim)
            throw new ArgumentException("key must be [batch, heads, keySequence, headDim] matching query.", nameof(key));
        if (value._shape[0] != batch || value._shape[1] != heads || value._shape[2] != seqK)
            throw new ArgumentException("value must be [batch, heads, keySequence, valueDim] matching key.", nameof(value));
        if (double.IsNaN(softcap) || softcap < 0) throw new ArgumentOutOfRangeException(nameof(softcap), "softcap must be >= 0.");
        double scaleValue = scale ?? 1.0 / Math.Sqrt(headDim);
        if (mask is not null)
        {
            ValidateAttentionMask(mask, batch, heads, seqQ, seqK);
            // As the weights-returning overload: a boolean mask is a heterogeneous input an inference capture rejects.
            GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        }

        // The fused kernels are host float32. Other element types and GPU engines run the materialized primitive
        // chain, which every engine differentiates on its own device.
        if (typeof(T) != typeof(float) || this is DirectGpuTensorEngine)
            return FusedAttentionDecomposed(query, key, value, mask, scaleValue, isCausal, softcap);

        var plan = new FusedAttentionPlan(batch, heads, seqQ, seqK, headDim, valueDim, scaleValue, softcap, isCausal, mask);
        var q = (Tensor<float>)(object)query;
        var k = (Tensor<float>)(object)key;
        var v = (Tensor<float>)(object)value;

        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            // Each replayed forward rewrites the node's log-sum-exp rows, which the same step's backward reads.
            var lse = new float[batch * heads * seqQ];
            var inputs = new[] { q, k, v };
            var outputShape = new[] { batch, heads, seqQ, valueDim };
            return (Tensor<T>)(object)scope.RecordVariadic(LazyNodeType.Custom, "ScaledDotProductAttentionFused", inputs,
                outputShape,
                (eng, output) =>
                {
                    if (output.IsContiguous && output._gpuBuffer is null && !output.HasPendingGpuData)
                    {
                        FusedAttentionForwardFloat(inputs[0], inputs[1], inputs[2], output, lse, plan);
                        output.IncrementVersion();
                    }
                    else
                    {
                        var staged = new Tensor<float>(outputShape);
                        FusedAttentionForwardFloat(inputs[0], inputs[1], inputs[2], staged, lse, plan);
                        DirectGpuTensorEngine.CopyResultInto(eng, staged, output);
                    }
                },
                FusedAttentionBackwardFloat, new object[] { lse, plan });
        }

        var result = new Tensor<float>(new[] { batch, heads, seqQ, valueDim });
        var lseEager = new float[batch * heads * seqQ];
        FusedAttentionForwardFloat(q, k, v, result, lseEager, plan);
        var resultT = (Tensor<T>)(object)result;
        DifferentiableOps.RecordIfActive("ScaledDotProductAttentionFused", resultT, new[] { query, key, value },
            (BackwardFunction<T>)(object)(BackwardFunction<float>)FusedAttentionBackwardFloat,
            new object[] { lseEager, plan });
        return resultT;
    }

    /// <summary>A mask must broadcast to [batch, heads, querySequence, keySequence] (right-aligned, extents 1 or full).</summary>
    private static void ValidateAttentionMask(Tensor<bool> mask, int batch, int heads, int seqQ, int seqK)
    {
        if (mask.Rank < 2 || mask.Rank > 4)
            throw new ArgumentException("mask must broadcast to [batch, heads, querySequence, keySequence].", nameof(mask));
        var full = new[] { batch, heads, seqQ, seqK };
        for (int d = 0; d < mask.Rank; d++)
        {
            int extent = mask._shape[mask.Rank - 1 - d], target = full[3 - d];
            if (extent != 1 && extent != target)
                throw new ArgumentException(
                    $"mask shape [{string.Join(", ", mask._shape)}] does not broadcast to [{string.Join(", ", full)}].", nameof(mask));
        }
    }

    /// <summary>
    /// The same function through recorded primitives (any element type, any engine): the mask (causal and supplied,
    /// broadcast) is expanded once and handed to the materialising attention, whose backward the tape already knows.
    /// </summary>
    private Tensor<T> FusedAttentionDecomposed<T>(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, Tensor<bool>? mask, double scale, bool isCausal, double softcap)
    {
        int batch = query._shape[0], heads = query._shape[1], seqQ = query._shape[2], seqK = key._shape[2];
        int valueDim = value._shape[3];
        Tensor<bool>? full = null;
        bool[]? emptyRows = null;
        if (mask is not null || isCausal)
        {
            var allowed = new bool[batch * heads * seqQ * seqK];
            var accessor = mask is null ? default : new MaskAccessor(mask);
            for (int b = 0, at = 0, row = 0; b < batch; b++)
                for (int h = 0; h < heads; h++)
                    for (int i = 0; i < seqQ; i++, row++)
                    {
                        bool any = false;
                        for (int j = 0; j < seqK; j++, at++)
                            any |= allowed[at] = (!isCausal || j <= i) && (mask is null || accessor.Allowed(b, h, i, j));
                        if (any) continue;
                        // A row with no allowed key outputs zeros, as the fused kernel does: give the softmax a finite
                        // (all-open) row and zero its output below, so it also passes no gradient instead of NaN.
                        for (int j = at - seqK; j < at; j++) allowed[j] = true;
                        (emptyRows ??= new bool[batch * heads * seqQ])[row] = true;
                    }
            full = new Tensor<bool>(allowed, new[] { batch, heads, seqQ, seqK });
        }
        var result = ScaledDotProductAttention(query, key, value, full, scale, out _, softcap);
        if (emptyRows is null) return result;
        var blocked = new bool[batch * heads * seqQ * valueDim];
        for (int row = 0; row < emptyRows.Length; row++)
            if (emptyRows[row])
                for (int d = 0; d < valueDim; d++) blocked[row * valueDim + d] = true;
        return TensorMaskedFill(result, new Tensor<bool>(blocked, new[] { batch, heads, seqQ, valueDim }),
            MathHelper.GetNumericOperations<T>().Zero);
    }

    /// <summary>Broadcast-aware reads of a boolean attention mask (stride 0 on broadcast axes).</summary>
    internal readonly struct MaskAccessor
    {
        private readonly bool[] _data;
        private readonly int _offset, _sb, _sh, _si, _sj;

        public MaskAccessor(Tensor<bool> mask)
        {
            // Row-major view of the mask's own extents (a copy only when it is a strided or offset view).
            _data = (mask.IsContiguous ? mask : mask.Contiguous()).GetFlattenedData();
            _offset = 0;
            var strides = new int[4];
            int stride = 1;
            for (int d = mask.Rank - 1; d >= 0; d--)
            {
                int axis = 4 - mask.Rank + d;
                strides[axis] = mask._shape[d] == 1 ? 0 : stride;
                stride *= mask._shape[d];
            }
            _sb = strides[0]; _sh = strides[1]; _si = strides[2]; _sj = strides[3];
        }

        public bool Allowed(int b, int h, int i, int j) => _data[_offset + b * _sb + h * _sh + i * _si + j * _sj];
    }

    /// <summary>Shape and options of one fused attention call, kept with its tape / graph node.</summary>
    internal sealed class FusedAttentionPlan
    {
        public FusedAttentionPlan(int batch, int heads, int seqQ, int seqK, int headDim, int valueDim, double scale,
            double softcap, bool causal, Tensor<bool>? mask)
        {
            Batch = batch; Heads = heads; SeqQ = seqQ; SeqK = seqK; HeadDim = headDim; ValueDim = valueDim;
            ScaleF = (float)scale; SoftcapF = (float)softcap; Causal = causal;
            Mask = mask is null ? null : new MaskAccessor(mask);
        }

        public int Batch { get; }
        public int Heads { get; }
        public int SeqQ { get; }
        public int SeqK { get; }
        public int HeadDim { get; }
        public int ValueDim { get; }
        public float ScaleF { get; }
        public float SoftcapF { get; }
        public bool Causal { get; }
        public MaskAccessor? Mask { get; }

        /// <summary>Keys row i can see at most: all, or j &lt;= i when causal (PyTorch's top-left is_causal).</summary>
        public int KeyLimit(int i) => Causal ? Math.Min(SeqK, i + 1) : SeqK;

        public bool Allowed(int b, int h, int i, int j) => (!Causal || j <= i) && (Mask is not { } m || m.Allowed(b, h, i, j));
    }

    /// <summary>Row-major [B, H, S, D] access to a (possibly strided) operand whose last axis is unit-stride.</summary>
    internal readonly struct AttentionOperand
    {
        public readonly float[] Data;
        public readonly int Offset, BatchStride, HeadStride, RowStride;

        public AttentionOperand(Tensor<float> tensor)
        {
            // Strided views (permutes of [B, S, H, D] projections, narrowed / sliced keys) are read in place; only a
            // view whose innermost axis is not unit-stride is copied (then once, row-major).
            var t = tensor._strides[3] == 1 ? tensor : tensor.Contiguous();
            var data = t.GetCpuBackingForStridedRead(out int offset);
            if (data is null)
            {
                t = t.Contiguous();
                data = t.GetFlattenedData();
                offset = 0;
            }
            Data = data; Offset = offset;
            BatchStride = t._strides[0]; HeadStride = t._strides[1]; RowStride = t._strides[2];
        }

        public int Base(int b, int h) => Offset + b * BatchStride + h * HeadStride;
    }

    /// <summary>
    /// out[b, h, i, :] = softmax_j(f(scale * q_i . k_j)) v_j over the allowed keys, f the optional soft-cap, with an
    /// online softmax over key tiles so no [seqQ, seqK] matrix exists; tiles every row of which is masked are skipped.
    /// Writes lse[(b*heads+h)*seqQ + i] for the backward; a row with no allowed key outputs zeros (lse = -inf).
    /// One task per (batch, head, query block): rows are independent, so the result does not depend on the split.
    /// </summary>
    internal static void FusedAttentionForwardFloat(
        Tensor<float> query, Tensor<float> key, Tensor<float> value, Tensor<float> output, float[] lse, FusedAttentionPlan p)
    {
        var q = new AttentionOperand(query);
        var k = new AttentionOperand(key);
        var v = new AttentionOperand(value);
        var oArr = output.GetCpuBackingForContiguousWrite(out int oOff)
            ?? throw new InvalidOperationException("ScaledDotProductAttention needs a host output buffer.");
        int br = Math.Min(AttentionCoreQueryBlock, p.SeqQ), queryBlocks = (p.SeqQ + br - 1) / br;
        int tasks = p.Batch * p.Heads * queryBlocks;
        long work = (long)p.Batch * p.Heads * p.SeqQ * p.SeqK * (p.HeadDim + p.ValueDim);
        CpuParallelSettings.ParallelForOrSerial(0, tasks, work,
            unit => FusedAttentionForwardBlock(q, k, v, oArr, oOff, lse, p, unit / queryBlocks, (unit % queryBlocks) * br),
            deterministicSafe: true);
    }

    // Applies the soft-cap and the mask to one score row in place: masked entries become -inf. Returns the row's max.
    private static float PrepareScoreRow(float[] sc, int off, int cols, FusedAttentionPlan p, int b, int h, int i, int j0)
    {
        float cap = p.SoftcapF, max = float.NegativeInfinity;
        int limit = Math.Min(cols, p.KeyLimit(i) - j0);
        for (int c = 0; c < cols; c++)
        {
            float s = sc[off + c];
            if (c >= limit || (p.Mask is { } m && !m.Allowed(b, h, i, j0 + c))) { sc[off + c] = float.NegativeInfinity; continue; }
            if (cap > 0f) { s = cap * (float)Math.Tanh(s / cap); sc[off + c] = s; }
            if (s > max) max = s;
        }
        return max;
    }

    // True when some row of the [rows, cols] tile at (i0, j0) may attend some key in it.
    private static bool TileHasAllowed(FusedAttentionPlan p, int b, int h, int i0, int rows, int j0, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            int limit = Math.Min(cols, p.KeyLimit(i0 + r) - j0);
            if (p.Mask is not { } m) { if (limit > 0) return true; continue; }
            for (int c = 0; c < limit; c++) if (m.Allowed(b, h, i0 + r, j0 + c)) return true;
        }
        return false;
    }

    private static void FusedAttentionForwardBlock(
        AttentionOperand q, AttentionOperand k, AttentionOperand v, float[] o, int oOff, float[] lse,
        FusedAttentionPlan p, int bh, int i0)
    {
        int b = bh / p.Heads, h = bh % p.Heads;
        int hd = p.HeadDim, vd = p.ValueDim;
        int qBase = q.Base(b, h), kBase = k.Base(b, h), vBase = v.Base(b, h), oBase = oOff + bh * p.SeqQ * vd;
        int br = Math.Min(AttentionCoreQueryBlock, p.SeqQ), bc = Math.Min(AttentionCoreKeyBlock, p.SeqK);
        var ws = AttentionScratch.ForThread();
        float[] qs = ws.Get(0, br * hd), kt = ws.Get(1, hd * bc), sc = ws.Get(2, br * bc);
        float[] acc = ws.Get(3, br * vd), rowMax = ws.Get(4, br), rowSum = ws.Get(5, br);
        int rows = Math.Min(br, p.SeqQ - i0);
        for (int r = 0; r < rows; r++)
        {
            ScaleCopy(q.Data, qBase + (i0 + r) * q.RowStride, qs, r * hd, hd, p.ScaleF);
            rowMax[r] = float.NegativeInfinity;
            rowSum[r] = 0f;
        }
        Array.Clear(acc, 0, rows * vd);
        int keyEnd = p.KeyLimit(i0 + rows - 1);
        for (int j0 = 0; j0 < keyEnd; j0 += bc)
        {
            int cols = Math.Min(bc, keyEnd - j0);
            if (!TileHasAllowed(p, b, h, i0, rows, j0, cols)) continue;   // a fully masked tile adds nothing
            TransposeKeyBlock(k.Data, kBase + j0 * k.RowStride, k.RowStride, kt, hd, cols, bc);
            TileGemm(qs, 0, hd, kt, 0, bc, sc, 0, bc, rows, cols, hd, accumulate: false);
            for (int r = 0; r < rows; r++)
            {
                int srow = r * bc;
                float blockMax = PrepareScoreRow(sc, srow, cols, p, b, h, i0 + r, j0);
                if (float.IsNegativeInfinity(blockMax)) { Array.Clear(sc, srow, cols); continue; }
                float newMax = Math.Max(rowMax[r], blockMax);
                float rescale = float.IsNegativeInfinity(rowMax[r]) ? 0f : (float)Math.Exp(rowMax[r] - newMax);
                ExpShifted(sc, srow, cols, newMax);   // masked (-inf) entries become 0
                float blockSum = 0f;
                for (int c = 0; c < cols; c++) blockSum += sc[srow + c];
                rowSum[r] = rowSum[r] * rescale + blockSum;
                rowMax[r] = newMax;
                if (rescale != 1f) Scale(acc, r * vd, vd, rescale);
            }
            TileGemm(sc, 0, bc, v.Data, vBase + j0 * v.RowStride, v.RowStride, acc, 0, vd, rows, vd, cols, accumulate: true);
        }
        int lseRow = bh * p.SeqQ + i0;
        for (int r = 0; r < rows; r++)
        {
            float sum = rowSum[r];
            int orow = oBase + (i0 + r) * vd;
            if (sum > 0f)
            {
                ScaleCopy(acc, r * vd, o, orow, vd, 1f / sum);
                lse[lseRow + r] = rowMax[r] + (float)Math.Log(sum);
            }
            else
            {
                Array.Clear(o, orow, vd);
                lse[lseRow + r] = float.NegativeInfinity;
            }
        }
    }

    /// <summary>
    /// Backward of the fused attention (FlashAttention-2 form). Each probability tile is rebuilt from the saved
    /// log-sum-exp; with s = f(raw) the soft-capped score, dS_raw = P * (dO V^T - delta) * f'(raw) where
    /// delta = rowsum(dO * O) and f'(raw) = 1 - tanh^2(raw / cap). Two passes, each split finely and race-free: one
    /// task per (batch, head, key block) owns that block's dK += dS_raw^T (scale Q) and dV += P^T dO, and one task per
    /// (batch, head, query block) owns its dQ = scale dS_raw K -- the tiles are recomputed rather than accumulated
    /// through atomics, so the result does not depend on the thread count.
    /// </summary>
    private static void FusedAttentionBackwardFloat(
        Tensor<float> gradOutput, Tensor<float>[] inputs, Tensor<float> output, object[] savedState, IEngine engine,
        Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var lse = (float[])savedState[0];
        var p = (FusedAttentionPlan)savedState[1];
        Tensor<float> query = inputs[0], key = inputs[1], value = inputs[2];
        bool needQ = DifferentiableOps.IsGradientRequired(query);
        bool needK = DifferentiableOps.IsGradientRequired(key);
        bool needV = DifferentiableOps.IsGradientRequired(value);
        if (!needQ && !needK && !needV) return;

        var q = new AttentionOperand(query);
        var k = new AttentionOperand(key);
        var v = new AttentionOperand(value);
        var oArr = ReadableBacking(output, out int oOff);
        var dOArr = ReadableBacking(gradOutput, out int dOOff);

        bool distinct = !ReferenceEquals(query, key) && !ReferenceEquals(query, value) && !ReferenceEquals(key, value);
        var dQ = GradTarget(needQ, query, grads, engine, distinct);
        var dK = GradTarget(needK, key, grads, engine, distinct);
        var dV = GradTarget(needV, value, grads, engine, distinct);
        if (distinct && dQ.Array is not null && (Overlap(dQ, dK) || Overlap(dQ, dV)))
            dQ = GradTarget(needQ, query, grads, engine, distinct: false);
        if (distinct && dK.Array is not null && Overlap(dK, dV))
            dK = GradTarget(needK, key, grads, engine, distinct: false);

        int vd = p.ValueDim, heads = p.Batch * p.Heads;
        int br = Math.Min(AttentionCoreQueryBlock, p.SeqQ), bc = Math.Min(AttentionCoreKeyBlock, p.SeqK);
        int queryBlocks = (p.SeqQ + br - 1) / br, keyBlocks = (p.SeqK + bc - 1) / bc;
        long rowWork = (long)p.SeqK * (p.HeadDim + p.ValueDim);

        // delta[bh, i] = dO_i . O_i, read by both passes.
        var delta = new float[heads * p.SeqQ];
        CpuParallelSettings.ParallelForOrSerial(0, heads, (long)heads * p.SeqQ * vd, bh =>
        {
            for (int i = 0; i < p.SeqQ; i++)
                delta[bh * p.SeqQ + i] = Dot(dOArr, dOOff + (bh * p.SeqQ + i) * vd, oArr, oOff + (bh * p.SeqQ + i) * vd, vd);
        }, deterministicSafe: true);

        if (dK.Array is not null || dV.Array is not null)
            CpuParallelSettings.ParallelForOrSerial(0, heads * keyBlocks, 2L * heads * p.SeqQ * rowWork,
                unit => FusedAttentionBackwardKeyBlock(q, k, v, dOArr, dOOff, lse, delta, p, unit / keyBlocks,
                    (unit % keyBlocks) * bc, dK, dV),
                deterministicSafe: true);
        if (dQ.Array is not null)
            CpuParallelSettings.ParallelForOrSerial(0, heads * queryBlocks, 2L * heads * p.SeqQ * rowWork,
                unit => FusedAttentionBackwardQueryBlock(q, k, v, dOArr, dOOff, lse, delta, p, unit / queryBlocks,
                    (unit % queryBlocks) * br, dQ),
                deterministicSafe: true);

        dQ.Commit(grads, query, engine);
        dK.Commit(grads, key, engine);
        dV.Commit(grads, value, engine);
    }

    // Rebuilds the [rows, cols] tile at (i0, j0): pr = P and ds = dS_raw (zero where masked). qs holds scale * Q rows,
    // kt / vt the transposed key / value block.
    private static void FusedAttentionGradientTile(
        float[] qs, float[] kt, float[] vt, float[] dO, int dOBase, float[] lse, float[] delta, FusedAttentionPlan p,
        int b, int h, int bh, int i0, int rows, int j0, int cols, float[] pr, float[] ds)
    {
        int hd = p.HeadDim, vd = p.ValueDim, bc = Math.Min(AttentionCoreKeyBlock, p.SeqK);
        float cap = p.SoftcapF;
        TileGemm(qs, 0, hd, kt, 0, bc, pr, 0, bc, rows, cols, hd, accumulate: false);              // raw scores
        TileGemm(dO, dOBase + i0 * vd, vd, vt, 0, bc, ds, 0, bc, rows, cols, vd, accumulate: false); // dP
        for (int r = 0; r < rows; r++)
        {
            int i = i0 + r, row = r * bc;
            float rowLse = lse[bh * p.SeqQ + i];
            int limit = Math.Min(cols, p.KeyLimit(i) - j0);
            if (float.IsNegativeInfinity(rowLse) || limit <= 0)
            {
                Array.Clear(pr, row, cols);
                Array.Clear(ds, row, cols);
                continue;
            }
            float d = delta[bh * p.SeqQ + i];
            for (int c = 0; c < cols; c++)
            {
                if (c >= limit || (p.Mask is { } m && !m.Allowed(b, h, i, j0 + c)))
                {
                    pr[row + c] = 0f;
                    ds[row + c] = 0f;
                    continue;
                }
                float raw = pr[row + c], s = raw, slope = 1f;
                if (cap > 0f)
                {
                    float t = (float)Math.Tanh(raw / cap);
                    s = cap * t;
                    slope = 1f - t * t;
                }
                float prob = (float)Math.Exp(s - rowLse);
                pr[row + c] = prob;
                ds[row + c] = prob * (ds[row + c] - d) * slope;
            }
        }
    }

    private static void FusedAttentionBackwardKeyBlock(
        AttentionOperand q, AttentionOperand k, AttentionOperand v, float[] dO, int dOOff, float[] lse, float[] delta,
        FusedAttentionPlan p, int bh, int j0, FusedGradTarget dK, FusedGradTarget dV)
    {
        int b = bh / p.Heads, h = bh % p.Heads;
        int hd = p.HeadDim, vd = p.ValueDim;
        int qBase = q.Base(b, h), kBase = k.Base(b, h), vBase = v.Base(b, h), dOBase = dOOff + bh * p.SeqQ * vd;
        int br = Math.Min(AttentionCoreQueryBlock, p.SeqQ), bc = Math.Min(AttentionCoreKeyBlock, p.SeqK);
        int cols = Math.Min(bc, p.SeqK - j0);
        var ws = AttentionScratch.ForThread();
        float[] qs = ws.Get(0, br * hd), kt = ws.Get(1, hd * bc), vt = ws.Get(2, vd * bc);
        float[] pr = ws.Get(3, br * bc), ds = ws.Get(4, br * bc);
        float[] dk = ws.Get(7, bc * hd), dv = ws.Get(8, bc * vd);
        Array.Clear(dk, 0, cols * hd);
        Array.Clear(dv, 0, cols * vd);
        TransposeKeyBlock(k.Data, kBase + j0 * k.RowStride, k.RowStride, kt, hd, cols, bc);
        TransposeKeyBlock(v.Data, vBase + j0 * v.RowStride, v.RowStride, vt, vd, cols, bc);
        // Causal: query rows before j0 see none of this block's keys.
        int iStart = p.Causal ? j0 / br * br : 0;
        for (int i0 = iStart; i0 < p.SeqQ; i0 += br)
        {
            int rows = Math.Min(br, p.SeqQ - i0);
            if (!TileHasAllowed(p, b, h, i0, rows, j0, cols)) continue;
            for (int r = 0; r < rows; r++) ScaleCopy(q.Data, qBase + (i0 + r) * q.RowStride, qs, r * hd, hd, p.ScaleF);
            FusedAttentionGradientTile(qs, kt, vt, dO, dOBase, lse, delta, p, b, h, bh, i0, rows, j0, cols, pr, ds);
            // P^T and dS^T are read in place (row stride 1, column stride bc): no transposed copies.
            TileGemm(pr, 0, 1, dO, dOBase + i0 * vd, vd, dv, 0, vd, cols, vd, rows, accumulate: true, aStep: bc);
            TileGemm(ds, 0, 1, qs, 0, hd, dk, 0, hd, cols, hd, rows, accumulate: true, aStep: bc);
        }
        if (dK.Array is not null)
            for (int c = 0; c < cols; c++)
                StoreRow(dk, c * hd, dK.Array, dK.Offset + (bh * p.SeqK + j0 + c) * hd, hd, 1f, dK.Overwrite);
        if (dV.Array is not null)
            for (int c = 0; c < cols; c++)
                StoreRow(dv, c * vd, dV.Array, dV.Offset + (bh * p.SeqK + j0 + c) * vd, vd, 1f, dV.Overwrite);
    }

    private static void FusedAttentionBackwardQueryBlock(
        AttentionOperand q, AttentionOperand k, AttentionOperand v, float[] dO, int dOOff, float[] lse, float[] delta,
        FusedAttentionPlan p, int bh, int i0, FusedGradTarget dQ)
    {
        int b = bh / p.Heads, h = bh % p.Heads;
        int hd = p.HeadDim, vd = p.ValueDim;
        int qBase = q.Base(b, h), kBase = k.Base(b, h), vBase = v.Base(b, h), dOBase = dOOff + bh * p.SeqQ * vd;
        int br = Math.Min(AttentionCoreQueryBlock, p.SeqQ), bc = Math.Min(AttentionCoreKeyBlock, p.SeqK);
        int rows = Math.Min(br, p.SeqQ - i0);
        var ws = AttentionScratch.ForThread();
        float[] qs = ws.Get(0, br * hd), kt = ws.Get(1, hd * bc), vt = ws.Get(2, vd * bc);
        float[] pr = ws.Get(3, br * bc), ds = ws.Get(4, br * bc), dq = ws.Get(5, br * hd);
        for (int r = 0; r < rows; r++) ScaleCopy(q.Data, qBase + (i0 + r) * q.RowStride, qs, r * hd, hd, p.ScaleF);
        Array.Clear(dq, 0, rows * hd);
        int keyEnd = p.KeyLimit(i0 + rows - 1);
        for (int j0 = 0; j0 < keyEnd; j0 += bc)
        {
            int cols = Math.Min(bc, keyEnd - j0);
            if (!TileHasAllowed(p, b, h, i0, rows, j0, cols)) continue;
            TransposeKeyBlock(k.Data, kBase + j0 * k.RowStride, k.RowStride, kt, hd, cols, bc);
            TransposeKeyBlock(v.Data, vBase + j0 * v.RowStride, v.RowStride, vt, vd, cols, bc);
            FusedAttentionGradientTile(qs, kt, vt, dO, dOBase, lse, delta, p, b, h, bh, i0, rows, j0, cols, pr, ds);
            TileGemm(ds, 0, bc, k.Data, kBase + j0 * k.RowStride, k.RowStride, dq, 0, hd, rows, hd, cols, accumulate: true);
        }
        for (int r = 0; r < rows; r++)
            StoreRow(dq, r * hd, dQ.Array!, dQ.Offset + (bh * p.SeqQ + i0 + r) * hd, hd, p.ScaleF, dQ.Overwrite);
    }
}
