using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> MultiHeadAttentionCore<T>(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, int numHeads, double? scale = null, bool causal = false)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (query.Rank != 3 || key.Rank != 3 || value.Rank != 3)
            throw new ArgumentException(
                $"MultiHeadAttentionCore expects rank-3 [batch, seq, heads*dim] query, key and value; got ranks "
                + $"{query.Rank}, {key.Rank}, {value.Rank}.");
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");

        int batch = query._shape[0], seqQ = query._shape[1], modelQ = query._shape[2];
        int seqK = key._shape[1], modelV = value._shape[2];
        if (key._shape[0] != batch || value._shape[0] != batch)
            throw new ArgumentException("query, key and value must have the same batch size.");
        if (value._shape[1] != seqK)
            throw new ArgumentException($"value has {value._shape[1]} positions but key has {seqK}.", nameof(value));
        if (key._shape[2] != modelQ)
            throw new ArgumentException($"key width {key._shape[2]} must equal query width {modelQ}.", nameof(key));
        if (modelQ % numHeads != 0 || modelV % numHeads != 0)
            throw new ArgumentException(
                $"query width {modelQ} and value width {modelV} must both be divisible by numHeads ({numHeads}).",
                nameof(numHeads));

        var shape = new AttentionCoreShape(batch, seqQ, seqK, numHeads, modelQ / numHeads, modelV / numHeads,
            (float)(scale ?? 1.0 / Math.Sqrt(modelQ / numHeads)), causal);

        // The fused kernels are host float32. GPU engines and other element types run the primitive chain, which
        // every backend already differentiates (and a GPU plan can capture).
        if (typeof(T) != typeof(float) || this is DirectGpuTensorEngine)
            return MultiHeadAttentionCoreDecomposed(query, key, value, shape);

        var q = (Tensor<float>)(object)query;
        var k = (Tensor<float>)(object)key;
        var v = (Tensor<float>)(object)value;

        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            // Each replayed forward rewrites the node's log-sum-exp rows, which the same step's backward reads.
            var lse = new float[batch * numHeads * seqQ];
            var inputs = new[] { q.IsContiguous ? q : q.Contiguous(), k.IsContiguous ? k : k.Contiguous(),
                                 v.IsContiguous ? v : v.Contiguous() };
            var outputShape = new[] { batch, seqQ, modelV };
            return (Tensor<T>)(object)scope.RecordVariadic(LazyNodeType.Custom, "MultiHeadAttentionCore", inputs,
                outputShape,
                (eng, output) =>
                {
                    if (output.IsContiguous && output._gpuBuffer is null && !output.HasPendingGpuData)
                    {
                        AttentionCoreForwardFloat(inputs[0], inputs[1], inputs[2], output, lse, shape);
                        output.IncrementVersion();
                    }
                    else
                    {
                        var staged = new Tensor<float>(outputShape);
                        AttentionCoreForwardFloat(inputs[0], inputs[1], inputs[2], staged, lse, shape);
                        DirectGpuTensorEngine.CopyResultInto(eng, staged, output);
                    }
                },
                AttentionCoreBackwardFloat, new object[] { lse, shape });
        }

        var result = new Tensor<float>(new[] { batch, seqQ, modelV });
        var lseEager = new float[batch * numHeads * seqQ];
        AttentionCoreForwardFloat(q.IsContiguous ? q : q.Contiguous(), k.IsContiguous ? k : k.Contiguous(),
            v.IsContiguous ? v : v.Contiguous(), result, lseEager, shape);
        var resultT = (Tensor<T>)(object)result;
        DifferentiableOps.RecordIfActive("MultiHeadAttentionCore", resultT, new[] { query, key, value },
            (BackwardFunction<T>)(object)(BackwardFunction<float>)AttentionCoreBackwardFloat,
            new object[] { lseEager, shape });
        return resultT;
    }

    /// <summary>Shape and options of one <see cref="MultiHeadAttentionCore{T}"/> call, kept with its graph node.</summary>
    internal sealed class AttentionCoreShape
    {
        public AttentionCoreShape(int batch, int seqQ, int seqK, int heads, int headDim, int valueDim, float scale, bool causal)
        {
            Batch = batch; SeqQ = seqQ; SeqK = seqK; Heads = heads;
            HeadDim = headDim; ValueDim = valueDim; Scale = scale; Causal = causal;
        }

        public int Batch { get; }
        public int SeqQ { get; }
        public int SeqK { get; }
        public int Heads { get; }
        public int HeadDim { get; }
        public int ValueDim { get; }
        public float Scale { get; }
        public bool Causal { get; }
        public int QueryWidth => Heads * HeadDim;
        public int ValueWidth => Heads * ValueDim;

        /// <summary>How many keys query row <paramref name="i"/> attends: all of them, or when causal the keys
        /// j &lt;= i + (seqK - seqQ) (bottom-right aligned, so the last query sees every key, as in a decode step
        /// over a key cache; the same as PyTorch's <c>is_causal</c> when seqQ == seqK). Zero for a row that sees none.</summary>
        public int KeyLimit(int i) => Causal ? Math.Max(0, Math.Min(SeqK, i + 1 + SeqK - SeqQ)) : SeqK;
    }

    /// <summary>The same function through the recorded primitives (per-head permutes and SDPA).</summary>
    private Tensor<T> MultiHeadAttentionCoreDecomposed<T>(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, AttentionCoreShape s)
    {
        var q4 = TensorPermute(Reshape(query, new[] { s.Batch, s.SeqQ, s.Heads, s.HeadDim }), new[] { 0, 2, 1, 3 });
        var k4 = TensorPermute(Reshape(key, new[] { s.Batch, s.SeqK, s.Heads, s.HeadDim }), new[] { 0, 2, 1, 3 });
        var v4 = TensorPermute(Reshape(value, new[] { s.Batch, s.SeqK, s.Heads, s.ValueDim }), new[] { 0, 2, 1, 3 });
        Tensor<bool>? mask = null;
        if (s.Causal)
        {
            var allowed = new bool[s.Batch * s.Heads * s.SeqQ * s.SeqK];
            for (int bh = 0; bh < s.Batch * s.Heads; bh++)
                for (int i = 0; i < s.SeqQ; i++)
                {
                    int limit = s.KeyLimit(i), row = (bh * s.SeqQ + i) * s.SeqK;
                    for (int j = 0; j < limit; j++) allowed[row + j] = true;
                }
            mask = new Tensor<bool>(allowed, new[] { s.Batch, s.Heads, s.SeqQ, s.SeqK });
        }
        var context = ScaledDotProductAttention(q4, k4, v4, mask, s.Scale, out _);
        return Reshape(TensorPermute(context, new[] { 0, 2, 1, 3 }), new[] { s.Batch, s.SeqQ, s.ValueWidth });
    }

    // Query rows and key columns per tile. A tile's score block, its Q rows and the transposed K block stay in L1/L2
    // for the head dims transformers use (<= 128).
    private const int AttentionCoreQueryBlock = 32;
    private const int AttentionCoreKeyBlock = 64;

    /// <summary>
    /// out[b, i, h*dv:(h+1)*dv] = softmax(scale * q_bh,i . K_bh^T) V_bh for every batch row b and head h, read straight
    /// from the head-interleaved [batch, seq, heads*dim] layout (no per-head permute copies), with an online softmax
    /// over key blocks so no [seqQ, seqK] score matrix is materialised. Also writes lse[(b*heads+h)*seqQ+i], the
    /// log-sum-exp of row i's scaled scores, which the backward uses to rebuild the probabilities.
    /// </summary>
    internal static void AttentionCoreForwardFloat(
        Tensor<float> query, Tensor<float> key, Tensor<float> value, Tensor<float> output, float[] lse,
        AttentionCoreShape s)
    {
        var qArr = ReadableBacking(query, out int qOff);
        var kArr = ReadableBacking(key, out int kOff);
        var vArr = ReadableBacking(value, out int vOff);
        var oArr = output.GetCpuBackingForContiguousWrite(out int oOff)
            ?? throw new InvalidOperationException("MultiHeadAttentionCore needs a host output buffer.");
        int tasks = s.Batch * s.Heads;
        long work = (long)tasks * s.SeqQ * s.SeqK * (s.HeadDim + s.ValueDim);
        CpuParallelSettings.ParallelForOrSerial(0, tasks, work,
            bh => AttentionCoreForwardHead(qArr, qOff, kArr, kOff, vArr, vOff, oArr, oOff, lse, s, bh),
            deterministicSafe: true);
    }

    private static void AttentionCoreForwardHead(
        float[] q, int qOff, float[] k, int kOff, float[] v, int vOff, float[] o, int oOff, float[] lse,
        AttentionCoreShape s, int bh)
    {
        int b = bh / s.Heads, h = bh % s.Heads;
        int hd = s.HeadDim, vd = s.ValueDim, qw = s.QueryWidth, vw = s.ValueWidth;
        int qBase = qOff + b * s.SeqQ * qw + h * hd;
        int kBase = kOff + b * s.SeqK * qw + h * hd;
        int vBase = vOff + b * s.SeqK * vw + h * vd;
        int oBase = oOff + b * s.SeqQ * vw + h * vd;
        int br = Math.Min(AttentionCoreQueryBlock, s.SeqQ), bc = Math.Min(AttentionCoreKeyBlock, s.SeqK);

        var ws = AttentionScratch.ForThread();
        float[] qs = ws.Get(0, br * hd), kt = ws.Get(1, hd * bc), sc = ws.Get(2, br * bc);
        float[] acc = ws.Get(3, br * vd), rowMax = ws.Get(4, br), rowSum = ws.Get(5, br);
        {
            for (int i0 = 0; i0 < s.SeqQ; i0 += br)
            {
                int rows = Math.Min(br, s.SeqQ - i0);
                for (int r = 0; r < rows; r++)
                {
                    ScaleCopy(q, qBase + (i0 + r) * qw, qs, r * hd, hd, s.Scale);
                    rowMax[r] = float.NegativeInfinity;
                    rowSum[r] = 0f;
                }
                Array.Clear(acc, 0, rows * vd);
                int keyEnd = s.KeyLimit(i0 + rows - 1);
                for (int j0 = 0; j0 < keyEnd; j0 += bc)
                {
                    int cols = Math.Min(bc, keyEnd - j0);
                    TransposeKeyBlock(k, kBase + j0 * qw, qw, kt, hd, cols, bc);
                    // S = (scale * Q_block) . K_block^T for the whole [rows, cols] tile.
                    TileGemm(qs, 0, hd, kt, 0, bc, sc, 0, bc, rows, cols, hd, accumulate: false);
                    for (int r = 0; r < rows; r++)
                    {
                        int valid = Math.Min(cols, s.KeyLimit(i0 + r) - j0);
                        int srow = r * bc;
                        if (valid <= 0) { Array.Clear(sc, srow, cols); continue; }   // contributes nothing below
                        float blockMax = float.NegativeInfinity;
                        for (int c = 0; c < valid; c++) if (sc[srow + c] > blockMax) blockMax = sc[srow + c];
                        float newMax = Math.Max(rowMax[r], blockMax);
                        float rescale = rowMax[r] == float.NegativeInfinity ? 0f : (float)Math.Exp(rowMax[r] - newMax);
                        ExpShifted(sc, srow, valid, newMax);
                        float blockSum = 0f;
                        for (int c = 0; c < valid; c++) blockSum += sc[srow + c];
                        rowSum[r] = rowSum[r] * rescale + blockSum;
                        rowMax[r] = newMax;
                        if (valid < cols) Array.Clear(sc, srow + valid, cols - valid);
                        if (rescale != 1f) Scale(acc, r * vd, vd, rescale);
                    }
                    // acc += P . V_block
                    TileGemm(sc, 0, bc, v, vBase + j0 * vw, vw, acc, 0, vd, rows, vd, cols, accumulate: true);
                }
                int lseRow = bh * s.SeqQ + i0;
                for (int r = 0; r < rows; r++)
                {
                    float sum = rowSum[r];
                    int orow = oBase + (i0 + r) * vw;
                    if (sum > 0f)
                    {
                        ScaleCopy(acc, r * vd, o, orow, vd, 1f / sum);
                        lse[lseRow + r] = rowMax[r] + (float)Math.Log(sum);
                    }
                    else
                    {
                        // A row with no visible key (causal, seqQ > seqK) attends to nothing: zero output.
                        Array.Clear(o, orow, vd);
                        lse[lseRow + r] = float.NegativeInfinity;
                    }
                }
            }
        }

    }

    /// <summary>
    /// Backward of <see cref="MultiHeadAttentionCore{T}"/> (FlashAttention-2 form): rebuilds each probability block from
    /// the saved log-sum-exp, then dV += P^T dO, dS = P * (dO V^T - rowsum(dO * O)), dQ = scale * dS K,
    /// dK = scale * dS^T Q. One task per (batch, head) owns that head's columns of every gradient, so no two tasks write
    /// the same element. Gradients go straight into the accumulator buffers when they exist.
    /// </summary>
    private static void AttentionCoreBackwardFloat(
        Tensor<float> gradOutput, Tensor<float>[] inputs, Tensor<float> output, object[] savedState, IEngine engine,
        Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var lse = (float[])savedState[0];
        var s = (AttentionCoreShape)savedState[1];
        Tensor<float> query = inputs[0], key = inputs[1], value = inputs[2];
        bool needQ = DifferentiableOps.IsGradientRequired(query);
        bool needK = DifferentiableOps.IsGradientRequired(key);
        bool needV = DifferentiableOps.IsGradientRequired(value);
        if (!needQ && !needK && !needV) return;

        var qArr = ReadableBacking(query, out int qOff);
        var kArr = ReadableBacking(key, out int kOff);
        var vArr = ReadableBacking(value, out int vOff);
        var oArr = ReadableBacking(output, out int oOff);
        var dOArr = ReadableBacking(gradOutput, out int dOOff);

        // Write each gradient into its accumulator in place when one exists, unless two of the three share a target
        // (the same tensor passed twice): the per-head tasks would then race on it, so those go through contributions.
        bool distinct = !ReferenceEquals(query, key) && !ReferenceEquals(query, value) && !ReferenceEquals(key, value);
        var dQ = GradTarget(needQ, query, grads, engine, distinct);
        var dK = GradTarget(needK, key, grads, engine, distinct);
        var dV = GradTarget(needV, value, grads, engine, distinct);
        if (distinct && dQ.Array is not null && (Overlap(dQ, dK) || Overlap(dQ, dV)))
        {
            dQ = GradTarget(needQ, query, grads, engine, distinct: false);
        }
        if (distinct && dK.Array is not null && Overlap(dK, dV))
        {
            dK = GradTarget(needK, key, grads, engine, distinct: false);
        }

        int tasks = s.Batch * s.Heads;
        long work = 3L * tasks * s.SeqQ * s.SeqK * (s.HeadDim + s.ValueDim);
        CpuParallelSettings.ParallelForOrSerial(0, tasks, work,
            bh => AttentionCoreBackwardHead(qArr, qOff, kArr, kOff, vArr, vOff, oArr, oOff, dOArr, dOOff, lse, s, bh,
                dQ, dK, dV),
            deterministicSafe: true);

        dQ.Commit(grads, query, engine);
        dK.Commit(grads, key, engine);
        dV.Commit(grads, value, engine);
    }

    /// <summary>Where one input's gradient goes: an accumulator written in place, or a fresh contribution tensor.</summary>
    private struct FusedGradTarget
    {
        public float[]? Array;
        public int Offset;
        public int Length;
        public bool Overwrite;
        public Tensor<float>? Accumulator;
        public Tensor<float>? Contribution;

        public void Commit(Dictionary<Tensor<float>, Tensor<float>> grads, Tensor<float> input, IEngine engine)
        {
            if (Accumulator is not null) Accumulator.IncrementVersion();
            else if (Contribution is not null) DifferentiableOps.AccumulateGrad(grads, input, Contribution, engine);
        }
    }

    private static FusedGradTarget GradTarget(
        bool needed, Tensor<float> input, Dictionary<Tensor<float>, Tensor<float>> grads, IEngine engine, bool distinct)
    {
        var target = new FusedGradTarget { Length = input.Length };
        if (!needed) return target;
        if (distinct)
        {
            var acc = DifferentiableOps.TryGetDirectGradTarget(grads, input, engine, out bool overwrite);
            var arr = acc?.GetCpuBackingForContiguousWrite(out target.Offset);
            if (acc is not null && arr is not null)
            {
                target.Array = arr;
                target.Overwrite = overwrite;
                target.Accumulator = acc;
                return target;
            }
        }
        var contribution = new Tensor<float>(input._shape);
        target.Array = contribution.GetCpuBackingForContiguousWrite(out target.Offset);
        target.Overwrite = true;
        target.Contribution = contribution;
        return target;
    }

    /// <summary>True when two in-place targets cover overlapping elements of one backing array.</summary>
    private static bool Overlap(FusedGradTarget a, FusedGradTarget b) =>
        a.Array is not null && ReferenceEquals(a.Array, b.Array)
        && a.Offset < b.Offset + b.Length && b.Offset < a.Offset + a.Length;

    private static void AttentionCoreBackwardHead(
        float[] q, int qOff, float[] k, int kOff, float[] v, int vOff, float[] o, int oOff, float[] dO, int dOOff,
        float[] lse, AttentionCoreShape s, int bh,
        FusedGradTarget dQ, FusedGradTarget dK, FusedGradTarget dV)
    {
        int b = bh / s.Heads, h = bh % s.Heads;
        int hd = s.HeadDim, vd = s.ValueDim, qw = s.QueryWidth, vw = s.ValueWidth;
        int qBase = qOff + b * s.SeqQ * qw + h * hd;
        int kBase = kOff + b * s.SeqK * qw + h * hd;
        int vBase = vOff + b * s.SeqK * vw + h * vd;
        int oBase = oOff + b * s.SeqQ * vw + h * vd;
        int dOBase = dOOff + b * s.SeqQ * vw + h * vd;
        int br = Math.Min(AttentionCoreQueryBlock, s.SeqQ), bc = Math.Min(AttentionCoreKeyBlock, s.SeqK);

        var ws = AttentionScratch.ForThread();
        float[] qs = ws.Get(0, br * hd), kt = ws.Get(1, hd * bc), vt = ws.Get(2, vd * bc);
        float[] p = ws.Get(3, br * bc), ds = ws.Get(4, br * bc), dq = ws.Get(5, br * hd), delta = ws.Get(6, br);
        float[] dk = ws.Get(7, s.SeqK * hd), dv = ws.Get(8, s.SeqK * vd);
        float[] pt = ws.Get(9, bc * br), dst = ws.Get(10, bc * br);
        {
            Array.Clear(dk, 0, s.SeqK * hd);
            Array.Clear(dv, 0, s.SeqK * vd);
            for (int i0 = 0; i0 < s.SeqQ; i0 += br)
            {
                int rows = Math.Min(br, s.SeqQ - i0);
                for (int r = 0; r < rows; r++)
                {
                    ScaleCopy(q, qBase + (i0 + r) * qw, qs, r * hd, hd, s.Scale);
                    delta[r] = Dot(dO, dOBase + (i0 + r) * vw, o, oBase + (i0 + r) * vw, vd);
                }
                Array.Clear(dq, 0, rows * hd);
                int keyEnd = s.KeyLimit(i0 + rows - 1);
                for (int j0 = 0; j0 < keyEnd; j0 += bc)
                {
                    int cols = Math.Min(bc, keyEnd - j0);
                    TransposeKeyBlock(k, kBase + j0 * qw, qw, kt, hd, cols, bc);
                    TransposeKeyBlock(v, vBase + j0 * vw, vw, vt, vd, cols, bc);
                    // P = exp(scale * Q.K^T - lse) and dP = dO.V^T for the whole tile, then dS = P * (dP - delta).
                    // Masked entries are zeroed rather than skipped: the dV and dK products read the tile by column.
                    TileGemm(qs, 0, hd, kt, 0, bc, p, 0, bc, rows, cols, hd, accumulate: false);
                    TileGemm(dO, dOBase + i0 * vw, vw, vt, 0, bc, ds, 0, bc, rows, cols, vd, accumulate: false);
                    for (int r = 0; r < rows; r++)
                    {
                        int valid = Math.Max(0, Math.Min(cols, s.KeyLimit(i0 + r) - j0));
                        float rowLse = lse[bh * s.SeqQ + i0 + r];
                        int prow = r * bc;
                        if (valid == 0 || float.IsNegativeInfinity(rowLse))
                        {
                            Array.Clear(p, prow, cols);
                            Array.Clear(ds, prow, cols);
                            continue;
                        }
                        ExpShifted(p, prow, valid, rowLse);
                        float d = delta[r];
                        for (int c = 0; c < valid; c++) ds[prow + c] = p[prow + c] * (ds[prow + c] - d);
                        if (valid < cols)
                        {
                            Array.Clear(p, prow + valid, cols - valid);
                            Array.Clear(ds, prow + valid, cols - valid);
                        }
                    }
                    // dV[j0..] += P^T . dO_block;  dK[j0..] += dS^T . (scale * Q_block);  dQ_block += dS . K_block
                    TransposeTile(p, bc, pt, rows, rows, cols);
                    TransposeTile(ds, bc, dst, rows, rows, cols);
                    TileGemm(pt, 0, rows, dO, dOBase + i0 * vw, vw, dv, j0 * vd, vd, cols, vd, rows, accumulate: true);
                    TileGemm(dst, 0, rows, qs, 0, hd, dk, j0 * hd, hd, cols, hd, rows, accumulate: true);
                    TileGemm(ds, 0, bc, k, kBase + j0 * qw, qw, dq, 0, hd, rows, hd, cols, accumulate: true);
                }
                if (dQ.Array is not null)
                    for (int r = 0; r < rows; r++)
                        StoreRow(dq, r * hd, dQ.Array, dQ.Offset + b * s.SeqQ * qw + (i0 + r) * qw + h * hd, hd,
                            s.Scale, dQ.Overwrite);
            }
            if (dK.Array is not null)
                for (int j = 0; j < s.SeqK; j++)
                    StoreRow(dk, j * hd, dK.Array, dK.Offset + b * s.SeqK * qw + j * qw + h * hd, hd, 1f, dK.Overwrite);
            if (dV.Array is not null)
                for (int j = 0; j < s.SeqK; j++)
                    StoreRow(dv, j * vd, dV.Array, dV.Offset + b * s.SeqK * vw + j * vw + h * vd, vd, 1f, dV.Overwrite);
        }

    }

    /// <summary>
    /// Per-thread scratch for the attention head kernels, reused across heads and steps. Renting 6-9 arrays per head
    /// from ArrayPool.Shared overflowed its one-array-per-bucket thread cache into the locked shared partitions: with
    /// 33 threads that contention was ~80% of the attention time. A slot grows to the largest size requested and is
    /// kept for the thread's lifetime (a few KB per slot at typical head sizes; dK/dV slots are seqK x headDim).
    /// </summary>
    private sealed class AttentionScratch
    {
        [ThreadStatic] private static AttentionScratch? t_instance;
        private readonly float[][] _slots = new float[11][];

        public static AttentionScratch ForThread() => t_instance ??= new AttentionScratch();

        public float[] Get(int slot, int length)
        {
            var a = _slots[slot];
            if (a is null || a.Length < length) _slots[slot] = a = new float[length];
            return a;
        }
    }

    private static float[] ReadableBacking(Tensor<float> tensor, out int offset)
    {
        if (tensor.IsContiguous)
        {
            var arr = tensor.GetCpuBackingForStridedRead(out offset);
            if (arr is not null) return arr;
        }
        offset = 0;
        return tensor.Contiguous().GetFlattenedData();
    }

    // ---- small vector helpers (System.Numerics.Vector<float>: AVX2 on x64, every target framework) ----

    /// <summary>dst[dOff..+n] = a * src[sOff..+n].</summary>
    private static void ScaleCopy(float[] src, int sOff, float[] dst, int dOff, int n, float a)
    {
        int w = System.Numerics.Vector<float>.Count, i = 0;
        var va = new System.Numerics.Vector<float>(a);
        for (; i <= n - w; i += w) (new System.Numerics.Vector<float>(src, sOff + i) * va).CopyTo(dst, dOff + i);
        for (; i < n; i++) dst[dOff + i] = src[sOff + i] * a;
    }

    /// <summary>x[off..+n] *= a.</summary>
    private static void Scale(float[] x, int off, int n, float a)
    {
        int w = System.Numerics.Vector<float>.Count, i = 0;
        var va = new System.Numerics.Vector<float>(a);
        for (; i <= n - w; i += w) (new System.Numerics.Vector<float>(x, off + i) * va).CopyTo(x, off + i);
        for (; i < n; i++) x[off + i] *= a;
    }

    /// <summary>y[yOff..+n] += a * x[xOff..+n].</summary>
    private static void Axpy(float a, float[] x, int xOff, float[] y, int yOff, int n)
    {
        int w = System.Numerics.Vector<float>.Count, i = 0;
        var va = new System.Numerics.Vector<float>(a);
        for (; i <= n - w; i += w)
            (new System.Numerics.Vector<float>(y, yOff + i) + va * new System.Numerics.Vector<float>(x, xOff + i))
                .CopyTo(y, yOff + i);
        for (; i < n; i++) y[yOff + i] += a * x[xOff + i];
    }

    private static float Dot(float[] x, int xOff, float[] y, int yOff, int n)
    {
        int w = System.Numerics.Vector<float>.Count, i = 0;
        var acc = System.Numerics.Vector<float>.Zero;
        for (; i <= n - w; i += w)
            acc += new System.Numerics.Vector<float>(x, xOff + i) * new System.Numerics.Vector<float>(y, yOff + i);
        float sum = System.Numerics.Vector.Dot(acc, System.Numerics.Vector<float>.One);
        for (; i < n; i++) sum += x[xOff + i] * y[yOff + i];
        return sum;
    }

    /// <summary>
    /// C[i, j] (= or +=) sum over p &lt; K of A[i, p] * B[p, j] for i &lt; M, j &lt; N, on row-major strided operands
    /// (lda/ldb/ldc are row strides; rows of B and C may sit inside wider head-interleaved rows). Register-blocked
    /// 4 rows x 16 columns: each loaded B vector feeds four rows, so a block's [M, N] product runs as small-GEMM tiles
    /// instead of M separate row-vector products (which ran the attention forward at ~6 GFLOP/s single-threaded).
    /// </summary>
    private static unsafe void TileGemm(
        float[] a, int aOff, int lda, float[] b, int bOff, int ldb, float[] c, int cOff, int ldc,
        int M, int N, int K, bool accumulate)
    {
        if (M <= 0 || N <= 0) return;
        fixed (float* pa0 = a, pb0 = b, pc0 = c)
        {
            float* pa = pa0 + aOff, pb = pb0 + bOff, pc = pc0 + cOff;
#if NET5_0_OR_GREATER
            if (System.Runtime.Intrinsics.X86.Fma.IsSupported)
            {
                int i = 0;
                for (; i + 4 <= M; i += 4) TileRows4Avx(pa + (long)i * lda, lda, pb, ldb, pc + (long)i * ldc, ldc, N, K, accumulate);
                for (; i < M; i++) TileRows1Avx(pa + (long)i * lda, pb, ldb, pc + (long)i * ldc, N, K, accumulate);
                return;
            }
#endif
            int w = System.Numerics.Vector<float>.Count;
            for (int i = 0; i < M; i++)
            {
                float* ar = pa + (long)i * lda, cr = pc + (long)i * ldc;
                int j = 0;
                for (; j + w <= N; j += w)
                {
                    var acc = accumulate ? System.Runtime.CompilerServices.Unsafe.Read<System.Numerics.Vector<float>>(cr + j)
                                         : System.Numerics.Vector<float>.Zero;
                    for (int p = 0; p < K; p++)
                        acc += new System.Numerics.Vector<float>(ar[p])
                               * System.Runtime.CompilerServices.Unsafe.Read<System.Numerics.Vector<float>>(pb + (long)p * ldb + j);
                    System.Runtime.CompilerServices.Unsafe.Write(cr + j, acc);
                }
                for (; j < N; j++)
                {
                    float sum = accumulate ? cr[j] : 0f;
                    for (int p = 0; p < K; p++) sum += ar[p] * pb[(long)p * ldb + j];
                    cr[j] = sum;
                }
            }
        }
    }

#if NET5_0_OR_GREATER
    private static unsafe void TileRows4Avx(float* a, int lda, float* b, int ldb, float* c, int ldc, int N, int K, bool accumulate)
    {
        float* a1 = a + lda, a2 = a1 + lda, a3 = a2 + lda;
        float* c1 = c + ldc, c2 = c1 + ldc, c3 = c2 + ldc;
        int j = 0;
        for (; j + 16 <= N; j += 16)
        {
            System.Runtime.Intrinsics.Vector256<float> x00, x01, x10, x11, x20, x21, x30, x31;
            if (accumulate)
            {
                x00 = Avx.LoadVector256(c + j); x01 = Avx.LoadVector256(c + j + 8);
                x10 = Avx.LoadVector256(c1 + j); x11 = Avx.LoadVector256(c1 + j + 8);
                x20 = Avx.LoadVector256(c2 + j); x21 = Avx.LoadVector256(c2 + j + 8);
                x30 = Avx.LoadVector256(c3 + j); x31 = Avx.LoadVector256(c3 + j + 8);
            }
            else
            {
                x00 = x01 = x10 = x11 = x20 = x21 = x30 = x31 = System.Runtime.Intrinsics.Vector256<float>.Zero;
            }
            float* bp = b + j;
            for (int p = 0; p < K; p++, bp += ldb)
            {
                var b0 = Avx.LoadVector256(bp);
                var b1 = Avx.LoadVector256(bp + 8);
                var s = System.Runtime.Intrinsics.Vector256.Create(a[p]);
                x00 = Fma.MultiplyAdd(s, b0, x00); x01 = Fma.MultiplyAdd(s, b1, x01);
                s = System.Runtime.Intrinsics.Vector256.Create(a1[p]);
                x10 = Fma.MultiplyAdd(s, b0, x10); x11 = Fma.MultiplyAdd(s, b1, x11);
                s = System.Runtime.Intrinsics.Vector256.Create(a2[p]);
                x20 = Fma.MultiplyAdd(s, b0, x20); x21 = Fma.MultiplyAdd(s, b1, x21);
                s = System.Runtime.Intrinsics.Vector256.Create(a3[p]);
                x30 = Fma.MultiplyAdd(s, b0, x30); x31 = Fma.MultiplyAdd(s, b1, x31);
            }
            Avx.Store(c + j, x00); Avx.Store(c + j + 8, x01);
            Avx.Store(c1 + j, x10); Avx.Store(c1 + j + 8, x11);
            Avx.Store(c2 + j, x20); Avx.Store(c2 + j + 8, x21);
            Avx.Store(c3 + j, x30); Avx.Store(c3 + j + 8, x31);
        }
        for (; j + 8 <= N; j += 8)
        {
            var x0 = accumulate ? Avx.LoadVector256(c + j) : System.Runtime.Intrinsics.Vector256<float>.Zero;
            var x1 = accumulate ? Avx.LoadVector256(c1 + j) : System.Runtime.Intrinsics.Vector256<float>.Zero;
            var x2 = accumulate ? Avx.LoadVector256(c2 + j) : System.Runtime.Intrinsics.Vector256<float>.Zero;
            var x3 = accumulate ? Avx.LoadVector256(c3 + j) : System.Runtime.Intrinsics.Vector256<float>.Zero;
            float* bp = b + j;
            for (int p = 0; p < K; p++, bp += ldb)
            {
                var b0 = Avx.LoadVector256(bp);
                x0 = Fma.MultiplyAdd(System.Runtime.Intrinsics.Vector256.Create(a[p]), b0, x0);
                x1 = Fma.MultiplyAdd(System.Runtime.Intrinsics.Vector256.Create(a1[p]), b0, x1);
                x2 = Fma.MultiplyAdd(System.Runtime.Intrinsics.Vector256.Create(a2[p]), b0, x2);
                x3 = Fma.MultiplyAdd(System.Runtime.Intrinsics.Vector256.Create(a3[p]), b0, x3);
            }
            Avx.Store(c + j, x0); Avx.Store(c1 + j, x1); Avx.Store(c2 + j, x2); Avx.Store(c3 + j, x3);
        }
        for (; j < N; j++)
        {
            float s0 = accumulate ? c[j] : 0f, s1 = accumulate ? c1[j] : 0f;
            float s2 = accumulate ? c2[j] : 0f, s3 = accumulate ? c3[j] : 0f;
            for (int p = 0; p < K; p++)
            {
                float bv = b[(long)p * ldb + j];
                s0 += a[p] * bv; s1 += a1[p] * bv; s2 += a2[p] * bv; s3 += a3[p] * bv;
            }
            c[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
        }
    }

    private static unsafe void TileRows1Avx(float* a, float* b, int ldb, float* c, int N, int K, bool accumulate)
    {
        int j = 0;
        for (; j + 8 <= N; j += 8)
        {
            var x = accumulate ? Avx.LoadVector256(c + j) : System.Runtime.Intrinsics.Vector256<float>.Zero;
            float* bp = b + j;
            for (int p = 0; p < K; p++, bp += ldb)
                x = Fma.MultiplyAdd(System.Runtime.Intrinsics.Vector256.Create(a[p]), Avx.LoadVector256(bp), x);
            Avx.Store(c + j, x);
        }
        for (; j < N; j++)
        {
            float sum = accumulate ? c[j] : 0f;
            for (int p = 0; p < K; p++) sum += a[p] * b[(long)p * ldb + j];
            c[j] = sum;
        }
    }
#endif

    /// <summary>dst[c * dstStride + r] = src[r * srcStride + c] for r &lt; rows, c &lt; cols.</summary>
    private static void TransposeTile(float[] src, int srcStride, float[] dst, int dstStride, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            int s = r * srcStride;
            for (int c = 0; c < cols; c++) dst[c * dstStride + r] = src[s + c];
        }
    }

    /// <summary>block[d * stride + c] = src[srcOff + c * rowStride + d] for d &lt; depth, c &lt; cols.</summary>
    private static void TransposeKeyBlock(float[] src, int srcOff, int rowStride, float[] block, int depth, int cols, int stride)
    {
        for (int c = 0; c < cols; c++)
        {
            int s = srcOff + c * rowStride;
            for (int d = 0; d < depth; d++) block[d * stride + c] = src[s + d];
        }
    }

    /// <summary>x[off..+n] = exp(x - shift).</summary>
    private static void ExpShifted(float[] x, int off, int n, float shift)
    {
        int w = System.Numerics.Vector<float>.Count, i = 0;
        var vs = new System.Numerics.Vector<float>(shift);
        for (; i <= n - w; i += w) (new System.Numerics.Vector<float>(x, off + i) - vs).CopyTo(x, off + i);
        for (; i < n; i++) x[off + i] -= shift;
        var span = new Span<float>(x, off, n);
        SimdKernels.Exp(span, span);
    }

    /// <summary>dst[dOff..+n] = a * src (overwrite) or dst += a * src.</summary>
    private static void StoreRow(float[] src, int sOff, float[] dst, int dOff, int n, float a, bool overwrite)
    {
        if (overwrite) ScaleCopy(src, sOff, dst, dOff, n, a);
        else Axpy(a, src, sOff, dst, dOff, n);
    }
}
