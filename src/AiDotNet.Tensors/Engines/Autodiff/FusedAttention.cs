using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Config for <see cref="FusedAttention{T}.Forward"/> — softmax scale,
/// causal mask, per-axis block sizes, dropout, and whether to return
/// the (large) attention-weight matrix.
///
/// <para>Uses nullable properties + industry-standard defaults so a
/// zero-config call still runs sensibly. Defaults match PyTorch's
/// <c>torch.nn.functional.scaled_dot_product_attention</c>: scale is
/// <c>1/sqrt(headDim)</c>, causal is false, block size is auto (picked
/// by the kernel based on seqLen), no dropout.</para>
/// </summary>
public sealed class FlashAttentionConfig
{
    /// <summary>Softmax scale. When null, defaults to <c>1 / sqrt(headDim)</c>.</summary>
    public double? Scale { get; set; }

    /// <summary>Synonym for <see cref="Scale"/> — the issue spec uses
    /// <c>ScaleFactor</c>; we accept either.</summary>
    public double? ScaleFactor { get => Scale; set => Scale = value; }

    /// <summary>When true, applies the autoregressive causal mask.
    /// Honours <see cref="QueryOffset"/> so <c>q_i</c> attends to all
    /// <c>k_j</c> where <c>j &lt;= queryOffset + i</c>.</summary>
    public bool IsCausal { get; set; }

    /// <summary>When true, returns the attention-weight matrix. Allocates
    /// an extra <c>[B, H, seqQ, seqK]</c> tensor; leave false for inference
    /// when you only need the output.</summary>
    public bool ReturnAttentionWeights { get; set; }

    /// <summary>Block size along the query axis. Null = auto.
    /// <para><b>RESERVED.</b> Forward currently dispatches to the engine's
    /// tuned SDPA kernel (no bias) or a GEMM chain (with bias); neither
    /// reads this value today. Persisted on the config so existing callers
    /// and the future block-tiled fast path agree on the knob — setting it
    /// is harmless but has no effect on the hot path. Wire-through lands
    /// with the block-tiled bias-aware kernel.</para></summary>
    public int? BlockSizeQ { get; set; }

    /// <summary>Block size along the key / value axis. Null = auto.
    /// <para><b>RESERVED.</b> Same status as <see cref="BlockSizeQ"/>:
    /// unread by the current Forward paths, persisted for future block-
    /// tiled kernels.</para></summary>
    public int? BlockSizeKV { get; set; }

    /// <summary>Back-compat single knob — sets both <see cref="BlockSizeQ"/>
    /// and <see cref="BlockSizeKV"/> at once. Reads <see cref="BlockSizeQ"/>.
    /// <para><b>RESERVED</b> — see <see cref="BlockSizeQ"/>.</para></summary>
    public int? BlockSize
    {
        get => BlockSizeQ;
        set { BlockSizeQ = value; BlockSizeKV = value; }
    }

    /// <summary>Offset of the query sequence within the full KV history.
    /// For KV-cache / autoregressive inference where Q is a slice (often
    /// length 1 at decode time) and K/V represent the full past-plus-
    /// current context. Causal mask honours the offset so the query
    /// token correctly attends to every past key.</summary>
    public int QueryOffset { get; set; }

    /// <summary>Dropout rate applied to the post-softmax weights. Null
    /// means "no dropout"; 0.0 matches but preserves explicit intent.
    /// Applied only during training — inference callers should leave
    /// this null.</summary>
    public double? DropoutRate { get; set; }

    /// <summary>
    /// When true, dispatch through the block-tiled online-softmax
    /// <see cref="FlashAttention2"/> kernel (forward + backward) instead
    /// of materialising the full score matrix. O(seqLen) intermediate
    /// memory — required for long-context (seq ≥ 4k). Today gated to
    /// <c>T == float</c>; falls back to the standard path for other T.
    /// </summary>
    public bool UseFlashAttention2 { get; set; }

    /// <summary>Default configuration — every field at its spec default.</summary>
    public static FlashAttentionConfig Default => new();
}

/// <summary>
/// Generic CPU-path fused attention with the full PyTorch-parity feature
/// set the existing <see cref="IEngine.ScaledDotProductAttention{T}"/>
/// lacks:
/// <list type="bullet">
/// <item>Rank-3 input support — <c>[batch, seqLen, headDim]</c> is
///       promoted to <c>[batch, 1, seqLen, headDim]</c>.</item>
/// <item>Optional <c>attentionBias</c> added to the score matrix
///       before softmax — covers ALiBi, T5 relative-position, and
///       custom additive masks.</item>
/// <item>Causal toggle, block-size override, optional weights return.</item>
/// </list>
///
/// <para>Delegates to the engine's tuned
/// <see cref="IEngine.ScaledDotProductAttention{T}"/> for the common
/// bias-free case. When <paramref name="attentionBias"/> is supplied
/// the routine composes the path manually via MatMul + bias add +
/// softmax + MatMul, which is still GEMM-dominated but loses the fast
/// block-tiled softmax. Fine for correctness / infrequent bias use;
/// block-tiled bias-aware kernel is future work.</para>
/// </summary>
public static class FusedAttention<T>
{
    // Supported element types. Attention uses fractional scale
    // (1/sqrt(headDim)), softmax probabilities, and -Inf causal masks —
    // none of which are meaningful for integer T. The IEngine's
    // ScaledDotProductAttention and the engine's softmax only ship
    // float / double / Half code paths; passing int/long would either
    // quantize the scale to zero or throw deep inside numOps.FromDouble.
    // Fail fast at entry with an actionable message instead of producing
    // silent garbage or a misleading exception from the numeric-ops layer.
    private static readonly HashSet<Type> s_supportedTypes = new()
    {
        typeof(float), typeof(double), typeof(System.Half),
    };

    private static void EnsureSupportedElementType()
    {
        if (!s_supportedTypes.Contains(typeof(T)))
            throw new NotSupportedException(
                $"FusedAttention<{typeof(T).Name}> is not supported. The implementation relies on " +
                "fractional softmax scale, softmax probabilities, and -Inf causal masks — use " +
                "float, double, or System.Half as the element type.");
    }

    /// <summary>
    /// Forward pass. Returns (Output, AttentionWeights) — weights is null
    /// unless <see cref="FlashAttentionConfig.ReturnAttentionWeights"/> is
    /// set.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// <typeparamref name="T"/> is not a floating-point type
    /// (<c>float</c>, <c>double</c>, or <c>System.Half</c>).
    /// </exception>
    public static (Tensor<T> Output, Tensor<T>? AttentionWeights) Forward(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        FlashAttentionConfig? config = null,
        Tensor<T>? attentionBias = null,
        IEngine? engine = null)
    {
        EnsureSupportedElementType();
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key   is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        config ??= new FlashAttentionConfig();
        engine ??= new CpuEngine();

        // Rank normalisation — 3D [B, S, D] → 4D [B, 1, S, D]. Dispatcher
        // below always sees 4D so shape-validation lives in one place.
        bool was3D = query.Rank == 3;
        if (was3D)
        {
            if (key.Rank != 3 || value.Rank != 3)
                throw new ArgumentException(
                    "When query is rank-3, key and value must also be rank-3.");
            query = PromoteToFourD(engine, query);
            key = PromoteToFourD(engine, key);
            value = PromoteToFourD(engine, value);
        }
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            throw new ArgumentException(
                $"Attention inputs must be rank-3 or rank-4; got query rank {query.Rank}.");

        // Per-dim shape agreement. The bias path below derives B/H/headDim/Sk
        // from `query` and then reshapes `key`/`value` using those dims —
        // without this check, inputs like q:[1,2,S,D] vs k:[2,1,S,D] would be
        // silently reinterpreted into a wrong batch/head split rather than
        // rejected, producing numerically wrong attention with no exception.
        if (query._shape[0] != key._shape[0] || query._shape[0] != value._shape[0])
            throw new ArgumentException(
                $"query/key/value batch dims must match: query[0]={query._shape[0]}, " +
                $"key[0]={key._shape[0]}, value[0]={value._shape[0]}.");
        if (query._shape[1] != key._shape[1] || query._shape[1] != value._shape[1])
            throw new ArgumentException(
                $"query/key/value head dims must match: query[1]={query._shape[1]}, " +
                $"key[1]={key._shape[1]}, value[1]={value._shape[1]}.");
        if (query._shape[3] != key._shape[3])
            throw new ArgumentException(
                $"query and key must share headDim (last axis): query[3]={query._shape[3]}, " +
                $"key[3]={key._shape[3]}.");
        if (key._shape[2] != value._shape[2])
            throw new ArgumentException(
                $"key and value must share sequence length (axis 2): key[2]={key._shape[2]}, " +
                $"value[2]={value._shape[2]}.");

        // KV-cache / query-offset support (issue #198 gap D): when the
        // caller supplies a queryOffset, q_i must attend to keys
        // k_j where j <= queryOffset + i. We validate queryOffset fits
        // within the KV history here; the mask itself is built below.
        int queryOffset = config.QueryOffset;
        if (queryOffset < 0 || queryOffset + query._shape[2] > key._shape[2])
            throw new ArgumentException(
                $"queryOffset={queryOffset} + seqQ={query._shape[2]} must be <= seqKV={key._shape[2]}.",
                nameof(config));

        // FlashAttention-2 block-tiled dispatch: O(seqLen) memory, uses
        // online softmax so long-context inference (seq ≥ 4k) stays
        // tractable. Gated to T=float today (the kernel is float-only in
        // the paper; bf16 / fp16 Tensor-Core variants follow).
        if (config.UseFlashAttention2 && typeof(T) == typeof(float))
        {
            var qf = (Tensor<float>)(object)query;
            var kf = (Tensor<float>)(object)key;
            var vf = (Tensor<float>)(object)value;
            var biasF = attentionBias is null ? null : (Tensor<float>)(object)attentionBias;
            int bsQ = config.BlockSizeQ ?? 64;
            int bsKV = config.BlockSizeKV ?? 64;
            var (fOut, _lse) = FlashAttention2.Forward(
                qf, kf, vf, bsQ, bsKV, config.Scale, config.IsCausal, queryOffset, biasF);
            var outT = (Tensor<T>)(object)fOut;
            if (was3D) outT = DemoteToThreeD(engine, outT);
            // The block-tiled kernel deliberately never materialises the
            // full weights matrix; if the caller needed weights they must
            // use the standard bias path.
            if (config.ReturnAttentionWeights)
                throw new ArgumentException(
                    "UseFlashAttention2 = true does not support ReturnAttentionWeights " +
                    "(the block-tiled kernel never materialises the [B,H,Sq,Sk] matrix).",
                    nameof(config));
            return (outT, null);
        }

        // Causal no-bias → synthesize the offset-aware -inf upper-triangle
        // bias and route through the bias path. Honours queryOffset so the
        // KV-cache decoder case (seqQ=1, queryOffset=t, seqKV=t+1) masks
        // only positions > t — which is none — letting the single decode
        // token attend to every past key as expected.
        if (attentionBias is null && config.IsCausal)
        {
            attentionBias = BuildCausalBias(
                engine, query._shape[2], key._shape[2], queryOffset);
        }

        // No bias → use the engine's tuned SDPA kernel. Engine handles the
        // fast float/double BLAS path; returns attention weights via out.
        //
        // Note on memory: the engine overload always materializes the
        // [B,H,Sq,Sk] weights tensor internally via the out parameter; we
        // discard it when ReturnAttentionWeights is false. A
        // true weights-less engine overload is a follow-up — flagged so the
        // public config docstring doesn't overclaim the memory win for
        // ReturnAttentionWeights=false on the no-bias path.
        if (attentionBias is null)
        {
            var result = engine.ScaledDotProductAttention(
                query, key, value, mask: null, scale: config.Scale,
                out var weights);
            if (was3D)
                result = DemoteToThreeD(engine, result);
            // Respect ReturnAttentionWeights: if caller didn't ask, null out
            // to avoid leaking a potentially large tensor to the caller
            // (the engine still allocated it internally — see note above).
            Tensor<T>? returnedWeights = config.ReturnAttentionWeights
                ? (was3D ? DemoteToThreeD(engine, weights) : weights)
                : null;
            return (result, returnedWeights);
        }

        // Bias path: compose manually. Output = softmax(Q @ K^T * scale + bias) @ V.
        // Uses engine primitives so it still goes through tuned GEMMs.
        // TensorBatchMatMul wants rank-3 [batch, M, K]; 4D [B,H,Sq,D] is
        // collapsed to [B*H, Sq, D] and reshaped back afterwards.
        var numOps = MathHelper.GetNumericOperations<T>();
        int B = query._shape[0], H = query._shape[1];
        int Sq = query._shape[2], headDim = query._shape[3];
        int Sk = key._shape[2], Dv = value._shape[3];
        double scaleVal = config.Scale ?? 1.0 / Math.Sqrt(headDim);

        var qFlat  = engine.Reshape(query, new[] { B * H, Sq, headDim });
        var kFlat  = engine.Reshape(key,   new[] { B * H, Sk, headDim });
        var vFlat  = engine.Reshape(value, new[] { B * H, Sk, Dv });
        var kFlatT = kFlat.TransposeLast2D();                       // [B*H, headDim, Sk]

        var scoresFlat = engine.TensorBatchMatMul(qFlat, kFlatT);   // [B*H, Sq, Sk]
        scoresFlat = engine.TensorMultiplyScalar(scoresFlat, numOps.FromDouble(scaleVal));
        var scores = engine.Reshape(scoresFlat, new[] { B, H, Sq, Sk });
        scores = AddBroadcastBias(engine, scores, attentionBias);
        // IsCausal is now handled via BuildCausalBias → attentionBias above,
        // so we skip a redundant ApplyCausalMask pass when attentionBias is
        // the synthesized causal bias. For the (bias + isCausal) case the
        // caller supplied an additive bias AND wants causal — layer both.
        if (config.IsCausal && attentionBias is not null)
        {
            // Only apply the offset-aware causal mask when the caller gave
            // us their own bias; otherwise the BuildCausalBias result
            // already handles masking and re-applying would be a no-op.
            // We detect "caller-supplied bias" by checking the bias shape
            // doesn't match BuildCausalBias's canonical [1,1,Sq,Sk] form.
            if (!(attentionBias._shape.Length == 4
                  && attentionBias._shape[0] == 1
                  && attentionBias._shape[1] == 1))
            {
                scores = ApplyCausalMask(engine, scores, numOps, config.QueryOffset);
            }
        }
        var weightsFull = engine.TensorSoftmax(scores, axis: 3);

        var weightsFlat = engine.Reshape(weightsFull, new[] { B * H, Sq, Sk });
        var resultFlat = engine.TensorBatchMatMul(weightsFlat, vFlat); // [B*H, Sq, Dv]
        var result2 = engine.Reshape(resultFlat, new[] { B, H, Sq, Dv });
        if (was3D)
            result2 = DemoteToThreeD(engine, result2);
        Tensor<T>? maybeWeights = config.ReturnAttentionWeights
            ? (was3D ? DemoteToThreeD(engine, weightsFull) : weightsFull)
            : null;
        return (result2, maybeWeights);
    }

    private static Tensor<T> PromoteToFourD(IEngine engine, Tensor<T> t)
    {
        var newShape = new[] { t._shape[0], 1, t._shape[1], t._shape[2] };
        return engine.Reshape(t, newShape);
    }

    private static Tensor<T> DemoteToThreeD(IEngine engine, Tensor<T> t)
    {
        // Collapse the 1-head dim. When heads > 1 we leave the tensor 4D
        // — the caller asked for 3D input but the computation legitimately
        // produced a head-aware result; downstream should reshape itself.
        if (t._shape.Length == 4 && t._shape[1] == 1)
        {
            return engine.Reshape(t, new[] { t._shape[0], t._shape[2], t._shape[3] });
        }
        return t;
    }

    private static Tensor<T> TransposeLastTwo(IEngine engine, Tensor<T> t)
    {
        // Tensor.TransposeLast2D returns a contiguous-on-demand view that
        // swaps the last two dims — exactly the K^T we need for Q @ K^T.
        return t.TransposeLast2D();
    }

    private static Tensor<T> AddBroadcastBias(IEngine engine, Tensor<T> scores, Tensor<T> bias)
    {
        // Engine TensorBroadcastAdd handles the broadcast rules —
        // bias can be [B, H, Sq, Sk], [H, Sq, Sk], [1, 1, Sq, Sk], etc.
        return engine.TensorBroadcastAdd(scores, bias);
    }

    private static Tensor<T> BuildCausalBias(IEngine engine, int sq, int sk, int queryOffset = 0)
    {
        // Offset-aware causal bias: position (q, k) is visible iff
        // k <= queryOffset + q. For the classic self-attention case
        // (queryOffset=0, sq=sk) this collapses to the usual lower-
        // triangular mask. For KV-cache decode (sq=1, queryOffset=t,
        // sk=t+1) only column t+0 is visible to the single query row.
        var numOps = MathHelper.GetNumericOperations<T>();
        T zero = numOps.Zero;
        T negInf = numOps.FromDouble(double.NegativeInfinity);
        var data = new T[sq * sk];
        for (int q = 0; q < sq; q++)
        {
            int rowBase = q * sk;
            int maxVisible = queryOffset + q;
            for (int k = 0; k < sk; k++)
                data[rowBase + k] = k <= maxVisible ? zero : negInf;
        }
        return new Tensor<T>(data, new[] { 1, 1, sq, sk });
    }

    private static Tensor<T> ApplyCausalMask(IEngine engine, Tensor<T> scores, INumericOperations<T> numOps, int queryOffset = 0)
    {
        // Offset-aware causal mask: positions k > queryOffset + q are masked.
        int[] shape = scores._shape;
        int b = shape[0], h = shape[1], sq = shape[2], sk = shape[3];
        if (!scores.IsContiguous) scores = scores.Contiguous();
        var data = scores.GetDataArray();
        T negInf = numOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < b; i++)
        {
            for (int j = 0; j < h; j++)
            {
                int baseIdx = ((i * h) + j) * sq * sk;
                for (int q = 0; q < sq; q++)
                {
                    int rowBase = baseIdx + q * sk;
                    int firstMasked = queryOffset + q + 1;
                    for (int k = firstMasked; k < sk; k++)
                        data[rowBase + k] = negInf;
                }
            }
        }
        return scores;
    }

    /// <summary>
    /// Backward pass — computes dQ / dK / dV given gradOutput, saved
    /// (Q, K, V, Output) from forward, and the same config used at forward.
    /// Uses the analytic formulation of softmax + matmul chain-rule; no
    /// reliance on the engine's autodiff tape, so it composes with any
    /// external gradient framework.
    ///
    /// <para><b>Derivation:</b> let <c>S = Q K^T * scale (+ bias)</c>,
    /// <c>P = softmax(S)</c>, <c>O = P V</c>. Then:
    /// <code>
    /// dV = P^T dO
    /// dP = dO V^T
    /// dS = softmax_backward(dP, P) = P * (dP - sum_row(dP * P))
    /// dQ = dS K * scale
    /// dK = dS^T Q * scale
    /// </code>
    /// </para>
    ///
    /// <para>Memory: this reference implementation materialises the full
    /// <c>[B, H, Sq, Sk]</c> attention-weight matrix. A tiled online-
    /// softmax backward that matches FlashAttention-2's O(seqLen) memory
    /// cost is future work; this version is sufficient for correctness
    /// testing and for small/medium sequences.</para>
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// <typeparamref name="T"/> is not a floating-point type
    /// (<c>float</c>, <c>double</c>, or <c>System.Half</c>).
    /// </exception>
    public static (Tensor<T> GradQuery, Tensor<T> GradKey, Tensor<T> GradValue) Backward(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        FlashAttentionConfig? config = null,
        Tensor<T>? attentionBias = null,
        IEngine? engine = null)
    {
        EnsureSupportedElementType();
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        config ??= new FlashAttentionConfig();
        engine ??= new CpuEngine();

        bool was3D = query.Rank == 3;
        if (was3D)
        {
            query = PromoteToFourD(engine, query);
            key = PromoteToFourD(engine, key);
            value = PromoteToFourD(engine, value);
            gradOutput = PromoteToFourD(engine, gradOutput);
        }

        // Recompute the forward softmax to get P — we need it for every
        // gradient below. Same pipeline as Forward's bias path.
        var numOps = MathHelper.GetNumericOperations<T>();
        int B = query._shape[0], H = query._shape[1];
        int Sq = query._shape[2], headDim = query._shape[3];
        int Sk = key._shape[2], Dv = value._shape[3];
        double scaleVal = config.Scale ?? 1.0 / Math.Sqrt(headDim);
        T scaleT = numOps.FromDouble(scaleVal);

        var qFlat  = engine.Reshape(query, new[] { B * H, Sq, headDim });
        var kFlat  = engine.Reshape(key,   new[] { B * H, Sk, headDim });
        var vFlat  = engine.Reshape(value, new[] { B * H, Sk, Dv });
        var kFlatT = kFlat.TransposeLast2D();

        var scoresFlat = engine.TensorBatchMatMul(qFlat, kFlatT);
        scoresFlat = engine.TensorMultiplyScalar(scoresFlat, scaleT);
        var scores = engine.Reshape(scoresFlat, new[] { B, H, Sq, Sk });
        if (attentionBias is not null)
            scores = AddBroadcastBias(engine, scores, attentionBias);
        else if (config.IsCausal)
            scores = ApplyCausalMask(engine, scores, numOps, config.QueryOffset);
        else if (config.IsCausal && attentionBias is not null)
            scores = ApplyCausalMask(engine, scores, numOps, config.QueryOffset);
        var P = engine.TensorSoftmax(scores, axis: 3);                           // [B, H, Sq, Sk]

        var pFlat = engine.Reshape(P, new[] { B * H, Sq, Sk });
        var pFlatT = pFlat.TransposeLast2D();                                    // [B*H, Sk, Sq]
        var goFlat = engine.Reshape(gradOutput, new[] { B * H, Sq, Dv });

        // dV = P^T @ dO                                                        [B*H, Sk, Dv]
        var dvFlat = engine.TensorBatchMatMul(pFlatT, goFlat);

        // dP = dO @ V^T                                                        [B*H, Sq, Sk]
        var vFlatT = vFlat.TransposeLast2D();
        var dpFlat = engine.TensorBatchMatMul(goFlat, vFlatT);

        // dS = P * (dP - rowsum(dP * P)) — softmax backward
        var dP = engine.Reshape(dpFlat, new[] { B, H, Sq, Sk });
        var dS = SoftmaxBackward(engine, numOps, dP, P);

        var dsFlat = engine.Reshape(dS, new[] { B * H, Sq, Sk });

        // dQ = dS @ K * scale                                                  [B*H, Sq, headDim]
        var dqFlat = engine.TensorBatchMatMul(dsFlat, kFlat);
        dqFlat = engine.TensorMultiplyScalar(dqFlat, scaleT);

        // dK = dS^T @ Q * scale                                                [B*H, Sk, headDim]
        var dsFlatT = dsFlat.TransposeLast2D();
        var dkFlat = engine.TensorBatchMatMul(dsFlatT, qFlat);
        dkFlat = engine.TensorMultiplyScalar(dkFlat, scaleT);

        var gradQ = engine.Reshape(dqFlat, new[] { B, H, Sq, headDim });
        var gradK = engine.Reshape(dkFlat, new[] { B, H, Sk, headDim });
        var gradV = engine.Reshape(dvFlat, new[] { B, H, Sk, Dv });

        if (was3D)
        {
            gradQ = DemoteToThreeD(engine, gradQ);
            gradK = DemoteToThreeD(engine, gradK);
            gradV = DemoteToThreeD(engine, gradV);
        }
        return (gradQ, gradK, gradV);
    }

    /// <summary>
    /// Softmax backward: dS = P * (dP - sum_row(dP * P)).
    /// Implemented element-wise because TensorSoftmaxBackward is tape-
    /// aware and we're outside the tape here.
    /// </summary>
    private static Tensor<T> SoftmaxBackward(IEngine engine, INumericOperations<T> numOps, Tensor<T> dP, Tensor<T> P)
    {
        if (!dP.IsContiguous) dP = dP.Contiguous();
        if (!P.IsContiguous) P = P.Contiguous();
        var shape = P._shape;
        int b = shape[0], h = shape[1], sq = shape[2], sk = shape[3];
        var pData = P.GetDataArray();
        var dpData = dP.GetDataArray();
        var dsData = new T[pData.Length];
        for (int i = 0; i < b; i++)
        {
            for (int j = 0; j < h; j++)
            {
                int batchBase = ((i * h) + j) * sq * sk;
                for (int q = 0; q < sq; q++)
                {
                    int rowBase = batchBase + q * sk;
                    // rowSum = Σ_k (dP[row, k] * P[row, k])
                    T rowSum = numOps.Zero;
                    for (int k = 0; k < sk; k++)
                        rowSum = numOps.Add(rowSum,
                            numOps.Multiply(dpData[rowBase + k], pData[rowBase + k]));
                    for (int k = 0; k < sk; k++)
                        dsData[rowBase + k] = numOps.Multiply(
                            pData[rowBase + k],
                            numOps.Subtract(dpData[rowBase + k], rowSum));
                }
            }
        }
        return new Tensor<T>(dsData, shape);
    }
}
