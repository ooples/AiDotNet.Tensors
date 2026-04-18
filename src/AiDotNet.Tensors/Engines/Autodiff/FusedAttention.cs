using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Config for <see cref="FusedAttention{T}.Forward"/> — softmax scale,
/// causal mask, optional block-size override, whether to return the
/// (large) attention-weight matrix.
///
/// <para>Uses nullable properties + industry-standard defaults so a
/// zero-config call still runs sensibly. Defaults match PyTorch's
/// <c>torch.nn.functional.scaled_dot_product_attention</c>: scale is
/// <c>1/sqrt(headDim)</c>, causal is false, block size is auto (picked
/// by the kernel based on seqLen).</para>
/// </summary>
public sealed class FlashAttentionConfig
{
    /// <summary>Softmax scale. When null, defaults to <c>1 / sqrt(headDim)</c>.</summary>
    public double? Scale { get; set; }

    /// <summary>When true, applies the autoregressive causal mask.</summary>
    public bool IsCausal { get; set; }

    /// <summary>When true, returns the attention-weight matrix. Allocates
    /// an extra <c>[B, H, seqQ, seqK]</c> tensor; leave false for inference
    /// when you only need the output.</summary>
    public bool ReturnAttentionWeights { get; set; }

    /// <summary>Block size for the block-tiled softmax. Null means "let
    /// the kernel pick" — chosen to balance L2 cache residency against
    /// softmax accumulation precision. Exposed so benchmarks can sweep.</summary>
    public int? BlockSize { get; set; }
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
    /// <summary>
    /// Forward pass. Returns (Output, AttentionWeights) — weights is null
    /// unless <see cref="FlashAttentionConfig.ReturnAttentionWeights"/> is
    /// set.
    /// </summary>
    public static (Tensor<T> Output, Tensor<T>? AttentionWeights) Forward(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        FlashAttentionConfig? config = null,
        Tensor<T>? attentionBias = null,
        IEngine? engine = null)
    {
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

        // Causal no-bias → synthesize a -inf upper-triangle bias and route
        // through the bias path. Avoids a NotImplementedException for the
        // common self-attention use case and still lets the bias path reuse
        // the tuned GEMM chain. A weights-less engine overload + block-tiled
        // causal fast path is a follow-up; this path keeps the API complete.
        if (attentionBias is null && config.IsCausal)
        {
            attentionBias = BuildCausalBias(engine, query._shape[2], key._shape[2]);
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
        if (config.IsCausal)
            scores = ApplyCausalMask(engine, scores, numOps);
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

    private static Tensor<T> BuildCausalBias(IEngine engine, int sq, int sk)
    {
        // Construct a [1, 1, sq, sk] bias tensor where position (q, k) is
        // 0 when k <= q (visible) and -Inf when k > q (masked). Shape is
        // chosen to broadcast over batch + head in AddBroadcastBias without
        // a per-(B, H) allocation. Used by the no-bias causal path to reuse
        // the bias pipeline rather than maintaining a second causal codepath.
        var numOps = MathHelper.GetNumericOperations<T>();
        T zero = numOps.Zero;
        T negInf = numOps.FromDouble(double.NegativeInfinity);
        var data = new T[sq * sk];
        for (int q = 0; q < sq; q++)
        {
            int rowBase = q * sk;
            for (int k = 0; k < sk; k++)
                data[rowBase + k] = k <= q ? zero : negInf;
        }
        return new Tensor<T>(data, new[] { 1, 1, sq, sk });
    }

    private static Tensor<T> ApplyCausalMask(IEngine engine, Tensor<T> scores, INumericOperations<T> numOps)
    {
        // Causal mask: add -Inf to upper-triangular positions. Scores shape:
        // [B, H, Sq, Sk]. We apply the mask column-wise via a manual pass —
        // the engine doesn't expose a dedicated causal op yet.
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
                    // Future positions (k > q) are masked.
                    for (int k = q + 1; k < sk; k++)
                        data[rowBase + k] = negInf;
                }
            }
        }
        return scores;
    }
}
