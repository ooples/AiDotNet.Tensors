using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX recurrent operator translators: LSTM and GRU. Implemented by
/// unrolling the timestep loop at import time into plain matmul +
/// activation engine ops — static ONNX graphs don't have a loop primitive
/// we can compile into, so every timestep becomes its own set of steps
/// in the resulting plan.
///
/// <para>Phase 2 scope (Issue #169 follow-up): forward direction only, no
/// peepholes, no sequence_lens (packed sequences), default activations.
/// Layout = 0 (seq-first). Bidirectional / reverse-direction / custom
/// activations throw NotSupportedException.</para>
/// </summary>
internal static class RecurrentOperators
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new Lstm<T>());
        r.Register(new Gru<T>());
    }

    /// <summary>
    /// ONNX LSTM — unrolls into per-timestep gate computations.
    ///
    /// <para>Gate order in W/R matrices is <b>[i, o, f, c]</b> (ONNX spec),
    /// distinct from PyTorch's [i, f, g, o]. We split W/R into four
    /// hidden×* slices and compute:
    /// <code>
    /// it = sigmoid(Wi·x_t + Ri·h_{t-1} + bi)
    /// ft = sigmoid(Wf·x_t + Rf·h_{t-1} + bf)
    /// ct_tilde = tanh(Wc·x_t + Rc·h_{t-1} + bc)
    /// ot = sigmoid(Wo·x_t + Ro·h_{t-1} + bo)
    /// c_t = ft · c_{t-1} + it · ct_tilde
    /// h_t = ot · tanh(c_t)
    /// </code></para>
    ///
    /// <para>Outputs Y and Y_h are produced; Y_c is optional. Y packs every
    /// timestep's hidden along the seq axis with num_directions = 1.</para>
    /// </summary>
    internal sealed class Lstm<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "LSTM";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            ValidateAttributes(ctx, node, "LSTM");
            int hiddenSize = ctx.GetIntAttrAsInt(node, "hidden_size", 0);
            if (hiddenSize <= 0) throw new InvalidDataException("LSTM requires hidden_size > 0.");

            // Required inputs: X, W, R
            var x = ctx.GetTensor(node.Input[0]);                   // [seq, batch, input]
            var wParam = ctx.GetTensor(node.Input[1]);              // [1, 4H, input]
            var rParam = ctx.GetTensor(node.Input[2]);              // [1, 4H, H]
            // Optional inputs (by position): B, sequence_lens, initial_h, initial_c, P
            Tensor<T>? bParam = InputOrNull(ctx, node, 3);          // [1, 8H]
            Tensor<T>? seqLens = InputOrNull(ctx, node, 4);
            Tensor<T>? initialH = InputOrNull(ctx, node, 5);        // [1, batch, H]
            Tensor<T>? initialC = InputOrNull(ctx, node, 6);        // [1, batch, H]
            Tensor<T>? peepholes = InputOrNull(ctx, node, 7);
            if (seqLens is not null)
                throw new NotSupportedException("LSTM sequence_lens (packed sequences) is not yet supported.");
            if (peepholes is not null)
                throw new NotSupportedException("LSTM peepholes are not yet supported.");

            int seqLen = x._shape[0];
            int batch = x._shape[1];
            int inputSize = x._shape[2];

            // Squeeze the num_directions dim (= 1) from W, R, B, initial_h, initial_c
            // by reshaping. These tensors are frozen initializers so the reshape
            // is free.
            var w = ctx.Engine.Reshape(wParam, new[] { 4 * hiddenSize, inputSize });
            var r = ctx.Engine.Reshape(rParam, new[] { 4 * hiddenSize, hiddenSize });
            Tensor<T>? bCombined = bParam is null ? null : ctx.Engine.Reshape(bParam, new[] { 8 * hiddenSize });

            // Split W into per-gate slices of shape [H, inputSize] and R into
            // [H, hiddenSize]. ONNX packs as [i, o, f, c]; our iteration uses
            // i, o, f, c.
            var gateWs = SplitRows(ctx, w, 4, hiddenSize);
            var gateRs = SplitRows(ctx, r, 4, hiddenSize);
            var gateBs = bCombined is null
                ? (null, null, null, null, null, null, null, null)
                : SplitBias(ctx, bCombined, hiddenSize);

            // Hidden / cell state initializers default to zeros when not
            // provided. Reshape to drop num_directions=1 → [batch, H].
            var hPrev = initialH is null
                ? ZeroTensor<T>(ctx, new[] { batch, hiddenSize })
                : ctx.Engine.Reshape(initialH, new[] { batch, hiddenSize });
            var cPrev = initialC is null
                ? ZeroTensor<T>(ctx, new[] { batch, hiddenSize })
                : ctx.Engine.Reshape(initialC, new[] { batch, hiddenSize });

            // Per-timestep outputs collected along seq axis.
            var yTimesteps = new List<Tensor<T>>(seqLen);

            // Pre-transpose W, R so gemm(x_t, Wgate^T) is plain matmul with
            // layouts that give row-major outputs (no TensorPermute in the
            // timestep loop).
            var (wi, wo, wf, wc) = gateWs;
            var (ri, ro, rf, rc) = gateRs;

            var wiT = ctx.Engine.TensorTranspose(wi);
            var woT = ctx.Engine.TensorTranspose(wo);
            var wfT = ctx.Engine.TensorTranspose(wf);
            var wcT = ctx.Engine.TensorTranspose(wc);
            var riT = ctx.Engine.TensorTranspose(ri);
            var roT = ctx.Engine.TensorTranspose(ro);
            var rfT = ctx.Engine.TensorTranspose(rf);
            var rcT = ctx.Engine.TensorTranspose(rc);

            for (int t = 0; t < seqLen; t++)
            {
                // x_t: slice [t, :, :] and drop the leading seq dim.
                var xtSlice = ctx.Engine.TensorSlice(x,
                    start: new[] { t, 0, 0 },
                    length: new[] { 1, batch, inputSize });
                var xt = ctx.Engine.Reshape(xtSlice, new[] { batch, inputSize });

                // Each gate: gate = sigmoid/tanh(xt·Wgate^T + hprev·Rgate^T + Wbgate + Rbgate)
                var it = GateWithActivation(ctx, xt, hPrev, wiT, riT, gateBs.Item1, gateBs.Item5, sigmoid: true);
                var ft = GateWithActivation(ctx, xt, hPrev, wfT, rfT, gateBs.Item3, gateBs.Item7, sigmoid: true);
                var ot = GateWithActivation(ctx, xt, hPrev, woT, roT, gateBs.Item2, gateBs.Item6, sigmoid: true);
                var ctTilde = GateWithActivation(ctx, xt, hPrev, wcT, rcT, gateBs.Item4, gateBs.Item8, sigmoid: false);

                // c_t = ft * c_{t-1} + it * ct_tilde
                var cForget = ctx.Engine.TensorMultiply(ft, cPrev);
                var cNew = ctx.Engine.TensorMultiply(it, ctTilde);
                cPrev = ctx.Engine.TensorAdd(cForget, cNew);

                // h_t = ot * tanh(c_t)
                var cTanh = ctx.Engine.Tanh(cPrev);
                hPrev = ctx.Engine.TensorMultiply(ot, cTanh);

                // Reshape h_t into [1, 1, batch, H] so we can concat along
                // the seq axis (ONNX Y shape is [seq, num_directions, batch, H]).
                yTimesteps.Add(ctx.Engine.Reshape(hPrev, new[] { 1, 1, batch, hiddenSize }));
            }

            // Produce outputs. Y is present when node has an output name at
            // index 0 (may be empty string to skip); Y_h at index 1; Y_c at
            // index 2.
            if (node.Output.Count > 0 && !string.IsNullOrEmpty(node.Output[0]))
            {
                var y = ctx.Engine.Concat(yTimesteps, axis: 0);
                ctx.PutTensor(node.Output[0], y);
            }
            if (node.Output.Count > 1 && !string.IsNullOrEmpty(node.Output[1]))
            {
                ctx.PutTensor(node.Output[1],
                    ctx.Engine.Reshape(hPrev, new[] { 1, batch, hiddenSize }));
            }
            if (node.Output.Count > 2 && !string.IsNullOrEmpty(node.Output[2]))
            {
                ctx.PutTensor(node.Output[2],
                    ctx.Engine.Reshape(cPrev, new[] { 1, batch, hiddenSize }));
            }
        }
    }

    /// <summary>
    /// ONNX GRU — gate order is <b>[z, r, h]</b> (update, reset, hidden).
    /// <code>
    /// zt = sigmoid(Wz·x_t + Rz·h_{t-1} + bz)
    /// rt = sigmoid(Wr·x_t + Rr·h_{t-1} + br)
    /// ht_tilde = tanh(Wh·x_t + rt·(Rh·h_{t-1}) + bh)   // linear_before_reset=0 (default)
    /// h_t = (1 - zt) · ht_tilde + zt · h_{t-1}
    /// </code>
    /// The <c>linear_before_reset=1</c> variant applies the reset gate on
    /// the pre-linear recurrent projection (<c>rt·Rh·h_{t-1}</c>) vs the
    /// default post-linear (<c>Rh·(rt·h_{t-1})</c>) — both forms accepted.
    /// </summary>
    internal sealed class Gru<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "GRU";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            ValidateAttributes(ctx, node, "GRU");
            int hiddenSize = ctx.GetIntAttrAsInt(node, "hidden_size", 0);
            if (hiddenSize <= 0) throw new InvalidDataException("GRU requires hidden_size > 0.");
            int linearBeforeReset = ctx.GetIntAttrAsInt(node, "linear_before_reset", 0);

            var x = ctx.GetTensor(node.Input[0]);                   // [seq, batch, input]
            var wParam = ctx.GetTensor(node.Input[1]);              // [1, 3H, input]
            var rParam = ctx.GetTensor(node.Input[2]);              // [1, 3H, H]
            Tensor<T>? bParam = InputOrNull(ctx, node, 3);          // [1, 6H]
            Tensor<T>? seqLens = InputOrNull(ctx, node, 4);
            Tensor<T>? initialH = InputOrNull(ctx, node, 5);
            if (seqLens is not null)
                throw new NotSupportedException("GRU sequence_lens is not yet supported.");

            int seqLen = x._shape[0];
            int batch = x._shape[1];
            int inputSize = x._shape[2];

            var w = ctx.Engine.Reshape(wParam, new[] { 3 * hiddenSize, inputSize });
            var r = ctx.Engine.Reshape(rParam, new[] { 3 * hiddenSize, hiddenSize });
            Tensor<T>? bCombined = bParam is null ? null : ctx.Engine.Reshape(bParam, new[] { 6 * hiddenSize });

            // Split into per-gate. ONNX order: z, r, h.
            var gateWs = SplitRows(ctx, w, 3, hiddenSize, _forGru: true);
            var gateRs = SplitRows(ctx, r, 3, hiddenSize, _forGru: true);

            Tensor<T>? bwz = null, bwr = null, bwh = null, brz = null, brr = null, brh = null;
            if (bCombined is not null)
            {
                bwz = SliceBias(ctx, bCombined, 0, hiddenSize);
                bwr = SliceBias(ctx, bCombined, 1, hiddenSize);
                bwh = SliceBias(ctx, bCombined, 2, hiddenSize);
                brz = SliceBias(ctx, bCombined, 3, hiddenSize);
                brr = SliceBias(ctx, bCombined, 4, hiddenSize);
                brh = SliceBias(ctx, bCombined, 5, hiddenSize);
            }

            var hPrev = initialH is null
                ? ZeroTensor<T>(ctx, new[] { batch, hiddenSize })
                : ctx.Engine.Reshape(initialH, new[] { batch, hiddenSize });

            var yTimesteps = new List<Tensor<T>>(seqLen);
            var (wz3, wr3, wh3) = (gateWs.Item1, gateWs.Item2, gateWs.Item3);
            var (rz3, rr3, rh3) = (gateRs.Item1, gateRs.Item2, gateRs.Item3);
            var wzT = ctx.Engine.TensorTranspose(wz3);
            var wrT = ctx.Engine.TensorTranspose(wr3);
            var whT = ctx.Engine.TensorTranspose(wh3);
            var rzT = ctx.Engine.TensorTranspose(rz3);
            var rrT = ctx.Engine.TensorTranspose(rr3);
            var rhT = ctx.Engine.TensorTranspose(rh3);

            for (int t = 0; t < seqLen; t++)
            {
                var xtSlice = ctx.Engine.TensorSlice(x,
                    start: new[] { t, 0, 0 },
                    length: new[] { 1, batch, inputSize });
                var xt = ctx.Engine.Reshape(xtSlice, new[] { batch, inputSize });

                var zt = GateWithActivation(ctx, xt, hPrev, wzT, rzT, bwz, brz, sigmoid: true);
                var rt = GateWithActivation(ctx, xt, hPrev, wrT, rrT, bwr, brr, sigmoid: true);

                // Candidate hidden. linear_before_reset changes whether we apply
                // rt on h_{t-1} before or after the recurrent projection Rh.
                Tensor<T> hTilde;
                if (linearBeforeReset == 0)
                {
                    // rh_proj = Rh · (rt · h_{t-1})
                    var rtH = ctx.Engine.TensorMultiply(rt, hPrev);
                    hTilde = GateWithActivation(ctx, xt, rtH, whT, rhT, bwh, brh, sigmoid: false);
                }
                else
                {
                    // rh_proj = rt · (Rh · h_{t-1})
                    var rhProj = MatMulWithBias(ctx, hPrev, rhT, brh);
                    var wxh = MatMulWithBias(ctx, xt, whT, bwh);
                    var rtRhProj = ctx.Engine.TensorMultiply(rt, rhProj);
                    var sum = ctx.Engine.TensorAdd(wxh, rtRhProj);
                    hTilde = ctx.Engine.Tanh(sum);
                }

                // h_t = (1 - zt) * hTilde + zt * h_{t-1}
                var oneMinusZ = OneMinus(ctx, zt);
                var term1 = ctx.Engine.TensorMultiply(oneMinusZ, hTilde);
                var term2 = ctx.Engine.TensorMultiply(zt, hPrev);
                hPrev = ctx.Engine.TensorAdd(term1, term2);

                yTimesteps.Add(ctx.Engine.Reshape(hPrev, new[] { 1, 1, batch, hiddenSize }));
            }

            if (node.Output.Count > 0 && !string.IsNullOrEmpty(node.Output[0]))
            {
                var y = ctx.Engine.Concat(yTimesteps, axis: 0);
                ctx.PutTensor(node.Output[0], y);
            }
            if (node.Output.Count > 1 && !string.IsNullOrEmpty(node.Output[1]))
            {
                ctx.PutTensor(node.Output[1],
                    ctx.Engine.Reshape(hPrev, new[] { 1, batch, hiddenSize }));
            }
        }
    }

    // ─── Shared helpers ────────────────────────────────────────────────

    private static Tensor<T>? InputOrNull<T>(OnnxTranslationContext<T> ctx, NodeProto node, int i) where T : unmanaged =>
        node.Input.Count > i && ctx.HasTensor(node.Input[i]) ? ctx.GetTensor(node.Input[i]) : null;

    private static void ValidateAttributes<T>(OnnxTranslationContext<T> ctx, NodeProto node, string opName) where T : unmanaged
    {
        string? direction = ctx.GetStringAttr(node, "direction", "forward");
        if (direction != "forward")
            throw new NotSupportedException(
                $"{opName} direction={direction} is not yet supported. Only 'forward' is covered.");
        int layout = ctx.GetIntAttrAsInt(node, "layout", 0);
        if (layout != 0)
            throw new NotSupportedException(
                $"{opName} layout={layout} is not yet supported. Only seq-first layout=0 is covered.");
    }

    /// <summary>
    /// Splits a rank-2 tensor of shape [G·H, N] into <paramref name="numGates"/>
    /// contiguous row-blocks of shape [H, N]. Returned as a tuple so each
    /// gate-slice can be captured locally without indexing.
    /// </summary>
    private static (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>) SplitRows<T>(
        OnnxTranslationContext<T> ctx, Tensor<T> stacked, int numGates, int hiddenSize) where T : unmanaged
    {
        if (numGates == 4)
        {
            var a = ctx.Engine.TensorSlice(stacked, new[] { 0 * hiddenSize, 0 }, new[] { hiddenSize, stacked._shape[1] });
            var b = ctx.Engine.TensorSlice(stacked, new[] { 1 * hiddenSize, 0 }, new[] { hiddenSize, stacked._shape[1] });
            var c = ctx.Engine.TensorSlice(stacked, new[] { 2 * hiddenSize, 0 }, new[] { hiddenSize, stacked._shape[1] });
            var d = ctx.Engine.TensorSlice(stacked, new[] { 3 * hiddenSize, 0 }, new[] { hiddenSize, stacked._shape[1] });
            return (a, b, c, d);
        }
        throw new ArgumentException("SplitRows numGates must be 4 for LSTM.");
    }

    private static (Tensor<T>, Tensor<T>, Tensor<T>) SplitRows<T>(
        OnnxTranslationContext<T> ctx, Tensor<T> stacked, int numGates, int hiddenSize, bool _forGru = true) where T : unmanaged
    {
        if (numGates == 3)
        {
            var a = ctx.Engine.TensorSlice(stacked, new[] { 0 * hiddenSize, 0 }, new[] { hiddenSize, stacked._shape[1] });
            var b = ctx.Engine.TensorSlice(stacked, new[] { 1 * hiddenSize, 0 }, new[] { hiddenSize, stacked._shape[1] });
            var c = ctx.Engine.TensorSlice(stacked, new[] { 2 * hiddenSize, 0 }, new[] { hiddenSize, stacked._shape[1] });
            return (a, b, c);
        }
        throw new ArgumentException("SplitRows numGates must be 3 for GRU.");
    }

    /// <summary>
    /// Split LSTM's [8H] combined bias into (Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc)
    /// following ONNX's [i, o, f, c] gate order. First 4H is W bias; next 4H
    /// is R bias.
    /// </summary>
    private static (Tensor<T>?, Tensor<T>?, Tensor<T>?, Tensor<T>?, Tensor<T>?, Tensor<T>?, Tensor<T>?, Tensor<T>?) SplitBias<T>(
        OnnxTranslationContext<T> ctx, Tensor<T> bCombined, int hiddenSize) where T : unmanaged
    {
        var wbi = SliceBias(ctx, bCombined, 0, hiddenSize);
        var wbo = SliceBias(ctx, bCombined, 1, hiddenSize);
        var wbf = SliceBias(ctx, bCombined, 2, hiddenSize);
        var wbc = SliceBias(ctx, bCombined, 3, hiddenSize);
        var rbi = SliceBias(ctx, bCombined, 4, hiddenSize);
        var rbo = SliceBias(ctx, bCombined, 5, hiddenSize);
        var rbf = SliceBias(ctx, bCombined, 6, hiddenSize);
        var rbc = SliceBias(ctx, bCombined, 7, hiddenSize);
        return (wbi, wbo, wbf, wbc, rbi, rbo, rbf, rbc);
    }

    private static Tensor<T> SliceBias<T>(OnnxTranslationContext<T> ctx, Tensor<T> b, int slot, int hiddenSize) where T : unmanaged =>
        ctx.Engine.TensorSlice(b, new[] { slot * hiddenSize }, new[] { hiddenSize });

    /// <summary>
    /// Compute gate = activation(xt · Wgate^T + hprev · Rgate^T + Wbgate + Rbgate).
    /// Pre-transposed <paramref name="wGateT"/> and <paramref name="rGateT"/>
    /// let us use plain MatMul for both halves instead of a transposed MatMul.
    /// </summary>
    private static Tensor<T> GateWithActivation<T>(
        OnnxTranslationContext<T> ctx,
        Tensor<T> xt, Tensor<T> hPrev,
        Tensor<T> wGateT, Tensor<T> rGateT,
        Tensor<T>? wBias, Tensor<T>? rBias,
        bool sigmoid) where T : unmanaged
    {
        var xw = MatMulWithBias(ctx, xt, wGateT, wBias);
        var hr = MatMulWithBias(ctx, hPrev, rGateT, rBias);
        var sum = ctx.Engine.TensorAdd(xw, hr);
        return sigmoid ? ctx.Engine.Sigmoid(sum) : ctx.Engine.Tanh(sum);
    }

    private static Tensor<T> MatMulWithBias<T>(
        OnnxTranslationContext<T> ctx, Tensor<T> a, Tensor<T> bT, Tensor<T>? bias) where T : unmanaged
    {
        var mm = ctx.Engine.TensorMatMul(a, bT);
        return bias is null ? mm : ctx.Engine.TensorBroadcastAdd(mm, bias);
    }

    private static Tensor<T> OneMinus<T>(OnnxTranslationContext<T> ctx, Tensor<T> t) where T : unmanaged
    {
        // Materialize a same-shape ones tensor and subtract.
        var ones = new Tensor<T>(t._shape);
        var span = ones.AsWritableSpan();
        var one = Helpers.MathHelper.GetNumericOperations<T>().FromDouble(1.0);
        for (int i = 0; i < span.Length; i++) span[i] = one;
        return ctx.Engine.TensorSubtract(ones, t);
    }

    private static Tensor<T> ZeroTensor<T>(OnnxTranslationContext<T> ctx, int[] shape) where T : unmanaged
    {
        // Tensor<T>(shape) zero-initializes. Return a traced reference via
        // Reshape so downstream ops see a LazyNode input under GraphMode
        // (engine ops consume LazySource to build the graph; a freshly-
        // allocated Tensor has no LazySource but that's fine — it's a leaf
        // like any other initializer).
        return new Tensor<T>(shape);
    }
}
