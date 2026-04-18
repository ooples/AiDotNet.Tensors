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
            string direction = ctx.GetStringAttr(node, "direction", "forward") ?? "forward";
            int numDirs = direction == "bidirectional" ? 2 : 1;
            int layout = ctx.GetIntAttrAsInt(node, "layout", 0);
            int inputForget = ctx.GetIntAttrAsInt(node, "input_forget", 0);
            float? clip = null;
            var clipAttr = ctx.GetAttribute(node, "clip");
            if (clipAttr is not null) clip = clipAttr.F;

            // Required inputs: X, W, R
            var xRaw = ctx.GetTensor(node.Input[0]);
            // layout=1 swaps the first two dims to [batch, seq, input]; we
            // transpose to [seq, batch, input] for the internal loop.
            var x = layout == 0 ? xRaw : ctx.Engine.TensorPermute(xRaw, new[] { 1, 0, 2 });
            var wParam = ctx.GetTensor(node.Input[1]);              // [numDirs, 4H, input]
            var rParam = ctx.GetTensor(node.Input[2]);              // [numDirs, 4H, H]
            Tensor<T>? bParam = InputOrNull(ctx, node, 3);          // [numDirs, 8H]
            Tensor<T>? seqLens = InputOrNull(ctx, node, 4);         // [batch]
            Tensor<T>? initialH = InputOrNull(ctx, node, 5);        // [numDirs, batch, H] (or layout=1)
            Tensor<T>? initialC = InputOrNull(ctx, node, 6);        // [numDirs, batch, H]
            Tensor<T>? peepholes = InputOrNull(ctx, node, 7);       // [numDirs, 3H]

            int seqLen = x._shape[0];
            int batch = x._shape[1];
            int inputSize = x._shape[2];
            if (wParam._shape[0] != numDirs)
                throw new InvalidDataException(
                    $"LSTM W first dim {wParam._shape[0]} doesn't match direction count {numDirs}.");

            int[]? seqLensPerBatch = null;
            if (seqLens is not null) seqLensPerBatch = ToIntsCeil(seqLens);

            // Peepholes: [numDirs, 3H] — P_i, P_o, P_f (ONNX packing order).
            Tensor<T>? peepholesAll = peepholes is null ? null : ctx.Engine.Reshape(peepholes, new[] { numDirs, 3 * hiddenSize });

            // Initial states may be provided in layout=1 form [batch, numDirs, H];
            // normalize to layout=0 [numDirs, batch, H].
            if (layout == 1 && initialH is not null)
                initialH = ctx.Engine.TensorPermute(initialH, new[] { 1, 0, 2 });
            if (layout == 1 && initialC is not null)
                initialC = ctx.Engine.TensorPermute(initialC, new[] { 1, 0, 2 });

            var yPerDir = new List<Tensor<T>>(numDirs);
            var yHPerDir = new List<Tensor<T>>(numDirs);
            var yCPerDir = new List<Tensor<T>>(numDirs);
            for (int d = 0; d < numDirs; d++)
            {
                bool reverse = (direction == "reverse") || (direction == "bidirectional" && d == 1);
                // Slice per-direction weights.
                var w = SliceAlongFirstAxisAndSqueeze(ctx, wParam, d, new[] { 4 * hiddenSize, inputSize });
                var r = SliceAlongFirstAxisAndSqueeze(ctx, rParam, d, new[] { 4 * hiddenSize, hiddenSize });
                Tensor<T>? bCombined = bParam is null ? null
                    : SliceAlongFirstAxisAndSqueeze(ctx, bParam, d, new[] { 8 * hiddenSize });
                Tensor<T>? pThis = peepholesAll is null ? null
                    : SliceAlongFirstAxisAndSqueeze(ctx, peepholesAll, d, new[] { 3 * hiddenSize });

                var gateWs = SplitRows(ctx, w, 4, hiddenSize);
                var gateRs = SplitRows(ctx, r, 4, hiddenSize);
                var gateBs = bCombined is null
                    ? (null, null, null, null, null, null, null, null)
                    : SplitBias(ctx, bCombined, hiddenSize);

                // Peepholes: ONNX packs [P_i, P_o, P_f]; each is [H]. Reshape
                // to [1, H] so they broadcast against [batch, H] cell state.
                Tensor<T>? pI = null, pO = null, pF = null;
                if (pThis is not null)
                {
                    pI = ctx.Engine.Reshape(SliceAlongFirstAxisAndSqueeze(ctx, ctx.Engine.Reshape(pThis, new[] { 3, hiddenSize }), 0, new[] { hiddenSize }), new[] { 1, hiddenSize });
                    pO = ctx.Engine.Reshape(SliceAlongFirstAxisAndSqueeze(ctx, ctx.Engine.Reshape(pThis, new[] { 3, hiddenSize }), 1, new[] { hiddenSize }), new[] { 1, hiddenSize });
                    pF = ctx.Engine.Reshape(SliceAlongFirstAxisAndSqueeze(ctx, ctx.Engine.Reshape(pThis, new[] { 3, hiddenSize }), 2, new[] { hiddenSize }), new[] { 1, hiddenSize });
                }

                var hPrev = initialH is null
                    ? ZeroTensor<T>(ctx, new[] { batch, hiddenSize })
                    : SliceAlongFirstAxisAndSqueeze(ctx, initialH, d, new[] { batch, hiddenSize });
                var cPrev = initialC is null
                    ? ZeroTensor<T>(ctx, new[] { batch, hiddenSize })
                    : SliceAlongFirstAxisAndSqueeze(ctx, initialC, d, new[] { batch, hiddenSize });

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

                var yTimesteps = new List<Tensor<T>>(seqLen);
                for (int tIdx = 0; tIdx < seqLen; tIdx++)
                {
                    int t = reverse ? (seqLen - 1 - tIdx) : tIdx;
                    var xtSlice = ctx.Engine.TensorSlice(x,
                        start: new[] { t, 0, 0 },
                        length: new[] { 1, batch, inputSize });
                    var xt = ctx.Engine.Reshape(xtSlice, new[] { batch, inputSize });

                    // Pre-activation = xt·Wgate^T + hprev·Rgate^T + Wbgate + Rbgate
                    var iPre = GatePreActivation(ctx, xt, hPrev, wiT, riT, gateBs.Item1, gateBs.Item5);
                    var oPre = GatePreActivation(ctx, xt, hPrev, woT, roT, gateBs.Item2, gateBs.Item6);
                    var fPre = GatePreActivation(ctx, xt, hPrev, wfT, rfT, gateBs.Item3, gateBs.Item7);
                    var cPre = GatePreActivation(ctx, xt, hPrev, wcT, rcT, gateBs.Item4, gateBs.Item8);

                    // Peepholes: P_i * c_{t-1} to i, P_f * c_{t-1} to f.
                    if (pI is not null) iPre = ctx.Engine.TensorAdd(iPre, ctx.Engine.TensorBroadcastMultiply(cPrev, pI));
                    if (pF is not null) fPre = ctx.Engine.TensorAdd(fPre, ctx.Engine.TensorBroadcastMultiply(cPrev, pF));

                    var it = ctx.Engine.Sigmoid(iPre);
                    var ftGate = inputForget != 0 ? OneMinus(ctx, it) : ctx.Engine.Sigmoid(fPre);
                    var ctTilde = ctx.Engine.Tanh(cPre);

                    // c_t = ft * c_{t-1} + it * ct_tilde
                    var cForget = ctx.Engine.TensorMultiply(ftGate, cPrev);
                    var cNew = ctx.Engine.TensorMultiply(it, ctTilde);
                    var cNext = ctx.Engine.TensorAdd(cForget, cNew);

                    // P_o * c_t to o after cell update (ONNX peephole order).
                    if (pO is not null) oPre = ctx.Engine.TensorAdd(oPre, ctx.Engine.TensorBroadcastMultiply(cNext, pO));
                    var ot = ctx.Engine.Sigmoid(oPre);

                    // Clip: clamp cell state to [-clip, clip].
                    if (clip.HasValue)
                        cNext = ClipTensor(ctx, cNext, clip.Value);
                    var cTanh = ctx.Engine.Tanh(cNext);
                    var hNext = ctx.Engine.TensorMultiply(ot, cTanh);

                    // sequence_lens: keep previous h/c for rows where
                    // t >= seq_lens[row].
                    if (seqLensPerBatch is not null)
                    {
                        hNext = BlendByMask(ctx, hNext, hPrev, seqLensPerBatch, t, batch, hiddenSize);
                        cNext = BlendByMask(ctx, cNext, cPrev, seqLensPerBatch, t, batch, hiddenSize);
                    }

                    hPrev = hNext;
                    cPrev = cNext;

                    // Push hNext shaped [1, 1, batch, H]; for reverse we
                    // prepend so the final concat along seq axis is still in
                    // increasing-t order per the ONNX spec.
                    var piece = ctx.Engine.Reshape(hNext, new[] { 1, 1, batch, hiddenSize });
                    if (reverse) yTimesteps.Insert(0, piece);
                    else yTimesteps.Add(piece);
                }

                yPerDir.Add(ctx.Engine.Concat(yTimesteps, axis: 0));
                // Only materialize yH/yC if the node actually declares
                // those outputs — otherwise the unused Reshape ops become
                // leaves in the plan, survive DCE, and the reorderer can
                // place one of them AFTER the output wrap, causing
                // Plan.Execute() (which returns the last step's buffer) to
                // return the wrong tensor.
                bool needYH = node.Output.Count > 1 && !string.IsNullOrEmpty(node.Output[1]);
                bool needYC = node.Output.Count > 2 && !string.IsNullOrEmpty(node.Output[2]);
                yHPerDir.Add(needYH ? ctx.Engine.Reshape(hPrev, new[] { 1, batch, hiddenSize }) : hPrev);
                yCPerDir.Add(needYC ? ctx.Engine.Reshape(cPrev, new[] { 1, batch, hiddenSize }) : cPrev);
            }

            bool wantYH = node.Output.Count > 1 && !string.IsNullOrEmpty(node.Output[1]);
            bool wantYC = node.Output.Count > 2 && !string.IsNullOrEmpty(node.Output[2]);
            if (node.Output.Count > 0 && !string.IsNullOrEmpty(node.Output[0]))
            {
                var y = numDirs == 1 ? yPerDir[0] : ctx.Engine.Concat(yPerDir, axis: 1);
                if (layout == 1) y = ctx.Engine.TensorPermute(y, new[] { 2, 0, 1, 3 });
                ctx.PutTensor(node.Output[0], y);
            }
            if (wantYH)
            {
                var yH = numDirs == 1 ? yHPerDir[0] : ctx.Engine.Concat(yHPerDir, axis: 0);
                if (layout == 1) yH = ctx.Engine.TensorPermute(yH, new[] { 1, 0, 2 });
                ctx.PutTensor(node.Output[1], yH);
            }
            if (wantYC)
            {
                var yC = numDirs == 1 ? yCPerDir[0] : ctx.Engine.Concat(yCPerDir, axis: 0);
                if (layout == 1) yC = ctx.Engine.TensorPermute(yC, new[] { 1, 0, 2 });
                ctx.PutTensor(node.Output[2], yC);
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
            string direction = ctx.GetStringAttr(node, "direction", "forward") ?? "forward";
            int numDirs = direction == "bidirectional" ? 2 : 1;
            int layout = ctx.GetIntAttrAsInt(node, "layout", 0);
            float? clip = null;
            var clipAttr = ctx.GetAttribute(node, "clip");
            if (clipAttr is not null) clip = clipAttr.F;

            var xRaw = ctx.GetTensor(node.Input[0]);
            var x = layout == 0 ? xRaw : ctx.Engine.TensorPermute(xRaw, new[] { 1, 0, 2 });
            var wParam = ctx.GetTensor(node.Input[1]);
            var rParam = ctx.GetTensor(node.Input[2]);
            Tensor<T>? bParam = InputOrNull(ctx, node, 3);
            Tensor<T>? seqLens = InputOrNull(ctx, node, 4);
            Tensor<T>? initialH = InputOrNull(ctx, node, 5);

            int seqLen = x._shape[0];
            int batch = x._shape[1];
            int inputSize = x._shape[2];
            if (wParam._shape[0] != numDirs)
                throw new InvalidDataException(
                    $"GRU W first dim {wParam._shape[0]} doesn't match direction count {numDirs}.");

            int[]? seqLensPerBatch = seqLens is null ? null : ToIntsCeil(seqLens);
            if (layout == 1 && initialH is not null)
                initialH = ctx.Engine.TensorPermute(initialH, new[] { 1, 0, 2 });

            var yPerDir = new List<Tensor<T>>(numDirs);
            var yHPerDir = new List<Tensor<T>>(numDirs);
            for (int d = 0; d < numDirs; d++)
            {
                bool reverse = (direction == "reverse") || (direction == "bidirectional" && d == 1);
                var w = SliceAlongFirstAxisAndSqueeze(ctx, wParam, d, new[] { 3 * hiddenSize, inputSize });
                var r = SliceAlongFirstAxisAndSqueeze(ctx, rParam, d, new[] { 3 * hiddenSize, hiddenSize });
                Tensor<T>? bCombined = bParam is null ? null
                    : SliceAlongFirstAxisAndSqueeze(ctx, bParam, d, new[] { 6 * hiddenSize });

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
                    : SliceAlongFirstAxisAndSqueeze(ctx, initialH, d, new[] { batch, hiddenSize });

                var (wz3, wr3, wh3) = (gateWs.Item1, gateWs.Item2, gateWs.Item3);
                var (rz3, rr3, rh3) = (gateRs.Item1, gateRs.Item2, gateRs.Item3);
                var wzT = ctx.Engine.TensorTranspose(wz3);
                var wrT = ctx.Engine.TensorTranspose(wr3);
                var whT = ctx.Engine.TensorTranspose(wh3);
                var rzT = ctx.Engine.TensorTranspose(rz3);
                var rrT = ctx.Engine.TensorTranspose(rr3);
                var rhT = ctx.Engine.TensorTranspose(rh3);

                var yTimesteps = new List<Tensor<T>>(seqLen);
                for (int tIdx = 0; tIdx < seqLen; tIdx++)
                {
                    int t = reverse ? (seqLen - 1 - tIdx) : tIdx;
                    var xtSlice = ctx.Engine.TensorSlice(x,
                        start: new[] { t, 0, 0 },
                        length: new[] { 1, batch, inputSize });
                    var xt = ctx.Engine.Reshape(xtSlice, new[] { batch, inputSize });

                    var zt = GateWithActivation(ctx, xt, hPrev, wzT, rzT, bwz, brz, sigmoid: true);
                    var rt = GateWithActivation(ctx, xt, hPrev, wrT, rrT, bwr, brr, sigmoid: true);

                    Tensor<T> hTilde;
                    if (linearBeforeReset == 0)
                    {
                        var rtH = ctx.Engine.TensorMultiply(rt, hPrev);
                        hTilde = GateWithActivation(ctx, xt, rtH, whT, rhT, bwh, brh, sigmoid: false);
                    }
                    else
                    {
                        var rhProj = MatMulWithBias(ctx, hPrev, rhT, brh);
                        var wxh = MatMulWithBias(ctx, xt, whT, bwh);
                        var rtRhProj = ctx.Engine.TensorMultiply(rt, rhProj);
                        var sum = ctx.Engine.TensorAdd(wxh, rtRhProj);
                        hTilde = ctx.Engine.Tanh(sum);
                    }

                    var oneMinusZ = OneMinus(ctx, zt);
                    var term1 = ctx.Engine.TensorMultiply(oneMinusZ, hTilde);
                    var term2 = ctx.Engine.TensorMultiply(zt, hPrev);
                    var hNext = ctx.Engine.TensorAdd(term1, term2);
                    if (clip.HasValue) hNext = ClipTensor(ctx, hNext, clip.Value);

                    if (seqLensPerBatch is not null)
                        hNext = BlendByMask(ctx, hNext, hPrev, seqLensPerBatch, t, batch, hiddenSize);
                    hPrev = hNext;

                    var piece = ctx.Engine.Reshape(hNext, new[] { 1, 1, batch, hiddenSize });
                    if (reverse) yTimesteps.Insert(0, piece);
                    else yTimesteps.Add(piece);
                }
                yPerDir.Add(ctx.Engine.Concat(yTimesteps, axis: 0));
                bool needYH = node.Output.Count > 1 && !string.IsNullOrEmpty(node.Output[1]);
                yHPerDir.Add(needYH ? ctx.Engine.Reshape(hPrev, new[] { 1, batch, hiddenSize }) : hPrev);
            }

            bool wantYH = node.Output.Count > 1 && !string.IsNullOrEmpty(node.Output[1]);
            if (node.Output.Count > 0 && !string.IsNullOrEmpty(node.Output[0]))
            {
                var y = numDirs == 1 ? yPerDir[0] : ctx.Engine.Concat(yPerDir, axis: 1);
                if (layout == 1) y = ctx.Engine.TensorPermute(y, new[] { 2, 0, 1, 3 });
                ctx.PutTensor(node.Output[0], y);
            }
            if (wantYH)
            {
                var yH = numDirs == 1 ? yHPerDir[0] : ctx.Engine.Concat(yHPerDir, axis: 0);
                if (layout == 1) yH = ctx.Engine.TensorPermute(yH, new[] { 1, 0, 2 });
                ctx.PutTensor(node.Output[1], yH);
            }
        }
    }

    // ─── Shared helpers ────────────────────────────────────────────────

    private static Tensor<T>? InputOrNull<T>(OnnxTranslationContext<T> ctx, NodeProto node, int i) where T : unmanaged =>
        node.Input.Count > i && ctx.HasTensor(node.Input[i]) ? ctx.GetTensor(node.Input[i]) : null;

    private static void ValidateAttributes<T>(OnnxTranslationContext<T> ctx, NodeProto node, string opName) where T : unmanaged
    {
        // direction, layout, clip, input_forget, and linear_before_reset are
        // now supported (plumbed through the per-timestep loop below).
        // Custom activations (per-gate override) and activation_alpha /
        // activation_beta remain unsupported — honoring those would require
        // per-gate activation dispatch that our current fixed
        // sigmoid/tanh/tanh (LSTM) / sigmoid/tanh (GRU) engine path doesn't
        // expose. We keep those rejections so a model requesting e.g. ReLU
        // gates fails loud instead of silently producing sigmoid output.
        var activations = ctx.GetAttribute(node, "activations");
        if (activations is not null && activations.Strings.Count > 0)
        {
            // Accept the default set explicitly (["Sigmoid", "Tanh", "Tanh"]
            // for LSTM, ["Sigmoid", "Tanh"] for GRU — one entry per
            // direction-gate-set). Any deviation throws.
            var expected = opName == "LSTM"
                ? new[] { "Sigmoid", "Tanh", "Tanh" }
                : new[] { "Sigmoid", "Tanh" };
            string direction = ctx.GetStringAttr(node, "direction", "forward") ?? "forward";
            int dirCount = direction == "bidirectional" ? 2 : 1;
            int expectedLen = expected.Length * dirCount;
            if (activations.Strings.Count != expectedLen)
                throw new NotSupportedException(
                    $"{opName} 'activations' attribute with {activations.Strings.Count} entries " +
                    $"(expected {expectedLen} for direction={direction}) is not yet supported.");
            for (int i = 0; i < activations.Strings.Count; i++)
            {
                var got = System.Text.Encoding.UTF8.GetString(activations.Strings[i].ToByteArray());
                var want = expected[i % expected.Length];
                if (!string.Equals(got, want, StringComparison.OrdinalIgnoreCase))
                    throw new NotSupportedException(
                        $"{opName} 'activations[{i}]={got}' differs from the default '{want}' — " +
                        "per-gate activation override is not yet supported.");
            }
        }
        var alpha = ctx.GetAttribute(node, "activation_alpha");
        if (alpha is not null && alpha.Floats.Count > 0)
            throw new NotSupportedException(
                $"{opName} 'activation_alpha' attribute is not yet supported (requires per-gate " +
                "parametric activations, which our default sigmoid/tanh set doesn't accept).");
        var beta = ctx.GetAttribute(node, "activation_beta");
        if (beta is not null && beta.Floats.Count > 0)
            throw new NotSupportedException(
                $"{opName} 'activation_beta' attribute is not yet supported.");
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

    /// <summary>
    /// Computes the pre-activation sum <c>xt · wT + hPrev · rT + wBias + rBias</c>
    /// without applying sigmoid/tanh. Used by the bidirectional LSTM loop
    /// which needs the raw pre-activation to add peephole contributions.
    /// </summary>
    private static Tensor<T> GatePreActivation<T>(
        OnnxTranslationContext<T> ctx,
        Tensor<T> xt, Tensor<T> hPrev,
        Tensor<T> wGateT, Tensor<T> rGateT,
        Tensor<T>? wBias, Tensor<T>? rBias) where T : unmanaged
    {
        var xw = MatMulWithBias(ctx, xt, wGateT, wBias);
        var hr = MatMulWithBias(ctx, hPrev, rGateT, rBias);
        return ctx.Engine.TensorAdd(xw, hr);
    }

    /// <summary>
    /// Slices <paramref name="t"/> along axis 0 at <paramref name="index"/>
    /// and reshapes the result to <paramref name="squeezed"/> — dropping
    /// the leading singleton dim. Used to slice per-direction weight/state
    /// tensors of shape [numDirs, ...] down to the direction's own copy.
    /// </summary>
    private static Tensor<T> SliceAlongFirstAxisAndSqueeze<T>(
        OnnxTranslationContext<T> ctx, Tensor<T> t, int index, int[] squeezed) where T : unmanaged
    {
        var start = new int[t.Rank];
        var length = (int[])t._shape.Clone();
        start[0] = index;
        length[0] = 1;
        var sliced = ctx.Engine.TensorSlice(t, start, length);
        return ctx.Engine.Reshape(sliced, squeezed);
    }

    /// <summary>
    /// Clamps every element of <paramref name="t"/> to
    /// <c>[-clip, clip]</c>. Implemented via <c>TensorMax(lo, TensorMin(hi, t))</c>
    /// fallback when the engine's native Clamp isn't exposed.
    /// </summary>
    private static Tensor<T> ClipTensor<T>(OnnxTranslationContext<T> ctx, Tensor<T> t, float clip) where T : unmanaged
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        var hiT = new Tensor<T>(t._shape);
        var loT = new Tensor<T>(t._shape);
        var hiSpan = hiT.AsWritableSpan();
        var loSpan = loT.AsWritableSpan();
        var hi = ops.FromDouble(clip);
        var lo = ops.FromDouble(-clip);
        for (int i = 0; i < hiSpan.Length; i++) { hiSpan[i] = hi; loSpan[i] = lo; }
        return ctx.Engine.TensorMax(loT, ctx.Engine.TensorMin(t, hiT));
    }

    /// <summary>
    /// For each batch row <c>b</c>, picks <paramref name="active"/>'s row if
    /// <c>t &lt; seqLens[b]</c>, otherwise <paramref name="fallback"/>'s row.
    /// Used by LSTM/GRU's sequence_lens path so rows past their declared
    /// length freeze their h/c state at whatever the pre-freeze value was.
    /// </summary>
    private static Tensor<T> BlendByMask<T>(
        OnnxTranslationContext<T> ctx, Tensor<T> active, Tensor<T> fallback,
        int[] seqLens, int t, int batch, int hidden) where T : unmanaged
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        // Build a [batch, 1] float mask: 1 where t < seqLens[row] else 0.
        var maskT = new Tensor<T>(new[] { batch, 1 });
        var maskSpan = maskT.AsWritableSpan();
        var zero = ops.Zero;
        var one = ops.One;
        for (int b = 0; b < batch; b++)
            maskSpan[b] = t < seqLens[b] ? one : zero;
        // blended = mask * active + (1 - mask) * fallback, broadcasting mask
        // along the H axis.
        var maskedActive = ctx.Engine.TensorBroadcastMultiply(active, maskT);
        var onesMinusMask = new Tensor<T>(new[] { batch, 1 });
        var omm = onesMinusMask.AsWritableSpan();
        for (int b = 0; b < batch; b++)
            omm[b] = t < seqLens[b] ? zero : one;
        var maskedFallback = ctx.Engine.TensorBroadcastMultiply(fallback, onesMinusMask);
        return ctx.Engine.TensorAdd(maskedActive, maskedFallback);
    }

    /// <summary>
    /// Reads an ONNX <c>sequence_lens</c> tensor (typically int32) as int[].
    /// Our plan stores everything as float T, so round-to-int to recover
    /// the original integer values.
    /// </summary>
    private static int[] ToIntsCeil<T>(Tensor<T> t) where T : unmanaged
    {
        var span = t.AsSpan();
        var result = new int[span.Length];
        for (int i = 0; i < span.Length; i++)
        {
            double v = Convert.ToDouble(span[i]!);
            result[i] = (int)Math.Round(v);
        }
        return result;
    }
}
