using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Fused LSTM sequence forward — processes a full <c>[B, seq, in]</c> sequence
    /// through one LSTM cell in a single call, returning either the entire hidden
    /// state sequence <c>[B, seq, hidden]</c> (when <paramref name="returnSequences"/>
    /// is true) or only the last hidden state <c>[B, hidden]</c>.
    ///
    /// <para>
    /// Replaces the per-timestep loop that
    /// <c>AiDotNet.NeuralNetworks.Layers.LSTMLayer.Forward</c> currently runs.
    /// At PyTorch-default workload (<c>[128, 32, 32]</c>) the per-step loop spawns
    /// 32 separate dispatches through <c>Tensor&lt;T&gt;.MatMul</c> +
    /// <c>Sigmoid</c> + <c>Tanh</c>, each of which records on the autograd tape
    /// and allocates scratch tensors. That dispatch overhead, not the GEMM work
    /// itself, is what kept LSTM <c>Predict()</c> from finishing in reasonable
    /// wall time on the AIsEval benchmark (issue #436 P0 gap). Pre-computing
    /// the entire input-to-hidden product <c>Wx = input @ W_ih^T</c> as one
    /// <c>[B*seq, 4H]</c> GEMM and then running a tight per-step inner loop
    /// over raw spans closes the gap.
    /// </para>
    ///
    /// <para>
    /// Convention follows PyTorch's <c>nn.LSTM</c>: gates are
    /// <c>[i, f, g, o]</c> concatenated along the hidden axis, so
    /// <paramref name="wIh"/> and <paramref name="wHh"/> are shaped
    /// <c>[4*hidden, in]</c> / <c>[4*hidden, hidden]</c>. Bias terms are
    /// optional and are split the same way; passing one but not the other is
    /// allowed (each is folded in independently).
    /// </para>
    ///
    /// <para>
    /// <b>Forward-only.</b> This op is intended for the inference path
    /// (<c>Predict</c>). Calling it under an active <c>GradientTape</c> throws
    /// — training paths should keep using the existing decomposed
    /// <c>LSTMLayer.Forward</c> until a fused backward lands in a follow-up PR.
    /// </para>
    /// </summary>
    /// <param name="input">[B, seq, in] input sequence.</param>
    /// <param name="h0">[B, hidden] initial hidden state. Null = zeros.</param>
    /// <param name="c0">[B, hidden] initial cell state. Null = zeros.</param>
    /// <param name="wIh">[4*hidden, in] input-to-hidden weight (PyTorch order: i, f, g, o).</param>
    /// <param name="wHh">[4*hidden, hidden] hidden-to-hidden weight.</param>
    /// <param name="bIh">[4*hidden] input-to-hidden bias. Null = no bias.</param>
    /// <param name="bHh">[4*hidden] hidden-to-hidden bias. Null = no bias.</param>
    /// <param name="returnSequences">
    /// True returns the full <c>[B, seq, hidden]</c> stack. False returns just
    /// the last timestep's hidden state <c>[B, hidden]</c> — the common shape
    /// for classification heads (matches PyTorch's <c>output[:, -1, :]</c>
    /// pattern and AiDotNet's <c>SequenceTokenSliceLayer(Position.Last)</c>).
    /// </param>
    public virtual Tensor<T> LstmSequenceForward<T>(
        Tensor<T> input,
        Tensor<T>? h0,
        Tensor<T>? c0,
        Tensor<T> wIh,
        Tensor<T> wHh,
        Tensor<T>? bIh,
        Tensor<T>? bHh,
        bool returnSequences = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (wIh is null) throw new ArgumentNullException(nameof(wIh));
        if (wHh is null) throw new ArgumentNullException(nameof(wHh));
        if (input.Rank != 3)
            throw new ArgumentException($"LstmSequenceForward expects rank-3 input [B, seq, in]; got rank {input.Rank}.", nameof(input));
        if (wIh.Rank != 2 || wHh.Rank != 2)
            throw new ArgumentException("wIh and wHh must be rank-2.");

        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int inFeatures = input.Shape[2];
        int gateRows = wIh.Shape[0];
        if (gateRows % 4 != 0)
            throw new ArgumentException($"wIh first dim must be 4*hidden; got {gateRows}.", nameof(wIh));
        int hidden = gateRows / 4;
        if (wIh.Shape[1] != inFeatures)
            throw new ArgumentException($"wIh.Shape[1] ({wIh.Shape[1]}) must equal input feature count ({inFeatures}).", nameof(wIh));
        if (wHh.Shape[0] != gateRows || wHh.Shape[1] != hidden)
            throw new ArgumentException($"wHh must be [{gateRows}, {hidden}]; got [{wHh.Shape[0]}, {wHh.Shape[1]}].", nameof(wHh));
        if (h0 is not null && (h0.Rank != 2 || h0.Shape[0] != batch || h0.Shape[1] != hidden))
            throw new ArgumentException($"h0 must be [{batch}, {hidden}].", nameof(h0));
        if (c0 is not null && (c0.Rank != 2 || c0.Shape[0] != batch || c0.Shape[1] != hidden))
            throw new ArgumentException($"c0 must be [{batch}, {hidden}].", nameof(c0));
        if (bIh is not null && (bIh.Rank != 1 || bIh.Shape[0] != gateRows))
            throw new ArgumentException($"bIh must be [{gateRows}].", nameof(bIh));
        if (bHh is not null && (bHh.Rank != 1 || bHh.Shape[0] != gateRows))
            throw new ArgumentException($"bHh must be [{gateRows}].", nameof(bHh));

        // GraphMode is unsupported in this revision — backward is not yet
        // implemented. Training paths should not call this; they should use
        // the existing per-step LSTMLayer.Forward which has tape coverage.
        if (GraphMode.IsActive)
            throw new InvalidOperationException(
                "LstmSequenceForward is inference-only and does not yet support GradientTape. " +
                "Call it outside an active graph-mode scope, or route training through the " +
                "decomposed LSTMLayer.Forward path.");

        // Pre-compute Wx = input @ wIh^T + bIh for ALL timesteps as one big
        // GEMM. input is [B, seq, in] -> reshape to [B*seq, in]; wIh is
        // [4H, in]. TensorMatMulTransposed handles transB=true natively via
        // SimdGemm.Sgemm (float fast path), so this is a single dispatch
        // instead of (seq) dispatches.
        var inputFlat = input.Reshape(new[] { batch * seqLen, inFeatures });
        var wxFlat = TensorMatMulTransposed(inputFlat, wIh); // [B*seq, 4H]

        // Fold bIh into Wx in-place via raw span op (broadcast add along axis 0).
        // Skip the standard TensorAdd dispatch since the broadcast pattern is
        // trivial here and we want to avoid allocating another [B*seq, 4H] tensor.
        if (bIh is not null)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var wxSpan = wxFlat.AsWritableSpan();
            var bIhSpan = bIh.AsSpan();
            int rows = batch * seqLen;
            for (int r = 0; r < rows; r++)
            {
                int off = r * gateRows;
                for (int g = 0; g < gateRows; g++)
                    wxSpan[off + g] = ops.Add(wxSpan[off + g], bIhSpan[g]);
            }
        }

        // Allocate output(s) + scratch buffers.
        // h, c are alive across timesteps and are [B, H] each.
        // gates buffer is [B, 4H] per step (reused). hh buffer for h_prev @
        // wHh^T per step, also [B, 4H], also reused.
        var hShape = new[] { batch, hidden };
        var gateShape = new[] { batch, gateRows };

        var hPrev = h0 is null ? Tensor<T>.CreateZeros(hShape) : h0.Clone();
        var cPrev = c0 is null ? Tensor<T>.CreateZeros(hShape) : c0.Clone();

        // Output buffer: [B, seq, H] when returning sequences, else [B, H].
        Tensor<T> output = returnSequences
            ? AutoTensorCache.RentOrAllocate<T>(new[] { batch, seqLen, hidden })
            : AutoTensorCache.RentOrAllocate<T>(hShape);

        // Reusable scratch for h_prev @ wHh^T (gate accumulator per step).
        // We can't reuse a single Tensor<T> across the matmul calls (each
        // returns a fresh allocation), but we can keep references short-lived
        // so the GC reclaims them between steps. Keeping this honest for the
        // first revision; a future pass can pre-allocate gateScratch.

        var opsT = MathHelper.GetNumericOperations<T>();
        var wxSpanRO = wxFlat.AsSpan();

        // Per-timestep loop. We work primarily in spans here to keep the
        // inner loop tight; the only Tensor<T>-level call per step is the
        // h_prev @ wHh^T GEMM, which routes through SimdGemm for float.
        for (int t = 0; t < seqLen; t++)
        {
            // gates_t = Wx[:, t, :] + h_prev @ wHh^T + bHh
            var hh = TensorMatMulTransposed(hPrev, wHh); // [B, 4H]
            var hhSpan = hh.AsSpan();
            var hCurr = Tensor<T>.CreateZeros(hShape);
            var cCurr = Tensor<T>.CreateZeros(hShape);
            var hCurrSpan = hCurr.AsWritableSpan();
            var cCurrSpan = cCurr.AsWritableSpan();
            var hPrevSpan = hPrev.AsSpan();
            var cPrevSpan = cPrev.AsSpan();
            bool hasBhh = bHh is not null;
            ReadOnlySpan<T> bHhSpan = hasBhh ? bHh!.AsSpan() : default;

            // Wx slice for this timestep: rows [b * seqLen + t] for b in [0, B).
            // We index it inline.
            for (int b = 0; b < batch; b++)
            {
                int wxRow = b * seqLen + t;
                int wxOff = wxRow * gateRows;
                int hhOff = b * gateRows;
                int hOff = b * hidden;

                for (int h = 0; h < hidden; h++)
                {
                    // Indices into the 4 gates within this row.
                    int iIdx = hhOff + 0 * hidden + h;
                    int fIdx = hhOff + 1 * hidden + h;
                    int gIdx = hhOff + 2 * hidden + h;
                    int oIdx = hhOff + 3 * hidden + h;
                    int wxI = wxOff + 0 * hidden + h;
                    int wxF = wxOff + 1 * hidden + h;
                    int wxG = wxOff + 2 * hidden + h;
                    int wxO = wxOff + 3 * hidden + h;

                    // gate = Wx[b,t,gate] + hh[b,gate] + bHh[gate]
                    T iGate = opsT.Add(wxSpanRO[wxI], hhSpan[iIdx]);
                    T fGate = opsT.Add(wxSpanRO[wxF], hhSpan[fIdx]);
                    T gGate = opsT.Add(wxSpanRO[wxG], hhSpan[gIdx]);
                    T oGate = opsT.Add(wxSpanRO[wxO], hhSpan[oIdx]);
                    if (hasBhh)
                    {
                        iGate = opsT.Add(iGate, bHhSpan[0 * hidden + h]);
                        fGate = opsT.Add(fGate, bHhSpan[1 * hidden + h]);
                        gGate = opsT.Add(gGate, bHhSpan[2 * hidden + h]);
                        oGate = opsT.Add(oGate, bHhSpan[3 * hidden + h]);
                    }

                    // Activations: i, f, o = sigmoid; g = tanh.
                    T iAct = Sigmoid(opsT, iGate);
                    T fAct = Sigmoid(opsT, fGate);
                    T gAct = TanhScalar(opsT, gGate);
                    T oAct = Sigmoid(opsT, oGate);

                    // c = f * c_prev + i * g
                    T cNew = opsT.Add(opsT.Multiply(fAct, cPrevSpan[hOff + h]), opsT.Multiply(iAct, gAct));
                    // h = o * tanh(c)
                    T hNew = opsT.Multiply(oAct, TanhScalar(opsT, cNew));

                    cCurrSpan[hOff + h] = cNew;
                    hCurrSpan[hOff + h] = hNew;
                }
            }

            // Roll hPrev/cPrev to the just-computed states for next iteration.
            hPrev = hCurr;
            cPrev = cCurr;

            if (returnSequences)
            {
                // Copy hCurr into output[:, t, :].
                var outSpan = output.AsWritableSpan();
                for (int b = 0; b < batch; b++)
                {
                    int srcOff = b * hidden;
                    int dstOff = (b * seqLen + t) * hidden;
                    for (int h = 0; h < hidden; h++)
                        outSpan[dstOff + h] = hCurrSpan[srcOff + h];
                }
            }
        }

        if (!returnSequences)
        {
            // Output is just the last hCurr — copy it.
            var outSpan = output.AsWritableSpan();
            var hLastSpan = hPrev.AsSpan(); // hPrev was assigned to hCurr at end of last iter
            for (int i = 0; i < batch * hidden; i++)
                outSpan[i] = hLastSpan[i];
        }

        return output;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Sigmoid<T>(INumericOperations<T> ops, T x)
    {
        // 1 / (1 + exp(-x))
        var negX = ops.Multiply(ops.FromDouble(-1.0), x);
        var expNeg = ops.Exp(negX);
        return ops.Divide(ops.One, ops.Add(ops.One, expNeg));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T TanhScalar<T>(INumericOperations<T> ops, T x)
    {
        // tanh(x) = (e^2x - 1) / (e^2x + 1)
        var e2x = ops.Exp(ops.Multiply(ops.FromDouble(2.0), x));
        return ops.Divide(ops.Subtract(e2x, ops.One), ops.Add(e2x, ops.One));
    }
}
