using System;
using System.Buffers;
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
    /// Cached zero-length placeholder returned for the final hidden / cell out
    /// params when a caller of the allocation-free <c>LstmSequenceForward</c>
    /// overload discards them. Shared (and never mutated) so the hot path stays
    /// free of per-call state allocation.
    /// </summary>
    private static readonly Tensor<float> s_emptyState = new(new[] { 0 });

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
        // Allocation-free hot path (the AIsEval Predict path): compute only the
        // sequence output. wantState: false skips materializing finalHidden /
        // finalCell, so this overload never pays the two [batch, hidden] copies.
        => LstmSequenceForwardImpl(
            input, h0, c0, wIh, wHh, bIh, bHh, returnSequences,
            wantState: false, out _, out _);

    /// <summary>
    /// Fused LSTM sequence forward that also returns the final hidden / cell states,
    /// enabling chunked / streaming inference (feed <paramref name="finalHidden"/> /
    /// <paramref name="finalCell"/> back as <c>h0</c> / <c>c0</c> for the next chunk
    /// instead of recomputing the prefix).
    /// </summary>
    /// <param name="finalHidden">Receives the last timestep's hidden state <c>[batch, hidden]</c> (== h_n).</param>
    /// <param name="finalCell">Receives the last timestep's cell state <c>[batch, hidden]</c> (== c_n).</param>
    public virtual Tensor<T> LstmSequenceForward<T>(
        Tensor<T> input,
        Tensor<T>? h0,
        Tensor<T>? c0,
        Tensor<T> wIh,
        Tensor<T> wHh,
        Tensor<T>? bIh,
        Tensor<T>? bHh,
        out Tensor<T> finalHidden,
        out Tensor<T> finalCell,
        bool returnSequences = false)
        => LstmSequenceForwardImpl(
            input, h0, c0, wIh, wHh, bIh, bHh, returnSequences,
            wantState: true, out finalHidden, out finalCell);

    /// <summary>
    /// Shared implementation for both <c>LstmSequenceForward</c> overloads.
    /// <paramref name="wantState"/> gates final-state materialization: when
    /// false (the allocation-free overload), the per-step compute runs exactly
    /// as before but the two <c>[batch, hidden]</c> final-state tensors are
    /// neither allocated nor copied.
    /// </summary>
    private Tensor<T> LstmSequenceForwardImpl<T>(
        Tensor<T> input,
        Tensor<T>? h0,
        Tensor<T>? c0,
        Tensor<T> wIh,
        Tensor<T> wHh,
        Tensor<T>? bIh,
        Tensor<T>? bHh,
        bool returnSequences,
        bool wantState,
        out Tensor<T> finalHidden,
        out Tensor<T> finalCell)
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

        // Float fast path: SimdGemm + vectorized sigmoid/tanh + ArrayPool
        // scratch buffers reused across timesteps. This is the path that
        // closes the AIsEval LSTM gap on CPU. Generic-T path below stays
        // for double / decimal / BigInteger / custom numerics.
        if (typeof(T) == typeof(float))
        {
            var outF = LstmSequenceForwardFloat(
                (Tensor<float>)(object)input,
                (Tensor<float>?)(object?)h0,
                (Tensor<float>?)(object?)c0,
                (Tensor<float>)(object)wIh,
                (Tensor<float>)(object)wHh,
                (Tensor<float>?)(object?)bIh,
                (Tensor<float>?)(object?)bHh,
                batch, seqLen, inFeatures, hidden, gateRows, returnSequences, wantState,
                out var hnF, out var cnF);
            finalHidden = (Tensor<T>)(object)hnF;
            finalCell = (Tensor<T>)(object)cnF;
            return (Tensor<T>)(object)outF;
        }

        return LstmSequenceForwardGeneric(
            input, h0, c0, wIh, wHh, bIh, bHh,
            batch, seqLen, inFeatures, hidden, gateRows, returnSequences, wantState,
            out finalHidden, out finalCell);
    }

    /// <summary>
    /// Float32 fast path. Uses <see cref="SimdGemm.Sgemm"/> for both the
    /// big <c>Wx</c> pre-compute and the per-step recurrent GEMM, with
    /// vectorized <see cref="SimdKernels.SigmoidUnsafe(float*, float*, int)"/>
    /// and <see cref="SimdKernels.TanhUnsafe(float*, float*, int)"/> for the
    /// gate activations. All inter-timestep scratch is held in pooled arrays
    /// (one alloc up front from <see cref="ArrayPool{T}"/>) so the per-step
    /// loop is allocation-free.
    /// </summary>
    private unsafe Tensor<float> LstmSequenceForwardFloat(
        Tensor<float> input,
        Tensor<float>? h0,
        Tensor<float>? c0,
        Tensor<float> wIh,
        Tensor<float> wHh,
        Tensor<float>? bIh,
        Tensor<float>? bHh,
        int batch, int seqLen, int inFeatures, int hidden, int gateRows,
        bool returnSequences,
        bool wantState,
        out Tensor<float> finalHidden,
        out Tensor<float> finalCell)
    {
        // Empty-batch / empty-sequence early return — matches the generic LSTM path's
        // tolerance of degenerate inputs and shortcuts the per-step GEMM/activation work
        // (avoids any zero-size buffer surprises and pool churn on a no-op call).
        if (batch == 0 || seqLen == 0)
        {
            finalHidden = AutoTensorCache.RentOrAllocate<float>(new[] { batch, hidden });
            finalCell = AutoTensorCache.RentOrAllocate<float>(new[] { batch, hidden });
            return returnSequences
                ? AutoTensorCache.RentOrAllocate<float>(new[] { batch, seqLen, hidden })
                : AutoTensorCache.RentOrAllocate<float>(new[] { batch, hidden });
        }

        // Pre-compute Wx = input @ wIh^T as one big GEMM via SimdGemm.
        // input is [B, seq, in] — treat as [B*seq, in] row-major.
        // wIh is [4H, in] — we want input @ wIh^T which is [B*seq, 4H].
        // Avoid the Tensor<T> dispatch path for the contiguous fast case;
        // call SimdGemm directly using transB=true.
        int totalRows = batch * seqLen;
        var pool = ArrayPool<float>.Shared;

        var wxBuf = pool.Rent(totalRows * gateRows);
        var hCurrBuf = pool.Rent(batch * hidden);
        var cCurrBuf = pool.Rent(batch * hidden);
        var hPrevBuf = pool.Rent(batch * hidden);
        var cPrevBuf = pool.Rent(batch * hidden);
        var hhBuf = pool.Rent(batch * gateRows);     // h_prev @ wHh^T scratch
        var gateBuf = pool.Rent(batch * gateRows);   // gates pre-activation

        // Output buffer comes from AutoTensorCache so we don't pay an allocation.
        var output = returnSequences
            ? AutoTensorCache.RentOrAllocate<float>(new[] { batch, seqLen, hidden })
            : AutoTensorCache.RentOrAllocate<float>(new[] { batch, hidden });

        try
        {
            // Wx = input @ wIh^T using SimdGemm with transB=true.
            // input rows = totalRows, K = inFeatures, wIh rows = gateRows, ldb = inFeatures.
            SimdGemm.Sgemm(
                input.AsSpan(), inFeatures, false,
                wIh.AsSpan(), inFeatures, true,
                wxBuf.AsSpan(0, totalRows * gateRows),
                totalRows, inFeatures, gateRows);

            // Fold bIh into Wx (broadcast add) — SIMD-friendly contiguous loop.
            if (bIh is not null)
            {
                var bIhArr = bIh.AsSpan();
                for (int r = 0; r < totalRows; r++)
                {
                    int off = r * gateRows;
                    for (int g = 0; g < gateRows; g++)
                        wxBuf[off + g] += bIhArr[g];
                }
            }

            // Initial states.
            if (h0 is null) Array.Clear(hPrevBuf, 0, batch * hidden);
            else h0.AsSpan().Slice(0, batch * hidden).CopyTo(hPrevBuf.AsSpan(0, batch * hidden));
            if (c0 is null) Array.Clear(cPrevBuf, 0, batch * hidden);
            else c0.AsSpan().Slice(0, batch * hidden).CopyTo(cPrevBuf.AsSpan(0, batch * hidden));

            var wHhSpan = wHh.AsSpan();
            bool hasBhh = bHh is not null;
            ReadOnlySpan<float> bHhSpan = hasBhh ? bHh!.AsSpan() : default;
            var outSpan = output.AsWritableSpan();

            // #477: the recurrent GEMM h_prev @ wHh^T runs once per timestep (seqLen
            // serial calls). The old transB=true Sgemm re-transposed + re-packed the
            // constant wHh on EVERY step — per-GEMM dispatch over the sequence was the
            // measured bottleneck (thread-insensitive ~5.9 ms). Pre-transpose wHh
            // [gateRows, hidden] → wHhT [hidden, gateRows] ONCE, then drive the per-step
            // GEMM through SgemmWithCachedB: it keys the pre-packed B on the array
            // identity, so wHhT is packed once (step 0) and reused for every later step.
            // A fresh array (not pooled) guarantees a unique identity per call, so the
            // pack cache can never serve stale bytes from a recycled buffer.
            var wHhT = new float[hidden * gateRows];
            for (int g = 0; g < gateRows; g++)
                for (int hh2 = 0; hh2 < hidden; hh2++)
                    wHhT[hh2 * gateRows + g] = wHhSpan[g * hidden + hh2];

            // Per-timestep loop.
            for (int t = 0; t < seqLen; t++)
            {
                // hh = h_prev @ wHh^T = h_prev @ wHhT, [B, hidden] @ [hidden, gateRows].
                // No-transpose A·B with the pre-packed (cached) wHhT — pack happens once
                // on the first step, every later step reuses it. See the wHhT note above.
                // Kept SERIAL on purpose: parallelizing this tiny per-step GEMM across the
                // batch loses badly (32 parallel-for dispatches over the sequence dominate
                // the ~2M-FMA chunks — measured 4× slower), matching the issue's
                // thread-insensitive finding.
                SimdGemm.SgemmWithCachedB(
                    hPrevBuf.AsSpan(0, batch * hidden),
                    wHhT,
                    hhBuf.AsSpan(0, batch * gateRows),
                    batch, hidden, gateRows);

                // Combine into a DE-INTERLEAVED gate layout [gateType][batch][hidden]
                // (i-block | f-block | g-block | o-block, each batch*hidden contiguous):
                //   gate[gType][b][h] = Wx[(b*seq+t), gType*H+h] + hh[b, gType*H+h] + bHh[...]
                // This lets each activation run as ONE big SimdKernels call per step
                // (4/step) instead of 4-per-batch (512/step at B=128), and turns the
                // cell update into a flat contiguous pass. #477: the per-call activation
                // overhead over the 32-step sequence was ~half the kernel wall-clock.
                int bh = batch * hidden;
                for (int b = 0; b < batch; b++)
                {
                    int wxRowOff = (b * seqLen + t) * gateRows;
                    int hhOff = b * gateRows;
                    for (int gType = 0; gType < 4; gType++)
                    {
                        int srcG = gType * hidden;
                        int dst = gType * bh + b * hidden;
                        for (int h = 0; h < hidden; h++)
                        {
                            float v = wxBuf[wxRowOff + srcG + h] + hhBuf[hhOff + srcG + h];
                            if (hasBhh) v += bHhSpan[srcG + h];
                            gateBuf[dst + h] = v;
                        }
                    }
                }

                // Activate each gate-type block in a single vectorized call.
                fixed (float* gateP = gateBuf)
                {
                    SimdKernels.SigmoidUnsafe(gateP + 0 * bh, gateP + 0 * bh, bh);  // i
                    SimdKernels.SigmoidUnsafe(gateP + 1 * bh, gateP + 1 * bh, bh);  // f
                    SimdKernels.TanhUnsafe(gateP + 2 * bh, gateP + 2 * bh, bh);     // g
                    SimdKernels.SigmoidUnsafe(gateP + 3 * bh, gateP + 3 * bh, bh);  // o
                }

                // Cell + hidden update over the flat [batch*hidden] blocks:
                //   c = f * c_prev + i * g ;  h = o * tanh(c)
                // tanh(c) is now a single batched call (was per-batch).
                fixed (float* gateP = gateBuf)
                fixed (float* hCurrP = hCurrBuf)
                fixed (float* cCurrP = cCurrBuf)
                fixed (float* cPrevP = cPrevBuf)
                {
                    float* iB = gateP + 0 * bh;
                    float* fB = gateP + 1 * bh;
                    float* gB = gateP + 2 * bh;
                    float* oB = gateP + 3 * bh;
                    for (int idx = 0; idx < bh; idx++)
                        cCurrP[idx] = fB[idx] * cPrevP[idx] + iB[idx] * gB[idx];
                    SimdKernels.TanhUnsafe(cCurrP, hCurrP, bh);   // hCurr = tanh(c)
                    for (int idx = 0; idx < bh; idx++)
                        hCurrP[idx] = oB[idx] * hCurrP[idx];       // h = o * tanh(c)
                }

                if (returnSequences)
                {
                    // Copy hCurr into output[:, t, :] (B copies of [hidden] floats).
                    for (int b = 0; b < batch; b++)
                    {
                        int srcOff = b * hidden;
                        int dstOff = (b * seqLen + t) * hidden;
                        hCurrBuf.AsSpan(srcOff, hidden).CopyTo(outSpan.Slice(dstOff, hidden));
                    }
                }

                // Swap roles: next iter's h_prev/c_prev are this iter's h_curr/c_curr.
                (hPrevBuf, hCurrBuf) = (hCurrBuf, hPrevBuf);
                (cPrevBuf, cCurrBuf) = (cCurrBuf, cPrevBuf);
            }

            if (!returnSequences)
            {
                // hPrevBuf now holds the last hCurr (post-swap). Copy it out.
                hPrevBuf.AsSpan(0, batch * hidden).CopyTo(outSpan);
            }

            if (wantState)
            {
                // Final states (h_n, c_n) for streaming/chunked inference. After the
                // role-swap at the end of the last timestep, hPrevBuf / cPrevBuf hold
                // the last step's hidden / cell state. Copy into owned [B, hidden]
                // tensors so they outlive the pooled scratch returned below.
                finalHidden = new Tensor<float>(new[] { batch, hidden });
                finalCell = new Tensor<float>(new[] { batch, hidden });
                hPrevBuf.AsSpan(0, batch * hidden).CopyTo(finalHidden.AsWritableSpan());
                cPrevBuf.AsSpan(0, batch * hidden).CopyTo(finalCell.AsWritableSpan());
            }
            else
            {
                // Hot path: caller discards state — skip the two [B, hidden]
                // allocations + copies entirely. Share one cached empty tensor.
                finalHidden = s_emptyState;
                finalCell = s_emptyState;
            }

            return output;
        }
        finally
        {
            pool.Return(wxBuf);
            pool.Return(hCurrBuf);
            pool.Return(cCurrBuf);
            pool.Return(hPrevBuf);
            pool.Return(cPrevBuf);
            pool.Return(hhBuf);
            pool.Return(gateBuf);
        }
    }

    /// <summary>
    /// Generic-T path for non-float numerics (double, decimal, BigInteger,
    /// custom types). Uses <see cref="INumericOperations{T}"/> for the math
    /// and the existing tensor-level matmul for the Wx pre-compute.
    /// </summary>
    private Tensor<T> LstmSequenceForwardGeneric<T>(
        Tensor<T> input,
        Tensor<T>? h0,
        Tensor<T>? c0,
        Tensor<T> wIh,
        Tensor<T> wHh,
        Tensor<T>? bIh,
        Tensor<T>? bHh,
        int batch, int seqLen, int inFeatures, int hidden, int gateRows,
        bool returnSequences,
        bool wantState,
        out Tensor<T> finalHidden,
        out Tensor<T> finalCell)
    {
        // Pre-compute Wx = input @ wIh^T + bIh for ALL timesteps as one big GEMM.
        var inputFlat = input.Reshape(new[] { batch * seqLen, inFeatures });
        var wxFlat = TensorMatMulTransposed(inputFlat, wIh); // [B*seq, 4H]

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

        var hShape = new[] { batch, hidden };
        var hPrev = h0 is null ? Tensor<T>.CreateZeros(hShape) : h0.Clone();
        var cPrev = c0 is null ? Tensor<T>.CreateZeros(hShape) : c0.Clone();

        Tensor<T> output = returnSequences
            ? AutoTensorCache.RentOrAllocate<T>(new[] { batch, seqLen, hidden })
            : AutoTensorCache.RentOrAllocate<T>(hShape);

        var opsT = MathHelper.GetNumericOperations<T>();
        var wxSpanRO = wxFlat.AsSpan();

        for (int t = 0; t < seqLen; t++)
        {
            var hh = TensorMatMulTransposed(hPrev, wHh);
            var hhSpan = hh.AsSpan();
            var hCurr = Tensor<T>.CreateZeros(hShape);
            var cCurr = Tensor<T>.CreateZeros(hShape);
            var hCurrSpan = hCurr.AsWritableSpan();
            var cCurrSpan = cCurr.AsWritableSpan();
            var cPrevSpan = cPrev.AsSpan();
            bool hasBhh = bHh is not null;
            ReadOnlySpan<T> bHhSpan = hasBhh ? bHh!.AsSpan() : default;

            for (int b = 0; b < batch; b++)
            {
                int wxRow = b * seqLen + t;
                int wxOff = wxRow * gateRows;
                int hhOff = b * gateRows;
                int hOff = b * hidden;

                for (int h = 0; h < hidden; h++)
                {
                    int iIdx = hhOff + 0 * hidden + h;
                    int fIdx = hhOff + 1 * hidden + h;
                    int gIdx = hhOff + 2 * hidden + h;
                    int oIdx = hhOff + 3 * hidden + h;
                    int wxI = wxOff + 0 * hidden + h;
                    int wxF = wxOff + 1 * hidden + h;
                    int wxG = wxOff + 2 * hidden + h;
                    int wxO = wxOff + 3 * hidden + h;

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

                    T iAct = Sigmoid(opsT, iGate);
                    T fAct = Sigmoid(opsT, fGate);
                    T gAct = TanhScalar(opsT, gGate);
                    T oAct = Sigmoid(opsT, oGate);

                    T cNew = opsT.Add(opsT.Multiply(fAct, cPrevSpan[hOff + h]), opsT.Multiply(iAct, gAct));
                    T hNew = opsT.Multiply(oAct, TanhScalar(opsT, cNew));

                    cCurrSpan[hOff + h] = cNew;
                    hCurrSpan[hOff + h] = hNew;
                }
            }

            hPrev = hCurr;
            cPrev = cCurr;

            if (returnSequences)
            {
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
            var outSpan = output.AsWritableSpan();
            var hLastSpan = hPrev.AsSpan();
            for (int i = 0; i < batch * hidden; i++)
                outSpan[i] = hLastSpan[i];
        }

        if (wantState)
        {
            // Final states (h_n, c_n) for streaming/chunked inference. hPrev / cPrev
            // hold the last timestep's hidden / cell state; Clone so they are owned
            // tensors independent of the loop's intermediate buffers.
            finalHidden = hPrev.Clone();
            finalCell = cPrev.Clone();
        }
        else
        {
            // Hot path: caller discards state — skip the two clones.
            finalHidden = new Tensor<T>(new[] { 0 });
            finalCell = new Tensor<T>(new[] { 0 });
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
