using System;
using System.Buffers;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
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

    // Batch-parallel recurrence config. The LSTM recurrence is embarrassingly
    // parallel across the batch dimension — h_t[b] depends only on h_{t-1}[b]
    // and x_t[b], never on another batch row — so the entire T-timestep loop
    // (per-step hidden GEMM + fused cell) can be split into contiguous batch
    // chunks that run with ZERO inter-thread communication. The parallel region
    // is launched ONCE for the whole sequence (not per timestep): an earlier
    // attempt that relaunched the region every step lost to 32x dispatch
    // overhead. Gated on rows-per-chunk so small batches (b=1/8 — which already
    // beat PyTorch on the serial path) keep a single chunk and never pay the
    // park/wakeup floor; only large batches (b=32/128, where the serial
    // per-step M=batch GEMM is the bottleneck) fan out. Env-tunable for A/B.
    // Default 8 rows/chunk: a MINROWS sweep on the AIsEval LSTM ([*, 32, 32],
    // hidden=64) found 8 is the sweet spot. It keeps b<=8 on a single chunk (the
    // serial path they already WIN on — fanning b=8 out to 4 chunks regressed it
    // from 0.46x to 0.82x of PyTorch), while letting b=32 fan to 4 chunks
    // (1.58x -> 1.17x) and b=128 saturate the cores (1.89x -> ~1.28x). Going
    // lower (2/4 rows) over-fragments: the per-chunk M=2/4 recurrent GEMM is too
    // thin to amortize, so b=32 degraded back to 1.3-1.4x.
    private static readonly int _lstmParallelMinRowsPerChunk =
        int.TryParse(System.Environment.GetEnvironmentVariable("AIDOTNET_LSTM_PARALLEL_MINROWS"), out var lpr) && lpr > 0
            ? lpr : 8;

    // Master switch (default on). Set AIDOTNET_LSTM_PARALLEL=0 to force the
    // serial recurrence for drift-cancelled on/off A/B measurement.
    private static readonly bool _lstmParallelEnabled =
        System.Environment.GetEnvironmentVariable("AIDOTNET_LSTM_PARALLEL") != "0";

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

        // Under an active gradient tape, float routes to the fused training path: exact
        // activations, saved per-timestep state, and a single fused BPTT node
        // (CpuEngine.LstmSequenceBackward.cs). Collapses the per-timestep training graph
        // into one node (ooples/AiDotNet#1566). The tape (not GraphMode) is what
        // RecordIfActive keys on — GraphMode is the separate graph-compilation flag.
        if (Autodiff.DifferentiableOps.IsRecording<T>())
        {
            if (typeof(T) == typeof(float))
            {
                var trainOut = LstmSequenceForwardFloatTrain(
                    (Tensor<float>)(object)input,
                    (Tensor<float>?)(object?)h0,
                    (Tensor<float>?)(object?)c0,
                    (Tensor<float>)(object)wIh,
                    (Tensor<float>)(object)wHh,
                    (Tensor<float>?)(object?)bIh,
                    (Tensor<float>?)(object?)bHh,
                    batch, seqLen, inFeatures, hidden, gateRows, returnSequences, wantState,
                    out var hnT, out var cnT);
                finalHidden = (Tensor<T>)(object)hnT;
                finalCell = (Tensor<T>)(object)cnT;
                return (Tensor<T>)(object)trainOut;
            }

            throw new InvalidOperationException(
                "LstmSequenceForward supports GradientTape for float only in this revision. " +
                "Route non-float training through the decomposed LSTMLayer.Forward path.");
        }

        // Graph-compilation mode (distinct from the eager tape) is not yet supported.
        if (GraphMode.IsActive)
            throw new InvalidOperationException(
                "LstmSequenceForward does not yet support graph-compilation mode. " +
                "Call it outside an active graph-mode scope.");

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
        // #477: pool the transposed hidden weight too. It used to be `new float[]` per
        // call (64 KB at the AIsEval shape) because SgemmWithCachedB keyed its pre-pack
        // cache on the array identity. Now the recurrent GEMM is SgemmSequential (no
        // identity cache), so wHhT can come from the pool — removing the single biggest
        // per-call allocation (the main GC-churn / p95 source).
        var wHhTBuf = pool.Rent(hidden * gateRows);

        // Output buffer comes from AutoTensorCache so we don't pay an allocation.
        var output = returnSequences
            ? AutoTensorCache.RentOrAllocate<float>(new[] { batch, seqLen, hidden })
            : AutoTensorCache.RentOrAllocate<float>(new[] { batch, hidden });

        // Final-state tensors are allocated up front (when requested) so the
        // batch-parallel chunks can each write their own row range into them.
        finalHidden = wantState ? new Tensor<float>(new[] { batch, hidden }) : s_emptyState;
        finalCell = wantState ? new Tensor<float>(new[] { batch, hidden }) : s_emptyState;

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

            var wHhSpan = wHh.AsSpan();
            float[]? bHhArr = bHh?.ToArray();
            float[]? h0Arr = h0?.ToArray();
            float[]? c0Arr = c0?.ToArray();

            // #477: the recurrent GEMM h_prev @ wHh^T runs once per timestep. The old
            // transB=true Sgemm re-transposed wHh on EVERY step. Pre-transpose wHh
            // [gateRows, hidden] → wHhT [hidden, gateRows] ONCE; [hidden, gateRows] is
            // exactly the [K, N] row-major layout SgemmSequential expects, so the
            // per-step call needs no further transpose or packing setup.
            var wHhT = wHhTBuf;
            for (int g = 0; g < gateRows; g++)
                for (int hh2 = 0; hh2 < hidden; hh2++)
                    wHhT[hh2 * gateRows + g] = wHhSpan[g * hidden + hh2];

            var outArr = (float[])(object)output.GetDataArray();
            float[]? fhArr = wantState ? (float[])(object)finalHidden.GetDataArray() : null;
            float[]? fcArr = wantState ? (float[])(object)finalCell.GetDataArray() : null;

            // Decide batch-parallel fan-out. The recurrence is independent across
            // batch rows, so split [0,batch) into `numChunks` CONTIGUOUS ranges and
            // run the whole T-step loop per range. Only fan out when each chunk keeps
            // a GEMM-healthy row count (>= _lstmParallelMinRowsPerChunk): small
            // batches (b=1/8) collapse to one chunk and take the serial path they
            // already win on. The parallel region launches ONCE for the sequence.
            int numChunks = 1;
            if (_lstmParallelEnabled)
            {
                int byRows = batch / Math.Max(1, _lstmParallelMinRowsPerChunk);
                numChunks = Math.Max(1, Math.Min(byRows, CpuParallelSettings.MaxDegreeOfParallelism));
            }

            if (numChunks <= 1)
            {
                // Serial: one chunk covering the full batch, reusing pooled scratch.
                var hPrevBuf = pool.Rent(batch * hidden);
                var hCurrBuf = pool.Rent(batch * hidden);
                var cPrevBuf = pool.Rent(batch * hidden);
                var cCurrBuf = pool.Rent(batch * hidden);
                var hhBuf = pool.Rent(batch * gateRows);
                try
                {
                    RunLstmRecurrenceRange(
                        wxBuf, wHhT, bHhArr, h0Arr, c0Arr,
                        hPrevBuf, hCurrBuf, cPrevBuf, cCurrBuf, hhBuf,
                        outArr, returnSequences, fhArr, fcArr,
                        r0: 0, rows: batch, seqLen, hidden, gateRows);
                }
                finally
                {
                    pool.Return(hPrevBuf); pool.Return(hCurrBuf);
                    pool.Return(cPrevBuf); pool.Return(cCurrBuf); pool.Return(hhBuf);
                }
            }
            else
            {
                // Batch-parallel: contiguous row ranges, one region launch. Each task
                // rents its own scratch (no shared mutable state across threads — the
                // only shared reads are wxBuf/wHhT/bHh/h0/c0, the only shared writes are
                // disjoint row ranges of outArr/fhArr/fcArr).
                int chunkRows = (batch + numChunks - 1) / numChunks;
                Helpers.PersistentParallelExecutor.Instance.Execute(numChunks, chunk =>
                {
                    int r0 = chunk * chunkRows;
                    if (r0 >= batch) return;
                    int rows = Math.Min(chunkRows, batch - r0);
                    var hPrevL = pool.Rent(rows * hidden);
                    var hCurrL = pool.Rent(rows * hidden);
                    var cPrevL = pool.Rent(rows * hidden);
                    var cCurrL = pool.Rent(rows * hidden);
                    var hhL = pool.Rent(rows * gateRows);
                    try
                    {
                        RunLstmRecurrenceRange(
                            wxBuf, wHhT, bHhArr, h0Arr, c0Arr,
                            hPrevL, hCurrL, cPrevL, cCurrL, hhL,
                            outArr, returnSequences, fhArr, fcArr,
                            r0, rows, seqLen, hidden, gateRows);
                    }
                    finally
                    {
                        pool.Return(hPrevL); pool.Return(hCurrL);
                        pool.Return(cPrevL); pool.Return(cCurrL); pool.Return(hhL);
                    }
                });
            }

            return output;
        }
        finally
        {
            pool.Return(wxBuf);
            pool.Return(wHhTBuf);
        }
    }

    /// <summary>
    /// Runs the full T-timestep LSTM recurrence (per-step recurrent GEMM + fused
    /// elementwise cell) for a CONTIGUOUS range of batch rows <c>[r0, r0+rows)</c>.
    /// Shared with the serial path (one call covering the whole batch) and the
    /// batch-parallel path (one call per chunk). The <paramref name="hPrev"/> /
    /// <paramref name="hCurr"/> / <paramref name="cPrev"/> / <paramref name="cCurr"/>
    /// / <paramref name="hh"/> buffers are LOCAL scratch owned by the caller (sized
    /// for <paramref name="rows"/>, not the full batch) and are mutated/ping-ponged
    /// here. Shared reads: <paramref name="wxBuf"/> (pre-computed input projection,
    /// indexed by GLOBAL row), <paramref name="wHhT"/>, bias, h0/c0. Shared writes:
    /// disjoint global row ranges of <paramref name="outArr"/> and the optional final
    /// state arrays — no two ranges overlap, so concurrent chunks need no locking.
    /// </summary>
    private static unsafe void RunLstmRecurrenceRange(
        float[] wxBuf, float[] wHhT, float[]? bHhArr, float[]? h0Arr, float[]? c0Arr,
        float[] hPrev, float[] hCurr, float[] cPrev, float[] cCurr, float[] hh,
        float[] outArr, bool returnSequences, float[]? fhArr, float[]? fcArr,
        int r0, int rows, int seqLen, int hidden, int gateRows)
    {
        bool hasBhh = bHhArr is not null;

        // Initial states for this row range. Local buffers are indexed [localRow*hidden];
        // h0/c0 are global [batch, hidden] so they are read at the global offset.
        if (h0Arr is null) Array.Clear(hPrev, 0, rows * hidden);
        else Array.Copy(h0Arr, r0 * hidden, hPrev, 0, rows * hidden);
        if (c0Arr is null) Array.Clear(cPrev, 0, rows * hidden);
        else Array.Copy(c0Arr, r0 * hidden, cPrev, 0, rows * hidden);

        // Local ping-pong references (swapped each step). Using locals keeps the
        // caller's array handles fixed for the pool.Return in its finally.
        float[] hP = hPrev, hC = hCurr, cP = cPrev, cC = cCurr;

        bool fusedVec = false;
#if NET5_0_OR_GREATER
        fusedVec = Avx2.IsSupported && Fma.IsSupported && (hidden % 8 == 0);
#endif

        for (int t = 0; t < seqLen; t++)
        {
            // hh = h_prev @ wHhT, [rows, hidden] @ [hidden, gateRows]. SgemmSequential
            // is the DIRECT 6×16 AVX2 kernel (the cached-pre-packed path measured slower
            // at this small K=hidden). Serial WITHIN a thread — the batch-level fan-out
            // (when present) already supplies the parallelism, and re-threading this tiny
            // per-step GEMM would oversubscribe and lose to dispatch overhead.
            //
            // Kernel choice is settled by the LosingShapeGemmBench 4-way matrix at this
            // exact shape ([128,64,256], GF/s): managed-serial 103 BEATS managed-parallel
            // 87, the asm panel kernel serial 53 / parallel 88, and OpenBLAS 66. The
            // managed serial kernel is the measured optimum here — do NOT route this
            // through JitGemmAvx2 or native BLAS without re-running that bench.
            SimdGemm.SgemmSequential(
                hP.AsSpan(0, rows * hidden),
                wHhT.AsSpan(0, hidden * gateRows),
                hh.AsSpan(0, rows * gateRows),
                rows, hidden, gateRows);

            // #477 FUSED ELEMENTWISE CELL: read the four gate pre-activations straight
            // from the natural [row][gateRows] Wx + hh, apply sigmoid/tanh in registers,
            // and write c = f·c_prev + i·g, h = o·tanh(c) directly — gates never
            // round-trip through memory. `lr` is the local row; the GLOBAL row r0+lr
            // indexes the shared wxBuf / outArr / final-state arrays.
            if (fusedVec)
            {
#if NET5_0_OR_GREATER
                fixed (float* wxP = wxBuf)
                fixed (float* hhP = hh)
                fixed (float* cPrevP = cP)
                fixed (float* cCurrP = cC)
                fixed (float* hCurrP = hC)
                fixed (float* bHhP = bHhArr)
                fixed (float* outP = outArr)
                {
                    for (int lr = 0; lr < rows; lr++)
                    {
                        int gb = r0 + lr;
                        int wxRow = (gb * seqLen + t) * gateRows;
                        int hhRow = lr * gateRows;
                        int cRow = lr * hidden;
                        int outRow = (gb * seqLen + t) * hidden;
                        for (int h = 0; h < hidden; h += 8)
                        {
                            var iIn = Avx.Add(Avx.LoadVector256(wxP + wxRow + 0 * hidden + h), Avx.LoadVector256(hhP + hhRow + 0 * hidden + h));
                            var fIn = Avx.Add(Avx.LoadVector256(wxP + wxRow + 1 * hidden + h), Avx.LoadVector256(hhP + hhRow + 1 * hidden + h));
                            var gIn = Avx.Add(Avx.LoadVector256(wxP + wxRow + 2 * hidden + h), Avx.LoadVector256(hhP + hhRow + 2 * hidden + h));
                            var oIn = Avx.Add(Avx.LoadVector256(wxP + wxRow + 3 * hidden + h), Avx.LoadVector256(hhP + hhRow + 3 * hidden + h));
                            if (hasBhh)
                            {
                                iIn = Avx.Add(iIn, Avx.LoadVector256(bHhP + 0 * hidden + h));
                                fIn = Avx.Add(fIn, Avx.LoadVector256(bHhP + 1 * hidden + h));
                                gIn = Avx.Add(gIn, Avx.LoadVector256(bHhP + 2 * hidden + h));
                                oIn = Avx.Add(oIn, Avx.LoadVector256(bHhP + 3 * hidden + h));
                            }

                            var iG = SimdKernels.FastSigmoid256(iIn);
                            var fG = SimdKernels.FastSigmoid256(fIn);
                            var gG = SimdKernels.FastTanh256(gIn);
                            var oG = SimdKernels.FastSigmoid256(oIn);

                            var cp = Avx.LoadVector256(cPrevP + cRow + h);
                            var c = Fma.MultiplyAdd(fG, cp, Avx.Multiply(iG, gG)); // f·c_prev + i·g
                            Avx.Store(cCurrP + cRow + h, c);
                            var hv = Avx.Multiply(oG, SimdKernels.FastTanh256(c)); // o·tanh(c)
                            Avx.Store(hCurrP + cRow + h, hv);
                            if (returnSequences)
                                Avx.Store(outP + outRow + h, hv);
                        }
                    }
                }
#endif
            }
            else
            {
                // Scalar fused fallback (no AVX2/FMA, or hidden % 8 != 0).
                fixed (float* wxP = wxBuf)
                fixed (float* hhP = hh)
                fixed (float* cPrevP = cP)
                fixed (float* cCurrP = cC)
                fixed (float* hCurrP = hC)
                fixed (float* bHhP = bHhArr)
                fixed (float* outP = outArr)
                {
                    for (int lr = 0; lr < rows; lr++)
                    {
                        int gb = r0 + lr;
                        int wxRow = (gb * seqLen + t) * gateRows;
                        int hhRow = lr * gateRows;
                        int cRow = lr * hidden;
                        int outRow = (gb * seqLen + t) * hidden;
                        for (int h = 0; h < hidden; h++)
                        {
                            float iIn = wxP[wxRow + 0 * hidden + h] + hhP[hhRow + 0 * hidden + h];
                            float fIn = wxP[wxRow + 1 * hidden + h] + hhP[hhRow + 1 * hidden + h];
                            float gIn = wxP[wxRow + 2 * hidden + h] + hhP[hhRow + 2 * hidden + h];
                            float oIn = wxP[wxRow + 3 * hidden + h] + hhP[hhRow + 3 * hidden + h];
                            if (hasBhh)
                            {
                                iIn += bHhP[0 * hidden + h]; fIn += bHhP[1 * hidden + h];
                                gIn += bHhP[2 * hidden + h]; oIn += bHhP[3 * hidden + h];
                            }
                            float iG = 1f / (1f + (float)Math.Exp(-iIn));
                            float fG = 1f / (1f + (float)Math.Exp(-fIn));
                            float gG = (float)Math.Tanh(gIn);
                            float oG = 1f / (1f + (float)Math.Exp(-oIn));
                            float c = fG * cPrevP[cRow + h] + iG * gG;
                            cCurrP[cRow + h] = c;
                            float hv = oG * (float)Math.Tanh(c);
                            hCurrP[cRow + h] = hv;
                            if (returnSequences) outP[outRow + h] = hv;
                        }
                    }
                }
            }

            // Swap roles: next iter's prev is this iter's curr.
            (hP, hC) = (hC, hP);
            (cP, cC) = (cC, cP);
        }

        // After the loop, hP/cP hold the last timestep's hidden/cell state for these rows.
        if (!returnSequences)
            Array.Copy(hP, 0, outArr, r0 * hidden, rows * hidden);

        if (fhArr is not null) Array.Copy(hP, 0, fhArr, r0 * hidden, rows * hidden);
        if (fcArr is not null) Array.Copy(cP, 0, fcArr, r0 * hidden, rows * hidden);
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
        // #478: ping-pong TWO pooled hidden + TWO pooled cell buffers across timesteps instead of a
        // fresh Tensor<T>.CreateZeros for hCurr AND cCurr EVERY step (2*seqLen allocations that bypass
        // the arena). That fresh-per-step allocation measured ~25-28x the float fast path's per-call
        // allocation on a 64-step LSTM; reusing two buffers brings the generic (double/decimal/...)
        // path's per-call allocation down to a small constant. The inner loop fully overwrites every
        // element of the destination buffers each step, so no clear is needed between reuses.
        int stateLen = batch * hidden;
        var hbufs = new[] { AutoTensorCache.RentOrAllocate<T>(hShape), AutoTensorCache.RentOrAllocate<T>(hShape) };
        var cbufs = new[] { AutoTensorCache.RentOrAllocate<T>(hShape), AutoTensorCache.RentOrAllocate<T>(hShape) };
        SeedLstmState(hbufs[0], h0, stateLen);
        SeedLstmState(cbufs[0], c0, stateLen);
        int cur = 0;

        Tensor<T> output = returnSequences
            ? AutoTensorCache.RentOrAllocate<T>(new[] { batch, seqLen, hidden })
            : AutoTensorCache.RentOrAllocate<T>(hShape);

        var opsT = MathHelper.GetNumericOperations<T>();
        var wxSpanRO = wxFlat.AsSpan();

        // #478: pool the per-timestep hidden GEMM. The old TensorMatMulTransposed(hPrev, wHh) allocated
        // a fresh [batch, 4*hidden] output (plus the GEMM's packing scratch) EVERY step. Pre-transpose
        // wHh once into a rented buffer and MultiplyBlocked into ONE reused hh buffer, mirroring the
        // float fast path's wHhT + hhBuf — so the recurrence loop allocates nothing per step. This is
        // the dominant remaining allocation after the ping-pong state buffers above.
        var lstmPool = ArrayPool<T>.Shared;
        var wHhT = lstmPool.Rent(hidden * gateRows);
        var hhBuf = lstmPool.Rent(batch * gateRows);
        {
            var wHhSrc = wHh.AsSpan();
            for (int g = 0; g < gateRows; g++)
                for (int hcol = 0; hcol < hidden; hcol++)
                    wHhT[hcol * gateRows + g] = wHhSrc[g * hidden + hcol];
        }
        try
        {
        for (int t = 0; t < seqLen; t++)
        {
            int nxt = 1 - cur;
            // hh[batch, gateRows] = hbufs[cur][batch, hidden] @ wHhT[hidden, gateRows].
            // MultiplyBlocked accumulates into C, so clear the reused buffer first.
            var hhMem = new Memory<T>(hhBuf, 0, batch * gateRows);
            hhMem.Span.Clear();
            MatrixMultiplyHelper.MultiplyBlocked(opsT,
                new ReadOnlyMemory<T>(hbufs[cur].GetDataArray(), 0, stateLen),
                new ReadOnlyMemory<T>(wHhT, 0, hidden * gateRows),
                hhMem,
                batch, hidden, gateRows, hidden, gateRows, gateRows, allowParallel: true);
            var hhSpan = hhBuf.AsSpan();
            var hCurr = hbufs[nxt];
            var cCurr = cbufs[nxt];
            var hCurrSpan = hCurr.AsWritableSpan();
            var cCurrSpan = cCurr.AsWritableSpan();
            var cPrevSpan = cbufs[cur].AsSpan();
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

            cur = nxt;

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
        }
        finally
        {
            lstmPool.Return(wHhT);
            lstmPool.Return(hhBuf);
        }

        if (!returnSequences)
        {
            var outSpan = output.AsWritableSpan();
            var hLastSpan = hbufs[cur].AsSpan();
            for (int i = 0; i < batch * hidden; i++)
                outSpan[i] = hLastSpan[i];
        }

        if (wantState)
        {
            // Final states (h_n, c_n) for streaming/chunked inference. hPrev / cPrev
            // hold the last timestep's hidden / cell state; Clone so they are owned
            // tensors independent of the loop's intermediate buffers.
            finalHidden = hbufs[cur].Clone();
            finalCell = cbufs[cur].Clone();
        }
        else
        {
            // Hot path: caller discards state — skip the two clones.
            finalHidden = new Tensor<T>(new[] { 0 });
            finalCell = new Tensor<T>(new[] { 0 });
        }

        return output;
    }

    // Seed a ping-pong state buffer (#478): copy the initial state in, or zero it when none is given.
    // default(T) is the additive identity for the numeric structs this path serves, so Clear() zeroes.
    private static void SeedLstmState<T>(Tensor<T> buffer, Tensor<T>? source, int n)
    {
        var dst = buffer.AsWritableSpan();
        if (source is null)
            dst.Slice(0, n).Clear();
        else
            source.AsSpan().Slice(0, n).CopyTo(dst.Slice(0, n));
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
