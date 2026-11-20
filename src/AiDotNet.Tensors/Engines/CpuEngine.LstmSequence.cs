using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
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
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines;

[Flags]
internal enum LstmSequenceOptionalInputs : byte
{
    None = 0,
    InitialHidden = 1 << 0,
    InitialCell = 1 << 1,
    InputBias = 1 << 2,
    RecurrentBias = 1 << 3
}

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
    /// <b>Training.</b> Under an active <c>GradientTape</c>, float records one tape node whose backward is
    /// the fused BPTT (exact gradients for the input, both weights and both biases).
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

        if (seqLen == 0)
        {
            finalHidden = h0 ?? new Tensor<T>(new[] { batch, hidden });
            finalCell = c0 ?? new Tensor<T>(new[] { batch, hidden });
            return returnSequences ? new Tensor<T>(new[] { batch, 0, hidden }) : finalHidden;
        }

        if (GraphMode.IsActive)
        {
            // Host training records one fused kernel for either scalar type. State outputs are
            // views of a single packet so the backward receives all three adjoints together.
            if (!GraphMode.IsInferenceTrace && seqLen > 0
                && this is not DirectGpuTensorEngine && GraphMode.Current is { } trainScope
                && (typeof(T) == typeof(float) || typeof(T) == typeof(double)))
            {
                trainScope.BindEngineIfUnset(this);
                Tensor<T> packed;
                if (typeof(T) == typeof(float))
                    packed = (Tensor<T>)(object)RecordLstmSequenceTrainFloat(trainScope,
                        (Tensor<float>)(object)input, (Tensor<float>?)(object?)h0, (Tensor<float>?)(object?)c0,
                        (Tensor<float>)(object)wIh, (Tensor<float>)(object)wHh,
                        (Tensor<float>?)(object?)bIh, (Tensor<float>?)(object?)bHh,
                        batch, seqLen, inFeatures, hidden, returnSequences, wantState);
                else
                    packed = (Tensor<T>)(object)RecordLstmSequenceTrainDouble(trainScope,
                        (Tensor<double>)(object)input, (Tensor<double>?)(object?)h0, (Tensor<double>?)(object?)c0,
                        (Tensor<double>)(object)wIh, (Tensor<double>)(object)wHh,
                        (Tensor<double>?)(object?)bIh, (Tensor<double>?)(object?)bHh,
                        batch, seqLen, inFeatures, hidden, returnSequences, wantState);
                return UnpackLstmTrainingOutput(this, packed, batch, seqLen, hidden, returnSequences, wantState,
                    out finalHidden, out finalCell);
            }

            if (wantState || !GraphMode.IsInferenceTrace)
            {
                return LstmSequenceForwardGraph(
                    input, h0, c0, wIh, wHh, bIh, bHh,
                    batch, seqLen, inFeatures, hidden, returnSequences,
                    out finalHidden, out finalCell);
            }

            var optionalInputs = LstmSequenceOptionalInputs.None;
            var graphInputs = new List<Tensor<T>> { input, wIh, wHh };
            if (h0 is not null) { optionalInputs |= LstmSequenceOptionalInputs.InitialHidden; graphInputs.Add(h0); }
            if (c0 is not null) { optionalInputs |= LstmSequenceOptionalInputs.InitialCell; graphInputs.Add(c0); }
            if (bIh is not null) { optionalInputs |= LstmSequenceOptionalInputs.InputBias; graphInputs.Add(bIh); }
            if (bHh is not null) { optionalInputs |= LstmSequenceOptionalInputs.RecurrentBias; graphInputs.Add(bHh); }

            var capturedInputs = graphInputs.ToArray();
            var outputShape = returnSequences
                ? new[] { batch, seqLen, hidden }
                : new[] { batch, hidden };
            var scope = GraphMode.Current!; scope.BindEngineIfUnset(this);
            object[] savedState = { returnSequences, optionalInputs };
            finalHidden = Tensor<T>.Empty();
            finalCell = Tensor<T>.Empty();
            return scope.RecordVariadic(
                LazyNodeType.LstmSequenceForward,
                "LstmSequenceForward",
                capturedInputs,
                outputShape,
                (eng, output) =>
                {
                    UnpackLstmSequenceInputs(
                        capturedInputs, optionalInputs,
                        out var capturedInput, out var capturedH0, out var capturedC0,
                        out var capturedWIh, out var capturedWHh,
                        out var capturedBIh, out var capturedBHh);
                    var eager = eng.LstmSequenceForward(
                        capturedInput, capturedH0, capturedC0, capturedWIh, capturedWHh,
                        capturedBIh, capturedBHh, returnSequences);
                    DirectGpuTensorEngine.CopyResultInto(eng, eager, output);
                },
                // Tape-replay backward: the graph node used to carry none, so compiled training dropped every gradient
                // through the LSTM (weights never updated). Re-runs the eager op under a tape for the exact VJP.
                backwardFn: AiDotNet.Tensors.Engines.Autodiff.BackwardFunctions<T>.ReplayUnderTape(eng =>
                {
                    UnpackLstmSequenceInputs(
                        capturedInputs, optionalInputs,
                        out var rInput, out var rH0, out var rC0, out var rWIh, out var rWHh, out var rBIh, out var rBHh);
                    return eng.LstmSequenceForward(rInput, rH0, rC0, rWIh, rWHh, rBIh, rBHh, returnSequences);
                }),
                savedState: savedState);
        }

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

            if (typeof(T) == typeof(double))
            {
                var trainOut = LstmSequenceForwardDoubleTrain(
                    (Tensor<double>)(object)input,
                    (Tensor<double>?)(object?)h0,
                    (Tensor<double>?)(object?)c0,
                    (Tensor<double>)(object)wIh,
                    (Tensor<double>)(object)wHh,
                    (Tensor<double>?)(object?)bIh,
                    (Tensor<double>?)(object?)bHh,
                    batch, seqLen, inFeatures, hidden, gateRows, returnSequences, wantState,
                    out var hnD, out var cnD);
                finalHidden = (Tensor<T>)(object)hnD;
                finalCell = (Tensor<T>)(object)cnD;
                return (Tensor<T>)(object)trainOut;
            }

            throw new InvalidOperationException(
                "LstmSequenceForward supports GradientTape for float and double in this revision. " +
                "Route other element types through the decomposed LSTMLayer.Forward path.");
        }

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

        // Reuse the typed GEMM/activation core for double inference too. The generic
        // numeric loop defeats the fused layer dispatch and is slower than composed GEMMs.
        if (typeof(T) == typeof(double))
        {
            var output = LstmSequenceForwardDoubleTrain(
                (Tensor<double>)(object)input, (Tensor<double>?)(object?)h0, (Tensor<double>?)(object?)c0,
                (Tensor<double>)(object)wIh, (Tensor<double>)(object)wHh,
                (Tensor<double>?)(object?)bIh, (Tensor<double>?)(object?)bHh,
                batch, seqLen, inFeatures, hidden, gateRows, returnSequences, wantState,
                out var hn, out var cn);
            finalHidden = (Tensor<T>)(object)hn; finalCell = (Tensor<T>)(object)cn;
            return (Tensor<T>)(object)output;
        }

        return LstmSequenceForwardGeneric(
            input, h0, c0, wIh, wHh, bIh, bHh,
            batch, seqLen, inFeatures, hidden, gateRows, returnSequences, wantState,
            out finalHidden, out finalCell);
    }

    [MethodImpl(Hot)]
    private Tensor<T> LstmSequenceForwardGraph<T>(
        Tensor<T> input,
        Tensor<T>? h0,
        Tensor<T>? c0,
        Tensor<T> wIh,
        Tensor<T> wHh,
        Tensor<T>? bIh,
        Tensor<T>? bHh,
        int batch,
        int seqLen,
        int inFeatures,
        int hidden,
        bool returnSequences,
        out Tensor<T> finalHidden,
        out Tensor<T> finalCell)
    {
        if (seqLen == 0)
            throw new NotSupportedException(
                "Graph capture of an empty LSTM sequence has no timestep node from which to derive its outputs.");

        Tensor<T> currentH = h0 ?? new Tensor<T>(new[] { batch, hidden });
        Tensor<T> currentC = c0 ?? new Tensor<T>(new[] { batch, hidden });
        var hiddenStates = returnSequences ? new List<Tensor<T>>(seqLen) : null;

        for (int t = 0; t < seqLen; t++)
        {
            var xt = Reshape(
                TensorSlice(input, new[] { 0, t, 0 }, new[] { batch, 1, inFeatures }),
                new[] { batch, inFeatures });
            var gates = TensorAdd(
                TensorMatMulTransposed(xt, wIh),
                TensorMatMulTransposed(currentH, wHh));
            if (bIh is not null) gates = TensorAdd(gates, bIh);
            if (bHh is not null) gates = TensorAdd(gates, bHh);

            var inputGate = Sigmoid(TensorSlice(gates, new[] { 0, 0 }, new[] { batch, hidden }));
            var forgetGate = Sigmoid(TensorSlice(gates, new[] { 0, hidden }, new[] { batch, hidden }));
            var candidate = Tanh(TensorSlice(gates, new[] { 0, 2 * hidden }, new[] { batch, hidden }));
            var outputGate = Sigmoid(TensorSlice(gates, new[] { 0, 3 * hidden }, new[] { batch, hidden }));

            currentC = TensorAdd(
                TensorMultiply(forgetGate, currentC),
                TensorMultiply(inputGate, candidate));
            currentH = TensorMultiply(outputGate, Tanh(currentC));
            hiddenStates?.Add(Reshape(currentH, new[] { batch, 1, hidden }));
        }

        finalHidden = currentH;
        finalCell = currentC;
        return returnSequences
            ? TensorConcatenate(hiddenStates!.ToArray(), axis: 1)
            : currentH;
    }

    internal static void UnpackLstmSequenceInputs<T>(
        Tensor<T>[] inputs,
        LstmSequenceOptionalInputs optionalInputs,
        out Tensor<T> input,
        out Tensor<T>? h0,
        out Tensor<T>? c0,
        out Tensor<T> wIh,
        out Tensor<T> wHh,
        out Tensor<T>? bIh,
        out Tensor<T>? bHh)
    {
        const LstmSequenceOptionalInputs supportedOptionalInputs =
            LstmSequenceOptionalInputs.InitialHidden |
            LstmSequenceOptionalInputs.InitialCell |
            LstmSequenceOptionalInputs.InputBias |
            LstmSequenceOptionalInputs.RecurrentBias;
        if ((optionalInputs & ~supportedOptionalInputs) != 0)
            throw new InvalidDataException(
                $"LSTM replay contains unsupported optional-input flags 0x{(byte)(optionalInputs & ~supportedOptionalInputs):X2}.");

        const int requiredInputCount = 3;
        int expectedInputCount = requiredInputCount;
        if ((optionalInputs & LstmSequenceOptionalInputs.InitialHidden) != 0) expectedInputCount++;
        if ((optionalInputs & LstmSequenceOptionalInputs.InitialCell) != 0) expectedInputCount++;
        if ((optionalInputs & LstmSequenceOptionalInputs.InputBias) != 0) expectedInputCount++;
        if ((optionalInputs & LstmSequenceOptionalInputs.RecurrentBias) != 0) expectedInputCount++;
        if (inputs.Length != expectedInputCount)
            throw new InvalidDataException(
                $"LSTM replay flags describe {expectedInputCount} inputs, but the node contains {inputs.Length}.");

        int index = 0;
        input = inputs[index++];
        wIh = inputs[index++];
        wHh = inputs[index++];
        h0 = optionalInputs.HasFlag(LstmSequenceOptionalInputs.InitialHidden) ? inputs[index++] : null;
        c0 = optionalInputs.HasFlag(LstmSequenceOptionalInputs.InitialCell) ? inputs[index++] : null;
        bIh = optionalInputs.HasFlag(LstmSequenceOptionalInputs.InputBias) ? inputs[index++] : null;
        bHh = optionalInputs.HasFlag(LstmSequenceOptionalInputs.RecurrentBias) ? inputs[index++] : null;
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
    [MethodImpl(Hot)]
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

        // A single sequence can project the original recurrent weights without a transpose.
        if (batch == 1)
            return LstmSingleSequenceForwardFloat(input, h0, c0, wIh, wHh, bIh, bHh,
                seqLen, inFeatures, hidden, gateRows, returnSequences, wantState, out finalHidden, out finalCell);

        // Pre-compute Wx = input @ wIh^T as one big GEMM via SimdGemm.
        // input is [B, seq, in] — treat as [B*seq, in] row-major.
        // wIh is [4H, in] — we want input @ wIh^T which is [B*seq, 4H].
        // Avoid the Tensor<T> dispatch path for the contiguous fast case;
        // call SimdGemm directly using transB=true.
        int totalRows = batch * seqLen;
        var pool = ArrayPool<float>.Shared;

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

        const int batchTileRows = 6;
        int rowsPerWorker = 1 + (batch - 1) / numChunks;
        long packedElements = ((long)inFeatures + hidden) * gateRows +
            (long)numChunks * batchTileRows * (((long)seqLen + 1) * gateRows + 2L * hidden);
        // One complete GEMM row block and a short tail reuse the same projection storage.
        // With one worker, packing W_ih must cost less than the projection rows it replaces.
        if (rowsPerWorker > batchTileRows && rowsPerWorker <= 2 * batchTileRows && packedElements <= int.MaxValue / sizeof(float) &&
            (numChunks > 1 || (long)(batch - batchTileRows) * seqLen > inFeatures))
            return LstmBatchedTiledForwardFloat(input, h0, c0, wIh, wHh, bIh, bHh, batch, seqLen,
                inFeatures, hidden, gateRows, numChunks, rowsPerWorker, batchTileRows, returnSequences, wantState,
                out finalHidden, out finalCell);


        // #436: with a fan-out, each chunk projects its OWN rows (Wx for [rows*seq, in]) into chunk-local scratch
        // inside the same dispatch, instead of one parallel GEMM over every row followed by the recurrence
        // region. That removes a fork/join and the [B*seq, 4H] buffer every core wrote and every chunk then read
        // across chiplets step by step (4 MB at the AIsEval shape); the chunk-local projection stays in L2.
        bool perChunkProjection = numChunks > 1;

        // The whole-batch Wx buffer exists only without a fan-out.
        var wxBuf = perChunkProjection ? System.Array.Empty<float>() : CpuStorageAllocationDiagnostics.RentWorkspace<float>(totalRows * gateRows, "workspace.lstm_forward_pool");
        // #477: pool the transposed hidden weight too. It used to be `new float[]` per
        // call (64 KB at the AIsEval shape) because SgemmWithCachedB keyed its pre-pack
        // cache on the array identity. Now the recurrent GEMM is SgemmSequential (no
        // identity cache), so wHhT can come from the pool — removing the single biggest
        // per-call allocation (the main GC-churn / p95 source).
        var wHhTBuf = CpuStorageAllocationDiagnostics.RentWorkspace<float>(hidden * gateRows, "workspace.lstm_forward_pool");
        float[]? wIhT = null;   // [in, gateRows], rented only for the per-chunk projection

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
            // LSTM is also used after channel/sequence permutations (for example,
            // Demucs feeds a [B, seq, channels] view).  Keep the zero-copy hot path
            // for ordinary tensors while materializing only operands whose logical
            // row-major order is not represented by one contiguous span.
            var inputData = input.GetFlattenedData();
            var wIhData = wIh.GetFlattenedData();
            float[]? bIhFlat = perChunkProjection ? bIh?.GetFlattenedData() : null;

            // Wx = input @ wIh^T using SimdGemm with transB=true.
            // input rows = totalRows, K = inFeatures, wIh rows = gateRows, ldb = inFeatures.
            if (perChunkProjection)
            {
                // [gateRows, in] -> [in, gateRows]: the [K, N] row-major layout SgemmSequential takes.
                wIhT = CpuStorageAllocationDiagnostics.RentWorkspace<float>(inFeatures * gateRows, "workspace.lstm_forward_pool");
                for (int g = 0; g < gateRows; g++)
                    for (int f = 0; f < inFeatures; f++)
                        wIhT[f * gateRows + g] = wIhData[g * inFeatures + f];
            }
            else
            SimdGemm.Sgemm(
                inputData.AsSpan(), inFeatures, false,
                wIhData.AsSpan(), inFeatures, true,
                wxBuf.AsSpan(0, totalRows * gateRows),
                totalRows, inFeatures, gateRows);

            // Fold bIh into Wx (broadcast add) — SIMD-friendly contiguous loop.
            if (bIh is not null && !perChunkProjection)
            {
                var bIhArr = bIh.GetFlattenedData().AsSpan();
                for (int r = 0; r < totalRows; r++)
                {
                    int off = r * gateRows;
                    for (int g = 0; g < gateRows; g++)
                        wxBuf[off + g] += bIhArr[g];
                }
            }

            var wHhData = wHh.GetFlattenedData();
            float[]? bHhArr = bHh?.GetFlattenedData();
            float[]? h0Arr = h0?.GetFlattenedData();
            float[]? c0Arr = c0?.GetFlattenedData();

            // #477: the recurrent GEMM h_prev @ wHh^T runs once per timestep. The old
            // transB=true Sgemm re-transposed wHh on EVERY step. Pre-transpose wHh
            // [gateRows, hidden] → wHhT [hidden, gateRows] ONCE; [hidden, gateRows] is
            // exactly the [K, N] row-major layout SgemmSequential expects, so the
            // per-step call needs no further transpose or packing setup.
            var wHhT = wHhTBuf;
            SimdGemm.Transpose(wHhData, 0, hidden, wHhT, 0, gateRows, gateRows, hidden);

            using var outArrLease = output.LeaseArray();
            var outArr = (float[])(object)outArrLease.Array;
            float[]? fhArr = wantState ? (float[])(object)finalHidden.GetDataArrayUnmarked() : null;
            float[]? fcArr = wantState ? (float[])(object)finalCell.GetDataArrayUnmarked() : null;

            if (numChunks <= 1)
            {
                // Serial: one chunk covering the full batch, reusing pooled scratch.
                var hPrevBuf = CpuStorageAllocationDiagnostics.RentWorkspace<float>(batch * hidden, "workspace.lstm_forward_pool");
                var hCurrBuf = CpuStorageAllocationDiagnostics.RentWorkspace<float>(batch * hidden, "workspace.lstm_forward_pool");
                var cPrevBuf = CpuStorageAllocationDiagnostics.RentWorkspace<float>(batch * hidden, "workspace.lstm_forward_pool");
                var cCurrBuf = CpuStorageAllocationDiagnostics.RentWorkspace<float>(batch * hidden, "workspace.lstm_forward_pool");
                var hhBuf = CpuStorageAllocationDiagnostics.RentWorkspace<float>(batch * gateRows, "workspace.lstm_forward_pool");
                try
                {
                    RunLstmRecurrenceRange(
                        wxBuf, 0, wHhT, bHhArr, h0Arr, c0Arr,
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
                    var hPrevL = CpuStorageAllocationDiagnostics.RentWorkspace<float>(rows * hidden, "workspace.lstm_forward_pool");
                    var hCurrL = CpuStorageAllocationDiagnostics.RentWorkspace<float>(rows * hidden, "workspace.lstm_forward_pool");
                    var cPrevL = CpuStorageAllocationDiagnostics.RentWorkspace<float>(rows * hidden, "workspace.lstm_forward_pool");
                    var cCurrL = CpuStorageAllocationDiagnostics.RentWorkspace<float>(rows * hidden, "workspace.lstm_forward_pool");
                    var hhL = CpuStorageAllocationDiagnostics.RentWorkspace<float>(rows * gateRows, "workspace.lstm_forward_pool");
                    var wxL = CpuStorageAllocationDiagnostics.RentWorkspace<float>(rows * seqLen * gateRows, "workspace.lstm_forward_pool");
                    try
                    {
                        // This chunk's input projection: its [rows*seq, in] input rows @ wIhT, plus bIh.
                        SimdGemm.SgemmSequential(
                            inputData.AsSpan(r0 * seqLen * inFeatures, rows * seqLen * inFeatures),
                            wIhT!.AsSpan(0, inFeatures * gateRows),
                            wxL.AsSpan(0, rows * seqLen * gateRows),
                            rows * seqLen, inFeatures, gateRows);
                        if (bIhFlat is not null)
                            for (int r = 0; r < rows * seqLen; r++)
                            {
                                int off = r * gateRows;
                                for (int g = 0; g < gateRows; g++)
                                    wxL[off + g] += bIhFlat[g];
                            }
                        RunLstmRecurrenceRange(
                            wxL, r0, wHhT, bHhArr, h0Arr, c0Arr,
                            hPrevL, hCurrL, cPrevL, cCurrL, hhL,
                            outArr, returnSequences, fhArr, fcArr,
                            r0, rows, seqLen, hidden, gateRows);
                    }
                    finally
                    {
                        pool.Return(hPrevL); pool.Return(hCurrL);
                        pool.Return(cPrevL); pool.Return(cCurrL); pool.Return(hhL);
                        pool.Return(wxL);
                    }
                });
            }

            return output;
        }
        finally
        {
            if (!perChunkProjection) pool.Return(wxBuf);
            pool.Return(wHhTBuf);
            if (wIhT is not null) pool.Return(wIhT);
        }
    }

    private static unsafe Tensor<float> LstmBatchedTiledForwardFloat(Tensor<float> input, Tensor<float>? h0,
        Tensor<float>? c0, Tensor<float> wIh, Tensor<float> wHh, Tensor<float>? bIh, Tensor<float>? bHh,
        int batch, int steps, int inputs, int hidden, int gates, int workers, int rowsPerWorker, int tileRows,
        bool returnSequences, bool wantState, out Tensor<float> finalHidden, out Tensor<float> finalCell)
    {
        var output = AutoTensorCache.RentOrAllocate<float>(returnSequences ? new[] { batch, steps, hidden } : new[] { batch, hidden });
        finalHidden = wantState ? new Tensor<float>(new[] { batch, hidden }) : s_emptyState;
        finalCell = wantState ? new Tensor<float>(new[] { batch, hidden }) : s_emptyState;
        float[]? lastHidden = wantState ? finalHidden.GetDataArrayUnmarked() : null;
        float[]? lastCell = wantState ? finalCell.GetDataArrayUnmarked() : null;
        var inputValues = input.GetFlattenedData();
        var inputWeights = wIh.GetFlattenedData();
        var recurrentWeights = wHh.GetFlattenedData();
        var initialHidden = h0?.GetFlattenedData();
        var initialCell = c0?.GetFlattenedData();
        var inputBias = bIh?.GetFlattenedData();
        var hiddenBias = bHh?.GetFlattenedData();

        int recurrentWeightOffset = checked(inputs * gates);
        int workerStart = checked(recurrentWeightOffset + hidden * gates);
        int projectionLength = checked(tileRows * steps * gates);
        int recurrentOffset = projectionLength;
        int hiddenOffset = checked(recurrentOffset + tileRows * gates);
        int cellOffset = checked(hiddenOffset + tileRows * hidden);
        int stride = checked(cellOffset + tileRows * hidden);
        var workspace = CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(workerStart + workers * stride), "workspace.lstm_forward_pool");
        try
        {
            SimdGemm.Transpose(inputWeights, 0, inputs, workspace, 0, gates, gates, inputs);
            SimdGemm.Transpose(recurrentWeights, 0, hidden, workspace, recurrentWeightOffset, gates, gates, hidden);
            using var lease = output.LeaseArray();
            var outputValues = lease.Array;
            void RunWorker(int worker)
            {
                int workerOffset = workerStart + worker * stride;
                int endRow = Math.Min(batch, (worker + 1) * rowsPerWorker);
                for (int firstRow = worker * rowsPerWorker; firstRow < endRow; firstRow += tileRows)
                {
                    int rows = Math.Min(tileRows, endRow - firstRow);
                    var projected = workspace.AsSpan(workerOffset, rows * steps * gates);
                    var recurrent = workspace.AsSpan(workerOffset + recurrentOffset, rows * gates);
                    var hiddenState = workspace.AsSpan(workerOffset + hiddenOffset, rows * hidden);
                    var cellState = workspace.AsSpan(workerOffset + cellOffset, rows * hidden);
                    if (initialHidden is null) hiddenState.Clear();
                    else initialHidden.AsSpan(firstRow * hidden, hiddenState.Length).CopyTo(hiddenState);
                    if (initialCell is null) cellState.Clear();
                    else initialCell.AsSpan(firstRow * hidden, cellState.Length).CopyTo(cellState);

                    SimdGemm.SgemmSequential(inputValues.AsSpan(firstRow * steps * inputs, rows * steps * inputs),
                        workspace.AsSpan(0, recurrentWeightOffset), projected, rows * steps, inputs, gates);
                    if (inputBias is not null)
                        for (int r = 0; r < rows * steps; r++)
                        {
                            var row = projected.Slice(r * gates, gates);
                            SimdKernels.VectorAdd(row, inputBias.AsSpan(0, gates), row);
                        }

                    fixed (float* memory = workspace, bias = hiddenBias, result = outputValues)
                    {
                        for (int t = 0; t < steps; t++)
                        {
                            SimdGemm.SgemmRecurrentOverwrite(hiddenState,
                                workspace.AsSpan(recurrentWeightOffset, hidden * gates), recurrent, rows, hidden, gates);
                            for (int r = 0; r < rows; r++)
                            {
                                float* cell = memory + workerOffset + cellOffset + r * hidden;
                                float* state = memory + workerOffset + hiddenOffset + r * hidden;
                                UpdateLstmCellRow(memory + workerOffset + (r * steps + t) * gates,
                                    memory + workerOffset + recurrentOffset + r * gates, bias, cell, cell, state,
                                    returnSequences ? result + ((firstRow + r) * steps + t) * hidden : null, hidden);
                            }
                        }
                    }
                    if (!returnSequences) hiddenState.CopyTo(outputValues.AsSpan(firstRow * hidden, hiddenState.Length));
                    if (lastHidden is not null) hiddenState.CopyTo(lastHidden.AsSpan(firstRow * hidden, hiddenState.Length));
                    if (lastCell is not null) cellState.CopyTo(lastCell.AsSpan(firstRow * hidden, cellState.Length));
                }
            }
            if (workers == 1) RunWorker(0);
            else Helpers.PersistentParallelExecutor.Instance.Execute(workers, RunWorker);
            return output;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(workspace);
        }
    }


    private static unsafe Tensor<float> LstmSingleSequenceForwardFloat(
        Tensor<float> input, Tensor<float>? h0, Tensor<float>? c0, Tensor<float> wIh, Tensor<float> wHh,
        Tensor<float>? bIh, Tensor<float>? bHh, int seqLen, int inFeatures, int hidden, int gateRows,
        bool returnSequences, bool wantState, out Tensor<float> finalHidden, out Tensor<float> finalCell)
    {
        int tileRows = Math.Min(seqLen, 8);
        int gateOffset = checked(tileRows * gateRows);
        int hOffset = checked(gateOffset + gateRows);
        int cOffset = checked(hOffset + hidden);
        var output = AutoTensorCache.RentOrAllocate<float>(returnSequences ? new[] { 1, seqLen, hidden } : new[] { 1, hidden });
        finalHidden = wantState ? new Tensor<float>(new[] { 1, hidden }) : s_emptyState;
        finalCell = wantState ? new Tensor<float>(new[] { 1, hidden }) : s_emptyState;
        var inputData = input.GetFlattenedData();
        var inputWeights = wIh.GetFlattenedData();
        var hiddenWeights = wHh.GetFlattenedData();
        var inputBias = bIh?.GetFlattenedData();
        var hiddenBias = bHh?.GetFlattenedData();
        var workspace = CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(cOffset + hidden), "workspace.lstm_forward_pool");
        try
        {
            var gates = workspace.AsSpan(gateOffset, gateRows);
            var hiddenState = workspace.AsSpan(hOffset, hidden);
            var cellState = workspace.AsSpan(cOffset, hidden);
            if (h0 is null) hiddenState.Clear();
            else h0.GetFlattenedData().AsSpan(0, hidden).CopyTo(hiddenState);
            if (c0 is null) cellState.Clear();
            else c0.GetFlattenedData().AsSpan(0, hidden).CopyTo(cellState);
            using var outputLease = output.LeaseWritable();
            for (int first = 0; first < seqLen; first += tileRows)
            {
                int rows = Math.Min(tileRows, seqLen - first);
                var projection = workspace.AsSpan(0, rows * gateRows);
                SimdGemm.Sgemm(inputData.AsSpan(first * inFeatures, rows * inFeatures), inFeatures, false,
                    inputWeights, inFeatures, true, projection, rows, inFeatures, gateRows);
                for (int t = 0; t < rows; t++)
                {
                    // W_hh is [4H,H]. Matrix-times-vector writes all gates without a transpose.
                    BlasManaged.GemvKernel.Run<float>(hiddenWeights, hidden, false, hiddenState, 1, false,
                        gates, 1, gateRows, 1, hidden);
                    var wx = projection.Slice(t * gateRows, gateRows);
                    if (inputBias is not null) SimdKernels.VectorAdd(wx, inputBias.AsSpan(0, gateRows), wx);
                    fixed (float* buffer = workspace, bias = hiddenBias)
                        UpdateLstmCellRow(buffer + t * gateRows, buffer + gateOffset, bias,
                            buffer + cOffset, buffer + cOffset, buffer + hOffset, null, hidden);
                    if (returnSequences)
                        hiddenState.CopyTo(outputLease.Span.Slice((first + t) * hidden, hidden));
                }
            }
            if (!returnSequences) hiddenState.CopyTo(outputLease.Span);
            if (wantState)
            {
                using var h = finalHidden.LeaseWritable();
                using var c = finalCell.LeaseWritable();
                hiddenState.CopyTo(h.Span);
                cellState.CopyTo(c.Span);
            }
            return output;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(workspace);
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
    /// here. Shared reads: <paramref name="wxBuf"/> (pre-computed input projection; global batch row
    /// <c>gb</c> is its row <c>gb - wxRowBase</c>, so 0 for the whole-batch buffer and r0 for a chunk-local one), <paramref name="wHhT"/>, bias, h0/c0. Shared writes:
    /// disjoint global row ranges of <paramref name="outArr"/> and the optional final
    /// state arrays — no two ranges overlap, so concurrent chunks need no locking.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void RunLstmRecurrenceRange(
        float[] wxBuf, int wxRowBase, float[] wHhT, float[]? bHhArr, float[]? h0Arr, float[]? c0Arr,
        float[] hPrev, float[] hCurr, float[] cPrev, float[] cCurr, float[] hh,
        float[] outArr, bool returnSequences, float[]? fhArr, float[]? fcArr,
        int r0, int rows, int seqLen, int hidden, int gateRows)
    {
        // Initial states for this row range. Local buffers are indexed [localRow*hidden];
        // h0/c0 are global [batch, hidden] so they are read at the global offset.
        if (h0Arr is null) Array.Clear(hPrev, 0, rows * hidden);
        else Array.Copy(h0Arr, r0 * hidden, hPrev, 0, rows * hidden);
        if (c0Arr is null) Array.Clear(cPrev, 0, rows * hidden);
        else Array.Copy(c0Arr, r0 * hidden, cPrev, 0, rows * hidden);

        // Local ping-pong references (swapped each step). Using locals keeps the
        // caller's array handles fixed for the pool.Return in its finally.
        float[] hP = hPrev, hC = hCurr, cP = cPrev, cC = cCurr;

        for (int t = 0; t < seqLen; t++)
        {
            // hh = h_prev @ wHhT, [rows, hidden] @ [hidden, gateRows]. The write-first route
            // skips redundant clearing while retaining the serial direct kernel. It
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
            SimdGemm.SgemmRecurrentOverwrite(
                hP.AsSpan(0, rows * hidden),
                wHhT.AsSpan(0, hidden * gateRows),
                hh.AsSpan(0, rows * gateRows),
                rows, hidden, gateRows);

            fixed (float* wx = wxBuf, recurrent = hh, bias = bHhArr, previousCell = cP, nextCell = cC,
                nextHidden = hC, output = outArr)
            {
                for (int localRow = 0; localRow < rows; localRow++)
                {
                    int batchRow = r0 + localRow;
                    int projectionRow = ((batchRow - wxRowBase) * seqLen + t) * gateRows;
                    int stateRow = localRow * hidden;
                    UpdateLstmCellRow(wx + projectionRow, recurrent + localRow * gateRows, bias,
                        previousCell + stateRow, nextCell + stateRow, nextHidden + stateRow,
                        returnSequences ? output + (batchRow * seqLen + t) * hidden : null, hidden);
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

    // All recurrent gates for a row are ready before the cell and hidden state are written.
    // That allows the bounded workspace to update both states in place.
    [MethodImpl(Hot)]
    private static unsafe void UpdateLstmCellRow(float* projection, float* recurrent, float* bias,
        float* previousCell, float* nextCell, float* nextHidden, float* output, int hidden)
    {
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported && hidden % 8 == 0)
        {
            for (int h = 0; h < hidden; h += 8)
            {
                var iIn = Avx.Add(Avx.LoadVector256(projection + h), Avx.LoadVector256(recurrent + h));
                var fIn = Avx.Add(Avx.LoadVector256(projection + hidden + h), Avx.LoadVector256(recurrent + hidden + h));
                var gIn = Avx.Add(Avx.LoadVector256(projection + 2 * hidden + h), Avx.LoadVector256(recurrent + 2 * hidden + h));
                var oIn = Avx.Add(Avx.LoadVector256(projection + 3 * hidden + h), Avx.LoadVector256(recurrent + 3 * hidden + h));
                if (bias != null)
                {
                    iIn = Avx.Add(iIn, Avx.LoadVector256(bias + h));
                    fIn = Avx.Add(fIn, Avx.LoadVector256(bias + hidden + h));
                    gIn = Avx.Add(gIn, Avx.LoadVector256(bias + 2 * hidden + h));
                    oIn = Avx.Add(oIn, Avx.LoadVector256(bias + 3 * hidden + h));
                }

                var iG = SimdKernels.FastSigmoid256(iIn);
                var fG = SimdKernels.FastSigmoid256(fIn);
                var gG = SimdKernels.FastTanh256(gIn);
                var oG = SimdKernels.FastSigmoid256(oIn);

                var cp = Avx.LoadVector256(previousCell + h);
                var c = Fma.MultiplyAdd(fG, cp, Avx.Multiply(iG, gG)); // f·c_prev + i·g
                Avx.Store(nextCell + h, c);
                var hv = Avx.Multiply(oG, SimdKernels.FastTanh256(c)); // o·tanh(c)
                Avx.Store(nextHidden + h, hv);
                if (output != null)
                    Avx.Store(output + h, hv);
            }
            return;
        }
#endif
        for (int h = 0; h < hidden; h++)
        {
            float iIn = projection[h] + recurrent[h];
            float fIn = projection[hidden + h] + recurrent[hidden + h];
            float gIn = projection[2 * hidden + h] + recurrent[2 * hidden + h];
            float oIn = projection[3 * hidden + h] + recurrent[3 * hidden + h];
            if (bias != null)
            {
                iIn += bias[h]; fIn += bias[hidden + h];
                gIn += bias[2 * hidden + h]; oIn += bias[3 * hidden + h];
            }
            float iG = 1f / (1f + (float)Math.Exp(-iIn));
            float fG = 1f / (1f + (float)Math.Exp(-fIn));
            float gG = (float)Math.Tanh(gIn);
            float oG = 1f / (1f + (float)Math.Exp(-oIn));
            float c = fG * previousCell[h] + iG * gG;
            nextCell[h] = c;
            float hv = oG * (float)Math.Tanh(c);
            nextHidden[h] = hv;
            if (output != null) output[h] = hv;
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
        // #478: pool the one-time input projection Wx = input @ wIh^T + bIh too (the dominant residual
        // after the per-step buffers below). The old TensorMatMulTransposed(inputFlat, wIh) allocated a
        // fresh [B*seq, 4*hidden] output; pre-transpose wIh once and MultiplyBlocked into a rented wxBuf
        // (returned in the finally with the per-step buffers). Mirrors the float fast path's wxBuf.
        var lstmPool = ArrayPool<T>.Shared;
        int wxRows = batch * seqLen;
        var wIhT = CpuStorageAllocationDiagnostics.RentWorkspace<T>(inFeatures * gateRows, "workspace.lstm_forward_pool");
        var wxBuf = CpuStorageAllocationDiagnostics.RentWorkspace<T>(wxRows * gateRows, "workspace.lstm_forward_pool");
        {
            var wIhSrc = wIh.GetFlattenedData().AsSpan();
            for (int g = 0; g < gateRows; g++)
                for (int fcol = 0; fcol < inFeatures; fcol++)
                    wIhT[fcol * gateRows + g] = wIhSrc[g * inFeatures + fcol];
        }
        var wxMem = new Memory<T>(wxBuf, 0, wxRows * gateRows);
        wxMem.Span.Clear();
        MatrixMultiplyHelper.MultiplyBlocked(MathHelper.GetNumericOperations<T>(),
            new ReadOnlyMemory<T>(input.GetFlattenedData(), 0, wxRows * inFeatures),
            new ReadOnlyMemory<T>(wIhT, 0, inFeatures * gateRows),
            wxMem, wxRows, inFeatures, gateRows, inFeatures, gateRows, gateRows, allowParallel: true);
        lstmPool.Return(wIhT);

        if (bIh is not null)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var bIhSpan = bIh.GetFlattenedData().AsSpan();
            for (int r = 0; r < wxRows; r++)
            {
                int off = r * gateRows;
                for (int g = 0; g < gateRows; g++)
                    wxBuf[off + g] = ops.Add(wxBuf[off + g], bIhSpan[g]);
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
        var wxSpanRO = new ReadOnlySpan<T>(wxBuf, 0, wxRows * gateRows);
        var bHhData = bHh?.GetFlattenedData();

        // #478: pool the per-timestep hidden GEMM. The old TensorMatMulTransposed(hPrev, wHh) allocated
        // a fresh [batch, 4*hidden] output (plus the GEMM's packing scratch) EVERY step. Pre-transpose
        // wHh once into a rented buffer and MultiplyBlocked into ONE reused hh buffer, mirroring the
        // float fast path's wHhT + hhBuf — so the recurrence loop allocates nothing per step. This is
        // the dominant remaining allocation after the ping-pong state buffers above.
        var wHhT = CpuStorageAllocationDiagnostics.RentWorkspace<T>(hidden * gateRows, "workspace.lstm_forward_pool");
        var hhBuf = CpuStorageAllocationDiagnostics.RentWorkspace<T>(batch * gateRows, "workspace.lstm_forward_pool");
        {
            var wHhSrc = wHh.GetFlattenedData().AsSpan();
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
            using var hCurrSpanLease = hCurr.LeaseWritable();
            var hCurrSpan = hCurrSpanLease.Span;
            using var cCurrSpanLease = cCurr.LeaseWritable();
            var cCurrSpan = cCurrSpanLease.Span;
            var cPrevSpan = cbufs[cur].AsSpan();
            bool hasBhh = bHhData is not null;
            ReadOnlySpan<T> bHhSpan = hasBhh ? bHhData.AsSpan() : default;

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
                    T gAct = TanhScalar(gGate);
                    T oAct = Sigmoid(opsT, oGate);

                    T cNew = opsT.Add(opsT.Multiply(fAct, cPrevSpan[hOff + h]), opsT.Multiply(iAct, gAct));
                    T hNew = opsT.Multiply(oAct, TanhScalar(cNew));

                    cCurrSpan[hOff + h] = cNew;
                    hCurrSpan[hOff + h] = hNew;
                }
            }

            cur = nxt;

            if (returnSequences)
            {
                using var outSpanLease = output.LeaseWritable();
                var outSpan = outSpanLease.Span;
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
            lstmPool.Return(wxBuf);
        }

        if (!returnSequences)
        {
            using var outSpanLease = output.LeaseWritable();
            var outSpan = outSpanLease.Span;
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
    /// <summary>Seeds a ping-pong LSTM state buffer from an initial state, or zeroes it when none is given.</summary>
    /// <remarks>
    /// <c>default(T)</c> is the additive identity for the numeric structs this path serves, so
    /// <c>Clear()</c> is a correct zero rather than merely a cheap one.
    /// </remarks>
    private static void SeedLstmState<T>(Tensor<T> buffer, Tensor<T>? source, int n)
    {
        using var dstLease = buffer.LeaseWritable();
        var dst = dstLease.Span;
        if (source is null)
            dst.Slice(0, n).Clear();
        else
            source.GetFlattenedData().AsSpan(0, n).CopyTo(dst.Slice(0, n));
    }

    /// <summary>Logistic sigmoid over the generic numeric abstraction: 1 / (1 + e^-x).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Sigmoid<T>(INumericOperations<T> ops, T x)
    {
        // 1 / (1 + exp(-x))
        var negX = ops.Multiply(ops.FromDouble(-1.0), x);
        var expNeg = ops.Exp(negX);
        return ops.Divide(ops.One, ops.Add(ops.One, expNeg));
    }

    /// <summary>Hyperbolic tangent over the generic numeric abstraction, saturating to +/-1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T TanhScalar<T>(T x)
    {
        // (e^2x - 1) / (e^2x + 1) overflowed to Inf / Inf = NaN once e^2x left T's range (x > ~44 in float),
        // where tanh is exactly 1: a saturated LSTM cell state produced NaN hidden states.
        return MathHelper.Tanh(x);
    }
}
