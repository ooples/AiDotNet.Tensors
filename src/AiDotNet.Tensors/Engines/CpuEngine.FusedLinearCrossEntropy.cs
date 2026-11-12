using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // The float and double kernels below take the element type's own arrays and tensors. A T[] or Tensor<T> reaches
    // them only inside a typeof(T) branch, and C# has no direct conversion from an unconstrained T's array or tensor,
    // so this is the one checked reinterpretation (a wrong T throws InvalidCastException, never misreads memory).
    private static TTarget Reinterpret<TTarget>(object value) where TTarget : class => (TTarget)value;

    /// <summary>
    /// Fused linear (LM head) + cross-entropy-with-logits loss in a SINGLE op
    /// (forward scalar loss + custom autodiff backward), for the paper-scale LM head where the
    /// vocabulary is huge (issue ooples/AiDotNet#1464). Computes
    /// <c>logits = hidden·weight + bias</c> (<c>weight</c> is [d, vocab], matching DenseLayer's
    /// [inputSize, outputSize] layout), then the mean cross-entropy
    /// <c>L = -mean_r sum_v target[r,v]·log_softmax(logits_r)[v]</c>, and returns the scalar loss.
    ///
    /// <para>The win vs. the decomposed path (DenseLayer producing a [N, vocab] tape tensor, then the
    /// loss running log-softmax + multiply + reduce as separate tape ops): the full [N, vocab] logits
    /// are NEVER materialized on the autodiff tape, and the multiple full-vocab loss passes collapse
    /// into the single fused op. The (inherent) head GEMMs are still run on the fast parallel managed
    /// kernel — here via TensorMatMul inside a no-grad scope, so they execute without recording. The
    /// backward computes <c>dlogits = (softmax(logits) - target)/N</c> and then
    /// <c>dHidden = dlogits·weightᵀ</c>, <c>dWeight = hiddenᵀ·dlogits</c>, <c>dBias = sum_r dlogits</c>.</para>
    /// </summary>
    /// <param name="hidden">Pre-head hidden states [N, d] (N = batch·seq).</param>
    /// <param name="weight">LM head weight [d, vocab] (DenseLayer [inputSize, outputSize]).</param>
    /// <param name="bias">LM head bias [vocab].</param>
    /// <param name="targetIds">Per-row class id [N], each in [0, vocab).</param>
    /// <returns>Scalar (length-1, shape [1]) mean cross-entropy loss.</returns>
    public virtual Tensor<T> FusedLinearCrossEntropyWithLogits<T>(
        Tensor<T> hidden, Tensor<T> weight, Tensor<T> bias, Tensor<int> targetIds)
        => FusedLinearCrossEntropyWithLogits(hidden, weight, bias, targetIds, computeReadoutGrads: true);

    /// <summary>
    /// Fused linear (LM head) + cross-entropy-with-logits loss with optional readout-parameter gradient detachment.
    /// When <paramref name="computeReadoutGrads"/> is <see langword="false"/> (for example on intermediate layers in
    /// zero-backprop local-credit assignment or frozen-head distillation), only <paramref name="hidden"/> is recorded
    /// on the autodiff tape and neither <c>dWeight</c> (<c>[d, vocab]</c>) nor <c>dBias</c> (<c>[vocab]</c>) is
    /// allocated or computed.
    /// </summary>
    public virtual Tensor<T> FusedLinearCrossEntropyWithLogits<T>(
        Tensor<T> hidden, Tensor<T> weight, Tensor<T> bias, Tensor<int> targetIds, bool computeReadoutGrads)
    {
        if (hidden is null) throw new ArgumentNullException(nameof(hidden));
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (bias is null) throw new ArgumentNullException(nameof(bias));
        if (targetIds is null) throw new ArgumentNullException(nameof(targetIds));
        GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        GradientTape<T>.Current?.BindEngineIfUnset(this);
        if (hidden.Rank != 2)
            throw new ArgumentException($"hidden must be rank-2 [N, d]; got rank {hidden.Rank}.", nameof(hidden));
        if (weight.Rank != 2)
            throw new ArgumentException($"weight must be rank-2 [d, vocab]; got rank {weight.Rank}.", nameof(weight));

        int n = hidden.Shape[0];
        int d = hidden.Shape[1];
        int vocab = weight.Shape[1];
        if (n <= 0)
            throw new ArgumentException($"hidden must have a non-empty batch (N > 0); got N={n}.", nameof(hidden));
        if (vocab <= 0)
            throw new ArgumentException($"weight must have a non-empty vocab (dim1 > 0); got vocab={vocab}.", nameof(weight));
        if (weight.Shape[0] != d)
            throw new ArgumentException($"weight dim0 ({weight.Shape[0]}) must equal hidden dim1 ({d}).", nameof(weight));
        // Require rank-1 [vocab]: the backward returns a rank-1 [vocab] bias gradient, so a
        // higher-rank bias with the same flat length (e.g. [1, vocab]) would record a tape entry
        // whose gradient shape does not match the input.
        if (bias.Rank != 1 || bias.Length != vocab)
            throw new ArgumentException($"bias must be rank-1 [vocab={vocab}].", nameof(bias));
        if (targetIds.Rank != 1 || targetIds.Length != n)
            throw new ArgumentException($"targetIds must be rank-1 [N={n}].", nameof(targetIds));

        var ids = targetIds.GetDataArray();
        for (int r = 0; r < n; r++)
            if (ids[r] < 0 || ids[r] >= vocab)
                throw new ArgumentOutOfRangeException(nameof(targetIds),
                    $"targetIds[{r}] ({ids[r]}) must be in [0, vocab={vocab}).");

        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var capturedHidden = hidden;
            var capturedWeight = weight;
            var capturedBias = bias;
            var capturedTargetIds = targetIds;
            var readoutGuard = new ReadoutMutationGuard<T>(weight, bias);
            var compiledSavedState = new object[]
            {
                (int[])ids.Clone(), vocab, weight, bias, computeReadoutGrads, readoutGuard
            };
            if (!computeReadoutGrads)
            {
                // Keep weight and bias in the forward dependency graph with no backward writer so their producers
                // execute before the loss without allocating readout gradient buffers or adding readout backward edges.
                var lossForward = scope.RecordVariadic(
                    LazyNodeType.Custom,
                    "FusedLinearCrossEntropyReadoutForward",
                    new[] { hidden, weight, bias },
                    new[] { 1 },
                    (eng, output) =>
                    {
                        compiledSavedState[0] = (int[])capturedTargetIds.GetDataArray().Clone();
                        readoutGuard.Refresh(capturedWeight, capturedBias);
                        var result = eng.FusedLinearCrossEntropyWithLogits(
                            capturedHidden, capturedWeight, capturedBias, capturedTargetIds, computeReadoutGrads: false);
                        DirectGpuTensorEngine.CopyResultInto(eng, result, output);
                    },
                    backwardFn: null);
                return scope.RecordVariadic(
                    LazyNodeType.Custom,
                    "FusedLinearCrossEntropy",
                    new[] { hidden, lossForward },
                    new[] { 1 },
                    (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, lossForward, output),
                    FusedLinearCrossEntropyIndexBackward<T>,
                    compiledSavedState);
            }

            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "FusedLinearCrossEntropy",
                new[] { hidden, weight, bias },
                new[] { 1 },
                (eng, output) =>
                {
                    // Labels are not differentiable graph inputs, but they are live training data.
                    // Refresh the backward state on every replay instead of freezing the trace batch.
                    compiledSavedState[0] = (int[])capturedTargetIds.GetDataArray().Clone();
                    readoutGuard.Refresh(capturedWeight, capturedBias);
                    var result = eng.FusedLinearCrossEntropyWithLogits(
                        capturedHidden, capturedWeight, capturedBias, capturedTargetIds, computeReadoutGrads: true);
                    DirectGpuTensorEngine.CopyResultInto(eng, result, output);
                },
                FusedLinearCrossEntropyIndexBackward<T>,
                compiledSavedState);
        }

        // Training on the CPU in float: compute the gradients in the forward, while the logits are still in cache, so
        // the step runs the three head GEMMs (logits, dHidden, dWeight) once instead of recomputing the logits in the
        // backward (four), and the [N, vocab] buffer is freed before the forward returns.
        if (typeof(T) == typeof(float) && this is not DirectGpuTensorEngine && DifferentiableOps.IsRecording<T>())
            return FusedLinearCrossEntropyWithGradientsFloat(
                hidden, weight, bias, ids, n, d, vocab, FusedCeChunkElements, computeReadoutGrads);

        // logits = hidden·weight + bias, computed off-tape (this op records its own node).
        //
        // MATERIALIZE ONCE. TensorMatMul returns a GPU-RESIDENT tensor when this runs on the GPU engine
        // (which it does whenever a tape is active, because the DirectGpu override bails to this base
        // implementation). For a resident tensor GetDataArray() hands back a DETACHED COPY, and a fresh one
        // per call — so the previous code added the bias into one copy inside AddBiasRows and then computed
        // the loss from a SECOND copy that never saw the bias.
        //
        // Measured: forward loss with no tape agreed exactly (cpu 2.76591349 == gpu 2.76591349); with a tape
        // active the GPU value dropped to 2.35062456 — the bias-free loss — while CPU was unchanged. That
        // 4.153E-01 gap was the whole reported tape-gradient divergence for this op, and it reached every
        // leaf because the recorded backward reconstructs softmax from these same logits.
        Tensor<T> logits;
        T[] logitData;
        using (new NoGradScope<T>())
        {
            logits = TensorMatMul(hidden, weight);
            logitData = logits.GetDataArray();
            AddBiasRowsInto(logitData, bias, n, vocab);
        }

        double lossValue = typeof(T) == typeof(double)
            ? FusedCeLossIndexDouble(Reinterpret<double[]>(logitData), ids, n, vocab)
            : typeof(T) == typeof(float)
                ? FusedCeLossIndexFloat(Reinterpret<float[]>(logitData), ids, n, vocab)
                : FusedCeLossIndexGeneric<T>(logitData, ids, n, vocab);

        var output = new Tensor<T>(new[] { 1 }); // scalar loss (length-1)
        output.GetDataArray()[0] = MathHelper.GetNumericOperations<T>().FromDouble(lossValue);

        // target ids are supervision (no gradient): carry them in savedState rather than as a
        // differentiable tape input. Save a COPY so the backward is tied to the forward's labels —
        // targetIds is not a tape input, so mutating/recycling it after forward must not change the
        // gradient. Backward reconstructs dlogits = softmax - onehot(id).
        var tapeInputs = computeReadoutGrads ? new[] { hidden, weight, bias } : new[] { hidden };
        DifferentiableOps.RecordIfActive<T>(
            "FusedLinearCrossEntropy", output,
            tapeInputs,
            FusedLinearCrossEntropyIndexBackward<T>,
            savedState: new object[]
            {
                (int[])ids.Clone(), vocab, weight, bias, computeReadoutGrads,
                new ReadoutMutationGuard<T>(weight, bias)
            });

        return output;
    }

    /// <summary>
    /// Dense-target overload: <paramref name="target"/> is a full per-row distribution / one-hot
    /// [N, vocab] for soft / distillation supervision. Prefer the int-id overload for standard LM
    /// training to avoid the O(N·vocab) dense label storage/bandwidth.
    /// </summary>
    /// <param name="hidden">Pre-head hidden states [N, d] (N = batch·seq).</param>
    /// <param name="weight">LM head weight [d, vocab] (DenseLayer [inputSize, outputSize]).</param>
    /// <param name="bias">LM head bias [vocab].</param>
    /// <param name="target">Per-row categorical target distribution / one-hot [N, vocab].</param>
    /// <returns>Scalar (length-1, shape [1]) mean cross-entropy loss.</returns>
    public virtual Tensor<T> FusedLinearCrossEntropyWithLogits<T>(
        Tensor<T> hidden, Tensor<T> weight, Tensor<T> bias, Tensor<T> target)
    {
        if (hidden is null) throw new ArgumentNullException(nameof(hidden));
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (bias is null) throw new ArgumentNullException(nameof(bias));
        if (target is null) throw new ArgumentNullException(nameof(target));
        GradientTape<T>.Current?.BindEngineIfUnset(this);
        if (hidden.Rank != 2)
            throw new ArgumentException($"hidden must be rank-2 [N, d]; got rank {hidden.Rank}.", nameof(hidden));
        if (weight.Rank != 2)
            throw new ArgumentException($"weight must be rank-2 [d, vocab]; got rank {weight.Rank}.", nameof(weight));

        int n = hidden.Shape[0];
        int d = hidden.Shape[1];
        int vocab = weight.Shape[1];
        if (n <= 0)
            throw new ArgumentException($"hidden must have a non-empty batch (N > 0); got N={n}.", nameof(hidden));
        if (vocab <= 0)
            throw new ArgumentException($"weight must have a non-empty vocab (dim1 > 0); got vocab={vocab}.", nameof(weight));
        if (weight.Shape[0] != d)
            throw new ArgumentException($"weight dim0 ({weight.Shape[0]}) must equal hidden dim1 ({d}).", nameof(weight));
        // Require rank-1 [vocab]: the backward returns a rank-1 [vocab] bias gradient, so a
        // higher-rank bias with the same flat length (e.g. [1, vocab]) would record a tape entry
        // whose gradient shape does not match the input.
        if (bias.Rank != 1 || bias.Length != vocab)
            throw new ArgumentException($"bias must be rank-1 [vocab={vocab}].", nameof(bias));
        // Check rank before indexing Shape[Rank-1] (a rank-0 target would otherwise throw
        // IndexOutOfRangeException instead of this clear argument error).
        if (target.Rank != 2 || target.Shape[0] != n || target.Shape[1] != vocab)
            throw new ArgumentException($"target must be rank-2 [N={n}, vocab={vocab}].", nameof(target));

        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var capturedHidden = hidden;
            var capturedWeight = weight;
            var capturedBias = bias;
            var capturedTarget = target;
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "FusedLinearCrossEntropy",
                new[] { hidden, weight, bias, target },
                new[] { 1 },
                (eng, output) =>
                {
                    var result = eng.FusedLinearCrossEntropyWithLogits(
                        capturedHidden, capturedWeight, capturedBias, capturedTarget);
                    DirectGpuTensorEngine.CopyResultInto(eng, result, output);
                },
                FusedLinearCrossEntropyBackward<T>,
                savedState: null);
        }

        // logits = hidden·weight + bias, computed via the fast parallel GEMM but NOT recorded
        // (this op records its own single tape node). The logits tensor is transient — it never
        // enters the tape, so the big [N, vocab] allocation is freed after the forward.
        Tensor<T> logits;
        T[] logitData;
        using (new NoGradScope<T>())
        {
            logits = TensorMatMul(hidden, weight);              // [N, vocab]
            // Materialize ONCE — see the int-id forward: GetDataArray() returns a fresh detached copy
            // for a GPU-resident tensor, so adding the bias to one copy and reading another drops it.
            logitData = logits.GetDataArray();
            AddBiasRowsInto(logitData, bias, n, vocab);
        }

        double lossValue;
        if (typeof(T) == typeof(double))
        {
            lossValue = FusedCeLossDouble(
                Reinterpret<double[]>(logitData), Reinterpret<double[]>(target.GetDataArray()), n, vocab);
        }
        else
        {
            lossValue = FusedCeLossGeneric<T>(logitData, target.GetDataArray(), n, vocab);
        }

        var output = new Tensor<T>(new[] { 1 }); // scalar loss (length-1)
        output.GetDataArray()[0] = MathHelper.GetNumericOperations<T>().FromDouble(lossValue);

        DifferentiableOps.RecordIfActive<T>(
            "FusedLinearCrossEntropy", output,
            new[] { hidden, weight, bias, target },
            FusedLinearCrossEntropyBackward<T>,
            savedState: null);

        return output;
    }

    /// <summary>
    /// Adds the bias row-wise into an ALREADY-MATERIALIZED logits array.
    /// </summary>
    /// <remarks>
    /// Takes the array rather than the tensor so the caller performs exactly one GetDataArray(). For a
    /// GPU-resident tensor that call returns a detached copy each time, so fetching separately here and in
    /// the loss computation silently dropped the bias.
    /// </remarks>
    private static void AddBiasRowsInto<T>(T[] lo, Tensor<T> bias, int n, int vocab)
    {
        var bi = bias.GetDataArray();
        if (typeof(T) == typeof(float))
        {
            AddBiasRowsFloat(Reinterpret<float[]>(lo), Reinterpret<float[]>(bi), n, vocab);
        }
        else if (typeof(T) == typeof(double))
        {
            var l = Reinterpret<double[]>(lo); var b = Reinterpret<double[]>(bi);
            for (int r = 0; r < n; r++)
            {
                int off = r * vocab;
                for (int v = 0; v < vocab; v++) l[off + v] += b[v];
            }
        }
        else
        {
            var ops = MathHelper.GetNumericOperations<T>();
            for (int r = 0; r < n; r++)
            {
                int off = r * vocab;
                for (int v = 0; v < vocab; v++) lo[off + v] = ops.Add(lo[off + v], bi[v]);
            }
        }
    }

    // Mean CE = -(1/N) Σ_r Σ_v target[r,v]·(logit[r,v] - logSumExp_r). Numerically stable.
    [MethodImpl(Hot)]
    private static double FusedCeLossDouble(double[] logits, double[] target, int n, int vocab)
    {
        double total = 0.0;
        for (int r = 0; r < n; r++)
        {
            int off = r * vocab;
            double max = logits[off];
            for (int v = 1; v < vocab; v++) if (logits[off + v] > max) max = logits[off + v];
            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(logits[off + v] - max);
            double lse = max + Math.Log(sumExp);
            double rowLoss = 0.0;
            for (int v = 0; v < vocab; v++) rowLoss += target[off + v] * (logits[off + v] - lse);
            total += rowLoss;
        }
        return -total / n;
    }

    private static double FusedCeLossGeneric<T>(T[] logits, T[] target, int n, int vocab)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double total = 0.0;
        for (int r = 0; r < n; r++)
        {
            int off = r * vocab;
            double max = ops.ToDouble(logits[off]);
            for (int v = 1; v < vocab; v++) { double lv = ops.ToDouble(logits[off + v]); if (lv > max) max = lv; }
            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(ops.ToDouble(logits[off + v]) - max);
            double lse = max + Math.Log(sumExp);
            double rowLoss = 0.0;
            for (int v = 0; v < vocab; v++) rowLoss += ops.ToDouble(target[off + v]) * (ops.ToDouble(logits[off + v]) - lse);
            total += rowLoss;
        }
        return -total / n;
    }

    // Index-target mean CE = -(1/N) Σ_r (logit[r, id_r] - logSumExp_r). Numerically stable.
    [MethodImpl(Hot)]
    private static double FusedCeLossIndexDouble(double[] logits, int[] ids, int n, int vocab)
    {
        double total = 0.0;
        for (int r = 0; r < n; r++)
        {
            int off = r * vocab;
            double max = logits[off];
            for (int v = 1; v < vocab; v++) if (logits[off + v] > max) max = logits[off + v];
            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(logits[off + v] - max);
            double lse = max + Math.Log(sumExp);
            total += logits[off + ids[r]] - lse;
        }
        return -total / n;
    }

    private static double FusedCeLossIndexGeneric<T>(T[] logits, int[] ids, int n, int vocab)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double total = 0.0;
        for (int r = 0; r < n; r++)
        {
            int off = r * vocab;
            double max = ops.ToDouble(logits[off]);
            for (int v = 1; v < vocab; v++) { double lv = ops.ToDouble(logits[off + v]); if (lv > max) max = lv; }
            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(ops.ToDouble(logits[off + v]) - max);
            double lse = max + Math.Log(sumExp);
            total += ops.ToDouble(logits[off + ids[r]]) - lse;
        }
        return -total / n;
    }

    // Index-target backward. dlogits[r,v] = (softmax(logits)[r,v] - [v == id_r]) · g/N.
    internal static void FusedLinearCrossEntropyIndexBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        bool computeReadout = savedState.Length <= 4 || (bool)savedState[4];
        var hidden = inputs[0];
        var weight = computeReadout && inputs.Length > 1 ? inputs[1] : (Tensor<T>)savedState[2];
        var bias = computeReadout && inputs.Length > 2 ? inputs[2] : (Tensor<T>)savedState[3];
        var ids = (int[])savedState[0];
        int vocab = (int)savedState[1];
        ValidateDetachedReadoutVersions(weight, bias, savedState);

        bool needHidden = DifferentiableOps.IsGradientRequired(hidden);
        bool needWeight = computeReadout && inputs.Length > 1 && DifferentiableOps.IsGradientRequired(weight);
        bool needBias = computeReadout && inputs.Length > 2 && DifferentiableOps.IsGradientRequired(bias);
        if (!needHidden && !needWeight && !needBias)
            return;

        if (engine is DirectGpuTensorEngine gpu && gpu.TryFusedLinearCrossEntropyBackwardOnDevice(
                gradOutput, hidden, weight, bias, ids, null, needHidden, needWeight, needBias,
                out var deviceHidden, out var deviceWeight, out var deviceBias))
        {
            if (needHidden) DifferentiableOps.AccumulateGrad(grads, hidden, deviceHidden, engine);
            if (needWeight) DifferentiableOps.AccumulateGrad(grads, weight, deviceWeight, engine);
            if (needBias) DifferentiableOps.AccumulateGrad(grads, bias, deviceBias, engine);
            return;
        }

        int n = hidden.Shape[0];
        var ops = MathHelper.GetNumericOperations<T>();
        double g = ops.ToDouble(gradOutput.GetDataArray()[0]);

        var dLogits = new Tensor<T>(new[] { n, vocab });
        using (new NoGradScope<T>())
        {
            var logits = engine.TensorMatMul(hidden, weight);
            // Materialize ONCE — see the int-id forward: GetDataArray() returns a fresh detached copy
            // for a GPU-resident tensor, so adding the bias to one copy and reading another drops it.
            var logitData = logits.GetDataArray();
            AddBiasRowsInto(logitData, bias, n, vocab);
            if (typeof(T) == typeof(double))
            {
                SoftmaxMinusIndexDouble(
                    Reinterpret<double[]>(logitData), ids,
                    Reinterpret<double[]>(dLogits.GetDataArray()), n, vocab, g);
            }
            else if (typeof(T) == typeof(float))
            {
                SoftmaxMinusIndexFloat(
                    Reinterpret<float[]>(logitData), ids, Reinterpret<float[]>(dLogits.GetDataArray()), n, vocab, g);
            }
            else
            {
                SoftmaxMinusIndexGeneric<T>(logitData, ids, dLogits.GetDataArray(), n, vocab, g);
            }

            if (needHidden)
            {
                var dHidden = engine.TensorMatMulTransposed(dLogits, weight);
                DifferentiableOps.AccumulateGrad(grads, hidden, dHidden, engine);
            }
            if (needWeight)
            {
                var dWeight = engine.TensorMatMul(engine.TensorTranspose(hidden), dLogits);
                DifferentiableOps.AccumulateGrad(grads, weight, dWeight, engine);
            }
            if (needBias)
            {
                var dBias = ColumnSums(dLogits, n, vocab);
                DifferentiableOps.AccumulateGrad(grads, bias, dBias, engine);
            }
            // target ids are supervision — no gradient flows to them.
        }
    }

    /// <summary>
    /// Logit elements computed per chunk by the float training forward: 64M floats (256 MB). A 512 x 49152 LM-head
    /// batch is one chunk; larger batches are split by rows so the transient logits stay bounded.
    /// </summary>
    internal const long FusedCeChunkElements = 1L << 26;

    // The [rows, vocab] dLogits buffer of the float training forward, kept between steps. It never escapes the op, and
    // writing a fresh one is what made the logits GEMM slow: 64 threads faulting in new pages serialize on the
    // process's page tables, so h·W into a fresh 512 x 49152 buffer took 98.5 ms against 36.9 ms into a reused one on
    // a 3990X (median of 9). One buffer, taken with an exchange: a concurrent caller finding it taken allocates its own.
    private static float[]? s_fusedCeWorkspace;

    private static float[] TakeFusedCeWorkspace(long elements)
    {
        var buffer = System.Threading.Interlocked.Exchange(ref s_fusedCeWorkspace, null);
        return buffer is not null && buffer.LongLength == elements ? buffer : new float[elements];
    }

    private static void ReturnFusedCeWorkspace(float[] buffer) => System.Threading.Volatile.Write(ref s_fusedCeWorkspace, buffer);

    /// <summary>The gradients the float training forward computed, handed out by the first backward.</summary>
    internal sealed class FusedCePrecomputedGradients
    {
        public Tensor<float>? Hidden;
        public Tensor<float>? Weight;
        public Tensor<float>? Bias;
    }

    /// <summary>
    /// Float training forward: the loss plus all three gradients for a unit upstream gradient.
    /// </summary>
    /// <remarks>
    /// Per row chunk: logits = h·W + b (GEMM), then one fused SIMD pass per row that takes the row's loss and
    /// overwrites the row with dLogits = (softmax - onehot)/N, then dHidden = dLogits·Wᵀ and dWeight += hᵀ·dLogits (two
    /// GEMMs) and the bias column sums. The loss is a mean, so every gradient is linear in the upstream scalar: the
    /// backward scales these by it and runs no GEMM. A second backward over the same tape (a persistent tape) finds
    /// them handed out and recomputes through the ordinary backward.
    /// </remarks>
    internal Tensor<T> FusedLinearCrossEntropyWithGradientsFloat<T>(
        Tensor<T> hidden, Tensor<T> weight, Tensor<T> bias, int[] ids, int n, int d, int vocab, long chunkElements,
        bool computeReadoutGrads = true)
    {
        var h = Reinterpret<Tensor<float>>(hidden);
        var w = Reinterpret<Tensor<float>>(weight);
        var biasData = Reinterpret<Tensor<float>>(bias).GetDataArray();
        float scale = 1f / n;
        int rowsPerChunk = (int)Math.Max(1, Math.Min(n, chunkElements / vocab));
        var rowLoss = new double[n];
        var dBias = computeReadoutGrads ? new float[vocab] : null;
        Tensor<float> dHidden;
        Tensor<float>? dWeight = null;

        using (new NoGradScope<float>())
        {
            if (rowsPerChunk >= n)
            {
                var data = TakeFusedCeWorkspace((long)n * vocab);
                var dLogits = new Tensor<float>(data, new[] { n, vocab });
                MatMulInto(dLogits, h, w);
                CrossEntropyGradRowsFloat(data, biasData, ids, 0, n, vocab, scale, rowLoss);
                dHidden = DLogitsTimesWeightTransposed(data, w, n, d, vocab);
                if (computeReadoutGrads)
                {
                    dWeight = PrefaultedTensor(d, vocab);
                    MatMulInto(dWeight, TensorTranspose(h), dLogits);
                    ColumnSumsFloat(data, dBias!, n, vocab);
                }
                ReturnFusedCeWorkspace(data);
            }
            else
            {
                var hData = h.IsContiguous ? h.GetDataArray() : h.Contiguous().GetDataArray();
                var dHiddenData = new float[(long)n * d];
                // The first chunk's product becomes dWeight; later chunks write into one scratch and add it in.
                Tensor<float>? dwcScratch = null;
                for (int start = 0; start < n; start += rowsPerChunk)
                {
                    int rows = Math.Min(rowsPerChunk, n - start);
                    var hc = new Tensor<float>(new float[(long)rows * d], new[] { rows, d });
                    Array.Copy(hData, (long)start * d, hc.GetDataArray(), 0, (long)rows * d);
                    var data = TakeFusedCeWorkspace((long)rows * vocab);
                    var dLogits = new Tensor<float>(data, new[] { rows, vocab });
                    MatMulInto(dLogits, hc, w);
                    CrossEntropyGradRowsFloat(data, biasData, ids, start, rows, vocab, scale, rowLoss);
                    var dhc = DLogitsTimesWeightTransposed(data, w, rows, d, vocab);
                    Array.Copy(dhc.GetDataArray(), 0, dHiddenData, (long)start * d, (long)rows * d);
                    if (computeReadoutGrads)
                    {
                        if (dWeight is null)
                        {
                            dWeight = PrefaultedTensor(d, vocab);
                            MatMulInto(dWeight, TensorTranspose(hc), dLogits);
                        }
                        else
                        {
                            dwcScratch ??= PrefaultedTensor(d, vocab);
                            MatMulInto(dwcScratch, TensorTranspose(hc), dLogits);
                            AddInPlaceFloat(dWeight.GetDataArray(), dwcScratch.GetDataArray());
                        }
                        ColumnSumsFloat(data, dBias!, rows, vocab);
                    }
                    ReturnFusedCeWorkspace(data);
                }
                dHidden = new Tensor<float>(dHiddenData, new[] { n, d });
            }
        }

        double total = 0.0;
        for (int r = 0; r < n; r++) total += rowLoss[r];
        var output = new Tensor<T>(new[] { 1 });
        output.GetDataArray()[0] = MathHelper.GetNumericOperations<T>().FromDouble(total / n);

        var precomputed = new FusedCePrecomputedGradients
        {
            Hidden = dHidden,
            Weight = dWeight,
            Bias = dBias is not null ? new Tensor<float>(dBias, new[] { vocab }) : null,
        };
        var tapeInputs = computeReadoutGrads ? new[] { hidden, weight, bias } : new[] { hidden };
        DifferentiableOps.RecordIfActive<T>(
            "FusedLinearCrossEntropy", output,
            tapeInputs,
            FusedLinearCrossEntropyPrecomputedBackward<T>,
            savedState: new object[]
            {
                (int[])ids.Clone(), vocab, weight, bias, computeReadoutGrads,
                new ReadoutMutationGuard<T>(weight, bias), precomputed
            });
        return output;
    }

    internal sealed class ReadoutMutationGuard<T>
    {
        private Tensor<T> _weight;
        private int _weightVersion;
        private int _weightStorageVersion;
        private Tensor<T> _bias;
        private int _biasVersion;
        private int _biasStorageVersion;
        // Set when a rebase found the readout already mutated: the tape then hands Validate the snapshot, whose counters
        // say nothing about the original tensor and could coincide with the recorded ones.
        private bool _weightMutated;
        private bool _biasMutated;

        internal ReadoutMutationGuard(Tensor<T> weight, Tensor<T> bias)
        {
            _weight = weight;
            _weightVersion = weight.Version;
            _weightStorageVersion = weight.StorageMutationVersion;
            _bias = bias;
            _biasVersion = bias.Version;
            _biasStorageVersion = bias.StorageMutationVersion;
        }

        internal void Refresh(Tensor<T> weight, Tensor<T> bias)
        {
            _weight = weight;
            _weightVersion = weight.Version;
            _weightStorageVersion = weight.StorageMutationVersion;
            _bias = bias;
            _biasVersion = bias.Version;
            _biasStorageVersion = bias.StorageMutationVersion;
            _weightMutated = false;
            _biasMutated = false;
        }

        /// <summary>
        /// Repoints the guard at <paramref name="to"/>, the tape's snapshot of <paramref name="from"/> before an in-place
        /// write, only while <paramref name="from"/> still holds the recorded value. One already mutated since recording
        /// is marked mutated, so the backward reports that mutation whatever counters the snapshot carries.
        /// </summary>
        internal void Rebase(Tensor<T> from, Tensor<T> to)
        {
            if (ReferenceEquals(_weight, from))
            {
                if (from.Version == _weightVersion && from.StorageMutationVersion == _weightStorageVersion)
                {
                    _weight = to;
                    _weightVersion = to.Version;
                    _weightStorageVersion = to.StorageMutationVersion;
                }
                else _weightMutated = true;
            }
            if (ReferenceEquals(_bias, from))
            {
                if (from.Version == _biasVersion && from.StorageMutationVersion == _biasStorageVersion)
                {
                    _bias = to;
                    _biasVersion = to.Version;
                    _biasStorageVersion = to.StorageMutationVersion;
                }
                else _biasMutated = true;
            }
        }

        internal void Validate(Tensor<T> weight, Tensor<T> bias)
        {
            if (InferenceModeFlag.IsActive) return;
            if (_weightMutated || weight.Version != _weightVersion || weight.StorageMutationVersion != _weightStorageVersion)
            {
                throw new InvalidOperationException(
                    "Tensor input 1 of operation 'FusedLinearCrossEntropy' was mutated in-place after being recorded on the gradient tape. " +
                    "This produces incorrect gradients. Use non-in-place operations, or ensure in-place mutations happen before recording.");
            }
            if (_biasMutated || bias.Version != _biasVersion || bias.StorageMutationVersion != _biasStorageVersion)
            {
                throw new InvalidOperationException(
                    "Tensor input 2 of operation 'FusedLinearCrossEntropy' was mutated in-place after being recorded on the gradient tape. " +
                    "This produces incorrect gradients. Use non-in-place operations, or ensure in-place mutations happen before recording.");
            }
        }
    }

    private static void ValidateDetachedReadoutVersions<T>(Tensor<T> weight, Tensor<T> bias, object[] savedState)
    {
        if (InferenceModeFlag.IsActive || savedState.Length < 6) return;
        if (savedState[5] is ReadoutMutationGuard<T> guard)
            guard.Validate(weight, bias);
    }

    /// <summary>
    /// Backward of the float training forward: the precomputed gradients times the upstream scalar. Falls back to the
    /// recomputing backward if they were already handed out.
    /// </summary>
    internal static void FusedLinearCrossEntropyPrecomputedBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        bool computeReadout = (bool)savedState[4];
        var weight = computeReadout && inputs.Length > 1 ? inputs[1] : (Tensor<T>)savedState[2];
        var bias = computeReadout && inputs.Length > 2 ? inputs[2] : (Tensor<T>)savedState[3];
        ValidateDetachedReadoutVersions(weight, bias, savedState);
        var pre = (FusedCePrecomputedGradients)savedState[6];
        var dHidden = System.Threading.Interlocked.Exchange(ref pre.Hidden, null);
        var dWeight = System.Threading.Interlocked.Exchange(ref pre.Weight, null);
        var dBias = System.Threading.Interlocked.Exchange(ref pre.Bias, null);
        if (dHidden is null || (computeReadout && (dWeight is null || dBias is null)))
        {
            FusedLinearCrossEntropyIndexBackward(gradOutput, inputs, output, savedState, engine, grads);
            return;
        }

        float g = (float)MathHelper.GetNumericOperations<T>().ToDouble(gradOutput.GetDataArray()[0]);
        if (g != 1f)
        {
            ScaleInPlaceFloat(dHidden.GetDataArray(), g);
            if (computeReadout)
            {
                ScaleInPlaceFloat(dWeight!.GetDataArray(), g);
                ScaleInPlaceFloat(dBias!.GetDataArray(), g);
            }
        }
        if (DifferentiableOps.IsGradientRequired(inputs[0]))
            DifferentiableOps.AccumulateGrad(grads, inputs[0], Reinterpret<Tensor<T>>(dHidden), engine);
        if (computeReadout && inputs.Length > 1 && DifferentiableOps.IsGradientRequired(inputs[1]))
            DifferentiableOps.AccumulateGrad(grads, inputs[1], Reinterpret<Tensor<T>>(dWeight!), engine);
        if (computeReadout && inputs.Length > 2 && DifferentiableOps.IsGradientRequired(inputs[2]))
            DifferentiableOps.AccumulateGrad(grads, inputs[2], Reinterpret<Tensor<T>>(dBias!), engine);
    }

    // Rows [start, start + rows) of a chunk: rowLoss[start + i] is row i's CE, and row i becomes its scaled gradient.
    // Row-parallel; each row is computed by exactly one worker.
    private static unsafe void CrossEntropyGradRowsFloat(
        float[] logits, float[] bias, int[] ids, int start, int rows, int vocab, float scale, double[] rowLoss)
    {
        int chunks = RowChunks(rows, vocab);
        int rowsPerChunk = (rows + chunks - 1) / chunks;
        void Chunk(int c)
        {
            int lo = c * rowsPerChunk, hi = Math.Min(lo + rowsPerChunk, rows);
            fixed (float* pl = logits)
            fixed (float* pb = bias)
            {
                for (int r = lo; r < hi; r++)
                    rowLoss[start + r] = Simd.SimdKernels.CrossEntropyGradRowUnsafe(
                        pl + (long)r * vocab, pb, vocab, ids[start + r], scale);
            }
        }
        if (chunks == 1) Chunk(0);
        else PersistentParallelExecutor.Instance.Execute(chunks, Chunk);
    }

    /// <summary>Smallest K slice of the split-K dHidden product.</summary>
    internal const int FusedCeSplitKSlice = 1024;

    /// <summary>Most K slices of the split-K dHidden product: a constant, so the plan never follows the thread count.</summary>
    internal const int FusedCeSplitKMaxSlices = 48;

    // Split-K partials of dLogits·Wᵀ and the per-slice transposed W panels, kept between steps like the dLogits workspace.
    private static float[]? s_fusedCeSplitKWorkspace;
    private static float[]? s_fusedCeSplitKPanels;

    /// <summary>
    /// dHidden[rows, d] = dLogits[rows, vocab] · W[d, vocab]ᵀ, split over the vocabulary.
    /// </summary>
    /// <remarks>
    /// K (the vocabulary) is ~100-200x the output's sides. The packed transposed-B GEMM had a handful of output tiles
    /// to share between the workers and ran 512 x 256 x 49152 at ~172 GFLOP/s (71-81 ms) on a 3990X, and splitting
    /// the output rows instead re-streams all of W per row block. Split over K, each slice owns the whole (small)
    /// output and every input element is read once: a slice transposes its columns of W into a contiguous
    /// [slice, d] panel and runs the single-threaded untransposed GEMM into its own partial; the partials are summed
    /// in slice order. The slice count comes from the shape alone, so the result does not depend on the thread count.
    /// </remarks>
    private static unsafe Tensor<float> DLogitsTimesWeightTransposed(float[] dLogits, Tensor<float> w, int rows, int d, int vocab)
    {
        var wData = w.IsContiguous ? w.GetDataArray() : w.Contiguous().GetDataArray();
        var result = new float[(long)rows * d];
        int slices = Math.Min(FusedCeSplitKMaxSlices, vocab / FusedCeSplitKSlice);
        if (slices < 2)
        {
            Helpers.BlasProvider.GemmEx(rows, d, vocab, dLogits, 0, vocab, false, wData, 0, vocab, true, result, 0, d);
            return new Tensor<float>(result, new[] { rows, d });
        }

        long mn = (long)rows * d;
        int kPer = (vocab + slices - 1) / slices;
        kPer = (kPer + 7) & ~7;
        slices = (vocab + kPer - 1) / kPer;
        var partial = System.Threading.Interlocked.Exchange(ref s_fusedCeSplitKWorkspace, null);
        if (partial is null || partial.LongLength != mn * slices) partial = new float[mn * slices];
        long panelElems = (long)kPer * d;
        var panels = System.Threading.Interlocked.Exchange(ref s_fusedCeSplitKPanels, null);
        if (panels is null || panels.LongLength != panelElems * slices) panels = new float[panelElems * slices];
        void Slice(int sl)
        {
            int k0 = sl * kPer, kl = Math.Min(kPer, vocab - k0);
            fixed (float* pw = wData)
            fixed (float* pp = panels)
            {
                // panel[kk, j] = W[j, k0 + kk]
                float* panel = pp + sl * panelElems;
                for (int j = 0; j < d; j++)
                {
                    float* src = pw + (long)j * vocab + k0;
                    for (int kk = 0; kk < kl; kk++) panel[(long)kk * d + j] = src[kk];
                }
            }
            BlasManaged.BlasManaged.Gemm<float>(
                new ReadOnlySpan<float>(dLogits, k0, dLogits.Length - k0), vocab, false,
                new ReadOnlySpan<float>(panels, (int)(sl * panelElems), (int)panelElems), d, false,
                new Span<float>(partial, (int)(sl * mn), (int)mn), d,
                rows, d, kl, new BlasManaged.BlasOptions<float> { BetaZero = true, NumThreads = -1 });
        }
        PersistentParallelExecutor.Instance.Execute(slices, Slice);
        SumPartialsFloat(partial, result, slices, mn);
        System.Threading.Volatile.Write(ref s_fusedCeSplitKWorkspace, partial);
        System.Threading.Volatile.Write(ref s_fusedCeSplitKPanels, panels);
        return new Tensor<float>(result, new[] { rows, d });
    }

    // result[i] = sum_s partial[s * mn + i], slices in order; parallel over contiguous index ranges.
    private static unsafe void SumPartialsFloat(float[] partial, float[] result, int slices, long mn)
    {
        int chunks = (int)Math.Max(1, Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, mn / 4096));
        long per = (mn + chunks - 1) / chunks;
        per = (per + 7) & ~7L;
        void Chunk(int c)
        {
            long start = c * per, len = Math.Min(per, mn - start);
            if (len <= 0) return;
            fixed (float* pp = partial)
            fixed (float* pr = result)
            {
                float* dst = pr + start;
                Buffer.MemoryCopy(pp + start, dst, len * sizeof(float), len * sizeof(float));
                for (int sl = 1; sl < slices; sl++)
                    Simd.SimdKernels.VectorAddUnsafe(dst, pp + sl * mn + start, dst, (int)len);
            }
        }
        if (chunks == 1) Chunk(0);
        else PersistentParallelExecutor.Instance.Execute(chunks, Chunk);
    }

    // A fresh [rows, cols] float tensor whose pages are already faulted in, for a parallel kernel to write. The gradient
    // escapes, so it cannot come from the workspace; faulting it in from one thread (one store per 4 KiB page) costs a
    // few ms per 100 MB, where 64 GEMM workers faulting the same pages serialize on the page tables: h^T·dLogits into
    // a fresh 256 x 49152 output took 60-74 ms in the training step on a 3990X.
    private static Tensor<float> PrefaultedTensor(int rows, int cols)
    {
        var data = new float[(long)rows * cols];
        for (long i = 0; i < data.LongLength; i += 1024) data[i] = 0f;
        return new Tensor<float>(data, new[] { rows, cols });
    }

    private static unsafe void ScaleInPlaceFloat(float[] data, float g)
    {
        fixed (float* p = data) Simd.SimdKernels.MultiplyScalarUnsafe(p, g, p, data.Length);
    }

    private static unsafe void AddInPlaceFloat(float[] acc, float[] add)
    {
        fixed (float* pa = acc)
        fixed (float* pb = add)
            Simd.SimdKernels.VectorAddUnsafe(pa, pb, pa, acc.Length);
    }

    [MethodImpl(Hot)]
    private static void SoftmaxMinusIndexDouble(
        double[] logits, int[] ids, double[] dLogits, int n, int vocab, double g)
    {
        double scale = g / n;
        for (int r = 0; r < n; r++)
        {
            int off = r * vocab;
            double max = logits[off];
            for (int v = 1; v < vocab; v++) if (logits[off + v] > max) max = logits[off + v];
            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(logits[off + v] - max);
            double inv = 1.0 / sumExp;
            int id = ids[r];
            for (int v = 0; v < vocab; v++)
            {
                double sm = Math.Exp(logits[off + v] - max) * inv;
                dLogits[off + v] = (sm - (v == id ? 1.0 : 0.0)) * scale;
            }
        }
    }

    // float paths. The generic path widens every element through INumericOperations and calls the scalar double
    // Math.Exp - twice per element in the backward - on one thread; for an LM head (N rows x V ~ 5e4) that serial
    // softmax dominated the op while the GEMMs around it ran on every core. Rows are independent, so they are split
    // into contiguous chunks across the persistent pool (deterministic: each row is computed by exactly one worker,
    // and the per-row losses are reduced in row order) and each row uses the SIMD row kernels.
    private const int FloatRowParallelMinElements = 32768;

    private static int RowChunks(int n, int vocab)
        => n >= 2 && (long)n * vocab >= FloatRowParallelMinElements
            ? Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, n)
            : 1;

    // Index-target mean CE for float logits: row r contributes max + log(sum exp(logits[r] - max)) - logits[r, id_r].
    // The exponentials are float SIMD; their sum, the log and the per-row losses are double (a float sum loses the
    // small terms of a confident row, see ExpShiftedSumUnsafe), and the row losses are summed in row order.
    private static unsafe double FusedCeLossIndexFloat(float[] logits, int[] ids, int n, int vocab)
    {
        var rowLoss = new double[n];
        int chunks = RowChunks(n, vocab);
        int rowsPerChunk = (n + chunks - 1) / chunks;
        void Chunk(int c)
        {
            int start = c * rowsPerChunk, end = Math.Min(start + rowsPerChunk, n);
            if (start >= end) return;
            var scratch = System.Buffers.ArrayPool<float>.Shared.Rent(vocab);
            try
            {
                fixed (float* pl = logits)
                fixed (float* ps = scratch)
                {
                    for (int r = start; r < end; r++)
                    {
                        float* row = pl + (long)r * vocab;
                        float max = Simd.SimdKernels.MaxUnsafe(row, vocab);
                        double sumExp = Simd.SimdKernels.ExpShiftedSumUnsafe(row, ps, max, vocab);
                        rowLoss[r] = max + Math.Log(sumExp) - row[ids[r]];
                    }
                }
            }
            finally { System.Buffers.ArrayPool<float>.Shared.Return(scratch); }
        }
        if (chunks == 1) Chunk(0);
        else PersistentParallelExecutor.Instance.Execute(chunks, Chunk);

        double total = 0.0;
        for (int r = 0; r < n; r++) total += rowLoss[r];
        return total / n;
    }

    // dlogits[r, v] = (softmax(logits[r])[v] - [v == id_r]) * g / N, row-parallel: float SIMD exponentials, their sum
    // in double (as in the loss), and one multiply by (g / N) / sum.
    private static unsafe void SoftmaxMinusIndexFloat(float[] logits, int[] ids, float[] dLogits, int n, int vocab, double g)
    {
        double scaleD = g / n;
        float scale = (float)scaleD;
        int chunks = RowChunks(n, vocab);
        int rowsPerChunk = (n + chunks - 1) / chunks;
        void Chunk(int c)
        {
            int start = c * rowsPerChunk, end = Math.Min(start + rowsPerChunk, n);
            fixed (float* pl = logits)
            fixed (float* pd = dLogits)
            {
                for (int r = start; r < end; r++)
                {
                    float* src = pl + (long)r * vocab, dst = pd + (long)r * vocab;
                    float max = Simd.SimdKernels.MaxUnsafe(src, vocab);
                    double sumExp = Simd.SimdKernels.ExpShiftedSumUnsafe(src, dst, max, vocab);
                    Simd.SimdKernels.MultiplyScalarUnsafe(dst, (float)(scaleD / sumExp), dst, vocab);
                    dst[ids[r]] -= scale;
                }
            }
        }
        if (chunks == 1) Chunk(0);
        else PersistentParallelExecutor.Instance.Execute(chunks, Chunk);
    }

    // logits[r, :] += bias, row-parallel, SIMD.
    private static unsafe void AddBiasRowsFloat(float[] logits, float[] bias, int n, int vocab)
    {
        int chunks = RowChunks(n, vocab);
        int rowsPerChunk = (n + chunks - 1) / chunks;
        void Chunk(int c)
        {
            int start = c * rowsPerChunk, end = Math.Min(start + rowsPerChunk, n);
            fixed (float* pl = logits)
            fixed (float* pb = bias)
            {
                for (int r = start; r < end; r++)
                {
                    float* row = pl + (long)r * vocab;
                    Simd.SimdKernels.VectorAddUnsafe(row, pb, row, vocab);
                }
            }
        }
        if (chunks == 1) Chunk(0);
        else PersistentParallelExecutor.Instance.Execute(chunks, Chunk);
    }

    // result[v] = sum_r dLogits[r, v]. Parallel over column blocks: each worker owns a column range and adds the rows
    // in row order, so every element is summed in the same order as the serial loop (bit-identical, deterministic).
    private static unsafe void ColumnSumsFloat(float[] dLogits, float[] result, int n, int vocab)
    {
        int chunks = RowChunks(n, vocab) == 1 ? 1 : Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, vocab / 256));
        int colsPerChunk = (vocab + chunks - 1) / chunks;
        void Chunk(int c)
        {
            int start = c * colsPerChunk, len = Math.Min(start + colsPerChunk, vocab) - start;
            if (len <= 0) return;
            fixed (float* pd = dLogits)
            fixed (float* pr = result)
            {
                float* acc = pr + start;
                for (int r = 0; r < n; r++)
                    Simd.SimdKernels.VectorAddUnsafe(acc, pd + (long)r * vocab + start, acc, len);
            }
        }
        if (chunks == 1) Chunk(0);
        else PersistentParallelExecutor.Instance.Execute(chunks, Chunk);
    }

    private static void SoftmaxMinusIndexGeneric<T>(
        T[] logits, int[] ids, T[] dLogits, int n, int vocab, double g)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double scale = g / n;
        for (int r = 0; r < n; r++)
        {
            int off = r * vocab;
            double max = ops.ToDouble(logits[off]);
            for (int v = 1; v < vocab; v++) { double lv = ops.ToDouble(logits[off + v]); if (lv > max) max = lv; }
            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(ops.ToDouble(logits[off + v]) - max);
            double inv = 1.0 / sumExp;
            int id = ids[r];
            for (int v = 0; v < vocab; v++)
            {
                double sm = Math.Exp(ops.ToDouble(logits[off + v]) - max) * inv;
                dLogits[off + v] = ops.FromDouble((sm - (v == id ? 1.0 : 0.0)) * scale);
            }
        }
    }

    internal static void FusedLinearCrossEntropyBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var hidden = inputs[0];
        var weight = inputs[1];
        var bias = inputs[2];
        var target = inputs[3];

        bool needHidden = DifferentiableOps.IsGradientRequired(hidden);
        bool needWeight = DifferentiableOps.IsGradientRequired(weight);
        bool needBias = DifferentiableOps.IsGradientRequired(bias);
        if (!needHidden && !needWeight && !needBias)
            return;

        if (engine is DirectGpuTensorEngine gpu && gpu.TryFusedLinearCrossEntropyBackwardOnDevice(
                gradOutput, hidden, weight, bias, null, target, needHidden, needWeight, needBias,
                out var deviceHidden, out var deviceWeight, out var deviceBias))
        {
            if (needHidden) DifferentiableOps.AccumulateGrad(grads, hidden, deviceHidden, engine);
            if (needWeight) DifferentiableOps.AccumulateGrad(grads, weight, deviceWeight, engine);
            if (needBias) DifferentiableOps.AccumulateGrad(grads, bias, deviceBias, engine);
            return;
        }

        int n = hidden.Shape[0];
        int vocab = weight.Shape[1];

        // Upstream scalar grad (∂L_outer/∂loss). dlogits = g·(softmax(logits) - target)/N.
        var ops = MathHelper.GetNumericOperations<T>();
        double g = ops.ToDouble(gradOutput.GetDataArray()[0]);

        // Recompute logits → softmax → dlogits, all off-tape.
        var dLogits = new Tensor<T>(new[] { n, vocab });
        using (new NoGradScope<T>())
        {
            var logits = engine.TensorMatMul(hidden, weight);
            // Materialize ONCE — see the int-id forward: GetDataArray() returns a fresh detached copy
            // for a GPU-resident tensor, so adding the bias to one copy and reading another drops it.
            var logitData = logits.GetDataArray();
            AddBiasRowsInto(logitData, bias, n, vocab);
            if (typeof(T) == typeof(double))
            {
                SoftmaxMinusTargetDouble(
                    Reinterpret<double[]>(logitData), Reinterpret<double[]>(target.GetDataArray()),
                    Reinterpret<double[]>(dLogits.GetDataArray()), n, vocab, g);
            }
            else
            {
                SoftmaxMinusTargetGeneric<T>(
                    logitData, target.GetDataArray(), dLogits.GetDataArray(), n, vocab, g);
            }

            if (needHidden)
            {
                // dHidden = dLogits·weightᵀ  [N,d]  (weight is [d,vocab] → weightᵀ [vocab,d]).
                var dHidden = engine.TensorMatMulTransposed(dLogits, weight);
                DifferentiableOps.AccumulateGrad(grads, hidden, dHidden, engine);
            }
            if (needWeight)
            {
                // dWeight = hiddenᵀ·dLogits  [d,vocab]  (hiddenᵀ is [d,N], tiny — d,N small).
                var dWeight = engine.TensorMatMul(engine.TensorTranspose(hidden), dLogits);
                DifferentiableOps.AccumulateGrad(grads, weight, dWeight, engine);
            }
            if (needBias)
            {
                var dBias = ColumnSums(dLogits, n, vocab);
                DifferentiableOps.AccumulateGrad(grads, bias, dBias, engine);
            }
            // target is supervision — no gradient flows to it.
        }
    }

    [MethodImpl(Hot)]
    private static void SoftmaxMinusTargetDouble(
        double[] logits, double[] target, double[] dLogits, int n, int vocab, double g)
    {
        double scale = g / n;
        for (int r = 0; r < n; r++)
        {
            int off = r * vocab;
            double max = logits[off];
            for (int v = 1; v < vocab; v++) if (logits[off + v] > max) max = logits[off + v];
            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(logits[off + v] - max);
            double inv = 1.0 / sumExp;
            for (int v = 0; v < vocab; v++)
            {
                double sm = Math.Exp(logits[off + v] - max) * inv;
                dLogits[off + v] = (sm - target[off + v]) * scale;
            }
        }
    }

    private static void SoftmaxMinusTargetGeneric<T>(
        T[] logits, T[] target, T[] dLogits, int n, int vocab, double g)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double scale = g / n;
        for (int r = 0; r < n; r++)
        {
            int off = r * vocab;
            double max = ops.ToDouble(logits[off]);
            for (int v = 1; v < vocab; v++) { double lv = ops.ToDouble(logits[off + v]); if (lv > max) max = lv; }
            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(ops.ToDouble(logits[off + v]) - max);
            double inv = 1.0 / sumExp;
            for (int v = 0; v < vocab; v++)
            {
                double sm = Math.Exp(ops.ToDouble(logits[off + v]) - max) * inv;
                dLogits[off + v] = ops.FromDouble((sm - ops.ToDouble(target[off + v])) * scale);
            }
        }
    }

    private static Tensor<T> ColumnSums<T>(Tensor<T> dLogits, int n, int vocab)
    {
        var result = new Tensor<T>(new[] { vocab });
        var dl = dLogits.GetDataArray();
        var res = result.GetDataArray();
        if (typeof(T) == typeof(float))
        {
            ColumnSumsFloat(Reinterpret<float[]>(dl), Reinterpret<float[]>(res), n, vocab);
        }
        else if (typeof(T) == typeof(double))
        {
            var d = Reinterpret<double[]>(dl); var rr = Reinterpret<double[]>(res);
            for (int r = 0; r < n; r++)
            {
                int off = r * vocab;
                for (int v = 0; v < vocab; v++) rr[v] += d[off + v];
            }
        }
        else
        {
            var ops = MathHelper.GetNumericOperations<T>();
            for (int r = 0; r < n; r++)
            {
                int off = r * vocab;
                for (int v = 0; v < vocab; v++) res[v] = ops.Add(res[v], dl[off + v]);
            }
        }
        return result;
    }
}

/// <summary>
/// <see cref="IEngine"/> extension methods for the readout-detached <see cref="CpuEngine.FusedLinearCrossEntropyWithLogits{T}(Tensor{T}, Tensor{T}, Tensor{T}, Tensor{int}, bool)"/>
/// overload, preserving source and binary compatibility for external <see cref="IEngine"/> implementations.
/// </summary>
public static class FusedLinearCrossEntropyEngineExtensions
{
    /// <inheritdoc cref="CpuEngine.FusedLinearCrossEntropyWithLogits{T}(Tensor{T}, Tensor{T}, Tensor{T}, Tensor{int}, bool)"/>
    public static Tensor<T> FusedLinearCrossEntropyWithLogits<T>(
        this IEngine engine, Tensor<T> hidden, Tensor<T> weight, Tensor<T> bias, Tensor<int> targetIds,
        bool computeReadoutGrads)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (engine is CpuEngine cpuEngine)
            return cpuEngine.FusedLinearCrossEntropyWithLogits(hidden, weight, bias, targetIds, computeReadoutGrads);
        if (computeReadoutGrads)
            return engine.FusedLinearCrossEntropyWithLogits(hidden, weight, bias, targetIds);
        return new CpuEngine().FusedLinearCrossEntropyWithLogits(hidden, weight, bias, targetIds, computeReadoutGrads: false);
    }
}
