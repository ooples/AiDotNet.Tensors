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

        var ids = (int[])(object)targetIds.GetDataArray();
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
            var compiledSavedState = new object[] { (int[])ids.Clone(), vocab };
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
                    var result = eng.FusedLinearCrossEntropyWithLogits(
                        capturedHidden, capturedWeight, capturedBias, capturedTargetIds);
                    DirectGpuTensorEngine.CopyResultInto(eng, result, output);
                },
                FusedLinearCrossEntropyIndexBackward<T>,
                compiledSavedState);
        }

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
            ? FusedCeLossIndexDouble((double[])(object)logitData, ids, n, vocab)
            : typeof(T) == typeof(float)
                ? FusedCeLossIndexFloat((float[])(object)logitData, ids, n, vocab)
                : FusedCeLossIndexGeneric<T>(logitData, ids, n, vocab);

        var output = new Tensor<T>(new[] { 1 }); // scalar loss (length-1)
        output.GetDataArray()[0] = MathHelper.GetNumericOperations<T>().FromDouble(lossValue);

        // target ids are supervision (no gradient): carry them in savedState rather than as a
        // differentiable tape input. Save a COPY so the backward is tied to the forward's labels —
        // targetIds is not a tape input, so mutating/recycling it after forward must not change the
        // gradient. Backward reconstructs dlogits = softmax - onehot(id).
        DifferentiableOps.RecordIfActive<T>(
            "FusedLinearCrossEntropy", output,
            new[] { hidden, weight, bias },
            FusedLinearCrossEntropyIndexBackward<T>,
            savedState: new object[] { (int[])ids.Clone(), vocab });

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
                (double[])(object)logitData, (double[])(object)target.GetDataArray(), n, vocab);
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
            AddBiasRowsFloat((float[])(object)lo, (float[])(object)bi, n, vocab);
        }
        else if (typeof(T) == typeof(double))
        {
            var l = (double[])(object)lo; var b = (double[])(object)bi;
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
        var hidden = inputs[0];
        var weight = inputs[1];
        var bias = inputs[2];
        var ids = (int[])savedState[0];
        int vocab = (int)savedState[1];

        if (engine is DirectGpuTensorEngine gpu && gpu.TryFusedLinearCrossEntropyBackwardOnDevice(
                gradOutput, hidden, weight, bias, ids, null, out var deviceHidden, out var deviceWeight, out var deviceBias))
        {
            DifferentiableOps.AccumulateGrad(grads, hidden, deviceHidden, engine);
            DifferentiableOps.AccumulateGrad(grads, weight, deviceWeight, engine);
            DifferentiableOps.AccumulateGrad(grads, bias, deviceBias, engine);
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
                    (double[])(object)logitData, ids,
                    (double[])(object)dLogits.GetDataArray(), n, vocab, g);
            }
            else if (typeof(T) == typeof(float))
            {
                SoftmaxMinusIndexFloat(
                    (float[])(object)logitData, ids, (float[])(object)dLogits.GetDataArray(), n, vocab, g);
            }
            else
            {
                SoftmaxMinusIndexGeneric<T>(logitData, ids, dLogits.GetDataArray(), n, vocab, g);
            }

            var dHidden = engine.TensorMatMulTransposed(dLogits, weight);
            var dWeight = engine.TensorMatMul(engine.TensorTranspose(hidden), dLogits);
            var dBias = ColumnSums(dLogits, n, vocab);

            DifferentiableOps.AccumulateGrad(grads, hidden, dHidden, engine);
            DifferentiableOps.AccumulateGrad(grads, weight, dWeight, engine);
            DifferentiableOps.AccumulateGrad(grads, bias, dBias, engine);
            // target ids are supervision — no gradient flows to them.
        }
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

    // Index-target mean CE for float logits: row r contributes -logSoftmax(logits[r])[id_r]. Each row's log-softmax
    // is evaluated in float by the SIMD row kernel; per-row losses are summed in double, in row order.
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
                        Simd.SimdKernels.LogSoftmaxRowUnsafe(pl + (long)r * vocab, ps, vocab);
                        rowLoss[r] = -scratch[ids[r]];
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

    // dlogits[r, v] = (softmax(logits[r])[v] - [v == id_r]) * g / N, row-parallel, SIMD softmax + scale.
    private static unsafe void SoftmaxMinusIndexFloat(float[] logits, int[] ids, float[] dLogits, int n, int vocab, double g)
    {
        float scale = (float)(g / n);
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
                    Simd.SimdKernels.SoftmaxRowUnsafe(src, dst, vocab);
                    Simd.SimdKernels.MultiplyScalarUnsafe(dst, scale, dst, vocab);
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

        if (engine is DirectGpuTensorEngine gpu && gpu.TryFusedLinearCrossEntropyBackwardOnDevice(
                gradOutput, hidden, weight, bias, null, target, out var deviceHidden, out var deviceWeight, out var deviceBias))
        {
            DifferentiableOps.AccumulateGrad(grads, hidden, deviceHidden, engine);
            DifferentiableOps.AccumulateGrad(grads, weight, deviceWeight, engine);
            DifferentiableOps.AccumulateGrad(grads, bias, deviceBias, engine);
            return;
        }

        int n = hidden.Shape[0];
        int d = hidden.Shape[1];
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
                    (double[])(object)logitData, (double[])(object)target.GetDataArray(),
                    (double[])(object)dLogits.GetDataArray(), n, vocab, g);
            }
            else
            {
                SoftmaxMinusTargetGeneric<T>(
                    logitData, target.GetDataArray(), dLogits.GetDataArray(), n, vocab, g);
            }

            // dHidden = dLogits·weightᵀ  [N,d]  (weight is [d,vocab] → weightᵀ [vocab,d]).
            var dHidden = engine.TensorMatMulTransposed(dLogits, weight);
            // dWeight = hiddenᵀ·dLogits  [d,vocab]  (hiddenᵀ is [d,N], tiny — d,N small).
            var dWeight = engine.TensorMatMul(engine.TensorTranspose(hidden), dLogits);
            var dBias = ColumnSums(dLogits, n, vocab);

            DifferentiableOps.AccumulateGrad(grads, hidden, dHidden, engine);
            DifferentiableOps.AccumulateGrad(grads, weight, dWeight, engine);
            DifferentiableOps.AccumulateGrad(grads, bias, dBias, engine);
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
            ColumnSumsFloat((float[])(object)dl, (float[])(object)res, n, vocab);
        }
        else if (typeof(T) == typeof(double))
        {
            var d = (double[])(object)dl; var rr = (double[])(object)res;
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
