using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    /// <remarks>Dense host float logits take ONE recorded node: a forward that computes the loss in a single pass over
    /// the logits and a backward that writes their gradient in one more. The composed primitives were about ten recorded
    /// ops, each with its own dispatch, output and gradient buffer; on a batch-64, 10-class MLP training step they cost
    /// ~190 us against ~5 us for this node. Everything else (other element types, strided or device tensors) keeps the
    /// composed primitives.</remarks>
    public virtual Tensor<T> TensorSoftmaxCrossEntropy<T>(Tensor<T> logits, Tensor<T> target)
    {
        ValidateSoftmaxCrossEntropy(logits, target);
        if (typeof(T) != typeof(float) || this is DirectGpuTensorEngine
            || !IsDenseHost(logits) || !IsDenseHost(target) || logits.Length == 0)
            return SoftmaxCrossEntropyDecomposed(logits, target);

        int cols = logits._shape[logits.Rank - 1];
        int rows = logits.Length / cols;
        var node = new SoftmaxCrossEntropyState(rows, cols);
        var savedState = new object[] { node };
        if (GraphMode.IsActive)
        {
            if (GraphMode.Current is not { } scope || !scope.BindEngine(this))
                return SoftmaxCrossEntropyDecomposed(logits, target);
            var capturedLogits = logits;
            var capturedTarget = target;
            return scope.RecordVariadic(LazyNodeType.Custom, "SoftmaxCrossEntropy", new[] { logits, target },
                Array.Empty<int>(),
                (eng, output) => WriteScalar(output, SoftmaxCrossEntropyForward(capturedLogits, capturedTarget, node)),
                SoftmaxCrossEntropyBackward<T>, savedState);
        }

        var loss = new Tensor<T>(Array.Empty<int>());
        WriteScalar(loss, SoftmaxCrossEntropyForward(logits, target, node));
        DifferentiableOps.RecordIfActive("SoftmaxCrossEntropy", loss, new[] { logits, target },
            SoftmaxCrossEntropyBackward<T>, savedState);
        return loss;
    }

    // The forward's softmax and per-row target sum, read by the backward (so it needs no exp); reused by every replay.
    private sealed class SoftmaxCrossEntropyState
    {
        internal SoftmaxCrossEntropyState(int rows, int cols)
        {
            Probabilities = new float[rows * cols];
            TargetSum = new float[rows];
        }

        internal float[] Probabilities { get; }
        internal float[] TargetSum { get; }
        internal float Count { get; set; } = 1f;
    }

    private static bool IsDenseHost<T>(Tensor<T> tensor) =>
        tensor.IsContiguous && tensor._storageOffset == 0 && !tensor.HasPendingGpuData && tensor.TryGetGpuBuffer() is null;

    private static float[] HostFloats<T>(Tensor<T> tensor, string role) =>
        tensor.GetCpuBackingForStridedRead(out _) as float[]
        ?? throw new InvalidOperationException($"Softmax cross-entropy: the {role} has no host float buffer.");

    private static void WriteScalar<T>(Tensor<T> output, float value)
    {
        var dst = output.GetCpuBackingForContiguousWrite(out int offset) as float[]
            ?? throw new InvalidOperationException("Softmax cross-entropy: the loss has no host float buffer.");
        dst[offset] = value;
        output.IncrementVersion();
    }

    /// <summary>
    /// loss = -(sum over rows of sum_c t*log_softmax(z)) / max(supervised rows, 1), where a row is supervised when its
    /// target has mass -- the composed objective exactly, accumulated in double.
    /// </summary>
    private static float SoftmaxCrossEntropyForward<T>(Tensor<T> logits, Tensor<T> target, SoftmaxCrossEntropyState node)
    {
        var z = HostFloats(logits, "logits");
        var t = HostFloats(target, "target");
        var p = node.Probabilities;
        int rows = node.TargetSum.Length;
        int cols = logits.Length / rows;
        double total = 0.0, supervised = 0.0;
        for (int r = 0; r < rows; r++)
        {
            int o = r * cols;
            float max = z[o];
            for (int c = 1; c < cols; c++) max = Math.Max(max, z[o + c]);
            double sumExp = 0.0;
            for (int c = 0; c < cols; c++)
            {
                double e = Math.Exp(z[o + c] - max);
                p[o + c] = (float)e;
                sumExp += e;
            }
            double lse = max + Math.Log(sumExp);
            float invSum = (float)(1.0 / sumExp);
            for (int c = 0; c < cols; c++) p[o + c] *= invSum;
            double dot = 0.0, targetSum = 0.0, mass = 0.0;
            for (int c = 0; c < cols; c++)
            {
                float tc = t[o + c];
                dot += (double)tc * z[o + c];
                targetSum += tc;
                mass += Math.Abs(tc);
            }
            total += dot - targetSum * lse;
            // Sign of the mass, as the composed count: NaN propagates, zero mass counts zero.
            supervised += double.IsNaN(mass) ? double.NaN : mass > 0.0 ? 1.0 : 0.0;
            node.TargetSum[r] = (float)targetSum;
        }
        double count = double.IsNaN(supervised) ? double.NaN : Math.Max(supervised, 1.0);
        node.Count = (float)count;
        return (float)(-total / count);
    }

    /// <summary>d loss / d z = (g / count) * (target_sum_row * softmax(z) - t). The target is supervision: no gradient.</summary>
    private static void SoftmaxCrossEntropyBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var node = savedState[0] as SoftmaxCrossEntropyState
            ?? throw new InvalidOperationException("Softmax cross-entropy backward: the forward's row statistics are missing.");
        var logits = inputs[0];
        var p = node.Probabilities;
        var t = HostFloats(inputs[1], "target");
        float g = HostFloats(gradOutput, "loss gradient")[0];
        float scale = g / node.Count;
        int rows = node.TargetSum.Length;
        int cols = logits.Length / rows;
        var grad = TensorAllocator.RentUninitialized<T>(logits._shape);
        var dz = grad.GetCpuBackingForContiguousWrite(out int dzOffset) as float[]
            ?? throw new InvalidOperationException("Softmax cross-entropy backward: the gradient has no host float buffer.");
        for (int r = 0; r < rows; r++)
        {
            int o = r * cols;
            float rowScale = scale * node.TargetSum[r];
            for (int c = 0; c < cols; c++)
                dz[dzOffset + o + c] = rowScale * p[o + c] - scale * t[o + c];
        }
        grad.IncrementVersion();
        DifferentiableOps.AccumulateGrad(grads, logits, grad, engine);
    }

    private protected static void ValidateSoftmaxCrossEntropy<T>(Tensor<T> logits, Tensor<T> target)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (logits.Rank < 2)
            throw new ArgumentException($"Softmax cross-entropy expects logits of rank 2 or more; got rank {logits.Rank}.", nameof(logits));
        if (!logits._shape.SequenceEqual(target._shape))
            throw new ArgumentException(
                $"target shape [{string.Join(", ", target._shape)}] must equal logits shape [{string.Join(", ", logits._shape)}].",
                nameof(target));
    }

    /// <summary>
    /// The objective as recorded primitives -- log-softmax, the target-weighted sum, and a detached supervised-row
    /// count -- so every engine and element type differentiates it with the ops it already has.
    /// </summary>
    private Tensor<T> SoftmaxCrossEntropyDecomposed<T>(Tensor<T> logits, Tensor<T> target)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int classAxis = logits.Rank - 1;
        var batchAxes = Enumerable.Range(0, classAxis).ToArray();
        var logSoftmax = TensorLogSoftmax(logits, classAxis);
        var perSample = ReduceSum(TensorMultiply(target, logSoftmax), new[] { classAxis }, keepDims: false);
        var total = ReduceSum(perSample, batchAxes, keepDims: false);
        // A row with target mass is supervised; the count depends only on the target, so it is detached.
        var supervised = ReduceSum(TensorSign(ReduceSum(TensorAbs(target), new[] { classAxis }, keepDims: false)),
            batchAxes, keepDims: false);
        var safeCount = StopGradient(TensorClampMin(supervised, numOps.One));
        return TensorNegate(TensorDivide(total, safeCount));
    }
}
