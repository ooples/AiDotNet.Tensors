using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorSoftmaxCrossEntropy<T>(Tensor<T> logits, Tensor<T> target)
    {
        ValidateSoftmaxCrossEntropy(logits, target);
        return SoftmaxCrossEntropyDecomposed(logits, target);
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
