using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>Backward passes of the model-level ops in <c>CpuEngine.TorchModelOps</c>.</summary>
internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// Saved state of <see cref="CrfForwardBackward"/>: the forward's inputs as doubles and its log-space forward
    /// variables, so the backward runs the backward recursion once and reads the marginals off both.
    /// </summary>
    internal sealed class CrfForwardState
    {
        internal int Batch, SeqLen, NumTags;
        internal bool HasStart, HasEnd;
        internal double[] Emissions = Array.Empty<double>();   // [batch, seqLen, numTags]
        internal double[] Transitions = Array.Empty<double>(); // [numTags, numTags], [from, to]
        internal double[] Start = Array.Empty<double>();       // [numTags]
        internal double[] End = Array.Empty<double>();         // [numTags]
        internal double[] Alpha = Array.Empty<double>();       // [batch, seqLen, numTags]
        internal double[] LogZ = Array.Empty<double>();        // [batch]
    }

    /// <summary>
    /// Linear-chain CRF log-partition backward (forward-backward): ∂logZ/∂emissions[t, j] = p(y_t = j),
    /// ∂logZ/∂transitions[i, j] = Σ_t p(y_{t-1} = i, y_t = j), ∂logZ/∂start[j] = p(y_0 = j) and
    /// ∂logZ/∂end[j] = p(y_{S-1} = j), each scaled by the batch row's upstream gradient.
    /// </summary>
    internal static void CrfForwardBackward(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var s = (CrfForwardState)savedState[0];
        int batch = s.Batch, seqLen = s.SeqLen, k = s.NumTags;
        var g = RoIAlignPlan.ToDoubles(gradOutput);
        var emitGrad = new double[batch * seqLen * k];
        var transGrad = new double[k * k];
        var startGrad = new double[k];
        var endGrad = new double[k];
        var beta = new double[seqLen * k];
        var terms = new double[k];
        for (int b = 0; b < batch; b++)
        {
            int baseOffset = b * seqLen * k;
            double logZ = s.LogZ[b], scale = g[b];
            for (int j = 0; j < k; j++) beta[(seqLen - 1) * k + j] = s.End[j];
            for (int t = seqLen - 2; t >= 0; t--)
            {
                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < k; j++)
                        terms[j] = s.Transitions[i * k + j] + s.Emissions[baseOffset + (t + 1) * k + j] + beta[(t + 1) * k + j];
                    beta[t * k + i] = LogSumExp(terms);
                }
            }

            for (int t = 0; t < seqLen; t++)
            {
                for (int j = 0; j < k; j++)
                {
                    double marginal = Math.Exp(s.Alpha[baseOffset + t * k + j] + beta[t * k + j] - logZ) * scale;
                    emitGrad[baseOffset + t * k + j] = marginal;
                    if (t == 0) startGrad[j] += marginal;
                    if (t == seqLen - 1) endGrad[j] += marginal;
                }
            }

            for (int t = 1; t < seqLen; t++)
            {
                for (int i = 0; i < k; i++)
                {
                    double from = s.Alpha[baseOffset + (t - 1) * k + i];
                    for (int j = 0; j < k; j++)
                        transGrad[i * k + j] += Math.Exp(from + s.Transitions[i * k + j]
                            + s.Emissions[baseOffset + t * k + j] + beta[t * k + j] - logZ) * scale;
                }
            }
        }

        DifferentiableOps.AccumulateGrad(grads, inputs[0], RoIAlignPlan.FromDoubles<T>(emitGrad, (int[])inputs[0]._shape.Clone()), engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], RoIAlignPlan.FromDoubles<T>(transGrad, (int[])inputs[1]._shape.Clone()), engine);
        int next = 2;
        if (s.HasStart)
        {
            DifferentiableOps.AccumulateGrad(grads, inputs[next], RoIAlignPlan.FromDoubles<T>(startGrad, (int[])inputs[next]._shape.Clone()), engine);
            next++;
        }
        if (s.HasEnd)
            DifferentiableOps.AccumulateGrad(grads, inputs[next], RoIAlignPlan.FromDoubles<T>(endGrad, (int[])inputs[next]._shape.Clone()), engine);
    }

    /// <summary>log Σ exp(values), shifted by the maximum so it neither overflows nor underflows.</summary>
    internal static double LogSumExp(double[] values)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < values.Length; i++) if (values[i] > max) max = values[i];
        if (double.IsNegativeInfinity(max) || double.IsNaN(max)) return max;
        double sum = 0;
        for (int i = 0; i < values.Length; i++) sum += Math.Exp(values[i] - max);
        return max + Math.Log(sum);
    }

    /// <summary>
    /// Straight-through backward restricted to a saved pass-through mask (1 where the forward kept the value in range,
    /// 0 where it clamped): dx = dy·mask, as <c>torch.fake_quantize_per_tensor_affine</c> computes it.
    /// </summary>
    internal static void MaskedPassThroughBackward(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var mask = (Tensor<T>)savedState[0];
        DifferentiableOps.AccumulateGrad(grads, inputs[0], engine.TensorMultiply(gradOutput, mask), engine);
    }

    /// <summary>
    /// Backward of a softmax over a subset of the last axis whose other entries are zero: dx = y·(dy - Σ dy·y) per row.
    /// The unselected entries have y = 0, so the full-row sum is the selected-entry sum and their gradient is zero.
    /// Built from engine ops, so it is itself differentiable.
    /// </summary>
    internal static void TopKSoftmaxBackward(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int last = output.Rank - 1;
        var rowDot = engine.ReduceSum(engine.TensorMultiply(gradOutput, output), new[] { last }, keepDims: true);
        var centered = engine.TensorSubtract(gradOutput, engine.TensorBroadcastTo(rowDot, (int[])output._shape.Clone()));
        DifferentiableOps.AccumulateGrad(grads, inputs[0], engine.TensorMultiply(output, centered), engine);
    }

    /// <summary>
    /// Sinusoidal time embedding backward: dt_b = Σ_i f_i·(dy[b, i]·cos(t_b·f_i) - dy[b, i + half]·sin(t_b·f_i)).
    /// </summary>
    internal static void SinusoidalTimeEmbeddingBackward(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var freq = (double[])savedState[0];
        int dim = (int)savedState[1];
        int half = freq.Length;
        var t = RoIAlignPlan.ToDoubles(inputs[0]);
        var g = RoIAlignPlan.ToDoubles(gradOutput);
        var dt = new double[t.Length];
        for (int b = 0; b < t.Length; b++)
        {
            double sum = 0;
            for (int i = 0; i < half; i++)
            {
                double angle = t[b] * freq[i];
                sum += freq[i] * (g[b * dim + i] * Math.Cos(angle) - g[b * dim + i + half] * Math.Sin(angle));
            }
            dt[b] = sum;
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], RoIAlignPlan.FromDoubles<T>(dt, (int[])inputs[0]._shape.Clone()), engine);
    }
}
