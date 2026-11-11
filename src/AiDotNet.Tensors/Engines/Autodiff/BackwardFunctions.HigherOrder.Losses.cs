using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Create-graph (double-backward) paths of the normalization, loss, fused-linear, distance, detection and sequence
/// ops whose first-order backward runs a fused kernel or a host loop.
/// </summary>
/// <remarks>
/// <para>
/// Each <c>TrySecondOrder…</c> method returns false (doing nothing) when no graph is being created, so the op's fast
/// first-order path runs unchanged. Under <c>createGraph</c> it differentiates the op's forward written as a
/// composition of recorded engine ops (<see cref="TryCompositeBackwardFor"/>), so the gradient it accumulates is itself
/// a node of the graph — differentiable in the op's inputs and in the upstream gradient, to any order.
/// </para>
/// <para>
/// Only the inputs the first-order backward differentiates are sources: a target, a label or a saved running statistic
/// stays a constant here exactly as it is there.
/// </para>
/// </remarks>
internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// <see cref="TryCompositeBackward"/> restricted to the inputs at <paramref name="sourceIndices"/>; the composite's
    /// result is reshaped to the upstream gradient's shape when only the shapes differ (a [1] loss against a [] sum).
    /// </summary>
    internal static bool TryCompositeBackwardFor(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, int[] sourceIndices, Func<IEngine, Tensor<T>> composite)
    {
        if (!CreatesGraph) return false;
        var outer = GradientTape<T>.Current;
        var sources = new List<Tensor<T>>(sourceIndices.Length);
        foreach (int i in sourceIndices)
        {
            if (i >= inputs.Length) continue;
            var input = inputs[i];
            if (input is not null && !sources.Exists(s => ReferenceEquals(s, input))) sources.Add(input);
        }
        if (sources.Count == 0) return true;
        using (var tape = new GradientTape<T>())
        {
            var result = composite(engine);
            if (!ShapesEqual(result._shape, gradOutput._shape))
                result = engine.Reshape(result, (int[])gradOutput._shape.Clone());
            var g = tape.ComputeGradients(result, sources, createGraph: true,
                seedOverride: new[] { new KeyValuePair<Tensor<T>, Tensor<T>>(result, gradOutput) });
            // Hand the nested tape's entries to the outer tape before accumulating (see TryCompositeBackward).
            if (outer is not null) tape.TransferEntriesTo(outer);
            using (GradientTape<T>.RecordOn(outer))
            {
                foreach (var input in sources)
                {
                    if (g.TryGetValue(input, out var gi) && gi is not null)
                        DifferentiableOps.AccumulateGrad(grads, input, gi, engine);
                }
            }
        }
        return true;
    }

    private static readonly int[] FirstInput = { 0 };
    private static readonly int[] FirstTwoInputs = { 0, 1 };
    private static readonly int[] FirstThreeInputs = { 0, 1, 2 };

    private static T Num(double value) => MathHelper.GetNumericOperations<T>().FromDouble(value);

    /// <summary>A constant tensor (no graph) from host values.</summary>
    private static Tensor<T> Constant(double[] values, int[] shape) => RoIAlignPlan.FromDoubles<T>(values, shape);

    /// <summary><paramref name="t"/> broadcast to <paramref name="shape"/> by a recorded reshape and broadcast.</summary>
    internal static Tensor<T> Bcast(IEngine e, Tensor<T> t, int[] shape)
    {
        if (ShapesEqual(t._shape, shape)) return t;
        if (t.Rank < shape.Length)
        {
            var padded = new int[shape.Length];
            int offset = shape.Length - t.Rank;
            for (int i = 0; i < offset; i++) padded[i] = 1;
            Array.Copy(t._shape, 0, padded, offset, t.Rank);
            t = e.Reshape(t, padded);
        }
        return ShapesEqual(t._shape, shape) ? t : e.TensorBroadcastTo(t, shape);
    }

    /// <summary>Σ over every element, as a recorded reduction.</summary>
    private static Tensor<T> SumAll(IEngine e, Tensor<T> t) => e.ReduceSum(t, null, keepDims: false);

    /// <summary>A per-channel parameter laid out to broadcast over axis 1: [1, C, 1, …].</summary>
    private static int[] ChannelShape(int rank, int channels)
    {
        var shape = new int[rank];
        for (int i = 0; i < rank; i++) shape[i] = 1;
        shape[1] = channels;
        return shape;
    }

    // (x - mean) / sqrt(var + eps) over the given axes (population variance), as recorded ops.
    private static Tensor<T> NormalizeRecorded(IEngine e, Tensor<T> x, int[] axes, double eps)
    {
        var mean = e.ReduceMean(x, axes, keepDims: true);
        var centred = e.TensorSubtract(x, Bcast(e, mean, x._shape));
        var variance = e.ReduceMean(e.TensorMultiply(centred, centred), axes, keepDims: true);
        var std = e.TensorSqrt(e.TensorAddScalar(variance, Num(eps)));
        return e.TensorDivide(centred, Bcast(e, std, x._shape));
    }

    private static Tensor<T> AffineRecorded(IEngine e, Tensor<T> xhat, Tensor<T> gamma, Tensor<T>? beta, int[] paramShape)
    {
        var y = e.TensorMultiply(xhat, Bcast(e, e.Reshape(gamma, paramShape), xhat._shape));
        return beta is null ? y : e.TensorAdd(y, Bcast(e, e.Reshape(beta, paramShape), xhat._shape));
    }

    // ── normalization ─────────────────────────────────────────────────

    internal static bool TrySecondOrderBatchNorm(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double epsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            var x = inputs[0];
            var axes = new[] { 0 }.Concat(Enumerable.Range(2, Math.Max(0, x.Rank - 2))).ToArray();
            return AffineRecorded(e, NormalizeRecorded(e, x, axes, epsilon), inputs[1], inputs[2],
                ChannelShape(x.Rank, x._shape[1]));
        });

    internal static bool TrySecondOrderInstanceNorm(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double epsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            var x = inputs[0];
            var axes = Enumerable.Range(2, x.Rank - 2).ToArray();
            return AffineRecorded(e, NormalizeRecorded(e, x, axes, epsilon), inputs[1], inputs[2],
                ChannelShape(x.Rank, x._shape[1]));
        });

    internal static bool TrySecondOrderLayerNorm(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double epsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            var x = inputs[0];
            var gamma = inputs[1];
            int lead = x.Rank - gamma.Rank;
            var axes = Enumerable.Range(lead, gamma.Rank).ToArray();
            var paramShape = Enumerable.Repeat(1, lead).Concat(gamma._shape).ToArray();
            return AffineRecorded(e, NormalizeRecorded(e, x, axes, epsilon), gamma, inputs[2], paramShape);
        });

    internal static bool TrySecondOrderGroupNorm(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, int numGroups, double epsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            var x = inputs[0];
            int n = x._shape[0];
            var grouped = e.Reshape(x, new[] { n, numGroups, x.Length / (n * numGroups) });
            var xhat = e.Reshape(NormalizeRecorded(e, grouped, new[] { 2 }, epsilon), (int[])x._shape.Clone());
            return AffineRecorded(e, xhat, inputs[1], inputs[2], ChannelShape(x.Rank, x._shape[1]));
        });

    internal static bool TrySecondOrderRMSNorm(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double epsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstTwoInputs, e =>
        {
            var x = inputs[0];
            var gamma = inputs[1];
            int lead = x.Rank - gamma.Rank;
            var axes = Enumerable.Range(lead, gamma.Rank).ToArray();
            var paramShape = Enumerable.Repeat(1, lead).Concat(gamma._shape).ToArray();
            var meanSquare = e.ReduceMean(e.TensorMultiply(x, x), axes, keepDims: true);
            var rms = e.TensorSqrt(e.TensorAddScalar(meanSquare, Num(epsilon)));
            return AffineRecorded(e, e.TensorDivide(x, Bcast(e, rms, x._shape)), gamma, null, paramShape);
        });

    /// <summary>The inference-form batch norm: <paramref name="mean"/> and <paramref name="variance"/> are constants.</summary>
    internal static bool TrySecondOrderBatchNormAffine(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> mean, Tensor<T> variance, double epsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            var x = inputs[0];
            var paramShape = ChannelShape(x.Rank, x._shape[1]);
            var centred = e.TensorSubtract(x, Bcast(e, e.Reshape(mean, paramShape), x._shape));
            var std = e.TensorSqrt(e.TensorAddScalar(e.Reshape(variance, paramShape), Num(epsilon)));
            return AffineRecorded(e, e.TensorDivide(centred, Bcast(e, std, x._shape)), inputs[1], inputs[2], paramShape);
        });

    /// <summary>
    /// log(var + ε) over the saved axes. ε is not saved, but var + ε is: the composite adds the constant
    /// (var + ε) − var, which is ε to rounding.
    /// </summary>
    internal static bool TrySecondOrderReduceLogVariance(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, int[] axes, Tensor<T> variancePlusEpsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            var x = inputs[0];
            var mean = e.ReduceMean(x, axes, keepDims: true);
            var centred = e.TensorSubtract(x, Bcast(e, mean, x._shape));
            var variance = e.ReduceMean(e.TensorMultiply(centred, centred), axes, keepDims: true);
            Tensor<T> offset;
            using (new NoGradScope<T>())
            {
                var plain = RoIAlignPlan.ToDoubles(variance);
                var saved = RoIAlignPlan.ToDoubles(variancePlusEpsilon);
                var eps = new double[plain.Length];
                for (int i = 0; i < eps.Length; i++) eps[i] = saved[i % saved.Length] - plain[i];
                offset = Constant(eps, (int[])variance._shape.Clone());
            }
            return e.TensorLog(e.TensorAdd(variance, offset));
        });

    // ── losses (scalar mean losses; targets are constants) ───────────

    internal static bool TrySecondOrderMSELoss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            var d = e.TensorSubtract(inputs[0], inputs[1]);
            return e.TensorMultiplyScalar(SumAll(e, e.TensorMultiply(d, d)), Num(1.0 / inputs[0].Length));
        });

    internal static bool TrySecondOrderL1Loss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
            e.TensorMultiplyScalar(SumAll(e, e.TensorAbs(e.TensorSubtract(inputs[0], inputs[1]))), Num(1.0 / inputs[0].Length)));

    internal static bool TrySecondOrderHuberLoss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double delta)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            var d = e.TensorSubtract(inputs[0], inputs[1]);
            // The quadratic / linear branch is chosen per element (constant on each side of |d| = δ).
            Tensor<T> quadratic, linear;
            using (new NoGradScope<T>())
            {
                var dv = RoIAlignPlan.ToDoubles(d);
                var q = new double[dv.Length];
                var l = new double[dv.Length];
                for (int i = 0; i < dv.Length; i++) { bool inside = Math.Abs(dv[i]) <= delta; q[i] = inside ? 1 : 0; l[i] = inside ? 0 : 1; }
                quadratic = Constant(q, (int[])d._shape.Clone());
                linear = Constant(l, (int[])d._shape.Clone());
            }
            var quad = e.TensorMultiply(quadratic, e.TensorMultiplyScalar(e.TensorMultiply(d, d), Num(0.5)));
            var lin = e.TensorMultiply(linear, e.TensorMultiplyScalar(e.TensorAddScalar(e.TensorAbs(d), Num(-0.5 * delta)), Num(delta)));
            return e.TensorMultiplyScalar(SumAll(e, e.TensorAdd(quad, lin)), Num(1.0 / inputs[0].Length));
        });

    internal static bool TrySecondOrderBCEWithLogitsLoss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            // max(x, 0) - x·t + log(1 + exp(-|x|)), the forward's stable form.
            var x = inputs[0];
            var softplusTail = e.TensorLog(e.TensorAddScalar(e.TensorExp(e.TensorNegate(e.TensorAbs(x))), Num(1.0)));
            var loss = e.TensorAdd(e.TensorSubtract(e.ReLU(x), e.TensorMultiply(x, inputs[1])), softplusTail);
            return e.TensorMultiplyScalar(SumAll(e, loss), Num(1.0 / x.Length));
        });

    internal static bool TrySecondOrderBinaryCrossEntropy(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double epsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            var p = e.TensorClamp(inputs[0], Num(epsilon), Num(1.0 - epsilon));
            var t = inputs[1];
            var first = e.TensorMultiply(t, e.TensorLog(p));
            var second = e.TensorMultiply(OneMinus(e, t), e.TensorLog(OneMinus(e, p)));
            return e.TensorNegate(e.TensorAdd(first, second));
        });

    // log_softmax over the last axis of a [batch, classes] tensor, as recorded ops.
    private static Tensor<T> LogSoftmaxRows(IEngine e, Tensor<T> logits)
    {
        var lse = e.TensorLogSumExp(logits, logits.Rank - 1, keepDims: true);
        return e.TensorSubtract(logits, Bcast(e, lse, logits._shape));
    }

    // Per-row class weights of a cross-entropy target: a one-hot row for a class index, else the positive part of the
    // soft target (the forward skips non-positive entries). Index targets out of range contribute nothing.
    private static Tensor<T> ClassWeights(Tensor<T> targets, int batch, int classes)
    {
        var t = RoIAlignPlan.ToDoubles(targets);
        var w = new double[batch * classes];
        if (targets.Rank == 1)
        {
            for (int b = 0; b < batch; b++)
            {
                int c = (int)t[b];
                if (c >= 0 && c < classes) w[b * classes + c] = 1;
            }
        }
        else
        {
            for (int i = 0; i < w.Length; i++) w[i] = t[i] > 0 ? t[i] : 0;
        }
        return Constant(w, new[] { batch, classes });
    }

    internal static bool TrySecondOrderCrossEntropyLoss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            var logits = inputs[0];
            int batch = logits._shape[0], classes = logits._shape[1];
            var weights = ClassWeights(inputs[1], batch, classes);
            return e.TensorMultiplyScalar(SumAll(e, e.TensorMultiply(weights, LogSoftmaxRows(e, logits))), Num(-1.0 / batch));
        });

    internal static bool TrySecondOrderNLLLoss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            var logProbs = inputs[0];
            int batch = logProbs._shape[0], classes = logProbs.Length / batch;
            var targets = RoIAlignPlan.ToDoubles(inputs[1]);
            var w = new double[batch * classes];
            for (int b = 0; b < batch; b++)
            {
                int c = (int)targets[b];
                if (c >= 0 && c < classes) w[b * classes + c] = 1;
            }
            var weights = Constant(w, (int[])logProbs._shape.Clone());
            return e.TensorMultiplyScalar(SumAll(e, e.TensorMultiply(weights, logProbs)), Num(-1.0 / batch));
        });

    internal static bool TrySecondOrderKLDivLoss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            // mean(t·(log t - x)): linear in x with slope -t/N (the t·log t part is a constant).
            var target = inputs[1];
            return e.TensorMultiplyScalar(SumAll(e, e.TensorMultiply(target, inputs[0])), Num(-1.0 / inputs[0].Length));
        });

    internal static bool TrySecondOrderCosineSimilarityLoss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstTwoInputs, e =>
        {
            // The backward's form: dot / (sqrt(|a|² + 1e-8) · sqrt(|b|² + 1e-8)).
            var a = inputs[0];
            var b = inputs[1];
            var dot = SumAll(e, e.TensorMultiply(a, b));
            var na = e.TensorSqrt(e.TensorAddScalar(SumAll(e, e.TensorMultiply(a, a)), Num(1e-8)));
            var nb = e.TensorSqrt(e.TensorAddScalar(SumAll(e, e.TensorMultiply(b, b)), Num(1e-8)));
            return e.TensorDivide(dot, e.TensorMultiply(na, nb));
        });

    // ── fused linear ─────────────────────────────────────────────────

    /// <summary>The activation of a fused linear layer, applied to its recorded pre-activation.</summary>
    internal enum FusedLinearActivation { ReLU, Sigmoid, Tanh, GELU, Swish }

    internal static bool TrySecondOrderFusedLinear(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, FusedLinearActivation activation)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            var pre = e.TensorMatMul(inputs[0], inputs[1]);
            if (inputs.Length > 2 && inputs[2] is not null) pre = e.TensorAdd(pre, Bcast(e, inputs[2], pre._shape));
            return activation switch
            {
                FusedLinearActivation.ReLU => e.ReLU(pre),
                FusedLinearActivation.Sigmoid => e.TensorSigmoid(pre),
                FusedLinearActivation.Tanh => e.Tanh(pre),
                FusedLinearActivation.GELU => e.GELU(pre),
                _ => e.TensorMultiply(pre, e.TensorSigmoid(pre)),
            };
        });

    /// <summary>
    /// Fused linear + mean cross-entropy: <c>-mean_r Σ_v w[r,v]·log_softmax(h·W + b)[r,v]</c> with w the one-hot of
    /// <paramref name="ids"/> or the dense target (a constant: the fused backward gives it no gradient).
    /// </summary>
    internal static bool TrySecondOrderFusedLinearCrossEntropy(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, int[]? ids)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            var logits = e.TensorMatMul(inputs[0], inputs[1]);
            logits = e.TensorAdd(logits, Bcast(e, inputs[2], logits._shape));
            int n = logits._shape[0], vocab = logits._shape[1];
            Tensor<T> weights;
            if (ids is not null)
            {
                var w = new double[n * vocab];
                for (int r = 0; r < n; r++) w[r * vocab + ids[r]] = 1;
                weights = Constant(w, new[] { n, vocab });
            }
            else
            {
                weights = inputs[3];
            }
            return e.TensorMultiplyScalar(SumAll(e, e.TensorMultiply(weights, LogSoftmaxRows(e, logits))), Num(-1.0 / n));
        });

    // ── distances and kernels ────────────────────────────────────────

    // ‖d‖_p over the last axis as recorded ops (p = 2 through the square root, p = 1 through |d|).
    private static Tensor<T> PNormLastAxis(IEngine e, Tensor<T> d, double p)
    {
        int axis = d.Rank - 1;
        if (p == 2) return e.TensorSqrt(e.ReduceSum(e.TensorMultiply(d, d), new[] { axis }, keepDims: false));
        var abs = e.TensorAbs(d);
        if (p == 1) return e.ReduceSum(abs, new[] { axis }, keepDims: false);
        return e.TensorPow(e.ReduceSum(e.TensorPow(abs, Num(p)), new[] { axis }, keepDims: false), Num(1.0 / p));
    }

    private static bool IsSmoothPNorm(double p) => p > 0 && !double.IsInfinity(p) && !double.IsNaN(p);

    internal static bool TrySecondOrderCDist(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double p)
    {
        if (!IsSmoothPNorm(p)) return false;
        return TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstTwoInputs, e =>
        {
            var x1 = inputs[0];
            var x2 = inputs[1];
            int d = x1._shape[x1.Rank - 1];
            var full = output._shape.Concat(new[] { d }).ToArray();     // [...batch, m, n, d]
            var s1 = x1._shape.Take(x1.Rank - 1).Concat(new[] { 1, d }).ToArray();
            var s2 = x2._shape.Take(x2.Rank - 2).Concat(new[] { 1, x2._shape[x2.Rank - 2], d }).ToArray();
            var diff = e.TensorSubtract(Bcast(e, e.Reshape(x1, s1), full), Bcast(e, e.Reshape(x2, s2), full));
            return PNormLastAxis(e, diff, p);
        });
    }

    internal static bool TrySecondOrderPDist(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double p)
    {
        if (!IsSmoothPNorm(p)) return false;
        return TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            var x = inputs[0];
            int n = x._shape[0];
            int pairs = n * (n - 1) / 2;
            // Row r of S picks x_i - x_j for the r-th pair (i < j), in torch.pdist's order.
            var s = new double[pairs * n];
            int r = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++, r++) { s[r * n + i] = 1; s[r * n + j] = -1; }
            var diff = e.TensorMatMul(Constant(s, new[] { pairs, n }), x);
            return PNormLastAxis(e, diff, p);
        });
    }

    internal static bool TrySecondOrderCosineSimilarityDim(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, int dim, double eps)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstTwoInputs, e =>
        {
            // dot / (max(|a|, eps) · max(|b|, eps)) along dim.
            var a = inputs[0];
            var b = inputs[1];
            var axis = new[] { dim };
            var dot = e.ReduceSum(e.TensorMultiply(a, b), axis, keepDims: false);
            var big = Num(double.MaxValue);
            var na = e.TensorClamp(e.TensorSqrt(e.ReduceSum(e.TensorMultiply(a, a), axis, keepDims: false)), Num(eps), big);
            var nb = e.TensorClamp(e.TensorSqrt(e.ReduceSum(e.TensorMultiply(b, b), axis, keepDims: false)), Num(eps), big);
            return e.TensorDivide(dot, e.TensorMultiply(na, nb));
        });

    internal static bool TrySecondOrderRBFKernel(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            // exp(-ε_c · ‖x_b - c_c‖²)
            var x = inputs[0];
            var centers = inputs[1];
            int batch = x._shape[0], features = x._shape[1], count = centers._shape[0];
            var full = new[] { batch, count, features };
            var diff = e.TensorSubtract(Bcast(e, e.Reshape(x, new[] { batch, 1, features }), full),
                Bcast(e, e.Reshape(centers, new[] { 1, count, features }), full));
            var distSq = e.ReduceSum(e.TensorMultiply(diff, diff), new[] { 2 }, keepDims: false);
            var scaled = e.TensorMultiply(distSq, Bcast(e, e.Reshape(inputs[2], new[] { 1, count }), distSq._shape));
            return e.TensorExp(e.TensorNegate(scaled));
        });

    // ── detection: the IoU family ────────────────────────────────────

    /// <summary>The variant of <see cref="TrySecondOrderBoxIou"/>.</summary>
    internal enum IouKind { Iou, Generalized, Distance, Complete }

    internal static bool TrySecondOrderBoxIou(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, IouKind kind)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstTwoInputs, e =>
        {
            var a = inputs[0];
            var b = inputs[1];
            int n = a._shape[0], m = b._shape[0];
            var pair = new[] { n, m };
            Tensor<T> A(int k) => Bcast(e, e.TensorNarrow(a, 1, k, 1), pair);
            Tensor<T> B(int k) => Bcast(e, e.Reshape(e.TensorNarrow(b, 1, k, 1), new[] { 1, m }), pair);
            Tensor<T> Half(Tensor<T> t) => e.TensorMultiplyScalar(t, Num(0.5));
            // max / min as (s ± |d|)/2: exact away from ties, the same subgradient convention as a comparison.
            Tensor<T> Max(Tensor<T> u, Tensor<T> v) => Half(e.TensorAdd(e.TensorAdd(u, v), e.TensorAbs(e.TensorSubtract(u, v))));
            Tensor<T> Min(Tensor<T> u, Tensor<T> v) => Half(e.TensorSubtract(e.TensorAdd(u, v), e.TensorAbs(e.TensorSubtract(u, v))));
            // Where a forward guard (a positive denominator) fails, it yields 0: mask, and keep the denominator at 1.
            Tensor<T> Positive(Tensor<T> t)
            {
                using (new NoGradScope<T>())
                {
                    var v = RoIAlignPlan.ToDoubles(t);
                    var mask = new double[v.Length];
                    for (int i = 0; i < v.Length; i++) mask[i] = v[i] > 0 ? 1 : 0;
                    return Constant(mask, pair);
                }
            }
            Tensor<T> SafeDivide(Tensor<T> num, Tensor<T> den)
            {
                var ok = Positive(den);
                var safeDen = e.TensorAdd(e.TensorMultiply(den, ok), OneMinus(e, ok));
                return e.TensorMultiply(e.TensorDivide(num, safeDen), ok);
            }

            Tensor<T> ax1 = A(0), ay1 = A(1), ax2 = A(2), ay2 = A(3);
            Tensor<T> bx1 = B(0), by1 = B(1), bx2 = B(2), by2 = B(3);
            var aw = e.TensorSubtract(ax2, ax1);
            var ah = e.TensorSubtract(ay2, ay1);
            var bw = e.TensorSubtract(bx2, bx1);
            var bh = e.TensorSubtract(by2, by1);
            // BoxArea clamps each side at 0, as the forward does.
            var areaA = e.TensorMultiply(e.ReLU(aw), e.ReLU(ah));
            var areaB = e.TensorMultiply(e.ReLU(bw), e.ReLU(bh));
            var iw = e.ReLU(e.TensorSubtract(Min(ax2, bx2), Max(ax1, bx1)));
            var ih = e.ReLU(e.TensorSubtract(Min(ay2, by2), Max(ay1, by1)));
            var inter = e.TensorMultiply(iw, ih);
            var union = e.TensorSubtract(e.TensorAdd(areaA, areaB), inter);
            var iou = SafeDivide(inter, union);
            if (kind == IouKind.Iou) return iou;

            var ew = e.TensorSubtract(Max(ax2, bx2), Min(ax1, bx1));
            var eh = e.TensorSubtract(Max(ay2, by2), Min(ay1, by1));
            if (kind == IouKind.Generalized)
            {
                var enclose = e.TensorMultiply(ew, eh);
                return e.TensorSubtract(iou, SafeDivide(e.TensorSubtract(enclose, union), enclose));
            }

            var dx = Half(e.TensorSubtract(e.TensorAdd(ax1, ax2), e.TensorAdd(bx1, bx2)));
            var dy = Half(e.TensorSubtract(e.TensorAdd(ay1, ay2), e.TensorAdd(by1, by2)));
            var centreSq = e.TensorAdd(e.TensorMultiply(dx, dx), e.TensorMultiply(dy, dy));
            var diagSq = e.TensorAdd(e.TensorMultiply(ew, ew), e.TensorMultiply(eh, eh));
            var diouTerm = SafeDivide(centreSq, diagSq);
            var diou = e.TensorSubtract(iou, diouTerm);
            if (kind == IouKind.Distance) return diou;

            // CIoU: - α·v with v = 4/π²·(atan(wA/hA) - atan(wB/hB))², α = v / ((1 - IoU) + v) a stop-gradient (as the
            // first-order backward treats it), both only where the heights are positive.
            var heightsOk = e.TensorMultiply(Positive(ah), Positive(bh));
            var safeAh = e.TensorAdd(e.TensorMultiply(ah, heightsOk), OneMinus(e, heightsOk));
            var safeBh = e.TensorAdd(e.TensorMultiply(bh, heightsOk), OneMinus(e, heightsOk));
            var angle = e.TensorSubtract(e.TensorAtan(e.TensorDivide(aw, safeAh)), e.TensorAtan(e.TensorDivide(bw, safeBh)));
            var v = e.TensorMultiplyScalar(e.TensorMultiply(angle, angle), Num(4.0 / (Math.PI * Math.PI)));
            Tensor<T> alpha;
            using (new NoGradScope<T>())
            {
                var vv = RoIAlignPlan.ToDoubles(v);
                var raw = RoIAlignPlan.ToDoubles(iou);
                var ok = RoIAlignPlan.ToDoubles(heightsOk);
                var al = new double[vv.Length];
                for (int i = 0; i < al.Length; i++)
                {
                    double denom = (1.0 - raw[i]) + vv[i];
                    al[i] = ok[i] > 0 && denom > 0 ? vv[i] / denom : 0;
                }
                alpha = Constant(al, pair);
            }
            return e.TensorSubtract(diou, e.TensorMultiply(alpha, v));
        });

    // ── attention ────────────────────────────────────────────────────

    /// <summary>
    /// softmax(scale · Q Kᵀ + bias) V for the uncapped scaled-dot-product attention, as recorded ops. The mask is not
    /// saved, but the forward's probabilities are exactly zero where it masked: those scores get a large negative
    /// constant bias, so their probabilities stay exactly zero here too.
    /// </summary>
    internal static bool TrySecondOrderScaledDotProductAttention(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> attentionWeights, double scale)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstThreeInputs, e =>
        {
            var q = inputs[0];
            var k = inputs[1];
            var v = inputs[2];
            int rank = k.Rank;
            var perm = Enumerable.Range(0, rank).ToArray();
            perm[rank - 1] = rank - 2;
            perm[rank - 2] = rank - 1;
            var scores = e.TensorMultiplyScalar(e.TensorMatMul(q, e.TensorPermute(k, perm)), Num(scale));
            var weights = RoIAlignPlan.ToDoubles(attentionWeights);
            if (Array.Exists(weights, w => w == 0))
            {
                var bias = new double[weights.Length];
                for (int i = 0; i < bias.Length; i++) bias[i] = weights[i] == 0 ? -1e30 : 0;
                scores = e.TensorAdd(scores, Constant(bias, (int[])scores._shape.Clone()));
            }
            return e.TensorMatMul(e.Softmax(scores, -1), v);
        });

    // ── sequence models ──────────────────────────────────────────────

    /// <summary>
    /// The linear-chain CRF log-partition by the forward algorithm in log space, as recorded ops:
    /// α_0 = start + e_0, α_t[j] = e_t[j] + logsumexp_i(α_{t-1}[i] + trans[i, j]), logZ = logsumexp(α_T + end).
    /// </summary>
    internal static void CrfForwardBackwardWithSecondOrder(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var state = (CrfForwardState)savedState[0];
        var sourceIndices = Enumerable.Range(0, inputs.Length).ToArray();
        if (TryCompositeBackwardFor(gradOutput, inputs, engine, grads, sourceIndices, e =>
            {
                int batch = state.Batch, steps = state.SeqLen, tags = state.NumTags;
                var emissions = e.Reshape(inputs[0], new[] { batch, steps, tags });
                var trans = inputs[1];
                int next = 2;
                var start = state.HasStart ? inputs[next++] : null;
                var end = state.HasEnd ? inputs[next] : null;
                var square = new[] { batch, tags, tags };
                var transB = Bcast(e, e.Reshape(trans, new[] { 1, tags, tags }), square);
                Tensor<T> Emit(int t) => e.Reshape(e.TensorNarrow(emissions, 1, t, 1), new[] { batch, tags });
                var alpha = Emit(0);
                if (start is not null) alpha = e.TensorAdd(alpha, Bcast(e, e.Reshape(start, new[] { 1, tags }), alpha._shape));
                for (int t = 1; t < steps; t++)
                {
                    var scores = e.TensorAdd(Bcast(e, e.Reshape(alpha, new[] { batch, tags, 1 }), square), transB);
                    alpha = e.TensorAdd(Emit(t), e.TensorLogSumExp(scores, 1, keepDims: false));
                }
                if (end is not null) alpha = e.TensorAdd(alpha, Bcast(e, e.Reshape(end, new[] { 1, tags }), alpha._shape));
                return e.TensorLogSumExp(alpha, 1, keepDims: false);
            }))
            return;
        CrfForwardBackward(gradOutput, inputs, output, savedState, engine, grads);
    }

    /// <summary>[sin(t·f), cos(t·f)] per timestep (a zero column for an odd width), as recorded ops.</summary>
    internal static void SinusoidalTimeEmbeddingBackwardWithSecondOrder(Tensor<T> gradOutput, Tensor<T>[] inputs,
        Tensor<T> output, object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var freq = (double[])savedState[0];
        int embeddingDim = (int)savedState[1];
        if (TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
            {
                int batch = inputs[0]._shape[0], half = freq.Length;
                var angle = e.TensorMultiply(Bcast(e, e.Reshape(inputs[0], new[] { batch, 1 }), new[] { batch, half }),
                    Bcast(e, Constant(freq, new[] { 1, half }), new[] { batch, half }));
                var parts = new List<Tensor<T>> { e.TensorSin(angle), e.TensorCos(angle) };
                if (embeddingDim > 2 * half) parts.Add(new Tensor<T>(new[] { batch, embeddingDim - 2 * half }));
                return e.Concat(parts, 1);
            }))
            return;
        SinusoidalTimeEmbeddingBackward(gradOutput, inputs, output, savedState, engine, grads);
    }

    /// <summary>
    /// The CTC loss −log P(target | x) per batch element by the alpha recursion in log space, as recorded ops over the
    /// extended (blank-interleaved) label sequence. Unreachable states carry a large negative finite score instead of
    /// −∞, so the log-sum-exp stays finite and their weight underflows to exactly zero.
    /// </summary>
    internal static bool TrySecondOrderCtcLoss(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<int> targets, int[] inputLengths, int[] targetLengths, int blank)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput, e =>
        {
            const double Unreachable = -1e30;
            var logProbs = inputs[0];               // [T, batch, classes]
            int batchSize = logProbs._shape[1], classes = logProbs._shape[2];
            var losses = new List<Tensor<T>>(batchSize);
            int targetOffset = 0;
            for (int n = 0; n < batchSize; n++)
            {
                int steps = inputLengths[n], u = targetLengths[n], s = 2 * u + 1;
                var labels = new int[s];
                for (int i = 0; i < s; i++) labels[i] = i % 2 == 0 ? blank : targets.GetFlat(targetOffset + i / 2);
                targetOffset += u;

                // pick[c, i] = 1 when extended label i is class c: lp_t · pick = the label scores at step t.
                var pick = new double[classes * s];
                for (int i = 0; i < s; i++) pick[labels[i] * s + i] = 1;
                var pickT = Constant(pick, new[] { classes, s });
                // Transitions into state i from i-1, and from i-2 when allowed; a state with none gets Unreachable.
                var shift1 = new double[s * s];
                var shift2 = new double[s * s];
                var bias1 = new double[s];
                var bias2 = new double[s];
                for (int i = 0; i < s; i++)
                {
                    if (i >= 1) shift1[(i - 1) * s + i] = 1; else bias1[i] = Unreachable;
                    if (i >= 2 && labels[i] != blank && labels[i] != labels[i - 2]) shift2[(i - 2) * s + i] = 1; else bias2[i] = Unreachable;
                }
                var shift1T = Constant(shift1, new[] { s, s });
                var shift2T = Constant(shift2, new[] { s, s });
                var bias1T = Constant(bias1, new[] { 1, s });
                var bias2T = Constant(bias2, new[] { 1, s });

                Tensor<T> Scores(int t) => e.TensorMatMul(
                    e.Reshape(e.TensorNarrow(e.TensorNarrow(logProbs, 0, t, 1), 1, n, 1), new[] { 1, classes }), pickT);
                var init = new double[s];
                for (int i = 2; i < s; i++) init[i] = Unreachable;
                var alpha = e.TensorAdd(Scores(0), Constant(init, new[] { 1, s }));
                for (int t = 1; t < steps; t++)
                {
                    var stay = alpha;
                    var step1 = e.TensorAdd(e.TensorMatMul(alpha, shift1T), bias1T);
                    var step2 = e.TensorAdd(e.TensorMatMul(alpha, shift2T), bias2T);
                    var prev = e.TensorLogSumExp(e.Concat(new[] { stay, step1, step2 }, 0), 0, keepDims: true);
                    alpha = e.TensorAdd(prev, Scores(t));
                }
                var ends = new double[s * 2];
                ends[(s - 1) * 2] = 1;
                var endBias = new double[2];
                if (s >= 2) ends[(s - 2) * 2 + 1] = 1; else endBias[1] = Unreachable;
                var final = e.TensorAdd(e.TensorMatMul(alpha, Constant(ends, new[] { s, 2 })), Constant(endBias, new[] { 1, 2 }));
                losses.Add(e.Reshape(e.TensorNegate(e.TensorLogSumExp(final, 1, keepDims: false)), new[] { 1 }));
            }
            return losses.Count == 1 ? losses[0] : e.Concat(losses, 0);
        });
}
