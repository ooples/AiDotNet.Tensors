// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NN.Losses;

/// <summary>
/// The "long tail" of <c>torch.nn.functional</c> losses missing from
/// the engine surface. Each entry mirrors PyTorch's reduction semantics:
/// <c>"none"</c> returns the per-element loss tensor, <c>"mean"</c> averages
/// over every element, <c>"sum"</c> sums everything to a scalar.
/// </summary>
public enum LossReduction
{
    /// <summary>Per-element loss; output has the same shape as input.</summary>
    None,
    /// <summary>Average over all elements; scalar output.</summary>
    Mean,
    /// <summary>Sum over all elements; scalar output.</summary>
    Sum,
}

/// <summary>
/// Functional loss surface added by #223. Engine-level losses
/// (HuberLoss, CTCLoss, CosineSimilarityLoss) already exist on
/// <c>IEngine</c> — this file fills in the remainder PyTorch lists in
/// <c>torch.nn.functional</c>.
/// </summary>
public static class Losses
{
    /// <summary>SmoothL1Loss with the configurable <paramref name="beta"/>
    /// transition point. Mirrors <c>F.smooth_l1_loss</c>; matches Huber
    /// when <c>beta == 1</c>.</summary>
    public static Tensor<T> SmoothL1Loss<T>(Tensor<T> input, Tensor<T> target, double beta = 1.0,
        LossReduction reduction = LossReduction.Mean)
    {
        EnsureSameShape(input, target);
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var inSpan = input.AsSpan();
        var tgtSpan = target.AsSpan();
        var outSpan = output.AsWritableSpan();

        // #294 NumericFastPath: bypass per-element INumericOperations<T>
        // dispatch for the float / double primitive cases. Audit
        // measured 4× speedup on this exact loop shape (per-element
        // ToDouble + scalar arithmetic + FromDouble). T is unconstrained
        // here so we pattern-match on the concrete data array.
        var inArr = input.GetDataArray();
        var tgtArr = target.GetDataArray();
        var outArr = output.GetDataArray();
        if (inArr is double[] iD && tgtArr is double[] tD && outArr is double[] oD)
        {
            double half = 0.5;
            double halfBeta = half * beta;
            int n = inSpan.Length;
            for (int i = 0; i < n; i++)
            {
                double diff = iD[i] - tD[i];
                double absDiff = Math.Abs(diff);
                oD[i] = absDiff < beta ? half * diff * diff / beta : absDiff - halfBeta;
            }
            return Reduce(output, reduction);
        }
        if (inArr is float[] iF && tgtArr is float[] tF && outArr is float[] oF)
        {
            float betaF = (float)beta;
            float halfBetaF = 0.5f * betaF;
            int n = inSpan.Length;
            for (int i = 0; i < n; i++)
            {
                float diff = iF[i] - tF[i];
                float absDiff = MathF.Abs(diff);
                oF[i] = absDiff < betaF ? 0.5f * diff * diff / betaF : absDiff - halfBetaF;
            }
            return Reduce(output, reduction);
        }

        for (int i = 0; i < inSpan.Length; i++)
        {
            double diff = ops.ToDouble(ops.Subtract(inSpan[i], tgtSpan[i]));
            double absDiff = Math.Abs(diff);
            double loss = absDiff < beta
                ? 0.5 * diff * diff / beta
                : absDiff - 0.5 * beta;
            outSpan[i] = ops.FromDouble(loss);
        }
        return Reduce(output, reduction);
    }

    /// <summary>Poisson NLL: <c>input − target · log(input + eps)</c>
    /// when <paramref name="logInput"/> is false; <c>exp(input) − target · input</c>
    /// when true. Matches <c>F.poisson_nll_loss</c>.</summary>
    public static Tensor<T> PoissonNllLoss<T>(Tensor<T> input, Tensor<T> target,
        bool logInput = true, bool full = false, double eps = 1e-8,
        LossReduction reduction = LossReduction.Mean)
    {
        EnsureSameShape(input, target);
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var inSpan = input.AsSpan();
        var tgtSpan = target.AsSpan();
        var outSpan = output.AsWritableSpan();
        for (int i = 0; i < inSpan.Length; i++)
        {
            double xi = ops.ToDouble(inSpan[i]);
            double ti = ops.ToDouble(tgtSpan[i]);
            double loss = logInput
                ? Math.Exp(xi) - ti * xi
                : xi - ti * Math.Log(xi + eps);
            // Stirling approximation for the full target-factorial term.
            if (full && ti > 1)
                loss += ti * Math.Log(ti) - ti + 0.5 * Math.Log(2.0 * Math.PI * ti);
            outSpan[i] = ops.FromDouble(loss);
        }
        return Reduce(output, reduction);
    }

    /// <summary>Gaussian NLL with predicted mean and variance.
    /// <c>0.5 · (log(var) + (target − input)² / var) [+ const]</c>.</summary>
    public static Tensor<T> GaussianNllLoss<T>(Tensor<T> input, Tensor<T> target, Tensor<T> variance,
        bool full = false, double eps = 1e-6,
        LossReduction reduction = LossReduction.Mean)
    {
        EnsureSameShape(input, target);
        EnsureSameShape(input, variance);
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var inSpan = input.AsSpan();
        var tgtSpan = target.AsSpan();
        var varSpan = variance.AsSpan();
        var outSpan = output.AsWritableSpan();
        double constTerm = full ? 0.5 * Math.Log(2.0 * Math.PI) : 0.0;

        // #294 NumericFastPath: float / double primitive fast paths
        // bypass INumericOperations<T>'s per-element virtual dispatch
        // via concrete-array pattern match.
        var inArr = input.GetDataArray();
        var tgtArr = target.GetDataArray();
        var varArr = variance.GetDataArray();
        var outArr = output.GetDataArray();
        if (inArr is double[] iD && tgtArr is double[] tD && varArr is double[] vD && outArr is double[] oD)
        {
            int n = inSpan.Length;
            for (int i = 0; i < n; i++)
            {
                double xi = iD[i];
                double ti = tD[i];
                double vi = vD[i] > eps ? vD[i] : eps;
                double diff = ti - xi;
                oD[i] = 0.5 * (Math.Log(vi) + diff * diff / vi) + constTerm;
            }
            return Reduce(output, reduction);
        }
        if (inArr is float[] iF && tgtArr is float[] tF && varArr is float[] vF && outArr is float[] oF)
        {
            float epsF = (float)eps;
            float constTermF = (float)constTerm;
            int n = inSpan.Length;
            for (int i = 0; i < n; i++)
            {
                float xi = iF[i];
                float ti = tF[i];
                float vi = vF[i] > epsF ? vF[i] : epsF;
                float diff = ti - xi;
                oF[i] = 0.5f * (MathF.Log(vi) + diff * diff / vi) + constTermF;
            }
            return Reduce(output, reduction);
        }

        for (int i = 0; i < inSpan.Length; i++)
        {
            double xi = ops.ToDouble(inSpan[i]);
            double ti = ops.ToDouble(tgtSpan[i]);
            double vi = Math.Max(eps, ops.ToDouble(varSpan[i]));
            double diff = ti - xi;
            double loss = 0.5 * (Math.Log(vi) + diff * diff / vi) + constTerm;
            outSpan[i] = ops.FromDouble(loss);
        }
        return Reduce(output, reduction);
    }

    /// <summary>
    /// MultiMarginLoss — for each row, sum over j of
    /// <c>max(0, margin − x[y] + x[j])^p</c> for j ≠ y.
    /// Mirrors <c>F.multi_margin_loss</c>.
    /// </summary>
    public static Tensor<T> MultiMarginLoss<T>(Tensor<T> input, Tensor<int> target,
        int p = 1, double margin = 1.0, Tensor<T>? weight = null,
        LossReduction reduction = LossReduction.Mean)
    {
        if (input.Rank != 2)
            throw new ArgumentException("MultiMarginLoss expects rank-2 input [batch, classes].", nameof(input));
        if (target.Rank != 1 || target.Length != input._shape[0])
            throw new ArgumentException("Target must be rank-1 with length = batch.", nameof(target));
        var ops = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0], classes = input._shape[1];
        if (weight is not null && weight.Length != classes)
            throw new ArgumentException(
                $"weight length {weight.Length} must equal classes {classes}.", nameof(weight));
        var output = new Tensor<T>(new[] { batch });
        var inSpan = input.AsSpan();
        var tSpan = target.AsSpan();
        var wSpan = weight is null ? ReadOnlySpan<T>.Empty : weight.AsSpan();
        var outSpan = output.AsWritableSpan();
        for (int b = 0; b < batch; b++)
        {
            int yi = tSpan[b];
            if ((uint)yi >= (uint)classes)
                throw new ArgumentException(
                    $"Target class index {yi} out of range [0, {classes}).", nameof(target));
            double loss = 0;
            double xy = ops.ToDouble(inSpan[b * classes + yi]);
            for (int j = 0; j < classes; j++)
            {
                if (j == yi) continue;
                double margin_xj = margin - xy + ops.ToDouble(inSpan[b * classes + j]);
                if (margin_xj > 0)
                {
                    double term = p == 1 ? margin_xj : Math.Pow(margin_xj, p);
                    if (weight is not null) term *= ops.ToDouble(wSpan[yi]);
                    loss += term;
                }
            }
            outSpan[b] = ops.FromDouble(loss / classes);
        }
        return Reduce(output, reduction);
    }

    /// <summary>MultiLabelMarginLoss — multi-label variant where the
    /// target row contains class indices terminated by -1.
    /// Mirrors <c>F.multi_label_margin_loss</c>.</summary>
    public static Tensor<T> MultiLabelMarginLoss<T>(Tensor<T> input, Tensor<int> target,
        LossReduction reduction = LossReduction.Mean)
    {
        if (input.Rank != 2)
            throw new ArgumentException("MultiLabelMarginLoss expects rank-2 input.", nameof(input));
        if (target.Rank != 2 || target._shape[0] != input._shape[0] || target._shape[1] != input._shape[1])
            throw new ArgumentException("target shape must equal input shape.", nameof(target));
        var ops = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0], classes = input._shape[1];
        var output = new Tensor<T>(new[] { batch });
        var inSpan = input.AsSpan();
        var tSpan = target.AsSpan();
        var outSpan = output.AsWritableSpan();
        var positive = new bool[classes];
        for (int b = 0; b < batch; b++)
        {
            // Mark every positive class index until -1 sentinel.
            Array.Clear(positive, 0, classes);
            for (int k = 0; k < classes; k++)
            {
                int idx = tSpan[b * classes + k];
                if (idx < 0) break;
                if ((uint)idx >= (uint)classes)
                    throw new ArgumentException($"Target index {idx} out of range.", nameof(target));
                positive[idx] = true;
            }

            double loss = 0;
            for (int pos = 0; pos < classes; pos++)
            {
                if (!positive[pos]) continue;
                for (int neg = 0; neg < classes; neg++)
                {
                    if (positive[neg]) continue;
                    double margin = 1.0 - ops.ToDouble(inSpan[b * classes + pos])
                                        + ops.ToDouble(inSpan[b * classes + neg]);
                    if (margin > 0) loss += margin;
                }
            }
            outSpan[b] = ops.FromDouble(loss / classes);
        }
        return Reduce(output, reduction);
    }

    /// <summary>MultiLabelSoftMarginLoss — element-wise sigmoid
    /// log-likelihood. Mirrors <c>F.multi_label_soft_margin_loss</c>.</summary>
    public static Tensor<T> MultiLabelSoftMarginLoss<T>(Tensor<T> input, Tensor<T> target,
        Tensor<T>? weight = null, LossReduction reduction = LossReduction.Mean)
    {
        EnsureSameShape(input, target);
        var ops = MathHelper.GetNumericOperations<T>();
        int classes = input._shape[input.Rank - 1];
        var output = new Tensor<T>(DropLastDim(input._shape));
        var inSpan = input.AsSpan();
        var tSpan = target.AsSpan();
        var wSpan = weight is null ? ReadOnlySpan<T>.Empty : weight.AsSpan();
        var outSpan = output.AsWritableSpan();
        int rows = output.Length;
        for (int r = 0; r < rows; r++)
        {
            double rowLoss = 0;
            for (int c = 0; c < classes; c++)
            {
                int idx = r * classes + c;
                double xi = ops.ToDouble(inSpan[idx]);
                double ti = ops.ToDouble(tSpan[idx]);
                // log(1 + exp(x)) = softplus, numerically stable.
                double sp = xi >= 0 ? xi + Math.Log(1 + Math.Exp(-xi))
                                    : Math.Log(1 + Math.Exp(xi));
                double term = ti * (xi - sp) + (1 - ti) * (-sp);
                if (weight is not null) term *= ops.ToDouble(wSpan[c]);
                rowLoss -= term;
            }
            outSpan[r] = ops.FromDouble(rowLoss / classes);
        }
        return Reduce(output, reduction);
    }

    /// <summary>Triplet margin loss with L2 distance —
    /// <c>max(0, ||a − p||₂ − ||a − n||₂ + margin)</c>.
    /// Mirrors <c>F.triplet_margin_loss</c>.</summary>
    public static Tensor<T> TripletMarginLoss<T>(
        Tensor<T> anchor, Tensor<T> positive, Tensor<T> negative,
        double margin = 1.0, double p = 2.0, double eps = 1e-6,
        LossReduction reduction = LossReduction.Mean)
    {
        EnsureSameShape(anchor, positive);
        EnsureSameShape(anchor, negative);
        if (p <= 0)
            throw new ArgumentOutOfRangeException(nameof(p), "p must be > 0.");
        var ops = MathHelper.GetNumericOperations<T>();
        int last = anchor._shape[anchor.Rank - 1];
        int rows = anchor.Length / last;
        var output = new Tensor<T>(DropLastDim(anchor._shape));
        var aSpan = anchor.AsSpan();
        var pSpan = positive.AsSpan();
        var nSpan = negative.AsSpan();
        var outSpan = output.AsWritableSpan();
        for (int r = 0; r < rows; r++)
        {
            double dPos = 0, dNeg = 0;
            for (int c = 0; c < last; c++)
            {
                double a = ops.ToDouble(aSpan[r * last + c]);
                double pp = ops.ToDouble(pSpan[r * last + c]);
                double n = ops.ToDouble(nSpan[r * last + c]);
                // Standard Lp distance: sum |a-p|^p, then root.
                // Adding eps INSIDE the |·| would inflate negative
                // deltas — eps is a numerical-stability nudge applied
                // AFTER the per-element power so the gradient stays
                // finite at zero.
                dPos += Math.Pow(Math.Abs(a - pp), p);
                dNeg += Math.Pow(Math.Abs(a - n), p);
            }
            double dPosNorm = Math.Pow(dPos + eps, 1.0 / p);
            double dNegNorm = Math.Pow(dNeg + eps, 1.0 / p);
            double loss = Math.Max(0.0, dPosNorm - dNegNorm + margin);
            outSpan[r] = ops.FromDouble(loss);
        }
        return Reduce(output, reduction);
    }

    /// <summary>Triplet margin loss with a custom distance function.
    /// Mirrors <c>F.triplet_margin_with_distance_loss</c>.</summary>
    public static Tensor<T> TripletMarginWithDistanceLoss<T>(
        Tensor<T> anchor, Tensor<T> positive, Tensor<T> negative,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> distanceFn,
        double margin = 1.0, bool swap = false,
        LossReduction reduction = LossReduction.Mean)
    {
        if (distanceFn is null) throw new ArgumentNullException(nameof(distanceFn));
        var dPos = distanceFn(anchor, positive);
        var dNeg = distanceFn(anchor, negative);
        // Reject scalar / mismatched-shape distance fn outputs up
        // front instead of letting them truncate or read past spans
        // in the swap merge / loss loop.
        EnsureSameShape(dPos, dNeg);
        if (swap)
        {
            // PyTorch's "swap" replaces dNeg with min(dNeg, distance(positive, negative)).
            var dPosNeg = distanceFn(positive, negative);
            EnsureSameShape(dNeg, dPosNeg);
            var ops = MathHelper.GetNumericOperations<T>();
            var swapped = new Tensor<T>((int[])dNeg._shape.Clone());
            var srcN = dNeg.AsSpan();
            var srcPn = dPosNeg.AsSpan();
            var dst = swapped.AsWritableSpan();
            for (int i = 0; i < srcN.Length; i++)
                dst[i] = ops.LessThan(srcPn[i], srcN[i]) ? srcPn[i] : srcN[i];
            dNeg = swapped;
        }
        var loss = new Tensor<T>((int[])dPos._shape.Clone());
        var ops2 = MathHelper.GetNumericOperations<T>();
        var dPosSpan = dPos.AsSpan();
        var dNegSpan = dNeg.AsSpan();
        var outSpan = loss.AsWritableSpan();
        for (int i = 0; i < outSpan.Length; i++)
            outSpan[i] = ops2.FromDouble(Math.Max(0.0,
                ops2.ToDouble(dPosSpan[i]) - ops2.ToDouble(dNegSpan[i]) + margin));
        return Reduce(loss, reduction);
    }

    /// <summary>Margin ranking loss — <c>max(0, −y · (x1 − x2) + margin)</c>
    /// where <paramref name="y"/> is +1 or -1.
    /// Mirrors <c>F.margin_ranking_loss</c>.</summary>
    public static Tensor<T> MarginRankingLoss<T>(Tensor<T> input1, Tensor<T> input2, Tensor<T> y,
        double margin = 0.0, LossReduction reduction = LossReduction.Mean)
    {
        EnsureSameShape(input1, input2);
        EnsureSameShape(input1, y);
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input1._shape.Clone());
        var s1 = input1.AsSpan();
        var s2 = input2.AsSpan();
        var sy = y.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < s1.Length; i++)
        {
            double a = ops.ToDouble(s1[i]);
            double b = ops.ToDouble(s2[i]);
            double yy = ops.ToDouble(sy[i]);
            dst[i] = ops.FromDouble(Math.Max(0.0, -yy * (a - b) + margin));
        }
        return Reduce(output, reduction);
    }

    /// <summary>Cosine embedding loss with margin.
    /// <c>1 − cos(x1, x2)</c> when <paramref name="y"/> = +1;
    /// <c>max(0, cos(x1, x2) − margin)</c> when <paramref name="y"/> = -1.</summary>
    public static Tensor<T> CosineEmbeddingLoss<T>(Tensor<T> x1, Tensor<T> x2, Tensor<int> y,
        double margin = 0.0, LossReduction reduction = LossReduction.Mean)
    {
        EnsureSameShape(x1, x2);
        if (x1.Rank != 2)
            throw new ArgumentException("CosineEmbeddingLoss expects rank-2 inputs.", nameof(x1));
        var ops = MathHelper.GetNumericOperations<T>();
        int batch = x1._shape[0], features = x1._shape[1];
        if (y.Rank != 1 || y.Length != batch)
            throw new ArgumentException(
                "y must be rank-1 with one label per input row.", nameof(y));
        var output = new Tensor<T>(new[] { batch });
        var s1 = x1.AsSpan();
        var s2 = x2.AsSpan();
        var ySpan = y.AsSpan();
        var dst = output.AsWritableSpan();
        for (int b = 0; b < batch; b++)
        {
            int yi = ySpan[b];
            if (yi != 1 && yi != -1)
                throw new ArgumentException(
                    $"y values must be +1 or -1; got {yi} at index {b}.", nameof(y));
            double dot = 0, n1 = 0, n2 = 0;
            for (int f = 0; f < features; f++)
            {
                double a = ops.ToDouble(s1[b * features + f]);
                double bb = ops.ToDouble(s2[b * features + f]);
                dot += a * bb;
                n1 += a * a;
                n2 += bb * bb;
            }
            double cos = dot / Math.Max(1e-12, Math.Sqrt(n1 * n2));
            double loss = yi == 1 ? 1.0 - cos : Math.Max(0.0, cos - margin);
            dst[b] = ops.FromDouble(loss);
        }
        return Reduce(output, reduction);
    }

    /// <summary>Hinge embedding loss — <c>x</c> when y=+1, otherwise
    /// <c>max(0, margin − x)</c>. Mirrors <c>F.hinge_embedding_loss</c>.</summary>
    public static Tensor<T> HingeEmbeddingLoss<T>(Tensor<T> input, Tensor<int> y,
        double margin = 1.0, LossReduction reduction = LossReduction.Mean)
    {
        if (input.Length != y.Length)
            throw new ArgumentException("input and y must have the same length.");
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var ySpan = y.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double xi = ops.ToDouble(src[i]);
            double loss = ySpan[i] == 1 ? xi : Math.Max(0.0, margin - xi);
            dst[i] = ops.FromDouble(loss);
        }
        return Reduce(output, reduction);
    }

    /// <summary>Kullback-Leibler divergence with optional <paramref name="logTarget"/>.
    /// <c>log_target=true</c> means <paramref name="target"/> is in log-space —
    /// the variant PyTorch added to <c>F.kl_div</c> for stable training.</summary>
    public static Tensor<T> KLDiv<T>(Tensor<T> input, Tensor<T> target,
        bool logTarget = false, LossReduction reduction = LossReduction.Mean)
    {
        EnsureSameShape(input, target);
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>((int[])input._shape.Clone());
        var src = input.AsSpan();
        var tgt = target.AsSpan();
        var dst = output.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double inputI = ops.ToDouble(src[i]);
            double targetI = ops.ToDouble(tgt[i]);
            double term = logTarget
                ? Math.Exp(targetI) * (targetI - inputI)
                : targetI * (Math.Log(Math.Max(1e-30, targetI)) - inputI);
            dst[i] = ops.FromDouble(term);
        }
        return Reduce(output, reduction);
    }

    private static Tensor<T> Reduce<T>(Tensor<T> input, LossReduction reduction)
    {
        if (reduction == LossReduction.None) return input;
        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        T acc = ops.Zero;
        for (int i = 0; i < src.Length; i++) acc = ops.Add(acc, src[i]);
        if (reduction == LossReduction.Mean)
            acc = ops.Divide(acc, ops.FromDouble(src.Length));
        var result = new Tensor<T>(new[] { 1 });
        result.AsWritableSpan()[0] = acc;
        return result;
    }

    private static int[] DropLastDim(int[] shape)
    {
        if (shape.Length <= 1) return new[] { 1 };
        var result = new int[shape.Length - 1];
        Array.Copy(shape, result, shape.Length - 1);
        return result;
    }

    private static void EnsureSameShape<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length || a.Rank != b.Rank)
            throw new ArgumentException("Shape mismatch between inputs.");
        for (int i = 0; i < a.Rank; i++)
            if (a._shape[i] != b._shape[i])
                throw new ArgumentException($"Shape mismatch on axis {i}: {a._shape[i]} vs {b._shape[i]}.");
    }
}
