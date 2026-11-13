using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The blank-extended CTC label lattice shared by <see cref="CpuEngine.TensorCTCLoss{T}"/>, its backward,
/// <see cref="CpuEngine.TensorCTCForwardBackward{T}"/> and <see cref="CpuEngine.TensorCTCLabelPosteriors{T}"/>: one
/// validation, one alpha recursion and one beta recursion, so the loss, its gradient and the exposed lattice
/// variables cannot drift apart.
/// </summary>
/// <remarks>
/// Lattice position s of item n holds the blank for even s and target label s / 2 for odd s (S_n = 2 U_n + 1
/// positions). The recursions run in double in the log domain (Graves et al. 2006, eq. 6-10), exactly as the loss
/// always has: <c>alpha[t, s] = lse(alpha[t-1, s], alpha[t-1, s-1], alpha[t-1, s-2] if skippable) + lp[t, l_s]</c> and
/// <c>beta[t, s] = lse over s' in {s, s+1, s+2 if skippable} of beta[t+1, s'] + lp[t+1, l_s']</c>.
/// </remarks>
internal static class CtcLattice
{
    /// <summary>A validated CTC problem: the dimensions, the targets as ints and each item's target offset.</summary>
    internal sealed class Problem
    {
        internal int MaxTime, Batch, Classes, Blank, MaxTargetLength, TotalTargets;
        internal int[] InputLengths = Array.Empty<int>();
        internal int[] TargetLengths = Array.Empty<int>();
        internal int[] Offsets = Array.Empty<int>();
        internal int[] Targets = Array.Empty<int>();

        /// <summary>Lattice positions of the longest item, the stride of the exposed alpha and beta rows.</summary>
        internal int MaxStates => 2 * MaxTargetLength + 1;

        /// <summary>The label at lattice position s of item n.</summary>
        internal int Label(int n, int s) => (s & 1) == 0 ? Blank : Targets[Offsets[n] + (s >> 1)];
    }

    /// <summary>Validates the CTC arguments (the messages TensorCTCLoss has always thrown) and packs them.</summary>
    internal static Problem Prepare<T>(Tensor<T> logProbs, Tensor<int> targets, int[] inputLengths, int[] targetLengths, int blank)
    {
        if (logProbs is null) throw new ArgumentNullException(nameof(logProbs));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (inputLengths is null) throw new ArgumentNullException(nameof(inputLengths));
        if (targetLengths is null) throw new ArgumentNullException(nameof(targetLengths));
        if (logProbs.Rank != 3)
            throw new ArgumentException("logProbs must be 3D [T, N, C].");
        var p = new Problem
        {
            MaxTime = logProbs._shape[0],
            Batch = logProbs._shape[1],
            Classes = logProbs._shape[2],
            Blank = blank,
            InputLengths = inputLengths,
            TargetLengths = targetLengths,
        };
        if (inputLengths.Length != p.Batch || targetLengths.Length != p.Batch)
            throw new ArgumentException("inputLengths and targetLengths must have length == batchSize.");
        if (blank < 0 || blank >= p.Classes)
            throw new ArgumentOutOfRangeException(nameof(blank), $"Blank index {blank} must be in [0, {p.Classes}).");

        p.Offsets = new int[p.Batch];
        int total = 0;
        for (int n = 0; n < p.Batch; n++)
        {
            int tn = inputLengths[n], un = targetLengths[n];
            if (tn < 0 || tn > p.MaxTime)
                throw new ArgumentOutOfRangeException(nameof(inputLengths), $"inputLengths[{n}]={tn} must be in [0, {p.MaxTime}].");
            if (un < 0)
                throw new ArgumentOutOfRangeException(nameof(targetLengths), $"targetLengths[{n}]={un} must be non-negative.");
            p.Offsets[n] = total;
            total = checked(total + un);
            p.MaxTargetLength = Math.Max(p.MaxTargetLength, un);
        }
        if (total > targets.Length)
            throw new ArgumentException($"Sum of targetLengths ({total}) exceeds targets length ({targets.Length}).");
        p.TotalTargets = total;

        p.Targets = new int[total];
        using (var lease = targets.Lease())
            lease.Span.Slice(0, total).CopyTo(p.Targets);
        foreach (int label in p.Targets)
        {
            if (label < 0 || label >= p.Classes)
                throw new ArgumentOutOfRangeException(nameof(targets), $"Target label {label} must be in [0, {p.Classes}).");
            // A blank target would interleave blank with blank, and the lattice would count duplicate paths.
            if (label == p.Blank)
                throw new ArgumentException($"Target labels must exclude the blank index {p.Blank}.", nameof(targets));
        }
        return p;
    }

    /// <summary>log(exp(a) + exp(b)), exact when either side is negative infinity.</summary>
    internal static double LogSumExp(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;
        double max = Math.Max(a, b);
        return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
    }

    // lp is logProbs [T, N, C] flattened row-major.
    private static double Emission(double[] lp, Problem p, int t, int n, int label) => lp[(t * p.Batch + n) * p.Classes + label];

    /// <summary>
    /// The alpha recursion of item n into <paramref name="alpha"/> (row t at <c>t * stride</c>, positions
    /// 0 .. S_n - 1 of rows 0 .. T_n - 1 written; the caller pre-fills the rest with negative infinity when it exposes
    /// them). Returns log P(target_n | input_n).
    /// </summary>
    internal static double Alpha(double[] lp, Problem p, int n, Span<double> alpha, int stride)
    {
        int tn = p.InputLengths[n], states = 2 * p.TargetLengths[n] + 1;
        if (tn == 0) return states == 1 ? 0.0 : double.NegativeInfinity;   // the empty alignment
        double negInf = double.NegativeInfinity;
        for (int s = 0; s < states; s++) alpha[s] = negInf;
        alpha[0] = Emission(lp, p, 0, n, p.Label(n, 0));
        if (states > 1) alpha[1] = Emission(lp, p, 0, n, p.Label(n, 1));
        for (int t = 1; t < tn; t++)
        {
            int row = t * stride, prev = row - stride;
            for (int s = 0; s < states; s++)
            {
                int label = p.Label(n, s);
                double sum = alpha[prev + s];
                if (s >= 1) sum = LogSumExp(sum, alpha[prev + s - 1]);
                if (s >= 2 && label != p.Blank && label != p.Label(n, s - 2))
                    sum = LogSumExp(sum, alpha[prev + s - 2]);
                alpha[row + s] = sum + Emission(lp, p, t, n, label);
            }
        }
        int last = (tn - 1) * stride;
        double logP = alpha[last + states - 1];
        if (states >= 2) logP = LogSumExp(logP, alpha[last + states - 2]);
        return logP;
    }

    /// <summary>
    /// The beta recursion of item n into <paramref name="beta"/> (the layout of <see cref="Alpha"/>): the emissions
    /// after t only, so the last step's final blank and final label are zero.
    /// </summary>
    internal static void Beta(double[] lp, Problem p, int n, Span<double> beta, int stride)
    {
        int tn = p.InputLengths[n], states = 2 * p.TargetLengths[n] + 1;
        if (tn == 0) return;
        double negInf = double.NegativeInfinity;
        int lastRow = (tn - 1) * stride;
        for (int s = 0; s < states; s++) beta[lastRow + s] = negInf;
        beta[lastRow + states - 1] = 0;
        if (states >= 2) beta[lastRow + states - 2] = 0;
        for (int t = tn - 2; t >= 0; t--)
        {
            int row = t * stride, next = row + stride;
            for (int s = states - 1; s >= 0; s--)
            {
                int label = p.Label(n, s);
                double sum = beta[next + s] + Emission(lp, p, t + 1, n, label);
                if (s + 1 < states)
                    sum = LogSumExp(sum, beta[next + s + 1] + Emission(lp, p, t + 1, n, p.Label(n, s + 1)));
                if (s + 2 < states)
                {
                    int skip = p.Label(n, s + 2);
                    if (skip != p.Blank && skip != label)
                        sum = LogSumExp(sum, beta[next + s + 2] + Emission(lp, p, t + 1, n, skip));
                }
                beta[row + s] = sum;
            }
        }
    }

    /// <summary>
    /// The label posteriors of item n at time t into <paramref name="gamma"/> (its C classes, zeroed first):
    /// <c>sum over s labelled c of exp(alpha + beta - logP)</c> in ascending s, NaN everywhere when logP is negative
    /// infinity (no alignment, an infinite loss).
    /// </summary>
    internal static void Posteriors(Problem p, int n, int t, ReadOnlySpan<double> alphaRow, ReadOnlySpan<double> betaRow,
        double logP, Span<double> gamma)
    {
        gamma.Clear();
        if (t >= p.InputLengths[n]) return;
        if (double.IsNegativeInfinity(logP))
        {
            gamma.Fill(double.NaN);
            return;
        }
        int states = 2 * p.TargetLengths[n] + 1;
        for (int s = 0; s < states; s++)
        {
            double ab = alphaRow[s] + betaRow[s];
            if (!double.IsNegativeInfinity(ab)) gamma[p.Label(n, s)] += Math.Exp(ab - logP);
        }
    }
}

public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual (Tensor<T> LogAlpha, Tensor<T> LogBeta) TensorCTCForwardBackward<T>(Tensor<T> logProbs, Tensor<int> targets,
        int[] inputLengths, int[] targetLengths, int blank = 0)
    {
        GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        var p = CtcLattice.Prepare(logProbs, targets, inputLengths, targetLengths, blank);
        var (alpha, beta, _) = CtcLatticeHost(logProbs, p);
        int[] shape = { p.Batch, p.MaxTime, p.MaxStates };
        return (RoIAlignPlan.FromDoubles<T>(alpha, shape), RoIAlignPlan.FromDoubles<T>(beta, shape));
    }

    /// <inheritdoc />
    public virtual Tensor<T> TensorCTCLabelPosteriors<T>(Tensor<T> logProbs, Tensor<int> targets, int[] inputLengths,
        int[] targetLengths, int blank = 0)
    {
        GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        var p = CtcLattice.Prepare(logProbs, targets, inputLengths, targetLengths, blank);
        return RoIAlignPlan.FromDoubles<T>(CtcPosteriorsHost(logProbs, p), new[] { p.MaxTime, p.Batch, p.Classes });
    }

    /// <summary>
    /// Alpha and beta of every item ([N, T, maxStates], negative infinity where unreachable) and each item's
    /// log-likelihood, items in parallel.
    /// </summary>
    internal static (double[] Alpha, double[] Beta, double[] LogLikelihood) CtcLatticeHost<T>(Tensor<T> logProbs, CtcLattice.Problem p)
    {
        var lp = RoIAlignPlan.ToDoubles(logProbs);
        int stride = p.MaxStates, plane = p.MaxTime * stride;
        var alpha = new double[checked(p.Batch * plane)];
        var beta = new double[alpha.Length];
        alpha.AsSpan().Fill(double.NegativeInfinity);
        beta.AsSpan().Fill(double.NegativeInfinity);
        var logLikelihood = new double[p.Batch];
        CpuParallelSettings.ParallelForOrSerial(0, p.Batch, (long)alpha.Length * 4, n =>
        {
            logLikelihood[n] = CtcLattice.Alpha(lp, p, n, alpha.AsSpan(n * plane, plane), stride);
            CtcLattice.Beta(lp, p, n, beta.AsSpan(n * plane, plane), stride);
        }, deterministicSafe: true);
        return (alpha, beta, logLikelihood);
    }

    /// <summary>gamma [T, N, C] of every item (see <see cref="CtcLattice.Posteriors"/>).</summary>
    internal static double[] CtcPosteriorsHost<T>(Tensor<T> logProbs, CtcLattice.Problem p)
    {
        var (alpha, beta, logLikelihood) = CtcLatticeHost(logProbs, p);
        int stride = p.MaxStates, plane = p.MaxTime * stride;
        var gamma = new double[checked(p.MaxTime * p.Batch * p.Classes)];
        CpuParallelSettings.ParallelForOrSerial(0, p.MaxTime * p.Batch, (long)gamma.Length + alpha.Length, idx =>
        {
            int t = idx / p.Batch, n = idx - t * p.Batch;
            int row = n * plane + t * stride;
            CtcLattice.Posteriors(p, n, t, alpha.AsSpan(row, stride), beta.AsSpan(row, stride), logLikelihood[n],
                gamma.AsSpan(idx * p.Classes, p.Classes));
        }, deterministicSafe: true);
        return gamma;
    }
}
