using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Einsum;

/// <summary>
/// Executes a planned einsum by walking the <see cref="EinsumPath"/> and
/// applying each pairwise contraction.
/// </summary>
/// <remarks>
/// <para>
/// The v1 implementation uses a correctness-first generic pairwise
/// contraction kernel (O(∏ merged-label sizes) per step). It handles
/// diagonals (repeated labels within one operand), partial reductions
/// (labels dropped before contraction), ellipsis broadcasting, and multi-way
/// contractions ordered by the greedy path optimiser.
/// </para>
/// <para>
/// A subsequent commit will add a BatchMatMul fast-path that routes
/// non-diagonal two-operand contractions through <c>TensorMatMul</c> /
/// <c>BatchMatMul</c> for speed.
/// </para>
/// </remarks>
public static class EinsumExecutor
{
    /// <summary>
    /// Evaluates the equation + path against the given operand tensors and
    /// returns the resulting tensor. Operand shapes must match
    /// <see cref="EinsumShapeBinding.OperandShapes"/>.
    /// </summary>
    public static Tensor<T> Execute<T>(
        EinsumShapeBinding binding,
        EinsumPath path,
        Tensor<T>[] operands)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        if (path is null) throw new ArgumentNullException(nameof(path));
        if (operands is null) throw new ArgumentNullException(nameof(operands));
        if (operands.Length != binding.Equation.Operands.Count)
            throw new ArgumentException(
                $"Expected {binding.Equation.Operands.Count} tensor(s), got {operands.Length}");

        var eq = binding.Equation;
        int batchRank = binding.BatchDims.Length;

        // 1. Expand each operand's ellipsis dims (if any) into synthetic
        //    digit-labeled batch dims, right-aligned with the global batch
        //    shape. Operands without ellipsis get leading size-1 dims when
        //    any other operand has ellipsis.
        var liveTensors = new List<Tensor<T>>(operands.Length);
        var liveLabels = new List<char[]>(operands.Length);
        for (int i = 0; i < operands.Length; i++)
        {
            var (t, lbls) = ExpandOperand(operands[i], eq.Operands[i], binding, batchRank);
            liveTensors.Add(t);
            liveLabels.Add(lbls);
        }

        // Output labels (with ellipsis expanded to synth digit labels).
        var outputLabels = ExpandEllipsis(eq.Output, batchRank).ToArray();

        // 2. Single-operand case: normalise to the output directly.
        if (liveTensors.Count == 1)
        {
            return NormalizeSingle(liveTensors[0], liveLabels[0], outputLabels, binding.LabelSizes, binding.BatchDims);
        }

        // 3. Multi-operand: walk path steps.
        for (int s = 0; s < path.Steps.Count; s++)
        {
            var step = path.Steps[s];
            var A = liveTensors[step.LeftIndex];
            var la = liveLabels[step.LeftIndex];
            var B = liveTensors[step.RightIndex];
            var lb = liveLabels[step.RightIndex];

            // Expand step.ResultLabels '@' marker to explicit digit labels.
            char[] stepResultLabels = ExpandMarker(step.ResultLabels, batchRank);

            // Reduce labels from A that appear only in A and not in lb/result,
            // and same for B. This keeps the pairwise kernel's iteration
            // volume minimal.
            char[] aKeep = FilterKeep(la, lb, stepResultLabels);
            char[] bKeep = FilterKeep(lb, la, stepResultLabels);

            var aNorm = NormalizeSingle(A, la, aKeep, binding.LabelSizes, binding.BatchDims);
            var bNorm = NormalizeSingle(B, lb, bKeep, binding.LabelSizes, binding.BatchDims);

            var C = PairwiseContract(aNorm, aKeep, bNorm, bKeep, stepResultLabels, binding.LabelSizes, binding.BatchDims);

            // Replace in live lists: remove two, append one.
            int hi = Math.Max(step.LeftIndex, step.RightIndex);
            int lo = Math.Min(step.LeftIndex, step.RightIndex);
            liveTensors.RemoveAt(hi);
            liveTensors.RemoveAt(lo);
            liveLabels.RemoveAt(hi);
            liveLabels.RemoveAt(lo);

            liveTensors.Add(C);
            liveLabels.Add(stepResultLabels);
        }

        // 4. The last live tensor carries the result. Its labels may need to
        //    be permuted to match the final outputLabels order.
        var finalTensor = liveTensors[0];
        var finalLabels = liveLabels[0];
        if (!finalLabels.SequenceEqual(outputLabels))
        {
            finalTensor = NormalizeSingle(finalTensor, finalLabels, outputLabels, binding.LabelSizes, binding.BatchDims);
        }
        return finalTensor;
    }

    // -- Ellipsis expansion --------------------------------------------------

    private static (Tensor<T>, char[]) ExpandOperand<T>(
        Tensor<T> operand,
        OperandLabels opLabels,
        EinsumShapeBinding binding,
        int batchRank)
    {
        // If the equation has no ellipsis at all, operand is unchanged and
        // labels are as-is.
        if (!binding.Equation.HasEllipsis)
        {
            return (operand, opLabels.Labels.ToArray());
        }

        int localEllipsisCount = opLabels.HasEllipsis
            ? operand.Rank - opLabels.MinimumRank
            : 0;
        int padLeadingOnes = batchRank - localEllipsisCount;

        // Assemble new shape: before-ellipsis dims + size-1 pads + local
        // ellipsis dims + after-ellipsis dims.
        var newShape = new List<int>(operand.Rank + padLeadingOnes);
        var newLabels = new List<char>(opLabels.Labels.Count + batchRank);

        int cursor = 0;
        // Before-ellipsis explicit dims.
        for (int i = 0; i < opLabels.CountBeforeEllipsis; i++)
        {
            newShape.Add(operand.Shape[cursor]);
            newLabels.Add(opLabels.Labels[i]);
            cursor++;
        }

        // Ellipsis region: padLeadingOnes size-1 dims, then the operand's
        // local ellipsis dims.
        for (int i = 0; i < padLeadingOnes; i++)
        {
            newShape.Add(1);
            newLabels.Add((char)('0' + i));
        }
        for (int i = 0; i < localEllipsisCount; i++)
        {
            newShape.Add(operand.Shape[cursor]);
            newLabels.Add((char)('0' + padLeadingOnes + i));
            cursor++;
        }

        // After-ellipsis explicit dims.
        int afterStart = opLabels.Labels.Count - opLabels.CountAfterEllipsis;
        for (int i = 0; i < opLabels.CountAfterEllipsis; i++)
        {
            newShape.Add(operand.Shape[cursor]);
            newLabels.Add(opLabels.Labels[afterStart + i]);
            cursor++;
        }

        // If operand has no ellipsis but equation does, the above adds the
        // size-1 synth batch labels at position 0 (since CountBeforeEllipsis = 0
        // and padLeadingOnes = batchRank). That's the intended behaviour.

        // Reshape is O(1) for contiguous tensors (a stride-view rewrite), so
        // we don't bother short-circuiting the equal-shape case.
        var shapeArr = newShape.ToArray();
        var reshaped = operand.Reshape(shapeArr);
        return (reshaped, newLabels.ToArray());
    }

    private static IEnumerable<char> ExpandEllipsis(OperandLabels labels, int batchRank)
    {
        if (!labels.HasEllipsis)
        {
            foreach (var c in labels.Labels) yield return c;
            yield break;
        }
        for (int i = 0; i < labels.EllipsisPosition; i++) yield return labels.Labels[i];
        for (int i = 0; i < batchRank; i++) yield return (char)('0' + i);
        for (int i = labels.EllipsisPosition; i < labels.Labels.Count; i++) yield return labels.Labels[i];
    }

    private static char[] ExpandMarker(IReadOnlyList<char> labels, int batchRank)
    {
        // The path optimiser uses '@' as a single marker for the batch block.
        // Expand it to the synth digit labels.
        var result = new List<char>(labels.Count + batchRank);
        foreach (var c in labels)
        {
            if (c == EinsumPathOptimizer.EllipsisMarker)
            {
                for (int i = 0; i < batchRank; i++) result.Add((char)('0' + i));
            }
            else
            {
                result.Add(c);
            }
        }
        return result.ToArray();
    }

    private static char[] FilterKeep(char[] selfLabels, char[] otherLabels, char[] outLabels)
    {
        // Keep any label of self that also appears in (other ∪ out).
        var keepSet = new HashSet<char>(otherLabels);
        foreach (var c in outLabels) keepSet.Add(c);
        var seen = new HashSet<char>();
        var ordered = new List<char>(selfLabels.Length);
        foreach (var c in selfLabels)
        {
            if (keepSet.Contains(c) && seen.Add(c)) ordered.Add(c);
        }
        return ordered.ToArray();
    }

    // -- Single-operand normalisation ---------------------------------------

    /// <summary>
    /// Produces a new tensor whose label sequence is exactly <paramref name="targetLabels"/>.
    /// Handles diagonals (repeated labels collapse), reductions (labels not
    /// in target are summed out), and arbitrary permutations.
    /// </summary>
    private static Tensor<T> NormalizeSingle<T>(
        Tensor<T> source,
        char[] sourceLabels,
        char[] targetLabels,
        IReadOnlyDictionary<char, int> labelSizes,
        int[] batchDims)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Size per label (for batch synth labels, look up batchDims).
        int SizeOf(char c) => c >= '0' && c <= '9'
            ? batchDims[c - '0']
            : labelSizes[c];

        // Source label → list of source-dim positions (multiple if diagonal).
        var srcPositions = new Dictionary<char, List<int>>();
        for (int i = 0; i < sourceLabels.Length; i++)
        {
            var c = sourceLabels[i];
            if (!srcPositions.TryGetValue(c, out var list))
            {
                list = new List<int>();
                srcPositions[c] = list;
            }
            list.Add(i);
        }

        // Target shape and label→position lookup.
        var targetShape = new int[targetLabels.Length];
        var targetLabelPos = new Dictionary<char, int>();
        for (int i = 0; i < targetLabels.Length; i++)
        {
            targetShape[i] = SizeOf(targetLabels[i]);
            targetLabelPos[targetLabels[i]] = i;
        }

        var result = new Tensor<T>(targetShape);
        // Fill with zero (Tensor<T> default-initialises; no-op for numeric
        // default-zero T, but we still want an explicit zero for unmanaged
        // types via the numeric ops to avoid relying on default(T)).
        FillZero(result, numOps);

        // Distinct source labels, in sourceLabels order.
        var distinctSourceLabels = new List<char>();
        var distinctSeen = new HashSet<char>();
        foreach (var c in sourceLabels)
            if (distinctSeen.Add(c)) distinctSourceLabels.Add(c);

        // Distinct label sizes (for outer iteration).
        var distinctSizes = distinctSourceLabels.Select(SizeOf).ToArray();

        // Source shape (for bounds).
        var sourceShape = source.Shape;

        // Iterate over all distinct source labels' index tuples.
        var iter = new int[distinctSizes.Length];
        var srcIdx = new int[sourceLabels.Length];
        var tgtIdx = new int[targetLabels.Length];
        while (true)
        {
            // Build srcIdx from iter: for each source position, look up the
            // iter value for its label.
            for (int i = 0; i < sourceLabels.Length; i++)
            {
                int labelPos = distinctSourceLabels.IndexOf(sourceLabels[i]);
                int idx = iter[labelPos];
                // Broadcast: if the actual source dim is 1, force index 0.
                if (sourceShape[i] == 1) idx = 0;
                srcIdx[i] = idx;
            }

            // Build tgtIdx from iter via targetLabelPos.
            bool skip = false;
            for (int i = 0; i < targetLabels.Length; i++)
            {
                int labelPos = distinctSourceLabels.IndexOf(targetLabels[i]);
                if (labelPos < 0)
                {
                    // Target label absent from source: can't happen if target
                    // is a subset of distinct source labels. Skip defensively.
                    skip = true;
                    break;
                }
                tgtIdx[i] = iter[labelPos];
            }

            if (!skip)
            {
                var v = source[srcIdx];
                var cur = result[tgtIdx];
                result[tgtIdx] = numOps.Add(cur, v);
            }

            // Increment iter (lexicographic over distinctSizes).
            if (!AdvanceIndex(iter, distinctSizes)) break;
        }

        return result;
    }

    // -- Pairwise contraction ----------------------------------------------

    private static Tensor<T> PairwiseContract<T>(
        Tensor<T> A, char[] la,
        Tensor<T> B, char[] lb,
        char[] targetLabels,
        IReadOnlyDictionary<char, int> labelSizes,
        int[] batchDims)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int SizeOf(char c) => c >= '0' && c <= '9'
            ? batchDims[c - '0']
            : labelSizes[c];

        // Union of all labels involved in this pairwise contraction.
        var allLabels = new List<char>();
        var allLabelsSeen = new HashSet<char>();
        foreach (var c in la) if (allLabelsSeen.Add(c)) allLabels.Add(c);
        foreach (var c in lb) if (allLabelsSeen.Add(c)) allLabels.Add(c);
        foreach (var c in targetLabels) if (allLabelsSeen.Add(c)) allLabels.Add(c);

        var allSizes = allLabels.Select(SizeOf).ToArray();

        // Per-operand label → source dim position.
        var aPos = BuildPositions(la);
        var bPos = BuildPositions(lb);

        // Target shape and dim-position lookup.
        var targetShape = new int[targetLabels.Length];
        var targetPos = new Dictionary<char, int>();
        for (int i = 0; i < targetLabels.Length; i++)
        {
            targetShape[i] = SizeOf(targetLabels[i]);
            targetPos[targetLabels[i]] = i;
        }
        var result = new Tensor<T>(targetShape);
        FillZero(result, numOps);

        var aShape = A.Shape;
        var bShape = B.Shape;

        var iter = new int[allLabels.Count];
        var aIdx = new int[la.Length];
        var bIdx = new int[lb.Length];
        var tIdx = new int[targetLabels.Length];

        while (true)
        {
            // Assemble A index.
            for (int i = 0; i < la.Length; i++)
            {
                int labelPos = allLabels.IndexOf(la[i]);
                int idx = iter[labelPos];
                if (aShape[i] == 1) idx = 0;
                aIdx[i] = idx;
            }
            // Assemble B index.
            for (int i = 0; i < lb.Length; i++)
            {
                int labelPos = allLabels.IndexOf(lb[i]);
                int idx = iter[labelPos];
                if (bShape[i] == 1) idx = 0;
                bIdx[i] = idx;
            }
            // Assemble target index.
            for (int i = 0; i < targetLabels.Length; i++)
            {
                int labelPos = allLabels.IndexOf(targetLabels[i]);
                tIdx[i] = iter[labelPos];
            }

            var a = la.Length == 0 ? numOps.One : A[aIdx];
            var b = lb.Length == 0 ? numOps.One : B[bIdx];
            var prod = numOps.Multiply(a, b);
            var cur = result[tIdx];
            result[tIdx] = numOps.Add(cur, prod);

            if (!AdvanceIndex(iter, allSizes)) break;
        }

        return result;
    }

    // -- Helpers -----------------------------------------------------------

    private static Dictionary<char, List<int>> BuildPositions(char[] labels)
    {
        var d = new Dictionary<char, List<int>>();
        for (int i = 0; i < labels.Length; i++)
        {
            if (!d.TryGetValue(labels[i], out var list))
            {
                list = new List<int>();
                d[labels[i]] = list;
            }
            list.Add(i);
        }
        return d;
    }

    private static bool AdvanceIndex(int[] iter, int[] sizes)
    {
        // Lexicographic increment over `sizes`; return false if we wrapped
        // past the end of the iteration.
        if (iter.Length == 0) return false;
        int k = iter.Length - 1;
        while (k >= 0)
        {
            iter[k]++;
            if (iter[k] < sizes[k]) return true;
            iter[k] = 0;
            k--;
        }
        return false;
    }

    private static void FillZero<T>(Tensor<T> t, INumericOperations<T> numOps)
    {
        var zero = numOps.Zero;
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = zero;
    }
}
