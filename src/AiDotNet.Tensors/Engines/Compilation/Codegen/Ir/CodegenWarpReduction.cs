// Copyright (c) AiDotNet. All rights reserved.
// Which reductions run one WARP per output, and the lane-interleaved spec that lowering emits.
//
// One thread per output walks its own row, so the 32 threads of a warp read 32 different rows
// at once: 32 sectors per request where 4 would do. On a GTX 1660 Ti an 8192x1024 row softmax
// took 2.34 ms that way and 0.54 ms with a warp per row. The rule lives in the IR, not the PTX
// emitter, because the split planner has to know about it too: a reduction lowered as a warp
// per output launches 32 threads per output, and a planner counting one would split kernels
// that already fill the device.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>The warp-per-output reduction lowering's eligibility rule and spec rewrite.</summary>
public static class CodegenWarpReduction
{
    /// <summary>Lanes per warp, and so the interleave factor of the contiguous axis.</summary>
    public const int Lanes = 32;

    /// <summary>Smallest reduction worth a warp: two trips per lane.</summary>
    public const long MinimumTrips = 2L * Lanes;

    /// <summary>
    /// Whether <paramref name="spec"/> is lowered as one warp per output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Eligible means: a real-valued sum or max with no argmax and no indirection; at least
    /// <see cref="MinimumTrips"/> reduction trips; the first product operand's stride-1
    /// dimension indexed by a reduction axis with coefficient one and an extent divisible by
    /// <see cref="Lanes"/>; and no cross-output reuse worth a tile -- the lead operand reads
    /// every parallel axis and any operand that skips one is a broadcast no larger than the
    /// reduction.
    /// </para>
    /// <para>
    /// The last condition is what keeps matmul out. A[m,k] does not read n, so one load of it
    /// can serve a tile of outputs, and the affine emitter's tiling exploits exactly that. A
    /// warp per output gives that reuse up, which is a good trade for a row statistic (there
    /// is no reuse to lose) and a bad one for a contraction.
    /// </para>
    /// </remarks>
    public static bool IsEligible(CodegenKernelSpec spec) => ContiguousAxis(spec) >= 0;

    /// <summary>
    /// The lane-interleaved form of an eligible spec: the contiguous reduction axis r shrinks
    /// to extent/32, and a new LAST parallel axis <c>lane</c> of extent 32 supplies the low
    /// part of its index, <c>r_original = r * 32 + lane</c>. Every lane computes a partial
    /// over its residue class; the lowering combines them. The rewritten spec is NOT the
    /// original operator on its own -- interpreted, it would store 32 partials to one
    /// address -- so it exists only inside an emission, and the oracle stays the original.
    /// </summary>
    public static bool TryLaneInterleave(CodegenKernelSpec spec, out CodegenKernelSpec laneSpec)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        laneSpec = spec;
        int axis = ContiguousAxis(spec);
        if (axis < 0) return false;

        int extent = spec.Space.Axes[axis].Extent;
        var axes = new CodegenAxis[spec.Space.Axes.Count + 1];
        for (int a = 0; a < spec.Space.Axes.Count; a++)
            axes[a] = a == axis
                ? CodegenAxis.Reduce(spec.Space.Axes[a].Name, extent / Lanes)
                : spec.Space.Axes[a];
        int lane = spec.Space.Axes.Count;
        axes[lane] = CodegenAxis.Parallel(spec.Space.Axes[axis].Name + "_lane", Lanes);

        var inputs = new CodegenTensorBinding[spec.Inputs.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            var b = spec.Inputs[i];
            if (!References(b, axis)) { inputs[i] = b; continue; }
            var map = new CodegenAffineExpr[b.Map.Count];
            for (int d = 0; d < map.Length; d++)
            {
                var expr = b.Map[d];
                var terms = new List<CodegenAffineTerm>(expr.Terms.Count + 1);
                int laneCoefficient = 0;
                foreach (var term in expr.Terms)
                {
                    if (term.Axis == axis)
                    {
                        terms.Add(new CodegenAffineTerm(axis, term.Coefficient * Lanes));
                        laneCoefficient = term.Coefficient;
                    }
                    else
                    {
                        terms.Add(term);
                    }
                }
                if (laneCoefficient != 0) terms.Add(new CodegenAffineTerm(lane, laneCoefficient));
                map[d] = laneCoefficient == 0
                    ? expr
                    : new CodegenAffineExpr(terms.ToArray(), expr.Constant, expr.Divisor, expr.RequiresExactDivision);
            }
            inputs[i] = new CodegenTensorBinding(
                b.ParameterIndex, b.Name, ToArray(b.Shape), map, b.IsOutput, b.ElementType);
        }

        var extras = new CodegenExtraOutput[spec.ExtraOutputs.Count];
        for (int e = 0; e < extras.Length; e++) extras[e] = spec.ExtraOutputs[e];

        laneSpec = new CodegenKernelSpec(
            spec.Name, new CodegenIterationSpace(axes), inputs, spec.Output,
            ToArray(spec.ProductInputs), spec.Reduce,
            biasInput: spec.BiasInput, scaleInput: spec.ScaleInput, activation: spec.Activation,
            reduceScale: spec.ReduceScale, preReduce: spec.PreReduce,
            preBiasInput: spec.PreBiasInput, preBiasScale: spec.PreBiasScale,
            extraOutputs: extras,
            reduceOffset: spec.ReduceOffset,
            postScaleInput: spec.PostScaleInput, postBiasInput: spec.PostBiasInput,
            body: spec.Body);
        return true;
    }

    /// <summary>The reduction axis the warp interleaves, or -1 when the spec is not eligible.</summary>
    private static int ContiguousAxis(CodegenKernelSpec spec)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (spec.Algebra != CodegenAlgebra.Real) return -1;
        if (spec.Reduce is not (CodegenReduceKind.Sum or CodegenReduceKind.Max)) return -1;
        if (spec.SecondaryOutput is not null || spec.SecondaryIndexExpr is not null) return -1;
        foreach (var extra in spec.ExtraOutputs)
            if (extra.Kind == CodegenExtraOutputKind.ArgMaxIndex || extra.Binding.HasIndirection) return -1;
        if (spec.Output.HasIndirection || spec.Output.NeedsAtomicStore) return -1;
        foreach (var input in spec.Inputs) if (input.HasIndirection) return -1;
        if (spec.Space.ReductionTripCount < MinimumTrips) return -1;
        if (spec.Body is not null && spec.Reduce == CodegenReduceKind.Max && spec.ProductInputs.Count > 1) return -1;

        var lead = spec.Inputs[spec.ProductInputs[0]];
        if (lead.Map.Count == 0) return -1;
        var fastest = lead.Map[lead.Map.Count - 1];
        if (fastest.Divisor != 1) return -1;
        int axis = -1;
        foreach (var term in fastest.Terms)
        {
            if (term.Coefficient != 1) continue;
            foreach (int r in spec.Space.ReductionAxes)
                if (r == term.Axis) axis = r;
        }
        if (axis < 0 || spec.Space.Axes[axis].Extent % Lanes != 0) return -1;

        // No cross-output reuse worth a tile: the lead operand reads every parallel axis, and
        // any other body operand that skips one is a broadcast parameter no larger than the
        // reduction it runs along (LayerNorm backward's gamma[j]), which caches. A
        // contraction's second operand, B[k, n], is far larger than k and is the reuse the
        // tiled lowering exists to exploit.
        var body = new List<int>(spec.ProductInputs);
        if (spec.PreBiasInput is { } preBias) body.Add(preBias);
        foreach (int p in spec.Space.ParallelAxes)
            if (!References(lead, p)) return -1;
        long reductionLength = spec.Space.ReductionTripCount;
        foreach (int operand in body)
        {
            bool skips = false;
            foreach (int p in spec.Space.ParallelAxes)
                if (!References(spec.Inputs[operand], p)) skips = true;
            if (skips && spec.Inputs[operand].ElementCount > reductionLength) return -1;
        }

        // Only the body may read the interleaved axis. An epilogue operand indexed by a
        // reduction axis is not something any declared kernel does, and rewriting it would
        // be guessing what it meant.
        for (int i = 0; i < spec.Inputs.Count; i++)
            if (!body.Contains(i) && References(spec.Inputs[i], axis)) return -1;

        return axis;
    }

    private static bool References(CodegenTensorBinding binding, int target)
    {
        foreach (var expr in binding.Map)
            foreach (var term in expr.Terms)
                if (term.Axis == target) return true;
        return false;
    }

    private static int[] ToArray(IReadOnlyList<int> values)
    {
        var array = new int[values.Count];
        for (int i = 0; i < array.Length; i++) array[i] = values[i];
        return array;
    }
}
