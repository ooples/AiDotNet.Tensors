// Copyright (c) AiDotNet. All rights reserved.
// Split a reduction across blocks, so parallelism can come from the reduction.
//
// The emitter maps parallel axes to threads and reduction axes to loops, always. When
// the parallel axes are small and the reduction is enormous that leaves the machine
// idle, and no tile fixes it -- the autotuner tried every candidate on
// depthwise_conv2d_3x3_bwd_weights and none moved it:
//
//   output elements (dW is [C,3,3])   576
//   reduction length (n x oh x ow)    100,352
//   threads at one output per thread  576
//   blocks on a 68-SM device          3          <- 4% of one wave
//   measured                          4052.6 us
//   compute roofline                  3.8 us     <- 1081x off
//
// This is not specific to depthwise. Every weight gradient has a small output and a long
// reduction, as does any norm, any loss, and any global pooling.
//
// The transform promotes one reduction axis to a parallel axis, so its extent becomes
// threads instead of loop trips, and writes a partial result per position of it. A
// second kernel then reduces over that new dimension.
//
// TWO PASSES RATHER THAN ATOMICS, deliberately. An atomicAdd combine needs no temporary
// and no second launch, but fp32 atomic addition is order-nondeterministic, so results
// vary run to run and the exact 0.000E+000 agreement gate would have to become a
// tolerance. That gate has caught four real defects in this project that the structural
// gates passed. One extra launch was measured at about 4.3 us of marginal cost, against
// a 4052 us kernel, which is not a close trade.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// An executable split: two kernels and the temporary that connects them.
/// </summary>
/// <param name="Partial">
/// Pass one. Reads the original operands and writes <paramref name="TempElements"/>
/// floats to the temporary, which is its last parameter.
/// </param>
/// <param name="Combine">
/// Pass two. Reads the temporary as parameter 0 and writes the real output as
/// parameter 1. Must be launched after <paramref name="Partial"/> completes.
/// </param>
/// <param name="TempElements">Elements the caller has to allocate between the passes.</param>
/// <param name="PromotedAxes">Reduction axes of the original that became parallel.</param>
public sealed record CodegenSplitPlan(
    CodegenKernelSpec Partial,
    CodegenKernelSpec Combine,
    long TempElements,
    IReadOnlyList<int> PromotedAxes);

/// <summary>Splits a long reduction into a partial pass and a combine pass.</summary>
public static class CodegenSplitReduction
{
    /// <summary>
    /// Largest partial buffer worth materialising, in elements -- 64 Mi, so 256 MB of
    /// fp32. Above this the combine pass's own DRAM traffic dominates whatever the extra
    /// parallelism bought.
    /// </summary>
    private const long MaxPartialElements = 64L * 1024 * 1024;


    /// <summary>
    /// Builds the split worth running for <paramref name="spec"/>, or null when it should
    /// be left alone.
    /// </summary>
    /// <remarks>
    /// Promotes ONE axis, which is what was measured to win. The blocks-per-SM model
    /// preferred two and the hardware disagreed by 1.41x -- see <see cref="ChooseAxes"/>
    /// for the numbers. A caller with a tuner should measure the prefixes of
    /// <see cref="ChooseAxes"/> itself rather than take this default.
    /// </remarks>
    public static CodegenSplitPlan? TryPlan(
        CodegenKernelSpec spec, int multiprocessors = 68, int blockThreads = 256)
    {
        var ranked = ChooseAxes(spec, multiprocessors, blockThreads);
        if (ranked.Count == 0) return null;

        var promoted = new[] { ranked[0] };
        var (partial, combine) = Split(spec, promoted);
        return new CodegenSplitPlan(partial, combine, partial.Output.ElementCount, promoted);
    }

    /// <summary>
    /// Chooses the reduction axis worth promoting, or -1 when splitting would not help.
    /// </summary>
    public static int ChooseAxis(CodegenKernelSpec spec, int multiprocessors = 68, int blockThreads = 256)
    {
        var axes = ChooseAxes(spec, multiprocessors, blockThreads);
        return axes.Count == 0 ? -1 : axes[0];
    }

    /// <summary>
    /// Ranks the reduction axes worth promoting, most valuable first, or an empty list
    /// when splitting would not help.
    /// </summary>
    /// <remarks>
    /// This is a CANDIDATE ORDER, not a decision. Every prefix of it is a valid split,
    /// and which prefix is fastest has to be measured, because modelling it was wrong the
    /// first time it was checked. Promoting one axis took
    /// depthwise_conv2d_3x3_bwd_weights from 4079.6 us to 240.8 us; the blocks-per-SM
    /// model said a second axis should help again, since one axis reached only 126 blocks
    /// on 68 SMs, and the hardware said 328.6 us -- worse:
    ///
    ///                    partial   combine    total
    ///   one axis          235.9      11.0     240.8
    ///   two axes          209.9     119.1     328.6
    ///
    /// The partial pass did get faster. The combine pass is ITSELF a small-output,
    /// long-reduction kernel -- 576 threads, 3 blocks -- and promoting a second axis grew
    /// its reduction from 56 to 3136, so it inherited the exact problem being fixed. That
    /// is why the caller measures prefixes rather than trusting the ranking.
    /// </remarks>
    public static IReadOnlyList<int> ChooseAxes(
        CodegenKernelSpec spec, int multiprocessors = 68, int blockThreads = 256)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));

        var chosen = new List<int>();
        if (spec.Reduce != CodegenReduceKind.Sum) return chosen;
        if (spec.BiasInput.HasValue || spec.ScaleInput.HasValue ||
            spec.Activation != CodegenActivationKind.None)
            return chosen;

        // Four blocks per SM is the point past which latency is hidden; a kernel already
        // there does not need the extra launch or the temporary.
        long target = (long)multiprocessors * 4L;
        long threads = spec.Output.ElementCount;
        if ((threads + blockThreads - 1) / blockThreads >= target) return chosen;

        var candidates = new List<int>(spec.Space.ReductionAxes);
        candidates.Sort((a, b) => spec.Space.Axes[b].Extent.CompareTo(spec.Space.Axes[a].Extent));

        foreach (int axis in candidates)
        {
            if ((threads + blockThreads - 1) / blockThreads >= target) break;
            int extent = spec.Space.Axes[axis].Extent;
            if (extent <= 1) continue;

            // The combine pass reads the whole temporary, so promoting an axis adds
            // (partial elements x 4 bytes) of DRAM traffic that the unsplit kernel never
            // paid. Past this point the combine costs more than the parallelism is worth
            // -- promoting n as well on the motivating kernel would have made the
            // temporary 231 MB and the combine alone slower than the 240.8 us the first
            // two axes already achieved.
            if (threads * extent > MaxPartialElements) continue;

            chosen.Add(axis);
            threads *= extent;
        }

        // Left in priority order, largest extent first, so a caller can measure prefixes.
        // Split sorts whatever it is given into ascending axis order for layout.
        return chosen;
    }

    /// <summary>
    /// Splits <paramref name="spec"/> into a partial pass and a combine pass.
    /// </summary>
    /// <param name="spec">Operator with a long reduction and a small output.</param>
    /// <param name="reductionAxis">Reduction axis to promote to parallel.</param>
    /// <returns>
    /// The partial kernel, whose output gains a trailing dimension of the promoted
    /// axis's extent, and the combine kernel that sums over it.
    /// </returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the operator cannot be split: a non-sum reduction has no associative
    /// combine, and an epilogue would be applied once per partial instead of once per
    /// output.
    /// </exception>
    public static (CodegenKernelSpec Partial, CodegenKernelSpec Combine) Split(
        CodegenKernelSpec spec, int reductionAxis) => Split(spec, new[] { reductionAxis });

    /// <summary>
    /// Splits <paramref name="spec"/> by promoting several reduction axes at once.
    /// </summary>
    /// <param name="spec">Operator with a long reduction and a small output.</param>
    /// <param name="reductionAxes">Reduction axes to promote, in ascending order.</param>
    /// <returns>
    /// The partial kernel, whose output gains one trailing dimension per promoted axis,
    /// and the combine kernel that sums over all of them.
    /// </returns>
    public static (CodegenKernelSpec Partial, CodegenKernelSpec Combine) Split(
        CodegenKernelSpec spec, IReadOnlyList<int> reductionAxes)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (reductionAxes is null) throw new ArgumentNullException(nameof(reductionAxes));
        if (reductionAxes.Count == 0)
            throw new ArgumentException("At least one axis has to be promoted.", nameof(reductionAxes));

        if (spec.Reduce != CodegenReduceKind.Sum)
            throw new NotSupportedException(
                "Only a sum reduction splits: the combine pass has to be the same " +
                "associative operation, and " + spec.Reduce + " is not summed partials.");

        if (spec.BiasInput.HasValue || spec.ScaleInput.HasValue ||
            spec.Activation != CodegenActivationKind.None)
            throw new NotSupportedException(
                "An epilogue cannot be split: bias, scale and activation apply once to " +
                "the finished sum, and a partial pass would apply them to every partial. " +
                "Split the epilogue-free operator and fuse the epilogue into the combine.");

        var axes = spec.Space.Axes;
        var promote = new List<int>();
        for (int i = 0; i < reductionAxes.Count; i++)
        {
            int candidate = reductionAxes[i];
            bool isReduction = false;
            foreach (int a in spec.Space.ReductionAxes) if (a == candidate) isReduction = true;
            if (!isReduction)
                throw new ArgumentOutOfRangeException(nameof(reductionAxes),
                    "Axis " + candidate + " is not a reduction axis of this operator.");
            if (!promote.Contains(candidate)) promote.Add(candidate);
        }
        promote.Sort();

        // ---- Pass 1: the same operator with the chosen axes promoted to parallel.
        //
        // Every index map is untouched, because each axis keeps its index -- only its
        // ROLE changes, from a loop trip to a thread coordinate. The output gains the
        // promoted axes as its LAST dimensions, in ascending axis order, so the trailing
        // dimension is the fastest-varying axis in the thread decomposition and
        // consecutive threads write consecutive addresses.
        var partialAxes = new CodegenAxis[axes.Count];
        for (int a = 0; a < axes.Count; a++)
            partialAxes[a] = promote.Contains(a)
                ? CodegenAxis.Parallel(axes[a].Name, axes[a].Extent)
                : axes[a];
        var partialSpace = new CodegenIterationSpace(partialAxes);

        var partialInputs = new CodegenTensorBinding[spec.Inputs.Count];
        for (int i = 0; i < spec.Inputs.Count; i++)
        {
            var b = spec.Inputs[i];
            partialInputs[i] = new CodegenTensorBinding(
                b.ParameterIndex, b.Name, ToArray(b.Shape), ToArray(b.Map));
        }

        int rank = spec.Output.Shape.Count;
        var partialShape = new int[rank + promote.Count];
        var partialMap = new CodegenAffineExpr[rank + promote.Count];
        for (int d = 0; d < rank; d++)
        {
            partialShape[d] = spec.Output.Shape[d];
            partialMap[d] = spec.Output.Map[d];
        }
        for (int p = 0; p < promote.Count; p++)
        {
            partialShape[rank + p] = axes[promote[p]].Extent;
            partialMap[rank + p] = CodegenAffineExpr.Axis(promote[p]);
        }

        var partialOutput = new CodegenTensorBinding(
            spec.Output.ParameterIndex, "partial", partialShape, partialMap, isOutput: true);

        // Promoting EVERY reduction axis is legitimate -- it materialises the product
        // and leaves all the summing to the combine pass -- but then the partial pass
        // has nothing left to reduce and must not claim it does.
        var partial = new CodegenKernelSpec(
            spec.Name + "_partial", partialSpace, partialInputs, partialOutput,
            ToArray(spec.ProductInputs),
            AnyReductionLeft(partialSpace) ? CodegenReduceKind.Sum : CodegenReduceKind.None);

        // ---- Pass 2: sum the partials over every promoted dimension.
        var combineAxes = new CodegenAxis[rank + promote.Count];
        for (int d = 0; d < rank; d++)
            combineAxes[d] = CodegenAxis.Parallel("o" + I(d), spec.Output.Shape[d]);
        for (int p = 0; p < promote.Count; p++)
            combineAxes[rank + p] = CodegenAxis.Reduce("split" + I(p), axes[promote[p]].Extent);
        var combineSpace = new CodegenIterationSpace(combineAxes);

        var combineInMap = new CodegenAffineExpr[rank + promote.Count];
        for (int d = 0; d < rank + promote.Count; d++) combineInMap[d] = CodegenAffineExpr.Axis(d);

        var combineInput = new CodegenTensorBinding(
            0, "partial", (int[])partialShape.Clone(), combineInMap);

        var combineOutMap = new CodegenAffineExpr[rank];
        for (int d = 0; d < rank; d++) combineOutMap[d] = CodegenAffineExpr.Axis(d);
        var combineOutput = new CodegenTensorBinding(
            1, spec.Output.Name, ToArray(spec.Output.Shape), combineOutMap, isOutput: true);

        var combine = new CodegenKernelSpec(
            spec.Name + "_combine", combineSpace, new[] { combineInput }, combineOutput,
            new[] { 0 }, CodegenReduceKind.Sum);

        return (partial, combine);
    }

    private static bool AnyReductionLeft(CodegenIterationSpace space)
    {
        foreach (int _ in space.ReductionAxes) return true;
        return false;
    }

    private static string I(int v) => v.ToString(System.Globalization.CultureInfo.InvariantCulture);

    private static int[] ToArray(IReadOnlyList<int> source)
    {
        var copy = new int[source.Count];
        for (int i = 0; i < source.Count; i++) copy[i] = source[i];
        return copy;
    }

    private static CodegenAffineExpr[] ToArray(IReadOnlyList<CodegenAffineExpr> source)
    {
        var copy = new CodegenAffineExpr[source.Count];
        for (int i = 0; i < source.Count; i++) copy[i] = source[i];
        return copy;
    }
}
