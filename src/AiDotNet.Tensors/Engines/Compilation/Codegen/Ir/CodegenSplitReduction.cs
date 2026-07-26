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
    /// <para>
    /// Splits ONE axis. The blocks-per-SM model preferred two and the hardware disagreed
    /// by 1.41x — see <see cref="ChooseAxes"/> for the numbers.
    /// </para>
    /// <para>
    /// The partial pass must keep real reduction work, so the axis is CHUNKED rather than
    /// promoted whole unless other reduction axes remain. Promoting the only reduction
    /// axis leaves the combine doing the entire reduction with the original kernel's
    /// thread count — the combine IS the original kernel, plus a wasted copy — and that
    /// lost on every graph it was measured on, by up to 3.80x.
    /// </para>
    /// </remarks>
    public static CodegenSplitPlan? TryPlan(
        CodegenKernelSpec spec, int multiprocessors = 68, int blockThreads = 256)
    {
        var ranked = ChooseAxes(spec, multiprocessors, blockThreads);
        if (ranked.Count == 0) return null;

        int axis = ranked[0];
        int extent = spec.Space.Axes[axis].Extent;

        // Reduction trips that survive if this axis is taken whole.
        long remaining = 1;
        foreach (int a in spec.Space.ReductionAxes)
            if (a != axis) remaining *= spec.Space.Axes[a].Extent;

        if (remaining > 1)
        {
            // Other axes still reduce, so taking this one whole leaves the partial with
            // work and the combine with a genuinely shorter reduction.
            var promoted = new[] { axis };
            var (partial, combine) = Split(spec, promoted);
            return new CodegenSplitPlan(partial, combine, partial.Output.ElementCount, promoted);
        }

        // This is the whole reduction, so it has to be chunked. Aim the chunk count at a
        // full device and round DOWN to a divisor -- a chunk that does not divide exactly
        // would need a bounds guard on a reduction axis, which would read outside the
        // operand.
        long target = (long)multiprocessors * 4L * blockThreads;
        long want = (target + spec.Output.ElementCount - 1) / Math.Max(1L, spec.Output.ElementCount);
        int factor = LargestDivisorAtMost(extent, (int)Math.Min(want, extent / 2L));
        if (factor <= 1) return null;

        var (chunkedPartial, chunkedCombine) = SplitChunked(spec, axis, factor);
        return new CodegenSplitPlan(
            chunkedPartial, chunkedCombine, chunkedPartial.Output.ElementCount, new[] { axis });
    }

    /// <summary>Largest divisor of <paramref name="extent"/> not above <paramref name="cap"/>.</summary>
    private static int LargestDivisorAtMost(int extent, int cap)
    {
        if (cap < 2) return 1;
        for (int d = Math.Min(cap, extent); d >= 2; d--)
            if (extent % d == 0) return d;
        return 1;
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
    /// Splits a reduction axis into <paramref name="splitFactor"/> chunks, promoting the
    /// chunk index to a parallel axis and leaving the work within a chunk as a loop.
    /// </summary>
    /// <param name="spec">Operator with a long reduction and a small output.</param>
    /// <param name="reductionAxis">Reduction axis to chunk.</param>
    /// <param name="splitFactor">Number of chunks. Must divide the axis's extent.</param>
    /// <remarks>
    /// <para>
    /// Promoting an axis WHOLE only helps when other reduction axes remain, because the
    /// combine pass keeps the original kernel's output — and therefore the original
    /// kernel's thread count. If the promotion consumes the entire reduction, the partial
    /// pass reduces nothing and the combine IS the original kernel, so the split is a
    /// wasted copy. Measured on an idle device:
    /// </para>
    /// <code>
    ///   graph                       single   split   launch config
    ///   matmul 128x96x64             11.5    29.9    32blk -> 3072+32blk
    ///   matmul A-transposed           9.3    35.3    32blk -> 3072+32blk
    ///   linear 256x128x64            14.4    49.7    16blk -> 8192+64blk
    ///   reduce-sum [512,256]        175.1   186.6     1blk ->   512+1blk
    /// </code>
    /// <para>
    /// Every one of those has a single reduction axis, and the combine column shows the
    /// tell: its block count never improves on the original. Chunking fixes it — the
    /// partial keeps <c>extent/splitFactor</c> trips of real reduction, so the combine's
    /// reduction is genuinely shorter than the one it replaced.
    /// </para>
    /// </remarks>
    public static (CodegenKernelSpec Partial, CodegenKernelSpec Combine) SplitChunked(
        CodegenKernelSpec spec, int reductionAxis, int splitFactor)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (spec.Reduce != CodegenReduceKind.Sum)
            throw new NotSupportedException(
                "Only a sum reduction splits: the combine pass has to be the same " +
                "associative operation, and " + spec.Reduce + " is not summed partials.");

        var axes = spec.Space.Axes;
        bool isReduction = false;
        foreach (int a in spec.Space.ReductionAxes) if (a == reductionAxis) isReduction = true;
        if (!isReduction)
            throw new ArgumentOutOfRangeException(nameof(reductionAxis),
                "Axis " + reductionAxis + " is not a reduction axis of this operator.");

        int extent = axes[reductionAxis].Extent;
        if (splitFactor <= 1 || splitFactor > extent || extent % splitFactor != 0)
            throw new ArgumentOutOfRangeException(nameof(splitFactor),
                "Split factor " + splitFactor + " must be above one and divide the axis " +
                "extent " + extent + " exactly; a partial chunk would need a bounds guard " +
                "on a reduction axis, which would read outside the operand.");

        int chunk = extent / splitFactor;

        // The chunked axis KEEPS its index and shrinks to one chunk; the chunk index
        // becomes a new parallel axis appended at the end. Splitting this way means no
        // existing axis is renumbered, so every index map stays valid apart from the one
        // extra term below.
        int chunkAxis = axes.Count;
        var partialAxes = new CodegenAxis[axes.Count + 1];
        for (int a = 0; a < axes.Count; a++)
            partialAxes[a] = a == reductionAxis
                ? CodegenAxis.Reduce(axes[a].Name, chunk)
                : axes[a];
        partialAxes[chunkAxis] = CodegenAxis.Parallel(axes[reductionAxis].Name + "_chunk", splitFactor);
        var partialSpace = new CodegenIterationSpace(partialAxes);

        // Original index = chunkIndex * chunk + withinChunk. Anywhere a map read the axis
        // with coefficient c, it now also reads the chunk index with coefficient c*chunk.
        // Applied as an extra TERM rather than a substitution, so compound maps -- a
        // convolution window is `oh*stride + kh - pad` -- stay correct without unfolding.
        var partialInputs = new CodegenTensorBinding[spec.ProductInputs.Count];
        var partialProduct = new int[spec.ProductInputs.Count];
        for (int i = 0; i < spec.ProductInputs.Count; i++)
        {
            var b = spec.Inputs[spec.ProductInputs[i]];
            partialInputs[i] = new CodegenTensorBinding(
                i, b.Name, ToArray(b.Shape),
                AddChunkTerm(b.Map, reductionAxis, chunkAxis, chunk));
            partialProduct[i] = i;
        }

        int rank = spec.Output.Shape.Count;
        var partialShape = new int[rank + 1];
        var partialMap = new CodegenAffineExpr[rank + 1];
        for (int d = 0; d < rank; d++)
        {
            partialShape[d] = spec.Output.Shape[d];
            partialMap[d] = spec.Output.Map[d];
        }
        partialShape[rank] = splitFactor;
        partialMap[rank] = CodegenAffineExpr.Axis(chunkAxis);

        var partial = new CodegenKernelSpec(
            spec.Name + "_partial", partialSpace, partialInputs,
            new CodegenTensorBinding(partialInputs.Length, "partial", partialShape, partialMap, isOutput: true),
            partialProduct,
            AnyReductionLeft(partialSpace) ? CodegenReduceKind.Sum : CodegenReduceKind.None);

        return (partial, BuildCombine(spec, partialShape, new[] { splitFactor }));
    }

    private static CodegenAffineExpr[] AddChunkTerm(
        IReadOnlyList<CodegenAffineExpr> map, int reductionAxis, int chunkAxis, int chunk)
    {
        var rewritten = new CodegenAffineExpr[map.Count];
        for (int d = 0; d < map.Count; d++)
        {
            var expr = map[d];
            int coefficient = 0;
            foreach (var term in expr.Terms)
                if (term.Axis == reductionAxis) coefficient = term.Coefficient;

            if (coefficient == 0) { rewritten[d] = expr; continue; }

            var terms = new CodegenAffineTerm[expr.Terms.Count + 1];
            for (int t = 0; t < expr.Terms.Count; t++) terms[t] = expr.Terms[t];
            terms[expr.Terms.Count] = new CodegenAffineTerm(chunkAxis, coefficient * chunk);
            rewritten[d] = new CodegenAffineExpr(
                terms, expr.Constant, expr.Divisor, expr.RequiresExactDivision);
        }
        return rewritten;
    }

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

        // Only the PRODUCT operands feed the partial pass. Bias, scale and activation
        // apply once to the finished sum, so a partial pass that carried them would add
        // the bias once per partial; they move to the combine instead. Binding only what
        // the pass reads also keeps its parameter list free of unused pointers.
        var partialInputs = new CodegenTensorBinding[spec.ProductInputs.Count];
        var partialProduct = new int[spec.ProductInputs.Count];
        for (int i = 0; i < spec.ProductInputs.Count; i++)
        {
            var b = spec.Inputs[spec.ProductInputs[i]];
            partialInputs[i] = new CodegenTensorBinding(
                i, b.Name, ToArray(b.Shape), ToArray(b.Map));
            partialProduct[i] = i;
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
            partialInputs.Length, "partial", partialShape, partialMap, isOutput: true);

        // Promoting EVERY reduction axis is legitimate -- it materialises the product
        // and leaves all the summing to the combine pass -- but then the partial pass
        // has nothing left to reduce and must not claim it does.
        var partial = new CodegenKernelSpec(
            spec.Name + "_partial", partialSpace, partialInputs, partialOutput, partialProduct,
            AnyReductionLeft(partialSpace) ? CodegenReduceKind.Sum : CodegenReduceKind.None);

        var promotedExtents = new int[promote.Count];
        for (int p = 0; p < promote.Count; p++) promotedExtents[p] = axes[promote[p]].Extent;
        return (partial, BuildCombine(spec, partialShape, promotedExtents));
    }

    /// <summary>
    /// Builds the pass that sums the partials back down and applies the epilogue.
    /// </summary>
    /// <param name="spec">The original operator, whose output and epilogue this restores.</param>
    /// <param name="partialShape">Shape the partial pass wrote.</param>
    /// <param name="splitExtents">Trailing dimensions of that shape, which are reduced away.</param>
    private static CodegenKernelSpec BuildCombine(
        CodegenKernelSpec spec, int[] partialShape, int[] splitExtents)
    {
        int rank = spec.Output.Shape.Count;

        var combineAxes = new CodegenAxis[rank + splitExtents.Length];
        for (int d = 0; d < rank; d++)
            combineAxes[d] = CodegenAxis.Parallel("o" + I(d), spec.Output.Shape[d]);
        for (int p = 0; p < splitExtents.Length; p++)
            combineAxes[rank + p] = CodegenAxis.Reduce("split" + I(p), splitExtents[p]);
        var combineSpace = new CodegenIterationSpace(combineAxes);

        var combineInMap = new CodegenAffineExpr[rank + splitExtents.Length];
        for (int d = 0; d < rank + splitExtents.Length; d++) combineInMap[d] = CodegenAffineExpr.Axis(d);

        var combineInputs = new List<CodegenTensorBinding>
        {
            new(0, "partial", (int[])partialShape.Clone(), combineInMap),
        };

        // The epilogue moves here. Its operands were indexed against the ORIGINAL axes,
        // and the combine has its own, so their maps have to be rewritten rather than
        // reused -- a bias that kept the original numbering would read a different axis
        // and still emit.
        var originalToCombine = new Dictionary<int, int>();
        for (int d = 0; d < rank; d++)
        {
            var expr = spec.Output.Map[d];
            if (expr.Terms.Count == 1 && expr.Terms[0].Coefficient == 1 &&
                expr.Constant == 0 && expr.Divisor == 1)
                originalToCombine[expr.Terms[0].Axis] = d;
        }

        int? combineBias = null, combineScale = null;
        if (spec.BiasInput.HasValue)
        {
            combineBias = combineInputs.Count;
            combineInputs.Add(Rebind(spec.Inputs[spec.BiasInput.Value],
                combineInputs.Count, originalToCombine, "bias"));
        }
        if (spec.ScaleInput.HasValue)
        {
            combineScale = combineInputs.Count;
            combineInputs.Add(Rebind(spec.Inputs[spec.ScaleInput.Value],
                combineInputs.Count, originalToCombine, "scale"));
        }

        var combineOutMap = new CodegenAffineExpr[rank];
        for (int d = 0; d < rank; d++) combineOutMap[d] = CodegenAffineExpr.Axis(d);
        var combineOutput = new CodegenTensorBinding(
            combineInputs.Count, spec.Output.Name, ToArray(spec.Output.Shape),
            combineOutMap, isOutput: true);

        return new CodegenKernelSpec(
            spec.Name + "_combine", combineSpace, combineInputs.ToArray(), combineOutput,
            new[] { 0 }, CodegenReduceKind.Sum,
            biasInput: combineBias, scaleInput: combineScale, activation: spec.Activation);
    }

    /// <summary>
    /// Rebinds an epilogue operand onto the combine pass's axes.
    /// </summary>
    /// <remarks>
    /// Refuses anything it cannot translate exactly. An epilogue operand that referenced
    /// a reduction axis of the original could not have been an epilogue in the first
    /// place, and one whose map does not survive the renumbering would read the wrong
    /// element at full speed.
    /// </remarks>
    private static CodegenTensorBinding Rebind(
        CodegenTensorBinding binding, int parameterIndex,
        Dictionary<int, int> originalToCombine, string role)
    {
        var map = new CodegenAffineExpr[binding.Map.Count];
        for (int d = 0; d < binding.Map.Count; d++)
        {
            var expr = binding.Map[d];
            var terms = new CodegenAffineTerm[expr.Terms.Count];
            for (int t = 0; t < expr.Terms.Count; t++)
            {
                if (!originalToCombine.TryGetValue(expr.Terms[t].Axis, out int moved))
                    throw new NotSupportedException(
                        "The " + role + " reads axis " + expr.Terms[t].Axis + ", which the " +
                        "combine pass does not carry, so the epilogue cannot move to it.");
                terms[t] = new CodegenAffineTerm(moved, expr.Terms[t].Coefficient);
            }
            map[d] = new CodegenAffineExpr(
                terms, expr.Constant, expr.Divisor, expr.RequiresExactDivision);
        }
        return new CodegenTensorBinding(
            parameterIndex, binding.Name, ToArray(binding.Shape), map);
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
