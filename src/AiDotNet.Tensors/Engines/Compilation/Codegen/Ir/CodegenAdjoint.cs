// Copyright (c) AiDotNet. All rights reserved.
// Backward kernels derived from the forward spec, not written a second time.
//
// A forward operator is
//
//     out[F(p)] = sum over r of  data[G(p,r)] * weight[H(p,r)]
//
// and its gradient with respect to the data is the ADJOINT of that same map:
//
//     dData[j] = sum over every (p,r) with G(p,r) = j  of  dOut[F(p)] * weight[H(p,r)]
//
// which is another operator of exactly the same shape. So the backward kernel does
// not need to be authored, verified, released and benchmarked separately -- it can be
// DERIVED, and then carried through the same conveyor as the forward one.
//
// The derivation is mechanical once the index maps are first-class:
//
//   * an axis the data map DETERMINES (given the output index) becomes a parallel
//     axis of the backward kernel;
//   * an axis the data map leaves FREE becomes a reduction axis -- that is the set
//     being summed over;
//   * a forward gather window `oh*stride + kh - pad` inverts to the exact-division
//     map `(ih + pad - kh)/stride`, which the affine layer already models as
//     TransposedWindow. The exactness predicate is not a special case bolted on for
//     transposed convolution; it is what an adjoint index map *is*.
//
// This is where the index-map IR pays for itself. Hand-written backward kernels are
// where the bugs live -- the shipped grouped-deformable backward kernel computed
// zeros because its thread count was maintained by hand, separately from its
// reference. A derived adjoint cannot drift from the forward operator it came from.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Derives gradient kernels from a forward <see cref="CodegenKernelSpec"/>.</summary>
public static class CodegenAdjoint
{
    /// <summary>
    /// Builds the gradient-with-respect-to-data kernel of <paramref name="forward"/>.
    /// </summary>
    /// <param name="forward">The forward operator. Must be a plain sum-reduction of two operands.</param>
    /// <param name="dataInput">
    /// Index of the operand to differentiate with respect to. Passed explicitly rather
    /// than guessed: picking the wrong operand silently produces a kernel that computes
    /// the wrong gradient, and no shape check would catch it.
    /// </param>
    /// <returns>
    /// A spec taking (dOut, weight) and producing dData, with the same reduce kind and
    /// no epilogue.
    /// </returns>
    /// <exception cref="NotSupportedException">
    /// Thrown for forward operators whose adjoint this layer cannot express: a
    /// non-sum reduction, an activation (whose backward needs the forward output),
    /// a bias (which makes the operator affine rather than linear), or an index map
    /// that is not an axis or a single strided window.
    /// </exception>
    /// <summary>
    /// Gradient with respect to the WEIGHTS, without which these kernels cannot train.
    /// </summary>
    /// <remarks>
    /// The transform is not data-specific -- it differentiates with respect to whichever
    /// operand it is given -- so this is the same derivation with the roles swapped. For
    /// a convolution it produces
    ///
    ///     dW[k, c, kh, kw] = sum over n, oh, ow of dOut[n,k,oh,ow] * in[n,c,oh+kh-1,ow+kw-1]
    ///
    /// whose reduction is over the batch and spatial axes rather than over channels and
    /// taps. The weight's dimensions become the parallel axes and everything the weight
    /// index does not pin down becomes a reduction axis, which falls out of the same
    /// determined/free classification.
    /// </remarks>
    /// <param name="forward">The forward operator.</param>
    /// <param name="weightInput">Index of the operand to differentiate with respect to.</param>
    public static CodegenKernelSpec BackwardWeights(CodegenKernelSpec forward, int weightInput)
        => BackwardData(forward, weightInput);

    public static CodegenKernelSpec BackwardData(CodegenKernelSpec forward, int dataInput)
    {
        if (forward is null) throw new ArgumentNullException(nameof(forward));

        if (forward.Reduce != CodegenReduceKind.Sum)
            throw new NotSupportedException(
                "Only a sum reduction has a linear adjoint; " + forward.Reduce + " does not.");
        if (forward.Activation != CodegenActivationKind.None)
            throw new NotSupportedException(
                "Activation backward needs the forward pre-activation value, which is not " +
                "an index-map transform. Differentiate the activation-free operator.");
        if (forward.BiasInput.HasValue || forward.ScaleInput.HasValue)
            throw new NotSupportedException(
                "Bias and scale make the operator affine, not linear; drop the epilogue first.");
        if (forward.ProductInputs.Count != 2)
            throw new NotSupportedException(
                "The adjoint is defined here for a two-operand product, got " +
                forward.ProductInputs.Count + ".");

        // dataInput has to BE one of the operands. Deriving weightInput as "the other one"
        // cannot detect a bogus dataInput: with operands {0,1} and dataInput 7, the loop
        // simply assigns weightInput 1 and the guard below never fires, so the adjoint is
        // built against the wrong operand instead of being refused.
        bool namesAnOperand = false;
        foreach (int i in forward.ProductInputs) if (i == dataInput) namesAnOperand = true;
        if (!namesAnOperand)
            throw new ArgumentOutOfRangeException(nameof(dataInput),
                "dataInput " + dataInput + " is not one of this operator's product operands.");

        int weightInput = -1;
        foreach (int i in forward.ProductInputs)
            if (i != dataInput) weightInput = i;
        if (weightInput < 0)
            throw new ArgumentOutOfRangeException(nameof(dataInput),
                "dataInput must name one of the two product operands.");

        var fwdAxes = forward.Space.Axes;
        var data = forward.Inputs[dataInput];
        var weight = forward.Inputs[weightInput];

        // ---- 1. Classify every forward axis as determined or free.
        //
        // Determined: the backward output index pins it down. A data dim that is a
        // plain axis pins that axis; a window dim pins its SPATIAL axis once the tap
        // is chosen. Free: everything else -- the taps, and any axis the data map
        // never mentions (the output-channel axis of a dense convolution).
        var determined = new bool[fwdAxes.Count];
        var newAxes = new List<CodegenAxis>();
        var substitution = new CodegenAffineExpr?[fwdAxes.Count];
        var windowDims = new List<(int Dim, int Spatial, int Tap, int Stride, int Padding)>();

        for (int d = 0; d < data.Map.Count; d++)
        {
            var expr = data.Map[d];
            newAxes.Add(CodegenAxis.Parallel("o" + d.ToString(System.Globalization.CultureInfo.InvariantCulture),
                                             data.Shape[d]));

            if (expr.Divisor != 1)
                throw new NotSupportedException(
                    "Cannot adjoint a data map that already divides; dimension " + d + ".");

            if (expr.Terms.Count == 1 && expr.Terms[0].Coefficient == 1 && expr.Constant == 0)
            {
                int axis = expr.Terms[0].Axis;
                determined[axis] = true;
                substitution[axis] = CodegenAffineExpr.Axis(d);
                continue;
            }

            if (expr.Terms.Count == 2)
            {
                // A gather window: stride*spatial + 1*tap - padding. The tap is the
                // term whose axis is a forward reduction axis.
                int first = expr.Terms[0].Axis, second = expr.Terms[1].Axis;
                bool firstIsTap = fwdAxes[first].IsReduction;
                bool secondIsTap = fwdAxes[second].IsReduction;
                if (firstIsTap == secondIsTap)
                    throw new NotSupportedException(
                        "Window dimension " + d + " must combine exactly one parallel and one reduction axis.");

                int tap = firstIsTap ? first : second;
                int spatial = firstIsTap ? second : first;
                int stride = firstIsTap ? expr.Terms[1].Coefficient : expr.Terms[0].Coefficient;
                int tapCoefficient = firstIsTap ? expr.Terms[0].Coefficient : expr.Terms[1].Coefficient;
                if (tapCoefficient != 1)
                    throw new NotSupportedException(
                        "Window dimension " + d + " must use the tap with coefficient 1.");

                determined[spatial] = true;
                windowDims.Add((d, spatial, tap, stride, -expr.Constant));
                continue;
            }

            throw new NotSupportedException(
                "Data map dimension " + d + " is neither an axis nor a single strided window.");
        }

        // ---- 2. Every axis left free becomes a reduction axis of the backward kernel.
        int parallelCount = newAxes.Count;
        var freeIndex = new int[fwdAxes.Count];
        for (int a = 0; a < fwdAxes.Count; a++) freeIndex[a] = -1;
        for (int a = 0; a < fwdAxes.Count; a++)
        {
            if (determined[a]) continue;
            freeIndex[a] = newAxes.Count;
            newAxes.Add(CodegenAxis.Reduce(fwdAxes[a].Name, fwdAxes[a].Extent));
            substitution[a] = CodegenAffineExpr.Axis(freeIndex[a]);
        }

        // ---- 3. A determined spatial axis resolves through the ADJOINT window.
        // Forward  ih = oh*stride + kh - pad
        // Backward oh = (ih + pad - kh)/stride, valid only when it divides exactly.
        foreach (var w in windowDims)
        {
            if (freeIndex[w.Tap] < 0)
                throw new NotSupportedException("The tap axis of a window must be free in the adjoint.");
            substitution[w.Spatial] = CodegenAffineExpr.TransposedWindow(
                w.Dim, freeIndex[w.Tap], w.Stride, w.Padding);
        }

        var space = new CodegenIterationSpace(newAxes.ToArray());

        // ---- 4. Rewrite the forward output and weight maps in the new axes.
        var dOutMap = Substitute(forward.Output.Map, substitution, "forward output");
        var weightMap = Substitute(weight.Map, substitution, weight.Name);

        var dOut = new CodegenTensorBinding(0, "dOut", ToArray(forward.Output.Shape), dOutMap);
        var weightBinding = new CodegenTensorBinding(1, weight.Name, ToArray(weight.Shape), weightMap);

        var dDataMap = new CodegenAffineExpr[parallelCount];
        for (int d = 0; d < parallelCount; d++) dDataMap[d] = CodegenAffineExpr.Axis(d);
        var dData = new CodegenTensorBinding(2, "d" + data.Name, ToArray(data.Shape), dDataMap, isOutput: true);

        return new CodegenKernelSpec(
            forward.Name + "_bwd_" + data.Name,
            space,
            new[] { dOut, weightBinding },
            dData,
            new[] { 0, 1 },
            CodegenReduceKind.Sum);
    }

    /// <summary>
    /// Rewrites a map expressed in forward axes into the backward axis space. Only
    /// plain axis references are substitutable: a map that already combines axes
    /// would need composition of affine forms, which no operator here requires.
    /// </summary>
    private static CodegenAffineExpr[] Substitute(
        IReadOnlyList<CodegenAffineExpr> map, CodegenAffineExpr?[] substitution, string what)
    {
        var result = new CodegenAffineExpr[map.Count];
        for (int d = 0; d < map.Count; d++)
        {
            var expr = map[d];
            if (expr.Terms.Count == 0)
            {
                result[d] = CodegenAffineExpr.Const(expr.Constant);
                continue;
            }
            // Simple case: a plain axis reference maps straight to its image, which may
            // itself be compound (a transposed window, for the data gradient).
            if (expr.Terms.Count == 1 && expr.Terms[0].Coefficient == 1 &&
                expr.Constant == 0 && expr.Divisor == 1)
            {
                var replacement = substitution[expr.Terms[0].Axis];
                if (replacement is null)
                    throw new NotSupportedException(
                        what + " dimension " + d + " references an axis with no adjoint image.");
                result[d] = replacement;
                continue;
            }

            // COMPOUND MAP. Needed for the WEIGHT gradient: dW reduces over the batch and
            // spatial axes, so the activation keeps its gather window
            // `stride*oh + kh - pad` and that window has to be rewritten in the new axis
            // numbering. Refusing this is what made the weight gradient underivable and
            // left these kernels unable to train.
            //
            // Rewriting term-by-term is exact whenever every image is a plain axis, which
            // is the case when the axes are being renumbered rather than transformed. If
            // an image is itself compound the composition is no longer affine in one step,
            // and that is refused rather than approximated.
            var rewritten = new CodegenAffineTerm[expr.Terms.Count];
            for (int t = 0; t < expr.Terms.Count; t++)
            {
                var image = substitution[expr.Terms[t].Axis];
                if (image is null)
                    throw new NotSupportedException(
                        what + " dimension " + d + " references an axis with no adjoint image.");
                if (image.Terms.Count != 1 || image.Terms[0].Coefficient != 1 ||
                    image.Constant != 0 || image.Divisor != 1)
                    throw new NotSupportedException(
                        "Cannot substitute a compound image into a compound map: " +
                        what + " dimension " + d + ".");
                rewritten[t] = new CodegenAffineTerm(image.Terms[0].Axis, expr.Terms[t].Coefficient);
            }
            result[d] = new CodegenAffineExpr(
                rewritten, expr.Constant, expr.Divisor, expr.RequiresExactDivision);
        }
        return result;
    }

    private static int[] ToArray(IReadOnlyList<int> shape)
    {
        var copy = new int[shape.Count];
        for (int i = 0; i < shape.Count; i++) copy[i] = shape[i];
        return copy;
    }
}
