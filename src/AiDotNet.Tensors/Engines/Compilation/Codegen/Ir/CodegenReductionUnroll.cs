// Copyright (c) AiDotNet. All rights reserved.
// Partial unrolling of a long reduction (#1441): the spec's single reduction axis i (extent E = U * E') becomes an outer
// axis io (extent E', in i's place) and an inner axis ii (extent U, appended last), every index term c * i rewritten as
// c * U * io + c * ii. The emitter then peels io into a run-time loop and unrolls ii, so each loop trip issues U
// independent index loads where a fully rolled loop waited on one at a time. The terms are visited in the same order --
// io outer, ii inner, i = io * U + ii increasing -- so every reduction is bit-identical to the original's.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Splits a spec's single reduction axis for partial unrolling (see the file header).</summary>
public static class CodegenReductionUnroll
{
    /// <summary>
    /// The spec with its single reduction axis split by <paramref name="factor"/>, or null when it has not exactly one
    /// reduction axis, the extent is not a multiple of the factor, or it carries a data-dependent range.
    /// </summary>
    public static CodegenKernelSpec? TrySplit(CodegenKernelSpec spec, int factor)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (factor < 2 || spec.ReductionRange is not null || spec.Space.ReductionAxes.Length != 1) return null;
        int axis = spec.Space.ReductionAxes[0];
        int extent = spec.Space.Axes[axis].Extent;
        if (extent % factor != 0 || extent / factor < 2) return null;

        int inner = spec.Space.Axes.Count;   // ii, appended
        var axes = new CodegenAxis[inner + 1];
        for (int a = 0; a < inner; a++) axes[a] = spec.Space.Axes[a];
        axes[axis] = CodegenAxis.Reduce(spec.Space.Axes[axis].Name + "o", extent / factor);
        axes[inner] = CodegenAxis.Reduce(spec.Space.Axes[axis].Name + "i", factor);

        CodegenAffineExpr Rewrite(CodegenAffineExpr expr)
        {
            var terms = new List<CodegenAffineTerm>(expr.Terms.Count + 1);
            foreach (var term in expr.Terms)
            {
                if (term.Axis != axis) { terms.Add(term); continue; }
                terms.Add(new CodegenAffineTerm(axis, checked(term.Coefficient * factor)));
                terms.Add(new CodegenAffineTerm(inner, term.Coefficient));
            }
            return new CodegenAffineExpr(terms.ToArray(), expr.Constant, expr.Divisor, expr.RequiresExactDivision);
        }

        CodegenTensorBinding Bind(CodegenTensorBinding b)
        {
            var map = new CodegenAffineExpr[b.Map.Count];
            for (int d = 0; d < map.Length; d++) map[d] = Rewrite(b.Map[d]);
            CodegenIndirectIndex?[]? indirect = null;
            if (b.HasIndirection)
            {
                indirect = new CodegenIndirectIndex?[b.Indirect.Count];
                for (int d = 0; d < indirect.Length; d++)
                {
                    var ix = b.Indirect[d];
                    if (ix is null) continue;
                    indirect[d] = new CodegenIndirectIndex(ix.IndexInput, Rewrite(ix.Position), ix.Bound, ix.OutOfRange,
                        ix.FloatIndices, ix.Offset is null ? null : Rewrite(ix.Offset));
                }
            }
            var shape = new int[b.Shape.Count];
            for (int d = 0; d < shape.Length; d++) shape[d] = b.Shape[d];
            return new CodegenTensorBinding(b.ParameterIndex, b.Name, shape, map, b.IsOutput, b.ElementType, indirect,
                b.IsIntegerValue);
        }

        var inputs = new CodegenTensorBinding[spec.Inputs.Count];
        for (int i = 0; i < inputs.Length; i++) inputs[i] = Bind(spec.Inputs[i]);
        CodegenExtraOutput[]? extras = null;
        if (spec.ExtraOutputs.Count != 0)
        {
            extras = new CodegenExtraOutput[spec.ExtraOutputs.Count];
            for (int e = 0; e < extras.Length; e++)
            {
                var x = spec.ExtraOutputs[e];
                extras[e] = x with
                {
                    Binding = Bind(x.Binding),
                    IndexExpr = x.IndexExpr is null ? null : Rewrite(x.IndexExpr),
                };
            }
        }
        var products = new int[spec.ProductInputs.Count];
        for (int i = 0; i < products.Length; i++) products[i] = spec.ProductInputs[i];
        return new CodegenKernelSpec(spec.Name, new CodegenIterationSpace(axes), inputs, Bind(spec.Output), products,
            spec.Reduce, spec.BiasInput, spec.ScaleInput, spec.Activation, spec.ReduceScale, spec.PreReduce,
            spec.PreBiasInput, spec.PreBiasScale, spec.Algebra, extras, null, null, spec.ReduceOffset,
            spec.PostScaleInput, spec.PostBiasInput, spec.Body);
    }
}