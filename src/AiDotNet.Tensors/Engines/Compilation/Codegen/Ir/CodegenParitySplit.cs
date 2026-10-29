// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// Splits a reduction that reads an operand through EXACT-DIVISION windows into one pass per output parity class
/// (#890 phase 4b). A strided convolution's data gradient reads <c>dOut[(ih + pad - kh) / stride]</c>, defined only
/// where the division is exact, so for each input position most taps are absent: at stride 2 a 3x3 window keeps one,
/// two or four of its nine taps, and a lowering over the whole window gathers zeros for the rest.
/// </summary>
/// <remarks>
/// <para>
/// For a window <c>(p + c r + K) / d</c> -- p a parallel axis, r a reduction axis, c coprime with d -- the positions
/// <c>p = d p' + py</c> of one residue py keep exactly the taps <c>r = r0 + d t</c> with <c>py + c r0 + K</c>
/// divisible by d, and on them the window is the affine <c>p' + c t + (py + c r0 + K) / d</c>. Each combination of
/// residues becomes a pass over its own positions and taps, with no division left and no absent tap visited.
/// </para>
/// <para>
/// Axes keep their slots (only extents change), so every other map is a substitution. The passes write disjoint
/// slices of one output, which the program declares (<see cref="CodegenProgram.PassesWriteDisjointSlices"/>). A class
/// with positions but no taps would need its outputs zeroed, which a reduction pass does not do, so such a spec (a 1x1
/// window at stride 2) is refused.
/// </para>
/// </remarks>
public static class CodegenParitySplit
{
    private readonly struct Pair
    {
        public Pair(int parallel, int reduction, int coefficient, int divisor, int constant)
        {
            Parallel = parallel; Reduction = reduction; Coefficient = coefficient; Divisor = divisor; Constant = constant;
        }

        public int Parallel { get; }
        public int Reduction { get; }
        public int Coefficient { get; }
        public int Divisor { get; }
        public int Constant { get; }
    }

    /// <summary>The parity passes of <paramref name="spec"/>, or null with the reason it does not split.</summary>
    public static CodegenProgram? TrySplit(CodegenKernelSpec spec, out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        reason = "";
        if (spec.Algebra != CodegenAlgebra.Real || spec.Reduce != CodegenReduceKind.Sum || spec.ProductInputs.Count != 2 ||
            spec.HasExpressionBody || spec.PreBiasInput.HasValue || spec.PreReduce != CodegenPreReduceOp.None ||
            spec.BiasInput.HasValue || spec.ScaleInput.HasValue || spec.Activation != CodegenActivationKind.None ||
            spec.ExtraOutputs.Count != 0 || spec.SecondaryOutput is not null || spec.HasExtendedEpilogue)
        {
            reason = "a parity split rebuilds a plain sum of two operands";
            return null;
        }
        if (spec.Output.HasIndirection || HasDivisor(spec.Output.Map))
        {
            reason = "the output must be written through plain affine maps";
            return null;
        }

        var axes = spec.Space.Axes;
        var pairs = new List<Pair>();
        foreach (var input in spec.Inputs)
        {
            if (input.HasIndirection) { reason = "an indirect operand cannot be split by parity"; return null; }
            foreach (var expr in input.Map)
            {
                if (expr.Divisor == 1) continue;
                if (!TryPair(expr, axes, out var pair, out reason)) return null;
                foreach (var other in pairs)
                {
                    if (other.Parallel == pair.Parallel || other.Reduction == pair.Reduction)
                    {
                        reason = "axis " + pair.Parallel + " or " + pair.Reduction + " is in two exact-division windows";
                        return null;
                    }
                }
                pairs.Add(pair);
            }
        }
        if (pairs.Count == 0) { reason = "no operand is read through an exact-division window"; return null; }

        // One pass per combination of residues, the first pair's residue slowest.
        int classes = 1;
        foreach (var pair in pairs) classes *= pair.Divisor;
        var passes = new List<CodegenKernelSpec>(classes);
        var residue = new int[pairs.Count];
        for (int combination = 0; combination < classes; combination++)
        {
            int rest = combination;
            for (int i = pairs.Count - 1; i >= 0; i--) { residue[i] = rest % pairs[i].Divisor; rest /= pairs[i].Divisor; }

            var newAxes = new CodegenAxis[axes.Count];
            for (int a = 0; a < axes.Count; a++) newAxes[a] = axes[a];
            var parallelScale = new int[axes.Count];
            var parallelShift = new int[axes.Count];
            var tapStart = new int[axes.Count];
            bool empty = false;
            for (int i = 0; i < pairs.Count; i++)
            {
                var pair = pairs[i];
                int d = pair.Divisor, py = residue[i];
                int positions = (axes[pair.Parallel].Extent - py + d - 1) / d;
                int r0 = -1;
                for (int r = 0; r < d && r0 < 0; r++)
                    if (Mod(py + pair.Coefficient * r + pair.Constant, d) == 0) r0 = r;
                int taps = r0 < 0 ? 0 : (axes[pair.Reduction].Extent - r0 + d - 1) / d;
                if (positions <= 0) { empty = true; break; }   // no output has this residue
                if (taps <= 0)
                {
                    reason = "residue " + py + " of axis " + pair.Parallel + " has positions but no taps, " +
                        "which would need its outputs zeroed";
                    return null;
                }
                newAxes[pair.Parallel] = CodegenAxis.Parallel(axes[pair.Parallel].Name, positions);
                newAxes[pair.Reduction] = CodegenAxis.Reduce(axes[pair.Reduction].Name, taps);
                parallelScale[pair.Parallel] = d;
                parallelShift[pair.Parallel] = py;
                parallelScale[pair.Reduction] = d;
                tapStart[pair.Reduction] = r0;
            }
            if (empty) continue;

            var inputs = new CodegenTensorBinding[spec.Inputs.Count];
            for (int k = 0; k < inputs.Length; k++)
            {
                var b = spec.Inputs[k];
                inputs[k] = new CodegenTensorBinding(b.ParameterIndex, b.Name, ToArray(b.Shape),
                    Substitute(b.Map, parallelScale, parallelShift, tapStart), elementType: b.ElementType);
            }
            var o = spec.Output;
            var output = new CodegenTensorBinding(o.ParameterIndex, o.Name, ToArray(o.Shape),
                Substitute(o.Map, parallelScale, parallelShift, tapStart), isOutput: true, elementType: o.ElementType);
            var name = new System.Text.StringBuilder(spec.Name).Append("_parity");
            foreach (int r in residue) name.Append(r.ToString(CultureInfo.InvariantCulture));
            passes.Add(new CodegenKernelSpec(name.ToString(), new CodegenIterationSpace(newAxes), inputs, output,
                ToArray(spec.ProductInputs), CodegenReduceKind.Sum, reduceScale: spec.ReduceScale));
        }
        return new CodegenProgram(passes, Array.Empty<long>(), spec.Name + " split by output parity")
        {
            PassesWriteDisjointSlices = true,
        };
    }

    // (p + c r + K) / d exactly: one parallel axis at coefficient 1, one reduction axis at a coefficient coprime
    // with d (then each residue of p keeps the taps of one residue of r).
    private static bool TryPair(CodegenAffineExpr expr, IReadOnlyList<CodegenAxis> axes, out Pair pair, out string reason)
    {
        pair = default;
        reason = "";
        if (!expr.RequiresExactDivision) { reason = "a floor division is not a window"; return false; }
        if (expr.Terms.Count != 2) { reason = "an exact-division window reads one position and one tap"; return false; }
        int parallel = -1, reduction = -1, coefficient = 0;
        foreach (var term in expr.Terms)
        {
            if (axes[term.Axis].IsReduction) { reduction = term.Axis; coefficient = term.Coefficient; }
            else if (term.Coefficient == 1) parallel = term.Axis;
        }
        if (parallel < 0 || reduction < 0) { reason = "the window must read a position at stride 1 and a tap"; return false; }
        if (Gcd(Math.Abs(coefficient), expr.Divisor) != 1)
        {
            reason = "a tap step sharing a factor with the stride keeps all or none of a residue's taps";
            return false;
        }
        pair = new Pair(parallel, reduction, coefficient, expr.Divisor, expr.Constant);
        return true;
    }

    // Position p -> scale p' + shift, tap r -> start + scale t; a window's numerator then divides exactly.
    private static CodegenAffineExpr[] Substitute(IReadOnlyList<CodegenAffineExpr> map, int[] scale, int[] shift,
        int[] tapStart)
    {
        var result = new CodegenAffineExpr[map.Count];
        for (int d = 0; d < map.Count; d++)
        {
            var expr = map[d];
            var terms = new CodegenAffineTerm[expr.Terms.Count];
            long constant = expr.Constant;
            for (int t = 0; t < terms.Length; t++)
            {
                var term = expr.Terms[t];
                int s = scale[term.Axis] == 0 ? 1 : scale[term.Axis];
                terms[t] = new CodegenAffineTerm(term.Axis, term.Coefficient * s);
                constant += (long)term.Coefficient * (shift[term.Axis] + tapStart[term.Axis]);
            }
            if (expr.Divisor == 1)
            {
                result[d] = new CodegenAffineExpr(terms, checked((int)constant));
                continue;
            }
            // The pairing made every coefficient and the constant multiples of the divisor.
            for (int t = 0; t < terms.Length; t++)
            {
                if (terms[t].Coefficient % expr.Divisor != 0)
                    throw new InvalidOperationException("A parity window kept a coefficient the stride does not divide.");
                terms[t] = new CodegenAffineTerm(terms[t].Axis, terms[t].Coefficient / expr.Divisor);
            }
            if (constant % expr.Divisor != 0)
                throw new InvalidOperationException("A parity window kept a remainder; the residue's taps are wrong.");
            result[d] = new CodegenAffineExpr(terms, checked((int)(constant / expr.Divisor)));
        }
        return result;
    }

    private static bool HasDivisor(IReadOnlyList<CodegenAffineExpr> map)
    {
        foreach (var expr in map) if (expr.Divisor != 1) return true;
        return false;
    }

    private static int Mod(int value, int divisor) => ((value % divisor) + divisor) % divisor;

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }

    private static int[] ToArray(IReadOnlyList<int> values)
    {
        var result = new int[values.Count];
        for (int i = 0; i < result.Length; i++) result[i] = values[i];
        return result;
    }
}
