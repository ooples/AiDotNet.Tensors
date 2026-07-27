// Copyright (c) AiDotNet. All rights reserved.
// Index-map layer for the codegen IR.
//
// WHY THIS EXISTS
// ---------------
// CodegenGraph describes *what* is computed (pointwise math, reductions).
// It has no notion of an iteration space or of how a tensor element is
// addressed from that space, so it cannot express a convolution and cannot
// fuse a convolution with its epilogue. Every conv-class kernel therefore had
// to be hand-written as PTX text.
//
// That hand-written path has a demonstrated failure mode. In the #841 campaign a
// bounds guard was hand-recomputed as Batch*taps*OH*OW, dropping a *DeformGroups
// factor that the launch grid still included. Half the threads retired at the
// guard and the tail of the output was never written -- and cubin export,
// PTX<->cubin identity and the SASS zero-spill audit ALL passed, because none of
// them check numerics.
//
// The fix is structural, not vigilance: here the launch grid and the in-kernel
// bounds predicate are both DERIVED from the same CodegenIterationSpace, and a
// load's validity predicate is DERIVED from its index map plus the tensor shape.
// Neither can be written by hand, so neither can drift from the other.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>One <c>coefficient * axis</c> term of an affine index expression.</summary>
public readonly struct CodegenAffineTerm
{
    /// <summary>Index into the enclosing <see cref="CodegenIterationSpace"/>'s axis list.</summary>
    public int Axis { get; }

    /// <summary>Integer multiplier applied to the axis value.</summary>
    public int Coefficient { get; }

    /// <summary>Creates a term.</summary>
    public CodegenAffineTerm(int axis, int coefficient)
    {
        if (axis < 0) throw new ArgumentOutOfRangeException(nameof(axis));
        Axis = axis;
        Coefficient = coefficient;
    }
}

/// <summary>
/// A quasi-affine index expression over iteration-space axes:
/// <c>(sum_i coeff_i * axis_i + constant) / divisor</c>.
/// </summary>
/// <remarks>
/// <para>
/// Pure-affine (<see cref="Divisor"/> = 1) covers direct convolution, gather,
/// broadcast, transpose and reshape. The divisor form additionally covers
/// transposed convolution, whose input index is
/// <c>(out + pad - k) / stride</c> and which is only valid when that numerator
/// divides exactly -- expressed by <see cref="RequiresExactDivision"/>.
/// </para>
/// <para><b>Deliberately not expressible:</b> data-dependent indices, where the
/// index is read from another tensor (deformable convolution's learned offsets).
/// Those need a gather escape hatch and are out of scope for this layer; the
/// emitter must reject them rather than silently mis-lower them.</para>
/// </remarks>
public sealed class CodegenAffineExpr
{
    private readonly CodegenAffineTerm[] _terms;

    /// <summary>The <c>coeff * axis</c> terms, in no particular order.</summary>
    public IReadOnlyList<CodegenAffineTerm> Terms => _terms;

    /// <summary>Constant offset added before dividing.</summary>
    public int Constant { get; }

    /// <summary>Positive divisor applied to the numerator. 1 for pure-affine.</summary>
    public int Divisor { get; }

    /// <summary>
    /// When true the expression is only valid where the numerator is an exact
    /// multiple of <see cref="Divisor"/>. The emitter turns this into a
    /// <c>rem == 0</c> term of the derived validity predicate.
    /// </summary>
    public bool RequiresExactDivision { get; }

    /// <summary>Creates an index expression.</summary>
    public CodegenAffineExpr(
        CodegenAffineTerm[] terms,
        int constant = 0,
        int divisor = 1,
        bool requiresExactDivision = false)
    {
        _terms = terms ?? throw new ArgumentNullException(nameof(terms));
        if (divisor <= 0)
            throw new ArgumentOutOfRangeException(nameof(divisor), "Divisor must be positive.");
        if (divisor == 1 && requiresExactDivision)
            throw new ArgumentException("Exact-division is meaningless with divisor 1.", nameof(requiresExactDivision));
        Constant = constant;
        Divisor = divisor;
        RequiresExactDivision = requiresExactDivision;
    }

    /// <summary>The identity map for a single axis: <c>axis</c>.</summary>
    public static CodegenAffineExpr Axis(int axis) =>
        new(new[] { new CodegenAffineTerm(axis, 1) });

    /// <summary>A constant index, independent of the iteration space.</summary>
    public static CodegenAffineExpr Const(int value) =>
        new(Array.Empty<CodegenAffineTerm>(), value);

    /// <summary>
    /// The direct-convolution window map <c>axis*stride + tap - padding</c> --
    /// the single most common conv index expression.
    /// </summary>
    public static CodegenAffineExpr Window(int spatialAxis, int tapAxis, int stride, int padding) =>
        new(
            new[]
            {
                new CodegenAffineTerm(spatialAxis, stride),
                new CodegenAffineTerm(tapAxis, 1)
            },
            constant: -padding);

    /// <summary>
    /// The transposed-convolution map <c>(axis + padding - tap) / stride</c>,
    /// valid only where the numerator divides exactly.
    /// </summary>
    public static CodegenAffineExpr TransposedWindow(int spatialAxis, int tapAxis, int stride, int padding) =>
        new(
            new[]
            {
                new CodegenAffineTerm(spatialAxis, 1),
                new CodegenAffineTerm(tapAxis, -1)
            },
            constant: padding,
            divisor: stride,
            requiresExactDivision: stride != 1);

    /// <summary>
    /// Evaluates the expression for concrete axis values. Used by the reference
    /// interpreter that validates an emitter's output without a GPU.
    /// </summary>
    /// <param name="axisValues">Value of every axis in the iteration space.</param>
    /// <param name="valid">
    /// False when <see cref="RequiresExactDivision"/> is set and the numerator
    /// does not divide exactly. The index value is then meaningless.
    /// </param>
    public int Evaluate(IReadOnlyList<int> axisValues, out bool valid)
    {
        if (axisValues is null) throw new ArgumentNullException(nameof(axisValues));
        int numerator = Constant;
        for (int i = 0; i < _terms.Length; i++)
            numerator += _terms[i].Coefficient * axisValues[_terms[i].Axis];

        if (Divisor == 1) { valid = true; return numerator; }

        // Floor-division semantics with an exactness check. A negative numerator
        // is always out of range for a tensor index, so it is reported invalid
        // rather than floored -- matching the emitted predicate.
        if (numerator < 0) { valid = false; return numerator; }
        valid = !RequiresExactDivision || (numerator % Divisor == 0);
        return numerator / Divisor;
    }

    /// <summary>Human-readable form, e.g. <c>(oh*2 + kh + -1)/2</c>.</summary>
    public string Describe(IReadOnlyList<CodegenAxis> axes)
    {
        var sb = new StringBuilder();
        if (Divisor != 1) sb.Append('(');
        bool first = true;
        for (int i = 0; i < _terms.Length; i++)
        {
            var t = _terms[i];
            string name = axes is null || t.Axis >= axes.Count ? "a" + t.Axis.ToString(CultureInfo.InvariantCulture) : axes[t.Axis].Name;
            if (!first) sb.Append(" + ");
            sb.Append(t.Coefficient == 1 ? name
                : t.Coefficient == -1 ? "-" + name
                : name + "*" + t.Coefficient.ToString(CultureInfo.InvariantCulture));
            first = false;
        }
        if (Constant != 0 || first)
        {
            // A constant-only expression has no preceding term, so there is no " - " to
            // carry the sign: it has to be written onto the number itself, or Const(-3)
            // renders as "3".
            if (!first) sb.Append(Constant < 0 ? " - " : " + ");
            else if (Constant < 0) sb.Append('-');
            sb.Append(Math.Abs(Constant).ToString(CultureInfo.InvariantCulture));
        }
        if (Divisor != 1)
        {
            sb.Append(")/").Append(Divisor.ToString(CultureInfo.InvariantCulture));
            if (RequiresExactDivision) sb.Append("[exact]");
        }
        return sb.ToString();
    }
}

/// <summary>
/// One axis of a kernel's iteration space.
/// </summary>
/// <remarks>
/// Parallel axes are flattened into the thread grid, in declaration order, with
/// the LAST axis varying fastest -- so declaring the contiguous tensor axis last
/// is what makes consecutive threads touch consecutive addresses. Reduction axes
/// become sequential loops inside each thread.
/// </remarks>
public sealed class CodegenAxis
{
    /// <summary>Short name used in dumps and generated symbol names.</summary>
    public string Name { get; }

    /// <summary>Number of values the axis takes, always &gt; 0.</summary>
    public int Extent { get; }

    /// <summary>True when the axis is reduced over (a loop) rather than parallelised.</summary>
    public bool IsReduction { get; }

    /// <summary>Creates an axis.</summary>
    public CodegenAxis(string name, int extent, bool isReduction = false)
    {
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("Axis needs a name.", nameof(name));
        if (extent <= 0) throw new ArgumentOutOfRangeException(nameof(extent), "Axis extent must be positive.");
        Name = name;
        Extent = extent;
        IsReduction = isReduction;
    }

    /// <summary>Declares a parallel (grid) axis.</summary>
    public static CodegenAxis Parallel(string name, int extent) => new(name, extent, false);

    /// <summary>Declares a reduction (loop) axis.</summary>
    public static CodegenAxis Reduce(string name, int extent) => new(name, extent, true);

    /// <inheritdoc/>
    public override string ToString() =>
        $"{Name}[{Extent.ToString(CultureInfo.InvariantCulture)}]{(IsReduction ? " reduce" : "")}";
}

/// <summary>
/// The ordered axis list a kernel iterates over, and the single authority on how
/// many threads that implies.
/// </summary>
/// <remarks>
/// <para><b>This type exists to make one specific bug unrepresentable.</b>
/// <see cref="TotalThreads"/> is consumed by the host launch code AND by the
/// emitter that writes the in-kernel bounds guard. Because there is exactly one
/// definition, the grid and the guard cannot disagree -- which is precisely the
/// defect that shipped silently-wrong gradients in the grouped deformable
/// kernels, where a hand-recomputed guard lost a factor the grid still had.</para>
/// </remarks>
public sealed class CodegenIterationSpace
{
    private readonly CodegenAxis[] _axes;

    /// <summary>All axes, parallel and reduction, in declaration order.</summary>
    public IReadOnlyList<CodegenAxis> Axes => _axes;

    /// <summary>Creates an iteration space.</summary>
    /// <exception cref="ArgumentException">If no parallel axis is declared.</exception>
    public CodegenIterationSpace(params CodegenAxis[] axes)
    {
        _axes = axes ?? throw new ArgumentNullException(nameof(axes));
        if (_axes.Length == 0) throw new ArgumentException("An iteration space needs at least one axis.", nameof(axes));

        bool anyParallel = false;
        for (int i = 0; i < _axes.Length; i++) if (!_axes[i].IsReduction) { anyParallel = true; break; }
        if (!anyParallel)
            throw new ArgumentException("An iteration space needs at least one parallel axis.", nameof(axes));
    }

    /// <summary>Indices of the parallel axes, fastest-varying last.</summary>
    public int[] ParallelAxes
    {
        get
        {
            var list = new List<int>(_axes.Length);
            for (int i = 0; i < _axes.Length; i++) if (!_axes[i].IsReduction) list.Add(i);
            return list.ToArray();
        }
    }

    /// <summary>Indices of the reduction axes, outermost first.</summary>
    public int[] ReductionAxes
    {
        get
        {
            var list = new List<int>(_axes.Length);
            for (int i = 0; i < _axes.Length; i++) if (_axes[i].IsReduction) list.Add(i);
            return list.ToArray();
        }
    }

    /// <summary>
    /// The number of threads the kernel must launch: the product of every
    /// parallel axis extent.
    /// <para><b>Single source of truth.</b> The host launch grid and the
    /// emitted bounds guard both read this property. Never recompute it.</para>
    /// </summary>
    public long TotalThreads
    {
        get
        {
            long total = 1;
            for (int i = 0; i < _axes.Length; i++)
                if (!_axes[i].IsReduction) total = checked(total * _axes[i].Extent);
            return total;
        }
    }

    /// <summary>Product of the reduction extents -- the per-thread loop trip count.</summary>
    public long ReductionTripCount
    {
        get
        {
            long total = 1;
            for (int i = 0; i < _axes.Length; i++)
                if (_axes[i].IsReduction) total = checked(total * _axes[i].Extent);
            return total;
        }
    }

    /// <summary>Human-readable dump.</summary>
    public string Describe()
    {
        var sb = new StringBuilder("iter(");
        for (int i = 0; i < _axes.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(_axes[i].ToString());
        }
        sb.Append(") threads=").Append(TotalThreads.ToString(CultureInfo.InvariantCulture));
        return sb.ToString();
    }
}

/// <summary>
/// Binds one tensor parameter to the iteration space: its shape, and one index
/// expression per tensor dimension.
/// </summary>
/// <remarks>
/// The validity predicate is NOT stored. It is derived on demand from the maps
/// and <see cref="Shape"/> (every dimension must satisfy <c>0 &lt;= idx &lt; dim</c>,
/// plus any exact-division requirement), so a caller cannot supply a guard that
/// disagrees with the addressing.
/// </remarks>
public sealed class CodegenTensorBinding
{
    private readonly CodegenAffineExpr[] _map;
    private readonly int[] _shape;

    /// <summary>Kernel parameter index this tensor is bound to.</summary>
    public int ParameterIndex { get; }

    /// <summary>Short name used in generated symbol names and dumps.</summary>
    public string Name { get; }

    /// <summary>Tensor shape, outermost dimension first.</summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>One index expression per tensor dimension.</summary>
    public IReadOnlyList<CodegenAffineExpr> Map => _map;

    /// <summary>True when the binding is written rather than read.</summary>
    public bool IsOutput { get; }

    /// <summary>
    /// True when at least one dimension can address outside the tensor, so the
    /// emitter must guard the access. Computed from the maps, never supplied.
    /// </summary>
    public bool NeedsBoundsCheck
    {
        get
        {
            for (int d = 0; d < _map.Length; d++)
            {
                var e = _map[d];
                if (e.RequiresExactDivision) return true;
                // A bare axis or constant that cannot leave [0, dim) needs no guard;
                // anything with a negative constant, a negative coefficient or a
                // multi-term sum can, so guard conservatively.
                if (e.Constant < 0) return true;
                int terms = e.Terms.Count;
                if (terms > 1) return true;
                for (int t = 0; t < terms; t++) if (e.Terms[t].Coefficient < 0) return true;
            }
            return false;
        }
    }

    /// <summary>Creates a binding.</summary>
    public CodegenTensorBinding(
        int parameterIndex,
        string name,
        int[] shape,
        CodegenAffineExpr[] map,
        bool isOutput = false)
    {
        if (parameterIndex < 0) throw new ArgumentOutOfRangeException(nameof(parameterIndex));
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("Binding needs a name.", nameof(name));
        _shape = shape ?? throw new ArgumentNullException(nameof(shape));
        _map = map ?? throw new ArgumentNullException(nameof(map));
        if (_shape.Length != _map.Length)
            throw new ArgumentException(
                $"'{name}' has {_shape.Length} dimensions but {_map.Length} index expressions.", nameof(map));
        for (int i = 0; i < _shape.Length; i++)
            if (_shape[i] <= 0) throw new ArgumentException($"'{name}' dimension {i} must be positive.", nameof(shape));
        ParameterIndex = parameterIndex;
        Name = name;
        IsOutput = isOutput;
    }

    /// <summary>Total element count.</summary>
    public long ElementCount
    {
        get { long n = 1; for (int i = 0; i < _shape.Length; i++) n = checked(n * _shape[i]); return n; }
    }

    /// <summary>Row-major stride of dimension <paramref name="dim"/>.</summary>
    public long Stride(int dim)
    {
        long s = 1;
        for (int i = dim + 1; i < _shape.Length; i++) s = checked(s * _shape[i]);
        return s;
    }

    /// <summary>
    /// Resolves the flat element offset for concrete axis values, deriving the
    /// validity predicate from the maps and the shape.
    /// </summary>
    /// <param name="axisValues">Value of every axis in the iteration space.</param>
    /// <param name="inBounds">
    /// False when any dimension indexes outside the tensor or fails an
    /// exact-division requirement. Callers must treat an out-of-bounds read as
    /// the additive identity and must not perform the store.
    /// </param>
    public long ResolveOffset(IReadOnlyList<int> axisValues, out bool inBounds)
    {
        long offset = 0;
        inBounds = true;
        for (int d = 0; d < _map.Length; d++)
        {
            int idx = _map[d].Evaluate(axisValues, out bool exact);
            if (!exact || idx < 0 || idx >= _shape[d]) { inBounds = false; return 0; }
            offset += idx * Stride(d);
        }
        return offset;
    }

    /// <summary>Human-readable dump, e.g. <c>input[n, c, oh*1 + kh - 1, ow*1 + kw - 1]</c>.</summary>
    public string Describe(IReadOnlyList<CodegenAxis> axes)
    {
        var sb = new StringBuilder(Name).Append('[');
        for (int d = 0; d < _map.Length; d++)
        {
            if (d > 0) sb.Append(", ");
            sb.Append(_map[d].Describe(axes));
        }
        return sb.Append(']').ToString();
    }
}
