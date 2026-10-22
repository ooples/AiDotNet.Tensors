// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>What to do when a data-dependent index falls outside the dimension it addresses.</summary>
/// <remarks>
/// There is no safe default here, which is why it must be stated. An out-of-range index is
/// not a bug in the kernel -- it is ordinary in the operators that need this: a padding row
/// in an embedding table, a masked position in a sequence, a sentinel of -1 meaning "no
/// entry". Guessing between the two behaviours silently corrupts one caller or the other.
/// </remarks>
public enum CodegenIndexOutOfRange
{
    /// <summary>
    /// Contribute nothing: a gather yields the reduction's identity, a scatter performs no
    /// write. This is what a padding index or a -1 sentinel means.
    /// </summary>
    Skip,

    /// <summary>
    /// Clamp into <c>[0, bound)</c>. Correct when indices are known valid and the guard is
    /// only there to keep a malformed input from reading out of the allocation.
    /// </summary>
    Clamp,
}

/// <summary>
/// An index read from another tensor at run time, rather than computed from the iteration
/// space.
/// </summary>
/// <remarks>
/// <para>
/// This is the gather/scatter escape hatch. Until it existed, <see cref="CodegenAffineExpr"/>
/// documented data-dependent indices as "deliberately not expressible ... out of scope for
/// this layer", which meant every operator that needs one -- embedding lookup and its
/// backward, one-hot projection, sparse accumulation, deformable convolution's learned
/// offsets -- had to be hand-written outside the generator.
/// </para>
/// <para>
/// It is attached to the BINDING and not to the affine expression on purpose. An affine
/// expression is a closed-form function of the axes, and every consumer relies on that: the
/// bounds predicate is derived from it, index folding assumes it, and the tensor-core
/// recogniser matches on its shape. An index fetched from memory has none of those
/// properties and must not be able to masquerade as one.
/// </para>
/// <para>
/// The <see cref="Position"/> within the index tensor is still affine, which is what keeps
/// this tractable: the emitter knows exactly where to read the index from, it just does not
/// know what it will be until the load returns.
/// </para>
/// </remarks>
public sealed class CodegenIndirectIndex
{
    /// <summary>Index into the spec's input list of the tensor holding the indices.</summary>
    public int IndexInput { get; }

    /// <summary>Where to read the index from, within the index tensor.</summary>
    public CodegenAffineExpr Position { get; }

    /// <summary>
    /// The extent of the dimension being addressed -- the range the loaded value must fall
    /// in for the access to be legal.
    /// </summary>
    public int Bound { get; }

    /// <summary>What happens when the loaded index falls outside <see cref="Bound"/>.</summary>
    public CodegenIndexOutOfRange OutOfRange { get; }

    /// <summary>
    /// The index tensor holds float32 values (a float tensor of indices, as torch.take_along_dim's callers pass), each
    /// converted by truncation toward zero -- <c>cvt.rzi.s32.f32</c>, the conversion C's <c>(int)</c> compiles to: NaN
    /// reads as 0 and a value outside the int range saturates. Declared, never inferred: an int32 index tensor read as
    /// float (or the reverse) would compute with a reinterpreted bit pattern.
    /// </summary>
    public bool FloatIndices { get; }

    /// <summary>
    /// Subtracted from the loaded index before the range test, or null for no shift.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is what makes a DETERMINISTIC scatter expressible. A scatter written the obvious way -- iterate the
    /// sources, write through <c>indices[i]</c> -- cannot prove two iterations miss each other, so the emitter has
    /// to accumulate with <c>red.global.add.f32</c>, and an atomic float add returns a different sum per run. #890
    /// records that as the determinism gap.
    /// </para>
    /// <para>
    /// With a shift the same operator inverts: iterate the DESTINATIONS in parallel, reduce over the sources, and
    /// let each source contribute only to its own destination. Shifting by the destination axis and testing against
    /// <see cref="Bound"/> 1 is exactly <c>indices[i] == d</c> -- in range for the one destination that matches,
    /// <see cref="CodegenIndexOutOfRange.Skip"/> (the reduction's identity) for every other. The reduction then runs
    /// in reduce-axis order, which is fixed, so the result is bit-identical run to run with no atomic anywhere.
    /// </para>
    /// </remarks>
    public CodegenAffineExpr? Offset { get; }

    /// <summary>Creates a data-dependent index.</summary>
    public CodegenIndirectIndex(
        int indexInput,
        CodegenAffineExpr position,
        int bound,
        CodegenIndexOutOfRange outOfRange = CodegenIndexOutOfRange.Skip,
        bool floatIndices = false,
        CodegenAffineExpr? offset = null)
    {
        if (indexInput < 0) throw new ArgumentOutOfRangeException(nameof(indexInput));
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        if (offset is not null && outOfRange == CodegenIndexOutOfRange.Clamp)
            throw new ArgumentException(
                "A shifted index cannot clamp: clamping a miss into [0, bound) would make every source contribute " +
                "to a destination it does not belong to. Use Skip, which contributes the reduction's identity.",
                nameof(outOfRange));

        IndexInput = indexInput;
        Position = position ?? throw new ArgumentNullException(nameof(position));
        Bound = bound;
        OutOfRange = outOfRange;
        FloatIndices = floatIndices;
        Offset = offset;
    }

    // cvt.rzi.s32.f32 (and so C's (int) on the device): truncation toward zero, NaN to 0, saturation outside the range.
    private static int TruncateLikeTheDevice(double value)
    {
        if (double.IsNaN(value)) return 0;
        if (value >= int.MaxValue) return int.MaxValue;
        if (value <= int.MinValue) return int.MinValue;
        return (int)value;
    }

    /// <summary>
    /// Resolves the index for concrete axis values, reporting whether the access happens at
    /// all. Used by the reference interpreter.
    /// </summary>
    /// <param name="indexTensor">Contents of the index tensor, holding integral values.</param>
    /// <param name="axisValues">Value of every axis in the iteration space.</param>
    /// <param name="active">False when the access must be skipped entirely.</param>
    public int Resolve(double[] indexTensor, System.Collections.Generic.IReadOnlyList<int> axisValues, out bool active)
    {
        if (indexTensor is null) throw new ArgumentNullException(nameof(indexTensor));

        int position = Position.Evaluate(axisValues, out bool positionValid);
        if (!positionValid || position < 0 || position >= indexTensor.Length)
        {
            active = false;
            return 0;
        }

        // The index tensor carries integral values in a double array, because the reference
        // interpreter is uniformly fp64. Truncation rather than rounding matches the i32
        // load the emitter performs.
        // A float index is first rounded to float32, as the device holds it, then converted as cvt.rzi does.
        int index = FloatIndices ? TruncateLikeTheDevice((float)indexTensor[position]) : (int)indexTensor[position];

        if (Offset is not null)
        {
            int shift = Offset.Evaluate(axisValues, out bool shiftValid);
            if (!shiftValid) { active = false; return 0; }
            index -= shift;
        }

        if (index >= 0 && index < Bound) { active = true; return index; }

        if (OutOfRange == CodegenIndexOutOfRange.Clamp)
        {
            active = true;
            return index < 0 ? 0 : Bound - 1;
        }

        active = false;
        return 0;
    }

    /// <summary>Human-readable form, e.g. <c>idx[i]&lt;512, skip&gt;</c>.</summary>
    public string Describe(System.Collections.Generic.IReadOnlyList<CodegenAxis> axes) =>
        "in" + IndexInput.ToString(System.Globalization.CultureInfo.InvariantCulture) +
        "[" + Position.Describe(axes) + "]<" +
        Bound.ToString(System.Globalization.CultureInfo.InvariantCulture) + ", " +
        (OutOfRange == CodegenIndexOutOfRange.Clamp ? "clamp" : "skip") + (FloatIndices ? ", f32" : "") + ">";
}
