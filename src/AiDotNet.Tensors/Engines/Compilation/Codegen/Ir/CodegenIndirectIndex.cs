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

    /// <summary>Creates a data-dependent index.</summary>
    public CodegenIndirectIndex(
        int indexInput,
        CodegenAffineExpr position,
        int bound,
        CodegenIndexOutOfRange outOfRange = CodegenIndexOutOfRange.Skip)
    {
        if (indexInput < 0) throw new ArgumentOutOfRangeException(nameof(indexInput));
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));

        IndexInput = indexInput;
        Position = position ?? throw new ArgumentNullException(nameof(position));
        Bound = bound;
        OutOfRange = outOfRange;
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
        int index = (int)indexTensor[position];

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
        (OutOfRange == CodegenIndexOutOfRange.Clamp ? "clamp" : "skip") + ">";
}
