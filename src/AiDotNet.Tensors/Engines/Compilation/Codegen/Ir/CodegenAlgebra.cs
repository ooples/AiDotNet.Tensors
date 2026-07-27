// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>The number system a kernel's arithmetic is carried out in.</summary>
/// <remarks>
/// <para>
/// A complex or quaternion element is <see cref="Components"/> consecutive fp32 values, and
/// the product of two of them is a fixed pattern of real multiply-accumulates. That pattern
/// is the ONLY thing that differs between these algebras, which is why one table drives all
/// of them rather than each getting its own emitter.
/// </para>
/// <para>
/// Everything else about a kernel -- the iteration space, the index maps, the bounds
/// predicates -- is unchanged, because an element's components are adjacent in memory and a
/// tensor of complex numbers has exactly the shape it says it has.
/// </para>
/// </remarks>
public enum CodegenAlgebra
{
    /// <summary>Ordinary fp32. One component; the product is a multiply.</summary>
    Real,

    /// <summary>
    /// Complex, stored as adjacent (real, imaginary) fp32 pairs.
    /// <c>(a0 + a1 i)(b0 + b1 i) = (a0b0 - a1b1) + (a0b1 + a1b0) i</c>.
    /// </summary>
    Complex,

    /// <summary>
    /// Quaternion, stored as adjacent (w, x, y, z) fp32 quadruples, with
    /// <c>i² = j² = k² = ijk = -1</c>. NOT commutative, so operand order is load-bearing.
    /// </summary>
    Quaternion,
}

/// <summary>One real multiply-accumulate of an algebra's product: <c>out[Out] += Sign * a[A] * b[B]</c>.</summary>
public readonly struct CodegenProductTerm
{
    /// <summary>Component of the result this term contributes to.</summary>
    public int Out { get; }

    /// <summary>Component of the left operand.</summary>
    public int A { get; }

    /// <summary>Component of the right operand.</summary>
    public int B { get; }

    /// <summary>+1 or -1.</summary>
    public int Sign { get; }

    /// <summary>Creates a term.</summary>
    public CodegenProductTerm(int outComponent, int a, int b, int sign)
    {
        Out = outComponent; A = a; B = b; Sign = sign;
    }
}

/// <summary>Component counts and multiplication tables for <see cref="CodegenAlgebra"/>.</summary>
public static class CodegenAlgebraTables
{
    /// <summary>fp32 values one element of the algebra occupies.</summary>
    public static int Components(this CodegenAlgebra algebra) => algebra switch
    {
        CodegenAlgebra.Real => 1,
        CodegenAlgebra.Complex => 2,
        CodegenAlgebra.Quaternion => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(algebra)),
    };

    /// <summary>
    /// The product, as real multiply-accumulates.
    /// </summary>
    /// <remarks>
    /// The quaternion table is written out rather than generated from the i/j/k relations,
    /// because a generator for it is harder to check by eye than the sixteen terms are, and
    /// a single wrong sign produces a kernel that runs and returns rotations that are subtly
    /// not rotations. <see cref="CodegenAlgebraTests"/> checks it against the defining
    /// relations instead of against itself.
    /// </remarks>
    public static CodegenProductTerm[] ProductTable(this CodegenAlgebra algebra) => algebra switch
    {
        CodegenAlgebra.Real => new[] { new CodegenProductTerm(0, 0, 0, +1) },

        CodegenAlgebra.Complex => new[]
        {
            new CodegenProductTerm(0, 0, 0, +1),
            new CodegenProductTerm(0, 1, 1, -1),
            new CodegenProductTerm(1, 0, 1, +1),
            new CodegenProductTerm(1, 1, 0, +1),
        },

        CodegenAlgebra.Quaternion => new[]
        {
            // w = a0b0 - a1b1 - a2b2 - a3b3
            new CodegenProductTerm(0, 0, 0, +1),
            new CodegenProductTerm(0, 1, 1, -1),
            new CodegenProductTerm(0, 2, 2, -1),
            new CodegenProductTerm(0, 3, 3, -1),

            // x = a0b1 + a1b0 + a2b3 - a3b2
            new CodegenProductTerm(1, 0, 1, +1),
            new CodegenProductTerm(1, 1, 0, +1),
            new CodegenProductTerm(1, 2, 3, +1),
            new CodegenProductTerm(1, 3, 2, -1),

            // y = a0b2 - a1b3 + a2b0 + a3b1
            new CodegenProductTerm(2, 0, 2, +1),
            new CodegenProductTerm(2, 1, 3, -1),
            new CodegenProductTerm(2, 2, 0, +1),
            new CodegenProductTerm(2, 3, 1, +1),

            // z = a0b3 + a1b2 - a2b1 + a3b0
            new CodegenProductTerm(3, 0, 3, +1),
            new CodegenProductTerm(3, 1, 2, +1),
            new CodegenProductTerm(3, 2, 1, -1),
            new CodegenProductTerm(3, 3, 0, +1),
        },

        _ => throw new ArgumentOutOfRangeException(nameof(algebra)),
    };

    /// <summary>
    /// Multiplies two elements component-wise, for the reference interpreter.
    /// </summary>
    public static void Multiply(this CodegenAlgebra algebra, double[] a, double[] b, double[] result)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (result is null) throw new ArgumentNullException(nameof(result));

        for (int c = 0; c < result.Length; c++) result[c] = 0.0;
        foreach (var term in algebra.ProductTable())
            result[term.Out] += term.Sign * a[term.A] * b[term.B];
    }
}
