// Copyright (c) AiDotNet. All rights reserved.
// Forward-mode AD dual numbers — packed primal + tangent tensors.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff.ForwardAD;

/// <summary>
/// A forward-mode AD "dual number" over tensors — pairs a primal
/// value with a tangent (directional derivative). Forward-mode AD
/// propagates tangents through operations in the same pass as the
/// primal computation, giving <c>O(output_dim)</c>-independent cost
/// for a single directional derivative. Essential for Jacobian-vector
/// products (JVP), <c>jacfwd</c>, and forward-over-reverse Hessians.
/// </summary>
/// <remarks>
/// <para><b>Design — side-band tangent, not packed SIMD:</b></para>
/// <para>
/// A true "pack primal+tangent into one <see cref="System.Numerics.Vector{T}"/>
/// register" approach is possible on CPU but would require a parallel
/// engine surface that understands interleaved layout. To ship a
/// correct forward-mode transform in a single PR we store primal and
/// tangent as two separate <see cref="Tensor{T}"/> instances. The SIMD
/// packing optimization can land later as a drop-in replacement of
/// <see cref="DualOps{T}"/> without changing this struct or callers.
/// </para>
/// <para><b>Why a readonly struct:</b></para>
/// <para>
/// Dual is copied by value through transforms (<see cref="TensorFunc{T}.Jvp"/>,
/// <see cref="TensorFunc{T}.JacFwd"/>). Making it a struct avoids
/// per-step allocations and keeps the tangent lifetime explicit —
/// there is no "shared tangent context" that could leak between
/// concurrent transforms.
/// </para>
/// <para><b>Shape contract:</b></para>
/// <para>
/// <see cref="Primal"/> and <see cref="Tangent"/> always share the
/// same shape; constructors validate this invariant so an op rule
/// can assume it without re-checking.
/// </para>
/// </remarks>
/// <typeparam name="T">Element numeric type.</typeparam>
public readonly struct Dual<T>
{
    /// <summary>Primal value — the ordinary tensor the computation
    /// would produce without differentiation.</summary>
    public Tensor<T> Primal { get; }

    /// <summary>Directional derivative of <see cref="Primal"/> with
    /// respect to the input along the direction the enclosing JVP
    /// transform was seeded with.</summary>
    public Tensor<T> Tangent { get; }

    /// <summary>Gets the shape of both <see cref="Primal"/> and
    /// <see cref="Tangent"/> (they are always identical).</summary>
    public int[] Shape => Primal._shape;

    /// <summary>
    /// Pairs a primal tensor with an explicit tangent tensor. Both
    /// tensors must have identical shape.
    /// </summary>
    /// <param name="primal">The primal value.</param>
    /// <param name="tangent">The tangent (derivative). Must match
    /// <paramref name="primal"/>'s shape.</param>
    /// <exception cref="ArgumentNullException">Thrown if either
    /// argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown if the shapes
    /// differ.</exception>
    public Dual(Tensor<T> primal, Tensor<T> tangent)
    {
        Primal = primal ?? throw new ArgumentNullException(nameof(primal));
        Tangent = tangent ?? throw new ArgumentNullException(nameof(tangent));
        if (!ShapesEqual(primal._shape, tangent._shape))
            throw new ArgumentException(
                $"Dual primal/tangent shapes must match — got {FormatShape(primal._shape)} and {FormatShape(tangent._shape)}.",
                nameof(tangent));
    }

    /// <summary>
    /// Wraps a primal value with a zero tangent. Use this when a
    /// tensor should participate in forward-mode computation but is
    /// independent of the direction the JVP is seeded along (e.g., a
    /// constant weight tensor during a JVP over the input).
    /// </summary>
    /// <param name="primal">The primal value.</param>
    /// <returns>A dual with zero tangent of the same shape.</returns>
    /// <exception cref="ArgumentNullException">Thrown if
    /// <paramref name="primal"/> is null.</exception>
    public static Dual<T> Constant(Tensor<T> primal)
    {
        if (primal is null) throw new ArgumentNullException(nameof(primal));
        var ops = MathHelper.GetNumericOperations<T>();
        var zero = ops.Zero;
        var zeroData = new T[primal.Length];
        for (int i = 0; i < zeroData.Length; i++) zeroData[i] = zero;
        return new Dual<T>(primal, new Tensor<T>(zeroData, (int[])primal._shape.Clone()));
    }

    /// <summary>
    /// Wraps a primal value with a one-seed tangent (derivative of
    /// identity). Use this to mark the direction along which the JVP
    /// is being computed. Combined with <see cref="Constant"/> for
    /// other inputs, this is how <see cref="TensorFunc{T}.Jvp"/>
    /// constructs its initial dual vector.
    /// </summary>
    /// <param name="primal">The primal value.</param>
    /// <param name="tangentSeed">The tangent direction — often a
    /// user-supplied "direction of differentiation". Must match
    /// <paramref name="primal"/>'s shape.</param>
    /// <returns>A dual that will propagate this tangent forward.</returns>
    /// <exception cref="ArgumentNullException">Thrown if either
    /// argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown if shapes differ.</exception>
    public static Dual<T> Seed(Tensor<T> primal, Tensor<T> tangentSeed)
        => new Dual<T>(primal, tangentSeed);

    /// <summary>
    /// Deconstruct to <c>(primal, tangent)</c>.
    /// </summary>
    public void Deconstruct(out Tensor<T> primal, out Tensor<T> tangent)
    {
        primal = Primal;
        tangent = Tangent;
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static string FormatShape(int[] shape)
        => "[" + string.Join(",", shape) + "]";
}
