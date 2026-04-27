// Copyright (c) AiDotNet. All rights reserved.
// Forward-mode autograd rule registry — closes the issue #214 gap
// "Forward-mode op rules registered through `OpRegistry`".

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Autodiff.ForwardAD;

/// <summary>
/// Process-wide registry of unary forward-mode (JVP) rules. Each rule
/// takes a primal+tangent <see cref="Dual{T}"/> and returns the
/// primal+tangent of <c>op(input)</c>. Pre-populated with the rules
/// from <see cref="DualOps{T}"/> for the standard pointwise ops;
/// callers can register additional rules for their own ops without
/// touching <see cref="DualOps{T}"/>.
/// </summary>
/// <remarks>
/// <para><b>Why a registry vs. static methods on DualOps:</b></para>
/// <para>
/// The static-method form forces every callsite to know the op
/// statically (e.g. <c>DualOps&lt;float&gt;.Add(engine, a, b)</c>).
/// A registry lets a forward-AD wrapper around an arbitrary
/// <see cref="IEngine"/> dispatch JVP rules by op name at runtime —
/// enabling <c>vmap(grad(fn))</c> and <c>jvp(fn)</c> over user code
/// that calls <c>engine.TensorAdd(...)</c> directly without the user
/// having to manually translate every call to the
/// <see cref="DualOps{T}"/> form. Issue #214 calls this out
/// explicitly: "Forward-mode op rules registered through
/// <c>OpRegistry</c>".
/// </para>
/// <para><b>Type-erased entries:</b></para>
/// <para>
/// Rules are stored as <see cref="Delegate"/> and cast at lookup time
/// to <c>Func&lt;IEngine, Dual&lt;T&gt;, Dual&lt;T&gt;&gt;</c> /
/// <c>Func&lt;IEngine, Dual&lt;T&gt;, Dual&lt;T&gt;, Dual&lt;T&gt;&gt;</c>.
/// The cast fails with <see cref="InvalidOperationException"/> if a
/// rule registered for one element type is looked up under another —
/// callers should register one entry per concrete <c>T</c>.
/// </para>
/// </remarks>
public static class ForwardAdRegistry
{
    private static readonly ConcurrentDictionary<string, Delegate> _unary = new(StringComparer.Ordinal);
    private static readonly ConcurrentDictionary<string, Delegate> _binary = new(StringComparer.Ordinal);

    static ForwardAdRegistry()
    {
        // Float seed registrations matching the DualOps surface.
        RegisterUnary<float>("TensorNegate", DualOps<float>.Negate);
        RegisterUnary<float>("TensorSquare", DualOps<float>.Square);
        RegisterUnary<float>("TensorSqrt", DualOps<float>.Sqrt);
        RegisterUnary<float>("TensorExp", DualOps<float>.Exp);
        RegisterUnary<float>("TensorLog", DualOps<float>.Log);
        RegisterUnary<float>("TensorSin", DualOps<float>.Sin);
        RegisterUnary<float>("TensorCos", DualOps<float>.Cos);
        RegisterUnary<float>("Tanh", DualOps<float>.Tanh);
        RegisterUnary<float>("Sigmoid", DualOps<float>.Sigmoid);
        RegisterUnary<float>("ReLU", DualOps<float>.ReLU);
        RegisterUnary<float>("ReduceSum", DualOps<float>.Sum);
        RegisterUnary<float>("TensorMean", DualOps<float>.Mean);

        RegisterBinary<float>("TensorAdd", DualOps<float>.Add);
        RegisterBinary<float>("TensorSubtract", DualOps<float>.Subtract);
        RegisterBinary<float>("TensorMultiply", DualOps<float>.Multiply);
        RegisterBinary<float>("TensorDivide", DualOps<float>.Divide);
        RegisterBinary<float>("TensorMatMul", DualOps<float>.MatMul);

        // Double counterparts.
        RegisterUnary<double>("TensorNegate", DualOps<double>.Negate);
        RegisterUnary<double>("TensorSquare", DualOps<double>.Square);
        RegisterUnary<double>("TensorSqrt", DualOps<double>.Sqrt);
        RegisterUnary<double>("TensorExp", DualOps<double>.Exp);
        RegisterUnary<double>("TensorLog", DualOps<double>.Log);
        RegisterUnary<double>("TensorSin", DualOps<double>.Sin);
        RegisterUnary<double>("TensorCos", DualOps<double>.Cos);
        RegisterUnary<double>("Tanh", DualOps<double>.Tanh);
        RegisterUnary<double>("Sigmoid", DualOps<double>.Sigmoid);
        RegisterUnary<double>("ReLU", DualOps<double>.ReLU);
        RegisterUnary<double>("ReduceSum", DualOps<double>.Sum);
        RegisterUnary<double>("TensorMean", DualOps<double>.Mean);

        RegisterBinary<double>("TensorAdd", DualOps<double>.Add);
        RegisterBinary<double>("TensorSubtract", DualOps<double>.Subtract);
        RegisterBinary<double>("TensorMultiply", DualOps<double>.Multiply);
        RegisterBinary<double>("TensorDivide", DualOps<double>.Divide);
        RegisterBinary<double>("TensorMatMul", DualOps<double>.MatMul);
    }

    /// <summary>
    /// Registers a unary JVP rule under <paramref name="opName"/> for
    /// element type <typeparamref name="T"/>. Replaces any existing
    /// rule for the same op+type.
    /// </summary>
    public static void RegisterUnary<T>(string opName, Func<IEngine, Dual<T>, Dual<T>> rule)
    {
        if (opName is null) throw new ArgumentNullException(nameof(opName));
        if (rule is null) throw new ArgumentNullException(nameof(rule));
        _unary[Key<T>(opName)] = rule;
    }

    /// <summary>
    /// Registers a binary JVP rule under <paramref name="opName"/>.
    /// </summary>
    public static void RegisterBinary<T>(string opName, Func<IEngine, Dual<T>, Dual<T>, Dual<T>> rule)
    {
        if (opName is null) throw new ArgumentNullException(nameof(opName));
        if (rule is null) throw new ArgumentNullException(nameof(rule));
        _binary[Key<T>(opName)] = rule;
    }

    /// <summary>
    /// Looks up the unary JVP rule for <paramref name="opName"/> at
    /// element type <typeparamref name="T"/>. Returns null if no rule
    /// is registered.
    /// </summary>
    public static Func<IEngine, Dual<T>, Dual<T>>? GetUnary<T>(string opName)
    {
        if (opName is null) throw new ArgumentNullException(nameof(opName));
        if (_unary.TryGetValue(Key<T>(opName), out var d)) return (Func<IEngine, Dual<T>, Dual<T>>)d;
        return null;
    }

    /// <summary>
    /// Looks up the binary JVP rule for <paramref name="opName"/> at
    /// element type <typeparamref name="T"/>.
    /// </summary>
    public static Func<IEngine, Dual<T>, Dual<T>, Dual<T>>? GetBinary<T>(string opName)
    {
        if (opName is null) throw new ArgumentNullException(nameof(opName));
        if (_binary.TryGetValue(Key<T>(opName), out var d)) return (Func<IEngine, Dual<T>, Dual<T>, Dual<T>>)d;
        return null;
    }

    /// <summary>
    /// Returns true if a unary or binary rule is registered for
    /// <paramref name="opName"/> at element type
    /// <typeparamref name="T"/>.
    /// </summary>
    public static bool IsRegistered<T>(string opName)
        => _unary.ContainsKey(Key<T>(opName)) || _binary.ContainsKey(Key<T>(opName));

    /// <summary>
    /// Enumerates every registered op name across element types.
    /// Useful for completeness audits paired with
    /// <see cref="OpRegistry"/>'s differentiable set.
    /// </summary>
    public static IEnumerable<string> RegisteredOpNames()
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var k in _unary.Keys) names.Add(StripTypeSuffix(k));
        foreach (var k in _binary.Keys) names.Add(StripTypeSuffix(k));
        return names;
    }

    private static string Key<T>(string opName) => opName + "::" + typeof(T).FullName;

    private static string StripTypeSuffix(string key)
    {
        int idx = key.IndexOf("::", StringComparison.Ordinal);
        return idx < 0 ? key : key.Substring(0, idx);
    }
}
