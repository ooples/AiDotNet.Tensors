using System;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Marks a member whose element-by-element tensor access is deliberate, exempting it from the element-wise gate
/// (TENS0010/TENS0011).
/// </summary>
/// <remarks>
/// The gate flags a <c>Tensor&lt;T&gt;</c>, <c>Vector&lt;T&gt;</c> or <c>Matrix&lt;T&gt;</c> indexer, or a per-element
/// <c>INumericOperations&lt;T&gt;</c> call, inside a loop. A member that must stay scalar - a reference implementation a
/// test compares against, or a loop over a handful of elements - says so here with the reason.
/// </remarks>
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor | AttributeTargets.Property | AttributeTargets.Class
    | AttributeTargets.Struct, Inherited = false)]
internal sealed class ScalarKernelAttribute : Attribute
{
    public ScalarKernelAttribute(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("State why the per-element access is intended.", nameof(reason));
        Reason = reason;
    }

    /// <summary>Why this member's per-element access is intended.</summary>
    public string Reason { get; }
}
