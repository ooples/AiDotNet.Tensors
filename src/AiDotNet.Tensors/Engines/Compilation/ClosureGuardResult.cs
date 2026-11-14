namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>Outcome of <see cref="ClosureObjectiveGuard{T}.TryRebind"/>.</summary>
internal enum ClosureGuardResult
{
    /// <summary>The objective traced to the compiled structure; its leaf data is now in the plan, which may step.</summary>
    Matched,

    /// <summary>The objective traced to a different structure (a shape, op, or captured scalar changed). The plan
    /// must not step for this objective.</summary>
    Mismatched,

    /// <summary>The objective read one of its own values while probed (or threw), so no plan can replay it.</summary>
    Aborted,

    /// <summary>The objective captured state whose value cannot be compared, so no plan can safely replay it.</summary>
    Unsignable,
}
