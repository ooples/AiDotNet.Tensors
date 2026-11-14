namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Thrown when a storage-free probe trace (<see cref="GraphMode.EnableProbe"/>) would have to compute a value: the
/// traced objective reads one of its own intermediates on the host (<c>ToArray()</c>, <c>.Item</c>, a data-dependent
/// branch) or calls an op that needs its input's data while tracing. Such an objective cannot replay a fixed compiled
/// graph, so the caller runs that step eagerly.
/// </summary>
internal sealed class ObjectiveProbeAbortedException : InvalidOperationException
{
    internal ObjectiveProbeAbortedException(string opName)
        : base($"The objective read the value of '{opName}' while being probed, so it cannot replay a compiled plan.")
    {
        OpName = opName;
    }

    /// <summary>The recorded op whose value was requested.</summary>
    internal string OpName { get; }
}
