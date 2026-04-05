namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Interface for optimization passes over the lazy computation graph.
/// Each pass transforms the node list in-place (e.g., fusing adjacent nodes,
/// eliminating dead code, reordering for better cache locality).
/// </summary>
internal interface ILazyGraphOptimizationPass
{
    /// <summary>Human-readable name for diagnostics.</summary>
    string Name { get; }

    /// <summary>
    /// Runs the optimization pass. Returns the (potentially modified) node list.
    /// Passes may remove nodes (dead code), replace nodes (fusion), or reorder them.
    /// </summary>
    List<ILazyNode> Run(List<ILazyNode> nodes);
}
