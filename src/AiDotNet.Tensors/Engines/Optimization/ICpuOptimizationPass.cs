using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Interface for CPU-level optimization passes that operate on compiled plans.
/// Unlike <see cref="ILazyGraphOptimizationPass"/> which transforms lazy graph nodes,
/// these passes operate at a higher level — analyzing weight matrices, layer adjacency,
/// and backward graph structure to decide which computational transformations to apply.
///
/// Each pass is independently toggleable via <see cref="TensorCodecOptions"/>.
/// </summary>
internal interface ICpuOptimizationPass
{
    /// <summary>Human-readable name for diagnostics and benchmarking.</summary>
    string Name { get; }

    /// <summary>Whether this pass is enabled in the current configuration.</summary>
    bool IsEnabled { get; }

    /// <summary>
    /// Analyzes the compiled steps and returns an optimized version.
    /// The pass may replace, reorder, or fuse steps as needed.
    /// Returns null if no optimization was applicable.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="steps">The compiled forward steps from the plan.</param>
    /// <param name="engine">The engine for executing operations.</param>
    /// <returns>Optimized steps, or null if no optimization was applied.</returns>
    CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine);
}
