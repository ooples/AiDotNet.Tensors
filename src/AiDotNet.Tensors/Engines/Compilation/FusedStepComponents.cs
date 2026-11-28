using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// The recorded operations a fused step replaced, in execution order, saved as the fused step's only saved-state item.
/// The fused step runs a generated kernel that has no per-operation form; consumers that need the individual
/// operations (the ONNX exporter) read them from here. Each component keeps its own op name, inputs, output and saved
/// state, so it translates exactly as it would have without the fusion.
/// </summary>
internal sealed class FusedStepComponents<T>
{
    internal FusedStepComponents(IReadOnlyList<LazyNode<T>> nodes)
    {
        var steps = new CompiledStep<T>[nodes.Count];
        for (int i = 0; i < steps.Length; i++)
        {
            var node = nodes[i];
            steps[i] = new CompiledStep<T>(
                node.OpName, node.Execute, node.Output, node.GetInputsArray(), node.BackwardFn, node.SavedState);
        }
        Steps = steps;
    }

    /// <summary>The replaced operations, in order; the last one writes the fused step's output.</summary>
    internal CompiledStep<T>[] Steps { get; }

    /// <summary>The operations <paramref name="step"/> replaced, or null when it is not a fused step that saved them.</summary>
    internal static CompiledStep<T>[]? Of(CompiledStep<T> step)
        => step.SavedState is { Length: 1 } saved && saved[0] is FusedStepComponents<T> components ? components.Steps : null;
}
