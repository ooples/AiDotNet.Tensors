using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// One inference trace, recorded before compilation mutates it: the nodes in recording order, every tensor
/// they touch numbered as a slot, and each slot's shape at trace time.
/// </summary>
/// <remarks>
/// Two recipes of the same forward at different batch sizes are what <see cref="BatchPolymorphicTemplate{T}"/>
/// compares. Compilation clears lazy sources and rebinds storage but never changes a recorded node's delegate,
/// its operands or their shapes, so the recipe stays comparable after its plan is built from the same nodes.
/// </remarks>
internal sealed class InferenceTraceRecipe<T>
{
    internal enum SlotKind { Leaf, NodeOutput }

    internal readonly List<LazyNode<T>> Nodes = new();
    internal readonly List<Tensor<T>> Slots = new();
    internal readonly List<int[]> SlotShapes = new();
    internal readonly List<SlotKind> SlotKinds = new();
    internal readonly Dictionary<object, int> SlotOf = new(ReferenceEqualityComparer<object>.Instance);
    // A slot's backing array, which a recorded delegate may capture in place of the tensor.
    internal readonly Dictionary<object, int> BackingSlotOf = new(ReferenceEqualityComparer<object>.Instance);
    internal readonly Dictionary<ILazyNode, int> NodeIndex = new(ReferenceEqualityComparer<ILazyNode>.Instance);
    internal int InputSlot = -1;
    internal int OutputSlot = -1;
    internal int[] InputShape = Array.Empty<int>();
    internal IEngine Engine = null!;

    /// <summary>Why this trace cannot seed a template, or null when it can.</summary>
    internal string? Failure;

    internal static InferenceTraceRecipe<T> Capture(LazyTensorScope scope, Tensor<T> explicitOutput, int[] inputShape)
    {
        var recipe = new InferenceTraceRecipe<T> { InputShape = (int[])inputShape.Clone(), Engine = scope.Engine };
        foreach (var node in scope.Nodes)
        {
            if (node is not LazyNode<T> typed)
                return recipe.Fail($"node {node.GetType().Name} crosses element types");
            recipe.NodeIndex[typed] = recipe.Nodes.Count;
            recipe.Nodes.Add(typed);
        }

        foreach (var node in recipe.Nodes)
        {
            var inputs = node.GetInputsArray();
            foreach (var input in inputs) recipe.AddSlot(input, SlotKind.Leaf);
            bool inPlace = ReferenceEquals(inputs[0], node.Output);
            if (!inPlace)
            {
                if (recipe.SlotOf.ContainsKey(node.Output))
                    return recipe.Fail($"{node.OpName} writes a tensor an earlier node already produced or read");
                foreach (var input in inputs)
                    if (ReferenceEquals(input.StorageIdentity, node.Output.StorageIdentity))
                        return recipe.Fail($"{node.OpName} is a view of its input");
                recipe.AddSlot(node.Output, SlotKind.NodeOutput);
            }
            if (node.ExternalPrerequisite is { } pre && !recipe.NodeIndex.ContainsKey(pre))
                return recipe.Fail($"{node.OpName} mutates a tensor produced outside the trace");
        }

        if (!recipe.SlotOf.TryGetValue(explicitOutput, out recipe.OutputSlot))
            return recipe.Fail("the forward's output is not a traced tensor");

        // The mutable input is the one leaf with the input's shape, as Compile selects it.
        for (int s = 0; s < recipe.Slots.Count; s++)
        {
            if (recipe.SlotKinds[s] != SlotKind.Leaf || !ShapesEqual(recipe.SlotShapes[s], inputShape)) continue;
            if (recipe.InputSlot >= 0) return recipe.Fail("two traced leaves have the input's shape");
            recipe.InputSlot = s;
        }
        if (recipe.InputSlot < 0) return recipe.Fail("no traced leaf has the input's shape");
        return recipe;
    }

    private void AddSlot(Tensor<T> tensor, SlotKind kind)
    {
        if (SlotOf.ContainsKey(tensor)) return;
        if (tensor.GetBackingArrayForCacheLookupUnsafe() is { } backing && !BackingSlotOf.ContainsKey(backing))
            BackingSlotOf[backing] = Slots.Count;
        SlotOf[tensor] = Slots.Count;
        Slots.Add(tensor);
        SlotShapes.Add((int[])tensor._shape.Clone());
        SlotKinds.Add(kind);
    }

    private InferenceTraceRecipe<T> Fail(string reason)
    {
        Failure = reason;
        return this;
    }

    internal static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }
}

