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
    // Captured arrays still all default at trace time: per-call scratch (see FindScratch).
    internal readonly HashSet<object> ScratchArrays = new(ReferenceEqualityComparer<object>.Instance);
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

        // Now, before the plan runs: a scratch array the delegate writes on replay is still all default here.
        var visited = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        foreach (var node in recipe.Nodes)
        {
            recipe.FindScratch(node.Execute, visited);
            recipe.FindScratch(node.SavedState, visited);
            recipe.FindScratch(node.BackwardFn, visited);
        }

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

    /// <summary>
    /// Records the arrays the recorded delegates captured that hold only default values at trace time: scratch made
    /// per call (<c>new float[n]</c>) that the replay writes, which a built plan needs its own copy of.
    /// </summary>
    private void FindScratch(object? o, HashSet<object> visited)
    {
        if (o is null) return;
        var type = o.GetType();
        if (type.IsPrimitive || type.IsEnum || o is string || ShapePolymorphicTemplate<T>.CaptureCloner.IsTensor(type)) return;
        if (!type.IsValueType && !visited.Add(o)) return;
        switch (o)
        {
            case Delegate d:
                foreach (var single in d.GetInvocationList()) FindScratch(single.Target, visited);
                return;
            case Array array:
                var et = type.GetElementType()!;
                if (et.IsPrimitive || et.IsEnum)
                {
                    if (!BackingSlotOf.ContainsKey(o) && IsAllDefault(array)) ScratchArrays.Add(o);
                    return;
                }
                foreach (var e in array) FindScratch(e, visited);
                return;
        }
        if (!ShapePolymorphicTemplate<T>.CaptureCloner.IsDescendable(type)) return;
        foreach (var field in ShapePolymorphicTemplate<T>.CaptureCloner.FieldsOf(type)) FindScratch(field.GetValue(o), visited);
    }

    private static bool IsAllDefault(Array array)
    {
        switch (array)
        {
            case float[] f: foreach (var v in f) if (v != 0f) return false; return true;
            case double[] f: foreach (var v in f) if (v != 0d) return false; return true;
            case int[] f: foreach (var v in f) if (v != 0) return false; return true;
            case long[] f: foreach (var v in f) if (v != 0) return false; return true;
            case byte[] f: foreach (var v in f) if (v != 0) return false; return true;
            case bool[] f: foreach (var v in f) if (v) return false; return true;
        }
        object zero = Activator.CreateInstance(array.GetType().GetElementType()!)!;
        foreach (var v in array) if (!zero.Equals(v)) return false;
        return true;
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

