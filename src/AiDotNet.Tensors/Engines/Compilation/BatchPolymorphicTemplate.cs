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

/// <summary>
/// A compiled inference graph that serves every batch size from two traces: the batch-polymorphic plan of
/// <c>torch.compile(dynamic=True)</c>, built the way torch's automatic-dynamic shapes are — the first trace is
/// static, the first trace at another batch marks what scales with it, and no batch after that is traced again.
/// </summary>
/// <remarks>
/// <para>Every tensor dimension and every integer a recorded delegate captured is either the same in both traces
/// or proportional to the batch (<c>value = k * batch</c>, the same integer <c>k</c> in both). Anything else — a
/// captured float that changed, a tensor made outside the graph whose contents differ, a different graph — means
/// the forward is not batch-polymorphic, and every batch keeps its own trace.</para>
/// <para><see cref="Instantiate"/> rebuilds the recorded graph at a new batch: fresh tensors at the scaled shapes,
/// each recorded delegate re-bound to them (its closure copied with the tensors and scaled integers replaced), and
/// the result compiled by the same pipeline a trace goes through, so the memory planner, kernel specialization and
/// every backend's lowering see an ordinary graph of the new shape.</para>
/// </remarks>
internal sealed class BatchPolymorphicTemplate<T>
{
    private readonly InferenceTraceRecipe<T> _recipe;
    private readonly int _batchDim;
    // Per slot: per dim, 0 = static, k > 0 = k * batch.
    private readonly int[][] _slotCoefficients;
    // Leaves other than the input: null = shared constant; otherwise the value every element holds.
    private readonly Dictionary<int, T> _uniformLeaves;
    private readonly CaptureFit _fit;

    private BatchPolymorphicTemplate(InferenceTraceRecipe<T> recipe, int batchDim, int[][] slotCoefficients,
        Dictionary<int, T> uniformLeaves, CaptureFit fit)
    {
        _recipe = recipe;
        _batchDim = batchDim;
        _slotCoefficients = slotCoefficients;
        _uniformLeaves = uniformLeaves;
        _fit = fit;
    }

    /// <summary>The input shape this template was built from; only <see cref="BatchDimension"/> may vary.</summary>
    internal int[] TemplateInputShape => _recipe.InputShape;

    internal int BatchDimension => _batchDim;

    /// <summary>Whether this template can build a plan for <paramref name="inputShape"/>.</summary>
    internal bool CanServe(int[] inputShape)
    {
        var template = _recipe.InputShape;
        if (inputShape.Length != template.Length || inputShape[_batchDim] < 1) return false;
        for (int i = 0; i < template.Length; i++)
            if (i != _batchDim && inputShape[i] != template[i]) return false;
        return true;
    }

    /// <summary>
    /// Builds a template from two traces of one forward whose inputs differ only in <paramref name="batchDim"/>,
    /// or returns null with the reason the forward is not batch-polymorphic.
    /// </summary>
    internal static BatchPolymorphicTemplate<T>? TryCreate(
        InferenceTraceRecipe<T> first, InferenceTraceRecipe<T> second, int batchDim, out string? reason)
    {
        reason = first.Failure ?? second.Failure;
        if (reason is not null) return null;
        int b1 = first.InputShape[batchDim], b2 = second.InputShape[batchDim];
        if (b1 == b2) { reason = "both traces have the same batch"; return null; }
        if (first.Nodes.Count != second.Nodes.Count) { reason = "the traces record different graphs"; return null; }
        if (first.Slots.Count != second.Slots.Count || first.InputSlot != second.InputSlot || first.OutputSlot != second.OutputSlot)
        { reason = "the traces touch different tensors"; return null; }

        var coefficients = new int[first.Slots.Count][];
        for (int s = 0; s < first.Slots.Count; s++)
        {
            if (first.SlotKinds[s] != second.SlotKinds[s]) { reason = $"slot {s} changes kind"; return null; }
            var c = FitShape(first.SlotShapes[s], second.SlotShapes[s], b1, b2);
            if (c is null) { reason = $"slot {s} shape [{string.Join(",", first.SlotShapes[s])}] -> [{string.Join(",", second.SlotShapes[s])}] does not scale with the batch"; return null; }
            coefficients[s] = c;
        }
        if (!IsBatchScaledInput(coefficients[first.InputSlot], batchDim)) { reason = "the input's batch dim does not scale"; return null; }

        var uniform = new Dictionary<int, T>();
        for (int s = 0; s < first.Slots.Count; s++)
        {
            if (first.SlotKinds[s] != InferenceTraceRecipe<T>.SlotKind.Leaf || s == first.InputSlot) continue;
            var a = first.Slots[s]; var b = second.Slots[s];
            if (ReferenceEquals(a, b)) continue;                     // a constant both traces read
            if (Array.TrueForAll(coefficients[s], k => k == 0) && SameContents(a, b)) continue;
            if (TryUniformValue(a, out var va) && TryUniformValue(b, out var vb)
                && EqualityComparer<T>.Default.Equals(va, vb))
            {
                uniform[s] = vb;                                      // e.g. a zero initial state made per call
                continue;
            }
            reason = $"leaf slot {s} is made per call with batch-dependent contents";
            return null;
        }

        var fit = new CaptureFit(first, second, b1, b2);
        for (int n = 0; n < first.Nodes.Count; n++)
        {
            var na = first.Nodes[n]; var nb = second.Nodes[n];
            if (na.OpType != nb.OpType || na.OpName != nb.OpName || na.InputCount != nb.InputCount)
            { reason = $"node {n} is {na.OpName} in one trace and {nb.OpName} in the other"; return null; }
            if (!SameNode(first, na.ExternalPrerequisite, second, nb.ExternalPrerequisite))
            { reason = $"node {n} ({na.OpName}) mutates a different predecessor"; return null; }
            var ia = na.GetInputsArray(); var ib = nb.GetInputsArray();
            for (int i = 0; i < ia.Length; i++)
                if (first.SlotOf[ia[i]] != second.SlotOf[ib[i]]) { reason = $"node {n} ({na.OpName}) reads different tensors"; return null; }
            if (!fit.Match(na.Execute, nb.Execute, na.OpName) ||
                !fit.Match(na.SavedState, nb.SavedState, na.OpName) ||
                !fit.Match(na.BackwardFn, nb.BackwardFn, na.OpName))
            { reason = fit.Failure; return null; }
        }

        return new BatchPolymorphicTemplate<T>(second, batchDim, coefficients, uniform, fit);
    }

    /// <summary>
    /// Builds and compiles the recorded graph at the batch of <paramref name="inputShape"/> without tracing.
    /// </summary>
    internal CompiledInferencePlan<T> Instantiate(int[] inputShape)
    {
        if (!CanServe(inputShape))
            throw new ArgumentException($"Shape [{string.Join(",", inputShape)}] differs from the template outside the batch dim.", nameof(inputShape));
        int batch = inputShape[_batchDim];
        var recipe = _recipe;

        var tensorMap = new Dictionary<object, object>(ReferenceEqualityComparer<object>.Instance);
        for (int s = 0; s < recipe.Slots.Count; s++)
        {
            var original = recipe.Slots[s];
            bool constant = recipe.SlotKinds[s] == InferenceTraceRecipe<T>.SlotKind.Leaf
                            && s != recipe.InputSlot && !_uniformLeaves.ContainsKey(s);
            if (constant) continue;
            var shape = ScaledShape(recipe.SlotShapes[s], _slotCoefficients[s], batch);
            Tensor<T> replacement;
            if (_uniformLeaves.TryGetValue(s, out var value))
            {
                replacement = new Tensor<T>(shape);
                replacement.Fill(value);
            }
            else
            {
                replacement = TensorAllocator.RentUninitialized<T>(shape);
            }
            tensorMap[original] = replacement;
        }

        var cloner = new CaptureCloner(tensorMap, _fit, batch);
        using var scope = GraphMode.EnableInference();
        scope.BindEngineIfUnset(recipe.Engine);
        var nodeMap = new Dictionary<ILazyNode, ILazyNode>(ReferenceEqualityComparer<ILazyNode>.Instance);
        foreach (var node in recipe.Nodes)
        {
            var inputs = node.GetInputsArray();
            var mapped = new Tensor<T>[inputs.Length];
            for (int i = 0; i < inputs.Length; i++) mapped[i] = Map(tensorMap, inputs[i]);
            var output = Map(tensorMap, node.Output);
            var execute = (Action<IEngine, Tensor<T>>)cloner.Clone(node.Execute)!;
            var backward = (BackwardFunction<T>?)cloner.Clone(node.BackwardFn);
            var saved = (object[]?)cloner.Clone(node.SavedState);
            ILazyNode? prerequisite = node.ExternalPrerequisite is { } pre ? nodeMap[pre] : null;

            var copy = node.InputsOverflow is null && node.InputCount <= 2 && prerequisite is null
                ? node.InputCount == 1
                    ? new LazyNode<T>(node.OpType, node.OpName, mapped[0], output, execute, backward, saved)
                    : new LazyNode<T>(node.OpType, node.OpName, mapped[0], mapped[1], output, execute, backward, saved)
                : new LazyNode<T>(node.OpType, node.OpName, mapped, output, execute, backward, saved, prerequisite);
            output.LazySource = copy;
            nodeMap[node] = copy;
            scope.AddInstantiatedNode(copy);
        }

        var explicitOutput = Map(tensorMap, recipe.Slots[recipe.OutputSlot]);
        var input = Map(tensorMap, recipe.Slots[recipe.InputSlot]);
        return scope.CompileInference(explicitOutput, new[] { input });
    }

    private static Tensor<T> Map(Dictionary<object, object> map, Tensor<T> tensor)
        => map.TryGetValue(tensor, out var mapped) ? (Tensor<T>)mapped : tensor;

    private static int[] ScaledShape(int[] shape, int[] coefficients, int batch)
    {
        var scaled = (int[])shape.Clone();
        for (int d = 0; d < scaled.Length; d++)
            if (coefficients[d] != 0) scaled[d] = checked(coefficients[d] * batch);
        return scaled;
    }

    private static bool IsBatchScaledInput(int[] coefficients, int batchDim)
    {
        for (int d = 0; d < coefficients.Length; d++)
            if ((coefficients[d] != 0) != (d == batchDim)) return false;
        return coefficients[batchDim] == 1;
    }

    /// <summary>Per dim: 0 when the dims agree, k when both equal k times their batch, null otherwise.</summary>
    private static int[]? FitShape(int[] a, int[] b, int b1, int b2)
    {
        if (a.Length != b.Length) return null;
        var c = new int[a.Length];
        for (int d = 0; d < a.Length; d++)
        {
            if (a[d] == b[d]) continue;
            if (!CaptureFit.TryScale(a[d], b[d], b1, b2, out var k)) return null;
            c[d] = (int)k;
        }
        return c;
    }

    private static bool SameNode(InferenceTraceRecipe<T> first, ILazyNode? a, InferenceTraceRecipe<T> second, ILazyNode? b)
    {
        if (a is null || b is null) return a is null && b is null;
        return first.NodeIndex.TryGetValue(a, out var ia) && second.NodeIndex.TryGetValue(b, out var ib) && ia == ib;
    }

    private static bool SameContents(Tensor<T> a, Tensor<T> b)
    {
        if (!InferenceTraceRecipe<T>.ShapesEqual(a._shape, b._shape)) return false;
        var sa = a.AsSpan(); var sb = b.AsSpan();
        var cmp = EqualityComparer<T>.Default;
        for (int i = 0; i < sa.Length; i++) if (!cmp.Equals(sa[i], sb[i])) return false;
        return true;
    }

    private static bool TryUniformValue(Tensor<T> t, out T value)
    {
        var span = t.AsSpan();
        value = span.Length > 0 ? span[0] : default!;
        if (span.Length == 0) return false;
        var cmp = EqualityComparer<T>.Default;
        for (int i = 1; i < span.Length; i++) if (!cmp.Equals(span[i], value)) return false;
        return true;
    }

    /// <summary>
    /// Matches what two traces' delegates captured and records which integers scale with the batch.
    /// </summary>
    internal sealed class CaptureFit
    {
        private readonly InferenceTraceRecipe<T> _first;
        private readonly InferenceTraceRecipe<T> _second;
        private readonly int _b1;
        private readonly int _b2;
        private readonly Dictionary<object, object> _pairs = new(ReferenceEqualityComparer<object>.Instance);
        private string _op = "";

        // Keyed by the second trace's objects, which the template instantiates from.
        internal readonly Dictionary<(object Owner, FieldInfo Field), long> FieldCoefficients = new(FieldKeyComparer.Instance);
        internal readonly Dictionary<object, long[]> ArrayCoefficients = new(ReferenceEqualityComparer<object>.Instance);
        internal string? Failure;

        internal CaptureFit(InferenceTraceRecipe<T> first, InferenceTraceRecipe<T> second, int b1, int b2)
        {
            _first = first; _second = second; _b1 = b1; _b2 = b2;
        }

        internal static bool TryScale(long a, long b, int b1, int b2, out long k)
        {
            k = 0;
            if (a % b1 != 0 || b % b2 != 0) return false;
            k = a / b1;
            return k > 0 && k == b / b2;
        }

        internal bool Match(object? a, object? b, string op)
        {
            _op = op;
            return MatchCore(a, b, "capture");
        }

        private bool Fail(string path, string why)
        {
            Failure = $"{_op}: {path} {why}";
            return false;
        }

        private bool MatchCore(object? a, object? b, string path)
        {
            if (a is null || b is null) return a is null && b is null || Fail(path, "is null in one trace");
            if (ReferenceEquals(a, b)) return true;
            var type = a.GetType();
            if (type != b.GetType()) return Fail(path, $"is {type.Name} in one trace and {b.GetType().Name} in the other");

            if (a is Tensor<T> ta) return MatchTensor(ta, (Tensor<T>)b, path);
            if (CaptureCloner.IsTensor(type)) return Fail(path, $"is a different {type.Name} per trace");
            if (type.IsPrimitive || type.IsEnum || a is string || a is decimal)
            {
                if (a.Equals(b)) return true;
                return Fail(path, $"captured {a} in one trace and {b} in the other");
            }
            if (_pairs.TryGetValue(a, out var paired))
                return ReferenceEquals(paired, b) || Fail(path, "aliases differently");
            if (!type.IsValueType) _pairs[a] = b;

            if (a is Delegate da)
            {
                var db = (Delegate)b;
                if (da.GetInvocationList().Length != 1 || db.GetInvocationList().Length != 1) return Fail(path, "is a multicast delegate");
                if (da.Method != db.Method) return Fail(path, "calls different methods");
                return MatchCore(da.Target, db.Target, path + "." + da.Method.Name);
            }
            if (a is Array arrA)
            {
                var arrB = (Array)b;
                if (arrA.Rank != 1 || arrA.Length != arrB.Length) return Fail(path, "has different lengths");
                var et = type.GetElementType()!;
                if (et == typeof(int) || et == typeof(long))
                {
                    long[]? coeffs = null;
                    for (int i = 0; i < arrA.Length; i++)
                    {
                        long va = Convert.ToInt64(arrA.GetValue(i)), vb = Convert.ToInt64(arrB.GetValue(i));
                        if (va == vb) continue;
                        if (!TryScale(va, vb, _b1, _b2, out var k)) return Fail($"{path}[{i}]", $"captured {va} and {vb}, not a batch multiple");
                        (coeffs ??= new long[arrA.Length])[i] = k;
                    }
                    if (coeffs is not null) ArrayCoefficients[b] = coeffs;
                    return true;
                }
                for (int i = 0; i < arrA.Length; i++)
                    if (!MatchCore(arrA.GetValue(i), arrB.GetValue(i), $"{path}[{i}]")) return false;
                return true;
            }
            if (!CaptureCloner.IsDescendable(type))
                return Fail(path, $"is a different {type.FullName} per trace");

            foreach (var field in CaptureCloner.FieldsOf(type))
            {
                object? fa = field.GetValue(a), fb = field.GetValue(b);
                var ft = field.FieldType;
                if ((ft == typeof(int) || ft == typeof(long)) && !Equals(fa, fb))
                {
                    if (type.IsValueType) return Fail(path + "." + field.Name, "is a batch-dependent integer inside a struct");
                    long va = Convert.ToInt64(fa), vb = Convert.ToInt64(fb);
                    if (!TryScale(va, vb, _b1, _b2, out var k)) return Fail(path + "." + field.Name, $"captured {va} and {vb}, not a batch multiple");
                    FieldCoefficients[(b, field)] = k;
                    continue;
                }
                if (!MatchCore(fa, fb, path + "." + field.Name)) return false;
            }
            return true;
        }

        private bool MatchTensor(Tensor<T> a, Tensor<T> b, string path)
        {
            bool sa = _first.SlotOf.TryGetValue(a, out var slotA);
            bool sb = _second.SlotOf.TryGetValue(b, out var slotB);
            if (sa || sb) return sa && sb && slotA == slotB || Fail(path, "captures different graph tensors");
            // A tensor the delegate holds outside the graph: the same values in both traces is a constant.
            if (SameContents(a, b)) return true;
            return Fail(path, "captures a per-call tensor whose contents differ");
        }
    }

    /// <summary>Copies captured state for a new batch: graph tensors replaced, scaled integers rescaled.</summary>
    internal sealed class CaptureCloner
    {
        private static readonly MethodInfo s_memberwiseClone =
            typeof(object).GetMethod("MemberwiseClone", BindingFlags.Instance | BindingFlags.NonPublic)!;
        private static readonly Dictionary<Type, FieldInfo[]> s_fields = new();

        private readonly Dictionary<object, object> _tensorMap;
        private readonly CaptureFit _fit;
        private readonly int _batch;
        private readonly Dictionary<object, object> _clones = new(ReferenceEqualityComparer<object>.Instance);

        internal CaptureCloner(Dictionary<object, object> tensorMap, CaptureFit fit, int batch)
        {
            _tensorMap = tensorMap; _fit = fit; _batch = batch;
        }

        internal static bool IsTensor(Type type)
        {
            for (var t = type; t is not null; t = t.BaseType)
                if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(TensorBase<>)) return true;
            return false;
        }

        /// <summary>Closure classes, value types and tuples: copied field by field. Everything else is shared.</summary>
        internal static bool IsDescendable(Type type)
        {
            if (type.IsValueType) return true;
            if (type.IsDefined(typeof(CompilerGeneratedAttribute), false) || type.Name.Contains("<>")) return true;
            return type.Namespace == "System" && type.Name.StartsWith("Tuple`", StringComparison.Ordinal);
        }

        internal static FieldInfo[] FieldsOf(Type type)
        {
            lock (s_fields)
            {
                if (!s_fields.TryGetValue(type, out var fields))
                {
                    fields = type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                    Array.Sort(fields, (x, y) => x.MetadataToken.CompareTo(y.MetadataToken));
                    s_fields[type] = fields;
                }
                return fields;
            }
        }

        internal object? Clone(object? o)
        {
            if (o is null) return null;
            if (_tensorMap.TryGetValue(o, out var mapped)) return mapped;
            var type = o.GetType();
            if (type.IsPrimitive || type.IsEnum || o is string || IsTensor(type)) return o;
            if (_clones.TryGetValue(o, out var done)) return done;

            if (o is Delegate d)
            {
                var target = Clone(d.Target);
                var copy = ReferenceEquals(target, d.Target) ? d : Delegate.CreateDelegate(d.GetType(), target, d.Method);
                _clones[o] = copy;
                return copy;
            }
            if (o is Array array)
            {
                if (_fit.ArrayCoefficients.TryGetValue(o, out var coeffs))
                {
                    var scaled = (Array)array.Clone();
                    for (int i = 0; i < coeffs.Length; i++)
                    {
                        if (coeffs[i] == 0) continue;
                        long v = coeffs[i] * _batch;
                        scaled.SetValue(type.GetElementType() == typeof(int) ? (object)checked((int)v) : v, i);
                    }
                    _clones[o] = scaled;
                    return scaled;
                }
                var et = type.GetElementType()!;
                if (et.IsPrimitive || et.IsEnum || et == typeof(string)) return o;
                Array? copy = null;
                for (int i = 0; i < array.Length; i++)
                {
                    var e = array.GetValue(i);
                    var ce = Clone(e);
                    if (ReferenceEquals(ce, e)) continue;
                    if (copy is null) { copy = (Array)array.Clone(); _clones[o] = copy; }
                    copy.SetValue(ce, i);
                }
                if (copy is null) _clones[o] = o;
                return copy ?? o;
            }
            if (!IsDescendable(type)) return o;

            // Copy first and register it, so a closure chain that refers back to itself resolves to the copy.
            var clone = s_memberwiseClone.Invoke(o, null)!;
            if (!type.IsValueType) _clones[o] = clone;
            foreach (var field in FieldsOf(type))
            {
                if (_fit.FieldCoefficients.TryGetValue((o, field), out var k))
                {
                    long v = k * _batch;
                    field.SetValue(clone, field.FieldType == typeof(int) ? (object)checked((int)v) : v);
                    continue;
                }
                var value = field.GetValue(o);
                var cloned = Clone(value);
                if (!ReferenceEquals(cloned, value) || field.FieldType.IsValueType && !field.FieldType.IsPrimitive && !field.FieldType.IsEnum)
                    field.SetValue(clone, cloned);
            }
            return clone;
        }
    }

    private sealed class FieldKeyComparer : IEqualityComparer<(object Owner, FieldInfo Field)>
    {
        internal static readonly FieldKeyComparer Instance = new();
        public bool Equals((object Owner, FieldInfo Field) x, (object Owner, FieldInfo Field) y)
            => ReferenceEquals(x.Owner, y.Owner) && x.Field == y.Field;
        public int GetHashCode((object Owner, FieldInfo Field) key)
            => RuntimeHelpers.GetHashCode(key.Owner) * 31 + key.Field.GetHashCode();
    }
}
