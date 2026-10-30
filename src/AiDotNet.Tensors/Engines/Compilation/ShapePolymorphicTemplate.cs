using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A compiled inference graph that serves every value of its input's symbolic dims (batch, sequence, ...) from a
/// few traces: the shape-polymorphic plan of <c>torch.compile(dynamic=True)</c>, learned the way torch's
/// automatic-dynamic shapes are. The first trace is the base; the first trace that changes one symbolic dim shows
/// what scales with that dim; once every dim a shape changes has been seen, that shape is built without tracing.
/// </summary>
/// <remarks>
/// <para>Every tensor dimension and every integer a recorded delegate captured is modelled as
/// <c>c * s1^p1 * s2^p2 ...</c> with each exponent 0 or 1: a value that changes when symbol <c>s</c> changes must
/// change in proportion to it (the same integer multiple in both traces). Anything else — a captured float that
/// changed, a tensor made outside the graph whose contents differ, a different graph — means the forward is not
/// polymorphic in that dim, and shapes that change it keep their own trace.</para>
/// <para><see cref="Instantiate"/> rebuilds the base trace at the new shape: fresh tensors at the scaled shapes,
/// each recorded delegate re-bound to them (its closure copied with the tensors and scaled integers replaced), and
/// the result compiled by the same pipeline a trace goes through, so the memory planner, kernel specialization and
/// every backend's lowering see an ordinary graph of the new shape.</para>
/// </remarks>
internal sealed class ShapePolymorphicTemplate<T>
{
    private readonly InferenceTraceRecipe<T> _base;
    private readonly int[] _symbolDims;
    // Per symbol (index into _symbolDims): what scales with it, or null until a trace has changed it.
    private readonly SymbolFit?[] _fits;
    private readonly string?[] _failures;

    internal ShapePolymorphicTemplate(InferenceTraceRecipe<T> baseRecipe, int[] symbolDims)
    {
        _base = baseRecipe;
        _symbolDims = (int[])symbolDims.Clone();
        _fits = new SymbolFit?[symbolDims.Length];
        _failures = new string?[symbolDims.Length];
    }

    /// <summary>Why the base trace or a symbol is not polymorphic, or null when nothing failed.</summary>
    internal string? Failure
    {
        get
        {
            if (_base.Failure is not null) return _base.Failure;
            for (int j = 0; j < _failures.Length; j++)
                if (_failures[j] is { } f) return $"dim {_symbolDims[j]}: {f}";
            return null;
        }
    }

    /// <summary>
    /// Whether every symbolic dim of <paramref name="shape"/> is at least 2. A size of 0 or 1 is specialized, as in
    /// torch: a forward legitimately records a different graph there (a size-1 operand turns a broadcast into a plain
    /// add, a kernel picks another path), so such a trace neither seeds a template nor is built from one.
    /// </summary>
    internal static bool IsGeneric(int[] shape, int[] symbolDims)
    {
        foreach (var d in symbolDims)
            if (d >= shape.Length || shape[d] < 2) return false;
        return true;
    }

    /// <summary>Whether this template can build a plan for <paramref name="inputShape"/> without a trace.</summary>
    internal bool CanServe(int[] inputShape)
    {
        if (_base.Failure is not null || !IsGeneric(inputShape, _symbolDims)) return false;
        var baseShape = _base.InputShape;
        if (inputShape.Length != baseShape.Length) return false;
        for (int d = 0; d < baseShape.Length; d++)
        {
            int j = Array.IndexOf(_symbolDims, d);
            if (inputShape[d] == baseShape[d]) continue;
            if (j < 0 || _fits[j] is null) return false;
        }
        return true;
    }

    /// <summary>
    /// Learns what scales with the one symbolic dim in which <paramref name="variant"/>'s input differs from the
    /// base. A trace that changes no symbolic dim, several, or a dim already learned teaches nothing and is ignored.
    /// </summary>
    internal void Observe(InferenceTraceRecipe<T> variant)
    {
        if (_base.Failure is not null || !IsGeneric(variant.InputShape, _symbolDims)) return;
        var a = _base.InputShape; var b = variant.InputShape;
        if (a.Length != b.Length) return;
        int changed = -1;
        for (int d = 0; d < a.Length; d++)
        {
            if (a[d] == b[d]) continue;
            int j = Array.IndexOf(_symbolDims, d);
            if (j < 0 || changed >= 0) return;
            changed = j;
        }
        if (changed < 0 || _fits[changed] is not null || _failures[changed] is not null) return;
        if (variant.Failure is not null) { _failures[changed] = variant.Failure; return; }

        var fit = new SymbolFit(_base, variant, a[_symbolDims[changed]], b[_symbolDims[changed]]);
        if (fit.Run()) _fits[changed] = fit;
        else _failures[changed] = fit.Failure;
    }

    /// <summary>Builds and compiles the base trace at <paramref name="inputShape"/> without tracing.</summary>
    internal CompiledInferencePlan<T> Instantiate(int[] inputShape)
    {
        if (!CanServe(inputShape))
            throw new ArgumentException($"Shape [{string.Join(",", inputShape)}] is not reachable from this template.", nameof(inputShape));
        var recipe = _base;
        var scale = new Scale(this, inputShape);

        var tensorMap = new Dictionary<object, object>(ReferenceEqualityComparer<object>.Instance);
        for (int s = 0; s < recipe.Slots.Count; s++)
        {
            var original = recipe.Slots[s];
            bool isInput = s == recipe.InputSlot;
            bool uniform = scale.UniformLeaf(s, out var value);
            if (recipe.SlotKinds[s] == InferenceTraceRecipe<T>.SlotKind.Leaf && !isInput && !uniform) continue;  // constant
            var shape = (int[])recipe.SlotShapes[s].Clone();
            for (int d = 0; d < shape.Length; d++) shape[d] = scale.Value(scale.SlotMask(s, d), shape[d]);
            Tensor<T> replacement;
            if (uniform)
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
        // A delegate that captured a replaced tensor's backing array gets the replacement's.
        foreach (var pair in recipe.BackingSlotOf)
        {
            if (!tensorMap.TryGetValue(recipe.Slots[pair.Value], out var replaced)) continue;
            if (((Tensor<T>)replaced).GetBackingArrayForCacheLookupUnsafe() is not { } backing)
                throw new InvalidOperationException($"slot {pair.Value}'s replacement has no backing array for a delegate that captured one.");
            tensorMap[pair.Key] = backing;
        }

        var cloner = new CaptureCloner(tensorMap, scale);
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
    /// The new value of every scaled site for one target shape: <c>base / prod(base symbols) * prod(new symbols)</c>
    /// over the symbols the site scales with.
    /// </summary>
    internal sealed class Scale
    {
        private readonly ShapePolymorphicTemplate<T> _t;
        private readonly long[] _baseSymbols;
        private readonly long[] _newSymbols;

        internal Scale(ShapePolymorphicTemplate<T> template, int[] inputShape)
        {
            _t = template;
            int n = template._symbolDims.Length;
            _baseSymbols = new long[n];
            _newSymbols = new long[n];
            for (int j = 0; j < n; j++)
            {
                _baseSymbols[j] = template._base.InputShape[template._symbolDims[j]];
                _newSymbols[j] = inputShape[template._symbolDims[j]];
            }
        }

        internal int SlotMask(int slot, int dim)
        {
            int mask = 0;
            for (int j = 0; j < _t._fits.Length; j++)
                if (_t._fits[j] is { } f && f.SlotScales(slot, dim)) mask |= 1 << j;
            return mask;
        }

        internal int FieldMask(object owner, FieldInfo field)
        {
            int mask = 0;
            for (int j = 0; j < _t._fits.Length; j++)
                if (_t._fits[j] is { } f && f.FieldScales(owner, field)) mask |= 1 << j;
            return mask;
        }

        internal int ElementMask(object array, int index)
        {
            int mask = 0;
            for (int j = 0; j < _t._fits.Length; j++)
                if (_t._fits[j] is { } f && f.ElementScales(array, index)) mask |= 1 << j;
            return mask;
        }

        /// <summary>A per-call scratch array of the base trace (all default when it was recorded).</summary>
        internal bool IsScratch(object array) => _t._base.ScratchArrays.Contains(array);

        internal int ScratchLengthMask(object array)
        {
            int mask = 0;
            for (int j = 0; j < _t._fits.Length; j++)
                if (_t._fits[j] is { } f && f.ScratchScales(array)) mask |= 1 << j;
            return mask;
        }

        internal bool HasScaledElements(object array)
        {
            foreach (var f in _t._fits) if (f is not null && f.HasScaledElements(array)) return true;
            return false;
        }

        internal int Value(int mask, int baseValue) => checked((int)Value(mask, (long)baseValue));

        internal long Value(int mask, long baseValue)
        {
            if (mask == 0) return baseValue;
            long v = baseValue;
            for (int j = 0; j < _baseSymbols.Length; j++)
            {
                if ((mask & (1 << j)) == 0) continue;
                // A value that scales with several dims must be a multiple of their product to be c * s1 * s2.
                if (v % _baseSymbols[j] != 0)
                    throw new ArithmeticException($"{baseValue} is not a multiple of the symbols it scales with.");
                v /= _baseSymbols[j];
            }
            for (int j = 0; j < _newSymbols.Length; j++)
                if ((mask & (1 << j)) != 0) v = checked(v * _newSymbols[j]);
            return v;
        }

        /// <summary>A per-call leaf every element of which holds one value in every trace.</summary>
        internal bool UniformLeaf(int slot, out T value)
        {
            foreach (var f in _t._fits)
            {
                if (f is not null && f.UniformLeaves.TryGetValue(slot, out var found))
                {
                    value = found;
                    return true;
                }
            }
            value = default!;
            return false;
        }
    }

    /// <summary>
    /// What changes between the base trace and a trace that changed one symbol. Keyed by the base trace's
    /// objects, which the template instantiates from.
    /// </summary>
    private sealed class SymbolFit
    {
        private readonly InferenceTraceRecipe<T> _first;
        private readonly InferenceTraceRecipe<T> _second;
        private readonly long _s1;
        private readonly long _s2;
        private readonly Dictionary<object, object> _pairs = new(ReferenceEqualityComparer<object>.Instance);
        private readonly HashSet<(int Slot, int Dim)> _slotDims = new();
        private readonly HashSet<(object Owner, FieldInfo Field)> _fields = new(FieldKeyComparer.Instance);
        private readonly Dictionary<object, HashSet<int>> _elements = new(ReferenceEqualityComparer<object>.Instance);
        internal readonly Dictionary<int, T> UniformLeaves = new();
        private string _op = "";

        internal string? Failure;

        internal SymbolFit(InferenceTraceRecipe<T> first, InferenceTraceRecipe<T> second, long s1, long s2)
        {
            _first = first; _second = second; _s1 = s1; _s2 = s2;
        }

        internal bool SlotScales(int slot, int dim) => _slotDims.Contains((slot, dim));
        internal bool FieldScales(object owner, FieldInfo field) => _fields.Contains((owner, field));
        internal bool ElementScales(object array, int index) => _elements.TryGetValue(array, out var set) && set.Contains(index);
        internal bool HasScaledElements(object array) => _elements.ContainsKey(array);
        internal bool ScratchScales(object array) => _scaledScratch.Contains(array);

        private readonly HashSet<object> _scaledScratch = new(ReferenceEqualityComparer<object>.Instance);

        private bool Scales(long a, long b) => a % _s1 == 0 && b % _s2 == 0 && a / _s1 > 0 && a / _s1 == b / _s2;

        private bool Fail(string why)
        {
            Failure = why;
            return false;
        }

        internal bool Run()
        {
            var first = _first; var second = _second;
            if (first.Nodes.Count != second.Nodes.Count) return Fail("the traces record different graphs");
            if (first.Slots.Count != second.Slots.Count || first.InputSlot != second.InputSlot || first.OutputSlot != second.OutputSlot)
                return Fail("the traces touch different tensors");

            for (int s = 0; s < first.Slots.Count; s++)
            {
                if (first.SlotKinds[s] != second.SlotKinds[s]) return Fail($"slot {s} changes kind");
                int[] a = first.SlotShapes[s], b = second.SlotShapes[s];
                if (a.Length != b.Length) return Fail($"slot {s} changes rank");
                for (int d = 0; d < a.Length; d++)
                {
                    if (a[d] == b[d]) continue;
                    if (!Scales(a[d], b[d]))
                        return Fail($"slot {s} shape [{string.Join(",", a)}] -> [{string.Join(",", b)}] does not scale with the dim");
                    _slotDims.Add((s, d));
                }
            }

            for (int s = 0; s < first.Slots.Count; s++)
            {
                if (first.SlotKinds[s] != InferenceTraceRecipe<T>.SlotKind.Leaf || s == first.InputSlot) continue;
                var a = first.Slots[s]; var b = second.Slots[s];
                if (ReferenceEquals(a, b)) continue;                      // a constant both traces read
                if (SameContents(a, b)) continue;                         // equal per-call constants: share the base's
                if (TryUniformValue(a, out var va) && TryUniformValue(b, out var vb) && EqualityComparer<T>.Default.Equals(va, vb))
                {
                    UniformLeaves[s] = va;                                // e.g. a zero initial state made per call
                    continue;
                }
                return Fail($"leaf slot {s} is made per call with shape-dependent contents");
            }

            for (int n = 0; n < first.Nodes.Count; n++)
            {
                var na = first.Nodes[n]; var nb = second.Nodes[n];
                if (na.OpType != nb.OpType || na.OpName != nb.OpName || na.InputCount != nb.InputCount)
                    return Fail($"node {n} is {na.OpName} in one trace and {nb.OpName} in the other");
                if (!SameNode(na.ExternalPrerequisite, nb.ExternalPrerequisite))
                    return Fail($"node {n} ({na.OpName}) mutates a different predecessor");
                var ia = na.GetInputsArray(); var ib = nb.GetInputsArray();
                for (int i = 0; i < ia.Length; i++)
                    if (first.SlotOf[ia[i]] != second.SlotOf[ib[i]]) return Fail($"node {n} ({na.OpName}) reads different tensors");
                _op = na.OpName;
                if (!Match(na.Execute, nb.Execute, "capture") ||
                    !Match(na.SavedState, nb.SavedState, "savedState") ||
                    !Match(na.BackwardFn, nb.BackwardFn, "backward"))
                    return false;
            }
            return CheckDivisible();
        }

        /// <summary>Every scaled base value is a whole multiple of the base symbol, so rescaling it is exact.</summary>
        private bool CheckDivisible()
        {
            foreach (var (slot, dim) in _slotDims)
                if (_first.SlotShapes[slot][dim] % _s1 != 0) return Fail($"slot {slot} dim {dim} is not a multiple of the base symbol");
            return true;
        }

        private bool SameNode(ILazyNode? a, ILazyNode? b)
        {
            if (a is null || b is null) return a is null && b is null;
            return _first.NodeIndex.TryGetValue(a, out var ia) && _second.NodeIndex.TryGetValue(b, out var ib) && ia == ib;
        }

        private bool FailAt(string path, string why) => Fail($"{_op}: {path} {why}");

        private bool Match(object? a, object? b, string path)
        {
            if (a is null || b is null) return a is null && b is null || FailAt(path, "is null in one trace");
            if (ReferenceEquals(a, b)) return true;
            var type = a.GetType();
            if (type != b.GetType()) return FailAt(path, $"is {type.Name} in one trace and {b.GetType().Name} in the other");

            if (a is Tensor<T> ta) return MatchTensor(ta, (Tensor<T>)b, path);
            if (CaptureCloner.IsTensor(type)) return FailAt(path, $"is a different {type.Name} per trace");
            if (type.IsPrimitive || type.IsEnum || a is string || a is decimal)
                return a.Equals(b) || FailAt(path, $"captured {a} in one trace and {b} in the other");
            if (_pairs.TryGetValue(a, out var paired))
                return ReferenceEquals(paired, b) || FailAt(path, "aliases differently");
            if (!type.IsValueType) _pairs[a] = b;

            if (a is Delegate da)
            {
                var db = (Delegate)b;
                if (da.GetInvocationList().Length != 1 || db.GetInvocationList().Length != 1) return FailAt(path, "is a multicast delegate");
                if (da.Method != db.Method) return FailAt(path, "calls different methods");
                return Match(da.Target, db.Target, path + "." + da.Method.Name);
            }
            // A graph tensor's backing array, captured instead of the tensor: it is that slot, not a constant, even
            // when the two traces' buffers happen to hold the same values.
            bool ba = _first.BackingSlotOf.TryGetValue(a, out var backingA);
            bool bb = _second.BackingSlotOf.TryGetValue(b, out var backingB);
            if (ba || bb) return ba && bb && backingA == backingB || FailAt(path, "captures different graph buffers");
            // Per-call scratch in both traces: a built plan gets a fresh one, as a trace would make.
            if (_first.ScratchArrays.Contains(a) && _second.ScratchArrays.Contains(b))
            {
                int la = ((Array)a).Length, lb = ((Array)b).Length;
                if (la == lb) return true;
                if (!Scales(la, lb)) return FailAt(path, $"is scratch of {la} and {lb} elements, not a multiple of the dim");
                _scaledScratch.Add(a);
                return true;
            }
            if (a is Array arrA)
            {
                var arrB = (Array)b;
                if (arrA.Rank != 1 || arrA.Length != arrB.Length) return FailAt(path, "has different lengths");
                var et = type.GetElementType()!;
                if (et == typeof(int) || et == typeof(long))
                {
                    for (int i = 0; i < arrA.Length; i++)
                    {
                        long va = Convert.ToInt64(arrA.GetValue(i)), vb = Convert.ToInt64(arrB.GetValue(i));
                        if (va == vb) continue;
                        if (!Scales(va, vb)) return FailAt($"{path}[{i}]", $"captured {va} and {vb}, not a multiple of the dim");
                        if (!_elements.TryGetValue(a, out var set)) _elements[a] = set = new HashSet<int>();
                        set.Add(i);
                    }
                    return true;
                }
                for (int i = 0; i < arrA.Length; i++)
                    if (!Match(arrA.GetValue(i), arrB.GetValue(i), $"{path}[{i}]")) return false;
                return true;
            }
            if (!CaptureCloner.IsDescendable(type))
                return FailAt(path, $"is a different {type.FullName} per trace");

            foreach (var field in CaptureCloner.FieldsOf(type))
            {
                object? fa = field.GetValue(a), fb = field.GetValue(b);
                var ft = field.FieldType;
                if ((ft == typeof(int) || ft == typeof(long)) && !Equals(fa, fb))
                {
                    if (type.IsValueType) return FailAt(path + "." + field.Name, "is a shape-dependent integer inside a struct");
                    long va = Convert.ToInt64(fa), vb = Convert.ToInt64(fb);
                    if (!Scales(va, vb)) return FailAt(path + "." + field.Name, $"captured {va} and {vb}, not a multiple of the dim");
                    _fields.Add((a, field));
                    continue;
                }
                if (!Match(fa, fb, path + "." + field.Name)) return false;
            }
            return true;
        }

        private bool MatchTensor(Tensor<T> a, Tensor<T> b, string path)
        {
            bool sa = _first.SlotOf.TryGetValue(a, out var slotA);
            bool sb = _second.SlotOf.TryGetValue(b, out var slotB);
            if (sa || sb) return sa && sb && slotA == slotB || FailAt(path, "captures different graph tensors");
            // A tensor the delegate holds outside the graph: the same values in both traces is a constant.
            return SameContents(a, b) || FailAt(path, "captures a per-call tensor whose contents differ");
        }
    }

    /// <summary>Copies captured state for a new shape: graph tensors replaced, scaled integers rescaled.</summary>
    internal sealed class CaptureCloner
    {
        private static readonly MethodInfo s_memberwiseClone =
            typeof(object).GetMethod("MemberwiseClone", BindingFlags.Instance | BindingFlags.NonPublic)!;
        private static readonly Dictionary<Type, FieldInfo[]> s_fields = new();

        private readonly Dictionary<object, object> _tensorMap;
        private readonly Scale _scale;
        private readonly Dictionary<object, object> _clones = new(ReferenceEqualityComparer<object>.Instance);

        internal CaptureCloner(Dictionary<object, object> tensorMap, Scale scale)
        {
            _tensorMap = tensorMap; _scale = scale;
        }

        internal static bool IsTensor(Type type)
        {
            for (var t = type; t is not null; t = t.BaseType)
                if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(TensorBase<>)) return true;
            return false;
        }

        /// <summary>
        /// Closure classes, value types, tuples, and plain data records (a sealed class straight off object holding
        /// only primitives, such as an op's shape descriptor): copied field by field. Everything else is shared.
        /// </summary>
        /// <remarks>A data record that differs between the traces was made per trace, so a copy per built plan is
        /// what another trace would have made.</remarks>
        internal static bool IsDescendable(Type type)
        {
            if (type.IsValueType) return true;
            if (type.IsDefined(typeof(CompilerGeneratedAttribute), false) || type.Name.Contains("<>")) return true;
            if (type.Namespace == "System" && type.Name.StartsWith("Tuple`", StringComparison.Ordinal)) return true;
            return IsPlainDataRecord(type);
        }

        private static readonly Dictionary<Type, bool> s_plainData = new();

        private static bool IsPlainDataRecord(Type type)
        {
            lock (s_plainData)
            {
                if (s_plainData.TryGetValue(type, out var known)) return known;
                bool plain = type.IsClass && type.IsSealed && type.BaseType == typeof(object) && !typeof(Delegate).IsAssignableFrom(type);
                if (plain)
                {
                    var fields = type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                    plain = fields.Length > 0;
                    foreach (var f in fields)
                        if (!(f.FieldType.IsPrimitive || f.FieldType.IsEnum || f.FieldType == typeof(string))) plain = false;
                }
                s_plainData[type] = plain;
                return plain;
            }
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
                var et = type.GetElementType()!;
                if (_scale.IsScratch(o))
                {
                    var fresh = Array.CreateInstance(et, checked((int)_scale.Value(_scale.ScratchLengthMask(o), (long)array.Length)));
                    _clones[o] = fresh;
                    return fresh;
                }
                if (_scale.HasScaledElements(o))
                {
                    var scaled = (Array)array.Clone();
                    for (int i = 0; i < scaled.Length; i++)
                    {
                        int mask = _scale.ElementMask(o, i);
                        if (mask == 0) continue;
                        long v = _scale.Value(mask, Convert.ToInt64(array.GetValue(i)));
                        scaled.SetValue(et == typeof(int) ? (object)checked((int)v) : v, i);
                    }
                    _clones[o] = scaled;
                    return scaled;
                }
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
                var ft = field.FieldType;
                if (ft == typeof(int) || ft == typeof(long))
                {
                    int mask = _scale.FieldMask(o, field);
                    if (mask == 0) continue;
                    long v = _scale.Value(mask, Convert.ToInt64(field.GetValue(o)));
                    field.SetValue(clone, ft == typeof(int) ? (object)checked((int)v) : v);
                    continue;
                }
                var value = field.GetValue(o);
                var cloned = Clone(value);
                if (!ReferenceEquals(cloned, value) || ft.IsValueType && !ft.IsPrimitive && !ft.IsEnum)
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
