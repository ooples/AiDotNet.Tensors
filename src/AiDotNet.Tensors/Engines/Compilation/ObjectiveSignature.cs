using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A recorded lazy node viewed type-erased, for <see cref="ObjectiveSignature"/>. Implemented by the two node types a
/// trace records (<see cref="LazyNode{T}"/> and <see cref="CrossTypeLazyNode{TIn,TOut}"/>).
/// </summary>
internal interface ISignableLazyNode : ILazyNode
{
    Type OutputElementType { get; }
    object OutputTensor { get; }
    int SignatureInputCount { get; }
    object? GetSignatureInput(int index);
    Delegate ExecuteDelegate { get; }
    Delegate? BackwardDelegate { get; }
    object[]? SavedStateObjects { get; }
    bool ReplaysHostStateFlag { get; }
}

/// <summary>
/// A tensor viewed type-erased, for <see cref="ObjectiveSignature"/> and <see cref="ClosureObjectiveGuard{T}"/>.
/// Implemented explicitly by <see cref="TensorBase{T}"/>.
/// </summary>
internal interface ISignatureTensor
{
    Type ElementType { get; }
    int[] SignatureShape { get; }
    int[] SignatureStrides { get; }
    int SignatureOffset { get; }
    bool SignatureIsContiguous { get; }
    object SignatureStorageIdentity { get; }

    /// <summary>Copies <paramref name="source"/>'s elements into this tensor and marks it modified (so a captured
    /// graph re-uploads it). Both must have this element type, the same shape and be contiguous.</summary>
    void CopyLeafFrom(ISignatureTensor source);
}

/// <summary>
/// The structural identity of a traced objective: everything a compiled plan freezes when it is built from the
/// trace. Two traces with equal signatures compile to the same plan up to the data held in their external leaves,
/// so a plan compiled from one can replay the other once the leaves' data is copied across.
/// </summary>
/// <remarks>
/// Per node, in record order: op type and name, output shape and element type, the recording engine's type, the
/// Execute and Backward method identities, each input as a reference class (an earlier node, a parameter, a
/// parameter view, or an external leaf by first-use position), every field of the Execute and Backward closures by
/// value, and the saved state by the same rules. A closure field of a type whose value cannot be compared (an
/// arbitrary object) makes the objective unsignable: such an objective must not replay a plan, because the plan
/// would freeze whatever that object held at trace time.
/// </remarks>
internal sealed class ObjectiveSignature : IEquatable<ObjectiveSignature>
{
    private const int MaxFieldDepth = 6;
    private static readonly ConcurrentDictionary<Type, FieldInfo[]> FieldCache = new();

    private readonly object[] _tokens;
    private readonly int _hash;

    private ObjectiveSignature(object[] tokens, int nodeCount, string? unsignableReason)
    {
        _tokens = tokens;
        NodeCount = nodeCount;
        UnsignableReason = unsignableReason;
        unchecked
        {
            int h = 17;
            foreach (var t in tokens) h = h * 31 + TokenHash(t);
            _hash = h;
        }
    }

    /// <summary>Whether every node and closure field could be compared by value.</summary>
    internal bool Signable => UnsignableReason is null;

    /// <summary>When <see cref="Signable"/> is false: the op and field that could not be compared.</summary>
    internal string? UnsignableReason { get; }

    /// <summary>Number of recorded nodes.</summary>
    internal int NodeCount { get; }

    /// <summary>
    /// Builds the signature of <paramref name="nodes"/> (a trace in record order). <paramref name="leaves"/> receives
    /// the external leaves (tensors the trace reads that no node produced, that are not parameters and do not share a
    /// parameter's storage) in first-use order; each is an <see cref="ISignatureTensor"/>.
    /// </summary>
    internal static ObjectiveSignature Build<T>(
        IReadOnlyList<ILazyNode> nodes, Tensor<T>[] parameters, out object[] leaves)
    {
        if (nodes is null) throw new ArgumentNullException(nameof(nodes));
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        var builder = new Builder(parameters);
        string? reason = null;
        for (int i = 0; i < nodes.Count && reason is null; i++)
            reason = builder.AddNode(i, nodes[i]);

        leaves = builder.Leaves.ToArray();
        return new ObjectiveSignature(builder.Tokens.ToArray(), nodes.Count, reason);
    }

    public bool Equals(ObjectiveSignature? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        if (_hash != other._hash || _tokens.Length != other._tokens.Length) return false;
        if (!Signable || !other.Signable) return false;
        for (int i = 0; i < _tokens.Length; i++)
            if (!TokenEquals(_tokens[i], other._tokens[i])) return false;
        return true;
    }

    public override bool Equals(object? obj) => obj is ObjectiveSignature s && Equals(s);

    public override int GetHashCode() => _hash;

    /// <summary>
    /// The first token position at which two signatures differ, described for a diagnostic, or null when equal.
    /// </summary>
    internal string? DescribeDifference(ObjectiveSignature other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        int n = Math.Min(_tokens.Length, other._tokens.Length);
        for (int i = 0; i < n; i++)
            if (!TokenEquals(_tokens[i], other._tokens[i]))
                return $"token {i}: compiled '{Describe(_tokens[i])}', now '{Describe(other._tokens[i])}' (near {NearestNodeName(i)})";
        return _tokens.Length == other._tokens.Length ? null : $"length {_tokens.Length} vs {other._tokens.Length}";
    }

    private string NearestNodeName(int tokenIndex)
    {
        for (int i = tokenIndex; i >= 0; i--)
            if (_tokens[i] is NodeMarker m) return $"node {m.Index} '{m.OpName}'";
        return "start";
    }

    private static string Describe(object t) => t switch
    {
        ArrayToken a => $"array[{a.Values.Length}]",
        NodeMarker m => $"node {m.Index} '{m.OpName}'",
        _ => t.ToString() ?? "null",
    };

    private static int TokenHash(object t) => t is ArrayToken a ? a.Hash : t.GetHashCode();

    private static bool TokenEquals(object a, object b)
    {
        if (a is ArrayToken x) return b is ArrayToken y && x.Equals(y);
        return a.Equals(b);
    }

    // ───────────────────────── tokens ─────────────────────────

    private sealed class NodeMarker : IEquatable<NodeMarker>
    {
        internal NodeMarker(int index, string opName) { Index = index; OpName = opName; }
        internal int Index { get; }
        internal string OpName { get; }
        public bool Equals(NodeMarker? other) => other is not null && other.Index == Index && other.OpName == OpName;
        public override bool Equals(object? obj) => obj is NodeMarker m && Equals(m);
        public override int GetHashCode() => unchecked(Index * 397 ^ OpName.GetHashCode());
    }

    private enum RefClass : byte { Null, Node, Parameter, ParameterView, Leaf }

    private readonly struct TensorRef : IEquatable<TensorRef>
    {
        internal TensorRef(RefClass kind, int index) { Kind = kind; Index = index; }
        internal RefClass Kind { get; }
        internal int Index { get; }
        public bool Equals(TensorRef other) => other.Kind == Kind && other.Index == Index;
        public override bool Equals(object? obj) => obj is TensorRef r && Equals(r);
        public override int GetHashCode() => unchecked((int)Kind * 1000003 ^ Index);
        public override string ToString() => $"{Kind}#{Index}";
    }

    /// <summary>Identifies a method across signatures without relying on MethodInfo instance caching.</summary>
    private readonly struct MethodToken : IEquatable<MethodToken>
    {
        internal MethodToken(MethodInfo method)
        {
            Method = method.MethodHandle.Value;
            DeclaringType = method.DeclaringType?.TypeHandle.Value ?? IntPtr.Zero;
            Name = method.Name;
        }

        private IntPtr Method { get; }
        private IntPtr DeclaringType { get; }
        private string Name { get; }
        public bool Equals(MethodToken other) => other.Method == Method && other.DeclaringType == DeclaringType;
        public override bool Equals(object? obj) => obj is MethodToken m && Equals(m);
        public override int GetHashCode() => Method.GetHashCode() ^ DeclaringType.GetHashCode();
        public override string ToString() => Name;
    }

    private sealed class ArrayToken : IEquatable<ArrayToken>
    {
        internal ArrayToken(Array source)
        {
            Values = (Array)source.Clone();
            Hash = StructuralComparisons.StructuralEqualityComparer.GetHashCode(Values) ^ Values.GetType().GetHashCode();
        }

        internal Array Values { get; }
        internal int Hash { get; }

        public bool Equals(ArrayToken? other)
            => other is not null && other.Hash == Hash && other.Values.GetType() == Values.GetType()
               && StructuralComparisons.StructuralEqualityComparer.Equals(Values, other.Values);

        public override bool Equals(object? obj) => obj is ArrayToken a && Equals(a);
        public override int GetHashCode() => Hash;
    }

    // ───────────────────────── builder ─────────────────────────

    private sealed class Builder
    {
        private readonly Dictionary<object, TensorRef> _classes = new(ReferenceComparer.Instance);
        private readonly Dictionary<object, int> _parameterStorage = new(ReferenceComparer.Instance);
        private readonly HashSet<object> _visiting = new(ReferenceComparer.Instance);

        internal List<object> Tokens { get; } = new();
        internal List<object> Leaves { get; } = new();

        internal Builder(Array parameters)
        {
            for (int p = 0; p < parameters.Length; p++)
            {
                var param = parameters.GetValue(p);
                if (param is null) continue;
                _classes[param] = new TensorRef(RefClass.Parameter, p);
                if (param is ISignatureTensor st && !_parameterStorage.ContainsKey(st.SignatureStorageIdentity))
                    _parameterStorage[st.SignatureStorageIdentity] = p;
            }
        }

        /// <summary>Appends one node's tokens; returns the unsignable reason, or null.</summary>
        internal string? AddNode(int index, ILazyNode node)
        {
            if (node is not ISignableLazyNode n)
                return $"node {index} ('{node.OpName}') is a {node.GetType().Name}, which has no signature";

            Tokens.Add(new NodeMarker(index, n.OpName));
            Tokens.Add(n.OpType);
            Tokens.Add(n.OutputElementType);
            Tokens.Add(new ArrayToken(n.OutputShape));
            Tokens.Add(n.ReplaysHostStateFlag);
            Tokens.Add(n.RecordingEngine?.GetType() ?? typeof(void));

            Tokens.Add(n.SignatureInputCount);
            for (int i = 0; i < n.SignatureInputCount; i++)
            {
                string? r = AddTensorReference(n.GetSignatureInput(i), n.OpName, $"input {i}");
                if (r is not null) return r;
            }

            // From here on the output is this node: an in-place write re-classes its target, and a closure that
            // captures its own output (most do) must see it as this node, never as an external leaf.
            _classes[n.OutputTensor] = new TensorRef(RefClass.Node, index);

            string? reason = AddDelegate(n.ExecuteDelegate, n.OpName, "Execute", 0)
                             ?? AddDelegate(n.BackwardDelegate, n.OpName, "Backward", 0);
            if (reason is not null) return reason;

            var saved = n.SavedStateObjects;
            Tokens.Add(saved?.Length ?? -1);
            if (saved is not null)
                for (int i = 0; i < saved.Length; i++)
                {
                    reason = AddValue(saved[i], n.OpName, $"SavedState[{i}]", 0);
                    if (reason is not null) return reason;
                }

            return null;
        }

        private string? AddTensorReference(object? tensor, string opName, string where)
        {
            if (tensor is null) { Tokens.Add(new TensorRef(RefClass.Null, 0)); return null; }
            if (_classes.TryGetValue(tensor, out var known)) { Tokens.Add(known); return null; }
            if (tensor is not ISignatureTensor st)
                return $"op '{opName}' {where} is a {tensor.GetType().Name}, not a tensor";

            if (_parameterStorage.TryGetValue(st.SignatureStorageIdentity, out int p))
            {
                // A view of a parameter (a reshape or slice): the plan reads the parameter's live storage through it,
                // so it is never rebound; its geometry is part of the structure.
                var view = new TensorRef(RefClass.ParameterView, p);
                _classes[tensor] = view;
                Tokens.Add(view);
                Tokens.Add(new ArrayToken(st.SignatureShape));
                Tokens.Add(new ArrayToken(st.SignatureStrides));
                Tokens.Add(st.SignatureOffset);
                return null;
            }

            if (!st.SignatureIsContiguous)
                return $"op '{opName}' {where} is a non-contiguous external tensor, which cannot be rebound";

            var leaf = new TensorRef(RefClass.Leaf, Leaves.Count);
            Leaves.Add(tensor);
            _classes[tensor] = leaf;
            Tokens.Add(leaf);
            Tokens.Add(st.ElementType);
            Tokens.Add(new ArrayToken(st.SignatureShape));
            return null;
        }

        private string? AddDelegate(Delegate? d, string opName, string where, int depth)
        {
            if (d is null) { Tokens.Add("no-delegate"); return null; }
            if (d.GetInvocationList().Length != 1)
                return $"op '{opName}' {where} is a multicast delegate";
            Tokens.Add(new MethodToken(d.Method));
            return AddTarget(d.Target, opName, where, depth);
        }

        private string? AddTarget(object? target, string opName, string where, int depth)
        {
            if (target is null) { Tokens.Add("static"); return null; }
            if (target is IEngine) { Tokens.Add(target.GetType()); return null; }
            var type = target.GetType();
            if (!IsCompilerGenerated(type))
                return $"op '{opName}' {where} is bound to a {type.FullName}, whose state cannot be compared";
            return AddObjectFields(target, type, opName, where, depth);
        }

        private string? AddObjectFields(object obj, Type type, string opName, string where, int depth)
        {
            if (depth >= MaxFieldDepth)
                return $"op '{opName}' {where} nests captured state deeper than {MaxFieldDepth}";
            if (!type.IsValueType && !_visiting.Add(obj)) { Tokens.Add("cycle"); return null; }
            try
            {
                Tokens.Add(type);
                foreach (var field in Fields(type))
                {
                    string? r = AddValue(field.GetValue(obj), opName, $"{where}.{field.Name}", depth + 1);
                    if (r is not null) return r;
                }
                return null;
            }
            finally
            {
                if (!type.IsValueType) _visiting.Remove(obj);
            }
        }

        private string? AddValue(object? value, string opName, string where, int depth)
        {
            switch (value)
            {
                case null:
                    Tokens.Add("null");
                    return null;
                case string s:
                    Tokens.Add(s);
                    return null;
                case ISignatureTensor:
                    return AddTensorReference(value, opName, where);
                case IEngine:
                    Tokens.Add(value.GetType());
                    return null;
                case LazyTensorScope:
                    // The recording scope itself (a display class shared with the recording branch can hold it). It
                    // is the trace, not state the plan replays.
                    Tokens.Add(typeof(LazyTensorScope));
                    return null;
                case Delegate d:
                    return AddDelegate(d, opName, where, depth);
                case Type t:
                    Tokens.Add(t);
                    return null;
                case Array a:
                    return AddArray(a, opName, where, depth);
            }

            var type = value.GetType();
            if (type.IsPrimitive || type.IsEnum || value is decimal || value is DateTime || value is Guid)
            {
                Tokens.Add(value);
                return null;
            }

            // Stateless numeric operations (one cached instance per element type) carry no per-step data.
            if (IsNumericOperations(type)) { Tokens.Add(type); return null; }

            if (type.IsValueType || IsCompilerGenerated(type))
                return AddObjectFields(value, type, opName, where, depth);

            return $"op '{opName}' field '{where}' holds a {type.FullName}, whose value cannot be compared";
        }

        private string? AddArray(Array a, string opName, string where, int depth)
        {
            var elementType = a.GetType().GetElementType() ?? typeof(object);
            if (a.Rank == 1 && (elementType.IsPrimitive || elementType.IsEnum))
            {
                Tokens.Add(new ArrayToken(a));
                return null;
            }

            Tokens.Add(a.GetType());
            Tokens.Add(a.Length);
            for (int i = 0; i < a.Length; i++)
            {
                string? r = AddValue(a.GetValue(i), opName, $"{where}[{i}]", depth + 1);
                if (r is not null) return r;
            }
            return null;
        }
    }

    private static bool IsCompilerGenerated(Type type)
        => type.IsDefined(typeof(CompilerGeneratedAttribute), inherit: false)
           || type.Name.IndexOf('<') >= 0;

    private static bool IsNumericOperations(Type type)
    {
        foreach (var i in type.GetInterfaces())
            if (i.IsGenericType && i.GetGenericTypeDefinition() == typeof(INumericOperations<>))
                return true;
        return false;
    }

    private static FieldInfo[] Fields(Type type)
        => FieldCache.GetOrAdd(type, t =>
            t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic));

    private sealed class ReferenceComparer : IEqualityComparer<object>
    {
        internal static readonly ReferenceComparer Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
