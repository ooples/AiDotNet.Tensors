using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Builds the ONNX graph for <see cref="OnnxExporter"/>: names every tensor the steps touch, turns leaf tensors into
/// initializers and translates each step through <see cref="OnnxStepTranslator{T}"/>.
/// </summary>
internal sealed class OnnxGraphWriter<T> where T : unmanaged
{
    private readonly OnnxExportOptions _options;
    private readonly GraphProto _graph;
    private readonly Dictionary<Tensor<T>, string> _names = new(ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly HashSet<string> _usedNames = new(StringComparer.Ordinal);
    private readonly HashSet<TensorStorage<T>> _dynamicStorages = new(ReferenceEqualityComparer<TensorStorage<T>>.Instance);
    private readonly HashSet<Tensor<T>> _initializers = new(ReferenceEqualityComparer<Tensor<T>>.Instance);
    private int _nextId;

    internal OnnxGraphWriter(OnnxExportOptions options)
    {
        _options = options;
        _graph = new GraphProto { Name = options.GraphName ?? "main" };
        Opset = options.EffectiveOpsetVersion;
    }

    internal long Opset { get; }

    internal static int ElementType
    {
        get
        {
            if (typeof(T) == typeof(float)) return (int)TensorProto.Types.DataType.Float;
            if (typeof(T) == typeof(double)) return (int)TensorProto.Types.DataType.Double;
            throw new NotSupportedException($"ONNX export supports float and double plans; got {typeof(T).Name}.");
        }
    }

    internal ModelProto Write(
        CompiledStep<T>[] steps,
        IReadOnlyDictionary<string, Tensor<T>> inputs,
        IReadOnlyDictionary<string, Tensor<T>> outputs)
    {
        int elementType = ElementType;

        foreach (var pair in inputs)
        {
            var name = Reserve(pair.Key);
            _names[pair.Value] = name;
            _dynamicStorages.Add(pair.Value._storage);
            _graph.Input.Add(ValueInfo(name, pair.Value._shape, elementType));
        }
        // Every buffer a step writes is runtime data, never an initializer, whichever tensor object views it.
        foreach (var step in steps) _dynamicStorages.Add(step.OutputBuffer._storage);

        var translator = new OnnxStepTranslator<T>(this);
        var unsupported = new List<string>();
        foreach (var step in steps)
        {
            if (!translator.CanTranslate(step) && !unsupported.Contains(step.OpName))
                unsupported.Add(step.OpName);
        }
        if (unsupported.Count > 0)
            throw new NotSupportedException(
                "The plan contains operations with no ONNX translation: " + string.Join(", ", unsupported) + ".");

        foreach (var step in steps)
        {
            var inputNames = new string[step.Inputs.Length];
            for (int i = 0; i < inputNames.Length; i++) inputNames[i] = NameOf(step.Inputs[i]);
            // A step that writes a tensor an earlier step already wrote (an in-place op) defines a new SSA value.
            string output = NewName("t");
            translator.Translate(step, inputNames, output);
            _names[step.OutputBuffer] = output;
        }

        foreach (var pair in outputs)
        {
            if (!_names.TryGetValue(pair.Value, out var produced))
                throw new ArgumentException($"Output '{pair.Key}' is not a tensor the plan computes.", nameof(outputs));
            var name = Reserve(pair.Key);
            AddNode("Identity", new[] { produced }, new[] { name });
            _graph.Output.Add(ValueInfo(name, pair.Value._shape, elementType));
        }

        var model = new ModelProto
        {
            IrVersion = 8,
            ProducerName = _options.ProducerName ?? "AiDotNet.Tensors.Onnx",
            ProducerVersion = typeof(OnnxExporter).Assembly.GetName().Version?.ToString() ?? string.Empty,
            Graph = _graph,
        };
        model.OpsetImport.Add(new OperatorSetIdProto { Domain = string.Empty, Version = Opset });
        return model;
    }

    // ── Names ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private string Reserve(string name)
    {
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Input and output names must be non-empty.");
        if (!_usedNames.Add(name)) throw new ArgumentException($"The name '{name}' is used twice.");
        return name;
    }

    internal string NewName(string prefix)
    {
        string name;
        do name = prefix + "_" + _nextId++;
        while (!_usedNames.Add(name));
        return name;
    }

    /// <summary>True when <paramref name="tensor"/> is runtime data (an input, a step's output, or a view of either).</summary>
    internal bool IsRuntimeValue(Tensor<T> tensor)
        => _dynamicStorages.Contains(tensor._storage) || (_names.ContainsKey(tensor) && !_initializers.Contains(tensor));

    /// <summary>The ONNX value a step input reads: a named input or step output, else a new initializer.</summary>
    private string NameOf(Tensor<T> tensor)
    {
        if (_names.TryGetValue(tensor, out var name)) return name;
        // A view of an input or a computed buffer that no step produced would freeze runtime data as a constant.
        if (_dynamicStorages.Contains(tensor._storage))
            throw new NotSupportedException(
                "A step reads a view of runtime data that no step of the plan produces; the plan cannot be exported faithfully.");
        name = AddInitializer(tensor);
        _names[tensor] = name;
        _initializers.Add(tensor);
        return name;
    }

    // ── Graph construction ─────────────────────────────────────────────────────────────────────────────────────────

    internal NodeProto AddNode(string opType, IReadOnlyList<string> inputs, IReadOnlyList<string> outputs, params AttributeProto[] attributes)
    {
        var node = new NodeProto { OpType = opType, Name = opType + "_" + _graph.Node.Count };
        foreach (var i in inputs) node.Input.Add(i);
        foreach (var o in outputs) node.Output.Add(o);
        foreach (var a in attributes) node.Attribute.Add(a);
        _graph.Node.Add(node);
        return node;
    }

    /// <summary>A new initializer holding <paramref name="tensor"/>'s current values in logical (row-major) order.</summary>
    internal string AddInitializer(Tensor<T> tensor)
    {
        if (tensor.Layout != TensorLayout.Nchw)
            throw new NotSupportedException($"A constant in the packed {tensor.Layout} layout cannot be exported.");
        var proto = new TensorProto { Name = NewName("const"), DataType = ElementType };
        foreach (var d in tensor._shape) proto.Dims.Add(d);
        T[] values = tensor.IsContiguous ? tensor.AsSpan().ToArray() : tensor.Contiguous().AsSpan().ToArray();
        proto.RawData = ByteString.CopyFrom(MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
        _graph.Initializer.Add(proto);
        return proto.Name;
    }

    /// <summary>An element-type scalar (rank 0) initializer.</summary>
    internal string AddScalar(double value)
    {
        var proto = new TensorProto { Name = NewName("const"), DataType = ElementType };
        byte[] bytes = typeof(T) == typeof(double) ? BitConverter.GetBytes(value) : BitConverter.GetBytes((float)value);
        proto.RawData = ByteString.CopyFrom(bytes);
        _graph.Initializer.Add(proto);
        return proto.Name;
    }

    /// <summary>A 1-D int64 initializer (shapes, axes, slice bounds, pads, indices).</summary>
    internal string AddInt64(IReadOnlyList<long> values, int[]? shape = null)
    {
        var proto = new TensorProto { Name = NewName("const"), DataType = (int)TensorProto.Types.DataType.Int64 };
        if (shape is null) proto.Dims.Add(values.Count);
        else foreach (var d in shape) proto.Dims.Add(d);
        var bytes = new byte[values.Count * sizeof(long)];
        for (int i = 0; i < values.Count; i++) BitConverter.GetBytes(values[i]).CopyTo(bytes, i * sizeof(long));
        proto.RawData = ByteString.CopyFrom(bytes);
        _graph.Initializer.Add(proto);
        return proto.Name;
    }

    internal string AddInt64(IReadOnlyList<int> values) => AddInt64(values.Select(v => (long)v).ToArray());

    internal static ValueInfoProto ValueInfo(string name, int[] shape, int elementType)
    {
        var tensorType = new TypeProto.Types.Tensor { ElemType = elementType, Shape = new TensorShapeProto() };
        foreach (var d in shape) tensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = d });
        return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = tensorType } };
    }

    private sealed class ReferenceEqualityComparer<TItem> : IEqualityComparer<TItem> where TItem : class
    {
        public static readonly ReferenceEqualityComparer<TItem> Instance = new();
        public bool Equals(TItem? x, TItem? y) => ReferenceEquals(x, y);
        public int GetHashCode(TItem obj) => RuntimeHelpers.GetHashCode(obj);
    }
}

/// <summary>ONNX attribute constructors.</summary>
internal static class OnnxAttributes
{
    internal static AttributeProto Int(string name, long value)
        => new() { Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value };

    internal static AttributeProto Float(string name, double value)
        => new() { Name = name, Type = AttributeProto.Types.AttributeType.Float, F = (float)value };

    internal static AttributeProto Str(string name, string value)
        => new() { Name = name, Type = AttributeProto.Types.AttributeType.String, S = ByteString.CopyFromUtf8(value) };

    internal static AttributeProto Ints(string name, IEnumerable<int> values)
    {
        var a = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var v in values) a.Ints.Add(v);
        return a;
    }
}
