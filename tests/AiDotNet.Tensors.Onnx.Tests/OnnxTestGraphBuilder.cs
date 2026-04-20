using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Programmatic builder for tiny ONNX models used to exercise individual
/// operator translators. Writes well-formed ModelProto blobs that both
/// ONNX Runtime and our importer consume; the tests then compare outputs
/// for numerical parity.
/// </summary>
internal static class OnnxTestGraphBuilder
{
    /// <summary>
    /// Builds a single-node ONNX model: <c>Op(inputs...) -&gt; output</c>.
    /// Inputs declared as graph inputs, output as graph output, no
    /// initializers. Caller supplies input tensors at execute time.
    /// </summary>
    internal static ModelProto SingleOp(
        string opType,
        IReadOnlyList<(string name, int[] shape, int elemType)> inputs,
        (string name, int[] shape, int elemType) output,
        IReadOnlyList<AttributeProto>? attributes = null,
        string? domain = null,
        IReadOnlyList<(string name, int[] shape, float[] data)>? initializers = null)
    {
        var graph = new GraphProto { Name = $"{opType}_test" };

        // Graph inputs
        foreach (var (name, shape, elemType) in inputs)
            graph.Input.Add(MakeValueInfo(name, shape, elemType));

        // Output
        graph.Output.Add(MakeValueInfo(output.name, output.shape, output.elemType));

        // Initializers
        if (initializers is not null)
        {
            foreach (var (name, shape, data) in initializers)
                graph.Initializer.Add(MakeInitializer(name, shape, data));
        }

        // Single node
        var node = new NodeProto { OpType = opType };
        if (!string.IsNullOrEmpty(domain)) node.Domain = domain;
        foreach (var (name, _, _) in inputs) node.Input.Add(name);
        if (initializers is not null)
            foreach (var (name, _, _) in initializers)
                node.Input.Add(name);
        node.Output.Add(output.name);
        if (attributes is not null)
            foreach (var a in attributes) node.Attribute.Add(a);
        graph.Node.Add(node);

        // Thread the node's non-default domain through to WrapModel so the
        // emitted ModelProto declares the matching OperatorSetIdProto.
        return WrapModel(graph, extraDomain: string.IsNullOrEmpty(domain) ? null : domain);
    }

    internal static ValueInfoProto MakeValueInfo(string name, int[] shape, int elemType)
    {
        var vi = new ValueInfoProto { Name = name };
        var tensorType = new TypeProto.Types.Tensor { ElemType = elemType };
        var tensorShape = new TensorShapeProto();
        foreach (var d in shape)
            tensorShape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = d });
        tensorType.Shape = tensorShape;
        vi.Type = new TypeProto { TensorType = tensorType };
        return vi;
    }

    internal static TensorProto MakeInitializer(string name, int[] shape, float[] data)
    {
        var t = new TensorProto { Name = name, DataType = (int)TensorProto.Types.DataType.Float };
        foreach (var d in shape) t.Dims.Add(d);
        // Use raw_data for compactness + round-trip parity with real exports.
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        t.RawData = ByteString.CopyFrom(bytes);
        return t;
    }

    internal static TensorProto MakeInt64Initializer(string name, int[] shape, long[] data)
    {
        var t = new TensorProto { Name = name, DataType = (int)TensorProto.Types.DataType.Int64 };
        foreach (var d in shape) t.Dims.Add(d);
        var bytes = new byte[data.Length * sizeof(long)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        t.RawData = ByteString.CopyFrom(bytes);
        return t;
    }

    // Overload: allow MakeInitializer(name, shape, long[]) to route to the
    // int64 path without forcing callers to remember the separate method
    // name. Tests that feed int64 tensor inputs (Reshape shape vectors,
    // Slice starts/ends, etc.) can write MakeInitializer(..., new long[]{…}).
    internal static TensorProto MakeInitializer(string name, int[] shape, long[] data)
        => MakeInt64Initializer(name, shape, data);

    // Raw-data int8 initializer for quantized-op parity tests.
    internal static TensorProto MakeInitializerInt8(string name, int[] shape, sbyte[] data)
    {
        var t = new TensorProto { Name = name, DataType = (int)TensorProto.Types.DataType.Int8 };
        foreach (var d in shape) t.Dims.Add(d);
        var bytes = new byte[data.Length];
        for (int i = 0; i < data.Length; i++) bytes[i] = (byte)data[i];
        t.RawData = ByteString.CopyFrom(bytes);
        return t;
    }

    /// <summary>
    /// Wraps a GraphProto in a minimal ModelProto with IR version and opset
    /// that ONNX Runtime accepts. Opset 13 is the baseline most modern
    /// exporters emit. When the graph uses a non-default domain on any of
    /// its nodes, pass <paramref name="extraDomain"/> so <c>opset_import</c>
    /// contains the matching <see cref="OperatorSetIdProto"/> — ONNX
    /// validation otherwise rejects the model with "No opset import for
    /// domain 'X'".
    /// </summary>
    internal static ModelProto WrapModel(GraphProto graph, int opsetVersion = 13, string? extraDomain = null, int extraDomainVersion = 1)
    {
        var model = new ModelProto
        {
            IrVersion = 7,
            ProducerName = "AiDotNet.Tensors.Onnx.Tests",
            Graph = graph,
            // Default domain "" must always be declared.
            OpsetImport = { new OperatorSetIdProto { Version = opsetVersion } },
        };
        if (!string.IsNullOrEmpty(extraDomain))
            model.OpsetImport.Add(new OperatorSetIdProto { Domain = extraDomain, Version = extraDomainVersion });
        return model;
    }

    internal static byte[] Serialize(ModelProto model)
    {
        var ms = new MemoryStream();
        model.WriteTo(ms);
        return ms.ToArray();
    }
}
