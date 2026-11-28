// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// The serialized form of a fused elementwise chain: one record of <see cref="Stride"/> ints per chain op —
/// <c>[LazyNodeType, source0, source1, scalarBits]</c>. A source at or above zero is an external input port; a
/// negative source <c>-(k + 1)</c> is the output of chain op <c>k</c>. Both the live fused node and a plan reloaded
/// from disk build their codegen graph and their exact sequential replay from this one program, so a saved plan
/// runs the same fused kernel it ran before it was saved.
/// </summary>
internal static class ElementwiseChainProgram
{
    internal const int Stride = 4;
    private const int NoSource = int.MinValue;

    internal static bool TryEncode(
        IReadOnlyList<LazyNode<float>> chain,
        out int[] program,
        out List<Tensor<float>> externalInputs)
    {
        program = new int[chain.Count * Stride];
        externalInputs = new List<Tensor<float>>(4);
        var produced = new Dictionary<Tensor<float>, int>(Autodiff.ReferenceEqualityComparer<Tensor<float>>.Instance);
        var ports = new Dictionary<Tensor<float>, int>(Autodiff.ReferenceEqualityComparer<Tensor<float>>.Instance);

        int Source(Tensor<float> t, List<Tensor<float>> ext)
        {
            if (produced.TryGetValue(t, out int k)) return -(k + 1);
            if (ports.TryGetValue(t, out int port)) return port;
            port = ext.Count;
            ext.Add(t);
            ports[t] = port;
            return port;
        }

        for (int i = 0; i < chain.Count; i++)
        {
            var node = chain[i];
            int scalarBits = 0;
            bool binary = false;
            switch (node.OpType)
            {
                case LazyNodeType.Negate:
                case LazyNodeType.ReLU:
                case LazyNodeType.GELU:
                case LazyNodeType.Sigmoid:
                case LazyNodeType.Tanh:
                case LazyNodeType.Swish:
                    break;
                case LazyNodeType.ELU:
                case LazyNodeType.AddScalar:
                case LazyNodeType.MultiplyScalar:
                case LazyNodeType.DivideScalar:
                    if (!PatternHelpers.TryGetFloatScalar(node, out float scalar)) return false;
                    scalarBits = SingleToBits(scalar);
                    break;
                case LazyNodeType.Add:
                case LazyNodeType.Subtract:
                case LazyNodeType.Multiply:
                case LazyNodeType.Divide:
                    if (node.Input1 is null) return false;
                    binary = true;
                    break;
                default:
                    return false;
            }

            int baseIdx = i * Stride;
            program[baseIdx] = (int)node.OpType;
            program[baseIdx + 1] = Source(node.Input0, externalInputs);
            program[baseIdx + 2] = binary && node.Input1 is not null ? Source(node.Input1, externalInputs) : NoSource;
            program[baseIdx + 3] = scalarBits;
            produced[node.Output] = i;
        }
        return true;
    }

    /// <summary>Validates a program read from a plan file against the inputs it names.</summary>
    internal static void Validate(int[] program, int inputCount)
    {
        if (program.Length == 0 || program.Length % Stride != 0)
            throw new System.IO.InvalidDataException("FusedElementwiseChain program length is invalid. The plan file is corrupt.");
        int ops = program.Length / Stride;
        for (int i = 0; i < ops; i++)
        {
            var type = (LazyNodeType)program[i * Stride];
            if (!IsSupported(type))
                throw new System.IO.InvalidDataException($"FusedElementwiseChain op {type} is not a chain op. The plan file is corrupt.");
            CheckSource(program[i * Stride + 1], i, inputCount);
            int s1 = program[i * Stride + 2];
            if (IsBinary(type)) CheckSource(s1, i, inputCount);
            else if (s1 != NoSource)
                throw new System.IO.InvalidDataException("FusedElementwiseChain unary op names a second source. The plan file is corrupt.");
        }
    }

    private static void CheckSource(int source, int op, int inputCount)
    {
        bool ok = source >= 0 ? source < inputCount : source != NoSource && -(source + 1) < op;
        if (!ok)
            throw new System.IO.InvalidDataException($"FusedElementwiseChain source {source} of op {op} is out of range. The plan file is corrupt.");
    }

    private static bool IsSupported(LazyNodeType type) => type is LazyNodeType.Negate or LazyNodeType.ReLU
        or LazyNodeType.GELU or LazyNodeType.Sigmoid or LazyNodeType.Tanh or LazyNodeType.Swish or LazyNodeType.ELU
        or LazyNodeType.AddScalar or LazyNodeType.MultiplyScalar or LazyNodeType.DivideScalar
        or LazyNodeType.Add or LazyNodeType.Subtract or LazyNodeType.Multiply or LazyNodeType.Divide;

    private static bool IsBinary(LazyNodeType type) => type is LazyNodeType.Add or LazyNodeType.Subtract
        or LazyNodeType.Multiply or LazyNodeType.Divide;

    internal static CodegenGraph BuildGraph(int[] program, IReadOnlyList<Tensor<float>> externalInputs, int[] outShape)
    {
        var graph = new CodegenGraph();
        var loads = new int[externalInputs.Count];
        for (int p = 0; p < loads.Length; p++) loads[p] = -1;
        int ops = program.Length / Stride;
        var results = new int[ops];

        int Resolve(int source)
        {
            if (source < 0) return results[-(source + 1)];
            if (loads[source] < 0)
                loads[source] = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
                    CodegenElementType.Float32, (int[])externalInputs[source]._shape.Clone(), source));
            return loads[source];
        }

        for (int i = 0; i < ops; i++)
        {
            var type = (LazyNodeType)program[i * Stride];
            int in0 = Resolve(program[i * Stride + 1]);
            float scalar = BitsToSingle(program[i * Stride + 3]);
            int created;
            switch (type)
            {
                case LazyNodeType.Negate: created = Unary(graph, CodegenOpKind.Negate, in0, outShape); break;
                case LazyNodeType.ReLU: created = Unary(graph, CodegenOpKind.ReLU, in0, outShape); break;
                case LazyNodeType.GELU: created = Unary(graph, CodegenOpKind.GELU, in0, outShape); break;
                case LazyNodeType.Sigmoid: created = Unary(graph, CodegenOpKind.Sigmoid, in0, outShape); break;
                case LazyNodeType.Tanh: created = Unary(graph, CodegenOpKind.Tanh, in0, outShape); break;
                case LazyNodeType.Swish: created = Unary(graph, CodegenOpKind.Swish, in0, outShape); break;
                case LazyNodeType.ELU:
                    created = graph.AddNode(new CodegenNode(CodegenOpKind.ELU, new[] { in0 }, CodegenElementType.Float32,
                        (int[])outShape.Clone(), scalar));
                    break;
                case LazyNodeType.AddScalar:
                case LazyNodeType.MultiplyScalar:
                case LazyNodeType.DivideScalar:
                {
                    int constant = graph.AddNode(new CodegenNode(CodegenOpKind.Constant, Array.Empty<int>(),
                        CodegenElementType.Float32, (int[])outShape.Clone(), scalar));
                    var kind = type == LazyNodeType.AddScalar ? CodegenOpKind.Add
                        : type == LazyNodeType.MultiplyScalar ? CodegenOpKind.Mul : CodegenOpKind.Div;
                    created = Binary(graph, kind, in0, constant, outShape);
                    break;
                }
                default:
                {
                    int in1 = Resolve(program[i * Stride + 2]);
                    var kind = type switch
                    {
                        LazyNodeType.Add => CodegenOpKind.Add,
                        LazyNodeType.Subtract => CodegenOpKind.Sub,
                        LazyNodeType.Multiply => CodegenOpKind.Mul,
                        _ => CodegenOpKind.Div
                    };
                    created = Binary(graph, kind, in0, in1, outShape);
                    break;
                }
            }
            results[i] = created;
        }
        return graph;
    }

    /// <summary>The chain op by op through the engine: the exact reference a reloaded plan falls back to.</summary>
    internal static Tensor<T> Replay<T>(IEngine engine, int[] program, IReadOnlyList<Tensor<T>> inputs)
    {
        int ops = program.Length / Stride;
        var results = new Tensor<T>[ops];
        Tensor<T> Get(int source) => source < 0 ? results[-(source + 1)] : inputs[source];
        for (int i = 0; i < ops; i++)
        {
            var type = (LazyNodeType)program[i * Stride];
            var a = Get(program[i * Stride + 1]);
            float scalar = BitsToSingle(program[i * Stride + 3]);
            results[i] = type switch
            {
                LazyNodeType.Negate => engine.TensorNegate(a),
                LazyNodeType.ReLU => engine.ReLU(a),
                LazyNodeType.GELU => engine.GELU(a),
                LazyNodeType.Sigmoid => engine.Sigmoid(a),
                LazyNodeType.Tanh => engine.Tanh(a),
                LazyNodeType.Swish => engine.Swish(a),
                LazyNodeType.ELU => engine.ELU(a, scalar),
                LazyNodeType.AddScalar => engine.TensorAddScalar(a, Scalar<T>(scalar)),
                LazyNodeType.MultiplyScalar => engine.TensorMultiplyScalar(a, Scalar<T>(scalar)),
                LazyNodeType.DivideScalar => engine.TensorDivideScalar(a, Scalar<T>(scalar)),
                LazyNodeType.Add => engine.TensorAdd(a, Get(program[i * Stride + 2])),
                LazyNodeType.Subtract => engine.TensorSubtract(a, Get(program[i * Stride + 2])),
                LazyNodeType.Multiply => engine.TensorMultiply(a, Get(program[i * Stride + 2])),
                LazyNodeType.Divide => engine.TensorDivide(a, Get(program[i * Stride + 2])),
                _ => throw new System.IO.InvalidDataException($"FusedElementwiseChain op {type} is not a chain op."),
            };
        }
        return results[ops - 1];
    }

    private static T Scalar<T>(float value) => typeof(T) == typeof(float)
        ? (T)(object)value
        : (T)(object)(double)value;

    private static int Unary(CodegenGraph g, CodegenOpKind kind, int input, int[] shape) =>
        g.AddNode(new CodegenNode(kind, new[] { input }, CodegenElementType.Float32, (int[])shape.Clone()));

    private static int Binary(CodegenGraph g, CodegenOpKind kind, int a, int b, int[] shape) =>
        g.AddNode(new CodegenNode(kind, new[] { a, b }, CodegenElementType.Float32, (int[])shape.Clone()));

#if NETCOREAPP2_0_OR_GREATER || NETSTANDARD2_1_OR_GREATER
    private static int SingleToBits(float value) => BitConverter.SingleToInt32Bits(value);
    private static float BitsToSingle(int bits) => BitConverter.Int32BitsToSingle(bits);
#else
    private static int SingleToBits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
    private static float BitsToSingle(int bits) => BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
#endif
}
