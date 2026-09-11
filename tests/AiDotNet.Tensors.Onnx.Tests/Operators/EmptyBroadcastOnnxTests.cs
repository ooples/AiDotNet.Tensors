using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Operators;
using AiDotNet.Tensors.Onnx.Protos;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// ONNX translators that compute a broadcast shape themselves must follow the NumPy rule: an
/// extent of 1 stretches to the other extent, including 0.
/// </summary>
/// <remarks>
/// <c>MathOperators.ComputeBroadcastShape</c> (used by Min, Max and Equal) and MatMul's batch-shape
/// loop both computed each axis as <c>Math.Max(a, b)</c>, which is right for positive extents but
/// turns (0, 1) into 1. ComputeBroadcastShape also had no compatibility check, so [2,3] vs [3,3]
/// was silently sized [3,3]. The translators are driven directly here so the tests do not depend
/// on how the importer treats a zero <c>dim_value</c> in a graph-input declaration.
/// </remarks>
public class EmptyBroadcastOnnxTests
{
    private static Tensor<float> Run(IOnnxOpTranslator<float> translator, string opType,
        params (string Name, Tensor<float> Value)[] inputs)
    {
        var tensors = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        var node = new NodeProto { OpType = opType };
        foreach (var (name, value) in inputs)
        {
            tensors[name] = value;
            node.Input.Add(name);
        }
        node.Output.Add("Y");
        var ctx = new OnnxTranslationContext<float>(new CpuEngine(), tensors, new OnnxImportOptions());
        translator.Translate(ctx, node);
        return ctx.GetTensor("Y");
    }

    private static Tensor<float> Filled(int[] shape, float start = 1f)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + i;
        return t;
    }

    public static TheoryData<int[], int[], int[]> EmptyPairs => new()
    {
        { new[] { 0, 3 }, new[] { 1, 3 }, new[] { 0, 3 } },
        { new[] { 1, 3 }, new[] { 0, 3 }, new[] { 0, 3 } },
        { new[] { 2, 0, 3 }, new[] { 2, 1, 1 }, new[] { 2, 0, 3 } },
        { new[] { 0, 2, 3 }, new[] { 2, 1 }, new[] { 0, 2, 3 } },
    };

    [Theory]
    [MemberData(nameof(EmptyPairs))]
    public void MinMax_ZeroSizeBroadcast_ReturnsEmptyBroadcastShape(int[] shapeA, int[] shapeB, int[] expected)
    {
        var min = Run(new MathOperators.Min<float>(), "Min", ("A", Filled(shapeA)), ("B", Filled(shapeB)));
        var max = Run(new MathOperators.Max<float>(), "Max", ("A", Filled(shapeA)), ("B", Filled(shapeB)));

        Assert.Equal(expected, min.Shape.ToArray());
        Assert.Equal(expected, max.Shape.ToArray());
        Assert.Equal(0, min.Length);
        Assert.Equal(0, max.Length);
    }

    [Theory]
    [MemberData(nameof(EmptyPairs))]
    public void Equal_ZeroSizeBroadcast_ReturnsEmptyBroadcastShape(int[] shapeA, int[] shapeB, int[] expected)
    {
        var eq = Run(new MathOperators.Equal<float>(), "Equal", ("A", Filled(shapeA)), ("B", Filled(shapeB)));

        Assert.Equal(expected, eq.Shape.ToArray());
        Assert.Equal(0, eq.Length);
    }

    [Theory]
    [MemberData(nameof(EmptyPairs))]
    public void Expand_ToZeroSizeShape_ReturnsEmptyBroadcastShape(int[] shapeA, int[] shapeB, int[] expected)
    {
        // Expand(input, shape) is the multidirectional broadcast of input.shape and `shape`.
        var shape = new Tensor<float>(new[] { shapeB.Length });
        for (int i = 0; i < shapeB.Length; i++) shape[i] = shapeB[i];

        var y = Run(new MathOperators.Expand<float>(), "Expand", ("X", Filled(shapeA)), ("S", shape));

        Assert.Equal(expected, y.Shape.ToArray());
        Assert.Equal(0, y.Length);
    }

    /// <summary>Incompatible shapes are rejected at import time with an ONNX-level error.</summary>
    [Theory]
    [InlineData(new[] { 2, 3 }, new[] { 3, 3 })]
    [InlineData(new[] { 0, 3 }, new[] { 2, 3 })]   // 0 is not a wildcard: it only stretches from 1
    public void MinMaxEqual_IncompatibleShapes_ThrowOnnxError(int[] shapeA, int[] shapeB)
    {
        foreach (var (translator, op) in new (IOnnxOpTranslator<float>, string)[]
                 {
                     (new MathOperators.Min<float>(), "Min"),
                     (new MathOperators.Max<float>(), "Max"),
                     (new MathOperators.Equal<float>(), "Equal"),
                 })
        {
            var ex = Assert.Throws<InvalidDataException>(
                () => Run(translator, op, ("A", Filled(shapeA)), ("B", Filled(shapeB))));
            Assert.Contains("not broadcast-compatible", ex.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void Expand_IncompatibleTarget_ThrowsOnnxError()
    {
        var input = Filled(new[] { 2, 3 });
        var shape = new Tensor<float>(new[] { 2 });
        shape[0] = 3; shape[1] = 3;

        var ex = Assert.Throws<InvalidDataException>(
            () => Run(new MathOperators.Expand<float>(), "Expand", ("X", input), ("S", shape)));
        Assert.Contains("not broadcast-compatible", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>Non-empty broadcasting keeps producing the NumPy result.</summary>
    [Fact]
    public void Max_NonEmptyBroadcast_ValuesUnchanged()
    {
        var a = Filled(new[] { 2, 3 });          // 1..6
        var b = new Tensor<float>(new[] { 2, 1 });
        b[0] = 2.5f; b[1] = 10f;

        var y = Run(new MathOperators.Max<float>(), "Max", ("A", a), ("B", b));

        Assert.Equal(new[] { 2, 3 }, y.Shape.ToArray());
        Assert.Equal(new[] { 2.5f, 2.5f, 3f, 10f, 10f, 10f }, y.ToArray());
    }

    [Theory]
    [InlineData(new[] { 0, 2, 3 }, new[] { 1, 3, 4 }, new[] { 0, 2, 4 })]
    [InlineData(new[] { 1, 2, 3 }, new[] { 0, 3, 4 }, new[] { 0, 2, 4 })]
    [InlineData(new[] { 2, 1, 2, 3 }, new[] { 1, 0, 3, 4 }, new[] { 2, 0, 2, 4 })]
    [InlineData(new[] { 0, 2, 3 }, new[] { 3, 4 }, new[] { 0, 2, 4 })]
    public void MatMul_ZeroSizeBatchBroadcast_ReturnsEmptyBatch(int[] shapeA, int[] shapeB, int[] expected)
    {
        var y = Run(new ArithOperators.MatMul<float>(), "MatMul", ("A", Filled(shapeA)), ("B", Filled(shapeB)));

        Assert.Equal(expected, y.Shape.ToArray());
        Assert.Equal(0, y.Length);
    }

    [Fact]
    public void MatMul_NonEmptyBatchBroadcast_ValuesUnchanged()
    {
        var a = Filled(new[] { 2, 2, 3 });
        var b = Filled(new[] { 1, 3, 2 }, start: -2f);

        var y = Run(new ArithOperators.MatMul<float>(), "MatMul", ("A", a), ("B", b));

        Assert.Equal(new[] { 2, 2, 2 }, y.Shape.ToArray());
        for (int n = 0; n < 2; n++)
        for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            float expected = 0f;
            for (int k = 0; k < 3; k++) expected += a[n, i, k] * b[0, k, j];
            Assert.Equal(expected, y[n, i, j], 4);
        }
    }
}
