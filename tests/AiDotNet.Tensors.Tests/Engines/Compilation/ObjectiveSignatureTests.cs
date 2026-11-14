using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The structural signature that decides whether a compiled plan may replay a newly traced closure objective.
/// Equal signatures must mean "same plan up to leaf data"; anything a plan would freeze must change it.
/// </summary>
public class ObjectiveSignatureTests
{
    private static Tensor<float> Filled(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < rows * cols; i++) t.SetFlat(i, (float)(rng.NextDouble() - 0.5));
        return t;
    }

    private static ObjectiveSignature Probe(Func<Tensor<float>> objective, Tensor<float>[] parameters, out object[] leaves)
    {
        using var scope = GraphMode.EnableProbe();
        objective();
        return ObjectiveSignature.Build(scope.Nodes, parameters, out leaves);
    }

    [Fact]
    public void Same_closure_twice_is_equal_and_lists_leaves_in_first_use_order()
    {
        var engine = new CpuEngine();
        var w = Filled(8, 4, 1);
        var parameters = new[] { w };
        Tensor<float>? x1 = null, x2 = null;

        var s1 = Probe(() => { x1 = Filled(5, 8, 2); return engine.ReduceSum(engine.Tanh(engine.TensorMatMul(x1, w))); },
            parameters, out var leaves1);
        var s2 = Probe(() => { x2 = Filled(5, 8, 3); return engine.ReduceSum(engine.Tanh(engine.TensorMatMul(x2, w))); },
            parameters, out var leaves2);

        Assert.True(s1.Signable, s1.UnsignableReason);
        Assert.Equal(s1, s2);
        Assert.Equal(s1.GetHashCode(), s2.GetHashCode());
        Assert.Equal(3, s1.NodeCount);
        Assert.Same(x1, Assert.Single(leaves1));
        Assert.Same(x2, Assert.Single(leaves2));
    }

    [Fact]
    public void Different_leaf_data_same_shapes_is_equal()
    {
        var engine = new CpuEngine();
        var w = Filled(4, 4, 1);
        var a = Filled(3, 4, 10);
        var b = Filled(3, 4, 11);

        var s1 = Probe(() => engine.ReduceSum(engine.TensorMatMul(a, w)), new[] { w }, out _);
        var s2 = Probe(() => engine.ReduceSum(engine.TensorMatMul(b, w)), new[] { w }, out _);

        Assert.Equal(s1, s2);
    }

    [Fact]
    public void Captured_scalar_difference_is_not_equal()
    {
        var engine = new CpuEngine();
        var w = Filled(4, 4, 1);
        var x = Filled(3, 4, 2);

        var half = Probe(() => engine.ReduceSum(engine.TensorMultiplyScalar(engine.TensorMatMul(x, w), 0.5f)), new[] { w }, out _);
        var quarter = Probe(() => engine.ReduceSum(engine.TensorMultiplyScalar(engine.TensorMatMul(x, w), 0.25f)), new[] { w }, out _);
        var halfAgain = Probe(() => engine.ReduceSum(engine.TensorMultiplyScalar(engine.TensorMatMul(x, w), 0.5f)), new[] { w }, out _);

        Assert.True(half.Signable, half.UnsignableReason);
        Assert.NotEqual(half, quarter);
        Assert.Equal(half, halfAgain);
        Assert.NotNull(half.DescribeDifference(quarter));
    }

    [Fact]
    public void Shape_change_is_not_equal()
    {
        var engine = new CpuEngine();
        var w = Filled(8, 4, 1);
        var full = Filled(4, 8, 2);
        var partial = Filled(3, 8, 3);

        var s1 = Probe(() => engine.ReduceSum(engine.TensorMatMul(full, w)), new[] { w }, out _);
        var s2 = Probe(() => engine.ReduceSum(engine.TensorMatMul(partial, w)), new[] { w }, out _);

        Assert.NotEqual(s1, s2);
    }

    [Fact]
    public void Unknown_reference_field_is_unsignable_and_named()
    {
        var x = Filled(2, 2, 1);
        var opaque = new List<int> { 1, 2, 3 };

        ObjectiveSignature signature;
        using (var scope = GraphMode.EnableProbe())
        {
            scope.RecordUnary(LazyNodeType.Custom, "OpaqueStateOp", x, new[] { 2, 2 },
                (eng, output) => output.SetFlat(0, opaque.Count));
            signature = ObjectiveSignature.Build(scope.Nodes, Array.Empty<Tensor<float>>(), out _);
        }

        Assert.False(signature.Signable);
        Assert.Contains("OpaqueStateOp", signature.UnsignableReason);
        Assert.Contains("opaque", signature.UnsignableReason);
    }
}
