using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// What a closure objective can capture, and how each kind of captured value enters the signature: compared by value
/// (so a change replays a different plan), or refused with a reason naming the op and the field.
/// </summary>
public class ObjectiveSignatureCaptureTests
{
    private enum Mode { First, Second }

    private struct Coefficients
    {
        public float Alpha;
        public int Step;
    }

    private sealed class Holder
    {
        public int Value;
        public void Write(IEngine engine, Tensor<float> output) => output.SetFlat(0, Value);
    }

    private static Func<int> Wrap(Func<int> inner) => () => inner() + 1;

    private static void StaticWrite(IEngine engine, Tensor<float> output) => output.SetFlat(0, 1f);

    private static Tensor<float> Filled(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < rows * cols; i++) t.SetFlat(i, (float)(rng.NextDouble() - 0.5));
        return t;
    }

    /// <summary>Signs one recorded unary node over <paramref name="input"/> whose execute delegate is
    /// <paramref name="execute"/>.</summary>
    private static ObjectiveSignature SignUnary(
        Tensor<float> input, Action<IEngine, Tensor<float>> execute, Tensor<float>[]? parameters = null,
        object[]? savedState = null)
        => SignUnary(input, execute, parameters, savedState, out _);

    private static ObjectiveSignature SignUnary(
        Tensor<float> input, Action<IEngine, Tensor<float>> execute, Tensor<float>[]? parameters,
        object[]? savedState, out object[] leaves)
    {
        using var scope = GraphMode.EnableProbe();
        scope.RecordUnary(LazyNodeType.Custom, "CaptureOp", input, input.Shape.ToArray(), execute, savedState: savedState);
        return ObjectiveSignature.Build(scope.Nodes, parameters ?? Array.Empty<Tensor<float>>(), out leaves);
    }

    private static Action<IEngine, Tensor<float>> Capturing(string label, Mode mode, Type type, decimal amount,
        Coefficients coefficients, int[] steps, object[] mixed, Func<int> nested)
        => (engine, output) =>
        {
            GC.KeepAlive(label);
            GC.KeepAlive(type);
            output.SetFlat(0, (float)mode + (float)amount + coefficients.Alpha + coefficients.Step + steps[0]
                              + mixed.Length + nested());
        };

    [Fact]
    public void Captured_values_of_every_comparable_kind_sign_and_compare_by_value()
    {
        var x = Filled(2, 3, 1);
        int seven = 7;
        Func<int> nested = () => seven;
        var c = new Coefficients { Alpha = 0.5f, Step = 3 };

        var a = SignUnary(x, Capturing("noise", Mode.First, typeof(float), 1.5m, c, new[] { 1, 2 },
            new object[] { 4, "s", null! }, nested));
        var same = SignUnary(x, Capturing("noise", Mode.First, typeof(float), 1.5m, c, new[] { 1, 2 },
            new object[] { 4, "s", null! }, nested));

        Assert.True(a.Signable, a.UnsignableReason);
        Assert.Equal(a, same);
        Assert.True(a.Equals((object)same));
        Assert.Null(a.DescribeDifference(same));

        // Each captured value is part of the signature: changing any one changes it.
        var variants = new[]
        {
            SignUnary(x, Capturing("other", Mode.First, typeof(float), 1.5m, c, new[] { 1, 2 }, new object[] { 4, "s", null! }, nested)),
            SignUnary(x, Capturing("noise", Mode.Second, typeof(float), 1.5m, c, new[] { 1, 2 }, new object[] { 4, "s", null! }, nested)),
            SignUnary(x, Capturing("noise", Mode.First, typeof(double), 1.5m, c, new[] { 1, 2 }, new object[] { 4, "s", null! }, nested)),
            SignUnary(x, Capturing("noise", Mode.First, typeof(float), 2.5m, c, new[] { 1, 2 }, new object[] { 4, "s", null! }, nested)),
            SignUnary(x, Capturing("noise", Mode.First, typeof(float), 1.5m, new Coefficients { Alpha = 0.5f, Step = 4 }, new[] { 1, 2 }, new object[] { 4, "s", null! }, nested)),
            SignUnary(x, Capturing("noise", Mode.First, typeof(float), 1.5m, c, new[] { 1, 3 }, new object[] { 4, "s", null! }, nested)),
            SignUnary(x, Capturing("noise", Mode.First, typeof(float), 1.5m, c, new[] { 1, 2 }, new object[] { 5, "s", null! }, nested)),
        };
        foreach (var v in variants)
        {
            Assert.True(v.Signable, v.UnsignableReason);
            Assert.NotEqual(a, v);
            Assert.Contains("CaptureOp", a.DescribeDifference(v));
        }

        seven = 8;   // the nested delegate's captured local
        var changedNested = SignUnary(x, Capturing("noise", Mode.First, typeof(float), 1.5m, c, new[] { 1, 2 },
            new object[] { 4, "s", null! }, nested));
        Assert.NotEqual(a, changedNested);
    }

    [Fact]
    public void Saved_state_is_compared_by_value()
    {
        var x = Filled(2, 2, 1);
        Action<IEngine, Tensor<float>> execute = (e, o) => o.SetFlat(0, 1f);

        var a = SignUnary(x, execute, savedState: new object[] { 0.1f, new[] { 2, 3 } });
        var b = SignUnary(x, execute, savedState: new object[] { 0.2f, new[] { 2, 3 } });
        var none = SignUnary(x, execute);

        Assert.True(a.Signable, a.UnsignableReason);
        Assert.NotEqual(a, b);
        Assert.NotEqual(a, none);
        Assert.Contains("SavedState", SignUnary(x, execute, savedState: new object[] { new List<int>() }).UnsignableReason);
    }

    [Fact]
    public void Static_method_and_engine_targets_sign()
    {
        var x = Filled(2, 2, 1);
        var engine = new CpuEngine();

        var stat = SignUnary(x, StaticWrite);
        var bound = SignUnary(x, (e, o) => o.SetFlat(0, engine.GetType().Name.Length));
        var other = SignUnary(x, (e, o) => o.SetFlat(0, 2f));

        Assert.True(stat.Signable, stat.UnsignableReason);
        Assert.True(bound.Signable, bound.UnsignableReason);
        Assert.NotEqual(stat, other);   // a different method is a different plan
    }

    [Fact]
    public void Self_referencing_closure_signs_without_recursing()
    {
        var x = Filled(2, 2, 1);
        Action<IEngine, Tensor<float>>? self = null;
        self = (e, o) => { if (o.Length < 0) self!(e, o); o.SetFlat(0, 1f); };

        var s = SignUnary(x, self);

        Assert.True(s.Signable, s.UnsignableReason);
    }

    [Fact]
    public void Recording_scope_captured_by_the_closure_is_not_state()
    {
        var x = Filled(2, 2, 1);
        ObjectiveSignature s;
        using (var scope = GraphMode.EnableProbe())
        {
            scope.RecordUnary(LazyNodeType.Custom, "ScopeOp", x, new[] { 2, 2 },
                (e, o) => o.SetFlat(0, scope.Nodes.Count));
            s = ObjectiveSignature.Build(scope.Nodes, Array.Empty<Tensor<float>>(), out _);
        }

        Assert.True(s.Signable, s.UnsignableReason);
    }

    [Fact]
    public void Parameter_view_is_structure_not_a_leaf()
    {
        var w = Filled(4, 3, 1);
        var parameters = new[] { w };
        var asRow = w.Reshape(1, 12);
        var asColumn = w.Reshape(12, 1);

        Action<IEngine, Tensor<float>> execute = (e, o) => o.SetFlat(0, 1f);
        var row = SignUnary(asRow, execute, parameters, null, out var rowLeaves);
        var rowAgain = SignUnary(w.Reshape(1, 12), execute, parameters, null, out _);
        var column = SignUnary(asColumn, execute, parameters, null, out _);

        Assert.True(row.Signable, row.UnsignableReason);
        Assert.Empty(rowLeaves);
        Assert.Equal(row, rowAgain);
        Assert.NotEqual(row, column);
    }

    [Fact]
    public void Non_contiguous_external_tensor_is_unsignable()
    {
        var x = Filled(3, 4, 1).Transpose();

        var s = SignUnary(x, (e, o) => o.SetFlat(0, 1f));

        Assert.False(s.Signable);
        Assert.Contains("non-contiguous", s.UnsignableReason);
        Assert.False(s.Equals((object?)null));
    }

    [Fact]
    public void Multicast_delegate_is_unsignable()
    {
        var x = Filled(2, 2, 1);
        Action<IEngine, Tensor<float>> both = (e, o) => o.SetFlat(0, 1f);
        both += (e, o) => o.SetFlat(1, 2f);

        var s = SignUnary(x, both);

        Assert.False(s.Signable);
        Assert.Contains("multicast", s.UnsignableReason);
    }

    [Fact]
    public void Delegate_bound_to_an_ordinary_object_is_unsignable()
    {
        var x = Filled(2, 2, 1);
        var holder = new Holder { Value = 3 };

        var s = SignUnary(x, holder.Write);

        Assert.False(s.Signable);
        Assert.Contains(nameof(Holder), s.UnsignableReason);
    }

    [Fact]
    public void Uncomparable_value_inside_a_captured_array_is_unsignable()
    {
        var x = Filled(2, 2, 1);
        var items = new object[] { 1, new List<int>() };

        var s = SignUnary(x, (e, o) => o.SetFlat(0, items.Length));

        Assert.False(s.Signable);
        Assert.Contains("items[1]", s.UnsignableReason);
    }

    [Fact]
    public void Deeply_nested_captures_are_unsignable()
    {
        var x = Filled(2, 2, 1);
        // Each Wrap call is its own closure class, so the captured delegates nest one level per call.
        Func<int> f = () => 1;
        for (int i = 0; i < 8; i++) f = Wrap(f);
        Func<int> f7 = f;

        var s = SignUnary(x, (e, o) => o.SetFlat(0, f7()));

        Assert.False(s.Signable);
        Assert.Contains("deeper", s.UnsignableReason);
    }

    [Fact]
    public void Unsignable_signatures_are_never_equal_even_to_an_identical_trace()
    {
        var x = Filled(2, 2, 1);
        var opaque = new List<int>();
        Action<IEngine, Tensor<float>> execute = (e, o) => o.SetFlat(0, opaque.Count);

        var a = SignUnary(x, execute);
        var b = SignUnary(x, execute);

        Assert.False(a.Signable);
        Assert.NotEqual(a, b);
        Assert.True(a.Equals(a));
    }

    [Fact]
    public void Different_node_counts_describe_a_length_difference()
    {
        var engine = new CpuEngine();
        var x = Filled(2, 2, 1);
        ObjectiveSignature one, two;
        using (var scope = GraphMode.EnableProbe())
        {
            engine.ReduceSum(x);
            one = ObjectiveSignature.Build(scope.Nodes, Array.Empty<Tensor<float>>(), out _);
        }
        using (var scope = GraphMode.EnableProbe())
        {
            engine.ReduceSum(engine.Tanh(x));
            two = ObjectiveSignature.Build(scope.Nodes, Array.Empty<Tensor<float>>(), out _);
        }

        Assert.NotEqual(one, two);
        Assert.NotNull(one.DescribeDifference(two));
    }

    [Fact]
    public void Null_arguments_are_rejected()
    {
        var x = Filled(2, 2, 1);
        var s = SignUnary(x, (e, o) => o.SetFlat(0, 1f));

        Assert.Throws<ArgumentNullException>(() => ObjectiveSignature.Build<float>(null!, Array.Empty<Tensor<float>>(), out _));
        Assert.Throws<ArgumentNullException>(() => ObjectiveSignature.Build<float>(Array.Empty<ILazyNode>(), null!, out _));
        Assert.Throws<ArgumentNullException>(() => s.DescribeDifference(null!));
        Assert.False(s.Equals((ObjectiveSignature?)null));
    }
}
