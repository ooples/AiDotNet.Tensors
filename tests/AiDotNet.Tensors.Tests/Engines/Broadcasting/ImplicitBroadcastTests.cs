using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Broadcasting;

/// <summary>
/// Pins the semantics of implicit broadcasting on the element-wise operators.
/// </summary>
/// <remarks>
/// <para>
/// Before this, <c>TensorMultiply</c> broadcast while <c>TensorAdd</c>, <c>TensorSubtract</c> and
/// <c>TensorDivide</c> threw — identical-looking operators with different shape semantics, which is
/// worse than either choice consistently applied.
/// </para>
/// <para>
/// Shape tests alone are not enough here. Broadcasting's backward pass has to SUM the gradient over
/// every stretched axis; get that subtly wrong and the forward values stay perfectly correct while
/// the model trains in the wrong direction. That is the failure these tests exist to catch.
/// </para>
/// </remarks>
public class ImplicitBroadcastTests
{
    private readonly IEngine _engine = new CpuEngine();

    private static Tensor<double> Filled(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    /// <summary>
    /// The reference contract: which shape pairs broadcast, and to what.
    /// </summary>
    private static readonly (int[] ShapeA, int[] ShapeB, int[] Expected)[] s_compatibleCases =
    {
        // rank-equal, one axis stretched
        (new[] { 4, 3 }, new[] { 4, 1 }, new[] { 4, 3 }),
        (new[] { 4, 1 }, new[] { 4, 3 }, new[] { 4, 3 }),
        (new[] { 4, 3 }, new[] { 1, 3 }, new[] { 4, 3 }),
        // both stretched, different axes
        (new[] { 4, 1 }, new[] { 1, 3 }, new[] { 4, 3 }),
        // rank padding
        (new[] { 4, 3 }, new[] { 3 },    new[] { 4, 3 }),
        (new[] { 3 },    new[] { 4, 3 }, new[] { 4, 3 }),
        // scalars in tensor form
        (new[] { 4, 3 }, new[] { 1, 1 }, new[] { 4, 3 }),
        (new[] { 1 },    new[] { 4, 3 }, new[] { 4, 3 }),
        // rank 3
        (new[] { 2, 4, 3 }, new[] { 4, 1 },    new[] { 2, 4, 3 }),
        (new[] { 2, 1, 3 }, new[] { 1, 4, 1 }, new[] { 2, 4, 3 }),
        // identical shapes must stay untouched
        (new[] { 4, 3 }, new[] { 4, 3 }, new[] { 4, 3 })
    };

    public static TheoryData<int[], int[], int[]> Compatible
    {
        get
        {
            var data = new TheoryData<int[], int[], int[]>();
            foreach (var (shapeA, shapeB, expected) in s_compatibleCases)
                data.Add(shapeA, shapeB, expected);
            return data;
        }
    }

    public static TheoryData<string> ElementwiseOperations => new()
    {
        "add",
        "subtract",
        "multiply",
        "divide"
    };

    public static TheoryData<string, int[], int[], int[]> CompatibleOperations
    {
        get
        {
            var data = new TheoryData<string, int[], int[], int[]>();
            foreach (string operation in new[] { "add", "subtract", "multiply", "divide" })
            foreach (var (shapeA, shapeB, expected) in s_compatibleCases)
                data.Add(operation, shapeA, shapeB, expected);
            return data;
        }
    }

    public static TheoryData<int[], int[]> Incompatible => new()
    {
        { new[] { 4, 3 }, new[] { 3, 4 } },   // transposed — the mistake broadcasting must NOT hide
        { new[] { 4, 3 }, new[] { 2, 3 } },
        { new[] { 4, 3 }, new[] { 4, 2 } },
        { new[] { 2, 4, 3 }, new[] { 3, 3 } },
    };

    [Theory]
    [MemberData(nameof(Compatible))]
    public void CompatibleShapes_BroadcastToTheExpectedShape(int[] shapeA, int[] shapeB, int[] expected)
    {
        var a = Filled(shapeA, 1);
        var b = Filled(shapeB, 2);

        foreach (var (name, result) in new (string, Tensor<double>)[]
                 {
                     ("add", _engine.TensorAdd(a, b)),
                     ("subtract", _engine.TensorSubtract(a, b)),
                     ("multiply", _engine.TensorMultiply(a, b)),
                     ("divide", _engine.TensorDivide(a, b)),
                 })
        {
            Assert.True(expected.SequenceEqual(result.Shape.ToArray()),
                $"{name}([{string.Join(",", shapeA)}], [{string.Join(",", shapeB)}]) produced " +
                $"[{string.Join(",", result.Shape.ToArray())}], expected [{string.Join(",", expected)}]");
        }
    }

    [Theory]
    [MemberData(nameof(Incompatible))]
    public void IncompatibleShapes_StillThrow(int[] shapeA, int[] shapeB)
    {
        var a = Filled(shapeA, 3);
        var b = Filled(shapeB, 4);

        // A transposed operand is the classic mistake. Broadcasting must not paper over it.
        Assert.ThrowsAny<ArgumentException>(() => _engine.TensorAdd(a, b));
        Assert.ThrowsAny<ArgumentException>(() => _engine.TensorSubtract(a, b));
        Assert.ThrowsAny<ArgumentException>(() => _engine.TensorMultiply(a, b));
        Assert.ThrowsAny<ArgumentException>(() => _engine.TensorDivide(a, b));
    }

    [Theory]
    [MemberData(nameof(ElementwiseOperations))]
    public void BroadcastValues_MatchAnExplicitlyExpandedComputation(string operation)
    {
        // [4,3] op [4,1] must equal the same operation against a manually tiled operand.
        var a = Filled(new[] { 4, 3 }, 5);
        var col = Filled(new[] { 4, 1 }, 6);

        var tiled = new Tensor<double>(new[] { 4, 3 });
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 3; c++)
                tiled[r, c] = col[r, 0];

        var broadcast = Apply(operation, a, col);
        var explicitly = Apply(operation, a, tiled);

        for (int i = 0; i < broadcast.Length; i++)
            Assert.Equal(explicitly[i], broadcast[i], 12);
    }

    [Fact]
    public void BroadcastElementwise_DensifiesSparseOperandsOnEitherSide()
    {
        var sparse = new SparseTensor<double>(
            rows: 2,
            columns: 1,
            rowIndices: new[] { 0, 1 },
            columnIndices: new[] { 0, 0 },
            values: new[] { 2.0, 5.0 });
        var dense = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < dense.Length; i++) dense[i] = 10.0 + i;

        var sparseLeft = sparse.BroadcastSubtract(dense);
        var sparseRight = dense.BroadcastSubtract(sparse);

        Assert.Equal(new[] { 2, 3 }, sparseLeft.Shape.ToArray());
        Assert.Equal(new[] { 2, 3 }, sparseRight.Shape.ToArray());
        for (int row = 0; row < 2; row++)
        for (int col = 0; col < 3; col++)
        {
            double sparseValue = row == 0 ? 2.0 : 5.0;
            Assert.Equal(sparseValue - dense[row, col], sparseLeft[row, col], 12);
            Assert.Equal(dense[row, col] - sparseValue, sparseRight[row, col], 12);
        }
    }

    [Fact]
    public void ElementwiseInto_ValidatesOperandsAgainstTheResultShape()
    {
        var singletonA = new Tensor<double>(new[] { 1, 1 });
        var singletonB = new Tensor<double>(new[] { 1 });
        var widerResult = new Tensor<double>(new[] { 2, 3 });
        singletonA[0] = 2.0;
        singletonB[0] = 4.0;

        Tensor<double>.ElementwiseInto(
            singletonA, singletonB, widerResult, Tensor<double>.BroadcastOp.Add);

        Assert.All(widerResult.AsSpan().ToArray(), value => Assert.Equal(6.0, value, 12));

        var incompatible = new Tensor<double>(new[] { 2, 2 });
        var exception = Assert.Throws<ArgumentException>(() => Tensor<double>.ElementwiseInto(
            incompatible, singletonB, widerResult, Tensor<double>.BroadcastOp.Add));
        Assert.Equal("a", exception.ParamName);
        Assert.Contains("cannot be broadcast", exception.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The gradient of a broadcast operand must be the SUM over the axes that were stretched.
    /// </summary>
    /// <remarks>
    /// This is the test most libraries do not ship. If the backward pass forgets to reduce, the
    /// gradient comes back with the broadcast shape rather than the operand's, or with each element
    /// counted once instead of once per stretched position — and every forward value still looks
    /// perfectly right while training goes the wrong way.
    /// </remarks>
    [Theory]
    [MemberData(nameof(CompatibleOperations))]
    public void BroadcastGradient_MatchesFiniteDifferences(
        string operation, int[] shapeA, int[] shapeB, int[] expected)
    {
        _ = expected;
        AssertGradientAgainstFiniteDifferences(operation, shapeA, shapeB, differentiateSecondOperand: false);
        AssertGradientAgainstFiniteDifferences(operation, shapeA, shapeB, differentiateSecondOperand: true);
    }

    private Tensor<double> Apply(string operation, Tensor<double> a, Tensor<double> b)
        => operation switch
        {
            "add" => _engine.TensorAdd(a, b),
            "subtract" => _engine.TensorSubtract(a, b),
            "multiply" => _engine.TensorMultiply(a, b),
            "divide" => _engine.TensorDivide(a, b),
            _ => throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown element-wise operation.")
        };

    private void AssertGradientAgainstFiniteDifferences(
        string operation, int[] shapeA, int[] shapeB, bool differentiateSecondOperand)
    {
        var a = Filled(shapeA, 7);
        var b = Filled(shapeB, 8);
        if (operation == "divide")
        {
            // Keep denominator probes away from zero so the finite-difference comparison measures
            // the broadcast gradient reduction rather than numerical conditioning at a pole.
            for (int i = 0; i < b.Length; i++) b[i] = 0.5 + Math.Abs(b[i]);
        }
        var target = differentiateSecondOperand ? b : a;

        double Loss()
        {
            var result = Apply(operation, a, b);
            var summed = _engine.ReduceSum(result, null, keepDims: false);
            return summed[0];
        }

        using var tape = new GradientTape<double>();
        var value = _engine.ReduceSum(Apply(operation, a, b), null, keepDims: false);
        var grads = tape.ComputeGradients(value, new[] { target });

        Assert.True(grads.ContainsKey(target),
            $"no gradient for the {(differentiateSecondOperand ? "second" : "first")} operand of " +
            $"{operation}([{string.Join(",", shapeA)}], [{string.Join(",", shapeB)}])");

        var g = grads[target];
        Assert.True(target.Shape.ToArray().SequenceEqual(g.Shape.ToArray()),
            $"gradient shape [{string.Join(",", g.Shape.ToArray())}] does not match the operand's " +
            $"[{string.Join(",", target.Shape.ToArray())}] — the backward pass did not reduce over " +
            "the broadcast axes.");

        const double eps = 1e-6;
        int probes = Math.Min(4, target.Length);
        for (int k = 0; k < probes; k++)
        {
            double original = target[k];
            target[k] = original + eps; double plus = Loss();
            target[k] = original - eps; double minus = Loss();
            target[k] = original;

            double numeric = (plus - minus) / (2 * eps);
            Assert.True(Math.Abs(numeric - g[k]) <= 1e-5 * Math.Max(1.0, Math.Abs(numeric)),
                $"gradient[{k}] for {operation}([{string.Join(",", shapeA)}], [{string.Join(",", shapeB)}]) " +
                $"is {g[k]:G10} but finite differences give {numeric:G10}. A stretched axis is most " +
                "likely being counted once instead of once per broadcast position.");
        }
    }

    [Theory]
    [MemberData(nameof(ElementwiseOperations))]
    public void StrictScope_RejectsBroadcastsThatWouldOtherwiseSucceed(string operation)
    {
        var a = Filled(new[] { 4, 3 }, 9);
        var b = Filled(new[] { 4, 1 }, 10);

        Assert.Equal(new[] { 4, 3 }, Apply(operation, a, b).Shape.ToArray());

        using (ShapePolicy.Strict())
        {
            var ex = Assert.ThrowsAny<ArgumentException>(() => Apply(operation, a, b));
            Assert.Contains("Strict", ex.Message, StringComparison.Ordinal);
        }

        // Policy is restored on scope exit.
        Assert.Equal(new[] { 4, 3 }, Apply(operation, a, b).Shape.ToArray());
    }

    [Fact]
    public void StrictScope_Nests()
    {
        Assert.False(ShapePolicy.IsStrict);
        using (ShapePolicy.Strict())
        {
            Assert.True(ShapePolicy.IsStrict);
            using (ShapePolicy.Strict()) Assert.True(ShapePolicy.IsStrict);
            // The inner scope must not re-enable broadcasting while the outer one is open.
            Assert.True(ShapePolicy.IsStrict);
        }
        Assert.False(ShapePolicy.IsStrict);
    }

    [Fact]
    public void StrictScope_AliasesCannotDisposeTheSameScopeTwice()
    {
        Assert.False(ShapePolicy.IsStrict);
        using (ShapePolicy.Strict())
        {
            var inner = ShapePolicy.Strict();
            var alias = inner;

            inner.Dispose();
            alias.Dispose();

            // Both variables reference one idempotent handle. Disposing the alias must not consume
            // the still-active outer scope, which was possible when StrictScope was a copyable struct.
            Assert.True(ShapePolicy.IsStrict);
        }
        Assert.False(ShapePolicy.IsStrict);
    }

    [Fact]
    public void StrictScope_HasNoPublicConstructor()
    {
        Assert.Empty(typeof(ShapePolicy.StrictScope).GetConstructors());
    }

    /// <summary>
    /// In-place operators must refuse a broadcast that would need to grow their destination.
    /// </summary>
    /// <remarks>
    /// <c>a += b</c> can broadcast b up to a, but never a up to b — there is nowhere to put the
    /// larger result. Silently truncating or writing past the destination is the kind of corruption
    /// that surfaces far from its cause.
    /// </remarks>
    [Fact]
    public void InPlaceAdd_RefusesToGrowItsDestination()
    {
        var small = Filled(new[] { 4, 1 }, 11);
        var large = Filled(new[] { 4, 3 }, 12);

        Assert.ThrowsAny<ArgumentException>(() => _engine.TensorAddInPlace(small, large));
    }
}
