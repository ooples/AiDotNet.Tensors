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
    public static TheoryData<int[], int[], int[]> Compatible => new()
    {
        // rank-equal, one axis stretched
        { new[] { 4, 3 }, new[] { 4, 1 }, new[] { 4, 3 } },
        { new[] { 4, 1 }, new[] { 4, 3 }, new[] { 4, 3 } },
        { new[] { 4, 3 }, new[] { 1, 3 }, new[] { 4, 3 } },
        // both stretched, different axes
        { new[] { 4, 1 }, new[] { 1, 3 }, new[] { 4, 3 } },
        // rank padding
        { new[] { 4, 3 }, new[] { 3 },    new[] { 4, 3 } },
        { new[] { 3 },    new[] { 4, 3 }, new[] { 4, 3 } },
        // scalars in tensor form
        { new[] { 4, 3 }, new[] { 1, 1 }, new[] { 4, 3 } },
        { new[] { 1 },    new[] { 4, 3 }, new[] { 4, 3 } },
        // rank 3
        { new[] { 2, 4, 3 }, new[] { 4, 1 },    new[] { 2, 4, 3 } },
        { new[] { 2, 1, 3 }, new[] { 1, 4, 1 }, new[] { 2, 4, 3 } },
        // identical shapes must stay untouched
        { new[] { 4, 3 }, new[] { 4, 3 }, new[] { 4, 3 } },
    };

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

    [Fact]
    public void BroadcastValues_MatchAnExplicitlyExpandedComputation()
    {
        // [4,3] + [4,1] must equal the same sum against a manually tiled operand.
        var a = Filled(new[] { 4, 3 }, 5);
        var col = Filled(new[] { 4, 1 }, 6);

        var tiled = new Tensor<double>(new[] { 4, 3 });
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 3; c++)
                tiled[r, c] = col[r, 0];

        var broadcast = _engine.TensorAdd(a, col);
        var explicitly = _engine.TensorAdd(a, tiled);

        for (int i = 0; i < broadcast.Length; i++)
            Assert.Equal(explicitly[i], broadcast[i], 12);
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
    [MemberData(nameof(Compatible))]
    public void BroadcastGradient_MatchesFiniteDifferences(int[] shapeA, int[] shapeB, int[] expected)
    {
        _ = expected;
        AssertGradientAgainstFiniteDifferences(shapeA, shapeB, differentiateSecondOperand: false);
        AssertGradientAgainstFiniteDifferences(shapeA, shapeB, differentiateSecondOperand: true);
    }

    private void AssertGradientAgainstFiniteDifferences(int[] shapeA, int[] shapeB, bool differentiateSecondOperand)
    {
        var a = Filled(shapeA, 7);
        var b = Filled(shapeB, 8);
        var target = differentiateSecondOperand ? b : a;

        double Loss()
        {
            var product = _engine.TensorMultiply(a, b);
            var summed = _engine.ReduceSum(product, null, keepDims: false);
            return summed[0];
        }

        using var tape = new GradientTape<double>();
        var value = _engine.ReduceSum(_engine.TensorMultiply(a, b), null, keepDims: false);
        var grads = tape.ComputeGradients(value, new[] { target });

        Assert.True(grads.ContainsKey(target),
            $"no gradient for the {(differentiateSecondOperand ? "second" : "first")} operand of " +
            $"multiply([{string.Join(",", shapeA)}], [{string.Join(",", shapeB)}])");

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
                $"gradient[{k}] for multiply([{string.Join(",", shapeA)}], [{string.Join(",", shapeB)}]) " +
                $"is {g[k]:G10} but finite differences give {numeric:G10}. A stretched axis is most " +
                "likely being counted once instead of once per broadcast position.");
        }
    }

    [Fact]
    public void StrictScope_RejectsBroadcastsThatWouldOtherwiseSucceed()
    {
        var a = Filled(new[] { 4, 3 }, 9);
        var b = Filled(new[] { 4, 1 }, 10);

        Assert.Equal(new[] { 4, 3 }, _engine.TensorAdd(a, b).Shape.ToArray());

        using (ShapePolicy.Strict())
        {
            var ex = Assert.ThrowsAny<ArgumentException>(() => _engine.TensorAdd(a, b));
            Assert.Contains("Strict", ex.Message, StringComparison.Ordinal);
        }

        // Policy is restored on scope exit.
        Assert.Equal(new[] { 4, 3 }, _engine.TensorAdd(a, b).Shape.ToArray());
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
