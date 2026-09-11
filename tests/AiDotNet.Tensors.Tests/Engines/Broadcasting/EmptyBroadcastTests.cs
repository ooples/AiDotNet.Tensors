using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Broadcasting;

/// <summary>
/// Broadcasting against a zero-size axis yields a zero-size result, exactly as NumPy, PyTorch and
/// JAX define it: an extent of 1 stretches to whatever the other operand has, and that includes 0.
/// </summary>
/// <remarks>
/// <para>
/// The defect: the broadcast-shape helpers computed each output extent as
/// <c>Math.Max(dim1, dim2)</c>. That equals the broadcast rule for every pair of POSITIVE extents,
/// but for the pair (0, 1) it returns 1 where the rule gives 0. So
/// <c>[0,2,2,2] + [1,2,1,1]</c> was sized as <c>[1,2,2,2]</c> — eight output elements drawn from a
/// left operand that has none — and the SIMD walk's bounds guard threw
/// <c>"Inner left-operand block exceeds its span."</c>. Rank 2 looked fine only because the
/// <c>[N,M] + [1,M]</c> bias fast path never asks for a broadcast shape.
/// </para>
/// <para>
/// The same (0, 1) blind spot sat in the backward reduction: an axis the operand had stretched from
/// 1 to 0 was not summed ("grad dim &gt; 1" is false for 0), so the gradient was reshaped from a
/// zero-length tensor to the operand's non-empty shape and threw. The gradient of an operand that
/// only ever met an empty partner is zeros of the operand's shape — a sum over nothing.
/// </para>
/// </remarks>
public class EmptyBroadcastTests
{
    private readonly CpuEngine _engine = new();

    /// <summary>(left shape, right shape, expected broadcast shape).</summary>
    private static readonly (int[] A, int[] B, int[] Expected)[] s_cases =
    {
        // rank 2 — the bias fast path already handled the first; the others reach the general walk
        (new[] { 0, 2 }, new[] { 1, 2 }, new[] { 0, 2 }),
        (new[] { 0, 2 }, new[] { 1, 1 }, new[] { 0, 2 }),
        (new[] { 2, 0 }, new[] { 2, 1 }, new[] { 2, 0 }),
        (new[] { 1, 2 }, new[] { 0, 2 }, new[] { 0, 2 }),
        // rank 3 — zero in the leading dim, a middle dim, and on the broadcast-from side
        (new[] { 0, 2, 3 }, new[] { 1, 2, 1 }, new[] { 0, 2, 3 }),
        (new[] { 2, 0, 3 }, new[] { 2, 1, 1 }, new[] { 2, 0, 3 }),
        (new[] { 1, 2, 1 }, new[] { 0, 2, 3 }, new[] { 0, 2, 3 }),
        (new[] { 2, 1, 3 }, new[] { 1, 0, 1 }, new[] { 2, 0, 3 }),
        (new[] { 0, 2, 1 }, new[] { 0, 1, 3 }, new[] { 0, 2, 3 }),
        // rank 4 — the reported repro and its mirror, plus a middle zero
        (new[] { 0, 2, 2, 2 }, new[] { 1, 2, 1, 1 }, new[] { 0, 2, 2, 2 }),
        (new[] { 1, 2, 1, 1 }, new[] { 0, 2, 2, 2 }, new[] { 0, 2, 2, 2 }),
        (new[] { 2, 2, 0, 2 }, new[] { 2, 1, 1, 1 }, new[] { 2, 2, 0, 2 }),
        (new[] { 2, 2, 0, 2 }, new[] { 1, 2, 1, 1 }, new[] { 2, 2, 0, 2 }),
        // rank 5
        (new[] { 0, 2, 2, 2, 2 }, new[] { 1, 2, 1, 1, 1 }, new[] { 0, 2, 2, 2, 2 }),
        (new[] { 2, 2, 2, 0, 2 }, new[] { 1, 2, 1, 1, 1 }, new[] { 2, 2, 2, 0, 2 }),
        (new[] { 1, 2, 1, 1, 1 }, new[] { 0, 2, 2, 2, 2 }, new[] { 0, 2, 2, 2, 2 }),
        // rank padding: the padded-in leading 1 must stretch to 0 as well
        (new[] { 0, 2, 3 }, new[] { 2, 1 }, new[] { 0, 2, 3 }),
        (new[] { 2, 1 }, new[] { 0, 2, 3 }, new[] { 0, 2, 3 }),
    };

    private static readonly string[] s_operations = { "add", "subtract", "multiply", "divide" };

    public static TheoryData<string, int[], int[], int[]> Cases
    {
        get
        {
            var data = new TheoryData<string, int[], int[], int[]>();
            foreach (string op in s_operations)
            foreach (var (a, b, expected) in s_cases)
                data.Add(op, a, b, expected);
            return data;
        }
    }

    public static TheoryData<int[], int[], int[]> Shapes
    {
        get
        {
            var data = new TheoryData<int[], int[], int[]>();
            foreach (var (a, b, expected) in s_cases)
                data.Add(a, b, expected);
            return data;
        }
    }

    internal static Tensor<T> Filled<T>(int[] shape, Func<int, T> value)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = value(i);
        return t;
    }

    internal static Tensor<T> Apply<T>(IEngine engine, string op, Tensor<T> a, Tensor<T> b) => op switch
    {
        "add" => engine.TensorAdd(a, b),
        "subtract" => engine.TensorSubtract(a, b),
        "multiply" => engine.TensorMultiply(a, b),
        "divide" => engine.TensorDivide(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op), op, "Unknown element-wise operation."),
    };

    internal static void AssertEmptyWithShape<T>(Tensor<T> result, int[] expected, string context)
    {
        Assert.True(expected.SequenceEqual(result.Shape.ToArray()),
            $"{context} produced [{string.Join(",", result.Shape.ToArray())}], " +
            $"expected [{string.Join(",", expected)}]");
        Assert.Equal(0, result.Length);
    }

    internal static string Describe(string op, int[] a, int[] b)
        => $"{op}([{string.Join(",", a)}], [{string.Join(",", b)}])";

    [Theory]
    [MemberData(nameof(Cases))]
    public void EngineOp_Double_ZeroSizeBroadcast_ReturnsEmptyResultOfBroadcastShape(
        string op, int[] shapeA, int[] shapeB, int[] expected)
    {
        var a = Filled<double>(shapeA, i => 1.0 + i);
        var b = Filled<double>(shapeB, i => 2.0 + i);

        AssertEmptyWithShape(Apply(_engine, op, a, b), expected, Describe(op, shapeA, shapeB));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void EngineOp_Float_ZeroSizeBroadcast_ReturnsEmptyResultOfBroadcastShape(
        string op, int[] shapeA, int[] shapeB, int[] expected)
    {
        var a = Filled<float>(shapeA, i => 1f + i);
        var b = Filled<float>(shapeB, i => 2f + i);

        AssertEmptyWithShape(Apply(_engine, op, a, b), expected, Describe(op, shapeA, shapeB));
    }

    [Fact]
    public void ReportedRepro_TensorAddRank4_DoesNotThrow()
    {
        var result = _engine.TensorAdd(new Tensor<double>(new[] { 0, 2, 2, 2 }), new Tensor<double>(new[] { 1, 2, 1, 1 }));

        AssertEmptyWithShape(result, new[] { 0, 2, 2, 2 }, "TensorAdd([0,2,2,2], [1,2,1,1])");
    }

    /// <summary>The public <c>Tensor.Broadcast*</c> methods share the same shape helper.</summary>
    [Theory]
    [MemberData(nameof(Shapes))]
    public void TensorBroadcastMethods_ZeroSizeBroadcast_ReturnEmptyResultOfBroadcastShape(
        int[] shapeA, int[] shapeB, int[] expected)
    {
        var a = Filled<double>(shapeA, i => 1.0 + i);
        var b = Filled<double>(shapeB, i => 2.0 + i);

        AssertEmptyWithShape(a.BroadcastAdd(b), expected, Describe("BroadcastAdd", shapeA, shapeB));
        AssertEmptyWithShape(a.BroadcastSubtract(b), expected, Describe("BroadcastSubtract", shapeA, shapeB));
        AssertEmptyWithShape(a.BroadcastMultiply(b), expected, Describe("BroadcastMultiply", shapeA, shapeB));
        AssertEmptyWithShape(a.BroadcastDivide(b), expected, Describe("BroadcastDivide", shapeA, shapeB));
    }

    /// <summary>
    /// A zero extent only stretches FROM 1. Pairing it with any other extent is still a mismatch,
    /// exactly as it is in NumPy — the fix must not turn 0 into a wildcard.
    /// </summary>
    [Theory]
    [InlineData(new[] { 0, 2 }, new[] { 2, 2 })]
    [InlineData(new[] { 2, 0, 3 }, new[] { 2, 2, 3 })]
    [InlineData(new[] { 0, 2, 2, 2 }, new[] { 3, 2, 1, 1 })]
    public void ZeroAgainstNonUnitExtent_StillThrows(int[] shapeA, int[] shapeB)
    {
        var a = new Tensor<double>(shapeA);
        var b = new Tensor<double>(shapeB);

        foreach (string op in s_operations)
        {
            Assert.ThrowsAny<ArgumentException>(() => Apply(_engine, op, a, b));
            Assert.ThrowsAny<ArgumentException>(() => Apply(_engine, op, b, a));
        }
    }

    /// <summary>Non-empty broadcasts keep producing the values they always did.</summary>
    [Fact]
    public void NonEmptyBroadcast_ValuesUnchanged()
    {
        var a = Filled<double>(new[] { 2, 3, 2, 2 }, i => 1.0 + i);
        var b = Filled<double>(new[] { 1, 3, 1, 2 }, i => 10.0 * (i + 1));

        var sum = _engine.TensorAdd(a, b);

        Assert.Equal(new[] { 2, 3, 2, 2 }, sum.Shape.ToArray());
        for (int n = 0; n < 2; n++)
        for (int c = 0; c < 3; c++)
        for (int h = 0; h < 2; h++)
        for (int w = 0; w < 2; w++)
            Assert.Equal(a[n, c, h, w] + b[0, c, 0, w], sum[n, c, h, w], 12);
    }

    [Theory]
    [InlineData(new[] { 0, 2, 3 }, new[] { 1, 2, 1 })]
    [InlineData(new[] { 0, 2, 2, 2 }, new[] { 1, 2, 1, 1 })]
    [InlineData(new[] { 2, 0, 2, 2 }, new[] { 1, 1, 2, 1 })]
    [InlineData(new[] { 2, 0 }, new[] { 0 })]
    [InlineData(new[] { 3, 0 }, new[] { 1 })]
    public void BroadcastAddInPlace_EmptyTarget_IsANoOp(int[] shapeA, int[] shapeB)
    {
        var a = new Tensor<double>(shapeA);
        var b = Filled<double>(shapeB, i => 1.0 + i);

        _engine.TensorBroadcastAddInPlace(a, b);

        Assert.Equal(shapeA, a.Shape.ToArray());
        Assert.Equal(0, a.Length);
    }

    /// <summary>
    /// An in-place op cannot change its target's shape. Stretching a size-1 target axis to 0 is a
    /// shape change, so it must be rejected rather than silently copying zero elements into a
    /// non-empty target and returning it unmodified.
    /// </summary>
    [Fact]
    public void BroadcastAddInPlace_TargetWouldHaveToShrinkToEmpty_Throws()
    {
        var a = Filled<double>(new[] { 1, 2, 1 }, i => 1.0 + i);
        var b = new Tensor<double>(new[] { 0, 2, 3 });

        // Pin the dedicated rejection, not just "some ArgumentException": before the shape fix this
        // threw an unrelated out-of-range error from the SIMD walk, and after it — without the guard —
        // it threw nothing at all.
        var ex = Assert.ThrowsAny<ArgumentException>(() => _engine.TensorBroadcastAddInPlace(a, b));
        Assert.Contains("cannot resize its target", ex.Message, StringComparison.Ordinal);
        Assert.Equal(new[] { 1.0, 2.0 }, a.ToArray());
    }

    /// <summary>
    /// The resize guard compares lengths, so the pre-existing tolerance for a broadcast result that
    /// differs from the target only by padded leading 1s is unchanged.
    /// </summary>
    [Fact]
    public void BroadcastAddInPlace_LeadingOnePaddedOperand_StillAccumulates()
    {
        var a = Filled<double>(new[] { 3 }, i => 1.0 + i);
        var b = Filled<double>(new[] { 1, 3 }, i => 10.0 * (i + 1));

        _engine.TensorBroadcastAddInPlace(a, b);

        Assert.Equal(new[] { 3 }, a.Shape.ToArray());
        Assert.Equal(new[] { 11.0, 22.0, 33.0 }, a.ToArray());
    }

    /// <summary>
    /// The gradient of each operand must come back in the operand's own shape. For the operand that
    /// was stretched against an empty axis that is zeros — a sum over no positions.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void Backward_ZeroSizeBroadcast_GradientsHaveOperandShapesAndAreZero(
        string op, int[] shapeA, int[] shapeB, int[] expected)
    {
        _ = expected;
        var a = Filled<double>(shapeA, i => 1.0 + i);
        var b = Filled<double>(shapeB, i => 2.0 + i);

        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(Apply(_engine, op, a, b), null, keepDims: false);
        var grads = tape.ComputeGradients(loss, new[] { a, b });

        foreach (var (operand, name) in new[] { (a, "a"), (b, "b") })
        {
            Assert.True(grads.ContainsKey(operand), $"no gradient for {name} of {Describe(op, shapeA, shapeB)}");
            var g = grads[operand];
            Assert.True(operand.Shape.ToArray().SequenceEqual(g.Shape.ToArray()),
                $"gradient of {name} for {Describe(op, shapeA, shapeB)} has shape " +
                $"[{string.Join(",", g.Shape.ToArray())}], expected [{string.Join(",", operand.Shape.ToArray())}]");
            Assert.All(g.ToArray(), v => Assert.Equal(0.0, v));
        }
    }

    /// <summary>The lazy-graph recorder sizes its placeholder with the engine's own shape helper.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void GraphMode_ZeroSizeBroadcast_RecordsTheBroadcastShape(
        string op, int[] shapeA, int[] shapeB, int[] expected)
    {
        var a = Filled<float>(shapeA, i => 1f + i);
        var b = Filled<float>(shapeB, i => 2f + i);

        using var scope = GraphMode.Enable();
        // Replay on THIS engine, not the process-global one, so the result does not depend on
        // whether the host happens to auto-detect a GPU. The GPU overrides are pinned separately.
        scope.BindEngineIfUnset(_engine);
        var placeholder = Apply(_engine, op, a, b);

        Assert.True(expected.SequenceEqual(placeholder.Shape.ToArray()),
            $"{Describe(op, shapeA, shapeB)} recorded [{string.Join(",", placeholder.Shape.ToArray())}], " +
            $"expected [{string.Join(",", expected)}]");
    }
}

/// <summary>GPU-engine counterpart of <see cref="EmptyBroadcastTests"/>.</summary>
[Collection("DirectGpuSerial")]
public class EmptyBroadcastGpuTests
{
    /// <summary>
    /// <see cref="DirectGpuTensorEngine"/> overrides the four broadcast operators with last-axis fast
    /// paths guarded by <c>a.Length % b.Length == 0</c>, which divides by zero — outside any
    /// try/catch — as soon as the right operand is empty. Skips without a GPU.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(EmptyBroadcastTests.Cases), MemberType = typeof(EmptyBroadcastTests))]
    public void GpuEngine_Float_ZeroSizeBroadcast_ReturnsEmptyResultOfBroadcastShape(
        string op, int[] shapeA, int[] shapeB, int[] expected)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No GPU available");
            var a = EmptyBroadcastTests.Filled<float>(shapeA, i => 1f + i);
            var b = EmptyBroadcastTests.Filled<float>(shapeB, i => 2f + i);

            EmptyBroadcastTests.AssertEmptyWithShape(EmptyBroadcastTests.Apply(gpu, op, a, b), expected, "GPU " + EmptyBroadcastTests.Describe(op, shapeA, shapeB));
        }
    }
}
