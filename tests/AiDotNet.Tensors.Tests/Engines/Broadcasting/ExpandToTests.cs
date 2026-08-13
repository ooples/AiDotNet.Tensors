using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Broadcasting;

/// <summary>
/// Pins the contract of <see cref="Tensor{T}.ExpandTo"/>, the stride-0 view that broadcasting is
/// built on.
/// </summary>
/// <remarks>
/// <para>
/// The whole value of this type is that it does NOT copy. A <c>[4,1]</c> column viewed as
/// <c>[4,3]</c> must still be four numbers in memory, with the stretched axis carrying a stride of
/// zero. If it ever silently materializes, every broadcast in the library starts allocating the
/// full expanded tensor and nothing fails loudly — it just gets slower and hungrier. So these
/// tests assert the sharing behaviourally, by writing through the source and reading through the
/// view.
/// </para>
/// </remarks>
public class ExpandToTests
{
    private static Tensor<double> Filled(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    [Fact]
    public void ExpandingToTheSameShape_ReturnsTheReceiverItself()
    {
        // The shape-equal case is the hot path of every element-wise op. It must not build a view.
        var t = Filled(new[] { 4, 3 }, 1);
        Assert.Same(t, t.ExpandTo(new[] { 4, 3 }));
        Assert.Same(t, t.ExpandTo(new[] { 4, 3 }.ToArray()));
    }

    [Fact]
    public void SparseTensor_IsRejectedBeforeSameShapeFastPath()
    {
        var sparse = new SparseTensor<double>(
            rows: 2,
            columns: 2,
            rowIndices: [0, 1],
            columnIndices: [0, 1],
            values: [1.0, 2.0]);

        Assert.Throws<InvalidOperationException>(() => sparse.ExpandTo([2, 2]));
    }

    [Fact]
    public void ExpandedView_SharesStorageWithItsSource()
    {
        var col = Filled(new[] { 4, 1 }, 2);
        var view = col.ExpandTo(new[] { 4, 3 });

        // Writing through the source must be visible at every stretched position. A copy would not
        // see this; only a genuine stride-0 alias does.
        col[2, 0] = 12345.0;

        Assert.Equal(12345.0, view[2, 0]);
        Assert.Equal(12345.0, view[2, 1]);
        Assert.Equal(12345.0, view[2, 2]);
    }

    [Fact]
    public void ExpandedView_ReportsItsTargetShapeButIsNotContiguous()
    {
        var view = Filled(new[] { 4, 1 }, 3).ExpandTo(new[] { 4, 3 });

        Assert.Equal(new[] { 4, 3 }, view.Shape.ToArray());
        Assert.Equal(12, view.Length);
        Assert.False(view.IsContiguous,
            "A stretched view must report non-contiguous. If it claims contiguity, AsSpan() hands " +
            "back a span of 4 elements for a tensor of logical length 12 and every kernel that " +
            "trusts it reads adjacent memory as data.");
    }

    [Fact]
    public void ExpandedView_RefusesToHandOutAContiguousSpan()
    {
        var view = Filled(new[] { 4, 1 }, 4).ExpandTo(new[] { 4, 3 });

        // Failing loudly here is the property that makes stride-0 views safe to pass around.
        Assert.Throws<InvalidOperationException>(() => view.AsSpan().ToArray());
    }

    [Fact]
    public void MaterializingAnExpandedView_ProducesTheTiledValues()
    {
        var col = Filled(new[] { 4, 1 }, 5);
        var dense = col.ExpandTo(new[] { 4, 3 }).Contiguous();

        Assert.True(dense.IsContiguous);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 3; c++)
                Assert.Equal(col[r, 0], dense[r, c], 12);
    }

    [Fact]
    public void ExpandingAddsLeadingAxesByRightAlignment()
    {
        var row = Filled(new[] { 3 }, 6);
        var view = row.ExpandTo(new[] { 2, 4, 3 });

        Assert.Equal(new[] { 2, 4, 3 }, view.Shape.ToArray());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 3; k++)
                    Assert.Equal(row[k], view[i, j, k], 12);
    }

    /// <summary>
    /// Expanding must compose with the strides a view already carries.
    /// </summary>
    /// <remarks>
    /// A transposed tensor is already non-contiguous. Expanding it must PRESERVE its permuted
    /// strides on the axes that are not being stretched — rebuilding row-major strides from the
    /// shape instead would read the transposed data in the original order and return numbers that
    /// look entirely reasonable while being transposed back.
    /// </remarks>
    [Fact]
    public void ExpandingATransposedView_KeepsThePermutedStrides()
    {
        var t = Filled(new[] { 2, 3 }, 7);
        var transposed = t.Transpose([1, 0]);       // [3,2]
        var view = transposed.ExpandTo(new[] { 4, 3, 2 });

        for (int b = 0; b < 4; b++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                    Assert.Equal(t[j, i], view[b, i, j], 12);
    }

    [Theory]
    [InlineData(new[] { 4, 3 }, new[] { 3, 4 })]     // transposed target
    [InlineData(new[] { 4, 3 }, new[] { 4, 2 })]
    [InlineData(new[] { 4, 3 }, new[] { 2, 3 })]
    [InlineData(new[] { 2, 4, 3 }, new[] { 4, 3 })]  // lower rank
    public void IncompatibleTargets_Throw(int[] from, int[] to)
    {
        var t = Filled(from, 8);
        Assert.Throws<ArgumentException>(() => t.ExpandTo(to));
    }

    /// <summary>
    /// The gradient of an expand is the SUM over every stretched axis.
    /// </summary>
    /// <remarks>
    /// One stored element is read once per position along a stride-0 axis, so it accumulates that
    /// many contributions. Omitting the sum leaves the operand's gradient too small by exactly the
    /// stretch factor, with every forward value still correct.
    /// </remarks>
    [Fact]
    public void ExpandGradient_SumsOverTheStretchedAxes()
    {
        var col = Filled(new[] { 4, 1 }, 9);
        var engine = new CpuEngine();

        using var tape = new GradientTape<double>();
        var expanded = col.ExpandTo(new[] { 4, 3 });
        var loss = engine.ReduceSum(expanded, null, keepDims: false);
        var grads = tape.ComputeGradients(loss, new[] { col });

        Assert.True(grads.ContainsKey(col), "expand did not record a gradient edge for its source");
        var g = grads[col];

        Assert.Equal(new[] { 4, 1 }, g.Shape.ToArray());
        // d(sum of 3 copies)/d(element) == 3, not 1.
        for (int i = 0; i < g.Length; i++)
            Assert.Equal(3.0, g[i], 12);
    }

    [Fact]
    public void ExpandGradient_MatchesFiniteDifferences()
    {
        var col = Filled(new[] { 4, 1 }, 10);
        var other = Filled(new[] { 4, 3 }, 11);
        var engine = new CpuEngine();

        double Loss()
        {
            var product = engine.TensorMultiply(col.ExpandTo([4, 3]).Contiguous(), other);
            return engine.ReduceSum(product, null, keepDims: false)[0];
        }

        using var tape = new GradientTape<double>();
        var value = engine.ReduceSum(
            engine.TensorMultiply(col.ExpandTo([4, 3]), other), null, keepDims: false);
        var analytic = tape.ComputeGradients(value, new[] { col })[col];

        const double eps = 1e-6;
        for (int k = 0; k < col.Length; k++)
        {
            double original = col[k];
            col[k] = original + eps; double plus = Loss();
            col[k] = original - eps; double minus = Loss();
            col[k] = original;

            double numeric = (plus - minus) / (2 * eps);
            Assert.True(Math.Abs(numeric - analytic[k]) <= 1e-6 * Math.Max(1.0, Math.Abs(numeric)),
                $"expand gradient[{k}] is {analytic[k]:G10} but finite differences give {numeric:G10}");
        }
    }
}
