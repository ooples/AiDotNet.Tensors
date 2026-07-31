using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Broadcasting;

/// <summary>
/// The element-wise kernel now consumes non-contiguous operands where they lie instead of
/// materializing them. These tests prove it reads them correctly.
/// </summary>
/// <remarks>
/// <para>
/// The kernel used to call <c>Contiguous()</c> on any strided operand and then re-derive row-major
/// broadcast strides from the shape. It now reads each operand's stored strides, which is what lets
/// a stride-0 expanded view through without a copy — but it also means a transposed or sliced
/// operand reaches the inner loop unmaterialized for the first time.
/// </para>
/// <para>
/// That is the risk this file exists for. A derived stride describes the layout an operand WOULD
/// have if freshly allocated; for a transposed view that is the wrong layout, and reading it walks
/// the right memory in the wrong order. Every result below is therefore checked against the same
/// computation on an explicitly materialized operand, where the two must agree exactly.
/// </para>
/// </remarks>
public class StridedOperandTests
{
    private readonly IEngine _engine = new CpuEngine();

    private static Tensor<double> Filled(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private void AssertMatchesMaterialized(
        string what, Tensor<double> strided, Tensor<double> other, bool stridedOnLeft = true)
    {
        Assert.False(strided.IsContiguous, $"{what}: operand was expected to be non-contiguous");

        var dense = strided.Contiguous();
        Assert.True(dense.IsContiguous);

        foreach (var (name, viaStrided, viaDense) in new (string, Tensor<double>, Tensor<double>)[]
                 {
                     ("add", stridedOnLeft ? _engine.TensorAdd(strided, other) : _engine.TensorAdd(other, strided),
                             stridedOnLeft ? _engine.TensorAdd(dense, other) : _engine.TensorAdd(other, dense)),
                     ("subtract", stridedOnLeft ? _engine.TensorSubtract(strided, other) : _engine.TensorSubtract(other, strided),
                                  stridedOnLeft ? _engine.TensorSubtract(dense, other) : _engine.TensorSubtract(other, dense)),
                     ("multiply", stridedOnLeft ? _engine.TensorMultiply(strided, other) : _engine.TensorMultiply(other, strided),
                                  stridedOnLeft ? _engine.TensorMultiply(dense, other) : _engine.TensorMultiply(other, dense)),
                 })
        {
            Assert.Equal(viaDense.Shape.ToArray(), viaStrided.Shape.ToArray());
            for (int i = 0; i < viaDense.Length; i++)
            {
                Assert.True(Math.Abs(viaDense[i] - viaStrided[i]) < 1e-12,
                    $"{what} / {name}: element {i} is {viaStrided[i]:G17} when consumed strided but " +
                    $"{viaDense[i]:G17} when materialized first. The kernel is walking the operand's " +
                    "memory in the wrong order.");
            }
        }
    }

    [Fact]
    public void TransposedOperand_AgainstAMatchingShape()
    {
        var t = Filled(new[] { 4, 3 }, 1).Transpose([1, 0]);   // [3,4], strides [1,3]
        AssertMatchesMaterialized("transposed [3,4]", t, Filled(new[] { 3, 4 }, 2));
    }

    [Fact]
    public void TransposedOperand_OnTheRightHandSide()
    {
        var t = Filled(new[] { 4, 3 }, 3).Transpose([1, 0]);
        AssertMatchesMaterialized("transposed rhs", t, Filled(new[] { 3, 4 }, 4), stridedOnLeft: false);
    }

    [Fact]
    public void TransposedOperand_BroadcastAgainstARow()
    {
        // Both a non-unit stride AND a stretched axis in the same operation.
        var t = Filled(new[] { 4, 3 }, 5).Transpose([1, 0]);   // [3,4]
        AssertMatchesMaterialized("transposed + broadcast", t, Filled(new[] { 1, 4 }, 6));
    }

    [Fact]
    public void TransposedRank3Operand()
    {
        var t = Filled(new[] { 2, 4, 3 }, 7).Transpose([0, 2, 1]);   // [2,3,4]
        AssertMatchesMaterialized("transposed rank 3", t, Filled(new[] { 2, 3, 4 }, 8));
    }

    [Fact]
    public void ExpandedViewAsAnOperand()
    {
        var view = Filled(new[] { 4, 1 }, 9).ExpandTo(new[] { 4, 3 });
        AssertMatchesMaterialized("expanded view", view, Filled(new[] { 4, 3 }, 10));
    }

    [Fact]
    public void BothOperandsStrided()
    {
        var a = Filled(new[] { 4, 3 }, 11).Transpose([1, 0]);        // [3,4]
        var b = Filled(new[] { 3, 1 }, 12).ExpandTo(new[] { 3, 4 }); // stride-0 stretch

        Assert.False(a.IsContiguous);
        Assert.False(b.IsContiguous);

        var strided = _engine.TensorAdd(a, b);
        var dense = _engine.TensorAdd(a.Contiguous(), b.Contiguous());

        for (int i = 0; i < dense.Length; i++)
            Assert.True(Math.Abs(dense[i] - strided[i]) < 1e-12,
                $"both-strided add: element {i} is {strided[i]:G17} vs {dense[i]:G17} materialized");
    }

    /// <summary>
    /// A view into the middle of a larger buffer must be read from its own origin.
    /// </summary>
    /// <remarks>
    /// The kernel indexes the full backing span and starts each operand at its <c>_storageOffset</c>.
    /// Ignoring that offset reads from the start of the parent buffer — data that belongs to a
    /// different slice entirely, and which is perfectly valid-looking floating point.
    /// </remarks>
    [Fact]
    public void SlicedOperand_ReadsFromItsOwnOffsetNotTheParentStart()
    {
        var parent = Filled(new[] { 4, 3 }, 13);
        var row2 = parent.Slice(2);                 // row 2 as [3], storage offset 6

        Assert.Equal(new[] { 3 }, row2.Shape.ToArray());

        var other = Filled([3], 14);
        var result = _engine.TensorAdd(row2, other);

        for (int c = 0; c < 3; c++)
            Assert.True(Math.Abs((parent[2, c] + other[c]) - result[c]) < 1e-12,
                $"sliced operand at [{c}]: got {result[c]:G17}, expected " +
                $"{parent[2, c] + other[c]:G17} — the slice's storage offset was dropped, so the " +
                "kernel read row 0 of the parent instead of row 2.");
    }
}
