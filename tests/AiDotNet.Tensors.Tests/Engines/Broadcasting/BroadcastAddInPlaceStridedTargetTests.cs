using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Broadcasting;

/// <summary>
/// <c>TensorBroadcastAddInPlace</c> must mutate the tensor the caller passed, including when that
/// tensor is a non-contiguous view.
/// </summary>
/// <remarks>
/// The defect: the method rebound a non-contiguous target to <c>a.Contiguous()</c> — a fresh copy —
/// and every kernel after that added into the copy. The call returned normally and the caller's
/// tensor was unchanged. Each shape below reaches a different branch of the method (same shape,
/// channel bias, scalar, last-axis bias, general fallback), and every branch had the defect.
/// </remarks>
public class BroadcastAddInPlaceStridedTargetTests
{
    private readonly CpuEngine _engine = new();

    private static Tensor<double> Filled(int[] shape, double start)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + i;
        return t;
    }

    /// <summary>
    /// (target's contiguous source shape, permutation that makes the view, operand shape).
    /// </summary>
    public static TheoryData<int[], int[], int[]> Cases => new()
    {
        { new[] { 3, 2 }, new[] { 1, 0 }, new[] { 2, 3 } },             // same shape
        { new[] { 3, 2 }, new[] { 1, 0 }, new[] { 1 } },                // scalar
        { new[] { 3, 2 }, new[] { 1, 0 }, new[] { 3 } },                // last-axis bias
        { new[] { 3, 2 }, new[] { 1, 0 }, new[] { 2, 1 } },             // general fallback
        { new[] { 2, 2, 3, 3 }, new[] { 0, 1, 3, 2 }, new[] { 1, 2, 1, 1 } }, // conv-bias pattern
        { new[] { 2, 4, 3 }, new[] { 0, 2, 1 }, new[] { 1, 3, 1 } },    // rank-3 general
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void NonContiguousTarget_IsMutated(int[] sourceShape, int[] permutation, int[] operandShape)
    {
        var source = Filled(sourceShape, 1.0);
        var view = source.Transpose(permutation);
        Assert.False(view.IsContiguous, "test precondition: the target must be a strided view");
        var before = view.ToArray();                  // logical (row-major) order of the view
        var operand = Filled(operandShape, 100.0);

        // Reference: the out-of-place broadcast add of the same logical values.
        var expected = _engine.TensorAdd(new Tensor<double>(before, view.Shape.ToArray()), operand).ToArray();

        _engine.TensorBroadcastAddInPlace(view, operand);

        Assert.Equal(expected, view.ToArray());
    }

    /// <summary>
    /// An in-place op on a view writes through to the storage the view shares with its source —
    /// that aliasing is what distinguishes it from returning a new tensor.
    /// </summary>
    [Fact]
    public void NonContiguousTarget_WritesThroughToItsSource()
    {
        var source = Filled(new[] { 3, 2 }, 1.0);        // [[1,2],[3,4],[5,6]]
        var view = source.Transpose(new[] { 1, 0 });     // [[1,3,5],[2,4,6]]
        var row = new Tensor<double>(new[] { 10.0, 20.0, 30.0 }, new[] { 1, 3 });

        _engine.TensorBroadcastAddInPlace(view, row);

        Assert.Equal(new[] { 11.0, 23.0, 35.0, 12.0, 24.0, 36.0 }, view.ToArray());
        Assert.Equal(new[] { 11.0, 12.0, 23.0, 24.0, 35.0, 36.0 }, source.ToArray());
    }

    /// <summary>
    /// A stride-0 (expanded) target has several elements in one memory slot, so there is no
    /// well-defined in-place result. It must be rejected clearly, not written last-wins.
    /// </summary>
    [Fact]
    public void ExpandedTarget_ThrowsClearly()
    {
        var column = Filled(new[] { 3, 1 }, 1.0);
        var expanded = column.ExpandTo(new[] { 3, 4 });
        Assert.False(expanded.IsContiguous, "test precondition: ExpandTo returns a stride-0 view");
        var operand = Filled(new[] { 1, 4 }, 10.0);

        var ex = Assert.Throws<ArgumentException>(() => _engine.TensorBroadcastAddInPlace(expanded, operand));

        Assert.Contains("stride 0", ex.Message, StringComparison.Ordinal);
        Assert.Equal(new[] { 1.0, 2.0, 3.0 }, column.ToArray());
    }
}
