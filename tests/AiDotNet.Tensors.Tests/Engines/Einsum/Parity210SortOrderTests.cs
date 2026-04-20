using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210SortOrderTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void Sort_Ascending_ReturnsValuesAndIndices()
    {
        var x = T(new[] { 3f, 1f, 4f, 1f, 5f }, 5);
        var (v, idx) = E.TensorSort(x);
        Assert.Equal(new[] { 1f, 1f, 3f, 4f, 5f }, v.AsSpan().ToArray());
        // Input positions of sorted values (first and second 1 both came from input index 1 or 3).
        Assert.Contains(0, idx.AsSpan().ToArray().Skip(2).Take(1)); // 3 came from index 0
        Assert.Contains(2, idx.AsSpan().ToArray().Skip(3).Take(1)); // 4 came from index 2
        Assert.Contains(4, idx.AsSpan().ToArray().Skip(4).Take(1)); // 5 came from index 4
    }

    [Fact]
    public void Sort_Descending_ReversesOrder()
    {
        var x = T(new[] { 3f, 1f, 4f }, 3);
        var (v, _) = E.TensorSort(x, descending: true);
        Assert.Equal(new[] { 4f, 3f, 1f }, v.AsSpan().ToArray());
    }

    [Fact]
    public void Sort_2D_AlongAxis1_SortsRowwise()
    {
        var x = T(new[] { 3f, 1f, 2f, 5f, 4f, 0f }, 2, 3);
        var (v, _) = E.TensorSort(x, axis: 1);
        Assert.Equal(new[] { 1f, 2f, 3f, 0f, 4f, 5f }, v.AsSpan().ToArray());
    }

    [Fact]
    public void Sort_2D_AlongAxis0_SortsColumnwise()
    {
        var x = T(new[] { 3f, 1f, 2f, 5f, 4f, 0f }, 2, 3);
        var (v, _) = E.TensorSort(x, axis: 0);
        // Column 0: 3, 5 -> 3, 5; column 1: 1, 4 -> 1, 4; column 2: 2, 0 -> 0, 2.
        Assert.Equal(new[] { 3f, 1f, 0f, 5f, 4f, 2f }, v.AsSpan().ToArray());
    }

    [Fact]
    public void Kthvalue_Returns1BasedKth()
    {
        var x = T(new[] { 5f, 2f, 8f, 1f, 9f }, 5);
        var (v, _) = E.TensorKthvalue(x, 3); // 3rd smallest = 5
        Assert.Equal(5f, v);
    }

    [Fact]
    public void Median_Odd_ReturnsMiddle()
    {
        var x = T(new[] { 7f, 1f, 3f, 9f, 5f }, 5);
        Assert.Equal(5f, E.TensorMedian(x));
    }

    [Fact]
    public void Median_Even_ReturnsLowerMedian()
    {
        var x = T(new[] { 4f, 2f, 1f, 3f }, 4);
        // sorted = 1, 2, 3, 4 — lower median is 2.
        Assert.Equal(2f, E.TensorMedian(x));
    }

    [Fact]
    public void Unique_SortedDefault_ReturnsAscending()
    {
        var x = T(new[] { 3f, 1f, 2f, 3f, 1f }, 5);
        var r = E.TensorUnique(x);
        Assert.Equal(new[] { 1f, 2f, 3f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Unique_Unsorted_KeepsFirstOccurrenceOrder()
    {
        var x = T(new[] { 3f, 1f, 2f, 3f, 1f }, 5);
        var r = E.TensorUnique(x, sorted: false);
        Assert.Equal(new[] { 3f, 1f, 2f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void SearchSorted_Finds_InsertionIndex()
    {
        var seq = T(new[] { 1f, 3f, 5f, 7f, 9f }, 5);
        var vals = T(new[] { 4f, 6f, 10f, 0f }, 4);
        var idx = E.TensorSearchSorted(seq, vals);
        Assert.Equal(new[] { 2, 3, 5, 0 }, idx.AsSpan().ToArray());
    }

    [Fact]
    public void SearchSorted_Right_PlacesEqualToRight()
    {
        var seq = T(new[] { 1f, 3f, 3f, 5f }, 4);
        var vals = T(new[] { 3f }, 1);
        var left = E.TensorSearchSorted(seq, vals, right: false);
        var right = E.TensorSearchSorted(seq, vals, right: true);
        Assert.Equal(1, left[0]);
        Assert.Equal(3, right[0]);
    }

    [Fact]
    public void Histogram_CountsEqualWidthBins()
    {
        // 10 values in [0, 10]; 5 bins → each bin width = 2.
        var x = T(new[] { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f }, 10);
        var hist = E.TensorHistogram(x, 5, 0f, 10f);
        // Each bin [0,2], [2,4], [4,6], [6,8], [8,10] gets 2 values.
        Assert.Equal(new[] { 2, 2, 2, 2, 2 }, hist.AsSpan().ToArray());
    }

    [Fact]
    public void Histogram_UpperEdge_MapsIntoLastBin()
    {
        var x = T(new[] { 10f }, 1);
        var hist = E.TensorHistogram(x, 5, 0f, 10f);
        Assert.Equal(new[] { 0, 0, 0, 0, 1 }, hist.AsSpan().ToArray());
    }
}
