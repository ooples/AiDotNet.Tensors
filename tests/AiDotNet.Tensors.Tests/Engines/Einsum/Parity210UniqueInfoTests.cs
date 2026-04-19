using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210UniqueInfoTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void UniqueWithInfo_ReturnsSortedValues()
    {
        var x = new Tensor<int>(new[] { 3, 1, 2, 1, 3, 2 }, new[] { 6 });
        var (values, inv, cnt) = E.TensorUniqueWithInfo(x, sorted: true);
        Assert.Equal(new[] { 1, 2, 3 }, values.GetDataArray());
        Assert.Null(inv);
        Assert.Null(cnt);
    }

    [Fact]
    public void UniqueWithInfo_ReturnInverse_MapsBackToUniques()
    {
        var x = new Tensor<int>(new[] { 3, 1, 2, 1, 3, 2 }, new[] { 6 });
        var (values, inv, _) = E.TensorUniqueWithInfo(x, sorted: true, returnInverse: true);
        // Sorted uniques: [1, 2, 3]
        // x[0]=3 -> 2, x[1]=1 -> 0, x[2]=2 -> 1, x[3]=1 -> 0, x[4]=3 -> 2, x[5]=2 -> 1
        Assert.NotNull(inv);
        Assert.Equal(new[] { 2, 0, 1, 0, 2, 1 }, inv!.GetDataArray());
        // Round-trip: values[inv[i]] == x[i]
        for (int i = 0; i < x.Length; i++)
            Assert.Equal(x[i], values[inv[i]]);
    }

    [Fact]
    public void UniqueWithInfo_ReturnCounts_CountsOccurrences()
    {
        var x = new Tensor<int>(new[] { 3, 1, 2, 1, 3, 2 }, new[] { 6 });
        var (values, _, cnt) = E.TensorUniqueWithInfo(x, sorted: true, returnCounts: true);
        Assert.NotNull(cnt);
        // Sorted uniques [1, 2, 3] each appear 2x
        Assert.Equal(new[] { 2, 2, 2 }, cnt!.GetDataArray());
    }

    [Fact]
    public void UniqueWithInfo_Unsorted_PreservesFirstSeenOrder()
    {
        var x = new Tensor<int>(new[] { 3, 1, 2, 1, 3, 2 }, new[] { 6 });
        var (values, _, _) = E.TensorUniqueWithInfo(x, sorted: false);
        Assert.Equal(new[] { 3, 1, 2 }, values.GetDataArray());
    }

    [Fact]
    public void UniqueConsecutiveWithInfo_CollapsesRuns()
    {
        var x = new Tensor<int>(new[] { 1, 1, 2, 2, 2, 3, 1, 1 }, new[] { 8 });
        var (values, inv, cnt) = E.TensorUniqueConsecutiveWithInfo(x, returnInverse: true, returnCounts: true);
        Assert.Equal(new[] { 1, 2, 3, 1 }, values.GetDataArray());
        Assert.NotNull(inv);
        Assert.Equal(new[] { 0, 0, 1, 1, 1, 2, 3, 3 }, inv!.GetDataArray());
        Assert.NotNull(cnt);
        Assert.Equal(new[] { 2, 3, 1, 2 }, cnt!.GetDataArray());
    }

    [Fact]
    public void UniqueWithInfo_EmptyInput_ReturnsEmptyResults()
    {
        var x = new Tensor<int>(new int[0], new[] { 0 });
        var (values, inv, cnt) = E.TensorUniqueWithInfo(x, returnInverse: true, returnCounts: true);
        Assert.Empty(values.GetDataArray());
        Assert.NotNull(inv);
        Assert.Empty(inv!.GetDataArray());
        Assert.NotNull(cnt);
        Assert.Empty(cnt!.GetDataArray());
    }
}
