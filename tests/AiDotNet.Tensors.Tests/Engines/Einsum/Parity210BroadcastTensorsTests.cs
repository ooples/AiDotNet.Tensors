using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210BroadcastTensorsTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void BroadcastTensors_MakesAllMatchCommonShape()
    {
        var a = T(new[] { 1f, 2f }, 1, 2);
        var b = T(new[] { 3f, 4f, 5f }, 3, 1);
        var arr = E.TensorBroadcastTensors(new[] { a, b });
        Assert.Equal(2, arr.Length);
        Assert.Equal(new[] { 3, 2 }, arr[0].Shape.ToArray());
        Assert.Equal(new[] { 3, 2 }, arr[1].Shape.ToArray());
        // a broadcast: row-repeated 3×2: 1 2 | 1 2 | 1 2
        Assert.Equal(new[] { 1f, 2f, 1f, 2f, 1f, 2f }, arr[0].AsSpan().ToArray());
        // b broadcast: col-repeated 3×2: 3 3 | 4 4 | 5 5
        Assert.Equal(new[] { 3f, 3f, 4f, 4f, 5f, 5f }, arr[1].AsSpan().ToArray());
    }

    [Fact]
    public void BroadcastTensors_Incompatible_Throws()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 3f, 4f, 5f }, 3);
        Assert.Throws<System.ArgumentException>(() => E.TensorBroadcastTensors(new[] { a, b }));
    }

    [Fact]
    public void BroadcastTensors_Empty_ReturnsEmpty()
    {
        var arr = E.TensorBroadcastTensors(System.Array.Empty<Tensor<float>>());
        Assert.Empty(arr);
    }

    [Fact]
    public void UniqueConsecutive_CollapsesRuns()
    {
        var x = T(new[] { 1f, 1f, 2f, 2f, 2f, 3f, 1f, 1f }, 8);
        var r = E.TensorUniqueConsecutive(x);
        // Consecutive runs collapsed: 1, 2, 3, 1 — the final 1 remains because
        // it's not adjacent to the first run.
        Assert.Equal(new[] { 1f, 2f, 3f, 1f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void UniqueConsecutive_Empty_IsEmpty()
    {
        var x = T(System.Array.Empty<float>(), 0);
        var r = E.TensorUniqueConsecutive(x);
        Assert.Equal(new[] { 0 }, r.Shape.ToArray());
    }
}
