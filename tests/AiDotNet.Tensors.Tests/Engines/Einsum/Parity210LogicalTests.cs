using System.Linq;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210LogicalTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<Bit> M(bool[] data, params int[] shape) =>
        new Tensor<Bit>(data.Select(b => b ? Bit.True : Bit.False).ToArray(), shape);

    [Fact]
    public void LogicalAnd_MatchesTruthTable()
    {
        var a = M(new[] { false, false, true, true }, 4);
        var b = M(new[] { false, true, false, true }, 4);
        var r = E.TensorLogicalAnd(a, b);
        Assert.Equal(new[] { false, false, false, true }, r.AsSpan().ToArray().Select(x => (bool)x));
    }

    [Fact]
    public void LogicalOr_MatchesTruthTable()
    {
        var a = M(new[] { false, false, true, true }, 4);
        var b = M(new[] { false, true, false, true }, 4);
        var r = E.TensorLogicalOr(a, b);
        Assert.Equal(new[] { false, true, true, true }, r.AsSpan().ToArray().Select(x => (bool)x));
    }

    [Fact]
    public void LogicalXor_MatchesTruthTable()
    {
        var a = M(new[] { false, false, true, true }, 4);
        var b = M(new[] { false, true, false, true }, 4);
        var r = E.TensorLogicalXor(a, b);
        Assert.Equal(new[] { false, true, true, false }, r.AsSpan().ToArray().Select(x => (bool)x));
    }

    [Fact]
    public void LogicalNot_Inverts()
    {
        var a = M(new[] { true, false, true, false }, 4);
        var r = E.TensorLogicalNot(a);
        Assert.Equal(new[] { false, true, false, true }, r.AsSpan().ToArray().Select(x => (bool)x));
    }

    [Fact]
    public void LogicalAnd_ShapeMismatch_Throws()
    {
        var a = M(new[] { true, true }, 2);
        var b = M(new[] { true, true, false }, 3);
        Assert.Throws<System.ArgumentException>(() => E.TensorLogicalAnd(a, b));
    }
}
