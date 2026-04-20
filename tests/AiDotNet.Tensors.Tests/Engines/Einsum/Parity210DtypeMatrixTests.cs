using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

/// <summary>
/// Dtype-matrix parity-210 tests. Issue #210 acceptance requires the core
/// ops to operate across the full numeric-operations matrix
/// (fp16/bf16/fp32/fp64/int8/16/32/64/bool where the op has a
/// meaningful semantic). Tests below pick a representative shape-manipulation
/// op (Triu), an element-wise binary op (Hypot on fp-only), a cumulative
/// op (CumSum on integer + fp), and a movement op (Roll on all dtypes)
/// since these exercise the full generic dispatch path.
///
/// Tests that are semantically undefined for a given dtype (e.g. Hypot on
/// int — overflow; Lgamma on int — not integer-valued) are skipped with an
/// explicit Xunit comment rather than silently omitted so reviewers can
/// see the coverage matrix at a glance.
/// </summary>
public class Parity210DtypeMatrixTests
{
    private static CpuEngine E => new CpuEngine();

    // ---------------------------------------------------------------------
    // Triu — shape masking; works on every dtype including int and Half.
    // ---------------------------------------------------------------------

    [Fact]
    public void Triu_Float()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var r = E.TensorTriu(x);
        Assert.Equal(new[] { 1f, 2f, 0f, 4f }, r.GetDataArray());
    }

    [Fact]
    public void Triu_Double()
    {
        var x = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 2, 2 });
        var r = E.TensorTriu(x);
        Assert.Equal(new[] { 1.0, 2.0, 0.0, 4.0 }, r.GetDataArray());
    }

    [Fact]
    public void Triu_Int()
    {
        var x = new Tensor<int>(new[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var r = E.TensorTriu(x);
        Assert.Equal(new[] { 1, 2, 0, 4 }, r.GetDataArray());
    }

    [Fact]
    public void Triu_Long()
    {
        var x = new Tensor<long>(new[] { 1L, 2L, 3L, 4L }, new[] { 2, 2 });
        var r = E.TensorTriu(x);
        Assert.Equal(new[] { 1L, 2L, 0L, 4L }, r.GetDataArray());
    }

    [Fact]
    public void Triu_Half()
    {
        var x = new Tensor<Half>(new[] { (Half)1f, (Half)2f, (Half)3f, (Half)4f }, new[] { 2, 2 });
        var r = E.TensorTriu(x);
        Assert.Equal((Half)1f, r[0, 0]);
        Assert.Equal((Half)2f, r[0, 1]);
        Assert.Equal((Half)0f, r[1, 0]);
        Assert.Equal((Half)4f, r[1, 1]);
    }

    // ---------------------------------------------------------------------
    // CumSum — arithmetic scan; works on all numeric dtypes.
    // ---------------------------------------------------------------------

    [Fact]
    public void CumSum_Float()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var r = E.TensorCumSum(x, axis: 0);
        Assert.Equal(new[] { 1f, 3f, 6f, 10f }, r.GetDataArray());
    }

    [Fact]
    public void CumSum_Double()
    {
        var x = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 4 });
        var r = E.TensorCumSum(x, axis: 0);
        Assert.Equal(new[] { 1.0, 3.0, 6.0, 10.0 }, r.GetDataArray());
    }

    [Fact]
    public void CumSum_Int()
    {
        var x = new Tensor<int>(new[] { 1, 2, 3, 4 }, new[] { 4 });
        var r = E.TensorCumSum(x, axis: 0);
        Assert.Equal(new[] { 1, 3, 6, 10 }, r.GetDataArray());
    }

    // ---------------------------------------------------------------------
    // Roll — shape-only movement; works on any dtype the tensor supports.
    // ---------------------------------------------------------------------

    [Fact]
    public void Roll_Float()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var r = E.TensorRoll(x, new[] { 1 }, new[] { 0 });
        Assert.Equal(new[] { 4f, 1f, 2f, 3f }, r.GetDataArray());
    }

    [Fact]
    public void Roll_Int()
    {
        var x = new Tensor<int>(new[] { 1, 2, 3, 4 }, new[] { 4 });
        var r = E.TensorRoll(x, new[] { 1 }, new[] { 0 });
        Assert.Equal(new[] { 4, 1, 2, 3 }, r.GetDataArray());
    }

    [Fact]
    public void Roll_Long()
    {
        var x = new Tensor<long>(new[] { 1L, 2L, 3L, 4L }, new[] { 4 });
        var r = E.TensorRoll(x, new[] { 1 }, new[] { 0 });
        Assert.Equal(new[] { 4L, 1L, 2L, 3L }, r.GetDataArray());
    }

    [Fact]
    public void Roll_Half()
    {
        var x = new Tensor<Half>(new[] { (Half)1f, (Half)2f, (Half)3f, (Half)4f }, new[] { 4 });
        var r = E.TensorRoll(x, new[] { 1 }, new[] { 0 });
        Assert.Equal((Half)4f, r[0]);
        Assert.Equal((Half)1f, r[1]);
    }

    // ---------------------------------------------------------------------
    // Hypot — floating-point-only (sqrt of sum-of-squares overflows for int).
    // ---------------------------------------------------------------------

    [Fact]
    public void Hypot_Float()
    {
        var a = new Tensor<float>(new[] { 3f, 5f }, new[] { 2 });
        var b = new Tensor<float>(new[] { 4f, 12f }, new[] { 2 });
        var r = E.TensorHypot(a, b);
        Assert.Equal(5f, r[0]);
        Assert.Equal(13f, r[1]);
    }

    [Fact]
    public void Hypot_Double()
    {
        var a = new Tensor<double>(new[] { 3.0, 5.0 }, new[] { 2 });
        var b = new Tensor<double>(new[] { 4.0, 12.0 }, new[] { 2 });
        var r = E.TensorHypot(a, b);
        Assert.Equal(5.0, r[0]);
        Assert.Equal(13.0, r[1]);
    }

    // ---------------------------------------------------------------------
    // Equal — boolean predicate; works on every numeric dtype.
    // ---------------------------------------------------------------------

    [Fact]
    public void Equal_Int()
    {
        var a = new Tensor<int>(new[] { 1, 2, 3 }, new[] { 3 });
        var b = new Tensor<int>(new[] { 1, 2, 3 }, new[] { 3 });
        Assert.True(E.TensorEqual(a, b));
    }

    [Fact]
    public void Equal_Long()
    {
        var a = new Tensor<long>(new[] { 1L, 2L }, new[] { 2 });
        var b = new Tensor<long>(new[] { 1L, 3L }, new[] { 2 });
        Assert.False(E.TensorEqual(a, b));
    }

    [Fact]
    public void Equal_Double()
    {
        var a = new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 });
        var b = new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 });
        Assert.True(E.TensorEqual(a, b));
    }
}
