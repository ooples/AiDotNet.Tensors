using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A reduction over every axis without keepDims returns a rank-0 tensor, as torch does
/// (<c>torch.amax(x)</c>, <c>x.sum()</c>, <c>x.max()</c> all have shape <c>[]</c>) and as the
/// GPU engine does. With keepDims every reduced axis stays as a size-1 dimension. Issue #1132:
/// the CPU max family returned <c>[1]</c> while ReduceSum and the GPU returned rank 0.
/// </summary>
public class FullReductionRankTests
{
    public enum Reduction { Sum, Mean, Max, MaxWithIndices, Amin, Variance, Std, LogVariance }

    private static readonly CpuEngine Engine = new CpuEngine();

    private static Tensor<double> Reduce(Reduction op, Tensor<double> x, int[] axes, bool keepDims) => op switch
    {
        Reduction.Sum => Engine.ReduceSum(x, axes, keepDims),
        Reduction.Mean => Engine.ReduceMean(x, axes, keepDims),
        Reduction.Max => Engine.ReduceMax(x, axes, keepDims),
        Reduction.MaxWithIndices => Engine.ReduceMaxWithTensorIndices(x, axes, keepDims, out _),
        Reduction.Amin => Engine.TensorAmin(x, axes, keepDims),
        Reduction.Variance => Engine.ReduceVariance(x, axes, keepDims),
        Reduction.Std => Engine.ReduceStd(x, axes, keepDims),
        Reduction.LogVariance => Engine.ReduceLogVariance(x, axes, keepDims),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    public static TheoryData<Reduction, int> Cases()
    {
        var data = new TheoryData<Reduction, int>();
        foreach (Reduction op in Enum.GetValues(typeof(Reduction)))
            foreach (int rank in new[] { 1, 2, 3 })
                data.Add(op, rank);
        return data;
    }

    private static Tensor<double> Input(int rank)
    {
        int[] shape = rank switch { 1 => new[] { 6 }, 2 => new[] { 2, 3 }, _ => new[] { 1, 2, 3 } };
        return new Tensor<double>(new[] { 3.0, -1.0, 7.5, 2.0, -4.0, 0.5 }, shape);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void AllAxes_NoKeepDims_IsRankZero(Reduction op, int rank)
    {
        var x = Input(rank);
        var allAxes = new int[rank];
        for (int i = 0; i < rank; i++) allAxes[i] = i;

        var explicitAxes = Reduce(op, x, allAxes, keepDims: false);
        Assert.Empty(explicitAxes.Shape.ToArray());
        Assert.Equal(1, explicitAxes.Length);

        var emptyAxes = Reduce(op, x, Array.Empty<int>(), keepDims: false);
        Assert.Empty(emptyAxes.Shape.ToArray());
        Assert.Equal(explicitAxes[0], emptyAxes[0], 12);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void AllAxes_KeepDims_KeepsEveryAxisAsOne(Reduction op, int rank)
    {
        var x = Input(rank);
        var allAxes = new int[rank];
        for (int i = 0; i < rank; i++) allAxes[i] = i;

        var r = Reduce(op, x, allAxes, keepDims: true);
        var expected = new int[rank];
        for (int i = 0; i < rank; i++) expected[i] = 1;
        Assert.Equal(expected, r.Shape.ToArray());
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void AllAxes_NoKeepDims_CompiledMatchesEagerRank(Reduction op, int rank)
    {
        if (op == Reduction.MaxWithIndices) return; // heterogeneous output: not graph-capturable by design
        var x = Input(rank);
        var allAxes = new int[rank];
        for (int i = 0; i < rank; i++) allAxes[i] = i;

        var eager = Reduce(op, x, allAxes, keepDims: false);
        using var cache = new CompiledModelCache<double>();
        var plan = cache.GetOrCompileInference(x._shape, () => Reduce(op, x, allAxes, keepDims: false));
        var compiled = plan.Execute();

        Assert.Empty(compiled.Shape.ToArray());
        Assert.Equal(eager[0], compiled[0], 10);
    }

    [Fact]
    public void ReduceMaxWithTensorIndices_FullReduction_IndexIsRankZero()
    {
        var x = Input(2);
        var r = Engine.ReduceMaxWithTensorIndices(x, new[] { 0, 1 }, keepDims: false, out var idx);
        Assert.Empty(r.Shape.ToArray());
        Assert.Empty(idx.Shape.ToArray());
        Assert.Equal(2, idx[0]);
        Assert.Equal(7.5, r[0], 12);
    }
}
