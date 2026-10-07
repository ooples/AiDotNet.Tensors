using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The destination-taking ops must write into the destination rather than compute a result and copy
/// it (#653). <c>MatMulInto</c> on batched inputs, <c>TransposeInto</c> in 2D and <c>ConcatInto</c>
/// used to allocate the whole result first: 4.1 MB, 769 KB and 1.5 MB per call at attention-block
/// shapes, which made each slower than the out-of-place op it was meant to replace.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class IntoOpsWriteThroughTests
{
    private readonly CpuEngine _engine = new();

    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static void AssertSame(Tensor<float> expected, Tensor<float> actual)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 4);
    }

    [Fact]
    public void MatMulInto_SameRankBatched_MatchesBatchMatMul()
    {
        var q = Random(new[] { 12, 32, 16 }, 1);
        var kT = Random(new[] { 12, 16, 24 }, 2);
        var destination = new Tensor<float>(new[] { 12, 32, 24 });

        _engine.MatMulInto(destination, q, kT);

        AssertSame(_engine.BatchMatMul(q, kT), destination);
    }

    [Fact]
    public void MatMulInto_BatchedTimesMatrix_MatchesTensorMatMul()
    {
        var x = Random(new[] { 3, 20, 16 }, 3);
        var w = Random(new[] { 16, 8 }, 4);
        var destination = new Tensor<float>(new[] { 3, 20, 8 });

        _engine.MatMulInto(destination, x, w);

        AssertSame(_engine.TensorMatMul(x, w), destination);
    }

    [Fact]
    public void MatMulInto_Double_Batched_MatchesBatchMatMul()
    {
        var a = new Tensor<double>(new[] { 2, 5, 4 });
        var b = new Tensor<double>(new[] { 2, 4, 3 });
        for (int i = 0; i < a.Length; i++) a[i] = 0.1 * i - 1.0;
        for (int i = 0; i < b.Length; i++) b[i] = 0.05 * i + 0.3;
        var destination = new Tensor<double>(new[] { 2, 5, 3 });

        _engine.MatMulInto(destination, a, b);

        var expected = _engine.BatchMatMul(a, b);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], destination[i], 12);
    }

    [Fact]
    public void MatMulInto_RejectsAWrongSizedDestination()
    {
        var q = Random(new[] { 4, 8, 6 }, 5);
        var kT = Random(new[] { 4, 6, 8 }, 6);
        var tooSmall = new Tensor<float>(new[] { 4, 8, 6 });

        Assert.Throws<ArgumentException>(() => _engine.MatMulInto(tooSmall, q, kT));
    }

    [Fact]
    public void MatMulInto_RejectsMismatchedInnerDimensions()
    {
        var a = Random(new[] { 2, 4, 5 }, 7);
        var b = Random(new[] { 2, 6, 3 }, 8);
        var destination = new Tensor<float>(new[] { 2, 4, 3 });

        Assert.Throws<ArgumentException>(() => _engine.MatMulInto(destination, a, b));
    }

    [Fact]
    public void TransposeInto_2D_MatchesTensorTranspose()
    {
        var x = Random(new[] { 7, 13 }, 9);
        var destination = new Tensor<float>(new[] { 13, 7 });

        _engine.TransposeInto(destination, x, new[] { 1, 0 });

        AssertSame(_engine.TensorTranspose(x), destination);
    }

    [Fact]
    public void TransposeInto_2D_RejectsAnUntransposedDestination()
    {
        var x = Random(new[] { 7, 13 }, 10);
        var wrong = new Tensor<float>(new[] { 7, 13 });

        Assert.Throws<ArgumentException>(() => _engine.TransposeInto(wrong, x, new[] { 1, 0 }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(-1)]
    public void ConcatInto_MatchesConcat_OnEveryAxis(int axis)
    {
        var a = Random(new[] { 2, 3, 4 }, 11);
        var b = Random(new[] { 2, 3, 4 }, 12);
        var c = Random(new[] { 2, 3, 4 }, 13);
        var expected = _engine.Concat(new[] { a, b, c }, axis);
        var destination = new Tensor<float>(expected.Shape.ToArray());

        _engine.ConcatInto(destination, new[] { a, b, c }, axis);

        AssertSame(expected, destination);
    }

    [Fact]
    public void ConcatInto_UnevenSlabs_MatchConcat()
    {
        var a = Random(new[] { 3, 2, 5 }, 14);
        var b = Random(new[] { 3, 4, 5 }, 15);
        var expected = _engine.Concat(new[] { a, b }, 1);
        var destination = new Tensor<float>(new[] { 3, 6, 5 });

        _engine.ConcatInto(destination, new[] { a, b }, 1);

        AssertSame(expected, destination);
    }

    [Fact]
    public void ConcatInto_RejectsADestinationOfTheWrongLength()
    {
        var a = Random(new[] { 2, 3 }, 16);
        var b = Random(new[] { 2, 3 }, 17);
        var wrong = new Tensor<float>(new[] { 2, 5 });

        Assert.Throws<ArgumentException>(() => _engine.ConcatInto(wrong, new[] { a, b }, 1));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1)]
    [InlineData(0)]
    public void SoftmaxInto_InPlace_MatchesSoftmax(int axis)
    {
        var x = Random(new[] { 4, 6, 5 }, 21);
        var expected = _engine.Softmax(x, axis);

        _engine.SoftmaxInto(x, x, axis);

        AssertSame(expected, x);
    }

    [Fact]
    public void SoftmaxInto_InPlace_Double_MatchesSoftmax()
    {
        var x = new Tensor<double>(new[] { 3, 7 });
        for (int i = 0; i < x.Length; i++) x[i] = 0.3 * i - 2.0;
        var expected = _engine.Softmax(x, -1);

        _engine.SoftmaxInto(x, x, -1);

        for (int i = 0; i < x.Length; i++) Assert.Equal(expected[i], x[i], 12);
    }

    [Fact]
    public void SoftmaxInto_RejectsADestinationOfTheWrongShape()
    {
        var x = Random(new[] { 4, 6 }, 22);
        var wrong = new Tensor<float>(new[] { 4, 5 });

        Assert.Throws<ArgumentException>(() => _engine.SoftmaxInto(wrong, x, -1));
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void IntoOps_DoNotAllocateTheirResult()
    {
        // Single-threaded, so every byte lands on this thread's allocation counter.
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = 1;
        try
        {
            AssertWriteThrough();
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }

    private void AssertWriteThrough()
    {
        var q = Random(new[] { 12, 256, 64 }, 18);
        var kT = Random(new[] { 12, 64, 256 }, 19);
        var scores = new Tensor<float>(new[] { 12, 256, 256 });
        var x = Random(new[] { 256, 768 }, 20);
        var xT = new Tensor<float>(new[] { 768, 256 });
        var both = new Tensor<float>(new[] { 256, 1536 });

        // Each result is 0.75-3 MB. A write-through op allocates a few hundred bytes of bookkeeping.
        const long budget = 64 * 1024;
        Assert.True(AllocatedBy(() => _engine.MatMulInto(scores, q, kT)) < budget, "MatMulInto allocated its result");
        Assert.True(AllocatedBy(() => _engine.TransposeInto(xT, x, new[] { 1, 0 })) < budget, "TransposeInto allocated its result");
        Assert.True(AllocatedBy(() => _engine.ConcatInto(both, new[] { x, x }, 1)) < budget, "ConcatInto allocated its result");
    }

    private static long AllocatedBy(Action op)
    {
        op();
        long before = GC.GetAllocatedBytesForCurrentThread();
        op();
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }
#endif
}
