using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the channel-repeat broadcast fast path in
/// <c>CpuEngine.TensorBroadcast{Add,Subtract,Multiply,Divide}</c> — the
/// new fast path for the <c>[B, C, H, W] op [1, C, 1, 1]</c> Conv2D-bias /
/// BatchNorm-rescale shape pattern that <c>TryBroadcastTrailingRepeat</c>
/// rejects. Verifies parity with the slow generic-indexer fallback by
/// constructing a known-correct reference and comparing every element.
/// </summary>
public class BroadcastChannelRepeatTests
{
    private readonly IEngine _engine = new CpuEngine();

    // ── Reference implementation (the math is unambiguous; just walks indices). ──
    private static Tensor<double> RefBroadcast(Tensor<double> a, Tensor<double> b, Func<double, double, double> op)
    {
        var res = new Tensor<double>(a._shape);
        int rank = a.Rank;
        int batch = a._shape[0], channels = a._shape[1];
        int spatial = 1;
        for (int i = 2; i < rank; i++) spatial *= a._shape[i];
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                double scalar = b.AsSpan()[c];
                int off = (n * channels + c) * spatial;
                for (int i = 0; i < spatial; i++)
                    res.AsWritableSpan()[off + i] = op(a.AsSpan()[off + i], scalar);
            }
        }
        return res;
    }

    private static Tensor<double> MakeRandom(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t.AsWritableSpan()[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    [Theory]
    [InlineData(1, 64, 8, 8)]          // small NCHW
    [InlineData(2, 16, 14, 14)]        // batch > 1
    [InlineData(1, 64, 224, 224)]      // VGG block 1 — the actual hot shape
    [InlineData(1, 128, 112, 112)]     // VGG block 2
    [InlineData(4, 32, 4, 4)]          // batch=4, channels not multiple of vector
    public void BroadcastAdd_ChannelRepeat_MatchesGeneric(int n, int c, int h, int w)
    {
        var a = MakeRandom([n, c, h, w], seed: 17);
        var b = MakeRandom([1, c, 1, 1], seed: 42);
        var fast = _engine.TensorBroadcastAdd(a, b);
        var slow = RefBroadcast(a, b, (x, y) => x + y);
        AssertEqual(slow, fast);
    }

    [Theory]
    [InlineData(1, 64, 8, 8)]
    [InlineData(2, 16, 14, 14)]
    [InlineData(1, 64, 224, 224)]
    public void BroadcastMultiply_ChannelRepeat_MatchesGeneric(int n, int c, int h, int w)
    {
        var a = MakeRandom([n, c, h, w], seed: 23);
        var b = MakeRandom([1, c, 1, 1], seed: 7);
        var fast = _engine.TensorBroadcastMultiply(a, b);
        var slow = RefBroadcast(a, b, (x, y) => x * y);
        AssertEqual(slow, fast);
    }

    [Theory]
    [InlineData(1, 32, 14, 14)]
    [InlineData(2, 16, 8, 8)]
    public void BroadcastSubtract_ChannelRepeat_MatchesGeneric(int n, int c, int h, int w)
    {
        var a = MakeRandom([n, c, h, w], seed: 11);
        var b = MakeRandom([1, c, 1, 1], seed: 13);
        var fast = _engine.TensorBroadcastSubtract(a, b);
        var slow = RefBroadcast(a, b, (x, y) => x - y);
        AssertEqual(slow, fast);
    }

    [Theory]
    [InlineData(1, 32, 14, 14)]
    [InlineData(2, 16, 8, 8)]
    public void BroadcastDivide_ChannelRepeat_MatchesGeneric(int n, int c, int h, int w)
    {
        var a = MakeRandom([n, c, h, w], seed: 31);
        // Keep b away from zero so we don't compare ±Inf
        var b = new Tensor<double>([1, c, 1, 1]);
        var rng = new Random(33);
        for (int i = 0; i < b.Length; i++) b.AsWritableSpan()[i] = 0.1 + rng.NextDouble();
        var fast = _engine.TensorBroadcastDivide(a, b);
        var slow = RefBroadcast(a, b, (x, y) => x / y);
        AssertEqual(slow, fast);
    }

    [Fact]
    public void BroadcastAdd_Rank3_NCL_ChannelRepeat_MatchesGeneric()
    {
        // Conv1D NCL shape — verifies rank != 4 still hits the path.
        var a = MakeRandom([2, 8, 17], seed: 41);
        var b = MakeRandom([1, 8, 1], seed: 43);
        var fast = _engine.TensorBroadcastAdd(a, b);
        var slow = RefBroadcast(a, b, (x, y) => x + y);
        AssertEqual(slow, fast);
    }

    [Fact]
    public void BroadcastAdd_Float_ChannelRepeat_MatchesGeneric()
    {
        // Float specialisation — fp32 AVX kernel must also be correct.
        var rng = new Random(91);
        int n = 1, c = 16, h = 14, w = 14;
        var a = new Tensor<float>([n, c, h, w]);
        var b = new Tensor<float>([1, c, 1, 1]);
        for (int i = 0; i < a.Length; i++) a.AsWritableSpan()[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b.AsWritableSpan()[i] = (float)(rng.NextDouble() * 2 - 1);
        var fast = _engine.TensorBroadcastAdd(a, b);
        // Reference: just walk
        var slow = new Tensor<float>([n, c, h, w]);
        int spatial = h * w;
        for (int nn = 0; nn < n; nn++)
            for (int cc = 0; cc < c; cc++)
            {
                float scalar = b.AsSpan()[cc];
                int off = (nn * c + cc) * spatial;
                for (int i = 0; i < spatial; i++)
                    slow.AsWritableSpan()[off + i] = a.AsSpan()[off + i] + scalar;
            }
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(slow.AsSpan()[i], fast.AsSpan()[i]);
    }

    private static void AssertEqual(Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected.AsSpan()[i], actual.AsSpan()[i]);
    }
}
