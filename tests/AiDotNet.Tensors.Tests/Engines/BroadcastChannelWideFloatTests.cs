using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The float channel-repeat broadcast's wide paths: the row broadcast <c>[N, C] OP [1, C]</c> (one-element planes,
/// vectorised across C and parallel over N - a tied LM head's per-token gain multiply is <c>[rows, V] * [1, V]</c>) and
/// large tensors parallel over planes. Every element must equal the scalar float result exactly: the paths are
/// elementwise, so neither vectorisation nor the parallel split may change a single bit.
/// </summary>
public class BroadcastChannelWideFloatTests
{
    private readonly IEngine _engine = new CpuEngine();

    private static Tensor<float> Random(int[] shape, int seed, bool nonZero = false)
    {
        var t = new Tensor<float>(shape);
        var rng = RandomHelper.CreateSeededRandom(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++)
        {
            float v = (float)(rng.NextDouble() * 2 - 1);
            s[i] = nonZero ? (v >= 0 ? v + 0.5f : v - 0.5f) : v;
        }
        return t;
    }

    private static void AssertExact(Tensor<float> a, Tensor<float> b, Tensor<float> got, Func<float, float, float> op)
    {
        int channels = a._shape[1], spatial = a.Length / (a._shape[0] * channels);
        var sa = a.AsSpan(); var sb = b.AsSpan(); var sg = got.AsSpan();
        for (int i = 0; i < sa.Length; i++)
        {
            float expected = op(sa[i], sb[(i / spatial) % channels]);
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected) == TestHelpers.MathCompat.SingleToInt32Bits(sg[i]),
                $"element {i}: expected {expected:R}, got {sg[i]:R}");
        }
    }

    public static TheoryData<int, int> RowShapes => new()
    {
        { 3, 7 }, { 4, 8 }, { 1, 8 }, { 3, 13 }, { 1, 49152 }, { 512, 1001 }, { 64, 49152 },
    };

    [Theory]
    [MemberData(nameof(RowShapes))]
    public void RowBroadcast_AllOps_BitExact(int rows, int cols)
    {
        var a = Random([rows, cols], 1);
        var b = Random([1, cols], 2, nonZero: true);
        AssertExact(a, b, _engine.TensorAdd(a, b), (x, y) => x + y);
        AssertExact(a, b, _engine.TensorSubtract(a, b), (x, y) => x - y);
        AssertExact(a, b, _engine.TensorMultiply(a, b), (x, y) => x * y);
        AssertExact(a, b, _engine.TensorDivide(a, b), (x, y) => x / y);
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// The row path takes over at one full vector of channels: a 7-wide row stays on the serial plane loop and an
    /// 8-wide row takes the wide kernel. The end-to-end cases above cover both results; this pins the dispatch.
    /// </summary>
    [SkippableTheory]
    [InlineData(7, false)]
    [InlineData(8, true)]
    public void RowBroadcast_DispatchesAtOneFullVector(int cols, bool wide)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx.IsSupported, "AVX is not available on this host.");
        const int rows = 4;
        var a = new float[rows * cols];
        var b = new float[cols];
        var r = new float[rows * cols];
        Assert.Equal(wide, CpuEngine.TryBroadcastChannelFloatWide(a, 0, b, 0, r, 0, rows, cols, 1, CpuEngine.BroadcastOp.Add));
    }
#endif

    [Theory]
    [InlineData(4, 64, 32, 32)]   // 262,144 elements: parallel over planes
    [InlineData(2, 3, 300, 129)]  // planes not a multiple of the vector width
    public void LargePlanes_AllOps_BitExact(int n, int c, int h, int w)
    {
        var a = Random([n, c, h, w], 3);
        var b = Random([1, c, 1, 1], 4, nonZero: true);
        AssertExact(a, b, _engine.TensorAdd(a, b), (x, y) => x + y);
        AssertExact(a, b, _engine.TensorSubtract(a, b), (x, y) => x - y);
        AssertExact(a, b, _engine.TensorMultiply(a, b), (x, y) => x * y);
        AssertExact(a, b, _engine.TensorDivide(a, b), (x, y) => x / y);
    }
}
