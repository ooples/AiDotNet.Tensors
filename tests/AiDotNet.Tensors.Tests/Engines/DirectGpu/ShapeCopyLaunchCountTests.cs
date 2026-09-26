// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// GPU tile / concatenate / last-axis narrow used to issue one device copy per outer slice (tile: outer*repeats,
/// concat: outer per input, slice: one per row). With tiny slices that is millions of launches -- tiling a
/// [1024, 1] ReduceSum gradient to [1024, 49152] was 50.3M one-element copies and made one LM training step take
/// minutes. They now use single-launch kernels (TileAxis, Copy2DStrided, CopyRows, StridedGather + Transpose).
/// These tests pin the RESULTS to the CPU engine across the shapes each new path handles, including the ones that
/// still take the general path.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ShapeCopyLaunchCountTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly CpuEngine _cpu = new();

    public ShapeCopyLaunchCountTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private IEngine Gpu => _fixture.Engine ?? throw new InvalidOperationException("No GPU engine.", _fixture.InitializationException);

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static void AssertSame(Tensor<float> expected, Tensor<float> actual)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(e[i] == a[i], $"element {i}: expected {e[i]}, got {a[i]}");
    }

    public static IEnumerable<object[]> TileCases() =>
    [
        [new[] { 64, 1 }, new[] { 1, 300 }],        // the ReduceSum-backward broadcast shape (block = 1)
        [new[] { 3, 4, 5 }, new[] { 1, 3, 1 }],     // middle axis, block = axis*inner = 20
        [new[] { 2, 3 }, new[] { 4, 1 }],           // leading axis
        [new[] { 5, 1, 7 }, new[] { 1, 6, 1 }],     // size-1 middle axis
    ];

    [SkippableTheory]
    [MemberData(nameof(TileCases))]
    public void Tile_MatchesCpu(int[] shape, int[] multiples)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var x = Rand(shape, 1);
        AssertSame(_cpu.TensorTile(x, multiples), Gpu.TensorTile(x, multiples));
    }

    public static IEnumerable<object[]> ConcatCases() =>
    [
        [new[] { 8, 16, 33, 1 }, 2, 3],   // last axis, inner 1 (the complex re/im stack)
        [new[] { 4, 5, 6 }, 3, 1],        // middle axis, three inputs
        [new[] { 2, 3 }, 2, 0],           // leading axis (outer == 1: single contiguous copy per input)
    ];

    [SkippableTheory]
    [MemberData(nameof(ConcatCases))]
    public void Concatenate_MatchesCpu(int[] shape, int count, int axis)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var inputs = Enumerable.Range(0, count).Select(i => Rand(shape, 10 + i)).ToArray();
        AssertSame(_cpu.TensorConcatenate(inputs, axis), Gpu.TensorConcatenate(inputs, axis));
    }

    public static IEnumerable<object[]> NarrowCases() =>
    [
        [new[] { 8, 16, 33, 2 }, 0, 1],   // width 1, start 0  -> strided gather
        [new[] { 8, 16, 33, 2 }, 1, 1],   // width 1, start 1  -> strided gather
        [new[] { 6, 7, 10 }, 0, 4],       // start 0           -> row-prefix copy
        [new[] { 6, 7, 10 }, 3, 5],       // start > 0, width > 1 -> column gathers + transpose
        [new[] { 6, 7, 10 }, 0, 10],      // full width        -> general path
    ];

    [SkippableTheory]
    [MemberData(nameof(NarrowCases))]
    public void LastAxisNarrow_MatchesCpu(int[] shape, int start, int length)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var x = Rand(shape, 3);
        int last = shape.Length - 1;
        AssertSame(_cpu.TensorNarrow(x, last, start, length), Gpu.TensorNarrow(x, last, start, length));
    }

    public static IEnumerable<object[]> IrfftCases() =>
    [
        [new[] { 8, 16 }, 257, 512],   // the LM spectral-FFN shape family: E = 512
        [new[] { 3 }, 9, 16],          // small, full length
        [new[] { 2, 5 }, 9, 13],       // trimmed output length
    ];

    /// <summary>
    /// Batched IRFFT (gather-based conjugate-symmetric expansion + one inverse BatchedFFT) must match the CPU
    /// transform, including spectra whose DC and Nyquist bins carry an imaginary part (a real inverse transform
    /// ignores it).
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(IrfftCases))]
    public void Irfft_MatchesCpu(int[] batchShape, int numFreqs, int outputLength)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var spectrum = Rand([.. batchShape, numFreqs * 2], 5);
        var expected = _cpu.IRFFT(spectrum, outputLength).ToArray();
        var actual = Gpu.IRFFT(spectrum, outputLength);
        Assert.Equal([.. batchShape, outputLength], actual.Shape.ToArray());
        var a = actual.ToArray();
        double scale = expected.Max(Math.Abs);
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(a[i] - expected[i]) <= 1e-4 * Math.Max(scale, 1e-6), $"element {i}: expected {expected[i]}, got {a[i]}");
    }

    [SkippableFact]
    public void LeadingAxisSlice_StillMatchesCpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var x = Rand([6, 7, 10], 4);
        int[] start = [1, 2, 3], length = [3, 4, 5];
        AssertSame(_cpu.TensorSlice(x, start, length), Gpu.TensorSlice(x, start, length));
    }
}
