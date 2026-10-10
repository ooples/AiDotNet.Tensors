using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// SimdKernels.MultiplyScalar(double) handles leading elements scalar until the destination is 32-byte aligned (an
/// unaligned 32-byte store that splits a cache line made the kernel ~2x slower in some processes). Every destination
/// offset 0..3 elements, so every head length, against every length around the 16-element unroll.
/// </summary>
public class AlignedScalarMultiplyTests
{
    [Fact]
    public void MultiplyScalar_Double_MatchesScalarForEveryHeadAndTail()
    {
        var rnd = new Random(5);
        var source = new double[1100];
        for (int i = 0; i < source.Length; i++) source[i] = rnd.NextDouble() * 4 - 2;
        int[] lengths = { 0, 1, 3, 4, 5, 15, 16, 17, 19, 20, 31, 32, 33, 35, 64, 99, 100, 1000 };
        for (int dstOffset = 0; dstOffset < 4; dstOffset++)
            for (int srcOffset = 0; srcOffset < 4; srcOffset++)
                foreach (int n in lengths)
                {
                    var dst = new double[n + 8];
                    for (int i = 0; i < dst.Length; i++) dst[i] = double.NaN;
                    SimdKernels.MultiplyScalar(source.AsSpan(srcOffset, n), -1.75, dst.AsSpan(dstOffset, n));
                    for (int i = 0; i < n; i++)
                        Assert.Equal(source[srcOffset + i] * -1.75, dst[dstOffset + i]);
                    for (int i = 0; i < dstOffset; i++) Assert.True(double.IsNaN(dst[i]), $"wrote before the destination (offset {dstOffset}, n {n})");
                    for (int i = dstOffset + n; i < dst.Length; i++) Assert.True(double.IsNaN(dst[i]), $"wrote past the destination (offset {dstOffset}, n {n})");
                }
    }
}
