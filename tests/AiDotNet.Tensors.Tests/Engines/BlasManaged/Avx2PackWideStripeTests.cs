using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #653: Avx2Pack.PackB_Fp32 packs stripes wider than 8 (PackBoth's 16/24/32-column kernels) with vector copies instead
/// of the scalar fallback. Packing only moves values, so the output must be bit-identical to ScalarPack, including the
/// zero-padded tail stripe and a row stride wider than the panel.
/// </summary>
public sealed class Avx2PackWideStripeTests
{
    [SkippableTheory]
    [InlineData(16, 256, 64, 300)]
    [InlineData(16, 37, 19, 40)]     // tail stripe of 5 columns
    [InlineData(24, 240, 33, 1000)]
    [InlineData(24, 50, 7, 50)]      // tail of 2
    [InlineData(32, 256, 16, 256)]
    [InlineData(32, 31, 5, 31)]      // tail only
    [InlineData(8, 64, 9, 70)]       // the existing nr == 8 path, as a control
    public void PackB_Fp32_WideStripes_AreBitIdenticalToScalar(int nr, int nc, int kc, int ldb)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported, "AVX2 is required.");
        var b = new float[kc * ldb];
        for (int i = 0; i < b.Length; i++) b[i] = (float)Math.Sin(i * 0.37) * (i % 7 - 3);
        int packedLength = ((nc + nr - 1) / nr) * nr * kc;
        var expected = new float[packedLength];
        var actual = new float[packedLength];
        for (int i = 0; i < packedLength; i++) { expected[i] = float.NaN; actual[i] = float.NaN; }
        ScalarPack.PackB<float>(b, ldb, false, expected, nc, kc, nr);
        Avx2Pack.PackB_Fp32(b, ldb, false, actual, nc, kc, nr);
        for (int i = 0; i < packedLength; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"nr={nr} nc={nc} kc={kc}: packed[{i}] {actual[i]} != {expected[i]}");
    }
}
