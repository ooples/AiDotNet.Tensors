#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #681: the half-weight GEMM converts B inside GotoGemm's B-pack. It must equal converting B to
/// float first and running the same kernel with the same blocking, bit for bit, including the
/// M- and N-tail paths, and both must agree with a double reference.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class GotoGemmHalfBTests
{
    [SkippableTheory]
    [InlineData(256, 3072, 1024)]  // transformer FFN-like; whole tiles
    [InlineData(250, 520, 100)]    // K below the default kc of 256: one partial K panel
    [InlineData(250, 520, 300)]    // M % 6 = 4, N % 16 = 8: both tails
    [InlineData(64, 1000, 777)]    // short M, odd K, N-tail
    public void HalfB_MatchesFloatB_BitForBit_AndADoubleReference(int m, int n, int k)
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable && System.Runtime.Intrinsics.X86.Avx2.IsSupported,
            "The per-tile GotoGemm kernel needs AVX2 and its machine-code microkernel.");
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 16;
            var rng = RandomHelper.CreateSeededRandom(681);
            var a = new float[m * k];
            var bHalf = new Half[k * n];
            var bFloat = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < bHalf.Length; i++) bHalf[i] = (Half)(rng.NextDouble() - 0.5);
            // Decode edge cases: subnormals, signed zero, the largest finite half.
            bHalf[0] = BitConverter.UInt16BitsToHalf(0x0001);
            bHalf[1] = BitConverter.UInt16BitsToHalf(0x83FF);
            bHalf[2] = BitConverter.UInt16BitsToHalf(0x8000);
            bHalf[3] = (Half)0.0009765625f;
            bHalf[n] = BitConverter.UInt16BitsToHalf(0x7BFF);
            for (int i = 0; i < bHalf.Length; i++) bFloat[i] = (float)bHalf[i];

            var (mc, nc, kc) = GotoGemmFp32.ChooseParallelBlocks(m, n);
            var cHalf = new float[m * n];
            var cFloat = new float[m * n];
            for (int i = 0; i < cHalf.Length; i++) { cHalf[i] = float.NaN; cFloat[i] = float.NaN; }
            unsafe
            {
                fixed (float* pa = a)
                fixed (Half* pbh = bHalf)
                fixed (float* pbf = bFloat)
                fixed (float* pch = cHalf)
                fixed (float* pcf = cFloat)
                {
                    GotoGemmFp32.RunParallelHalfB(pa, k, (ushort*)pbh, n, pch, n, m, n, k, mc, nc, kc);
                    GotoGemmFp32.RunParallel(pa, k, pbf, n, pcf, n, m, n, k, mc, nc, kc);
                }
            }

            // The reference bound scales with sum |a*b|: B holds 65504, the largest finite half, so some
            // outputs reach ~3e4, where float rounding alone is ~1e-3 per step.
            double worstRatio = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    int idx = i * n + j;
                    Assert.Equal(BitConverter.SingleToInt32Bits(cFloat[idx]), BitConverter.SingleToInt32Bits(cHalf[idx]));
                    double expected = 0, magnitude = 0;
                    for (int p = 0; p < k; p++)
                    {
                        double term = (double)a[i * k + p] * bFloat[p * n + j];
                        expected += term;
                        magnitude += Math.Abs(term);
                    }
                    worstRatio = Math.Max(worstRatio, Math.Abs(expected - cHalf[idx]) / (magnitude + 1e-30));
                }
            Assert.True(worstRatio < 1e-5, $"max |C - C_ref| / sum|a*b| = {worstRatio:E3}");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }

    [SkippableFact]
    public void TensorMatMulFp16WeightB_MatchesConvertingTheWeightFirst()
    {
        // Only then does the engine dispatch the fused GotoGemm path this test is about.
        Skip.IfNot(GotoGemmFp32.IsAvailable && System.Runtime.Intrinsics.X86.Avx2.IsSupported,
            "The fused fp16-weight path needs AVX2 and the GotoGemm kernel.");
        var engine = new CpuEngine();
        var rng = RandomHelper.CreateSeededRandom(682);
        const int m = 256, k = 768, n = 3072;
        var a = new Tensor<float>(new[] { m, k });
        var w = new Tensor<Half>(new[] { k, n });
        var wFloat = new Tensor<float>(new[] { k, n });
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < w.Length; i++) { w[i] = (Half)(rng.NextDouble() - 0.5); wFloat[i] = (float)w[i]; }

        var fused = engine.TensorMatMulFp16WeightB(a, w);
        var upcast = engine.TensorMatMul(a, wFloat);

        Assert.Equal(new[] { m, n }, fused.Shape.ToArray());
        double worst = 0;
        for (int i = 0; i < fused.Length; i++) worst = Math.Max(worst, Math.Abs(fused[i] - upcast[i]));
        // Different blocking may reorder the K sums; the values themselves are identical.
        Assert.True(worst < 1e-4, $"max |fused - upcast| = {worst:E3}");
    }
}
#endif
