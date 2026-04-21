using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// B2 + B6 parity: AVX-512 SGEMM must match the AVX2 reference bit-exact
/// (within float summation tolerance) for aligned shapes, and must silently
/// fall through to AVX2 for unaligned / small shapes.
///
/// <para>These tests call SimdGemm.Sgemm directly — its entry point dispatches
/// into Avx512Sgemm.SgemmBlocked when the CPU supports AVX-512F. On AVX2-only
/// CPUs the whole test set exercises the feature-gate no-op, which is also a
/// useful regression (the gate stays bit-identical to AVX2).</para>
/// </summary>
public class Avx512SgemmParity
{
    [Theory]
    [InlineData(16, 16, 16)]   // minimum tile
    [InlineData(32, 64, 48)]   // 2×4 tiles, m % 16 == n % 16 == 0
    [InlineData(64, 128, 16)]  // skinny right
    [InlineData(16, 512, 32)]  // deep K
    public void Sgemm_AlignedShape_BitExact_vs_ReferenceMatmul(int m, int k, int n)
    {
        var a = Random(m * k, seed: 0xA512);
        var b = Random(k * n, seed: 0xB512);
        var cFast = new float[m * n];
        var cRef  = new float[m * n];

        SimdGemm.Sgemm(a, b, cFast, m, k, n);
        ReferenceMatmul(a, b, cRef, m, k, n);

        for (int i = 0; i < cRef.Length; i++)
        {
            float d = Math.Abs(cFast[i] - cRef[i]);
            float scale = Math.Max(Math.Abs(cRef[i]), 1f);
            Assert.True(d <= 1e-3f * scale,
                $"[{i}] fast={cFast[i]} ref={cRef[i]} diff={d}");
        }
    }

    [Theory]
    [InlineData(7, 13, 11)]    // all misaligned — exercises fallback
    [InlineData(8, 8, 8)]      // below minimum tile, exercises fallback
    [InlineData(15, 15, 15)]   // one below tile boundary
    public void Sgemm_UnalignedShape_FallsBackCorrectly(int m, int k, int n)
    {
        var a = Random(m * k, seed: 0xC);
        var b = Random(k * n, seed: 0xD);
        var cFast = new float[m * n];
        var cRef  = new float[m * n];

        SimdGemm.Sgemm(a, b, cFast, m, k, n);
        ReferenceMatmul(a, b, cRef, m, k, n);

        for (int i = 0; i < cRef.Length; i++)
        {
            float d = Math.Abs(cFast[i] - cRef[i]);
            float scale = Math.Max(Math.Abs(cRef[i]), 1f);
            Assert.True(d <= 1e-3f * scale,
                $"[{i}] fast={cFast[i]} ref={cRef[i]} diff={d}");
        }
    }

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    // Dead-simple O(mkn) reference. Not for speed — for truth.
    private static void ReferenceMatmul(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        Span<float> c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float s = 0f;
                for (int p = 0; p < k; p++)
                    s += a[i * k + p] * b[p * n + j];
                c[i * n + j] = s;
            }
        }
    }
}
