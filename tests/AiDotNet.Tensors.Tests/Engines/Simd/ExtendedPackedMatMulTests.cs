using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Tests for issue #207 B1 expansion — Int2 / Int3 / NF4 / FP4 matmul
/// kernels and the AVX-2-accelerated XNOR popcount helper.
/// </summary>
public class ExtendedPackedMatMulTests
{
    [Fact]
    public void Int2MatMul_ReferenceParity_WithinQuantError()
    {
        const int M = 2, K = 16, N = 3;
        var rng = new Random(1);
        var wFloat = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < wFloat.Length; i++) wFloat[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Quantize row-by-row with one scale per row (group=K).
        var packed = new PackedInt2[M * K / PackedInt2.ValuesPerByte];
        var rowScales = new float[M];
        for (int i = 0; i < M; i++)
        {
            var rowDst = packed.AsSpan(i * (K / 4), K / 4);
            var s = QuantizationHelpers.QuantizeInt2(wFloat.AsSpan(i * K, K), rowDst, groupSize: K);
            rowScales[i] = s.Scales[0];
        }
        var scale = new QuantizationScale(rowScales, groupSize: K);

        var c = new float[M * N];
        PackedMatMul.Int2WeightMatMul(packed, scale, b, c, M, K, N);

        // Reference: dequant the weights then float matmul.
        var dequant = new float[M * K];
        for (int i = 0; i < M; i++)
            QuantizationHelpers.DequantizeInt2(
                packed.AsSpan(i * (K / 4), K / 4),
                new QuantizationScale(new[] { rowScales[i] }, K),
                dequant.AsSpan(i * K, K));
        var cRef = NaiveMatMul(dequant, b, M, K, N);
        for (int i = 0; i < cRef.Length; i++) Assert.Equal(cRef[i], c[i], 3);
    }

    [Fact]
    public void Int3MatMul_ReferenceParity()
    {
        const int M = 2, K = 16, N = 3;
        var rng = new Random(2);
        var wFloat = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < wFloat.Length; i++) wFloat[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var packed = new PackedInt3Block[M * K / 8];
        var rowScales = new float[M];
        for (int i = 0; i < M; i++)
        {
            var rowDst = packed.AsSpan(i * (K / 8), K / 8);
            var s = QuantizationHelpers.QuantizeInt3(wFloat.AsSpan(i * K, K), rowDst, groupSize: K);
            rowScales[i] = s.Scales[0];
        }
        var scale = new QuantizationScale(rowScales, groupSize: K);

        var c = new float[M * N];
        PackedMatMul.Int3WeightMatMul(packed, scale, b, c, M, K, N);

        var dequant = new float[M * K];
        for (int i = 0; i < M; i++)
            QuantizationHelpers.DequantizeInt3(
                packed.AsSpan(i * (K / 8), K / 8),
                new QuantizationScale(new[] { rowScales[i] }, K),
                dequant.AsSpan(i * K, K));
        var cRef = NaiveMatMul(dequant, b, M, K, N);
        for (int i = 0; i < cRef.Length; i++) Assert.Equal(cRef[i], c[i], 3);
    }

    [Fact]
    public void NF4MatMul_ReferenceParity()
    {
        const int M = 2, K = 8, N = 3;
        var rng = new Random(3);
        var wFloat = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < wFloat.Length; i++) wFloat[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var packed = new PackedInt4[M * K / 2];
        var rowScales = new float[M];
        for (int i = 0; i < M; i++)
        {
            var rowDst = packed.AsSpan(i * (K / 2), K / 2);
            var s = QuantizationHelpers.QuantizeNF4(wFloat.AsSpan(i * K, K), rowDst, groupSize: K);
            rowScales[i] = s.Scales[0];
        }
        var scale = new QuantizationScale(rowScales, groupSize: K);

        var c = new float[M * N];
        PackedMatMul.NF4WeightMatMul(packed, scale, b, c, M, K, N);

        var dequant = new float[M * K];
        for (int i = 0; i < M; i++)
            QuantizationHelpers.DequantizeNF4(
                packed.AsSpan(i * (K / 2), K / 2),
                new QuantizationScale(new[] { rowScales[i] }, K),
                dequant.AsSpan(i * K, K));
        var cRef = NaiveMatMul(dequant, b, M, K, N);
        for (int i = 0; i < cRef.Length; i++) Assert.Equal(cRef[i], c[i], 3);
    }

    [Fact]
    public void Fp4MatMul_ReferenceParity()
    {
        const int M = 2, K = 8, N = 3;
        var rng = new Random(4);
        var wFloat = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < wFloat.Length; i++) wFloat[i] = (float)(rng.NextDouble() * 4 - 2);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var packed = new PackedInt4[M * K / 2];
        var rowScales = new float[M];
        for (int i = 0; i < M; i++)
        {
            var rowDst = packed.AsSpan(i * (K / 2), K / 2);
            var s = QuantizationHelpers.QuantizeFp4(wFloat.AsSpan(i * K, K), rowDst, groupSize: K);
            rowScales[i] = s.Scales[0];
        }
        var scale = new QuantizationScale(rowScales, groupSize: K);

        var c = new float[M * N];
        PackedMatMul.Fp4WeightMatMul(packed, scale, b, c, M, K, N);

        var dequant = new float[M * K];
        for (int i = 0; i < M; i++)
            QuantizationHelpers.DequantizeFp4(
                packed.AsSpan(i * (K / 2), K / 2),
                new QuantizationScale(new[] { rowScales[i] }, K),
                dequant.AsSpan(i * K, K));
        var cRef = NaiveMatMul(dequant, b, M, K, N);
        for (int i = 0; i < cRef.Length; i++) Assert.Equal(cRef[i], c[i], 3);
    }

    [Fact]
    public void XnorPopCountBlock_MatchesScalar_OnWideInput()
    {
        // Exercise both the AVX2 wide path (when available) and the
        // scalar tail. 96 bytes forces three 32-byte vectors.
        const int N = 96;
        var rng = new Random(5);
        var a = new byte[N];
        var b = new byte[N];
        for (int i = 0; i < N; i++)
        {
            a[i] = (byte)rng.Next(256);
            b[i] = (byte)rng.Next(256);
        }
        int got = PackedMatMul.XnorPopCountBlock(a, b, N);

        int expected = 0;
        for (int i = 0; i < N; i++)
        {
            byte xn = (byte)~(a[i] ^ b[i]);
            for (int k = 0; k < 8; k++) if (((xn >> k) & 1) != 0) expected++;
        }
        Assert.Equal(expected, got);
    }

    [Fact]
    public void XnorPopCountBlock_TailOnly_LessThan32Bytes()
    {
        var a = new byte[] { 0b11110000, 0b10101010, 0xFF };
        var b = new byte[] { 0b00001111, 0b01010101, 0x00 };
        int got = PackedMatMul.XnorPopCountBlock(a, b, 3);
        // All bits disagree in byte 0 (xnor=0), all bits disagree in byte 1
        // (xnor=0), byte 2: 0xFF xnor 0x00 = 0x00 (all disagree) → total = 0.
        Assert.Equal(0, got);
    }

    private static float[] NaiveMatMul(float[] a, float[] b, int m, int k, int n)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                c[i * n + j] = acc;
            }
        return c;
    }
}
