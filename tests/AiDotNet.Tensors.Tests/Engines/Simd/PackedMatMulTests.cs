using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Tests for issue #207 B1 — sub-byte matmul kernels.
///
/// <para><b>Int4 weight-only</b>: common LLM inference pattern (AWQ / GPTQ /
/// GGUF Q4_0). Quantized weights × float activations; accuracy bound by
/// the per-group scale. Verify against float reference within the quant
/// error band.</para>
///
/// <para><b>Int1 XNOR</b>: BitNet pattern — both sides quantized to sign,
/// multiplied via popcount of XNOR. For random signs the result magnitude
/// should approach (scale_a × scale_b × √K) expected by random walk
/// theory, and the exact-agreement case (A == B) should recover K × scale_a × scale_b.</para>
/// </summary>
public class PackedMatMulTests
{
    // ──────────── Int4 weight-only ────────────

    [Fact]
    public void Int4MatMul_ReferenceFloatParity_WithinQuantError()
    {
        const int M = 4, K = 32, N = 3;
        var rng = new Random(0xB177);
        var wFloat = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < wFloat.Length; i++) wFloat[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Quantize weights.
        var packed = new PackedInt4[M * K / 2];
        var scales = new float[M]; // one scale per row (group=K)
        // Build a flat scale array row-by-row then wrap as QuantizationScale.
        var rowPacked = new PackedInt4[K / 2];
        var rowScales = new float[M];
        for (int i = 0; i < M; i++)
        {
            ReadOnlySpan<float> row = wFloat.AsSpan(i * K, K);
            var dst = packed.AsSpan(i * (K / 2), K / 2);
            var s = QuantizationHelpers.QuantizeInt4(row, dst, groupSize: K);
            rowScales[i] = s.Scales[0];
        }
        var scale = new QuantizationScale(rowScales, groupSize: K);

        // Run packed matmul.
        var cPacked = new float[M * N];
        PackedMatMul.Int4WeightMatMul(packed, scale, b, cPacked, M, K, N);

        // Reference — dequantize weights, do float matmul.
        var wDequant = new float[M * K];
        for (int i = 0; i < M; i++)
        {
            QuantizationHelpers.DequantizeInt4(
                packed.AsSpan(i * (K / 2), K / 2),
                new QuantizationScale(new[] { rowScales[i] }, groupSize: K),
                wDequant.AsSpan(i * K, K));
        }

        var cRef = new float[M * N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                for (int p = 0; p < K; p++) acc += wDequant[i * K + p] * b[p * N + j];
                cRef[i * N + j] = acc;
            }
        }

        for (int i = 0; i < cRef.Length; i++)
            Assert.Equal(cRef[i], cPacked[i], 3);
    }

    [Fact]
    public void Int4MatMul_OddK_Throws()
    {
        var packed = new PackedInt4[1];
        var b = new float[3];
        var c = new float[1];
        var scale = new QuantizationScale(new[] { 1f }, 1);
        Assert.Throws<ArgumentException>(() =>
            PackedMatMul.Int4WeightMatMul(packed, scale, b, c, 1, 3, 1));
    }

    [Fact]
    public void Int4MatMul_PerTensorScale_Throws()
    {
        // Per-tensor (groupSize=0) disallowed on the matmul path — it's
        // too lossy to be useful; callers must quantize per-group.
        var packed = new PackedInt4[1];
        var b = new float[2];
        var c = new float[1];
        var scale = new QuantizationScale(new[] { 1f }, groupSize: 0);
        Assert.Throws<ArgumentException>(() =>
            PackedMatMul.Int4WeightMatMul(packed, scale, b, c, 1, 2, 1));
    }

    // ──────────── Int1 XNOR ────────────

    [Fact]
    public void Int1MatMul_ExactAgreement_RecoversScaleProductTimesK()
    {
        // A == B (same sign matrix). Each row-col dot is exactly K in ±1
        // algebra. Result = K × aScale × bScale.
        const int M = 2, K = 16, N = 2;
        var signs = new float[M * K];
        var rng = new Random(17);
        for (int i = 0; i < signs.Length; i++) signs[i] = rng.Next(2) == 0 ? -1f : 1f;

        var aPacked = new PackedInt1[M * K / 8];
        var aScale = QuantizationHelpers.QuantizeInt1(signs, aPacked);

        // B = same sign pattern but transposed-packed [N × K/8].
        // Simplest equality: make B's columns equal to A's rows.
        // For M=N=2, K=16, arrange B so that column j == row j of A.
        var bLayout = new float[K * N];
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                bLayout[k * N + j] = signs[j * K + k]; // col j = row j of A

        var bPacked = new PackedInt1[N * K / 8];
        var bScale = PackedMatMul.PackBTransposed(bLayout, K, N, bPacked);

        var c = new float[M * N];
        PackedMatMul.Int1MatMulXnor(aPacked, aScale, bPacked, bScale, c, M, K, N);

        // Per-tensor aScale → single element; per-column bScale → length N.
        // Diagonal (i == j): rows of A == cols of B → dot = K → c[i,i] = K*sA*sB.
        float sA = aScale.Scales[0];
        Assert.Equal(K * sA * bScale.Scales[0], c[0 * N + 0], 3);
        Assert.Equal(K * sA * bScale.Scales[1], c[1 * N + 1], 3);
    }

    [Fact]
    public void Int1MatMul_AntiAgreement_RecoversNegativeKTimesScaleProduct()
    {
        // A column == -(B row) → every sign disagrees → dot = -K.
        const int M = 1, K = 16, N = 1;
        var signs = new float[K];
        for (int i = 0; i < K; i++) signs[i] = 1f;
        var aPacked = new PackedInt1[K / 8];
        var aScale = QuantizationHelpers.QuantizeInt1(signs, aPacked);

        var bLayout = new float[K];
        for (int i = 0; i < K; i++) bLayout[i] = -1f;
        var bPacked = new PackedInt1[K / 8];
        var bScale = PackedMatMul.PackBTransposed(bLayout, K, N, bPacked);

        var c = new float[M * N];
        PackedMatMul.Int1MatMulXnor(aPacked, aScale, bPacked, bScale, c, M, K, N);
        Assert.Equal(-K * aScale.Scales[0] * bScale.Scales[0], c[0], 3);
    }

    [Fact]
    public void Int1MatMul_KNotMultipleOf8_Throws()
    {
        var a = new PackedInt1[1];
        var b = new PackedInt1[1];
        var c = new float[1];
        var s = new QuantizationScale(new[] { 1f }, 0);
        Assert.Throws<ArgumentException>(() =>
            PackedMatMul.Int1MatMulXnor(a, s, b, s, c, 1, 7, 1));
    }

    [Fact]
    public void PackBTransposed_RoundTripsSignsAndProducesPerColumnScale()
    {
        const int K = 8, N = 3;
        var b = new float[K * N];
        var rng = new Random(4);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var packed = new PackedInt1[N * K / 8];
        var scale = PackedMatMul.PackBTransposed(b, K, N, packed);

        Assert.Equal(N, scale.Scales.Length);
        for (int j = 0; j < N; j++)
        {
            // Expected scale is absmean of column j.
            float absSum = 0f;
            for (int k = 0; k < K; k++) absSum += Math.Abs(b[k * N + j]);
            Assert.Equal(absSum / K, scale.Scales[j], 3);
        }

        // Signs round-trip.
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                int byteIdx = j * (K / 8) + k / 8;
                int lane = k % 8;
                sbyte expected = b[k * N + j] >= 0 ? (sbyte)1 : (sbyte)-1;
                Assert.Equal(expected, packed[byteIdx].GetLane(lane));
            }
        }
    }
}
