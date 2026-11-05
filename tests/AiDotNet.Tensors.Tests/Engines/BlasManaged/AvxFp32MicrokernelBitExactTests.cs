using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Asserts <see cref="AvxFp32_8x8"/> and <see cref="AvxStreaming"/> compute every element as the FP32
/// <c>c = c + a * b</c> chain over k in order, bit for bit: the arithmetic of the scalar N-tail they share a GEMM with.
/// </summary>
/// <remarks>
/// <para>
/// Bit-identity, not a tolerance: BlasMode.Deterministic promises the same bits whichever kernel a row lands in, and
/// the parallel split moves rows between the 8x8 tile, the scalar tail and the streaming kernel. A fused multiply-add,
/// or a reordered k loop, would agree to 1e-7 and still break that promise.
/// </para>
/// <para>
/// These kernels only run where AVX2+FMA is unavailable, so on an AVX2 host every test but the gate check returns at
/// once. build.yml runs this namespace a second time with <c>DOTNET_EnableAVX2=0</c>, which is where they execute.
/// </para>
/// </remarks>
public class AvxFp32MicrokernelBitExactTests
{
    private const int Mr = AvxFp32_8x8.Mr;
    private const int Nr = AvxFp32_8x8.Nr;

    private static float[] Filled(int count, int seed)
    {
        var rng = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((rng.NextDouble() * 2.0) - 1.0);
        return values;
    }

    // C[i, j] = C[i, j] + A[i, p] * B[p, j] in FP32, p ascending: product rounded, then sum rounded.
    private static void Reference(Func<int, int, float> a, Func<int, int, float> b, float[] c, int ldc,
        int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = c[(i * ldc) + j];
                for (int p = 0; p < k; p++)
                {
                    float product = a(i, p) * b(p, j);
                    sum += product;
                }
                c[(i * ldc) + j] = sum;
            }
        }
    }

    // BitConverter.SingleToInt32Bits is not in net462/net471.
    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    private static void AssertBitIdentical(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            int e = Bits(expected[i]);
            int a = Bits(actual[i]);
            if (e != a)
            {
                Assert.Fail($"{what}: element {i} differs at the BIT level: reference {expected[i]:R} (0x{e:X8}) vs "
                    + $"kernel {actual[i]:R} (0x{a:X8}).");
            }
        }
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(7, 8)]
    [InlineData(129, 12)]     // ldc > Nr, so C rows are not adjacent
    [InlineData(512, 16)]     // a full Kc panel
    public void RunMatchesTheFp32ChainBitForBit(int kc, int ldc)
    {
        if (!AvxFp32_8x8.IsSupported) return;

        var packedA = Filled(kc * Mr, seed: 31);
        var packedB = Filled(kc * Nr, seed: 32);
        var initial = Filled(Mr * ldc, seed: 33);
        var expected = (float[])initial.Clone();
        var actual = (float[])initial.Clone();

        Reference((i, p) => packedA[(p * Mr) + i], (p, j) => packedB[(p * Nr) + j], expected, ldc, Mr, Nr, kc);
        AvxFp32_8x8.Run(packedA, packedB, actual, ldc, kc);

        AssertBitIdentical(expected, actual, $"Run(kc={kc}, ldc={ldc})");
    }

    [Theory]
    [InlineData(5, 8, 8)]
    [InlineData(33, 11, 9)]   // ldb > Nr and ldc > Nr
    [InlineData(300, 40, 16)]
    public void RunStridedBMatchesTheFp32ChainBitForBit(int kc, int ldb, int ldc)
    {
        if (!AvxFp32_8x8.IsSupported) return;

        var packedA = Filled(kc * Mr, seed: 41);
        var b = Filled(kc * ldb, seed: 42);
        var initial = Filled(Mr * ldc, seed: 43);
        var expected = (float[])initial.Clone();
        var actual = (float[])initial.Clone();

        Reference((i, p) => packedA[(p * Mr) + i], (p, j) => b[(p * ldb) + j], expected, ldc, Mr, Nr, kc);
        AvxFp32_8x8.RunStridedB(packedA, b, ldb, actual, ldc, kc);

        AssertBitIdentical(expected, actual, $"RunStridedB(kc={kc}, ldb={ldb}, ldc={ldc})");
    }

    [Fact]
    public void SpecialValuesPropagateLikeTheChain()
    {
        // 0 * Infinity is NaN and -0.0 stays signed: an `a == 0` early-out would pass random data and diverge here.
        if (!AvxFp32_8x8.IsSupported) return;

        const int kc = 4;
        var packedA = new float[kc * Mr];
        var packedB = new float[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = 1f;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = 1f;
        packedA[0] = 0f;
        packedB[1] = float.PositiveInfinity;
        packedA[Mr + 1] = -0f;
        packedB[Nr + 2] = float.NaN;
        packedA[(2 * Mr) + 2] = float.NegativeInfinity;
        var initial = new float[Mr * Nr];
        for (int i = 0; i < initial.Length; i++) initial[i] = i % 2 == 0 ? 0f : -0f;
        var expected = (float[])initial.Clone();
        var actual = (float[])initial.Clone();

        Reference((i, p) => packedA[(p * Mr) + i], (p, j) => packedB[(p * Nr) + j], expected, Nr, Mr, Nr, kc);
        AvxFp32_8x8.Run(packedA, packedB, actual, Nr, kc);

        AssertBitIdentical(expected, actual, "special values");
    }

    [Theory]
    [InlineData(5, 19, 13, false, false)]   // two vector column steps and a three-column tail
    [InlineData(4, 16, 64, true, false)]
    [InlineData(3, 10, 7, false, true)]     // transposed B: element by element
    [InlineData(6, 9, 21, true, true)]
    public void StreamingMatchesTheFp32ChainBitForBit(int m, int n, int k, bool transA, bool transB)
    {
        if (!AvxStreaming.IsSupported) return;

        int lda = transA ? m : k, ldb = transB ? k : n, ldc = n + 3;
        var a = Filled(m * k, seed: 51);
        var b = Filled(k * n, seed: 52);
        var initial = Filled(m * ldc, seed: 53);
        var expected = (float[])initial.Clone();
        var actual = (float[])initial.Clone();

        Reference((i, p) => transA ? a[(p * lda) + i] : a[(i * lda) + p],
            (p, j) => transB ? b[(j * ldb) + p] : b[(p * ldb) + j], expected, ldc, m, n, k);
        AvxStreaming.RunFp32(a, lda, transA, b, ldb, transB, actual, ldc, m, n, k);

        AssertBitIdentical(expected, actual, $"RunFp32(m={m}, n={n}, k={k}, transA={transA}, transB={transB})");
    }

    [Fact]
    public void TheGateYieldsToTheAvx2Tier()
    {
#if NET5_0_OR_GREATER
        bool avx2Tier = System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported;
        Assert.Equal(System.Runtime.Intrinsics.X86.Avx.IsSupported && !avx2Tier, AvxFp32_8x8.IsSupported);
#else
        Assert.False(AvxFp32_8x8.IsSupported);
#endif
        Assert.Equal(AvxFp32_8x8.IsSupported, AvxStreaming.IsSupported);
    }
}
