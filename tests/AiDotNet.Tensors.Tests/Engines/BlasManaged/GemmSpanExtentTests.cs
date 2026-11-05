using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// <see cref="BlasManagedLib.Gemm{T}"/> rejects a span too short for its shape before any kernel runs. The SIMD
/// kernels address A, B and C through raw pointers with no checks of their own, so without this an undersized operand
/// read or wrote past its end on every SIMD host, silently instead of with the scalar kernel's IndexOutOfRange.
/// </summary>
public class GemmSpanExtentTests
{
    private const int M = 8, N = 16, K = 12;

    public static TheoryData<PackingMode> Modes() => new()
    {
        PackingMode.Auto, PackingMode.ForcePackBoth, PackingMode.ForcePackAOnly, PackingMode.ForceStreaming,
    };

    [Theory]
    [MemberData(nameof(Modes))]
    public void ShortA_Throws(PackingMode mode)
    {
        var ex = Assert.Throws<ArgumentException>(() => Run(mode, M * K - 1, K * N, M * N, transA: false, transB: false));
        Assert.Equal("a", ex.ParamName);
        // Transposed A is [K, M] with lda = M: the extent is (K - 1) * M + M.
        ex = Assert.Throws<ArgumentException>(() => Run(mode, K * M - 1, K * N, M * N, transA: true, transB: false));
        Assert.Equal("a", ex.ParamName);
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void ShortB_Throws(PackingMode mode)
    {
        var ex = Assert.Throws<ArgumentException>(() => Run(mode, M * K, K * N - 1, M * N, transA: false, transB: false));
        Assert.Equal("b", ex.ParamName);
        ex = Assert.Throws<ArgumentException>(() => Run(mode, M * K, N * K - 1, M * N, transA: false, transB: true));
        Assert.Equal("b", ex.ParamName);
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void ShortC_Throws(PackingMode mode)
    {
        var ex = Assert.Throws<ArgumentException>(() => Run(mode, M * K, K * N, M * N - 1, transA: false, transB: false));
        Assert.Equal("c", ex.ParamName);
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void ExactExtents_Run(PackingMode mode)
    {
        // The minimum lengths are accepted: a check one element too strict would reject every exact-size caller.
        Run(mode, M * K, K * N, M * N, transA: false, transB: false);
        Run(mode, K * M, N * K, M * N, transA: true, transB: true);
    }

    private static void Run(PackingMode mode, int aLength, int bLength, int cLength, bool transA, bool transB)
    {
        var a = new float[aLength];
        var b = new float[bLength];
        var c = new float[cLength];
        for (int i = 0; i < a.Length; i++) a[i] = 0.5f;
        for (int i = 0; i < b.Length; i++) b[i] = 0.25f;
        BlasManagedLib.Gemm<float>(
            a, transA ? M : K, transA,
            b, transB ? K : N, transB,
            c, N,
            M, N, K,
            new BlasOptions<float> { PackingMode = mode });
    }
}
