using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// An operand too small for its own strides must be rejected, not walked past the end.
/// </summary>
/// <remarks>
/// <para>
/// The GEMM kernels take <c>fixed</c> pointers into the operand spans and index them with lda / ldb
/// / n. Nothing validated the spans were long enough for those strides, so getting one wrong did not
/// raise - it read and wrote past the end of the buffer. In CI that surfaced as
/// <c>System.AccessViolationException: Attempted to read or write protected memory</c> thrown from
/// <c>SimdGemm.SgemmDirect</c> during a backward pass, which cannot be caught: the process dies,
/// the shard reports "Test host process crashed : Fatal error", and no shape information survives.
/// </para>
/// <para>
/// These cases pin the boundary rather than a single bad call, because the interesting property is
/// that an undersized operand is refused AT THE ENTRY POINT, while an exactly-sized one is still
/// accepted. The exactly-sized cases matter as much as the failing ones: the last row of a matrix
/// needs only its own columns, so the minimum is (rows-1)*stride + cols, and a guard written as
/// rows*stride would reject legitimate callers.
/// </para>
/// </remarks>
public class SimdGemmBoundsPreconditionTests
{
    private const int M = 4;
    private const int K = 3;
    private const int N = 5;

    private static float[] Filled(int length)
    {
        var buffer = new float[length];
        for (int i = 0; i < length; i++) buffer[i] = i * 0.125f + 1f;
        return buffer;
    }

    [Fact]
    public void ExactlySizedOperands_AreAccepted()
    {
        // (rows-1)*stride + cols for A and B; m*n for C.
        var a = Filled((M - 1) * K + K);
        var b = Filled((K - 1) * N + N);
        var c = new float[M * N];

        SimdGemm.Sgemm(a, K, false, b, N, false, c, M, K, N);

        // A real product, not an untouched buffer - proves the guard did not short-circuit the call.
        Assert.Contains(c, value => value != 0f);
    }

    [Fact]
    public void UndersizedA_ThrowsInsteadOfReadingPastTheEnd()
    {
        var a = Filled(M * K - 1);
        var b = Filled(K * N);
        var c = new float[M * N];

        var error = Assert.Throws<ArgumentException>(
            () => SimdGemm.Sgemm(a, K, false, b, N, false, c, M, K, N));
        Assert.Contains("operand A", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void UndersizedB_ThrowsInsteadOfReadingPastTheEnd()
    {
        var a = Filled(M * K);
        var b = Filled(K * N - 1);
        var c = new float[M * N];

        var error = Assert.Throws<ArgumentException>(
            () => SimdGemm.Sgemm(a, K, false, b, N, false, c, M, K, N));
        Assert.Contains("operand B", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void UndersizedC_ThrowsInsteadOfWritingPastTheEnd()
    {
        var a = Filled(M * K);
        var b = Filled(K * N);
        var c = new float[M * N - 1];

        var error = Assert.Throws<ArgumentException>(
            () => SimdGemm.Sgemm(a, K, false, b, N, false, c, M, K, N));
        Assert.Contains("output C", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StrideNarrowerThanTheColumnCount_Throws()
    {
        // lda below k cannot describe a [m,k] matrix at all; walking it would interleave rows.
        var a = Filled(M * K);
        var b = Filled(K * N);
        var c = new float[M * N];

        Assert.Throws<ArgumentException>(
            () => SimdGemm.Sgemm(a, K - 1, false, b, N, false, c, M, K, N));
    }

    [Fact]
    public void TransposedOperands_SizeAgainstTheTransposedShape()
    {
        // transA: A is [k,m] with row stride lda, so the minimum is (k-1)*lda + m, NOT (m-1)*lda + k.
        // Sizing a transposed operand against the untransposed shape is exactly the mistake that
        // walks off the end when m and k differ.
        var aT = Filled((K - 1) * M + M);
        var b = Filled((K - 1) * N + N);
        var c = new float[M * N];

        SimdGemm.Sgemm(aT, M, true, b, N, false, c, M, K, N);

        var tooSmall = Filled((K - 1) * M + M - 1);
        Assert.Throws<ArgumentException>(
            () => SimdGemm.Sgemm(tooSmall, M, true, b, N, false, c, M, K, N));
    }

    [Fact]
    public void DegenerateShapes_StillReturnWithoutValidating()
    {
        // k = 0 has no work to do and callers rely on it returning quietly; the guard must not
        // turn a documented no-op into an exception.
        var c = new float[M * N];
        SimdGemm.Sgemm(Array.Empty<float>(), 0, false, Array.Empty<float>(), N, false, c, M, 0, N);
    }
}
