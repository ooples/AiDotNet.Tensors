using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Locks <see cref="PortableSimdStreaming"/> (the BCL <c>Vector{T}</c> FP64 streaming tier that
/// keeps net471 off the scalar GEMM path) against <see cref="ScalarStreaming"/>.
///
/// <para>
/// These assert BIT equality via <see cref="BitConverter.DoubleToInt64Bits"/>, not a tolerance.
/// The kernel's whole design constraint is that vectorizing across <c>j</c> leaves each element's
/// accumulation over <c>kk</c> in the original order, so "close enough" would silently permit the
/// k-axis reassociation the design specifically avoids — and reassociation is what would break
/// ScalarStreaming's documented bit-exact contract with ScalarFp32_4x4. Comparing raw bits also
/// distinguishes -0.0 from 0.0, which a value comparison would not.
/// </para>
///
/// <para>
/// Deliberately NOT gated on a TFM. net471 is the reason this kernel exists, so the contract has
/// to be verified there; on net10.0 the same code guards the non-intrinsic fallback tier.
/// </para>
/// </summary>
public class PortableSimdStreamingTests
{
    private static double[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = (rng.NextDouble() * 2.0) - 1.0;
        return a;
    }

    /// <summary>
    /// Runs both kernels over identical inputs and asserts every C element matches bit-for-bit.
    /// Leading dimensions are passed deliberately larger than the logical extents where the shape
    /// allows, because real GEMM callers hand these kernels row strides into a bigger buffer and a
    /// kernel that assumed ld == extent would still pass a tightly-packed test.
    /// </summary>
    private static void AssertBitExact(int m, int n, int k, bool transA, bool transB, int seed, int ldPad = 0)
    {
        int lda = (transA ? m : k) + ldPad;
        int ldb = (transB ? k : n) + ldPad;
        int ldc = n + ldPad;

        // Size to the highest index each kernel can touch: (rows-1)*ld + extent.
        int aLen = (((transA ? k : m) - 1) * lda) + (transA ? m : k);
        int bLen = (((transB ? n : k) - 1) * ldb) + (transB ? k : n);
        int cLen = ((m - 1) * ldc) + n;

        var a = Random(aLen, seed);
        var b = Random(bLen, seed + 1);
        var cSeed = Random(cLen, seed + 2);

        // C is read-modify-write, so both runs must start from the same non-zero C.
        var cScalar = (double[])cSeed.Clone();
        var cVector = (double[])cSeed.Clone();

        ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, cScalar, ldc, m, n, k);
        PortableSimdStreaming.RunFp64(a, lda, transA, b, ldb, transB, cVector, ldc, m, n, k);

        for (int i = 0; i < cLen; i++)
        {
            long want = BitConverter.DoubleToInt64Bits(cScalar[i]);
            long got = BitConverter.DoubleToInt64Bits(cVector[i]);
            Assert.True(want == got,
                $"C[{i}] differs: scalar={cScalar[i]:R} (0x{want:X16}) vector={cVector[i]:R} (0x{got:X16}) " +
                $"for m={m} n={n} k={k} transA={transA} transB={transB} ldPad={ldPad}");
        }
    }

    [Theory]
    // n straddling the vector width in both directions, so the tail loop and the pure-tail
    // (n < lanes) bail are both exercised whether Vector<double>.Count is 2, 4 or 8.
    [InlineData(1, 1, 1)]
    [InlineData(1, 2, 3)]
    [InlineData(3, 3, 3)]
    [InlineData(4, 4, 4)]
    [InlineData(2, 5, 7)]
    [InlineData(5, 8, 2)]
    [InlineData(8, 9, 5)]
    [InlineData(7, 16, 3)]
    [InlineData(6, 17, 11)]
    [InlineData(16, 31, 8)]
    [InlineData(9, 64, 13)]
    public void MatchesScalarBitExactly_ForBothTransA(int m, int n, int k)
    {
        AssertBitExact(m, n, k, transA: false, transB: false, seed: (m * 131) + (n * 17) + k);
        AssertBitExact(m, n, k, transA: true, transB: false, seed: (m * 977) + (n * 31) + k);
    }

    [Theory]
    [InlineData(4, 8, 4)]
    [InlineData(6, 17, 11)]
    [InlineData(3, 5, 9)]
    public void MatchesScalarBitExactly_WithPaddedLeadingDimensions(int m, int n, int k)
    {
        // ldPad > 0 means the row stride exceeds the logical width, i.e. the kernel is writing
        // into a window of a larger matrix. Getting this wrong corrupts neighbouring columns,
        // which the padded region catches because it must come back untouched.
        AssertBitExact(m, n, k, transA: false, transB: false, seed: 4242, ldPad: 3);
        AssertBitExact(m, n, k, transA: true, transB: false, seed: 4243, ldPad: 5);
    }

    [Theory]
    [InlineData(4, 8, 4)]
    [InlineData(6, 17, 11)]
    public void TransposedB_DelegatesToScalarAndStaysBitExact(int m, int n, int k)
    {
        // transB strides B along j so it cannot be j-vectorized; the kernel must hand it to the
        // scalar path rather than emulate a gather. Either way the numbers must be identical.
        AssertBitExact(m, n, k, transA: false, transB: true, seed: 909);
        AssertBitExact(m, n, k, transA: true, transB: true, seed: 910);
    }

    [Fact]
    public void ZeroTimesInfinityStillPropagatesNaN()
    {
        // The kernel documents that it must NOT skip aval == 0. An `if (aval == 0) continue`
        // early-out is a tempting optimisation that changes results here: 0 * Infinity is NaN,
        // and skipping would leave C's original value instead.
        // WIDE ENOUGH TO REACH THE VECTOR BODY. RunFp64 hands off to ScalarStreaming when
        // n < Vector<double>.Count, so a hard-coded n = 4 tested only the scalar path on any
        // runtime with 8 lanes -- AVX-512 -- and this test's whole subject is what the VECTOR
        // loop does with 0 * Infinity. It would have passed there while asserting nothing about
        // the code it names. Sized from the runtime instead, so it reaches the vector body at
        // 2, 4 or 8 lanes.
        const int m = 1, k = 2;
        int n = Math.Max(4, System.Numerics.Vector<double>.Count);
        var a = new double[] { 0.0, 1.0 };
        var b = new double[n * k];
        for (int j = 0; j < n; j++)
        {
            b[j] = double.PositiveInfinity;   // k-row 0, multiplied by aval 0.0
            b[n + j] = 1.0;                   // k-row 1, multiplied by aval 1.0
        }

        // The property this sizing exists for, asserted rather than assumed: if a future
        // edit hard-codes n again, this fails here instead of silently testing the scalar
        // path on a wide machine and reporting green.
        Assert.True(n >= System.Numerics.Vector<double>.Count,
            $"n={n} must reach the vector body (lanes={System.Numerics.Vector<double>.Count})");

        var cScalar = new double[n];
        var cVector = new double[n];
        ScalarStreaming.RunFp64(a, k, false, b, n, false, cScalar, n, m, n, k);
        PortableSimdStreaming.RunFp64(a, k, false, b, n, false, cVector, n, m, n, k);

        for (int j = 0; j < n; j++)
        {
            Assert.True(double.IsNaN(cScalar[j]), $"reference should be NaN at {j}, got {cScalar[j]:R}");
            Assert.Equal(BitConverter.DoubleToInt64Bits(cScalar[j]), BitConverter.DoubleToInt64Bits(cVector[j]));
        }
    }

    [Fact]
    public void NegativeZeroSignIsPreserved()
    {
        // -0.0 + -0.0 stays -0.0, while any accidental re-zeroing of the accumulator would give
        // +0.0. Bit comparison is what makes this observable.
        // Sized from the runtime for the same reason as the test above: at n = 4 on an 8-lane
        // runtime this never entered the vector body, so the signed-zero claim was only ever
        // checked against the scalar path it is supposed to be compared WITH.
        const int m = 1, k = 1;
        int n = Math.Max(4, System.Numerics.Vector<double>.Count);
        var a = new double[] { -1.0 };
        var b = new double[n];
        var cScalar = new double[n];
        var cVector = new double[n];
        for (int j = 0; j < n; j++)
        {
            b[j] = 0.0;
            cScalar[j] = -0.0;
            cVector[j] = -0.0;
        }

        // The property this sizing exists for, asserted rather than assumed: if a future
        // edit hard-codes n again, this fails here instead of silently testing the scalar
        // path on a wide machine and reporting green.
        Assert.True(n >= System.Numerics.Vector<double>.Count,
            $"n={n} must reach the vector body (lanes={System.Numerics.Vector<double>.Count})");

        ScalarStreaming.RunFp64(a, k, false, b, n, false, cScalar, n, m, n, k);
        PortableSimdStreaming.RunFp64(a, k, false, b, n, false, cVector, n, m, n, k);

        for (int j = 0; j < n; j++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(cScalar[j]), BitConverter.DoubleToInt64Bits(cVector[j]));
    }

    [Fact]
    public void DegenerateExtentsAreNoOps()
    {
        var c = new double[] { 1.0, 2.0, 3.0, 4.0 };
        var expected = (double[])c.Clone();
        var a = new double[] { 5.0 };
        var b = new double[] { 6.0 };

        PortableSimdStreaming.RunFp64(a, 1, false, b, 1, false, c, 4, 0, 4, 1);
        PortableSimdStreaming.RunFp64(a, 1, false, b, 1, false, c, 4, 1, 0, 1);
        PortableSimdStreaming.RunFp64(a, 1, false, b, 1, false, c, 4, 1, 4, 0);

        Assert.Equal(expected, c);
    }

    [Fact]
    public void IsSupportedAgreesWithHardwareAcceleration()
    {
        // The tier must decline when Vector<T> is software-emulated, because emulated lanes are
        // slower than the scalar kernel it would displace.
        bool expected = System.Numerics.Vector.IsHardwareAccelerated
                        && System.Numerics.Vector<double>.Count > 1;
        Assert.Equal(expected, PortableSimdStreaming.IsSupported);
    }
}
