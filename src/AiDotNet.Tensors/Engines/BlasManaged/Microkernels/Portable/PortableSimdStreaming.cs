using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Streaming FP64 microkernel built on the BCL's <see cref="Vector{T}"/> rather than
/// <c>System.Runtime.Intrinsics</c>. Fills the gap between <see cref="NeonStreaming"/> and
/// <see cref="ScalarStreaming"/> in <see cref="StreamingStrategy"/>'s dispatch chain.
///
/// <para>
/// WHY THIS EXISTS. <c>System.Runtime.Intrinsics</c> does not exist before net6, so on net471
/// <see cref="Avx2Streaming"/> and <see cref="Avx512Streaming"/> compile to stubs whose
/// <c>IsSupported</c> is a constant <c>false</c> and which delegate straight to
/// <see cref="ScalarStreaming"/>. Every GEMM on net471 therefore ran a scalar triple loop.
/// Measured on AiDotNet's <c>ShiftNetTests.MoreData_ShouldNotDegrade</c> (250 training
/// iterations): 33.3 s on net10.0 versus a &gt;120 s timeout on net471, with a PerfView
/// /ThreadTime trace attributing 29.77% of leaf CPU to <see cref="ScalarStreaming.RunFp64"/>
/// and 12.00% to <see cref="ScalarFp64_4x4"/> — about 45% of the process in scalar fp64 GEMM.
/// <see cref="Vector{T}"/> IS available and JIT-vectorized on net471 (via System.Numerics.Vectors),
/// so the fallback tier did not have to be scalar.
/// </para>
///
/// <para>
/// NOT built on <c>Compatibility/SimdCompat.cs</c>. That file's net471 <c>Vector128</c>/
/// <c>Vector256</c> shims are backed by <c>T[] _elements</c> and loop element-wise, allocating a
/// fresh array in every <c>Create</c>. They exist so intrinsics-shaped code COMPILES on net471,
/// not so it runs fast; using them in a GEMM inner loop would be slower than the scalar kernel
/// and would allocate per call.
/// </para>
/// </summary>
/// <remarks>
/// <para><b>Bit-exactness.</b> This kernel vectorizes across <c>j</c> (the n axis), never across
/// <c>k</c>. For a fixed (i, j) the accumulation still walks <c>kk</c> in ascending order, one
/// rounded multiply and one rounded add per step — exactly <see cref="ScalarStreaming.RunFp64"/>'s
/// sequence — so results are bit-identical and the different loop nesting (i→kk→j here versus
/// i→j→kk there) does not change any element's summation order. Vectorizing across <c>k</c>
/// instead would need a horizontal sum of partial lanes, which reassociates the additions and
/// would break the bit-exact contract <see cref="ScalarStreaming"/> documents against
/// <see cref="ScalarFp32_4x4"/> for the tiny-shape bypass guarantee.
/// </para>
/// <para><b>Why only <c>transB == false</c>.</b> j-vectorization needs both operands contiguous
/// along j. With <c>transB == false</c>, <c>b[kk * ldb + j]</c> and <c>c[i * ldc + j]</c> both are,
/// and <c>aval</c> depends only on (i, kk) so it broadcasts — which is why <c>transA</c> needs no
/// special case. With <c>transB == true</c>, <c>b[j * ldb + kk]</c> strides by <c>ldb</c> in j and
/// would require a gather; that shape stays on the scalar kernel rather than being emulated
/// element-wise for no gain.
/// </para>
/// <para><b>Why FP64 only.</b> <see cref="ScalarStreaming.RunFp32"/> deliberately accumulates in
/// <c>double</c> to stay bit-exact with <see cref="ScalarFp32_4x4"/>. A <c>Vector{float}</c> path
/// would accumulate in float and break that contract, and the measured cost is fp64 anyway, so
/// FP32 is left on the scalar kernel.
/// </para>
/// </remarks>
internal static class PortableSimdStreaming
{
    internal static int Fp64ColumnTileWidth => Vector<double>.Count;

    /// <summary>
    /// True when the BCL reports real SIMD and a <see cref="Vector{T}"/> of double holds more than
    /// one lane. Without hardware acceleration <see cref="Vector{T}"/> is emulated in software and
    /// would be slower than the scalar kernel, so this tier declines and dispatch falls through.
    /// </summary>
    public static bool IsSupported => Vector.IsHardwareAccelerated && Vector<double>.Count > 1;

    /// <summary>
    /// Compute C += op(A) · op(B) without packing. C is read-modify-write.
    /// Bit-identical to <see cref="ScalarStreaming.RunFp64"/>; see the type remarks.
    /// </summary>
    public static unsafe void RunFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k)
    {
        if (m <= 0 || n <= 0 || k <= 0)
        {
            return;
        }

        int lanes = Fp64ColumnTileWidth;

        // transB strides B along j (gather); a sub-lane n has no vector work to do. Both are
        // handled by the scalar kernel so the numerics stay on one code path.
        if (transB || n < lanes)
        {
            ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            return;
        }

        int simdN = n - (n % lanes);

        fixed (double* aBase = &MemoryMarshal.GetReference(a))
        fixed (double* bBase = &MemoryMarshal.GetReference(b))
        fixed (double* cBase = &MemoryMarshal.GetReference(c))
        {
            for (int i = 0; i < m; i++)
            {
                double* cRow = cBase + ((long)i * ldc);
                double* aRow = aBase + ((long)i * lda);   // only read when !transA

                for (int kk = 0; kk < k; kk++)
                {
                    // Depends on (i, kk) only, so it broadcasts across the j lanes.
                    double aval = transA ? aBase[((long)kk * lda) + i] : aRow[kk];

                    // NOTE: no `aval == 0` early-out. Skipping would suppress the 0 * Infinity
                    // => NaN propagation and the signed-zero result that the scalar kernel
                    // produces, so the zero row is multiplied like any other.
                    var va = new Vector<double>(aval);
                    double* bRow = bBase + ((long)kk * ldb);

                    int j = 0;
                    for (; j < simdN; j += lanes)
                    {
                        var vb = Unsafe.ReadUnaligned<Vector<double>>((void*)(bRow + j));
                        var vc = Unsafe.ReadUnaligned<Vector<double>>((void*)(cRow + j));
                        Unsafe.WriteUnaligned((void*)(cRow + j), vc + (va * vb));
                    }

                    // j tail, same expression as the vector body so rounding matches.
                    for (; j < n; j++)
                    {
                        cRow[j] += aval * bRow[j];
                    }
                }
            }
        }
    }
}
