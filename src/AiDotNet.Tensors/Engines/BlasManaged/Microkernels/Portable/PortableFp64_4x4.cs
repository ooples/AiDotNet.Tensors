using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// BCL <see cref="Vector{T}"/> FP64 4x4 packed microkernel: a SIMD sibling of
/// <see cref="ScalarFp64_4x4"/>, bit-identical to it, for the packed strategies on net471.
/// </summary>
/// <remarks>
/// <para>
/// ADDED ALONGSIDE, NEVER REPLACING. <see cref="ScalarFp64_4x4"/> is the ground-truth reference that other
/// kernels assert against, so it keeps its scalar body untouched and this tier is dispatched to only when
/// it can be proven to produce identical bits. Rewriting the reference in SIMD would leave nothing to
/// compare against.
/// </para>
/// <para><b>Why this is bit-exact, and it is a proof rather than a tolerance.</b> Vectorization is along
/// the j (column) axis. One <see cref="Vector{T}"/> of double holds a single C row's four cells
/// <c>[c_i0, c_i1, c_i2, c_i3]</c>, and each k-step adds <c>broadcast(a_i) * [b_0, b_1, b_2, b_3]</c>. Lane
/// j therefore accumulates exactly <c>c_ij += a_i * b_j</c>, over exactly the same k values, in exactly the
/// same order as the scalar kernel's <c>c_ij += a_i * b_j</c>. There is no horizontal reduction anywhere,
/// so no addition is reassociated. Vectorizing along k instead would need a lane sum at the end, which
/// DOES reassociate and would break the contract.
/// </para>
/// <para>
/// The multiply and the add stay separate operations (<c>vc + (va * vb)</c>). .NET does not contract
/// multiply-add into FMA implicitly — that requires the explicit FusedMultiplyAdd intrinsics — so the
/// rounding sequence matches the scalar kernel's two-step arithmetic. Using an FMA intrinsic here would
/// round once instead of twice and silently break bit-exactness.
/// </para>
/// <para><b>Why exactly four lanes.</b> <c>Nr</c> is 4, so a four-lane vector covers one C row precisely.
/// Two lanes (SSE) would need two vectors per row and a second tail path; eight (AVX-512) would read past
/// the row into the next one. Rather than special-case both, this tier declines unless
/// <c>Vector&lt;double&gt;.Count == 4</c> and the caller falls through to the scalar kernel — which is
/// always correct, just slower.
/// </para>
/// <para>
/// It also declines when <see cref="Vector.IsHardwareAccelerated"/> is false, where
/// <see cref="Vector{T}"/> is a software emulation that would be slower than the scalar kernel it replaces.
/// </para>
/// </remarks>
internal static class PortableFp64_4x4
{
    /// <summary>Microkernel row-tile width, matching <see cref="ScalarFp64_4x4"/>.</summary>
    public const int Mr = 4;

    /// <summary>Microkernel column-tile width, matching <see cref="ScalarFp64_4x4"/>.</summary>
    public const int Nr = 4;

    /// <summary>
    /// True when the BCL reports real SIMD and a <see cref="Vector{T}"/> of double holds exactly
    /// <see cref="Nr"/> lanes. See the type remarks for why both conditions are required.
    /// </summary>
    public static bool IsSupported => Vector.IsHardwareAccelerated && Vector<double>.Count == Nr;

    /// <summary>
    /// Compute <c>C[0..Mr, 0..Nr] += packedA · packedB</c>. Bit-identical to
    /// <see cref="ScalarFp64_4x4.Run"/>.
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc x Mr] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc x Nr] row-major.</param>
    /// <param name="c">Output buffer; the kernel reads and writes the C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C, at least <see cref="Nr"/>.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static unsafe void Run(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc)
    {
        fixed (double* aBase = &MemoryMarshal.GetReference(packedA))
        fixed (double* bBase = &MemoryMarshal.GetReference(packedB))
        fixed (double* cBase = &MemoryMarshal.GetReference(c))
        {
            // One vector per C row: [c_i0 .. c_i3]. Held across the whole K-loop, exactly as the scalar
            // kernel holds its 16 cells in locals.
            var c0 = Unsafe.ReadUnaligned<Vector<double>>((void*)cBase);
            var c1 = Unsafe.ReadUnaligned<Vector<double>>((void*)(cBase + ldc));
            var c2 = Unsafe.ReadUnaligned<Vector<double>>((void*)(cBase + (2 * ldc)));
            var c3 = Unsafe.ReadUnaligned<Vector<double>>((void*)(cBase + (3 * ldc)));

            for (int k = 0; k < kc; k++)
            {
                var vb = Unsafe.ReadUnaligned<Vector<double>>((void*)(bBase + (k * Nr)));
                double* aK = aBase + (k * Mr);

                // No `a == 0` early-out: skipping would suppress the 0 * Infinity => NaN propagation and
                // the signed-zero results the scalar kernel produces, which are part of bit-exactness.
                c0 += new Vector<double>(aK[0]) * vb;
                c1 += new Vector<double>(aK[1]) * vb;
                c2 += new Vector<double>(aK[2]) * vb;
                c3 += new Vector<double>(aK[3]) * vb;
            }

            Unsafe.WriteUnaligned((void*)cBase, c0);
            Unsafe.WriteUnaligned((void*)(cBase + ldc), c1);
            Unsafe.WriteUnaligned((void*)(cBase + (2 * ldc)), c2);
            Unsafe.WriteUnaligned((void*)(cBase + (3 * ldc)), c3);
        }
    }

    /// <summary>
    /// Variant of <see cref="Run"/> reading B directly from caller memory with stride
    /// <paramref name="ldb"/>. Bit-identical to <see cref="ScalarFp64_4x4.RunStridedB"/>.
    /// </summary>
    /// <remarks>
    /// Supports only <c>transB == false</c>, i.e. B laid out [K, N] row-major, for the same reason the
    /// streaming tier does: j-vectorization needs B contiguous along j. With <c>transB == true</c> the four
    /// columns are <c>ldb</c> apart and would need a gather, so that shape stays on the scalar kernel.
    /// </remarks>
    /// <param name="packedA">Packed-A vpanel, layout [Kc x Mr] row-major.</param>
    /// <param name="b">B slice positioned at the (pc, jc) corner of the current panel.</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="c">Output buffer; reads and writes the C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static unsafe void RunStridedB(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> b,
        int ldb,
        Span<double> c,
        int ldc,
        int kc)
    {
        fixed (double* aBase = &MemoryMarshal.GetReference(packedA))
        fixed (double* bBase = &MemoryMarshal.GetReference(b))
        fixed (double* cBase = &MemoryMarshal.GetReference(c))
        {
            var c0 = Unsafe.ReadUnaligned<Vector<double>>((void*)cBase);
            var c1 = Unsafe.ReadUnaligned<Vector<double>>((void*)(cBase + ldc));
            var c2 = Unsafe.ReadUnaligned<Vector<double>>((void*)(cBase + (2 * ldc)));
            var c3 = Unsafe.ReadUnaligned<Vector<double>>((void*)(cBase + (3 * ldc)));

            for (int k = 0; k < kc; k++)
            {
                var vb = Unsafe.ReadUnaligned<Vector<double>>((void*)(bBase + ((long)k * ldb)));
                double* aK = aBase + (k * Mr);

                c0 += new Vector<double>(aK[0]) * vb;
                c1 += new Vector<double>(aK[1]) * vb;
                c2 += new Vector<double>(aK[2]) * vb;
                c3 += new Vector<double>(aK[3]) * vb;
            }

            Unsafe.WriteUnaligned((void*)cBase, c0);
            Unsafe.WriteUnaligned((void*)(cBase + ldc), c1);
            Unsafe.WriteUnaligned((void*)(cBase + (2 * ldc)), c2);
            Unsafe.WriteUnaligned((void*)(cBase + (3 * ldc)), c3);
        }
    }
}
