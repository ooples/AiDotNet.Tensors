using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Pack-A-Only strategy — packs A into vpanel layout but reads B directly
/// from caller-supplied memory. Used by <see cref="BlasManaged.Gemm{T}"/>
/// when packing B is not worthwhile (typically: B fits in L1 already, or
/// the K depth is too small for pack amortization to pay off).
///
/// <para>
/// This Phase B implementation supports only transB=false (B stored
/// row-major [K, N]). transB=true is handled by <see cref="PackBothStrategy"/>
/// (which packs B and absorbs the transpose) or <see cref="StreamingStrategy"/>.
/// </para>
/// </summary>
internal static class PackAOnlyStrategy
{
    /// <summary>
    /// Compute C += op(A) · B with no B-side pack. C is read-modify-write.
    /// </summary>
    /// <typeparam name="T">Element type. Must be float or double.</typeparam>
    /// <param name="a">Source A buffer.</param>
    /// <param name="lda">Leading dimension of A.</param>
    /// <param name="transA">True if A is stored transposed: [K, M] layout for logical [M, K].</param>
    /// <param name="b">Source B buffer, row-major [K, N] (transB=false only).</param>
    /// <param name="ldb">Leading dimension of B (number of columns in B, i.e. N).</param>
    /// <param name="c">Output matrix C, row-major [M, N] with leading dimension ldc.</param>
    /// <param name="ldc">Leading dimension of C (number of columns in the full C matrix).</param>
    /// <param name="m">Number of rows in op(A) and C.</param>
    /// <param name="n">Number of columns in B and C.</param>
    /// <param name="k">Shared inner dimension: columns of op(A), rows of B.</param>
    /// <param name="mc">Row blocking factor (Mc); each A panel covers mc rows.</param>
    /// <param name="kc">K blocking factor (Kc); each packed panel covers kc K-steps.</param>
    /// <param name="mr">Microkernel row-tile width (Mr). Must divide mc exactly (Phase G adds tail).</param>
    /// <param name="nr">Microkernel column-tile width (Nr). Must divide n exactly (Phase G adds tail).</param>
    /// <param name="options">Allocator options: workspace buffer, pre-pack handles, packing mode.</param>
    public static void Run<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb,
        Span<T> c, int ldc,
        int m, int n, int k,
        int mc, int kc,
        int mr, int nr,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        // Compute byte size for the worst-case pack-A panel buffer.
        int elemSize = Unsafe.SizeOf<T>();
        int packABytes = mc * kc * elemSize;

        // Layer 5 → 4 → 1 selection for pack-A byte buffer.
        // Layer 3 (pre-pack handle) is checked per-iteration inside the loops.
        var carver = new WorkspaceCarver(options.Workspace);

        Span<byte> packABytesSpan = carver.HasWorkspace ? carver.TryCarve(packABytes) : Span<byte>.Empty;
        if (packABytesSpan.IsEmpty) packABytesSpan = ArenaIntegration.TryRentBytes(packABytes);
        if (packABytesSpan.IsEmpty) packABytesSpan = PerThreadPool.Current.RentPackA(packABytes);

        Span<T> packA = MemoryMarshal.Cast<byte, T>(packABytesSpan).Slice(0, mc * kc);

        for (int pc = 0; pc < k; pc += kc)
        {
            int effectiveKc = Math.Min(kc, k - pc);

            for (int ic = 0; ic < m; ic += mc)
            {
                int effectiveMc = Math.Min(mc, m - ic);

                // Layer 3: short-circuit pack-A when pre-pack handle is current.
                // Use effective tile bytes, not nominal, because PrePackA clamps Mc/Kc to
                // the actual matrix dimensions (small matrices produce smaller handles).
                int effectivePackABytes = effectiveMc * effectiveKc * elemSize;
                bool packAFromPrePack = false;
                Span<T> activePackA = packA;
                if (options.PackedA != null && WeightPackCache.IsCacheCurrent(options.PackedA)
                    && options.PackedA.PackedBuffer.Length >= effectivePackABytes)
                {
                    activePackA = MemoryMarshal.Cast<byte, T>(options.PackedA.PackedBuffer.AsSpan(0, effectivePackABytes));
                    packAFromPrePack = true;
                }

                if (!packAFromPrePack)
                {
                    // Pack A[ic..ic+effectiveMc, pc..pc+effectiveKc] into packA.
                    // transA=false: A is [M, K] row-major, panel starts at a[ic * lda + pc].
                    // transA=true:  A is [K, M] row-major, panel starts at a[pc * lda + ic].
                    int aSliceOffset = transA ? pc * lda + ic : ic * lda + pc;
                    ScalarPack.PackA<T>(
                        a: a.Slice(aSliceOffset), lda, transA,
                        packed: activePackA.Slice(0, effectiveMc * effectiveKc),
                        mc: effectiveMc, kc: effectiveKc, mr);
                }

                // Iterate microkernel tiles. B is read in-place at (pc, jc).
                for (int jc = 0; jc < n; jc += nr)
                {
                    if (jc + nr > n) break;  // Phase G tail handling

                    for (int ir = 0; ir < effectiveMc; ir += mr)
                    {
                        // Offset into packA for the current Mr-stripe (ir/mr th stripe).
                        // Stripe layout: [numStripes, Kc, Mr] → stripe * Kc * Mr.
                        int packedAStripeOff = (ir / mr) * effectiveKc * mr;
                        // C tile starts at row (ic + ir), col jc in the full C matrix.
                        int cTileOff = (ic + ir) * ldc + jc;
                        // B slice starts at row pc, col jc for transB=false: b[pc * ldb + jc].
                        int bSliceOffset = pc * ldb + jc;

                        DispatchStridedMicrokernel<T>(
                            activePackA.Slice(packedAStripeOff, effectiveKc * mr),
                            b.Slice(bSliceOffset), ldb,
                            c.Slice(cTileOff), ldc, effectiveKc);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Routes to the scalar strided-B microkernel matching T. Dispatches
    /// <see cref="ScalarFp64_4x4.RunStridedB"/> or
    /// <see cref="ScalarFp32_4x4.RunStridedB"/> based on the element type.
    /// </summary>
    private static void DispatchStridedMicrokernel<T>(
        ReadOnlySpan<T> packedA, ReadOnlySpan<T> b, int ldb,
        Span<T> c, int ldc, int kc) where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            ScalarFp64_4x4.RunStridedB(
                MemoryMarshal.Cast<T, double>(packedA),
                MemoryMarshal.Cast<T, double>(b), ldb,
                MemoryMarshal.Cast<T, double>(c), ldc, kc);
        }
        else if (typeof(T) == typeof(float))
        {
            ScalarFp32_4x4.RunStridedB(
                MemoryMarshal.Cast<T, float>(packedA),
                MemoryMarshal.Cast<T, float>(b), ldb,
                MemoryMarshal.Cast<T, float>(c), ldc, kc);
        }
        else
        {
            throw new NotSupportedException($"PackAOnlyStrategy does not support T={typeof(T).Name}.");
        }
    }
}
