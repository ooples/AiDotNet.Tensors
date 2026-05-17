using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Three-level Goto loop nest for the Pack-Both strategy: pack A panels into
/// vpanel layout, pack B panels into stripe layout, dispatch the microkernel
/// over the (M, N) tile grid. Used when both A and B benefit from cache-blocked
/// packing — typically large K shapes where pack cost amortizes across the
/// inner loops.
///
/// <para>
/// Microkernel dispatch follows an AVX-512 → AVX2 → Neon → scalar hierarchy:
/// FP64 prefers <see cref="Avx512Fp64_8x16"/> (Mr=8, Nr=16), falls back to
/// <see cref="Avx2Fp64_4x8"/> (Mr=4, Nr=8), then <see cref="NeonFp64_4x4"/>
/// (Mr=4, Nr=4) on ARM64, then <see cref="ScalarFp64_4x4"/>;
/// FP32 prefers <see cref="Avx512Fp32_16x16"/> (Mr=16, Nr=16), falls back to
/// <see cref="Avx2Fp32_8x8"/> (Mr=8, Nr=8), then <see cref="NeonFp32_8x4"/>
/// (Mr=8, Nr=4) on ARM64, then <see cref="ScalarFp32_4x4"/>.
/// </para>
///
/// <para>
/// Caller-supplied (mc, nc, kc) blocking parameters; (mr, nr) are fixed to
/// the microkernel's tile widths. The implementation handles partial outer
/// blocks (effective_mc/nc/kc = min(block, remaining)) but requires every
/// effective block to be exactly divisible by mr/nr — tail handling is added
/// in Phase G.
/// </para>
/// </summary>
internal static class PackBothStrategy
{
    /// <summary>
    /// Compute C += op(A) · op(B) using the Pack-Both 3-level loop nest.
    /// C is zeroed by the caller (BlasManaged.Gemm); the kernel accumulates.
    /// </summary>
    /// <typeparam name="T">Element type. Must be float or double.</typeparam>
    /// <param name="a">Source A buffer.</param>
    /// <param name="lda">Leading dimension of A.</param>
    /// <param name="transA">True if A is stored transposed: [K, M] layout for logical [M, K].</param>
    /// <param name="b">Source B buffer.</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="transB">True if B is stored transposed: [N, K] layout for logical [K, N].</param>
    /// <param name="c">Output matrix C, row-major [M, N] with leading dimension ldc.</param>
    /// <param name="ldc">Leading dimension of C (number of columns in the full C matrix).</param>
    /// <param name="m">Number of rows in op(A) and C.</param>
    /// <param name="n">Number of columns in op(B) and C.</param>
    /// <param name="k">Shared inner dimension: columns of op(A), rows of op(B).</param>
    /// <param name="mc">Row blocking factor (Mc); each A panel covers mc rows.</param>
    /// <param name="nc">Column blocking factor (Nc); each B stripe covers nc cols.</param>
    /// <param name="kc">K blocking factor (Kc); each packed panel covers kc K-steps.</param>
    /// <param name="mr">Microkernel row-tile width (Mr). Must divide mc exactly (Phase G adds tail).</param>
    /// <param name="nr">Microkernel column-tile width (Nr). Must divide nc exactly (Phase G adds tail).</param>
    /// <param name="options">Allocator options: workspace buffer, pre-pack handles, packing mode.</param>
    public static void Run<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        int mc, int nc, int kc,
        int mr, int nr,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        // Compute byte sizes for worst-case panel buffers.
        int elemSize = Unsafe.SizeOf<T>();
        int packABytes = mc * kc * elemSize;
        int packBBytes = kc * nc * elemSize;

        // Layer 5 → 4 → 1 selection for pack-A and pack-B byte buffers.
        // Layer 2 (ArrayPool overflow) is deferred to a future task.
        // Layer 3 (pre-pack handles) is checked per-iteration inside the loops.
        var carver = new WorkspaceCarver(options.Workspace);

        Span<byte> packABytesSpan = carver.HasWorkspace ? carver.TryCarve(packABytes) : Span<byte>.Empty;
        if (packABytesSpan.IsEmpty) packABytesSpan = ArenaIntegration.TryRentBytes(packABytes);
        if (packABytesSpan.IsEmpty) packABytesSpan = PerThreadPool.Current.RentPackA(packABytes);

        Span<byte> packBBytesSpan = carver.HasWorkspace ? carver.TryCarve(packBBytes) : Span<byte>.Empty;
        if (packBBytesSpan.IsEmpty) packBBytesSpan = ArenaIntegration.TryRentBytes(packBBytes);
        if (packBBytesSpan.IsEmpty) packBBytesSpan = PerThreadPool.Current.RentPackB(packBBytes);

        Span<T> packA = MemoryMarshal.Cast<byte, T>(packABytesSpan).Slice(0, mc * kc);
        Span<T> packB = MemoryMarshal.Cast<byte, T>(packBBytesSpan).Slice(0, kc * nc);

        for (int jc = 0; jc < n; jc += nc)
        {
            int effectiveNc = Math.Min(nc, n - jc);

            for (int pc = 0; pc < k; pc += kc)
            {
                int effectiveKc = Math.Min(kc, k - pc);

                // Layer 3: short-circuit pack-B when pre-pack handle is current.
                // Caller is responsible for the handle covering the full Kc × Nc panel.
                // Use effective tile bytes, not nominal, because PrePackB clamps Kc/Nc to
                // the actual matrix dimensions (small matrices produce smaller handles).
                int effectivePackBBytes = effectiveKc * effectiveNc * elemSize;
                bool packBFromPrePack = false;
                Span<T> activePackB = packB;
                if (options.PackedB != null && WeightPackCache.IsCacheCurrent(options.PackedB)
                    && options.PackedB.PackedBuffer.Length >= effectivePackBBytes)
                {
                    activePackB = MemoryMarshal.Cast<byte, T>(options.PackedB.PackedBuffer.AsSpan(0, effectivePackBBytes));
                    packBFromPrePack = true;
                }

                if (!packBFromPrePack)
                {
                    // Pack B[pc..pc+effectiveKc, jc..jc+effectiveNc] into packB.
                    // transB=false: B is [K, N] row-major, panel starts at b[pc * ldb + jc].
                    // transB=true:  B is [N, K] row-major, panel starts at b[jc * ldb + pc].
                    int bSliceOffset = transB ? jc * ldb + pc : pc * ldb + jc;
                    ScalarPack.PackB<T>(
                        b: b.Slice(bSliceOffset), ldb, transB,
                        packed: activePackB.Slice(0, effectiveKc * effectiveNc),
                        nc: effectiveNc, kc: effectiveKc, nr);
                }

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

                    // Iterate microkernel tiles within this Mc × Nc panel.
                    for (int jr = 0; jr < effectiveNc; jr += nr)
                    {
                        for (int ir = 0; ir < effectiveMc; ir += mr)
                        {
                            // Offset into packA for the current Mr-stripe (ir/mr th stripe).
                            // Stripe layout: [numStripes, Kc, Mr] → stripe * Kc * Mr.
                            int packedAStripeOff = (ir / mr) * effectiveKc * mr;
                            // Offset into packB for the current Nr-stripe (jr/nr th stripe).
                            // Stripe layout: [numStripes, Kc, Nr] → stripe * Kc * Nr.
                            int packedBStripeOff = (jr / nr) * effectiveKc * nr;

                            // C tile starts at row (ic + ir), col (jc + jr) in the full C matrix.
                            int cTileOff = (ic + ir) * ldc + (jc + jr);
                            DispatchMicrokernel<T>(
                                activePackA.Slice(packedAStripeOff, effectiveKc * mr),
                                activePackB.Slice(packedBStripeOff, effectiveKc * nr),
                                c.Slice(cTileOff),
                                ldc, effectiveKc,
                                mr, nr);
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Routes to the appropriate microkernel for T based on (mr, nr) and
    /// runtime SIMD availability. Dispatch order: AVX-512 → AVX2 → Neon → scalar,
    /// so the widest available ISA is always selected.
    /// </summary>
    private static void DispatchMicrokernel<T>(
        ReadOnlySpan<T> packedA, ReadOnlySpan<T> packedB,
        Span<T> c, int ldc, int kc,
        int mr, int nr) where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            if (mr == 8 && nr == 16 && Avx512Fp64_8x16.IsSupported)
            {
                Avx512Fp64_8x16.Run(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedA),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedB),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(c),
                    ldc, kc);
                return;
            }
            if (mr == 4 && nr == 8 && Avx2Fp64_4x8.IsSupported)
            {
                Avx2Fp64_4x8.Run(
                    MemoryMarshal.Cast<T, double>(packedA),
                    MemoryMarshal.Cast<T, double>(packedB),
                    MemoryMarshal.Cast<T, double>(c),
                    ldc, kc);
                return;
            }
            if (mr == 4 && nr == 4)
            {
                if (NeonFp64_4x4.IsSupported)
                {
                    NeonFp64_4x4.Run(
                        MemoryMarshal.Cast<T, double>(packedA),
                        MemoryMarshal.Cast<T, double>(packedB),
                        MemoryMarshal.Cast<T, double>(c),
                        ldc, kc);
                    return;
                }
                ScalarFp64_4x4.Run(
                    MemoryMarshal.Cast<T, double>(packedA),
                    MemoryMarshal.Cast<T, double>(packedB),
                    MemoryMarshal.Cast<T, double>(c),
                    ldc, kc);
                return;
            }
            throw new NotSupportedException($"Unsupported FP64 microkernel shape Mr={mr} Nr={nr}");
        }
        if (typeof(T) == typeof(float))
        {
            if (mr == 16 && nr == 16 && Avx512Fp32_16x16.IsSupported)
            {
                Avx512Fp32_16x16.Run(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedA),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedB),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(c),
                    ldc, kc);
                return;
            }
            if (mr == 8 && nr == 8 && Avx2Fp32_8x8.IsSupported)
            {
                Avx2Fp32_8x8.Run(
                    MemoryMarshal.Cast<T, float>(packedA),
                    MemoryMarshal.Cast<T, float>(packedB),
                    MemoryMarshal.Cast<T, float>(c),
                    ldc, kc);
                return;
            }
            // Neon FP32 uses Mr=8 Nr=4. PickMicrokernelTile returns (8, 4) on ARM64
            // hosts where Neon is available but AVX is not.
            if (mr == 8 && nr == 4 && NeonFp32_8x4.IsSupported)
            {
                NeonFp32_8x4.Run(
                    MemoryMarshal.Cast<T, float>(packedA),
                    MemoryMarshal.Cast<T, float>(packedB),
                    MemoryMarshal.Cast<T, float>(c),
                    ldc, kc);
                return;
            }
            // (4, 4) is only reached when no SIMD is available (no AVX, no Neon).
            if (mr == 4 && nr == 4)
            {
                ScalarFp32_4x4.Run(
                    MemoryMarshal.Cast<T, float>(packedA),
                    MemoryMarshal.Cast<T, float>(packedB),
                    MemoryMarshal.Cast<T, float>(c),
                    ldc, kc);
                return;
            }
            throw new NotSupportedException($"Unsupported FP32 microkernel shape Mr={mr} Nr={nr}");
        }
        throw new NotSupportedException($"PackBothStrategy does not support T={typeof(T).Name}.");
    }
}
