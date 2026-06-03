#if NET5_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.BlasManaged.Jit;

/// <summary>
/// #378 — AVX-512-BF16 native path, Phase 1: emit and execute a <c>VDPBF16PS</c> probe to
/// verify the EVEX encoding (<see cref="X64Assembler.Vdpbf16ps256"/>) and the instruction's
/// BF16-dot-product semantics.
///
/// <para>
/// .NET 10 exposes no <c>Avx512Bf16</c> intrinsic, so the only way to reach VDPBF16PS from
/// managed code is to emit the raw EVEX bytes and run them. Because this is raw machine code
/// (not a .NET intrinsic), <b>Intel SDE emulates the instruction on any host</b> — including
/// AVX2-only dev boxes and AMD CI runners — so the encoding is verifiable without
/// Sapphire-Rapids / Zen-4 hardware. (On a host without AVX-512-BF16 and without SDE, the
/// instruction faults <c>#UD</c>; the caller must only run the probe under SDE there.)
/// </para>
/// </summary>
internal static class MachineCodeBf16Kernel
{
    // Windows x64 ABI integer arg registers: rcx, rdx, r8.
    private const int RCX = 1, RDX = 2, R8 = 8;

    /// <summary>
    /// Emits <c>void probe(ushort* a, ushort* b, float* c)</c>:
    /// loads 16 BF16 from <c>a</c> → ymm1 and 16 from <c>b</c> → ymm2, zeroes ymm0, then
    /// <c>vdpbf16ps ymm0, ymm1, ymm2</c> so <c>c[i] = a[2i]·b[2i] + a[2i+1]·b[2i+1]</c>
    /// (BF16 widened to FP32, accumulated in FP32), and stores the 8 FP32 results.
    /// </summary>
    internal static byte[] EmitVdpbf16ProbeWindows()
    {
        var asm = new X64Assembler();
        asm.VmovupdLoad(1, RCX, 0);   // ymm1 = a[0..15]  (32 bytes = 16 BF16)
        asm.VmovupdLoad(2, RDX, 0);   // ymm2 = b[0..15]
        asm.Vxorpd(0, 0, 0);          // ymm0 = 0  (8 FP32 lanes)
        asm.Vdpbf16ps256(0, 1, 2);    // ymm0[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]
        asm.VmovupdStore(R8, 0, 0);   // c[0..7] = ymm0
        asm.Vzeroupper();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>
    /// Emit + execute the probe. <paramref name="a"/>/<paramref name="b"/> are 16 raw BF16
    /// bit-patterns each; <paramref name="c"/> receives 8 FP32 results. Returns false only if
    /// executable memory could not be allocated (NativeAOT / hardened runtime). MUST be run
    /// under Intel SDE on a host lacking AVX-512-BF16 (otherwise VDPBF16PS faults).
    /// </summary>
    internal static unsafe bool TryRunProbe(ushort[] a, ushort[] b, float[] c)
    {
        if (a.Length < 16 || b.Length < 16 || c.Length < 8)
            throw new ArgumentException("VDPBF16PS probe needs a/b >= 16 BF16 and c >= 8 floats.");

        byte[] code = EmitVdpbf16ProbeWindows();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null || mem.Pointer == IntPtr.Zero) return false;

        var fn = (delegate* unmanaged<ushort*, ushort*, float*, void>)mem.Pointer;
        fixed (ushort* ap = a)
        fixed (ushort* bp = b)
        fixed (float* cp = c)
            fn(ap, bp, cp);
        return true;
    }

    // ── Phase 2: full BF16 GEMM microkernel (MR=4 × NR=8, broadcast-A) ──────────

    private const int MR = 4;   // rows per tile (one accumulator ymm each)
    private const int NR = 8;   // cols per tile (one ymm of 8 FP32 lanes)

    /// <summary>
    /// Emit <c>void kernel(uint* packedA, uint* packedB, float* cTile, nuint kPairs)</c> — a
    /// register-resident MR×NR FP32 GEMM tile driven by VDPBF16PS over a runtime K-pair loop.
    /// <para>
    /// Each iteration consumes one K-pair (k0=2·kp, k1=2·kp+1):
    /// <c>packedB[kp]</c> is a 32-byte vector of NR columns × {B[k0,c], B[k1,c]} (BF16);
    /// <c>packedA[kp]</c> is MR × {A[m,k0], A[m,k1]} packed as one uint per row. For each row the
    /// A pair is broadcast (vbroadcastss, 32-bit) across all 8 lanes, then
    /// <c>vdpbf16ps acc_m, aBroadcast, bVec</c> adds A[m,k0]·B[k0,c]+A[m,k1]·B[k1,c] to lane c —
    /// i.e. the two-deep K contribution to C[m,c], accumulated in FP32.
    /// </para>
    /// Disjoint accumulators → bit-stable; no .NET intrinsic exists for VDPBF16PS so the body is
    /// raw EVEX. Verify under Intel SDE on non-AVX-512-BF16 hosts.
    /// </summary>
    internal static byte[] EmitGemmMicrokernelWindows()
    {
        // Windows x64 ABI: rcx=packedA, rdx=packedB, r8=cTile, r9=kPairs.
        const int RCX = 1, RDX = 2, R8 = 8, R9 = 9;
        var asm = new X64Assembler();

        for (int m = 0; m < MR; m++) asm.Vxorpd(m, m, m); // zero ymm0..ymm3 accumulators

        int loop = asm.NewLabel();
        asm.MarkLabel(loop);
        asm.VmovupdLoad(4, RDX, 0);                 // ymm4 = packedB[kp]  (NR cols × 2 BF16)
        for (int m = 0; m < MR; m++)
        {
            asm.VbroadcastSs(5, RCX, (sbyte)(m * 4)); // ymm5 = broadcast {A[m,k0],A[m,k1]}
            asm.Vdpbf16ps256(m, 5, 4);                // acc_m += dpbf16(aBroadcast, bVec)
        }
        asm.AddRegImm8(RDX, NR * 4);                // advance B one 32-byte K-pair vector
        asm.AddRegImm8(RCX, MR * 4);                // advance A one K-pair block (MR uints)
        asm.DecReg(R9);
        asm.JnzLabel(loop);

        for (int m = 0; m < MR; m++) asm.VmovupdStore(R8, (sbyte)(m * NR * 4), m); // cTile rows
        asm.Vzeroupper();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>
    /// Native AVX-512-BF16 GEMM: <c>C[m,n] = Σ_k A[m,k]·B[k,n]</c> with A/B as raw BF16 bit
    /// patterns (row-major, A is M×K, B is K×N) and C accumulated in FP32 (row-major M×N).
    /// Tiles M by <see cref="MR"/> and N by <see cref="NR"/>, packs each tile into the
    /// K-pair-interleaved layout the microkernel expects (odd K and ragged M/N edges zero-padded
    /// — BF16 0x0000 = +0.0, a no-op), and runs the emitted kernel. Returns false only if
    /// executable memory is unavailable. MUST run under SDE on hosts lacking AVX-512-BF16.
    /// </summary>
    internal static unsafe bool TryGemm(
        ReadOnlySpan<ushort> a, ReadOnlySpan<ushort> b, Span<float> c, int m, int k, int n)
    {
        if (m <= 0 || k <= 0 || n <= 0) return true;
        // Size math in long so large shapes can't wrap int, bypass the guard, and then fault in
        // packing/allocation with negative lengths (a span never exceeds int.MaxValue, so an
        // over-large requirement throws here). kPairs likewise from long before the int narrow.
        long aNeeded = (long)m * k, bNeeded = (long)k * n, cNeeded = (long)m * n;
        if (aNeeded > a.Length || bNeeded > b.Length || cNeeded > c.Length)
            throw new ArgumentException("BF16 GEMM: a/b/c spans smaller than M·K / K·N / M·N.");

        int kPairs = (int)(((long)k + 1) / 2);
        byte[] code = EmitGemmMicrokernelWindows();
        using var mem = ExecutableMemory.TryAllocate(code);
        if (mem is null || mem.Pointer == IntPtr.Zero) return false;
        var fn = (delegate* unmanaged<uint*, uint*, float*, nuint, void>)mem.Pointer;

        // Per-tile packed buffers (reused across tiles).
        var packA = new uint[kPairs * MR];
        var packB = new uint[kPairs * NR];
        var cTile = new float[MR * NR];

        fixed (uint* pA = packA)
        fixed (uint* pB = packB)
        fixed (float* pC = cTile)
        {
            for (int n0 = 0; n0 < n; n0 += NR)
            {
                int cols = Math.Min(NR, n - n0);
                // Pack B[*, n0..n0+cols) → [kPairs][NR] of {B[k0,c]|B[k1,c]<<16}; pad cols/k with 0.
                Array.Clear(packB, 0, packB.Length);
                for (int kp = 0; kp < kPairs; kp++)
                {
                    int k0 = 2 * kp, k1 = k0 + 1;
                    int baseB = kp * NR;
                    for (int cc = 0; cc < cols; cc++)
                    {
                        uint lo = b[k0 * n + n0 + cc];
                        uint hi = (k1 < k) ? b[k1 * n + n0 + cc] : 0u;
                        packB[baseB + cc] = lo | (hi << 16);
                    }
                }

                for (int m0 = 0; m0 < m; m0 += MR)
                {
                    int rows = Math.Min(MR, m - m0);
                    // Pack A[m0..m0+rows, *) → [kPairs][MR] of {A[m,k0]|A[m,k1]<<16}; pad rows/k.
                    Array.Clear(packA, 0, packA.Length);
                    for (int kp = 0; kp < kPairs; kp++)
                    {
                        int k0 = 2 * kp, k1 = k0 + 1;
                        int baseA = kp * MR;
                        for (int rr = 0; rr < rows; rr++)
                        {
                            uint lo = a[(m0 + rr) * k + k0];
                            uint hi = (k1 < k) ? a[(m0 + rr) * k + k1] : 0u;
                            packA[baseA + rr] = lo | (hi << 16);
                        }
                    }

                    fn(pA, pB, pC, (nuint)kPairs);

                    for (int rr = 0; rr < rows; rr++)
                        for (int cc = 0; cc < cols; cc++)
                            c[(m0 + rr) * n + n0 + cc] = cTile[rr * NR + cc];
                }
            }
        }
        return true;
    }
}
#endif
