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
}
#endif
