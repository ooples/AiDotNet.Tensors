namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409) proof: emit a 12-accumulator FP64 FMA-throughput loop as raw
/// x86-64 machine code (Windows x64 ABI). RyuJIT craters past 8 vector
/// accumulators (FmaCeilingProbe: 8ch=44, 9ch=18 GFLOPS); by allocating the
/// registers ourselves we use 12 (ymm0–11) with the two operands in ymm12/13,
/// so the loop sustains enough in-flight FMA chains to approach 2 FMA/cycle —
/// the hardware peak RyuJIT can't reach. First-party code, no dependency.
///
/// <para>
/// Signature of the emitted function: <c>void(long iters, double* result, double* ab)</c>.
/// ab[0]=a, ab[1]=b are broadcast into ymm12/13; the loop does 12 FMAs/iter for
/// <c>iters</c> iterations; the 12 accumulators are summed and written to result[0..3].
/// Windows x64: rcx=iters, rdx=result, r8=ab.
/// </para>
/// </summary>
internal static class MachineCodeFmaKernel
{
    /// <summary>Emit the Windows-x64 12-accumulator FP64 FMA loop. Returns the machine-code bytes.</summary>
    internal static byte[] EmitFp64x12Windows()
    {
        const int RSP = 4, R8 = 8, RDX = 2;
        const int A = 12, Bv = 13; // operand vectors
        var asm = new X64Assembler();

        // Prologue: preserve nonvolatile xmm6–11 (Windows x64 ABI). 6 × 16 bytes.
        asm.SubRsp(0x60);
        for (int i = 0; i < 6; i++) asm.VmovupsXmmStore(RSP, (sbyte)(i * 0x10), 6 + i);

        // Broadcast a, b into ymm12, ymm13.
        asm.VbroadcastSd(A, R8, 0);
        asm.VbroadcastSd(Bv, R8, 8);

        // Zero the 12 accumulators ymm0..ymm11.
        for (int i = 0; i < 12; i++) asm.Vxorpd(i, i, i);

        // Loop: 12 independent FMAs, dec rcx, jnz.
        int loop = asm.NewLabel();
        asm.MarkLabel(loop);
        for (int i = 0; i < 12; i++) asm.Vfmadd231pd(i, A, Bv);
        asm.DecRcx();
        asm.JnzLabel(loop);

        // Reduce ymm0 += ymm1..ymm11, store ymm0 to result.
        for (int i = 1; i < 12; i++) asm.Vaddpd(0, 0, i);
        asm.VmovupdStore(RDX, 0, 0);

        // Epilogue: restore xmm6–11, vzeroupper, ret.
        for (int i = 0; i < 6; i++) asm.VmovupsXmmLoad(6 + i, RSP, (sbyte)(i * 0x10));
        asm.AddRsp(0x60);
        asm.Vzeroupper();
        asm.Ret();

        return asm.ToArray();
    }
}
