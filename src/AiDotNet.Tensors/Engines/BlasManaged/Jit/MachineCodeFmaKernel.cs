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

    /// <summary>
    /// Emit a 6×8 FP64 packed GEMM microkernel (Windows x64):
    ///   C[0..6, 0..8] += packedA[Kc×6] · packedB[Kc×8]  over kc K-steps, C read-modify-write.
    /// Signature: <c>void(double* packedA, double* packedB, double* c, long ldc, long kc)</c>
    ///   rcx=packedA, rdx=packedB, r8=c, r9=ldc(elements), [rsp+0x28]=kc.
    /// 12 accumulators (ymm0–11) + bLo/bHi (ymm12/13) + 2 A-broadcasts (ymm14/15) —
    /// all 16 YMM, our allocation, which RyuJIT can't sustain. Row r: lo=ymm(2r), hi=ymm(2r+1).
    /// </summary>
    internal static byte[] EmitFp64_6x8_PackedWindows()
    {
        const int RCX = 1, RDX = 2, R8 = 8, R9 = 9, R10 = 10, R11 = 11, RAX = 0, RSP = 4;
        const int BLO = 12, BHI = 13, A0 = 14, A1 = 15;
        const int Mr = 6, Nr = 8; // Nr*8=64 bytes/Kstep for B; Mr*8=48 for A.
        var asm = new X64Assembler();

        // kc (5th arg) is on the stack at [rsp+0x28] at entry — load before we move rsp.
        asm.MovRegFromRsp(R10, 0x28);

        // Prologue: preserve nonvolatile xmm6–15 (10 regs × 16 bytes = 0xA0).
        asm.SubRsp(0xA0);
        for (int i = 0; i < 10; i++) asm.VmovupsXmmStore(RSP, (sbyte)(i * 0x10), 6 + i);

        // rowStride (bytes) = ldc * 8  -> rax.
        asm.LeaRaxIndexScale8(R9);

        // Load 12 accumulators from C (read-modify-write). r11 walks the C rows.
        asm.MovRegReg(R11, R8);
        for (int r = 0; r < Mr; r++)
        {
            asm.VmovupdLoad(2 * r, R11, 0);          // c[r] lo
            asm.VmovupdLoad(2 * r + 1, R11, 32);     // c[r] hi
            if (r < Mr - 1) asm.AddRegReg(R11, RAX); // next row
        }

        // Guard kc==0 (do-while would loop forever): skip to store.
        int store = asm.NewLabel();
        asm.TestRegSelf(R10);
        asm.JzLabel(store);

        // K-loop.
        int loop = asm.NewLabel();
        asm.MarkLabel(loop);
        asm.VmovupdLoad(BLO, RDX, 0);
        asm.VmovupdLoad(BHI, RDX, 32);
        for (int r = 0; r < Mr; r++)
        {
            int areg = (r % 2 == 0) ? A0 : A1;       // alternate broadcast regs to pipeline
            asm.VbroadcastSd(areg, RCX, (sbyte)(r * 8));
            asm.Vfmadd231pd(2 * r, areg, BLO);       // c[r]lo += a[r]*bLo
            asm.Vfmadd231pd(2 * r + 1, areg, BHI);   // c[r]hi += a[r]*bHi
        }
        asm.AddRegImm8(RCX, Mr * 8);                 // aPtr += 6 doubles
        asm.AddRegImm8(RDX, Nr * 8);                 // bPtr += 8 doubles
        asm.DecReg(R10);                             // kc--
        asm.JnzLabel(loop);

        // Store 12 accumulators back to C.
        asm.MarkLabel(store);
        asm.MovRegReg(R11, R8);
        for (int r = 0; r < Mr; r++)
        {
            asm.VmovupdStore(R11, 0, 2 * r);
            asm.VmovupdStore(R11, 32, 2 * r + 1);
            if (r < Mr - 1) asm.AddRegReg(R11, RAX);
        }

        // Epilogue: restore xmm6–15, vzeroupper, ret.
        for (int i = 0; i < 10; i++) asm.VmovupsXmmLoad(6 + i, RSP, (sbyte)(i * 0x10));
        asm.AddRsp(0xA0);
        asm.Vzeroupper();
        asm.Ret();

        return asm.ToArray();
    }

    /// <summary>
    /// Emit a 6×16 FP32 packed GEMM microkernel (Windows x64):
    ///   C[0..6, 0..16] += packedA[Kc×6] · packedB[Kc×16]  over kc K-steps, C read-modify-write.
    /// Signature: <c>void(float* packedA, float* packedB, float* c, long ldc, long kc)</c>
    ///   rcx=packedA, rdx=packedB, r8=c, r9=ldc(elements), [rsp+0x28]=kc.
    /// Same 12-accumulator structure as the FP64 6×8 kernel, but each YMM holds 8
    /// floats: row r occupies lo=ymm(2r) (cols 0–7) + hi=ymm(2r+1) (cols 8–15). The
    /// per-step strides differ (A: 6×4=24 bytes, B: 16×4=64) and the row stride is
    /// ldc×4 bytes; the ps opcodes (vbroadcastss / vfmadd231ps / vmovups) replace pd.
    /// </summary>
    internal static byte[] EmitFp32_6x16_PackedWindows()
    {
        var asm = new X64Assembler();
        EmitWindowsPrologue(asm);                    // kc->r10, frame, save xmm6–15
        EmitBody_Fp32_6x16(asm);
        EmitWindowsEpilogue(asm);                    // restore xmm6–15, frame, vzeroupper, ret
        return asm.ToArray();
    }

    /// <summary>System V AMD64 (Linux/macOS) variant of <see cref="EmitFp32_6x16_PackedWindows"/>.
    /// Same body; only the ABI differs (args rdi/rsi/rdx/rcx/r8, no nonvolatile xmm).</summary>
    internal static byte[] EmitFp32_6x16_PackedSysV()
    {
        var asm = new X64Assembler();
        EmitSysVPrologue(asm);                       // shuffle SysV args into rcx/rdx/r8/r9/r10
        EmitBody_Fp32_6x16(asm);
        asm.Vzeroupper();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>
    /// Register-level FP32 6×16 body. Assumes rcx=packedA, rdx=packedB, r8=c,
    /// r9=ldc(elements), r10=kc; clobbers ymm0–15, rax, r11 and advances rcx/rdx/r10.
    /// ABI-agnostic — the caller supplies the prologue/epilogue. ymm6–15 are used as
    /// accumulators/scratch (callee must preserve them on Windows; on SysV they are
    /// volatile so nothing is saved).
    /// </summary>
    private static void EmitBody_Fp32_6x16(X64Assembler asm)
    {
        const int RCX = 1, RDX = 2, R8 = 8, R9 = 9, R10 = 10, R11 = 11, RAX = 0;
        const int BLO = 12, BHI = 13, A0 = 14, A1 = 15;
        const int Mr = 6, Nr = 16; // Nr*4=64 bytes/Kstep for B; Mr*4=24 for A.

        asm.LeaRaxIndexScale4(R9);                   // rowStride bytes = ldc*4 -> rax

        // Load 12 accumulators from C (read-modify-write). r11 walks the C rows.
        asm.MovRegReg(R11, R8);
        for (int r = 0; r < Mr; r++)
        {
            asm.VmovupsLoad(2 * r, R11, 0);          // c[r] cols 0–7
            asm.VmovupsLoad(2 * r + 1, R11, 32);     // c[r] cols 8–15
            if (r < Mr - 1) asm.AddRegReg(R11, RAX);
        }

        int store = asm.NewLabel();
        asm.TestRegSelf(R10);
        asm.JzLabel(store);

        int loop = asm.NewLabel();
        asm.MarkLabel(loop);
        asm.VmovupsLoad(BLO, RDX, 0);
        asm.VmovupsLoad(BHI, RDX, 32);
        for (int r = 0; r < Mr; r++)
        {
            int areg = (r % 2 == 0) ? A0 : A1;
            asm.VbroadcastSs(areg, RCX, (sbyte)(r * 4));
            asm.Vfmadd231ps(2 * r, areg, BLO);
            asm.Vfmadd231ps(2 * r + 1, areg, BHI);
        }
        asm.AddRegImm8(RCX, Mr * 4);                 // aPtr += 6 floats
        asm.AddRegImm8(RDX, Nr * 4);                 // bPtr += 16 floats
        asm.DecReg(R10);
        asm.JnzLabel(loop);

        asm.MarkLabel(store);
        asm.MovRegReg(R11, R8);
        for (int r = 0; r < Mr; r++)
        {
            asm.VmovupsStore(R11, 0, 2 * r);
            asm.VmovupsStore(R11, 32, 2 * r + 1);
            if (r < Mr - 1) asm.AddRegReg(R11, RAX);
        }
    }

    // ── ABI prologue/epilogue helpers (shared by Windows + SysV emitters) ───────
    private const int RSP_ = 4, R8_ = 8, R9_ = 9, R10_ = 10, RCX_ = 1, RDX_ = 2, RSI_ = 6, RDI_ = 7;

    /// <summary>Windows x64: kc (5th arg) is at [rsp+0x28]; xmm6–15 are nonvolatile.
    /// Load kc→r10 (before moving rsp), reserve frame, save xmm6–15.</summary>
    private static void EmitWindowsPrologue(X64Assembler asm)
    {
        asm.MovRegFromRsp(R10_, 0x28);
        asm.SubRsp(0xA0);
        for (int i = 0; i < 10; i++) asm.VmovupsXmmStore(RSP_, (sbyte)(i * 0x10), 6 + i);
    }

    private static void EmitWindowsEpilogue(X64Assembler asm)
    {
        for (int i = 0; i < 10; i++) asm.VmovupsXmmLoad(6 + i, RSP_, (sbyte)(i * 0x10));
        asm.AddRsp(0xA0);
        asm.Vzeroupper();
        asm.Ret();
    }

    /// <summary>System V AMD64: args in rdi(packedA), rsi(packedB), rdx(c), rcx(ldc),
    /// r8(kc). Shuffle into the body's rcx/rdx/r8/r9/r10 layout. No nonvolatile xmm,
    /// and the body touches no callee-saved GP regs, so no stack frame is needed.
    /// Order avoids clobbering a source before it's read.</summary>
    private static void EmitSysVPrologue(X64Assembler asm)
    {
        asm.MovRegReg(R10_, R8_);   // r10 = kc      (r8 was kc)
        asm.MovRegReg(R9_, RCX_);   // r9  = ldc     (rcx was ldc)
        asm.MovRegReg(R8_, RDX_);   // r8  = c       (rdx was c — read before overwrite)
        asm.MovRegReg(RDX_, RSI_);  // rdx = packedB
        asm.MovRegReg(RCX_, RDI_);  // rcx = packedA
    }

    /// <summary>
    /// Unroll-by-4 variant of <see cref="EmitFp64_6x8_PackedWindows"/> (disp32
    /// addressing, no per-step pointer advance; disp8 remainder loop for kc % 4).
    /// Same 6×8 / 12-accumulator math; bit-identical result.
    /// <para>
    /// Measured (best-of-7, Ryzen 9 3950X): <b>~57.8 GFLOPS vs the base loop's
    /// ~56.8</b> — only ~2% faster. Both sit at ~89% of the ~64 hardware peak and
    /// ~95% of OpenBLAS's ~60, i.e. the kernel is FMA-bound near the practical
    /// ceiling and the loop overhead the base loop pays is already tiny (predicted
    /// branch off the FMA ports). So unrolling is a marginal, optional tweak here,
    /// not a big lever. Kept as a tested variant; the compact base loop is the
    /// simpler default. (An early single-shot read showed 38.7 — that was
    /// measurement noise on this box, hence the best-of-N bench.)
    /// </para>
    /// </summary>
    internal static byte[] EmitFp64_6x8_PackedWindowsU4()
    {
        var asm = new X64Assembler();
        EmitWindowsPrologue(asm);
        EmitBody_Fp64_6x8_U4(asm);
        EmitWindowsEpilogue(asm);
        return asm.ToArray();
    }

    /// <summary>System V AMD64 (Linux/macOS) variant of <see cref="EmitFp64_6x8_PackedWindowsU4"/>.
    /// Same body; only the ABI differs (args rdi/rsi/rdx/rcx/r8, no nonvolatile xmm).</summary>
    internal static byte[] EmitFp64_6x8_PackedSysVU4()
    {
        var asm = new X64Assembler();
        EmitSysVPrologue(asm);
        EmitBody_Fp64_6x8_U4(asm);
        asm.Vzeroupper();
        asm.Ret();
        return asm.ToArray();
    }

    /// <summary>
    /// Register-level FP64 6×8 unroll-by-4 body. Assumes rcx=packedA, rdx=packedB,
    /// r8=c, r9=ldc(elements), r10=kc; clobbers ymm0–15, rax, r11 and advances
    /// rcx/rdx/r10. ABI-agnostic — the caller supplies the prologue/epilogue.
    /// </summary>
    private static void EmitBody_Fp64_6x8_U4(X64Assembler asm)
    {
        const int RCX = 1, RDX = 2, R8 = 8, R9 = 9, R10 = 10, R11 = 11, RAX = 0;
        const int BLO = 12, BHI = 13, A0 = 14, A1 = 15;
        const int Mr = 6, Nr = 8, U = 4;

        asm.LeaRaxIndexScale8(R9);                   // rowStride bytes

        // C read.
        asm.MovRegReg(R11, R8);
        for (int r = 0; r < Mr; r++)
        {
            asm.VmovupdLoad(2 * r, R11, 0);
            asm.VmovupdLoad(2 * r + 1, R11, 32);
            if (r < Mr - 1) asm.AddRegReg(R11, RAX);
        }

        // Main unrolled-by-4 loop: while (kc >= 4).
        int cond = asm.NewLabel();
        int rem = asm.NewLabel();
        int store = asm.NewLabel();
        asm.MarkLabel(cond);
        asm.CmpRegImm8(R10, U);
        asm.JlLabel32(rem);
        for (int s = 0; s < U; s++)
        {
            asm.VmovupdLoadD32(BLO, RDX, s * Nr * 8);
            asm.VmovupdLoadD32(BHI, RDX, s * Nr * 8 + 32);
            for (int r = 0; r < Mr; r++)
            {
                int areg = (r % 2 == 0) ? A0 : A1;
                asm.VbroadcastSdD32(areg, RCX, s * Mr * 8 + r * 8);
                asm.Vfmadd231pd(2 * r, areg, BLO);
                asm.Vfmadd231pd(2 * r + 1, areg, BHI);
            }
        }
        asm.AddRegImm32(RCX, U * Mr * 8);            // += 4*6 doubles
        asm.AddRegImm32(RDX, U * Nr * 8);            // += 4*8 doubles
        asm.SubRegImm8(R10, U);
        asm.JmpLabel32(cond);

        // Remainder (kc % 4): single K-step, disp8.
        asm.MarkLabel(rem);
        asm.TestRegSelf(R10);
        asm.JzLabel(store);
        int remLoop = asm.NewLabel();
        asm.MarkLabel(remLoop);
        asm.VmovupdLoad(BLO, RDX, 0);
        asm.VmovupdLoad(BHI, RDX, 32);
        for (int r = 0; r < Mr; r++)
        {
            int areg = (r % 2 == 0) ? A0 : A1;
            asm.VbroadcastSd(areg, RCX, (sbyte)(r * 8));
            asm.Vfmadd231pd(2 * r, areg, BLO);
            asm.Vfmadd231pd(2 * r + 1, areg, BHI);
        }
        asm.AddRegImm8(RCX, Mr * 8);
        asm.AddRegImm8(RDX, Nr * 8);
        asm.DecReg(R10);
        asm.JnzLabel(remLoop);

        // C write.
        asm.MarkLabel(store);
        asm.MovRegReg(R11, R8);
        for (int r = 0; r < Mr; r++)
        {
            asm.VmovupdStore(R11, 0, 2 * r);
            asm.VmovupdStore(R11, 32, 2 * r + 1);
            if (r < Mr - 1) asm.AddRegReg(R11, RAX);
        }
    }
}
