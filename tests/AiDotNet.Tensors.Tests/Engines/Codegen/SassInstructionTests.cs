// Copyright (c) AiDotNet. All rights reserved.
// The own SASS backend's bit model (#1442) against a real ptxas cubin: k1 below is
//   y[i] = fma(a[i], b[i], a[i]) for i < 1000, 256 threads a block
// compiled by ptxas 13.1 for sm_75. Its instructions (nvdisasm -hex) are quoted where a test reads them.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class SassInstructionTests
{
    // LEA R6, R3, R6, 0x8 ;  (waits on scoreboard 0, the S2Rs')
    private static readonly SassInstruction Lea = new(0x0000000603067211UL, 0x001fc800078e40ffUL);
    // @P0 EXIT ;
    private static readonly SassInstruction GuardedExit = new(0x000000000000094dUL, 0x000fea0003800000UL);
    // EXIT ;
    private static readonly SassInstruction Exit = new(0x000000000000794dUL, 0x000fea0003800000UL);
    // S2R R6, SR_TID.X ;  (variable latency: sets write scoreboard 0)
    private static readonly SassInstruction S2R = new(0x0000000000067919UL, 0x000e280000002100UL);
    // FFMA R9, R2, R5, R2 ;
    private static readonly SassInstruction Ffma = new(0x0000000502097223UL, 0x004fd00000000002UL);

    /// <summary>The same ptxas cubin, for the allocator's tests.</summary>
    internal static byte[] K1CubinImage() => K1Cubin();

    private static byte[] K1Cubin() => Convert.FromBase64String(
        "f0VMRgIBAUEIAAAAAAAAAAIAvgABAAAAAAAAAAAAAAAACwAAAAAAAIAHAAAAAAAABEsABkAAOAADAEAADgABAAAuc2hzdHJ0YWIA" +
        "LnN0cnRhYgAuc3ltdGFiAC5zeW10YWJfc2huZHgALm5vdGUubnYudGtpbmZvAC5ub3RlLm52LmN1aW5mbwAubnYuaW5mbwAudGV4" +
        "dC5rMQAubnYuaW5mby5rMQAubnYuc2hhcmVkLmsxAC5udi5jb25zdGFudDAuazEALnJlbC5udi5jb25zdGFudDAuazEALmRlYnVn" +
        "X2ZyYW1lAC5yZWwuZGVidWdfZnJhbWUALnJlbGEuZGVidWdfZnJhbWUALm52LmNhbGxncmFwaAAubnYucHJvdG90eXBlAC5udi5y" +
        "ZWwuYWN0aW9uAAAuc2hzdHJ0YWIALnN0cnRhYgAuc3ltdGFiAC5zeW10YWJfc2huZHgALm5vdGUubnYudGtpbmZvAC5ub3RlLm52" +
        "LmN1aW5mbwAubnYuaW5mbwAudGV4dC5rMQAubnYuaW5mby5rMQAubnYuc2hhcmVkLmsxAC5yZWwubnYuY29uc3RhbnQwLmsxAC5u" +
        "di5jb25zdGFudDAuazEALmRlYnVnX2ZyYW1lAC5yZWwuZGVidWdfZnJhbWUALnJlbGEuZGVidWdfZnJhbWUALm52LmNhbGxncmFw" +
        "aAAubnYucHJvdG90eXBlAC5udi5yZWwuYWN0aW9uAGsxAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAApAAAAAwAFAAAAAAAAAAAA" +
        "AAAAAAAAAAA5AAAAAwAGAAAAAAAAAAAAAAAAAAAAAABSAAAAAwANAAAAAAAAAAAAAAAAAAAAAACKAAAAAwAMAAAAAAAAAAAAAAAA" +
        "AAAAAACbAAAAAwAEAAAAAAAAAAAAAAAAAAAAAADLAAAAAwAJAAAAAAAAAAAAAAAAAAAAAADnAAAAAwAKAAAAAAAAAAAAAAAAAAAA" +
        "AAD2AAAAEhANAAAAAAAAAAAAAAEAAAAAAAD/////JAAAAAAAAAD//////////wMABHz/////DwyBgIAoAAj/gYAoCIGAgCgAAAD/" +
        "////NAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAEBAAAAAQUAAAADIGAgCgABCAAAAAAAAAAAAAADAAAAIgAAADQBwAA" +
        "TlZJRElBIENvcnAAAgAAAAAAAAABAAAABwAAADcAAABhAAAAAHB0eGFzAEN1ZGEgY29tcGlsYXRpb24gdG9vbHMsIHJlbGVhc2Ug" +
        "MTMuMSwgVjEzLjEuMTE1AEJ1aWxkIGN1ZGFfMTMuMS5yMTMuMS9jb21waWxlci4zNzA2MTk5NV8wAC1hcmNoIHNtXzc1IAAAAAwA" +
        "AAAIAAAA6AMAAE5WSURJQSBDb3JwAAIASwCDAAAABC8IAAgAAAAMAAAABBEIAAgAAAAAAAAABBIIAAgAAAAAAAAABDYEAAEAAAAE" +
        "NwQAgwAAAAQKCAAEAAAAYAEYAAMZGAAEFwwAAAAAAAIAEAAA8CEABBcMAAAAAAABAAgAAPAhAAQXDAAAAAAAAAAAAADwIQADG/8A" +
        "A18AAAQcCABQAAAA4AAAAAAAAAD/////AAAAAP7///8AAAAA/f///wAAAAD8////cwAAAAAAAAAAAAARJQAFNkQAAAAAAAAAAgAA" +
        "AAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAACegEAAAoAAAAPAAAA5A8AGXkGAAAAAAAAIQAAACgOABl5AwAAAAAAACUAAAAkDgARcgYDBgAAAP9AjgcA" +
        "yB8ADHgABugDAABwYvADANgPAE0JAAAAAAAAAACAAwDqDwACeAcABAAAAAAPAAAAyg8AJXYCBgBYAAAHAo4HAMgPACV2BAYAWgAA" +
        "BwKOBwDIDwCBcwICAAAAAADpHgAAqA4AgXMFBAAAAAAA6R4AAKIOACV2BgYAXAAABwKOBwDIDwAjcgkCBQAAAAIAAAAA0E8AhnMA" +
        "BgkAAAAA6RAAAOIPAE15AAAAAAAAAACAAwDqDwBHeQAA8P//////gwMAwA8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAADAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAA9gAAAAAAAAAA" +
        "AAAAAAAAAAEAAAAAAAAAAAAAAAAAAAALAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAA2AQAAAAAAAPkAAAAAAAAAAAAAAAAAAAABAAAA" +
        "AAAAAAAAAAAAAAAAEwAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAMAIAAAAAAADYAAAAAAAAAAIAAAAIAAAACAAAAAAAAAAYAAAAAAAA" +
        "AJsAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAgDAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAApAAAABwAAAAAA" +
        "AAIAAAAAAAAAAAAAAAB4AwAAAAAAAKAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAOQAAAAcAAAAAAAABAAAAAAAAAAAA" +
        "AAAAGAQAAAAAAAAgAAAAAAAAAAUAAAAAAAAABAAAAAAAAAAAAAAAAAAAAEkAAAAAAABwAAAAAAAAAAAAAAAAAAAAADgEAAAAAAAA" +
        "JAAAAAAAAAADAAAAAAAAAAQAAAAAAAAAAAAAAAAAAABbAAAAAAAAcEAAAAAAAAAAAAAAAAAAAABcBAAAAAAAAGQAAAAAAAAAAwAA" +
        "AA0AAAAEAAAAAAAAAAAAAAAAAAAAywAAAAEAAHAAAAAAAAAAAAAAAAAAAAAAwAQAAAAAAAAgAAAAAAAAAAMAAAAAAAAABAAAAAAA" +
        "AAAIAAAAAAAAAOcAAAALAABwAAAAAAAAAAAAAAAAAAAAAOAEAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAACAAAAAAAAACo" +
        "AAAACQAAAEAAAAAAAAAAAAAAAAAAAADwBAAAAAAAABAAAAAAAAAAAwAAAAQAAAAIAAAAAAAAABAAAAAAAAAAdQAAAAEAAABCAAAA" +
        "AAAAAAAAAAAAAAAAAAUAAAAAAAB4AQAAAAAAAAAAAAANAAAABAAAAAAAAAAAAAAAAAAAAFIAAAABAAAABgAAAAAAAAAAAAAAAAAA" +
        "AIAGAAAAAAAAAAEAAAAAAAADAAAACAAADIAAAAAAAAAAAAAAAAAAAAAGAAAABQAAAAALAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKgA" +
        "AAAAAAAAqAAAAAAAAAAIAAAAAAAAAAEAAAAFAAAAAAUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAIAAAAAAACAAgAAAAAAAAgAAAAA" +
        "AAAAAQAAAAUAAAAACwAAAAAAAAAAAAAAAAAAAAAAAAAAAACoAAAAAAAAAKgAAAAAAAAACAAAAAAAAAA=");

    [Fact]
    public void RegisterFieldsReadTheOperands()
    {
        Assert.Equal(6, Lea.Register(SassInstruction.RdField));
        Assert.Equal(3, Lea.Register(SassInstruction.RaField));
        Assert.Equal(6, Lea.Register(SassInstruction.RbField));
        Assert.Equal(9, Ffma.Register(SassInstruction.RdField));
        Assert.Equal(2, Ffma.Register(SassInstruction.RaField));
        Assert.Equal(5, Ffma.Register(SassInstruction.RbField));
        Assert.Equal(2, Ffma.Register(SassInstruction.RcField));
    }

    [Fact]
    public void TheGuardIsThePredicateField()
    {
        Assert.Equal(0, GuardedExit.GuardPredicate);
        Assert.False(GuardedExit.GuardNegated);
        Assert.Equal(SassInstruction.TruePredicate, Exit.GuardPredicate);
        Assert.Equal(GuardedExit.Opcode, Exit.Opcode);
    }

    [Fact]
    public void TheControlWordDecodes()
    {
        var s2r = S2R.Control;
        Assert.Equal(4, s2r.Stall);
        Assert.Equal(0, s2r.WriteScoreboard);
        Assert.Equal(SassInstruction.NoScoreboard, s2r.ReadScoreboard);
        Assert.Equal(0, s2r.WaitMask);
        var lea = Lea.Control;
        Assert.Equal(SassInstruction.NoScoreboard, lea.WriteScoreboard);
        Assert.Equal(1, lea.WaitMask);   // waits on the S2Rs' scoreboard 0
        Assert.Equal(4, Ffma.Control.WaitMask);   // waits on scoreboard 2, the loads'
    }

    [Fact]
    public void WritingAFieldChangesOnlyThatField()
    {
        var lea = Lea;
        lea.SetRegister(SassInstruction.RdField, 40);
        Assert.Equal(40, lea.Register(SassInstruction.RdField));
        Assert.Equal(3, lea.Register(SassInstruction.RaField));
        Assert.Equal(Lea.Control, lea.Control);
        lea.SetRegister(SassInstruction.RdField, 6);
        Assert.Equal(Lea, lea);

        var control = Lea.Control with { Stall = 13, WaitMask = 0x22, Reuse = 5 };
        lea.Control = control;
        Assert.Equal(control, lea.Control);
        Assert.Equal(Lea.Low, lea.Low);   // the control word lives in the high half
        lea.Control = Lea.Control;
        Assert.Equal(Lea, lea);
        Assert.Throws<ArgumentOutOfRangeException>(() => lea.SetBits(105, 4, 16));
    }

    [Fact]
    public void ACubinRoundTripsAndItsRegisterCountIsRewrittenInBothPlaces()
    {
        byte[] image = K1Cubin();
        var cubin = new SassCubin(image);
        Assert.Equal(new[] { "k1" }, cubin.Kernels);
        Assert.Equal(12, cubin.RegisterCount("k1"));
        var instructions = cubin.Instructions("k1");
        Assert.Equal(16, instructions.Length);
        Assert.Contains(Lea, instructions);
        Assert.Contains(Ffma, instructions);

        cubin.SetInstructions("k1", instructions);
        Assert.Equal(image, cubin.ToArray());

        cubin.SetRegisterCount("k1", 24);
        var reread = new SassCubin(cubin.ToArray());
        Assert.Equal(24, reread.RegisterCount("k1"));
        // .nv.info's EIATTR_REGCOUNT: format 0x04, attribute 0x2f, size 8, symbol, then the count.
        byte[] rewritten = cubin.ToArray();
        int changed = 0;
        for (int i = 0; i < image.Length; i++) if (image[i] != rewritten[i]) changed++;
        Assert.Equal(2, changed);   // one byte in the section header, one in the attribute
        Assert.Throws<ArgumentException>(() => cubin.SetInstructions("k1", new SassInstruction[15]));
        Assert.Throws<ArgumentException>(() => cubin.Instructions("missing"));
    }

    /// <summary>
    /// SassOpcodeTable on k1's instructions: what each reads and writes. (The table is checked against ptxas's own
    /// scoreboard waits over the whole catalog by --sass-learn; this pins a few by hand.)
    /// </summary>
    [Fact]
    public void TheOpcodeTableDecodesK1()
    {
        var ffma = SassOpcodeTable.Decode(Ffma) ?? throw new InvalidOperationException("FFMA unknown");
        Assert.Equal("FFMA", ffma.Mnemonic);
        Assert.Equal(SassLatency.Fixed, ffma.Latency);
        Assert.Equal(new[] { SassInstruction.RaField, SassInstruction.RbField, SassInstruction.RcField },
            ffma.Reads.Select(r => r.Field));
        var s2r = SassOpcodeTable.Decode(S2R) ?? throw new InvalidOperationException("S2R unknown");
        Assert.Equal(SassLatency.Variable, s2r.Latency);
        Assert.Empty(s2r.Reads);
        var exit = SassOpcodeTable.Decode(GuardedExit) ?? throw new InvalidOperationException("EXIT unknown");
        Assert.Equal(SassLatency.Control, exit.Latency);
        Assert.Null(SassOpcodeTable.Decode(new SassInstruction(0x0000000000000fffUL, 0)));
    }

    /// <summary>
    /// The scheduler's own control words for k1: every variable-latency result is waited on by its reader, the guarded
    /// EXIT issues 12 cycles after the compare that sets its predicate, no stall exceeds 12, and an unknown opcode
    /// declines the kernel.
    /// </summary>
    [Fact]
    public void TheSchedulerRecomputesK1sControlWords()
    {
        var code = new SassCubin(K1Cubin()).Instructions("k1");
        var scheduled = SassScheduler.RecomputeControl(code) ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.Equal(code.Length, scheduled.Length);
        for (int i = 0; i < code.Length; i++)
        {
            // Only the control word changes.
            Assert.Equal(code[i].Low, scheduled[i].Low);
            Assert.Equal(code[i].High & ((1UL << 41) - 1), scheduled[i].High & ((1UL << 41) - 1));
            Assert.InRange(scheduled[i].Control.Stall, 1, 12);
        }
        int lea = Array.IndexOf(code, Lea), ffma = Array.IndexOf(code, Ffma), exit = Array.IndexOf(code, GuardedExit);
        // LEA reads both S2R results: it waits on the scoreboard(s) they set.
        int s2rBoards = 0;
        for (int i = 0; i < lea; i++)
            if (SassOpcodeTable.Decode(scheduled[i])?.Mnemonic == "S2R") s2rBoards |= 1 << scheduled[i].Control.WriteScoreboard;
        Assert.Equal(s2rBoards, scheduled[lea].Control.WaitMask & s2rBoards);
        // FFMA reads both loads' results.
        int loadBoards = 0;
        for (int i = 0; i < ffma; i++)
            if (SassOpcodeTable.Decode(scheduled[i])?.Mnemonic == "LDG") loadBoards |= 1 << scheduled[i].Control.WriteScoreboard;
        Assert.NotEqual(0, loadBoards);
        Assert.Equal(loadBoards, scheduled[ffma].Control.WaitMask & loadBoards);
        // The compare is the instruction before the guarded EXIT: its stall carries the 12-cycle guard latency.
        Assert.True(scheduled[exit - 1].Control.Stall >= 12);

        var unknown = (SassInstruction[])code.Clone();
        unknown[3] = new SassInstruction(0x0000000000000fffUL, unknown[3].High);
        Assert.Null(SassScheduler.RecomputeControl(unknown));
        Assert.Contains("unknown opcode", SassScheduler.LastDecline);
    }

    // Instructions from ptxas's sm_75 register-blocked GEMM (PtxSgemmGenerator b128x128x8-t8x8-v1-p4-db), with their
    // nvdisasm text.
    private static readonly SassInstruction IaddCarryOut = new(0x0000002010107810UL, 0x000fe20007fde0ffUL);   // IADD3 R16, P6, R16, 0x20, RZ
    private static readonly SassInstruction ImadCarryIn = new(0x000000ffff117224UL, 0x000fe200030e0611UL);    // IMAD.X R17, RZ, RZ, R17, P6
    private static readonly SassInstruction IaddUniform = new(0x00000007440f7c10UL, 0x000fc6000fffe0ffUL);    // IADD3 R15, R68, UR7, RZ
    private static readonly SassInstruction BarSync = new(0x0000000000007b1dUL, 0x000fea0000000000UL);        // BAR.SYNC 0x0
    private static readonly SassInstruction S2ur = new(0x00000000000b79c3UL, 0x000ea20000002600UL);           // S2UR UR11, SR_CTAID.Y
    private static readonly SassInstruction Uimad = new(0x000000090c0972a4UL, 0x000fe2000f8e0205UL);          // UIMAD UR9, UR12, UR9, UR5
    private static readonly SassInstruction Uldc64 = new(0x0000660000087ab9UL, 0x000fe20000000a00UL);         // ULDC.64 UR8, c[0x0][0x198]
    private static readonly SassInstruction Lds128 = new(0x000000004c307984UL, 0x000ff00000001c00UL);         // LDS.U.128 R48, [R76]

    /// <summary>
    /// Uniform registers are a separate file: a vector instruction's form-6 operand and every uniform instruction's
    /// fields decode as uniform, with URZ (63) and the constant/immediate forms contributing no register.
    /// </summary>
    [Fact]
    public void UniformOperandsDecodeIntoTheirOwnFile()
    {
        var iadd = SassOpcodeTable.Decode(IaddUniform) ?? throw new InvalidOperationException("IADD3 form 6 unknown");
        Assert.Contains(iadd.Reads, o => o.Field == SassInstruction.RaField && !o.Uniform && IaddUniform.Register(o.Field) == 68);
        Assert.Contains(iadd.Reads, o => o.Field == SassInstruction.RbField && o.Uniform && IaddUniform.UniformRegister(o.Field) == 7);
        Assert.Contains(iadd.Writes, o => !o.Uniform && IaddUniform.Register(o.Field) == 15);

        var uimad = SassOpcodeTable.Decode(Uimad) ?? throw new InvalidOperationException("UIMAD unknown");
        Assert.All(uimad.Reads.Concat(uimad.Writes), o => Assert.True(o.Uniform));
        Assert.Equal(new[] { 12, 9, 5 }, uimad.Reads.Select(o => Uimad.UniformRegister(o.Field)).ToArray());
        Assert.Equal(9, Uimad.UniformRegister(uimad.Writes.Single().Field));

        var uldc = SassOpcodeTable.Decode(Uldc64) ?? throw new InvalidOperationException("ULDC unknown");
        Assert.Equal(SassLatency.Fixed, uldc.Latency);
        Assert.Empty(uldc.Reads);
        Assert.Equal(new SassRegisterOperand(SassInstruction.RdField, 2, true), uldc.Writes.Single());

        var s2ur = SassOpcodeTable.Decode(S2ur) ?? throw new InvalidOperationException("S2UR unknown");
        Assert.Equal(SassLatency.Variable, s2ur.Latency);
        Assert.Equal(11, S2ur.UniformRegister(s2ur.Writes.Single().Field));
    }

    /// <summary>
    /// IMAD.X reads the carry an IADD3 wrote: the scheduler holds it the ALU latency after the IADD3. Missing this
    /// read computed a wrong address high word on device and the GEMM's next load faulted.
    /// </summary>
    [Fact]
    public void ACarryInIsADependency()
    {
        var imad = SassOpcodeTable.Decode(ImadCarryIn) ?? throw new InvalidOperationException("IMAD.X unknown");
        Assert.Contains(imad.PredicateReads, field => ImadCarryIn.Bits(field, 3) == 6);
        var iadd = SassOpcodeTable.Decode(IaddCarryOut) ?? throw new InvalidOperationException("IADD3 unknown");
        Assert.Contains(iadd.PredicateWrites, field => IaddCarryOut.Bits(field, 3) == 6);

        var scheduled = SassScheduler.RecomputeControl(new[] { IaddCarryOut, ImadCarryIn, Exit })
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.True(scheduled[0].Control.Stall >= 6, "IMAD.X issued " + scheduled[0].Control.Stall + " cycles after its carry.");
    }

    /// <summary>
    /// Renaming rewrites every field naming a register (and nothing else), and refuses a map that would split or
    /// misalign a multi-register operand.
    /// </summary>
    [Fact]
    public void RenamingMovesEveryOperandAndKeepsGroupsWhole()
    {
        var map = Enumerable.Range(0, 128).ToArray();
        (map[16], map[20]) = (map[20], map[16]);
        var renamed = SassRegisterRenamer.Rename(new[] { IaddCarryOut, IaddUniform }, map)
            ?? throw new InvalidOperationException(SassRegisterRenamer.LastDecline);
        Assert.Equal(20, renamed[0].Register(SassInstruction.RdField));    // IADD3 R16, P6, R16, 0x20, RZ -> R20, R20
        Assert.Equal(20, renamed[0].Register(SassInstruction.RaField));
        Assert.Equal(IaddCarryOut.Bits(32, 32), renamed[0].Bits(32, 32));   // the immediate is untouched
        Assert.Equal(IaddCarryOut.Control, renamed[0].Control);
        Assert.Equal(IaddUniform, renamed[1]);                              // R15, R68 and UR7 are not in the swap
        Assert.Equal(7, renamed[1].UniformRegister(SassInstruction.RbField));

        // LDS.U.128 R48, [R76] writes R48..R51: moving the group whole is fine, moving R48 alone is not.
        var whole = Enumerable.Range(0, 128).ToArray();
        for (int k = 0; k < 4; k++) (whole[48 + k], whole[52 + k]) = (whole[52 + k], whole[48 + k]);
        Assert.Equal(52, SassRegisterRenamer.Rename(new[] { Lds128 }, whole)?[0].Register(SassInstruction.RdField));
        var split = Enumerable.Range(0, 128).ToArray();
        (split[48], split[52]) = (split[52], split[48]);
        Assert.Null(SassRegisterRenamer.Rename(new[] { Lds128 }, split));
        Assert.Contains("splits", SassRegisterRenamer.LastDecline);
    }

    /// <summary>
    /// The scheduler is per architecture: sm_86's barrier holds its successor 6 cycles, and an architecture without
    /// timings is declined rather than scheduled with another's.
    /// </summary>
    [Fact]
    public void TimingsArePerArchitecture()
    {
        var sm86 = SassScheduler.RecomputeControl(new[] { BarSync, Lds128, Exit }, 86)
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.True(sm86[0].Control.Stall >= 6);
        Assert.Null(SassScheduler.RecomputeControl(new[] { BarSync, Lds128, Exit }, 90));
        Assert.Contains("sm_90", SassScheduler.LastDecline);
    }

    // Instructions from the driver's sm_86 cubins (RTX 3080), with their nvdisasm text.
    private static readonly SassInstruction UimadWide86 = new(0x00000004080672a5UL, 0x000fc8000f8e003fUL);  // UIMAD.WIDE.U32 UR6, UR8, UR4, URZ
    private static readonly SassInstruction Uimad86 = new(0xfffffffe070778a4UL, 0x000fc6000f8e0204UL);      // UIMAD UR7, UR7, -0x2, UR4
    private static readonly SassInstruction UmovUr4 = new(0x0000003f00047c82UL, 0x000fe40008000000UL);      // UMOV UR4, URZ
    private static readonly SassInstruction UisetpUp0 = new(0x000000080400788cUL, 0x000fcc000bf06270UL);    // UISETP.GE.AND UP0, UPT, UR4, 0x8, UPT
    private static readonly SassInstruction PlopFromUp0 = new(0x000000000000781cUL, 0x000fe20003f0f008UL);  // PLOP3.LUT P0, PT, PT, PT, UP0, 0x80, 0x0
    private static readonly SassInstruction FaddImmediate = new(0x3f80000014147421UL, 0x000fc60000000000UL); // FADD R20, R20, 1
    private static readonly SassInstruction I2fp = new(0x0000000000007245UL, 0x000fca0000201400UL);        // I2FP.F32.S32 R0, R0

    /// <summary>
    /// sm_86 pipe issue spacing: two integer-ALU instructions issue 2 cycles apart (ptxas never leaves less on sm_86),
    /// while sm_75 keeps its measured schedule.
    /// </summary>
    [Fact]
    public void Sm86SpacesTheIntegerPipe()
    {
        var code = new[] { IaddCarryOut, IaddUniform, Exit };   // independent: R16/P6 against R68/UR7
        var sm86 = SassScheduler.RecomputeControl(code, 86) ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.True(sm86[0].Control.Stall >= 2);
        var sm75 = SassScheduler.RecomputeControl(code, 75) ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.Equal(1, sm75[0].Control.Stall);
        Assert.Equal(("uniform", 2), SassScheduler.PipeIssue("UIMAD", 86));
        Assert.Null(SassScheduler.PipeIssue("FFMA", 86));
    }

    /// <summary>
    /// A UMOV overwriting a register an earlier UIMAD reads issues 3 cycles after it on sm_86: the GEMM whose UMOV UR4
    /// issued 1 (then 2) cycles after UIMAD UR7, UR7, -0x2, UR4 faulted on the RTX 3080.
    /// </summary>
    [Fact]
    public void AnEarlyUniformWriteWaitsForTheLateRead()
    {
        var scheduled = SassScheduler.RecomputeControl(new[] { UimadWide86, Uimad86, UmovUr4, Exit }, 86)
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.True(scheduled[1].Control.Stall >= 3, "UMOV issued " + scheduled[1].Control.Stall + " cycles after the UIMAD.");
        Assert.True(scheduled[0].Control.Stall >= 2);   // the uniform pipe's spacing
    }

    /// <summary>
    /// PLOP3 with bit 67 set reads its third source from the uniform predicate file: the scheduler holds it the ALU
    /// latency after the UISETP writing UP0. Decoding it as P0 issued the PLOP3 1 cycle on and two convolutions
    /// computed wrong results on the RTX 3080.
    /// </summary>
    [Fact]
    public void PlopReadsAUniformPredicate()
    {
        var plop = SassOpcodeTable.Decode(PlopFromUp0) ?? throw new InvalidOperationException("PLOP3 unknown");
        Assert.Equal(new[] { 68 }, plop.UniformPredicateReads);
        Assert.DoesNotContain(68, plop.PredicateReads);
        var scheduled = SassScheduler.RecomputeControl(new[] { UisetpUp0, PlopFromUp0, Exit }, 86)
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.True(scheduled[0].Control.Stall >= 6, "PLOP3 issued " + scheduled[0].Control.Stall + " cycles after UISETP.");
    }

    /// <summary>
    /// Two-source operations with an immediate (FADD R, R, imm is form 2) read only A; I2FP (sm_80+) is a
    /// fixed-latency conversion reading B.
    /// </summary>
    [Fact]
    public void ImmediateFormsAndI2fpDecode()
    {
        var fadd = SassOpcodeTable.Decode(FaddImmediate) ?? throw new InvalidOperationException("FADD form 2 unknown");
        Assert.Equal(new[] { SassInstruction.RaField }, fadd.Reads.Select(o => o.Field));
        var i2fp = SassOpcodeTable.Decode(I2fp) ?? throw new InvalidOperationException("I2FP unknown");
        Assert.Equal("I2FP", i2fp.Mnemonic);
        Assert.Equal(SassLatency.Fixed, i2fp.Latency);
        Assert.Equal(new[] { SassInstruction.RbField }, i2fp.Reads.Select(o => o.Field));
    }

    /// <summary>A barrier holds its successor 5 cycles: an LDS issued 1 cycle after BAR.SYNC read a stale tile.</summary>
    [Fact]
    public void ABarrierHoldsItsSuccessorFiveCycles()
    {
        var scheduled = SassScheduler.RecomputeControl(new[] { BarSync, Lds128, Exit })
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.True(scheduled[0].Control.Stall >= 5);
        Assert.Equal(4, SassOpcodeTable.Decode(Lds128)?.Writes.Single().Width);
    }


    // BRA to instruction `target` placed at index `at` (the offset counts from the next instruction).
    private static SassInstruction Branch(int at, int target)
    {
        var bra = new SassInstruction(0x0000000000007947UL, 0x000fea0003800000UL);
        bra.SetBits(32, 50, (ulong)((target - (at + 1)) * 16L) & ((1UL << 50) - 1));
        return bra;
    }

    private static SassInstruction WithRd(SassInstruction instruction, int rd)
    {
        instruction.SetRegister(SassInstruction.RdField, rd);
        return instruction;
    }

    // FFMA Rd, Ra, Rb, Rc (register form).
    private static SassInstruction FfmaOf(int d, int a, int b, int c)
    {
        var ffma = Ffma;
        ffma.SetRegister(SassInstruction.RdField, d);
        ffma.SetRegister(SassInstruction.RaField, a);
        ffma.SetRegister(SassInstruction.RbField, b);
        ffma.SetRegister(SassInstruction.RcField, c);
        return ffma;
    }

    /// <summary>
    /// A variable-latency instruction gets a read scoreboard only when something can overwrite its sources: a later
    /// writer, or an earlier one inside a loop around it. The overwrite waits on that scoreboard.
    /// </summary>
    [Fact]
    public void AReadScoreboardGuardsOnlySourcesThatAreOverwritten()
    {
        var leaR76 = WithRd(Lea, 76);   // overwrites LDS.U.128 R48, [R76]'s address
        var alone = SassScheduler.RecomputeControl(new[] { Lds128, Exit })
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.Equal(SassInstruction.NoScoreboard, alone[0].Control.ReadScoreboard);

        var later = SassScheduler.RecomputeControl(new[] { Lds128, leaR76, Exit })
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        int board = later[0].Control.ReadScoreboard;
        Assert.NotEqual(SassInstruction.NoScoreboard, board);
        Assert.NotEqual(0, later[1].Control.WaitMask & (1 << board));

        // LEA R76 at 0, the LDS at 1, a branch back to 0: the next iteration's LEA overwrites the address.
        var loop = SassScheduler.RecomputeControl(new[] { leaR76, Lds128, Branch(2, 0), Exit })
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.NotEqual(SassInstruction.NoScoreboard, loop[1].Control.ReadScoreboard);
        var straight = SassScheduler.RecomputeControl(new[] { leaR76, Lds128, Branch(2, 3), Exit })
            ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.Equal(SassInstruction.NoScoreboard, straight[1].Control.ReadScoreboard);
    }

    /// <summary>
    /// No scoreboard counts more than its depth (16) of outstanding operations: 120 independent S2Rs fill all six
    /// boards, and each further one first waits on its board.
    /// </summary>
    [Fact]
    public void NoScoreboardHoldsMoreThanItsDepth()
    {
        var code = Enumerable.Range(0, 120).Select(r => WithRd(S2R, r)).Concat(new[] { Exit }).ToArray();
        var scheduled = SassScheduler.RecomputeControl(code) ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        var outstanding = new int[6];
        foreach (var instruction in scheduled)
        {
            var control = instruction.Control;
            for (int s = 0; s < 6; s++) if ((control.WaitMask >> s & 1) != 0) outstanding[s] = 0;
            if (control.WriteScoreboard != SassInstruction.NoScoreboard)
                Assert.True(++outstanding[control.WriteScoreboard] <= 16, "scoreboard " + control.WriteScoreboard + " overfilled");
        }
        Assert.Contains(scheduled.Take(120), i => i.Control.WaitMask != 0);
    }

    /// <summary>
    /// cp.async: the scoreboard DEPBAR.LE names is the copies' alone. LDGDEPBAR commits to it, nothing else sets it,
    /// and a block boundary does not wait on it. DEPBARs naming two scoreboards are declined.
    /// </summary>
    [Fact]
    public void TheCopyScoreboardIsReserved()
    {
        var ldgsts = new SassInstruction(0x0000000002047fae, 0x000fe2000b800000UL);   // LDGSTS [R4], [R2.64]
        var commit = new SassInstruction(0x00000000000079afUL, 0x000fe20000000000UL);   // LDGDEPBAR
        var depbar = new SassInstruction(0x000000000000791aUL, 0x000fe20000000000UL);   // DEPBAR.LE SB1, 0x0
        depbar.SetBits(44, 3, 1);
        var code = new[] { ldgsts, commit }.Concat(Enumerable.Range(10, 8).Select(r => WithRd(S2R, r)))
            .Concat(new[] { depbar, Exit }).ToArray();
        var scheduled = SassScheduler.RecomputeControl(code, 86) ?? throw new InvalidOperationException(SassScheduler.LastDecline);
        Assert.Equal(1, scheduled[1].Control.WriteScoreboard);
        for (int i = 0; i < scheduled.Length; i++)
        {
            if (i != 1) Assert.NotEqual(1, scheduled[i].Control.WriteScoreboard);
            Assert.NotEqual(1, scheduled[i].Control.ReadScoreboard);
            Assert.Equal(0, scheduled[i].Control.WaitMask & 0b10);
        }

        var other = depbar;
        other.SetBits(44, 3, 2);
        Assert.Null(SassScheduler.RecomputeControl(code.Concat(new[] { other }).ToArray(), 86));
        Assert.Contains("cp.async", SassScheduler.LastDecline);
    }

    /// <summary>
    /// The FFMA reorder moves a run of independent FFMAs (here the bank-conflicted first one later) but never across a
    /// branch target: a loop entering mid-run would skip or repeat different FFMAs.
    /// </summary>
    [Fact]
    public void TheFfmaReorderKeepsBranchTargetsInPlace()
    {
        var run = new[] { FfmaOf(9, 2, 4, 6), FfmaOf(11, 3, 4, 11), FfmaOf(13, 3, 6, 13), FfmaOf(15, 5, 8, 15) };
        var free = SassFfmaReorder.Reorder(run.Concat(new[] { Exit }).ToArray());
        Assert.NotEqual(run[0].Low, free[0].Low);
        Assert.Equal(run.Select(i => i.Low).OrderBy(x => x), free.Take(4).Select(i => i.Low).OrderBy(x => x));

        var entered = run.Concat(new[] { Branch(4, 2), Exit }).ToArray();
        var kept = SassFfmaReorder.Reorder(entered);
        Assert.Equal(entered.Select(i => i.Low), kept.Select(i => i.Low));
    }

    /// <summary>A malformed image is refused with ArgumentException, not an index fault.</summary>
    [Fact]
    public void AMalformedCubinIsRefused()
    {
        var image = K1Cubin();
        var badNames = (byte[])image.Clone();
        badNames[0x3e] = 0xff;
        badNames[0x3f] = 0x7f;   // section-name table index past the section count
        Assert.Throws<ArgumentException>(() => new SassCubin(badNames));
        Assert.Throws<ArgumentException>(() => new SassCubin(image.Take(image.Length / 2).ToArray()));
    }
}
