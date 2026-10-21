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
}
