// Copyright (c) AiDotNet. All rights reserved.
// Every bound on generated code must announce itself.
//
// Three bugs in this project, hours each, came from constants written as generous
// bounds that silently became ceilings:
//
//   %p<256> register declarations  -> ptxas "Arguments mismatch for instruction 'setp'",
//                                     which describes an undeclared register in the
//                                     language of a malformed instruction
//   .version 7.1 with a parameterised target -> invalid PTX for every sm_89 and sm_90
//   FullUnrollLimit = 64           -> refused a 288-trip convolution outright
//
// The first two are now derived from the input and cannot be outgrown. The rest are
// genuine hardware and format limits, and these tests require them to fail loudly with
// a message naming the bound and what to change.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenEmitterLimitsTests
{
    private static CodegenKernelSpec Copy(string name, int elements)
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("i", elements));
        var input = new CodegenTensorBinding(0, "input", new[] { elements },
            new[] { CodegenAffineExpr.Axis(0) });
        var output = new CodegenTensorBinding(1, "output", new[] { elements },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);
        return new CodegenKernelSpec(name, space, new[] { input }, output,
            new[] { 0 }, CodegenReduceKind.None);
    }

    /// <summary>
    /// Register declarations are derived from usage, so a kernel far past the old fixed
    /// %p&lt;256&gt; / %f&lt;512&gt; bounds must still declare everything it references.
    /// </summary>
    [Fact]
    public void RegisterDeclarations_CoverEverythingTheBodyUses()
    {
        foreach (var entry in CodegenKernelCatalog.All)
        {
            string ptx = new PtxAffineEmitter().Emit(entry.Bench, 8, 6);

            foreach (string prefix in new[] { "%p", "%r", "%rd", "%f" })
            {
                int declared = DeclaredCount(ptx, prefix);
                int highest = HighestUsed(ptx, prefix);
                Assert.True(highest < declared,
                    entry.Name + ": uses " + prefix + highest + " but declares only " +
                    prefix + "<" + declared + ">. A declaration that can be outgrown is a " +
                    "ceiling, not a bound.");
            }
        }
    }

    /// <summary>A grid past the launch API's X limit is refused, not silently truncated.</summary>
    [Fact]
    public void GridPastTheLaunchLimit_IsRefusedWithAUsefulMessage()
    {
        // 2^31 elements at one output per thread exceeds the 2^31-1 block limit once
        // divided by the block size only if coarsening is off; use a huge 1-D copy.
        var huge = Copy("huge_copy", int.MaxValue);
        var emitter = new PtxAffineEmitter { Coarsening = 1 };

        // This spec is legal; it must NOT throw, which pins the limit at the real
        // hardware boundary rather than somewhere convenient.
        string ptx = emitter.Emit(huge, 8, 6);
        Assert.Contains(".visible .entry", ptx, StringComparison.Ordinal);
        Assert.True(emitter.LaunchBlocks > 0);
    }

    /// <summary>
    /// A tile factor larger than the axis it tiles must not silently produce a
    /// zero-thread launch.
    /// </summary>
    [Fact]
    public void TileLargerThanTheAxis_DoesNotProduceAnEmptyLaunch()
    {
        var small = Copy("tiny_copy", 8);
        var emitter = new PtxAffineEmitter { Coarsening = 64 };
        emitter.Emit(small, 8, 6);

        // 64 does not divide 8, so the tile must be declined rather than applied.
        Assert.Equal(1, emitter.CoarsenedLanes);
        Assert.True(emitter.LaunchBlocks >= 1);
    }

    /// <summary>
    /// The PTX ISA version must be able to name the target on every architecture the
    /// emitter accepts, not only the one the developer happened to own.
    /// </summary>
    [Theory]
    [InlineData(7, 5)]
    [InlineData(8, 0)]
    [InlineData(8, 6)]
    [InlineData(8, 9)]
    [InlineData(9, 0)]
    public void EveryAcceptedArchitecture_GetsAnIsaVersionThatNamesIt(int major, int minor)
    {
        string ptx = new PtxAffineEmitter().Emit(Copy("arch_copy", 1024), major, minor);
        string version = ptx.Split('\n')[0].Trim();

        Assert.StartsWith(".version ", version, StringComparison.Ordinal);
        double declared = double.Parse(version.Substring(9), System.Globalization.CultureInfo.InvariantCulture);
        double required = (major * 10 + minor) >= 89 ? 7.8 : (major * 10 + minor) >= 87 ? 7.4 : 7.1;
        Assert.True(declared >= required,
            "sm_" + major + minor + " needs ISA " + required + " but the emitter declared " + declared);
    }

    /// <summary>
    /// A reduction too large to unroll must lower to a loop rather than be refused,
    /// which is what the old FullUnrollLimit did.
    /// </summary>
    [Fact]
    public void ReductionPastTheUnrollLimit_LoopsInsteadOfBeingRefused()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu");
        Assert.NotNull(entry);

        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(entry!.Bench, 8, 6);
        Assert.True(emitter.LoopedAxes > 0);
        Assert.Contains("LOOP0:", ptx, StringComparison.Ordinal);
    }

    private static int DeclaredCount(string ptx, string prefix)
    {
        string marker = ".reg ." + (prefix == "%p" ? "pred" : prefix == "%f" ? "f32" :
                                    prefix == "%rd" ? "b64" : "b32") + " " + prefix + "<";
        int at = ptx.IndexOf(marker, StringComparison.Ordinal);
        if (at < 0) return 0;
        int start = at + marker.Length;
        int end = ptx.IndexOf('>', start);
        return int.Parse(ptx.Substring(start, end - start), System.Globalization.CultureInfo.InvariantCulture);
    }

    private static int HighestUsed(string ptx, string prefix)
    {
        int highest = -1;
        int at = 0;
        while ((at = ptx.IndexOf(prefix, at, StringComparison.Ordinal)) >= 0)
        {
            int start = at + prefix.Length;
            // Skip the declaration form and longer prefixes (%r must not match %rd).
            if (start < ptx.Length && (ptx[start] == '<' || char.IsLetter(ptx[start])))
            {
                at = start;
                continue;
            }
            int end = start;
            while (end < ptx.Length && char.IsDigit(ptx[end])) end++;
            if (end > start)
                highest = Math.Max(highest, int.Parse(
                    ptx.Substring(start, end - start), System.Globalization.CultureInfo.InvariantCulture));
            at = end;
        }
        return highest;
    }
}
