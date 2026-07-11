// Copyright (c) AiDotNet. All rights reserved.
// Runtime bridge for the generated per-op parity facts (#775). The source generator emits one
// [SkippableFact] per tensor-returning IEngine op that calls into here; this looks up the
// registered parity case(s) for that op and runs the harness, or skips with a visible NEEDS-SPEC
// message so the test explorer shows the full IEngine surface and exactly which ops still lack a spec.
#if !NETFRAMEWORK

using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

public static class GeneratedOpParitySupport
{
    private static readonly ILookup<string, OpCase> ByMethod =
        OpParityRegistry.All().ToLookup(o => o.OpMethod, System.StringComparer.Ordinal);

    /// <summary>Run every registered parity case that exercises the given IEngine op method; if none
    /// is registered yet, skip with a NEEDS-SPEC note (so full-surface coverage stays auditable).</summary>
    public static void RunForwardByMethod(string opMethod, OpParityFixture fx)
    {
        var cases = ByMethod[opMethod].ToList();
        Skip.If(cases.Count == 0, $"NEEDS SPEC: IEngine.{opMethod} has no CPU-vs-GPU parity case yet.");
        foreach (var c in cases)
            OpParityHarness.CheckForward(c, fx);
    }
}
#endif
