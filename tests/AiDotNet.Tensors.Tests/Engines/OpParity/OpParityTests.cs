// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775).
// Auto-covers every registered op: one theory case per op (keyed by serializable op name).
// The Roslyn generator (follow-up) will additionally emit a discoverable [Fact] per IEngine op
// that calls the SAME harness; this theory already provides full coverage over the registry.
#if !NETFRAMEWORK

using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

[Collection("OpParity")]
public sealed class OpParityTests
{
    private readonly OpParityFixture _fx;
    public OpParityTests(OpParityFixture fx) => _fx = fx;

    private static readonly IReadOnlyDictionary<string, OpCase> Cases =
        OpParityRegistry.ViTPath().ToDictionary(o => o.Name);

    public static IEnumerable<object[]> ForwardCases =>
        OpParityRegistry.ViTPath().Select(o => new object[] { o.Name });

    [SkippableTheory]
    [MemberData(nameof(ForwardCases))]
    public void Forward_CpuMatchesGpu(string opName)
        => OpParityHarness.CheckForward(Cases[opName], _fx);
}
#endif
