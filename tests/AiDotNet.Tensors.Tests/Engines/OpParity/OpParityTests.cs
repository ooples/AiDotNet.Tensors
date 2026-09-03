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
        OpParityRegistry.All().ToDictionary(o => o.Name);

    public static IEnumerable<object[]> ForwardCases =>
        OpParityRegistry.All()
            .Where(o => o.TensorOutputContract == TensorOutputContract.SingleTensor
                && ParityShard.Include(o.Name))
            .Select(o => new object[] { o.Name });

    public static IEnumerable<object[]> MultiOutputCases =>
        OpParityRegistry.All()
            .Where(o => o.TensorOutputContract != TensorOutputContract.SingleTensor && ParityShard.Include(o.Name))
            .Select(o => new object[] { o.Name });

    [SkippableTheory]
    [MemberData(nameof(ForwardCases))]
    public void Forward_CpuMatchesGpu(string opName)
        => OpParityHarness.CheckForward(Cases[opName], _fx);

    [SkippableTheory]
    [MemberData(nameof(MultiOutputCases))]
    public void MultipleOutputs_CpuMatchesGpu(string opName)
    {
        var op = Cases[opName];
        if (op.TensorOutputContract == TensorOutputContract.HeterogeneousMultiple)
            OpParityHarness.CheckHeterogeneousOutputs(op, _fx);
        else
            OpParityHarness.CheckMultipleOutputs(op, _fx);
    }

}
#endif
